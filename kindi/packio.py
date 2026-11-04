"""How a pack stores a block's scale, and how it comes back.

A ternary tensor is a grid of 128-weight blocks, and a block's scale is one fp16 number - the absmean of the weights it
quantises. That is 29 MB of the 2B pack's 506 MB, and 422 MB of a 27B pack's ~8 GB: the only part of a ternary tensor
that is still wider than it needs to be. Blocks of one tensor are quantised from the same distribution, so their scales
cluster tightly, and 8 bits address 256 levels - a level table written once per tensor costs nothing next to a scale per
block.

  pack_format 1   dpm.bin      fp16 per block, 2 bytes
  pack_format 2   dpm.bin      uint8 per block, 1 byte: an index into the tensor's table
                  dscale.bin   every tensor's fp16 level table, back to back
                  bcodes_index.json  dblk (block offset), dtab_off, dtab_n

`load_scales` returns one value per block in either format, so nothing downstream of it knows which one it read and the
kernels are untouched. `block_of` is the single rule from an index record to a block offset - v1 records carry
`dpm_off`, a byte offset at 2 bytes a block; v2 records carry `dblk` directly.

Levels are an even grid in log2 over the tensor's own range: the fit that minimises the *worst* block's relative error,
which is what the 2B NLL walk turned out to care about. Choosing levels by k-means instead spreads them into the dense
part of the distribution, which halves the average error and leaves the tails 57% off - and cost 0.11 nats, against
0.003 for this fit. A scale multiplies the block's codes, so relative error is the metric, and the grid caps it at half
a level for every block in the tensor - `worst_error` reads that cap off the levels actually written, and `kindi
convert` records the widest one in the manifest. A tensor with no more than 256 distinct scales keeps them exactly - its table is those values - so small tensors and the test fixtures convert bit-for-bit, and only tensors with
more scales than levels are approximated at all.
"""
import json
import os

import numpy as np

BITS = 8
KMAX = 1 << BITS


def log_grid(pos, k):
    """k levels evenly spaced in log2 over [min, max] of the positive values: the minimax fit, as float16, ascending."""
    lo, hi = float(np.log2(pos.min())), float(np.log2(pos.max()))
    if int(k) < 2 or hi <= lo:
        return np.asarray([2.0 ** lo], dtype=np.float16)
    return np.unique((2.0 ** np.linspace(lo, hi, int(k))).astype(np.float16))


def worst_error(table, lo, hi):
    """The largest relative error these levels can give any value in [lo, hi]: half the widest gap, and the ends.

    Read off the table that was actually written rather than from the grid that was asked for, because rounding a level
to fp16 moves it by a hair and the guarantee should describe the file.
    """
    t = np.asarray(table, dtype=np.float64)
    t = t[t > 0]
    if t.size == 0:
        return 0.0
    if t.size == 1:
        return float(max(abs(t[0] / lo - 1.0), abs(hi / t[0] - 1.0)))
    e = float(2.0 ** (np.diff(np.log2(t)).max() / 2.0) - 1.0)
    if lo < t[0]:
        e = max(e, float(t[0] / lo - 1.0))
    if hi > t[-1]:
        e = max(e, float(hi / t[-1] - 1.0))
    return e


def _assign(pos, table):
    """Index of the nearest table entry for each positive value, compared in log2 (ties go to the lower entry)."""
    t = np.asarray(table, dtype=np.float64)
    start = 0 if t[0] > 0 else 1
    lt, lp = np.log2(t[start:]), np.log2(np.asarray(pos, dtype=np.float64))
    j = np.searchsorted(lt, lp)
    lo = np.clip(j - 1, 0, lt.size - 1)
    hi = np.clip(j, 0, lt.size - 1)
    take_hi = np.abs(lt[hi] - lp) < np.abs(lp - lt[lo])
    return (start + np.where(take_hi, hi, lo)).astype(np.uint8)


def level_table(values, bits=BITS):
    """(n,) block scales -> (table fp16 (K,), index uint8 (n,), exact).

    `exact` is True when every scale is in the table, which happens whenever the tensor has no more distinct scales
    than the table has entries.
    """
    v = np.asarray(values, dtype=np.float32).ravel()
    if v.size == 0:
        return np.zeros(0, np.float16), np.zeros(0, np.uint8), True
    if not np.all(np.isfinite(v)):
        raise ValueError("level_table: %d block scales are not finite" % int((~np.isfinite(v)).sum()))
    if np.any(v < 0):
        raise ValueError("level_table: a ternary block scale is a magnitude, but %d are negative" % int((v < 0).sum()))
    k = 1 << int(bits)
    zero = v == 0
    pos = v[~zero].astype(np.float64)
    room = k - (1 if zero.any() else 0)
    if pos.size == 0:
        return np.zeros(1, np.float16), np.zeros(v.size, np.uint8), True
    uniq = np.unique(pos)
    exact = uniq.size <= room
    levels = uniq.astype(np.float16) if exact else log_grid(pos, room)
    levels = np.unique(levels)                            # rounding to fp16 can merge two levels
    table = np.concatenate([np.zeros(1, np.float16), levels]) if zero.any() else levels
    idx = np.zeros(v.size, dtype=np.uint8)
    if pos.size:
        idx[~zero] = _assign(pos, table)
    return table, idx, exact


def encode(scales, bits=BITS):
    """fp16 block scales -> (table fp16, index uint8 with the shape of `scales`, exact)."""
    a = np.asarray(scales)
    table, idx, exact = level_table(np.asarray(a, dtype=np.float16), bits)
    return table, idx.reshape(a.shape), exact


def decode(table, index):
    """(table, index) -> the fp16 scales the blocks stand for."""
    return np.asarray(table, dtype=np.float16)[np.asarray(index)]


def block_of(rec):
    """The block offset of a tensor's scales: `dblk` in a v2 record, `dpm_off` at 2 bytes a block in a v1 one."""
    return int(rec["dblk"]) if "dblk" in rec else int(rec["dpm_off"]) // 2


def tensor_scales(dp, rec):
    """One tensor's block scales out of the array `load_scales` returned, phase-major (P, R) values."""
    q = block_of(rec)
    return dp[q:q + int(rec["rows"]) * (int(rec["ne0"]) // 128)]


def load_scales(pack, manifest=None, bindex=None):
    """The pack's block scales as float16, one element per block, whichever way they are stored.

    A v1 pack's file is already that array, so it comes back mapped. A v2 pack stores indices, and this expands them
    through the per-tensor tables: an allocation of 2 bytes a block, which is what the reader had to hold anyway.
    """
    man = manifest if manifest is not None else json.load(open(os.path.join(pack, "kindi_pack.json")))
    if int(man.get("pack_format") or 1) < 2:
        return np.memmap(os.path.join(pack, "dpm.bin"), dtype=np.float16, mode="r")
    bi = bindex if bindex is not None else json.load(open(os.path.join(pack, "bcodes_index.json")))
    bi = bi.get("tensors", bi)
    idx = np.memmap(os.path.join(pack, "dpm.bin"), dtype=np.uint8, mode="r")
    tabs = np.memmap(os.path.join(pack, "dscale.bin"), dtype=np.float16, mode="r")
    out = np.empty(int(idx.shape[0]), dtype=np.float16)
    for name, rec in bi.items():
        q, n = block_of(rec), int(rec["rows"]) * (int(rec["ne0"]) // 128)
        if q + n > out.size:
            raise SystemExit("kindi: %s needs blocks %d..%d, but dpm.bin holds %d. The pack is truncated or its index "
                             "belongs to another pack; convert the model again." % (name, q, q + n, out.size))
        t0, tn = int(rec["dtab_off"]) // 2, int(rec["dtab_n"])
        if t0 + tn > tabs.shape[0]:
            raise SystemExit("kindi: %s needs table entries %d..%d, but dscale.bin holds %d." % (name, t0, t0 + tn,
                                                                                                tabs.shape[0]))
        out[q:q + n] = tabs[t0:t0 + tn][idx[q:q + n]]
    return out


# --- selftest ---------------------------------------------------------------------------------------------------------

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)


def selftest():
    import tempfile
    del PASS[:]
    rng = np.random.RandomState(3)

    # a tensor with few distinct scales keeps them exactly: the table is those values, and the decode is bit-identical
    few = np.asarray(rng.choice([0.0625, 0.125, 0.1875, 0.25], size=5000), dtype=np.float16)
    table, idx, exact = level_table(few)
    back = decode(table, idx)
    check("few distinct scales: the table is exact and the round trip is bit-identical",
          exact and table.size <= 5 and np.array_equal(back, few) and np.array_equal(np.unique(few), np.unique(back)),
          "%d levels" % table.size)

    # more scales than levels: 256 levels, every value within half a level of the fitted one, and the decode is sorted
    # so the table can be searched
    many = (10.0 ** rng.uniform(-2.0, 1.0, 20000)).astype(np.float16)
    table, idx, exact = level_table(many)
    back = decode(table, idx)
    check("more scales than levels: 8 bits, and every scale lands on one of the levels",
          (not exact) and table.size <= KMAX and np.all(np.isin(back, table)) and np.array_equal(np.sort(table), table),
          "%d levels for %d distinct scales" % (table.size, np.unique(many).size))
    rel = np.abs(back.astype(np.float64) - many.astype(np.float64)) / many.astype(np.float64)
    bound = worst_error(table, float(many.min()), float(many.max()))
    check("every block lands within the worst case the written levels promise, even across three decades",
          float(rel.max()) <= bound + 1e-12 and float(np.percentile(rel, 99)) < bound < 0.02,
          "worst %.5f against a bound of %.5f, mean %.5f" % (float(rel.max()), bound, float(rel.mean())))

    # zeros are the padding a reordered head leaves behind, and a scale of zero must survive as zero
    withzero = np.concatenate([np.zeros(50, np.float16), many[:200]])
    tz, iz, _ = level_table(withzero)
    check("a zero scale (padded rows) decodes back to zero", tz[0] == 0 and np.all(decode(tz, iz)[:50] == 0))

    # the error is bounded by the fitted levels, not by the value: one level is the whole table
    one = np.full(64, 0.375, dtype=np.float16)
    t1, i1, e1 = level_table(one)
    check("a tensor whose scales are one value keeps exactly that value", e1 and t1.size == 1 and
          np.all(decode(t1, i1) == one))

    # a negative scale is refused rather than fitted, because a ternary block scale is a magnitude
    try:
        level_table(np.asarray([-0.5, 0.5], dtype=np.float16))
        refused = False
    except ValueError:
        refused = True
    check("a negative scale is refused, not fitted", refused)

    # the two on-disk formats agree: a v1 pack (fp16 per block) and a v2 pack (8-bit index) load to the same scales
    tmp = tempfile.mkdtemp(prefix="packio-")
    blocks = 4096
    sc = (10.0 ** rng.uniform(-1.0, 0.5, blocks)).astype(np.float16)
    names = [("a", 0, 1024), ("b", 1024, 3072)]                 # two tensors, blocks [0,1024) and [1024,4096)
    os.makedirs(os.path.join(tmp, "v1"))
    os.makedirs(os.path.join(tmp, "v2"))
    with open(os.path.join(tmp, "v1", "dpm.bin"), "wb") as f:
        f.write(sc.tobytes())
    with open(os.path.join(tmp, "v1", "kindi_pack.json"), "w") as f:
        json.dump({"pack_format": 1}, f)
    b1 = {"tensors": dict((n, {"dpm_off": q * 2, "rows": 8, "ne0": (nb * 128) // 8}) for n, q, nb in names)}
    with open(os.path.join(tmp, "v1", "bcodes_index.json"), "w") as f:
        json.dump(b1, f)
    idxs, tables, off = np.zeros(0, np.uint8), [], 0
    b2 = {"tensors": {}}
    for n, q, nb in names:
        t, i, _ = level_table(sc[q:q + nb])
        rec = {"dblk": q, "dtab_off": off, "dtab_n": t.size, "rows": 8, "ne0": (nb * 128) // 8}
        b2["tensors"][n] = rec
        tables.append(t)
        off += 2 * t.size
        if q == 0:
            idxs = i.astype(np.uint8)
        else:
            idxs = np.concatenate([idxs, i.astype(np.uint8)])
    with open(os.path.join(tmp, "v2", "dpm.bin"), "wb") as f:
        f.write(idxs.tobytes())
    with open(os.path.join(tmp, "v2", "dscale.bin"), "wb") as f:
        f.write(np.concatenate(tables).astype(np.float16).tobytes())
    with open(os.path.join(tmp, "v2", "kindi_pack.json"), "w") as f:
        json.dump({"pack_format": 2}, f)
    with open(os.path.join(tmp, "v2", "bcodes_index.json"), "w") as f:
        json.dump(b2, f)
    d1 = load_scales(os.path.join(tmp, "v1"))
    d2 = load_scales(os.path.join(tmp, "v2"))
    check("a v1 pack and a v2 pack of the same scales load to one value per block, the same values",
          d1.shape == d2.shape == (blocks,) and np.allclose(d1.astype(np.float32), d2.astype(np.float32),
                                                            rtol=2e-2, atol=1e-6),
          "%d blocks" % blocks)
    check("tensor_scales cuts the same slice out of either format",
          all(np.array_equal(tensor_scales(d1, b1["tensors"][n]), tensor_scales(d2, b2["tensors"][n])) or
              np.allclose(tensor_scales(d1, b1["tensors"][n]).astype(np.float32),
                          tensor_scales(d2, b2["tensors"][n]).astype(np.float32), rtol=2e-2)
              for n, _q, _nb in names))

    # a v2 pack whose index points past its files is refused with a message, not an IndexError
    os.makedirs(os.path.join(tmp, "bad"))
    with open(os.path.join(tmp, "bad", "dpm.bin"), "wb") as f:
        f.write(idxs[:10].tobytes())
    with open(os.path.join(tmp, "bad", "dscale.bin"), "wb") as f:
        f.write(np.concatenate(tables).astype(np.float16).tobytes())
    with open(os.path.join(tmp, "bad", "kindi_pack.json"), "w") as f:
        json.dump({"pack_format": 2}, f)
    with open(os.path.join(tmp, "bad", "bcodes_index.json"), "w") as f:
        json.dump(b2, f)
    try:
        load_scales(os.path.join(tmp, "bad"))
        refused = False
    except SystemExit as e:
        refused = "truncated" in str(e)
    check("a v2 pack that is shorter than its index says is refused, naming the file", refused)

    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == 9 else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
