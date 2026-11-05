"""Merging co-activated expert groups into a real pack, and measuring what the merge costs in NLL.

`cluster_budget.py` groups experts by co-activation to ask how many bytes a paging schedule would fetch. This uses
the same grouping to ask the other question: what happens to the model when each group is *replaced* by one averaged
expert. That answer needs a pack, because NLL has to be computed by a model whose experts really are shared - a mask
that maps two routing slots onto one matrix would leave the router and the renormalisation untouched, which is not
the model a merged pack is.

So this writes a GGUF: every non-expert tensor is copied byte for byte, the three expert stacks are replaced by
`n_groups` merged matrices, the router loses the rows of the experts that were merged away, and `expert_count` says
so. The result is a normal MoE pack that `tools.expert_merge.ggufmoe` reads and `tools.expert_merge.moe` runs, which
is what makes the before/after comparison two models and not two settings.

    python -m tools.expert_merge.mergepack --pack SRC.gguf --routes routes.npz --groups 64 --out MERGED.gguf
    python -m tools.expert_merge.mergepack --pack SRC.gguf --routes routes.npz --groups 64 --out M.gguf \
        --gate --corpus docs/BENCHMARKS.md,kindi/tp.py --docs 2 --tokens 48 --json merge.json
    python -m tools.expert_merge.mergepack --selftest

Three decisions the code makes, and why:

**The grouping is one partition for the whole model**, from the layer-averaged co-activation matrix, by average
linkage with a size cap `ceil(experts / groups)`. A per-layer partition would be a different model per layer and
would make "the merge" 48 separate experiments; a single partition is what a deployable pack can be.

**The average is weighted by routing mass, not by a Fisher.** `merge.py` implements the specification's
Fisher-weighted, permutation-aligned fusion, and `docs/EXPERT-MERGE.md` records that its own alignment gate rejects
every real pair (mean aligned cosine 0.081 against a 0.85 gate): the hidden units of two co-activated experts are not
one function relabelled, so permuting them to an anchor aligns noise. The arithmetic here is therefore the plain
weighted mean - the case the specification falls back to - with the weight being how much of the layer's routing
weight each member actually carried in the log. A member nothing routes to contributes nothing; every member
contributes in proportion to the mass it was applied with.

**The router row of a group is the same weighted mean of its members' rows.** That is the choice a merged pack forces:
the new expert has one row, so the logits of the experts it replaces have to be pooled. It changes routing as well as
arithmetic, and the gate sees both together - the note reports that rather than pretending the router is untouched.
"""
import json
import math
import os
import sys

import numpy as np

from tools.bitnet_gemm.cluster_budget import coactivation, merged_mass

from .ggufmoe import Moepack
from .prune import mass_per_expert

EXPERT_STACKS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
ROUTER = "ffn_gate_inp.weight"


# --- grouping -------------------------------------------------------------------------------------------------

def groups(coc, n_groups, log=None):
    """Average-linkage agglomeration on the co-activation matrix, size-capped, deterministic.

    Merging the two groups with the highest mean co-activation between their members, and never letting a group grow
    past `ceil(experts / groups)`, gives a partition whose groups are as co-activated as a greedy rule can make them
    and whose sizes cannot run away (one huge group and 63 singletons would be a legal k-means answer and a useless
    merge). Ties break on the lower member id, so the answer does not depend on dictionary order.
    """
    coc = np.asarray(coc, dtype=np.float64)
    n = coc.shape[0]
    n_groups = int(n_groups)
    if not 1 <= n_groups <= n:
        raise SystemExit("mergepack: %d groups from %d experts is not a partition" % (n_groups, n))
    cap = int(math.ceil(n / float(n_groups)))
    members = [[i] for i in range(n)]
    while len(members) > n_groups:
        best = None
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if len(members[i]) + len(members[j]) > cap:
                    continue
                s = float(coc[np.ix_(members[i], members[j])].mean())
                if best is None or s > best[0] + 1e-12:
                    best = (s, i, j)
        if best is None:                                  # the cap cannot be met; the only way out is to raise it
            cap += 1
            continue
        _s, i, j = best
        members[i] = sorted(members[i] + members[j])
        members.pop(j)
    order = sorted(range(len(members)), key=lambda k: (-len(members[k]), members[k][0]))
    gids = np.zeros(n, dtype=np.int64)
    for new, k in enumerate(order):
        for e in members[k]:
            gids[e] = new
    if log:
        sizes = np.bincount(gids, minlength=n_groups)
        log("  %d experts -> %d groups (sizes %d-%d, cap %d)" % (n, n_groups, sizes.min(), sizes.max(), cap))
    return gids


def merge_matrices(mats, weights):
    """The weighted mean of a group's matrices, with the weights normalised inside the group.

    `mats` are [out, in] arrays (or [in] vectors for a router row); `weights` the routing mass each member carried.
    A group whose members carried no mass at all - possible for experts the log never routed to - falls back to a
    plain mean, because the alternative is dividing by zero and dropping the expert entirely.
    """
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    w = (w / s) if s > 0 else np.full(w.size, 1.0 / w.size)
    out = np.zeros_like(mats[0], dtype=np.float64)
    for wi, m in zip(w, mats):
        out += wi * np.asarray(m, dtype=np.float64)
    return out.astype(np.float32)


# --- writing the pack ------------------------------------------------------------------------------------------

def _value_type(v):
    from gguf import GGUFValueType

    if isinstance(v, bool):
        return GGUFValueType.BOOL, None
    if isinstance(v, str):
        return GGUFValueType.STRING, None
    if isinstance(v, float):
        return GGUFValueType.FLOAT32, None
    if isinstance(v, int):
        return (GGUFValueType.UINT32, None) if v >= 0 else (GGUFValueType.INT32, None)
    if isinstance(v, (list, tuple)):
        vals = list(v)
        if vals and all(isinstance(x, str) for x in vals):
            return GGUFValueType.ARRAY, GGUFValueType.STRING
        if vals and all(isinstance(x, bool) for x in vals):
            return GGUFValueType.ARRAY, GGUFValueType.BOOL
        if vals and all(isinstance(x, float) for x in vals):
            return GGUFValueType.ARRAY, GGUFValueType.FLOAT32
        neg = any(isinstance(x, int) and x < 0 for x in vals)
        return GGUFValueType.ARRAY, (GGUFValueType.INT32 if neg else GGUFValueType.UINT32)
    raise SystemExit("mergepack: cannot write a metadata value of type %s" % type(v).__name__)


def writable(ty):
    """Can this build of `gguf` actually *write* that type?

    `GGML_QUANT_SIZES` and the decode path cover every type a real pack contains; `quants.quantize` does not - this
    build raises `NotImplementedError` for every K-quant. A pack that must be re-quantised therefore cannot be written
    in the type its source used, which is why the merge runs at one working type for both arms (`--expert-type`)
    instead of at the source's: a merged arm written more precisely than its control would look better than it is.
    """
    from gguf import GGMLQuantizationType as T
    from gguf import quants as Q

    t = T(int(ty))
    if t in (T.F32, T.F16, T.BF16):
        return True
    try:
        Q.quantize(np.zeros((2, 256), dtype=np.float32), t)
    except NotImplementedError:
        return False
    return True


def _add(writer, name, ne, ty, flat):
    """One tensor, quantised to the type it had in the source.

    `ne` is GGUF's own order (fastest-first) and `flat` is the data in that order - which is the same bytes as the
    NumPy array's C order for the reversed shape. `writer.add_tensor` takes the *NumPy* shape and reverses it when
    it writes the header, so the shape handed over is `reversed(ne)`; passing `ne` here transposes every tensor, and a
    transposed weight still loads and still predicts something (which is why the selftest reads the file back).
    """
    from gguf import GGMLQuantizationType as T
    from gguf import quants as GQ

    t = T(int(ty))
    if t == T.F32:
        arr = np.ascontiguousarray(flat, dtype=np.float32).ravel()
    elif t == T.F16:
        arr = np.ascontiguousarray(flat, dtype=np.float16).ravel()
    else:
        cols = int(ne[0])
        q = GQ.quantize(np.ascontiguousarray(flat, dtype=np.float32).reshape(-1, cols), t)
        arr = np.ascontiguousarray(q).view(np.uint8).ravel()
    writer.add_tensor(name, arr, raw_shape=tuple(int(d) for d in reversed(ne)), raw_dtype=t)


def write_pack(src, dst, gids, mass, expert_type=None, used=None, log=print, progress=None):
    """Write `dst` as `src` with each group of experts replaced by one mass-weighted average.

    Non-expert tensors are copied as bytes, not re-quantised: a copy that dequantised and re-quantised the attention
    would put the source's own quantisation error into the comparison. Everything that changes is in the expert
    stacks, the router, and `expert_count`.
    """
    from gguf import GGUFWriter

    pack = Moepack(src)
    g = pack.g
    gids = np.asarray(gids, dtype=np.int64)
    n_groups = int(gids.max()) + 1
    n_expert = pack.n_expert
    if gids.size != n_expert:
        raise SystemExit("mergepack: %d group ids for %d experts" % (gids.size, n_expert))
    members = [np.nonzero(gids == k)[0] for k in range(n_groups)]

    et = int(expert_type) if expert_type is not None else None
    if et is not None and not writable(et):
        raise SystemExit("mergepack: this gguf cannot write type %d; pick one it can (e.g. Q8_0, Q5_0, Q4_0, F16)" % et)
    kv = dict(g.kv)
    kv["%s.expert_count" % pack.arch] = n_groups
    if used is not None:
        # how many experts a token reads is a routing decision, so a pack can carry fewer groups than the source's k
        # "same fraction" arm; it is written here rather than patched afterwards when the merge is being built anyway
        if not 1 <= int(used) <= n_groups:
            raise SystemExit("mergepack: --used %d of %d groups is not a routing breadth this pack can have"
                             % (int(used), n_groups))
        kv["%s.expert_used_count" % pack.arch] = int(used)
    writer = GGUFWriter(dst, pack.arch, use_temp_file=True)   # this writes general.architecture itself
    for k in sorted(kv):
        if k == "general.architecture":
            continue
        vtype, sub = _value_type(kv[k])
        writer.add_key_value(k, kv[k], vtype, sub)

    copied = 0
    for name, (ne, ty, off) in sorted(pack.info.items(), key=lambda kv2: kv2[1][2]):
        layer = None
        if name.startswith("blk."):
            layer = int(name.split(".")[1])
        short = ".".join(name.split(".")[2:]) if layer is not None else name
        if short in EXPERT_STACKS and layer is not None:
            n_in, n_out, n_exp = ne
            if n_exp != n_expert:
                raise SystemExit("mergepack: %s stacks %d experts, the metadata says %d" % (name, n_exp, n_expert))
            merged = np.zeros((n_groups, n_out, n_in), dtype=np.float32)
            m = mass[layer] if layer < mass.shape[0] else np.zeros(n_expert)
            for k, mem in enumerate(members):
                mats = [pack.expert_weight(name, int(e)) for e in mem]
                merged[k] = merge_matrices(mats, m[mem])
            _add(writer, name, (n_in, n_out, n_groups), et if et is not None else ty, merged.ravel())
        elif short == ROUTER and layer is not None:
            n_in, n_exp = ne
            rows = [pack.tensor(name)[int(e)] for e in range(n_exp)]       # [d_model] each
            merged = np.stack([merge_matrices([rows[e] for e in mem], mass[layer][mem]) for mem in members])
            _add(writer, name, (n_in, n_groups), et if et is not None else ty, merged.ravel())
        else:
            span = pack._span(ty, int(np.prod(ne)))
            start = g.data_start + off
            raw = np.ascontiguousarray(np.asarray(pack._raw[start:start + span]))
            writer.add_tensor(name, raw, raw_shape=tuple(int(d) for d in reversed(ne)), raw_dtype=from_type(ty))
            copied += int(span)
        if progress and layer is not None and short == EXPERT_STACKS[-1]:
            progress(layer)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dict(path=dst, groups=n_groups, bytes=int(os.path.getsize(dst)),
                source_bytes=int(os.path.getsize(src)), copied_bytes=int(copied))


def from_type(ty):
    from gguf import GGMLQuantizationType as T

    return T(int(ty))


# --- the comparison --------------------------------------------------------------------------------------------

def compare_arms(base_pack, arms, docs, max_delta=0.02, block=8, log=print):
    """One baseline walk, one walk a pack, one verdict each.

    The baseline is loaded and scored once, which is what makes several arms affordable: the Q4_0 control is 17 GB and
    each merged arm 8.9 GB, while the 94 scored tokens are the cheap part. Every walk sees the same tokenisation and
    the same positions, so an arm's difference from the baseline is comparable with another arm's.
    """
    from .moe import Qwen3Moe
    from .prune import gate, nll_moe

    b = Qwen3Moe(base_pack, log=lambda *a: None)
    per, b_docs = [], []
    for name, ids in docs:
        r = nll_moe(b, ids)
        per.append(r["per_token"])
        b_docs.append((name, r["nll"], r["tokens"]))
        log("    baseline %-24s %8d tokens  nll %.4f" % (name, r["tokens"], r["nll"]))
    del b
    baseline = np.concatenate(per)

    out = []
    for path in arms:
        arm = Qwen3Moe(path, log=lambda *a: None)
        aps, a_docs = [], []
        for name, ids in docs:
            r = nll_moe(arm, ids)
            aps.append(r["per_token"])
            a_docs.append((name, r["nll"], r["tokens"]))
            log("    %-12s %-24s %8d tokens  nll %.4f" % (os.path.basename(path)[:12], name, r["tokens"], r["nll"]))
        del arm
        after = np.concatenate(aps)
        v = gate(dict(per_token=baseline), [dict(per_token=after)], max_delta=max_delta, block=block)[0]
        out.append(dict(pack=path, baseline_nll=float(baseline.mean()), merged_nll=float(after.mean()),
                        tokens=int(baseline.size), baseline_docs=b_docs, merged_docs=a_docs, gate=v))
    return out


def compare(src, merged, docs, max_delta=0.02, block=8, log=print):
    """NLL of both packs on the same tokens, paired token by token, with a moving-block interval.

    One process, two models, the same `forward`: the source is the baseline and the merged pack is the arm, and the
    difference is indexed by position so a document cannot contribute to one side only.
    """
    return compare_arms(src, [merged], docs, max_delta=max_delta, block=block, log=log)[0]


# --- routing breadth --------------------------------------------------------------------------------------------

_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _kv_uint32_offset(path, key):
    """The byte offset of a uint32 metadata value, read from the header alone - so it can be rewritten in place.

    A GGUF key/value is a length-prefixed key, a type, then the value; only a scalar value can be rewritten without
    shifting everything after it, which is why this insists on type 4 and refuses a string or an array.
    """
    import struct

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("mergepack: %s is not a GGUF file" % path)
        ver = struct.unpack("<I", f.read(4))[0]
        if ver not in (2, 3):
            raise SystemExit("mergepack: %s is GGUF v%d, and this reader knows v2 and v3" % (path, ver))
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            n = struct.unpack("<Q", f.read(8))[0]
            k = f.read(n).decode("utf-8", errors="replace")
            ty = struct.unpack("<I", f.read(4))[0]
            pos = f.tell()
            if k == key:
                if ty != 4:
                    raise SystemExit("mergepack: %s holds %r as metadata type %d, not a uint32" % (path, key, ty))
                return pos
            _skip_value(f, ty, k, path)
    raise SystemExit("mergepack: %s has no %r in its header, so there is no routing breadth to rewrite" % (path, key))


def _skip_value(f, ty, key, path):
    import struct

    if ty == 8:                                              # string
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        return
    if ty == 9:                                              # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n):
                f.seek(struct.unpack("<Q", f.read(8))[0], 1)
            return
        size = _SCALAR_BYTES.get(et)
        if size is None:
            raise SystemExit("mergepack: %s: array %r has element type %d, which this reader cannot skip"
                             % (path, key, et))
        f.seek(size * n, 1)
        return
    size = _SCALAR_BYTES.get(ty)
    if size is None:
        raise SystemExit("mergepack: %s: %r has metadata type %d, which this reader cannot skip" % (path, key, ty))
    f.seek(size, 1)


def retopk(src, dst, used, log=print):
    """The same pack at a different routing breadth: every byte copied, `expert_used_count` rewritten.

    k is a header field, not a weight. Changing it changes how many of the (already merged) experts a token reads and
    renormalises over, and nothing else in the file moves - so a paired walk of the copy against its source prices
    routing breadth with the merge held fixed. That is the arm the k/N asymmetry needs: 8 of 128 experts and 4 of 64
    are the same fraction of the pool.
    """
    import shutil
    import struct

    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("mergepack: --retopk writes a new pack; it does not patch its source in place")
    pack = Moepack(src)
    n, used = pack.n_expert, int(used)
    if not 1 <= used <= n:
        raise SystemExit("mergepack: %d of %d experts is not a routing breadth this pack can have" % (used, n))
    key = "%s.expert_used_count" % pack.arch
    off = _kv_uint32_offset(src, key)
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(off)
        f.write(struct.pack("<I", used))
    out = Moepack(dst)
    if (out.n_expert, out.n_expert_used) != (n, used):
        raise SystemExit("mergepack: %s reads back as %d of %d experts, not %d of %d"
                         % (dst, out.n_expert_used, out.n_expert, used, n))
    if os.path.getsize(dst) != os.path.getsize(src):
        raise SystemExit("mergepack: %s is %d bytes where its source is %d, so a copy went wrong"
                         % (dst, os.path.getsize(dst), os.path.getsize(src)))
    if not out.payload_report()["ok"]:
        raise SystemExit("mergepack: %s does not decode after the rewrite" % dst)
    log("  %s: %d experts, %d used -> %d used; %d bytes, every tensor still decodes"
        % (dst, n, pack.n_expert_used, used, os.path.getsize(dst)))
    return dict(path=dst, source=src, experts=n, used_before=pack.n_expert_used, used_after=used,
                bytes=int(os.path.getsize(dst)), offset=int(off))


# --- the checks ------------------------------------------------------------------------------------------------

def _synthetic_pack(path, n_expert=4, d=32, ff=16, head=2, hkv=1, hd=16, vocab=64, seed=0):
    """A one-layer qwen3moe GGUF small enough to check byte for byte, written with the same writer the merge uses."""
    from gguf import GGUFWriter

    rs = np.random.RandomState(seed)
    kv = {"general.architecture": "qwen3moe", "general.name": "mergepack-selftest",
          "qwen3moe.block_count": 1, "qwen3moe.embedding_length": d, "qwen3moe.feed_forward_length": 2 * ff,
          "qwen3moe.attention.head_count": head, "qwen3moe.attention.head_count_kv": hkv,
          "qwen3moe.attention.key_length": hd, "qwen3moe.attention.value_length": hd,
          "qwen3moe.expert_count": n_expert, "qwen3moe.expert_used_count": min(2, n_expert),
          "qwen3moe.expert_feed_forward_length": ff, "qwen3moe.rope.freq_base": 10000.0,
          "qwen3moe.attention.layer_norm_rms_epsilon": 1e-6, "tokenizer.ggml.model": "gpt2"}
    w = GGUFWriter(path, "qwen3moe", use_temp_file=True)
    for k in sorted(kv):
        if k == "general.architecture":
            continue
        vtype, sub = _value_type(kv[k])
        w.add_key_value(k, kv[k], vtype, sub)
    n = lambda *shape: (rs.randn(*shape) * 0.05).astype(np.float32)
    tensors = [("token_embd.weight", n(vocab, d), (d, vocab)),
               ("output_norm.weight", (np.ones(d) * 0.5).astype(np.float32), (d,)),
               ("output.weight", n(vocab, d), (d, vocab))]
    s = "blk.0."
    tensors += [(s + "attn_q.weight", n(head * hd, d), (d, head * hd)),
                (s + "attn_k.weight", n(hkv * hd, d), (d, hkv * hd)),
                (s + "attn_v.weight", n(hkv * hd, d), (d, hkv * hd)),
                (s + "attn_output.weight", n(d, head * hd), (head * hd, d)),
                (s + "attn_q_norm.weight", (np.ones(hd) * 0.4).astype(np.float32), (hd,)),
                (s + "attn_k_norm.weight", (np.ones(hd) * 0.4).astype(np.float32), (hd,)),
                (s + "attn_norm.weight", (np.ones(d) * 0.6).astype(np.float32), (d,)),
                (s + "ffn_norm.weight", (np.ones(d) * 0.6).astype(np.float32), (d,)),
                (s + "ffn_gate_inp.weight", n(n_expert, d), (d, n_expert)),
                (s + "ffn_gate_exps.weight", n(n_expert, ff, d), (d, ff, n_expert)),
                (s + "ffn_up_exps.weight", n(n_expert, ff, d), (d, ff, n_expert)),
                (s + "ffn_down_exps.weight", n(n_expert, d, ff), (ff, d, n_expert))]
    for name, arr, ne in tensors:
        w.add_tensor(name, np.ascontiguousarray(arr).ravel(), raw_shape=tuple(reversed(ne)),
                     raw_dtype=from_type(0))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def selftest():
    import shutil
    import tempfile

    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    # --- the grouping
    coc = np.zeros((4, 4))
    for a, b in ((0, 1), (2, 3)):
        coc[a, b] = coc[b, a] = 9.0
        coc[a, a] = coc[b, b] = 9.0
    coc += np.eye(4)
    g = groups(coc, 2, log=None)
    check("mergepack: average linkage finds the two co-activated pairs",
          sorted(np.nonzero(g == g[0])[0].tolist()) == [0, 1] or sorted(np.nonzero(g == g[0])[0].tolist()) == [1, 0],
          str(g))
    check("mergepack: and they are the ones that co-fire", {tuple(sorted(np.nonzero(g == k)[0])) for k in set(g.tolist())}
          == {(0, 1), (2, 3)}, str(g))
    check("mergepack: every group is used and every expert is in one",
          len(set(g.tolist())) == 2 and np.bincount(g).sum() == 4, str(g))
    sprawling = groups(coc, 3, log=None)
    sizes = np.bincount(sprawling)
    check("mergepack: a size cap keeps the groups even at a group count that does not divide", sizes.max() <= 2,
          str(sizes))
    try:
        groups(coc, 5, log=None)
    except SystemExit:
        pass
    else:
        raise AssertionError("more groups than experts must be refused")

    # --- the fusion arithmetic
    a, b = np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32)
    check("mergepack: one member is itself", np.allclose(merge_matrices([a], [5.0]), a))
    check("mergepack: zero weights fall back to a plain mean", np.allclose(merge_matrices([a, b], [0.0, 0.0]), 0.5))
    check("mergepack: mass weights decide the mean", np.allclose(merge_matrices([a, b], [3.0, 1.0]), 0.25))
    check("mergepack: the result is f32, not f64", merge_matrices([a, b], [1.0, 1.0]).dtype == np.float32)

    # --- which types this build can actually write, since that decides what a matched pair of arms can be
    from gguf import GGMLQuantizationType as T

    check("mergepack: the types a matched pair of arms can use are writable",
          writable(int(T.Q4_0)) and writable(int(T.F16)) and writable(int(T.F32)))
    check("mergepack: and the writer's answer is the quantiser's, not a guess",
          writable(int(T.Q8_0)) == True, "Q8_0 must round-trip through quantize/dequantize")

    # --- a real file through the real writer, read back by the real reader
    td = tempfile.mkdtemp(prefix="mergepack-")
    try:
        src = _synthetic_pack(os.path.join(td, "src.gguf"))
        mass = np.array([[4.0, 2.0, 1.0, 1.0]])            # one layer, four experts
        gids = np.array([0, 0, 1, 1])
        dst = os.path.join(td, "merged.gguf")
        rep = write_pack(src, dst, gids, mass, log=lambda *a: None)
        check("mergepack: the written pack has one expert a group", rep["groups"] == 2, str(rep))
        m = Moepack(dst)
        check("mergepack: and its metadata says so", m.n_expert == 2, str(m.n_expert))
        check("mergepack: the rest of the metadata survived", m.n_embd == 32 and m.n_ff_exp == 16 and m.n_layer == 1)
        check("mergepack: every tensor decodes at its new shape",
              m.payload_report()["ok"] and m.expert_weight("blk.0.ffn_gate_exps.weight", 1).shape == (16, 32))
        src_pack = Moepack(src)
        w0 = src_pack.expert_weight("blk.0.ffn_gate_exps.weight", 0)
        w1 = src_pack.expert_weight("blk.0.ffn_gate_exps.weight", 1)
        got = m.expert_weight("blk.0.ffn_gate_exps.weight", 0)
        want = (4.0 * w0 + 2.0 * w1) / 6.0
        check("mergepack: group 0 is the mass-weighted mean of its members", np.allclose(got, want, atol=1e-6),
              "max diff %.3g" % float(np.abs(got - want).max()))
        other = m.expert_weight("blk.0.ffn_gate_exps.weight", 1)
        check("mergepack: and group 1 is a different matrix", not np.allclose(got, other))
        check("mergepack: the file is smaller than its source", rep["bytes"] < rep["source_bytes"])

        # the merged pack is a model: it loads and its forward produces finite logits
        from .moe import Qwen3Moe

        ids = [3, 5, 7, 11]
        before = Qwen3Moe(src, log=lambda *a: None).forward(ids)["logits"]
        after = Qwen3Moe(dst, log=lambda *a: None).forward(ids)["logits"]
        check("mergepack: the source forward is finite", np.isfinite(before).all())
        check("mergepack: the merged forward is finite too", np.isfinite(after).all())
        check("mergepack: and it is not the same model any more", not np.allclose(before, after, atol=1e-4))

        # routing breadth is a header field: the same weights at another k, with nothing else moving
        k1 = os.path.join(td, "merged-k1.gguf")
        rr = retopk(dst, k1, 1)
        m1 = Moepack(k1)
        with open(dst, "rb") as f:
            raw = f.read()
        with open(k1, "rb") as f:
            raw2 = f.read()
        moved = [i for i in range(len(raw)) if raw[i] != raw2[i]]
        check("mergepack: --retopk moves one four-byte field and nothing else",
              m1.n_expert == 2 and m1.n_expert_used == 1 and rr["used_before"] == 2 and len(raw) == len(raw2)
              and moved and min(moved) >= rr["offset"] and max(moved) < rr["offset"] + 4,
              "%d bytes differ" % len(moved))
        check("mergepack: and the copy still decodes and runs at its new breadth",
              m1.payload_report()["ok"]
              and np.isfinite(Qwen3Moe(k1, log=lambda *a: None).forward(ids)["logits"]).all())
        try:
            retopk(dst, os.path.join(td, "too-wide.gguf"), 3)
            refused = ""
        except SystemExit as e:
            refused = str(e)
        check("mergepack: a breadth wider than the expert count is refused", "3 of 2 experts" in refused, refused[:60])

        # the gate over existing packs: one baseline walk, one a pack, a verdict each
        reps = compare_arms(src, [src], [("synthetic", ids)], log=lambda *a: None)
        check("mergepack: --compare walks the baseline and every arm, and a pack against itself is zero",
              len(reps) == 1 and reps[0]["gate"]["delta"] == 0.0 and reps[0]["tokens"] > 0, str(reps[0]["gate"]))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return 0


# --- the CLI ---------------------------------------------------------------------------------------------------

def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main(argv):
    if "--selftest" in argv:
        selftest()
        print("checks passed")
        return 0
    if "--retopk" in argv:
        src = _arg(argv, "--retopk")
        dst, used = _arg(argv, "--out"), _arg(argv, "--used")
        if not (src and dst and used):
            print("usage: python -m tools.expert_merge.mergepack --retopk PACK.gguf --used 4 --out PACK-k4.gguf")
            return 2
        retopk(src, dst, used)
        return 0

    if "--compare" in argv:
        base = _arg(argv, "--compare")
        arms = [p for p in (_arg(argv, "--against") or "").split(",") if p.strip()]
        names = [n for n in (_arg(argv, "--corpus") or "").split(",") if n.strip()]
        if not (base and arms and names):
            print("usage: python -m tools.expert_merge.mergepack --compare BASE.gguf --against A.gguf[,B.gguf] "
                  "--corpus A,B --docs 2 --tokens 48 --json out.json")
            return 2
        from .moe import tokenizer
        from .routes import corpus

        docs = corpus(tokenizer(Moepack(base)), names, per_doc=int(_arg(argv, "--tokens", 48)))[:int(_arg(argv, "--docs", 2))]
        if not docs:
            raise SystemExit("mergepack --compare: no corpus to gate on")
        print("  gate corpus: %s" % ", ".join("%s(%d tokens)" % (n, len(i)) for n, i in docs))
        reps = compare_arms(base, arms, docs, max_delta=float(_arg(argv, "--max-delta", 0.02)))
        for r in reps:
            v = r["gate"]
            print("  %s: nll %.4f -> %.4f (%+.4f, 95%% CI %+.4f..%+.4f) over %d tokens, tolerance %+.3f: %s"
                  % (os.path.basename(r["pack"]), r["baseline_nll"], r["merged_nll"], v["delta"], v["ci_lo"],
                     v["ci_hi"], v["tokens"], v["max_delta"], "ACCEPTED" if v["passed"] else "REJECTED"))
        js = _arg(argv, "--json")
        if js:
            with open(js, "w", encoding="utf-8") as f:
                json.dump(dict(baseline=base, corpus=names, per_doc=int(_arg(argv, "--tokens", 48)), arms=reps),
                          f, indent=1, sort_keys=True)
            print("  wrote %s" % js)
        return 0

    src = _arg(argv, "--pack")
    routes_path = _arg(argv, "--routes")
    out = _arg(argv, "--out")
    if not (src and routes_path and out):
        print("usage: python -m tools.expert_merge.mergepack --pack SRC.gguf --routes ROUTES.npz --groups 64 "
              "--out MERGED.gguf [--used N] [--gate --corpus A,B --docs 2 --tokens 48 --json out.json]\n"
              "       python -m tools.expert_merge.mergepack --retopk PACK.gguf --used 4 --out PACK-k4.gguf\n"
              "       python -m tools.expert_merge.mergepack --compare BASE.gguf --against A.gguf[,B.gguf] "
              "--corpus A,B --docs 2 --tokens 48 --json out.json [--selftest]")
        return 2
    from .routes import corpus, load

    got = load(routes_path)
    meta = got.get("meta", {})
    n_expert = int(meta.get("experts", got["route_ids"].max() + 1))
    mass, _count = mass_per_expert(got, n_expert)
    n_groups = int(_arg(argv, "--groups", 64))
    print("  %s: %d layers, %d experts, %d tokens logged" % (os.path.basename(src), mass.shape[0], n_expert,
                                                             got["route_wts"].shape[1]))
    coc = coactivation(got["route_ids"], got["route_wts"], np.arange(got["route_wts"].shape[1]), n_expert)
    gids = groups(coc + coc.T, n_groups)
    exposed = merged_mass(gids[got["route_ids"].astype(np.int64)], got["route_wts"].astype(np.float32))
    print("  a token's two slots land in one group %.1f%% of the time (the share the merge averages)"
          % (100.0 * exposed))

    et_name = _arg(argv, "--expert-type")
    et = None
    if et_name:
        from gguf import GGMLQuantizationType as T

        et = int(T[et_name])
    rep = write_pack(src, out, gids, mass, expert_type=et, used=_arg(argv, "--used"),
                     progress=lambda il: il % 8 == 7 and print("    merged through layer %d" % il))
    if et_name:
        print("  expert stacks and router written as %s (the source's K-quants cannot be re-quantised by this build)"
              % et_name)
    print("  wrote %s: %.2f GB from %.2f GB (%.0f%%), %d experts -> %d groups"
          % (out, rep["bytes"] / 1e9, rep["source_bytes"] / 1e9,
             100.0 * rep["bytes"] / rep["source_bytes"], n_expert, n_groups))

    payload = dict(source=src, merged=out, groups=n_groups, gids=[int(x) for x in gids],
                   merged_mass=exposed, write=rep, expert_type=et_name or "as the source",
                   used=int(_arg(argv, "--used")) if _arg(argv, "--used") else None)
    if "--gate" in argv:
        from .moe import tokenizer

        pack = Moepack(src)
        tok = tokenizer(pack)
        names = [n for n in (_arg(argv, "--corpus") or "").split(",") if n.strip()]
        if not names:
            names = [d[0] for d in meta.get("docs", [])]
            print("  gate corpus: the log's own documents (in-sample)")
        docs = corpus(tok, names, per_doc=int(_arg(argv, "--tokens", 48)))[:int(_arg(argv, "--docs", 2))]
        if not docs:
            raise SystemExit("mergepack: no corpus to gate on")
        print("  gate corpus: %s" % ", ".join("%s(%d tokens)" % (n, len(i)) for n, i in docs))
        base = _arg(argv, "--baseline", src)
        if base != src:
            print("  baseline pack: %s (not the source: the two arms share one quantisation)" % base)
        c = compare(base, out, docs, max_delta=float(_arg(argv, "--max-delta", 0.02)))
        payload["gate"] = c
        v = c["gate"]
        print("  nll %.4f -> %.4f (%+.4f, 95%% CI %+.4f..%+.4f) over %d tokens, tolerance %+.3f: %s"
              % (c["baseline_nll"], c["merged_nll"], v["delta"], v["ci_lo"], v["ci_hi"], v["tokens"],
                 v["max_delta"], "ACCEPTED" if v["passed"] else "REJECTED"))
    js = _arg(argv, "--json")
    if js:
        with open(js, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
        print("  wrote %s" % js)
    gated = payload.get("gate")
    return 0 if (not gated or gated["gate"]["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
