"""kindi refit: keep every code, re-derive every block scale against a target.

A pack's ternary tensors are two things: the trits (codes) and one fp16 scale per 128-weight block. Only the second is
a fit. Give the same codes better scales and the weights get closer to a target without moving a single trit, which
costs nothing at all: the codes are what the engine reads, and the scale file is 1-2 bytes a block.

    from kindi.refit import refit
    refit(src_pack, dst_pack, target_pack)

What the solve is: for a block of weights `w` and its fixed codes `c` in {-1, 0, +1}, the scale that minimises
`|w - c*s|^2` closed-form is `s = <w, c> / <c, c>` (a dot product divided by the number of non-zero codes). It is the
plain least-squares projection, not a search: one pass over the tensor, no training, no codes rewritten, and the block
whose codes are all zero has no direction to fit, so it keeps the scale it came in with. A negative solution is
clipped to zero rather than written, because a pack's block scale is a magnitude (`packio.level_table` refuses the
negative), and that block is counted in the report instead of being silently flipped.

What the target is: another pack of the same model, at any precision. A ternary tensor of the target is read back
through its own codes and scales (`dequantise`), so a refit against a pack prices "these codes, that build's weights";
a target makes no claim beyond that. Its float tensors are not read at all, because a refit only touches scale blocks.

The target's codes are compared with the source's while streaming, and the count of differing bytes goes in the report
and the manifest: if the codes are identical, the measurement is about the scales alone, and that is worth knowing
before anyone reads a delta as a recipe difference.

Written files: `dpm.bin` and `dscale.bin` are new; `bcodes.bin`, `f32.bin`, `signs_*.f32`, `vocab.json`, `merges.txt`
are byte copies of the source (`--link` hard-links them instead, which costs nothing on one filesystem and is what a 27B
refit wants); `index.json` and `bcodes_index.json` keep every offset and gain this build's `dtab_off`/`dtab_n`;
`kindi_pack.json` carries a `refit` block naming the target, the method, and what changed. A source with `pcodes.bin` is
refused rather than copied, because that file stages its own copy of the scales and would go stale.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

from . import packio

GROUP = 128
COPY = ("bcodes.bin", "f32.bin", "vocab.json", "merges.txt")     # pcodes.bin is refused, not copied: it stages scales


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def trits(codes, C):
    """a tensor's stored base-3 bytes ([C/4][R], four trits a byte) -> int8 (R, C) in {-1, 0, +1}"""
    b = np.asarray(codes, dtype=np.uint8).T.reshape(-1, C // 4)
    t = np.stack([b % 3, (b // 3) % 3, (b // 9) % 3, (b // 27) % 3], axis=2).reshape(b.shape[0], C)
    return (t - np.uint8(1)).astype(np.int8)


def solve(c, w):
    """the closed form per block: <w, c> and the code mass <c, c> it is divided by, both (R, C/128) float64"""
    n, C = c.shape
    G, k = C // GROUP, GROUP
    cg = c.reshape(n, G, k).astype(np.float32)
    wg = w.reshape(n, G, k).astype(np.float32)
    return ((cg * wg).sum(axis=2, dtype=np.float64), (cg * cg).sum(axis=2, dtype=np.float64))


def fit(c, w, prior):
    """least-squares block scale for fixed codes: (R, C) int8 codes + (R, C) target -> (R, C/128) float64

    `prior` is what a block keeps when it has no codes to fit (all-zero trits) and the shape of the return.
    A negative solution is clipped to 0 rather than written; `solve` is exposed so a caller can count those.
    """
    num, den = solve(c, w)
    s = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.asarray(prior, dtype=np.float64))
    return np.maximum(s, 0.0)


class Pack(object):
    """the three records a refit reads, and memmapped access to the bytes they point at"""

    def __init__(self, path):
        self.path = path
        self.man = _json(os.path.join(path, "kindi_pack.json"))
        self.ix = _json(os.path.join(path, "index.json"))
        self.bi = _json(os.path.join(path, "bcodes_index.json"))["tensors"]
        self.fmt = int(self.man.get("pack_format") or 1)
        self._bc = self._dpm = self._tabs = self._f16 = None

    def codes(self, name):
        rec = self.bi[name]
        q, rows, C = int(rec["bc_off"]), int(rec["rows"]), int(rec["ne0"])
        n = rows * (C // 4)
        if self._bc is None:
            self._bc = np.memmap(os.path.join(self.path, "bcodes.bin"), dtype=np.uint8, mode="r")
        if q + n > self._bc.size:
            raise SystemExit("kindi refit: %s needs bytes %d..%d of %s/bcodes.bin, which holds %d"
                             % (name, q, q + n, self.path, self._bc.size))
        return self._bc[q:q + n].reshape(C // 4, rows), rows, C

    def scales(self, name):
        """one fp16 scale a block, in the pack's own flat order ([group][row])"""
        rec = self.bi[name]
        n = int(rec["rows"]) * (int(rec["ne0"]) // GROUP)
        q = packio.block_of(rec)
        if self.fmt < 2:
            if self._f16 is None:
                self._f16 = np.memmap(os.path.join(self.path, "dpm.bin"), dtype=np.float16, mode="r")
            return self._f16[q:q + n]
        if self._dpm is None:
            self._dpm = np.memmap(os.path.join(self.path, "dpm.bin"), dtype=np.uint8, mode="r")
        if self._tabs is None:
            self._tabs = np.memmap(os.path.join(self.path, "dscale.bin"), dtype=np.float16, mode="r")
        t0, tn = int(rec["dtab_off"]) // 2, int(rec["dtab_n"])
        if q + n > self._dpm.size or t0 + tn > self._tabs.size:
            raise SystemExit("kindi refit: %s points past dpm.bin or dscale.bin in %s" % (name, self.path))
        return self._tabs[t0:t0 + tn][self._dpm[q:q + n]]


def _v2_record(dpm, dtable, vals, dblk, dtab_off, *recs):
    """one tensor's scales as an 8-bit level table: write the index, keep the table, point every record at both.

    Returns (next dtab_off, table, index, exact). The refit and the retable both write through this, so whatever a
    paired walk of their output measures is the same storage path in both.
    """
    table, idx, exact = packio.level_table(vals)
    dpm.write(np.ascontiguousarray(idx).tobytes())
    dtable.append(table)
    for r in recs:
        r.update({"dblk": dblk, "dtab_off": dtab_off, "dtab_n": int(table.size)})
    return dtab_off + 2 * int(table.size), table, idx, exact


def _place(src, dst, link):
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def refit(src, dst, target, log=print, link=False, chunk=512):
    """every code copied, every block scale re-derived against `target`; returns the report dict"""
    t0 = time.time()
    s, t = Pack(src), Pack(target)
    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit("kindi refit: %s already holds files; a refit writes a new pack, it does not patch one" % dst)
    missing = [n for n in s.bi if n not in t.bi]
    if missing:
        raise SystemExit("kindi refit: the target pack has no %s, so %d of the source's ternary tensors have nothing "
                         "to be fitted against" % (", ".join(missing[:3]), len(missing)))
    # pcodes.bin stages an expert matrix with its own float32 copy of the block scales, so new scales in dpm.bin would
    # leave that copy stale and the experts running on the old ones: refuse rather than write a pack that disagrees
    # with itself
    if os.path.exists(os.path.join(src, "pcodes.bin")):
        raise SystemExit("kindi refit: %s holds pcodes.bin, which carries its own copy of the block scales for the "
                         "expert matrices and would go stale; convert without --paircodes first" % src)
    # codes live in the basis the Hadamard rotation put them in; a target rotated differently is a different coordinate
    # system, and a scale fitted across the two would be a number with no meaning attached
    if ((s.man.get("fold"), s.man.get("hadamard_block")) != (t.man.get("fold"), t.man.get("hadamard_block"))
            or s.ix.get("hadamard", {}).get("sign_widths") != t.ix.get("hadamard", {}).get("sign_widths")):
        raise SystemExit("kindi refit: %s and %s carry different Hadamard rotations, so their codes are not in the same "
                         "basis and a scale fitted across them means nothing" % (src, target))
    os.makedirs(dst, exist_ok=True)
    v2 = s.fmt >= 2
    dtable, dtab_off, dblk = [], 0, 0
    tot = dict(blocks=0, kept=0, clipped=0, code_bytes=0, code_bytes_differing=0, weights=0,
               sq_before=0.0, sq_after=0.0, ref_sq=0.0, tensors=0, max_levels=0, exact=0, max_rel=0.0)
    worst, keep_first = [], None
    dpm = open(os.path.join(dst, "dpm.bin"), "wb")
    for i, name in enumerate(s.bi):
        # codes are stored [column-slot][row], so a row range is a strided read: pull each tensor in once, contiguously,
        # and slice in memory - a 248k-row embedding is 248k seeks if the walk is done through the memmap. `copy=True`
        # matters: np.asarray on a memmap view keeps the subclass, and the slices below would still be file-backed.
        codes, rows, C = s.codes(name)
        tcodes, trows, tC = t.codes(name)
        codes = np.array(codes, dtype=np.uint8, copy=True)
        tcodes = np.array(tcodes, dtype=np.uint8, copy=True)
        if (trows, tC) != (rows, C):
            raise SystemExit("kindi refit: %s is %dx%d in the source and %dx%d in the target"
                             % (name, rows, C, trows, tC))
        G = C // GROUP
        real = min(int(s.ix["tensors"][name]["dims"][0]), rows)
        prior = np.asarray(s.scales(name)).reshape(G, rows).T.astype(np.float64)      # (rows, G), [group][row]
        tsc = np.asarray(t.scales(name)).reshape(G, rows).T                           # (rows, G) fp16
        new = prior.copy()
        sq_b, sq_a, ref = 0.0, 0.0, 0.0
        for r0 in range(0, real, chunk):
            r1 = min(real, r0 + chunk)
            c = trits(codes[:, r0:r1], C)
            w = trits(tcodes[:, r0:r1], C).astype(np.float32) * tsc[r0:r1].astype(np.float32).repeat(GROUP, axis=1)
            block = prior[r0:r1]
            num, den = solve(c, w)
            f = np.where(den > 0, np.maximum(num / np.where(den > 0, den, 1.0), 0.0), block)
            new[r0:r1] = f
            tot["kept"] += int((den == 0).sum())
            tot["clipped"] += int(((den > 0) & (num < 0)).sum())
            deq_b = c.astype(np.float32) * block.astype(np.float32).repeat(GROUP, axis=1)
            deq_a = c.astype(np.float32) * f.astype(np.float32).repeat(GROUP, axis=1)
            sq_b += float(((w - deq_b) ** 2).sum())
            sq_a += float(((w - deq_a) ** 2).sum())
            ref += float((w ** 2).sum())
            tot["code_bytes"] += int(codes[:, r0:r1].nbytes)
            tot["code_bytes_differing"] += int((codes[:, r0:r1] != tcodes[:, r0:r1]).sum())
            tot["weights"] += (r1 - r0) * C
        tot["sq_before"] += sq_b
        tot["sq_after"] += sq_a
        tot["ref_sq"] += ref
        tot["blocks"] += rows * G
        tot["tensors"] += 1
        vals = np.ascontiguousarray(new.T.reshape(-1)).astype(np.float16)        # flat [group][row], as packs store it
        if v2:
            dtab_off, table, idx, exact = _v2_record(dpm, dtable, vals, dblk, dtab_off,
                                                     s.ix["tensors"][name], s.bi[name])
            tot["max_levels"] = max(tot["max_levels"], int(table.size))
            tot["exact"] += int(exact)
            # what this tensor's 8-bit table costs the scales it was fitted to, block by block - the file's own error,
            # not a bound derived from the value range
            back = packio.decode(table, idx).astype(np.float64)
            v64 = vals.astype(np.float64)
            ok = v64 > 0
            if ok.any():
                rel = np.abs(back[ok] - v64[ok]) / v64[ok]
                tot["max_rel"] = max(tot["max_rel"], float(rel.max()))
        else:
            dpm.write(np.ascontiguousarray(vals).tobytes())
        dblk += rows * G
        if i == 0:
            keep_first = (name, vals.copy())
        if ref:
            worst.append((1.0 - sq_a / sq_b if sq_b else 0.0, name, (sq_b / ref) ** 0.5, (sq_a / ref) ** 0.5))
        if (i + 1) % 40 == 0:
            log("  %d / %d tensors" % (i + 1, len(s.bi)))
    dpm.close()
    if v2:
        with open(os.path.join(dst, "dscale.bin"), "wb") as f:
            f.write(np.concatenate(dtable).astype(np.float16).tobytes())
    for name in sorted(os.listdir(src)):
        if name in COPY or name.startswith("signs_"):
            _place(os.path.join(src, name), os.path.join(dst, name), link)
    with open(os.path.join(dst, "index.json"), "w") as f:
        json.dump(s.ix, f)
    with open(os.path.join(dst, "bcodes_index.json"), "w") as f:
        json.dump({"tensors": s.bi}, f)
    man = dict(s.man)
    man["pack_format"] = 2 if v2 else 1
    man["dpm_bytes"] = os.path.getsize(os.path.join(dst, "dpm.bin"))
    if v2:
        man["block_scales"] = {"bits": packio.BITS, "file": "dscale.bin",
                               "bytes": os.path.getsize(os.path.join(dst, "dscale.bin")),
                               "ternary_tensors": tot["tensors"], "exact_tensors": tot["exact"],
                               "max_levels": tot["max_levels"], "max_relative_error": round(tot["max_rel"], 6)}
    man["refit"] = {"target": os.path.abspath(target), "target_model": t.man.get("model"),
                    "method": "s = <w, c> / <c, c> a 128-weight block, codes fixed, negative clipped to 0",
                    "codes": "unchanged", "code_bytes": tot["code_bytes"],
                    "code_bytes_differing": tot["code_bytes_differing"],
                    "code_bytes_differing_pct": round(100.0 * tot["code_bytes_differing"] /
                                                       max(tot["code_bytes"], 1), 4),
                    "blocks": tot["blocks"],
                    "blocks_without_codes": tot["kept"], "blocks_clipped_at_zero": tot["clipped"],
                    "rmse_relative_before": round((tot["sq_before"] / max(tot["ref_sq"], 1e-30)) ** 0.5, 8),
                    "rmse_relative_after": round((tot["sq_after"] / max(tot["ref_sq"], 1e-30)) ** 0.5, 8),
                    "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(dst, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    # read the file back through the same code every reader uses: the first tensor's blocks must decode to the fit
    out = Pack(dst)
    name, want = keep_first
    got = np.asarray(out.scales(name))
    if got.shape != want.shape or not np.allclose(got.astype(np.float32), want.astype(np.float32),
                                                  rtol=max(2.0 * tot["max_rel"], 1e-3)):
        raise SystemExit("kindi refit: %s reads back from %s with scales that are not the ones fitted" % (name, dst))
    rep = man["refit"]
    log("refit %s -> %s against %s" % (src, dst, target))
    log("  %d blocks, %d tensors; %d blocks had no codes and kept their scale, %d clipped at zero"
        % (tot["blocks"], tot["tensors"], tot["kept"], tot["clipped"]))
    log("  weights %d, code bytes %d of which %d differ from the target (%.4f%%)"
        % (tot["weights"], tot["code_bytes"], tot["code_bytes_differing"], rep["code_bytes_differing_pct"]))
    log("  closeness to the target: relative RMSE %.5f -> %.5f (%.2f%% of the error left)"
        % (rep["rmse_relative_before"], rep["rmse_relative_after"],
           100.0 * (rep["rmse_relative_after"] / max(rep["rmse_relative_before"], 1e-30))))
    for gain, n, b, a in sorted(worst, reverse=True)[:5]:
        log("    %-40s %.5f -> %.5f" % (n, b, a))
    log("  %d s" % int(time.time() - t0))
    return rep


def retable(src, dst, log=print, link=False):
    """every code and every scale value kept; only the storage changes - the same values, re-encoded as tables

    A refit changes two things at once: which values a block scale takes, and how it is stored. A paired walk of a
    refit against its source therefore prices both in one number. This mode leaves the values exactly as they are and
    runs them through the refit's own encoder, so a walk of its output against the source measures the storage path
    and nothing else - and on a source this encoder already produced that walk must return zero, which is the control.
    """
    t0 = time.time()
    s = Pack(src)
    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit("kindi retable: %s already holds files; a retable writes a new pack, it does not patch one" % dst)
    if os.path.exists(os.path.join(src, "pcodes.bin")):
        raise SystemExit("kindi retable: %s holds pcodes.bin, which carries its own copy of the block scales for the "
                         "expert matrices and would go stale; convert without --paircodes first" % src)
    os.makedirs(dst, exist_ok=True)
    tot = dict(blocks=0, tensors=0, exact=0, max_levels=0, max_rel=0.0, changed=0, scale_bytes=0)
    dtable, dtab_off, dblk = [], 0, 0
    dpm = open(os.path.join(dst, "dpm.bin"), "wb")
    keep_first = None
    for i, name in enumerate(s.bi):
        # one copy: a v1 source hands back a memmap view, and the encoder indexes it in place
        vals = np.ascontiguousarray(np.asarray(s.scales(name)).astype(np.float16))     # flat [group][row]
        dtab_off, table, idx, exact = _v2_record(dpm, dtable, vals, dblk, dtab_off,
                                                 s.ix["tensors"][name], s.bi[name])
        back = packio.decode(table, idx).astype(np.float64)
        v64 = vals.astype(np.float64)
        ok = v64 > 0
        if ok.any():
            tot["max_rel"] = max(tot["max_rel"], float((np.abs(back[ok] - v64[ok]) / v64[ok]).max()))
        tot["changed"] += int((back != v64).sum())
        tot["blocks"] += int(vals.size)
        tot["tensors"] += 1
        tot["exact"] += int(exact)
        tot["max_levels"] = max(tot["max_levels"], int(table.size))
        tot["scale_bytes"] += 2 * int(table.size)
        dblk += int(vals.size)
        if keep_first is None:
            keep_first = (name, vals.copy())
        if (i + 1) % 40 == 0:
            log("  %d / %d tensors" % (i + 1, len(s.bi)))
    dpm.close()
    with open(os.path.join(dst, "dscale.bin"), "wb") as f:
        f.write(np.concatenate(dtable).astype(np.float16).tobytes())
    for name in sorted(os.listdir(src)):
        if name in COPY or name.startswith("signs_"):
            _place(os.path.join(src, name), os.path.join(dst, name), link)
    with open(os.path.join(dst, "index.json"), "w") as f:
        json.dump(s.ix, f)
    with open(os.path.join(dst, "bcodes_index.json"), "w") as f:
        json.dump({"tensors": s.bi}, f)
    man = dict(s.man)
    man["pack_format"] = 2
    man["dpm_bytes"] = os.path.getsize(os.path.join(dst, "dpm.bin"))
    man["block_scales"] = {"bits": packio.BITS, "file": "dscale.bin",
                           "bytes": os.path.getsize(os.path.join(dst, "dscale.bin")),
                           "ternary_tensors": tot["tensors"], "exact_tensors": tot["exact"],
                           "max_levels": tot["max_levels"], "max_relative_error": round(tot["max_rel"], 6)}
    man["retable"] = {"source": os.path.abspath(src),
                      "method": "every code and every scale value kept; the scales re-encoded as one 8-bit level "
                                "table a tensor through kindi.refit's own encoder",
                      "blocks": tot["blocks"], "tensors": tot["tensors"], "exact_tensors": tot["exact"],
                      "max_levels": tot["max_levels"], "max_relative_error": round(tot["max_rel"], 8),
                      "scale_values_changed": tot["changed"], "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(dst, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    out = Pack(dst)
    name, want = keep_first
    got = np.asarray(out.scales(name))
    if got.shape != want.shape or not np.allclose(got.astype(np.float32), want.astype(np.float32),
                                                  rtol=max(2.0 * tot["max_rel"], 1e-3)):
        raise SystemExit("kindi retable: %s reads back from %s with scales that are not the ones written" % (name, dst))
    log("retable %s -> %s" % (src, dst))
    log("  %d blocks, %d tensors; %d scale values changed, worst %.6f relative; %d tensors exact, %d levels at most"
        % (tot["blocks"], tot["tensors"], tot["changed"], tot["max_rel"], tot["exact"], tot["max_levels"]))
    log("  %d s" % int(time.time() - t0))
    return man["retable"]


# --- selftest ---------------------------------------------------------------------------------------------------------

PASS = []
FAILED = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    (PASS if ok else FAILED).append(name)


def selftest():
    import tempfile
    from . import quant
    del PASS[:], FAILED[:]
    rng = np.random.RandomState(5)
    C, R = 384, 24
    G = C // GROUP

    # the solve recovers a planted scale exactly, and beats the scale it was given on a target it cannot match
    c = rng.randint(0, 3, size=(R, C)).astype(np.int8) - 1
    w = c.astype(np.float32) * 2.5
    s = fit(c, w, np.full((R, G), 1.0))
    check("a planted scale comes back exactly, block for block", np.allclose(s, 2.5, rtol=1e-6),
          "worst %.2e off" % float(np.abs(s - 2.5).max()))
    noisy = c.astype(np.float32) * 2.5 + rng.randn(R, C).astype(np.float32) * 0.9
    s2 = fit(c, noisy, np.full((R, G), 1.0))
    e_before = float((((noisy - c * 1.0) ** 2)).sum())
    e_after = float((((noisy - c * s2[:, :, None].astype(np.float32).repeat(GROUP, axis=1).reshape(R, C)) ** 2)).sum())
    check("against a target it cannot match, the fit still cuts the error", e_after < 0.95 * e_before,
          "%.4f -> %.4f" % (e_before, e_after))

    # a block whose codes are all zero has no direction to fit: it keeps what it came with, and it is not a NaN
    cz = c.copy()
    cz[:2, :GROUP] = 0
    prior = np.full((R, G), 0.75)
    sz = fit(cz, cz.astype(np.float32) * 3.0 + 0.1, prior)
    check("an all-zero-code block keeps its scale instead of dividing by zero",
          np.allclose(sz[:2, 0], 0.75) and np.isfinite(sz).all(), "kept %.3f" % float(sz[0, 0]))

    # a negative solution is clipped, not written: a pack's block scale is a magnitude
    neg = -c
    sn = fit(c, neg.astype(np.float32) * 2.0, np.full((R, G), 0.25))
    check("a block whose target points the other way clips to zero rather than going negative",
          np.all(sn >= 0) and np.allclose(sn, 0.0), "max %.3f" % float(sn.max()))

    # a pack against itself is a fixed point: the scale this formula returns for w = c*s is s
    tmp = tempfile.mkdtemp(prefix="refit-")
    w1 = (rng.randn(64, 256) * 0.02).astype(np.float32)
    w2 = (rng.randn(128, 384) * 0.02).astype(np.float32)
    a = os.path.join(tmp, "a")
    quant.pack({"x.weight": w1, "y.weight": w2}, a, log=lambda *x: None)
    rep = refit(a, os.path.join(tmp, "self"), a, log=lambda *x: None)
    same = all(np.allclose(np.asarray(Pack(a).scales(n)), np.asarray(Pack(os.path.join(tmp, "self")).scales(n)),
                           rtol=1e-3, atol=1e-7) for n in ("x.weight", "y.weight"))
    check("a pack refitted against itself changes no block scale", same and rep["blocks"] > 0,
          "%d blocks, %.2e relative RMSE" % (rep["blocks"], rep["rmse_relative_after"]))

    # doubling the target's weights doubles the scales, because the codes of a doubled tensor are the same codes
    b = os.path.join(tmp, "b")
    quant.pack({"x.weight": 2 * w1, "y.weight": 2 * w2}, b, log=lambda *x: None)
    rd = refit(a, os.path.join(tmp, "dbl"), b, log=lambda *x: None)
    sa = np.concatenate([np.asarray(Pack(a).scales(n)) for n in ("x.weight", "y.weight")]).astype(np.float32)
    sd = np.concatenate([np.asarray(Pack(os.path.join(tmp, "dbl")).scales(n)) for n in ("x.weight", "y.weight")])
    check("a target at twice the magnitude gives twice the scales, with the codes untouched",
          np.allclose(sd, 2 * sa, rtol=2e-2) and rd["code_bytes_differing"] == 0,
          "worst %.3f relative, %d code bytes differ" % (float(np.abs(sd.astype(np.float32) / (2 * sa) - 1).max()),
                                                         rd["code_bytes_differing"]))

    # a retable writes the same codes and re-encodes the values; a tensor with more distinct scales than the table
    # has entries is where the 8-bit grid actually costs something, so build one deliberately
    vary = os.path.join(tmp, "vary")
    quant.pack({"x.weight": (rng.randn(512, 128) * (1.0 + np.arange(512) / 512.0)[:, None]).astype(np.float32)},
               vary, log=lambda *x: None)
    rp = os.path.join(tmp, "retab")
    rr = retable(vary, rp, log=lambda *x: None)
    same_codes = np.array_equal(np.asarray(Pack(rp).codes("x.weight")[0]), np.asarray(Pack(vary).codes("x.weight")[0]))
    check("a retable keeps every code and bounds the scale error by its 8-bit grid",
          same_codes and rr["scale_values_changed"] > 0 and 0.0 < rr["max_relative_error"] < 0.02,
          "worst %.5f relative, %d of %d values moved, codes %s"
          % (rr["max_relative_error"], rr["scale_values_changed"], rr["blocks"],
             "identical" if same_codes else "CHANGED"))
    rr2 = retable(rp, os.path.join(tmp, "retab2"), log=lambda *x: None)
    check("a second retable moves no scale value: a set of tables comes back bit for bit",
          rr2["scale_values_changed"] == 0 and rr2["max_relative_error"] == 0.0,
          "%d of %d values moved" % (rr2["scale_values_changed"], rr2["blocks"]))
    import contextlib
    import io
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            main(["--pack", vary, "--out", os.path.join(tmp, "retab3"), "--retable", "--target", vary])
        refused = ""
    except SystemExit:
        refused = err.getvalue()
    check("a retable with a --target is refused, because keeping the values leaves nothing to fit",
          "nothing to fit" in refused, refused.strip().splitlines()[-1][:70] if refused else "")

    # a target that cannot serve a tensor is refused by name, rather than leaving that tensor unfitted
    c_only = os.path.join(tmp, "half")
    quant.pack({"x.weight": w1}, c_only, log=lambda *x: None)
    try:
        refit(a, os.path.join(tmp, "bad"), c_only, log=lambda *x: None)
        refused = ""
    except SystemExit as e:
        refused = str(e)
    check("a target missing a tensor is refused, naming what it cannot fit", "no y.weight" in refused, refused[:60])

    # a target with the wrong width is refused before any byte is written
    d = os.path.join(tmp, "narrow")
    quant.pack({"x.weight": w1[:, :128], "y.weight": w2}, d, log=lambda *x: None)
    try:
        refit(a, os.path.join(tmp, "bad2"), d, log=lambda *x: None)
        refused = ""
    except SystemExit as e:
        refused = str(e)
    check("a target at another width is refused instead of fitted into the wrong shape",
          "x.weight is 24x256 in the source" in refused or "256 in the source" in refused, refused[:60])

    # a target whose codes were rotated differently is a different basis: the fit has to be refused, not fitted
    rx = _json(os.path.join(b, "index.json"))
    rx["hadamard"] = {"tied_output": False, "fold": True, "sign_widths": [5120]}
    with open(os.path.join(b, "index.json"), "w") as f:
        json.dump(rx, f)
    try:
        refit(a, os.path.join(tmp, "bad3"), b, log=lambda *x: None)
        refused = ""
    except SystemExit as e:
        refused = str(e)
    check("a target in another rotation basis is refused rather than fitted across the two",
          "different Hadamard rotations" in refused, refused[:60])

    # a pack that stages its experts with their own copy of the scales cannot be refitted in place: refuse it
    open(os.path.join(a, "pcodes.bin"), "wb").close()
    try:
        refit(a, os.path.join(tmp, "bad4"), a, log=lambda *x: None)
        refused = ""
    except SystemExit as e:
        refused = str(e)
    try:
        retable(a, os.path.join(tmp, "bad5"), log=lambda *x: None)
        stale = ""
    except SystemExit as e:
        stale = str(e)
    os.remove(os.path.join(a, "pcodes.bin"))
    check("a pack whose pcodes.bin holds its own scales is refused, not left disagreeing with itself",
          "pcodes.bin" in refused and "stale" in refused, refused[:60])
    check("a retable of a pack whose pcodes.bin holds its own scales is refused for the same reason",
          "pcodes.bin" in stale and "stale" in stale, stale[:60])

    print("\n%d checks passed, %d failed" % (len(PASS), len(FAILED)))
    return 0 if len(FAILED) == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kindi refit", description="keep every code, re-derive every block scale")
    ap.add_argument("--pack", help="the source pack: its codes are what the output keeps")
    ap.add_argument("--target", help="the pack whose weights the scales are fitted to")
    ap.add_argument("--out", help="a new directory for the refitted pack")
    ap.add_argument("--retable", action="store_true",
                    help="keep the scale values too, and only re-encode them as 8-bit tables (the storage control "
                         "for a refit, not a second fit)")
    ap.add_argument("--link", action="store_true", help="hard-link the files that do not change instead of copying")
    ap.add_argument("--chunk", type=int, default=512, help="rows a fit does at a time (default 512)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.retable:
        if not (a.pack and a.out):
            ap.error("--retable needs --pack and --out")
        if a.target:
            ap.error("--retable keeps the source's scale values, so a --target has nothing to fit; drop one")
        retable(a.pack, a.out, link=a.link)
        return 0
    if not (a.pack and a.target and a.out):
        ap.error("--pack, --target and --out are all needed (or --selftest, or --retable)")
    refit(a.pack, a.out, a.target, link=a.link, chunk=a.chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
