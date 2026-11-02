"""kindi quant: a float checkpoint -> the ternary weight form a Kindi pack holds.

Bonsai's packs arrive as a PTQ1_0 GGUF, already ternary: one fp16 scale per 128-weight group of a row, weights in
{-1, 0, +1} * scale, byte = t0 + 3 t1 + 9 t2 + 27 t3 for four consecutive columns. A QAT checkpoint is the same
quantiser seen from the other side - float weights, trained with it in the loop. This module runs it forwards so a
float checkpoint can be packed, and `kindi.drift` measures what it costs.

The scale is the group's mean |w| (absmean, as in BitNet b1.58 and in the PTQ1_0 GGUF Kindi already reads), rounded
to fp16 **before** the trits are chosen, so `dequantise(pack)` is the arithmetic the engine runs and not an
idealisation of it. A zero group keeps scale 0 and every trit at 1 (the zero trit): the block contributes nothing,
which is what the original weights were.

  from .quant import quantise
  codes, scales = quantise(w)                 # w (R, C) float32 -> (C//4, R) uint8, (C//128, R) fp16

`pack()` writes those bytes in the layout `kindi/cpu.py` loads: bcodes.bin (slot-major, [slot][row]), dpm.bin
([group][row] fp16) and f32.bin (every non-ternary tensor verbatim), with index.json and bcodes_index.json. It is
the same file set and the same offsets as `kindi convert` produces, so a pack written here loads through the same
code path; what it does not carry is a tokenizer or a Hadamard rotation, because a float state dict has neither.
"""
import json
import os

import numpy as np

from . import __version__

PACK_FORMAT = 1
PTQ1_0, F32 = 143, 0
GROUP = 128


def scales_for(w, group=GROUP):
    """absmean per (row, group) -> fp16, the scale the trits are chosen against"""
    R, C = w.shape
    if C % group:
        raise ValueError("width %d is not a multiple of the %d-weight group" % (C, group))
    g = np.abs(w.reshape(R, C // group, group)).mean(axis=2, dtype=np.float64)
    return g.astype(np.float16)


LSQ_GRID = np.linspace(0.6, 2.2, 33)


def _lsq_scales(w, group):
    """the scale that minimises the error of the ternarised group, not the group's mean |w|.

    absmean is the right scale for weights a ternary-aware training run has already shaped into three levels; on a
    checkpoint that has not been shaped that way it overshoots, because a Gaussian's mean |w| sits far below the
    values that clip. This is a per-group search over a fixed multiple grid of absmean, vectorised over rows, which
    is what the drift study found is worth doing: the error fell monotonically as the multiple rose past 1.0.
    """
    R, C = w.shape
    G, n = C // group, group
    base = np.abs(w.reshape(R, G, n)).mean(axis=2, dtype=np.float64)
    out = np.empty((R, G), dtype=np.float32)
    cand = (base[:, :, None] * LSQ_GRID[None, None, :]).astype(np.float32)
    for r0 in range(0, R, 128):
        r1 = min(R, r0 + 128)
        wg = w[r0:r1].reshape(r1 - r0, G, n)[:, :, None, :]          # (rows, groups, 1, n)
        c = cand[r0:r1]                                            # (rows, groups, K)
        s = np.where(c > 0, c, np.float32(1.0))[:, :, :, None]     # (rows, groups, K, 1)
        q = np.clip(np.rint(wg / s), -1, 1)
        err = ((wg - q * c[:, :, :, None]) ** 2).sum(axis=3)
        k = err.argmin(axis=2)
        out[r0:r1] = np.take_along_axis(cand[r0:r1], k[:, :, None], axis=2)[:, :, 0]
    return out.astype(np.float16)


def quantise(w, group=GROUP, scheme="absmean"):
    """(codes uint8 [C//4][R], scales fp16 [C//group][R], dequantised float32 [R][C]) for one scale scheme"""
    w = np.ascontiguousarray(w, dtype=np.float32)
    R, C = w.shape
    sc = scales_for(w, group) if scheme == "absmean" else _lsq_scales(w, group)
    sf = sc.astype(np.float32)
    safe = np.where(sf > 0, sf, np.float32(1.0))
    q = np.clip(np.rint(w.reshape(R, C // group, group) / safe[:, :, None]), -1, 1).astype(np.int8)
    q = np.where(sf[:, :, None] > 0, q, np.int8(0))
    q = q.reshape(R, C)
    deq = q.astype(np.float32) * sf.repeat(group, axis=1)
    t = (q + 1).astype(np.uint8).reshape(R, C // 4, 4)
    codes = (t[:, :, 0] + 3 * t[:, :, 1] + 9 * t[:, :, 2] + 27 * t[:, :, 3]).astype(np.uint8)
    return np.ascontiguousarray(codes.T), np.ascontiguousarray(sc.T), deq


def dequantise(codes, scales, R, C):
    """the inverse, straight off the pack layout: (C//4, R) bytes + (C//128, R) fp16 -> (R, C) float32"""
    b = np.asarray(codes, dtype=np.int32).T.reshape(R, C // 4)
    t = np.stack([b % 3, (b // 3) % 3, (b // 9) % 3, (b // 27) % 3], axis=2).reshape(R, C)
    q = (t - 1).astype(np.float32)
    sf = np.asarray(scales, dtype=np.float16).T.astype(np.float32)
    return q * sf.repeat(C // sf.shape[1], axis=1)


def is_ternary(name, a):
    """the 2-D weight matrices; norms, biases, conv kernels and the GDN parameters stay float"""
    if a.ndim != 2:
        return False
    if name.endswith(("norm.weight", "conv1d.weight")):
        return False
    return a.shape[1] % GROUP == 0


def _pad8(a):
    """the pair-code layout walks rows in blocks of 8, so the last block is padded with zero rows"""
    R = a.shape[0]
    if R % 8:
        a = np.concatenate([a, np.zeros((8 - R % 8, a.shape[1]), dtype=a.dtype)], axis=0)
    return a, int(a.shape[0])


def pack(sd, out_dir, log=print, group=GROUP, name=None, source=None, sha=None, extra=None, scheme="lsq"):
    """write bcodes.bin / dpm.bin / f32.bin / index.json / bcodes_index.json / kindi_pack.json for a float state dict"""
    os.makedirs(out_dir, exist_ok=True)
    order = sorted(sd)
    tern = [k for k in order if is_ternary(k, sd[k])]
    if not tern:
        raise SystemExit("kindi quant: no 2-D weight matrices in this checkpoint, so there is nothing to ternarise")
    index = {"tensors": {}, "base": 0, "hadamard": {}}
    bindex = {"tensors": {}}
    fb = open(os.path.join(out_dir, "bcodes.bin"), "wb")
    fd = open(os.path.join(out_dir, "dpm.bin"), "wb")
    ff = open(os.path.join(out_dir, "f32.bin"), "wb")
    bc_off = dpm_off = f32_off = 0
    R = C = 0
    for k in order:
        a = sd[k]
        if k in tern:
            R0, C0 = a.shape
            ap, rows = _pad8(np.ascontiguousarray(a, dtype=np.float32))
            codes, sc, _ = quantise(ap, group, scheme)
            fb.write(codes.tobytes())                       # [slot][row], as convert.py writes it
            fd.write(np.ascontiguousarray(sc.view(np.uint16)).tobytes())
            n = codes.shape[0] * rows
            index["tensors"][k] = {"dims": [R0, C0], "type": PTQ1_0, "folded": False, "rows": rows, "ne0": C0,
                                   "bc_off": bc_off, "dpm_off": dpm_off}
            bindex["tensors"][k] = {"bc_off": bc_off, "dpm_off": dpm_off, "rows": rows, "ne0": C0}
            bc_off += n
            dpm_off += sc.shape[0] * rows * 2
            R, C = rows, C0
        else:
            f = np.ascontiguousarray(a.astype(np.float32))
            ff.write(f.tobytes())
            index["tensors"][k] = {"dims": list(f.shape), "type": F32, "folded": False, "rows": f.shape[0],
                                   "ne0": int(f.size // max(1, f.shape[0])), "f32_off": f32_off, "nfloats": int(f.size)}
            f32_off += 4 * f.size
        log("  %-28s %-18s %s" % (k, "ternary" if k in tern else "f32", tuple(a.shape)))
    for h in (fb, fd, ff):
        h.close()
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(out_dir, "bcodes_index.json"), "w") as f:
        json.dump(bindex, f)
    man = {"kindi_version": __version__, "pack_format": PACK_FORMAT, "model": name or (source and os.path.basename(source)),
           "source_file": os.path.basename(source) if source else None, "source_sha256": sha,
           "tensors": len(order), "ternary_tensors": len(tern), "bcodes_bytes": bc_off, "dpm_bytes": dpm_off,
           "f32_bytes": f32_off,           "quant": {"scheme": scheme, "group": group, "scale": "fp16", "values": [-1, 0, 1]},
           "hadamard": {"enabled": False}, "dimensions": infer_dims(sd)}
    if extra:
        man.update(extra)
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def selftest():
    """round trip in both directions, and the same tensor as the engine's GEMV will see it"""
    rng = np.random.RandomState(11)
    w = (rng.randn(64, 256) * 0.02).astype(np.float32)
    codes, sc, deq = quantise(w)
    assert np.array_equal(dequantise(codes, sc, 64, 256), deq), "pack layout does not decode to the quantised weights"
    assert set(np.unique(deq / sc.T.astype(np.float32).repeat(128, axis=1))).issubset({-1.0, 0.0, 1.0})
    w0 = w.copy()
    w0[:8, 128:] = 0.0
    c0, s0, d0 = quantise(w0)
    assert np.all(s0[1, :8] == 0) and np.all(d0[:8, 128:] == 0), "a zero group must decode to zero"
    # the engine's own loader: paircodes() from the pack bytes, decoded back through its code table
    from .cpu import paircodes, _CODE
    inv = {int(_CODE[k]): (k % 3 - 1, k // 3 - 1) for k in range(9)}
    R, C = 64, 256
    P = paircodes(codes, R, C)
    got = np.zeros((R, C), dtype=np.float32)
    for rb in range(R // 8):
        for g in range(C // 128):
            for p in range(64):
                word = int(P[rb, g, p])
                for l in range(8):
                    a, b = inv[(word >> (4 * l)) & 15]
                    s = np.float32(sc[g, rb * 8 + l])
                    got[rb * 8 + l, g * 128 + 2 * p] = a * s
                    got[rb * 8 + l, g * 128 + 2 * p + 1] = b * s
    assert np.array_equal(got, deq), "paircodes() does not rebuild the quantised matrix"


def infer_dims(sd):
    """the geometry a state dict states outright; anything a checkpoint does not carry is left out, not guessed"""
    d = {}
    lay = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.") and k.split(".")[1].isdigit()})
    if lay:
        d["layers"] = len(lay)
    for key, name in (("embed.weight", None),):
        if key in sd:
            d["vocab"], d["width"] = int(sd[key].shape[0]), int(sd[key].shape[1])
    if "layers.0.D" in sd:
        d["gdn_inner"] = int(sd["layers.0.D"].shape[0])
    if "layers.0.A_log" in sd:
        d["heads"] = int(sd["layers.0.A_log"].shape[0])
    if "layers.0.conv1d.weight" in sd:
        d["conv_kernel"] = int(sd["layers.0.conv1d.weight"].shape[-1])
    if "layers.0.in_proj.weight" in sd and "gdn_inner" in d:
        d["in_proj_rows"] = int(sd["layers.0.in_proj.weight"].shape[0])
    return d
