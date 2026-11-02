"""kindi drift: what the ternary quantiser costs, per tensor and by depth.

Reads a float checkpoint (see `kindi.ptload`), ternarises every 2-D weight matrix at one or more scale schemes and
granularities, and reports for each matrix how far the quantised weights are from the float ones and how far that
moves the output they produce. The point is the shape of the drift rather than one average: a QAT run is supposed to
pull it down and spread it evenly over depth, and the scheme rows say how much of the error is the scale rather than
the trits.

  kindi drift qat_final.pt qat_step1000.pt                        # absmean and least-squares, 128-wide groups
  kindi drift model.pt --groups 128,row,tensor --schemes absmean,lsq
  kindi drift model.pt --pack /tmp/pack                           # write an actual pack and check it decodes back

`rel` is ||W - Wq||_F / ||W||_F per matrix, `out` is the same for the matrix's output on probe activations
(unit-RMS Gaussian, one seed for every checkpoint and scheme, so the columns are comparable), and the depth table
averages both over the layer's matrices. Weight-space numbers are exact for the checkpoint; `out` is an estimate on
synthetic activations, which is the price of not having the trainer's forward pass on this machine.
"""
import os
import time

import numpy as np

from . import ptload, quant

PROBES = 32


def rel_err(w, wq):
    return float(np.linalg.norm((w - wq).ravel()) / max(np.linalg.norm(w.ravel()), 1e-30))


def cos_sim(w, wq):
    a, b = w.ravel().astype(np.float64), wq.ravel().astype(np.float64)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def out_err(w, wq, x):
    """relative error of W x against Wq x for probe rows x (P, C)"""
    y, yq = x @ w.T, x @ wq.T
    return float(np.linalg.norm(y - yq) / max(np.linalg.norm(y), 1e-30))


def quantise_at(w, group, scheme):
    """granularity: 128 (the pack), None (per row), 0 (per tensor)"""
    if group is None:
        group = w.shape[1]
    if group == 0:
        w2 = w.reshape(1, -1)
        sc = quant.scales_for(w2, w2.shape[1]) if scheme == "absmean" else quant._lsq_scales(w2, w2.shape[1])
        sf = np.float32(sc[0, 0])
        return np.zeros_like(w) if sf == 0 else np.clip(np.rint(w / sf), -1, 1).astype(np.float32) * sf
    return quant.quantise(w, group, scheme)[2]


def layer_of(name):
    p = name.split(".")
    return int(p[1]) if p[0] == "layers" and p[1].isdigit() else None


def probe_for(C, rng):
    x = rng.randn(PROBES, C).astype(np.float32)
    return x / np.sqrt((x ** 2).mean(axis=1, keepdims=True))


def measure(sd, variants, log=print):
    """[(name, variant, rel, cos, out, params)] for every ternary matrix, at every requested variant"""
    rng = np.random.RandomState(1234)
    rows = []
    for name in sorted(sd):
        w = sd[name]
        if not quant.is_ternary(name, w):
            continue
        w = np.ascontiguousarray(w, dtype=np.float32)
        x = probe_for(w.shape[1], rng)
        for label, group, scheme in variants:
            wq = quantise_at(w, group, scheme)
            rows.append((name, label, rel_err(w, wq), cos_sim(w, wq), out_err(w, wq, x), int(w.size)))
    return rows


def summarise(rows, log=print):
    by = {}
    for name, label, rel, cos, out, n in rows:
        by.setdefault(label, []).append((rel, cos, out, n))
    for label, rs in by.items():
        wn = sum(r[3] for r in rs)
        log("  %-16s params %7.1f M   weight rel %.4f   output rel %.4f   mean cos %.4f"
            % (label, wn / 1e6, sum(r[0] * r[3] for r in rs) / wn, sum(r[2] * r[3] for r in rs) / wn,
               float(np.mean([r[1] for r in rs]))))


def curve(rows, label, log=print):
    """the depth curve for one variant, plus the matrices outside the layer stack"""
    by_layer = {}
    for name, lab, rel, cos, out, n in rows:
        if lab != label:
            continue
        L = layer_of(name)
        if L is not None:
            by_layer.setdefault(L, []).append((name, rel, out))
    log("\n  depth   weight rel   output rel   worst tensor")
    peak = 0.0
    for L in sorted(by_layer):
        rs = by_layer[L]
        w = float(np.mean([r[1] for r in rs]))
        peak = max(peak, w)
        log("  %5d   %10.4f   %10.4f   %s" % (L, w, float(np.mean([r[2] for r in rs])),
                                              max(rs, key=lambda r: r[1])[0].split(".")[-2]))
    others = [(nm, rel, out, n) for nm, lab, rel, cos, out, n in rows if lab == label and layer_of(nm) is None]
    if others:
        log("\n  outside the stack:")
        for nm, rel, out, n in others:
            log("    %-24s %6.1f M   weight rel %.4f   output rel %.4f" % (nm, n / 1e6, rel, out))
    log("\n  worst layer weight rel %.4f" % peak)
    return peak


def variants_for(a):
    schemes = [s.strip() for s in (a.schemes or "absmean,lsq").split(",")]
    groups = []
    for g in (a.groups or "128").split(","):
        g = g.strip()
        groups.append(None if g == "row" else (0 if g == "tensor" else int(g)))
    return [(("%s/%s" % (s, "row" if g is None else g)), g, s) for s in schemes for g in groups]


def run(a, log=print):
    variants = variants_for(a)
    rc = 0
    for i, path in enumerate(a.checkpoint):
        if not os.path.exists(path):
            log("kindi drift: no such checkpoint %s" % path)
            return 2
        log("\n%s" % path)
        t0 = time.time()
        sd = {k: ptload.widen(v) for k, v in ptload.load(path).items() if v.dtype.kind in "fiu"}
        tern = [k for k in sd if quant.is_ternary(k, sd[k])]
        log("  %d tensors, %.1f M parameters, %d ternary matrices"
            % (len(sd), sum(v.size for v in sd.values()) / 1e6, len(tern)))
        rows = measure(sd, variants, log=log)
        summarise(rows, log=log)
        labels = [v[0] for v in variants]
        curve(rows, "lsq/128" if "lsq/128" in labels else labels[0], log=log)
        log("  (%.0fs)" % (time.time() - t0))
        if a.pack:
            out_dir = a.pack if len(a.checkpoint) == 1 else "%s-%d" % (a.pack, i)
            scheme = "lsq" if any(v[2] == "lsq" for v in variants) else "absmean"
            man = quant.pack(sd, out_dir, log=lambda *x: None, name=os.path.basename(path), source=path,
                             scheme=scheme)
            log("  pack written: %s (%d ternary matrices, %.0f MB codes, %s scale)"
                % (out_dir, man["ternary_tensors"], man["bcodes_bytes"] / 2 ** 20, scheme))
            if not verify_pack(sd, out_dir, scheme, log=log):
                rc = 1
    return rc


def verify_pack(sd, pack_dir, scheme="lsq", log=print):
    """read the pack back through its own index and compare against the quantiser in memory"""
    import json
    idx = json.load(open(os.path.join(pack_dir, "bcodes_index.json")))["tensors"]
    bc = np.memmap(os.path.join(pack_dir, "bcodes.bin"), dtype=np.uint8, mode="r")
    dp = np.memmap(os.path.join(pack_dir, "dpm.bin"), dtype=np.float16, mode="r")
    bad = 0
    for name, rec in idx.items():
        R, C = rec["rows"], rec["ne0"]
        o, q = rec["bc_off"], rec["dpm_off"] // 2
        got = quant.dequantise(bc[o:o + (C // 4) * R].reshape(C // 4, R),
                               dp[q:q + (C // 128) * R].reshape(C // 128, R), R, C)
        w = np.ascontiguousarray(sd[name][:R], dtype=np.float32)
        want = quantise_at(w, 128, scheme)
        if not np.array_equal(got[:w.shape[0], :w.shape[1]], want):
            log("  pack round trip FAILED for %s" % name)
            bad += 1
    log("  pack round trip: %d/%d ternary matrices decode to the quantised weights exactly"
        % (len(idx) - bad, len(idx)))
    return bad == 0
