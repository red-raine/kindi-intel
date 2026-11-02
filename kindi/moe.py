"""kindi moe-check: does the C Mixture-of-experts FFN compute what the arithmetic says it should?

A dense forward pass cannot isolate the routing, so `kc_ffn_probe` runs one layer's FFN block on a given input and
this compares it against a NumPy implementation of the same block: RMS norm, the pack's Hadamard fold, the router,
the top-k renormalised weights, and each selected expert's gate/up/silu/down. The pack is the synthetic fixture
(`kindi.synthpack --experts`), whose quantised weights are handed back to the reference, so both sides see the same
ternary values and the comparison is about arithmetic and routing, not about quantisation.

  kindi moe-check                  # build a fixture with 4 experts, 2 active, and compare
  kindi moe-check --experts 8 --active 1 --layers 4
"""
import os
import tempfile

import numpy as np

DIMS = dict(width=256, ffn=512, layers=3)


def fwht_block(x):
    hb = x.shape[-1]
    y = x.copy()
    h = 1
    while h < hb:
        y = y.reshape(-1, hb // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, hb)
        h *= 2
    return y / np.sqrt(hb)


def fold(v, sg, hb):
    return fwht_block((sg * v).reshape(-1, hb)).reshape(-1)


def rmsfold(x, w, sg, hb):
    s = 1.0 / np.sqrt((x * x).mean() + 1e-6)
    return fold((x * s) * w, sg, hb)


def softmax_topk(logits, k):
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    idx = np.argsort(-p)[:k]
    sel = p[idx]
    return idx, sel / max(sel.sum(), 1e-30)


def silu(x):
    return x / (1.0 + np.exp(-x))


def ref_ffn(x, t, il, dims, hb):
    """t: quantised tensors of the fixture. Returns (out, selected experts, weights)."""
    p = "blk.%d." % il
    D = dims["width"]
    xp = rmsfold(x, t[p + "post_attention_norm.weight"], t["signs_%d" % D], hb)
    pr = t[p + "moe_router.weight"] @ xp
    idx, w = softmax_topk(pr, dims["experts_active"])
    E = t[p + "moe_gate.e0.weight"].shape[0]
    y = np.zeros(D, dtype=np.float32)
    for j, e in enumerate(idx):
        g = t[p + "moe_gate.e%d.weight" % e] @ xp
        u = t[p + "moe_up.e%d.weight" % e] @ xp
        m = fold(silu(g) * u, t["signs_%d" % E], hb)
        y += np.float32(w[j]) * (t[p + "moe_down.e%d.weight" % e] @ m)
    return y, idx, w


def run(a, log=print):
    from . import synthpack
    from .cpu import Engine
    from . import quant
    for _ in range(1):
        quant.selftest()
    dims = dict(DIMS, layers=a.layers, experts=a.experts, experts_active=a.active)
    tmp = tempfile.mkdtemp(prefix="kindi-moe-")
    pack, tens, floats = synthpack.build(tmp, log=lambda *x: None, width=dims["width"], ffn=dims["ffn"],
                                         layers=dims["layers"], experts=dims["experts"],
                                         experts_active=dims["experts_active"])
    t = dict(floats)
    t.update(tens)
    for w in (dims["width"], dims["ffn"] // 2, dims["ffn"]):
        f = os.path.join(pack, "signs_%d.f32" % w)
        if os.path.exists(f) and "signs_%d" % w not in t:
            t["signs_%d" % w] = np.fromfile(f, dtype=np.float32)
    hb = 128
    eng = Engine(pack, ctx=64, log=lambda *x: None)
    width = eng.dims["width"]
    rng = np.random.RandomState(5)
    log("kindi moe-check: %s, %d layers, %d experts (%d active), width %d"
        % (os.path.basename(pack), dims["layers"], a.experts, a.active, width))
    worst = 0.0
    for il in range(dims["layers"]):
        for trial in range(2):
            x = rng.randn(width).astype(np.float32)
            got, sel_c, p_c = eng.ffn_probe(il, x)
            want, sel_n, p_n = ref_ffn(x, t, il, dims, hb)
            same_route = list(sel_c) == list(sel_n)
            rel = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))
            worst = max(worst, rel)
            log("  layer %d: routed %s%s  weights %s  output rel %.2e"
                % (il, list(sel_c), "" if same_route else " (numpy: %s)" % list(sel_n),
                   " ".join("%.3f" % p for p in p_c[:a.active]), rel))
            if not same_route:
                log("  the C and NumPy routers disagree - that is a bug, not a tolerance")
                return 1
    ok = worst < 1e-4
    log("  worst output relative difference %.2e: %s" % (worst, "the MoE block matches" if ok else "DOES NOT MATCH"))
    if a.paged:
        ok = bench_paging(log) and ok
    return 0 if ok else 1


def bench_paging(log=print):
    """resident experts against paged ones on a fixture big enough for the difference to show in RSS and in time"""
    import gc
    import time
    from . import synthpack
    from .cpu import Engine
    dims = dict(width=1024, ffn=512, layers=6)
    nexp, nact = 12, 2
    tmp = tempfile.mkdtemp(prefix="kindi-moe-page-")
    synthpack.build(tmp, log=lambda *x: None, experts=nexp, experts_active=nact, **dims)
    ids = [3, 11, 19, 27, 35, 43, 51, 59]
    per_expert = 1.5 * (dims["ffn"] // 2) * dims["width"]        # pair codes plus scales, three matrices
    out = []
    for keep in (nexp, nexp // 2, 4, 2):
        label = "all %d" % keep if keep == nexp else "cache %d" % keep
        gc.collect()
        eng = Engine(tmp, ctx=64, log=lambda *x: None, expert_resident=None if keep == nexp else keep)
        if keep != nexp:
            eng.enable_paging()
        eng.generate(ids, 2)                              # warm the cache, as a server would
        t0 = time.time()
        a = eng.generate(ids, 6)
        dt = (time.time() - t0) / 6
        out.append((label, dt, dims["layers"] * keep * per_expert / 2 ** 20, a))
        del eng
        gc.collect()
    log("\n  experts %d, %d active, %d layers, width %d, expert width %d" % (nexp, nact, dims["layers"],
                                                                           dims["width"], dims["ffn"]))
    log("  %-9s %8s %12s" % ("experts", "ms/token", "code memory"))
    for label, dt, mb, a in out:
        log("  %-9s %8.1f %9.0f MB" % (label, dt * 1000, mb))
    # what a miss costs: one expert's three matrices through the Python pair-code conversion
    from .cpu import pair_mat
    import json as _json
    bi = _json.load(open(os.path.join(tmp, "bcodes_index.json")))["tensors"]
    bc = np.memmap(os.path.join(tmp, "bcodes.bin"), dtype=np.uint8, mode="r")
    dp = np.memmap(os.path.join(tmp, "dpm.bin"), dtype=np.float16, mode="r")
    t0 = time.time()
    for rec in (bi["blk.0.moe_gate.e0.weight"], bi["blk.0.moe_up.e0.weight"], bi["blk.0.moe_down.e0.weight"]):
        pair_mat(bc, dp, rec)
    ms = (time.time() - t0) * 1e3
    log("  one expert's staging (3 matrices, NumPy): %.0f ms   a token using 2 of 12 experts misses often" % ms)
    same = all(o[3] == out[0][3] for o in out)
    log("  every cache size gives the resident output: %s" % ("yes" if same else "NO - the tokens must not change"))
    return same
