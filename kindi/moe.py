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
    """The residency curve, with both staging paths: rebuild the pair codes on a miss, or copy them from the pack.

    Two fixture packs with the same weights - one as `kindi convert` writes it, one with `pcodes.bin`
    (`--paircodes`) - so the same cache size is measured with conversion staging and with memcpy staging. Every row
    must give the resident engine's tokens; timing is decode only, after a two-token warm-up.
    """
    import gc
    import json as _json
    import time
    from . import synthpack
    from .cpu import Engine, _pair_len, _pair_views, is_expert, pair_mat
    dims = dict(width=1024, ffn=512, layers=6)
    nexp, nact = 12, 2
    tmp = tempfile.mkdtemp(prefix="kindi-moe-page-")
    packs = {}
    for tag, want in (("trit", False), ("pair", True)):
        p = os.path.join(tmp, tag)
        synthpack.build(p, log=lambda *x: None, experts=nexp, experts_active=nact, paircodes=want, **dims)
        packs[tag] = p
    ids = [3, 11, 19, 27, 35, 43, 51, 59]
    # bytes the three matrices of one expert take, read off the indexes rather than estimated:
    # the pack's own (ternary codes and their scales, the resident form) and the staged form (pcodes.bin)
    _bi = _json.load(open(os.path.join(packs["trit"], "bcodes_index.json")))["tensors"]
    trecs = [_bi[n] for n in ("blk.0.moe_gate.e0.weight", "blk.0.moe_up.e0.weight", "blk.0.moe_down.e0.weight")]
    per_res = sum(r["rows"] * (r["ne0"] // 128) * 34 for r in trecs)
    per_pool = sum(_pair_len(r) for r in trecs)

    def run(pack, keep, staging="auto"):
        gc.collect()
        eng = Engine(pack, ctx=64, log=lambda *x: None, expert_resident=None if keep == nexp else keep)
        calls = [0]
        if keep != nexp:
            eng.enable_paging()
            if staging == "none":
                eng.set_fetcher(None)                     # the pool and its Mats, but nothing ever refills them
            elif staging == "hook":
                eng.set_fetcher(lambda *a: None)          # the callback crosses, the slots are left stale
            else:
                def counted(big, layer, expert):          # the real hook, counting only the cache misses
                    ins = eng.pager["in_slot"].get(layer)
                    if ins is not None and expert not in ins:
                        calls[0] += 1
                    eng._page_expert(eng._paging_src, layer, expert)
                eng.set_fetcher(counted)
        eng.generate(ids, 2)                              # warm the cache, as a server would
        calls[0] = 0
        t0 = time.time()
        got = eng.generate(ids, 6)
        dt = (time.time() - t0) / 6
        del eng
        gc.collect()
        return got, dt, calls[0] / 6.0

    resident, dt_res, _ = run(packs["trit"], nexp)
    log("\n  experts %d, %d active, %d layers, width %d, expert width %d" % (nexp, nact, dims["layers"],
                                                                           dims["width"], dims["ffn"]))
    log("  %-11s %12s %12s %12s %12s %8s %10s" % ("experts", "convert ms/t", "memcpy ms/t", "hook ms/t",
                                                 "no fetch ms/t", "miss/t", "code memory"))
    log("  %-11s %12.1f %12s %12s %12s %8s %7.1f MB" % ("all %d" % nexp, dt_res * 1000, "-", "-", "-", "-",
                                                        dims["layers"] * nexp * per_res / 2 ** 20))
    same = True
    for keep in (nexp // 2, 4, 2):
        a, ta, _ = run(packs["trit"], keep)
        b, tb, nb = run(packs["pair"], keep)
        _c, tc, _ = run(packs["pair"], keep, staging="hook")   # output is meaningless: the slots are stale
        _d, td, _ = run(packs["pair"], keep, staging="none")
        same = same and a == resident and b == resident
        log("  %-11s %12.1f %12.1f %12.1f %12.1f %8.1f %7.1f MB"
            % ("cache %d" % keep, ta * 1000, tb * 1000, tc * 1000, td * 1000, nb,
               dims["layers"] * keep * per_pool / 2 ** 20))

    # the variant has to be the same bytes as the conversion, not merely the same tokens: every expert matrix
    _pi = _json.load(open(os.path.join(packs["pair"], "bcodes_index.json")))["tensors"]
    bc = np.memmap(os.path.join(packs["trit"], "bcodes.bin"), dtype=np.uint8, mode="r")
    dp = np.memmap(os.path.join(packs["trit"], "dpm.bin"), dtype=np.float16, mode="r")
    pcm = np.memmap(os.path.join(packs["pair"], "pcodes.bin"), dtype=np.uint8, mode="r")
    same_bytes, n_cmp = True, 0
    for name in sorted(n for n in _bi if is_expert(n)):
        W, S = _pair_views(pcm, _pi[name])
        W2, S2 = pair_mat(bc, dp, _bi[name])
        same_bytes = same_bytes and np.array_equal(W, W2) and np.array_equal(S, S2)
        n_cmp += 1
    log("  pcodes.bin is byte-identical to the conversion for %d expert matrices: %s"
        % (n_cmp, "yes" if same_bytes else "NO"))
    same = same and same_bytes
    pcm = None
    gc.collect()

    # what one miss costs, the two ways: three matrices of layer 0, expert 0
    precs = [_pi[n] for n in ("blk.0.moe_gate.e0.weight", "blk.0.moe_up.e0.weight", "blk.0.moe_down.e0.weight")]
    reps = 20
    t0 = time.time()
    for _ in range(reps):
        for rec in trecs:
            pair_mat(bc, dp, rec)
    ms_conv = (time.time() - t0) * 1e3 / reps
    eng = Engine(packs["pair"], ctx=64, log=lambda *x: None, expert_resident=2)
    eng.enable_paging()
    pcm = eng.pc
    t0 = time.time()
    for _ in range(reps):
        for j, rec in enumerate(precs):
            W, S = _pair_views(pcm, rec)
            dw, ds = eng.pager["slots"][0][0][j]
            np.copyto(dw, W.reshape(dw.shape))
            np.copyto(ds, S.reshape(ds.shape))
    ms_copy = (time.time() - t0) * 1e3 / reps
    mb = sum(_pair_len(rec) for rec in precs) / 1e6
    del eng
    pcm = None
    gc.collect()
    log("  one expert staged (3 matrices, %.3f MB): %.3f ms rebuilt from the trits, %.3f ms copied from pcodes.bin"
        % (mb, ms_conv, ms_copy))
    log("  every cache size and both packs give the resident output: %s"
        % ("yes" if same else "NO - the tokens must not change"))
    return same
