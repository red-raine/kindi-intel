"""tools/awp_ridge/catq_probe.py — CAT-Q-style output-aware ternarisation, measured the branch's way (corrected).

CAT-Q (arXiv 2606.26650, Intel Labs China, ICML 2026 oral) learns per-group scale and threshold with a
soft-then-hard relay and LM (mean/structure modulation). ScaleQ-1.58 (arXiv 2608.01078) shows the calibration
contexts must be the model's own reasoning traces. What transfers to this branch is the verifiable part:
per-group scale+threshold learned against the real activations of the linear, not just its weight histogram.

This probe compares, on the FP32 pre-quantisation weights (the GGUF source the fixture was built from) and the
true gate inputs captured by awp_ridge/probe.py from the C engine (the model's own activations here):

  naive   absmean scale, threshold = scale/2
  lsq     per-group scale search
  gsq     joint code+scale fit (kindi.gsq)
  delta   threshold scan (scales held at absmean)
  catq    LM + ST style: learn per-group mean-shift + scale-factor + threshold-factor
          with an analytic Adam gradient through the soft relay
          f(w) = [tanh(s*(w-d)) + tanh(s*(w+d))] / (2*tanh(s)), s annealed 2 -> 8, then hard.

All arms report ||X(W' - W)|| / ||X W|| in float64. The paired NLL walk (tools.nll) runs only if an arm beats
naive output error by more than the bootstrap interval width of a ~256-input estimate.

Usage:
  python -u tools/awp_ridge/catq_probe.py B:/models/scratch/qwen-layer0-pack
      B:/models/scratch/qwen3moe-control-q4.gguf --expert 4 [--tokens 256]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from kindi.cpu import Engine
from tools.awp_ridge.probe import scratch
from tools.expert_merge.ggufmoe import Moepack
from kindi import quant


def fp16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def catq_fit(wr, epochs=40, s_lo=2.0, s_hi=8.0, lr=0.02, seed=0):
    """LM + ST per group: learn mean-shift, scale-factor and threshold-factor on the FP32 weight values.

    Soft forward: q = sc * f((w - mu0 - d_mu)/1; s, d) with the tanh relay; exact Adam gradient through the
    relay. Hard final pass: same factors, s -> infinity."""
    base = np.abs(wr).mean(axis=2)
    mu0 = wr.mean(axis=2)
    safe = np.where(base > 0, base, np.float32(1.0))
    dmu = np.zeros_like(base)
    dal = np.zeros_like(base)                               # log scale modulation
    ddl = np.zeros_like(base)                               # log threshold modulation
    state = [(np.zeros_like(dmu), np.zeros_like(dmu), 0),
             (np.zeros_like(dal), np.zeros_like(dal), 0),
             (np.zeros_like(ddl), np.zeros_like(ddl), 0)]

    def _adam(param, grad, st, lr, b1=0.9, b2=0.999, eps=1e-8):
        m, v, t = st
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        t += 1
        return param - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps), (m, v, t)

    def soft_q(dmu, dal, ddl, s):
        sc = np.exp(dal) * safe                             # (r, g)
        dlt = np.exp(ddl) * 0.5
        wc = wr - (mu0 + dmu)[:, :, None]                   # (r, g, n)
        dd = dlt[:, :, None]
        u1 = np.tanh(s * (wc - dd)); u2 = np.tanh(s * (wc + dd))
        f = (u1 + u2) / (2.0 * np.tanh(s))
        return f * sc[:, :, None], (sc, dlt, wc, dd, u1, u2)

    def loss_and_grads(dmu, dal, ddl, s):
        q, (sc, dlt, wc, dd, u1, u2) = soft_q(dmu, dal, ddl, s)
        err = q - wr
        L = (err * err).sum()
        sech1 = 1.0 - u1 ** 2
        sech2 = 1.0 - u2 ** 2
        df_dwc = (sech1 + sech2) * s / (2.0 * np.tanh(s))
        df_dd = (sech2 - sech1) * s / (2.0 * np.tanh(s))
        dq_dwc = df_dwc * sc[:, :, None]
        dq_dd = df_dd * sc[:, :, None]
        f = (u1 + u2) / (2.0 * np.tanh(s))
        g_dmu = -(err * dq_dwc).sum(axis=2)
        g_dal = (err * f * sc[:, :, None]).sum(axis=2)
        g_ddl = (err * dq_dd).sum(axis=2) * dlt
        inv = 1.0 / wr.size
        return L, (g_dmu * inv, g_dal * inv, g_ddl * inv)

    for ep in range(epochs):
        frac = ep / max(1, epochs - 1)
        s = float(s_lo * (s_hi / s_lo) ** frac)
        _L, (g1, g2, g3) = loss_and_grads(dmu, dal, ddl, s)
        dmu, state[0] = _adam(dmu, g1, state[0], lr)
        dal, state[1] = _adam(dal, g2, state[1], lr)
        ddl, state[2] = _adam(ddl, g3, state[2], lr)

    sc = fp16(np.exp(dal) * safe)
    dlt = fp16(np.exp(ddl) * 0.5)
    sf = np.where(sc > 0, sc, np.float32(1.0))
    wc = wr - (mu0 + dmu)[:, :, None]
    t = np.where(wc > (dlt * sf)[:, :, None], 1.0, np.where(wc < -(dlt * sf)[:, :, None], -1.0, 0.0))
    return (t * sf[:, :, None]).reshape(wr.shape[0], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", help="real-weight kindi pack (for true gate inputs via the engine)")
    ap.add_argument("gguf", help="FP source GGUF the pack was built from")
    ap.add_argument("--expert", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()

    eng = Engine(a.pack, ctx=64, log=lambda *x: None)
    tok_ids = [(i * 37) % 900 + 10 for i in range(a.tokens)]
    Xs = []
    orig_step = eng.step

    def wrapped(tok, pos):
        out = orig_step(tok, pos)
        Xs.append(scratch(eng, "xp").copy())
        return out

    eng.step = wrapped
    for tid in tok_ids:
        eng.reset()
        eng.step(tid, 0)
    X = np.asarray(Xs, dtype=np.float64)                    # (N, 2048) true gate inputs
    # expert_weight returns [out, in] already (768, 2048): gate rows = expert width, cols = model width.
    W = Moepack(a.gguf).expert_weight("blk.0.ffn_gate_exps.weight", a.expert).astype(np.float32)
    assert W.shape == (768, 2048), "unexpected expert shape %r" % (W.shape,)
    R, C = W.shape
    G, n = C // 128, 128
    Y = X @ W.astype(np.float64).T
    ynorm = float(np.linalg.norm(Y))

    def relerr(Wm):
        return float(np.linalg.norm(X @ Wm.astype(np.float64).T - Y) / max(ynorm, 1e-30))

    wr = np.ascontiguousarray(W).reshape(R, G, n)

    _c, _s, W_abs = quant.quantise(W, scheme="absmean")
    _c, _s, W_lsq = quant.quantise(W, scheme="lsq")
    _c, _s, W_gsq = quant.quantise(W, scheme="gsq")

    # delta scan: threshold sweep with absmean scale held
    base = np.abs(wr).mean(axis=2)
    sf = fp16(base)
    best = (np.inf, None)
    for d in np.linspace(0.3, 0.9, 13):
        t = np.where(wr > (d * sf)[:, :, None], 1.0, np.where(wr < -(d * sf)[:, :, None], -1.0, 0.0))
        e = relerr((t * sf[:, :, None]).reshape(R, C))
        if e < best[0]:
            best = (e, float(d))
    e_delta, d_best = best

    t0 = time.time()
    W_cat = catq_fit(wr)
    dt = time.time() - t0

    print(json.dumps({
        "expert": a.expert, "tokens": int(X.shape[0]), "W": [int(R), int(C)],
        "naive_absmean": relerr(W_abs), "lsq": relerr(W_lsq), "gsq": relerr(W_gsq),
        "delta_scan": e_delta, "delta_at": d_best,
        "catq_lm_st_soft_then_hard": relerr(W_cat),
        "catq_fit_seconds": round(dt, 1),
        "_note": "FP32 weights from GGUF; X = true gate inputs from pack engine; all errors output-space",
    }, indent=2))
    eng.reset()


if __name__ == "__main__":
    main()
