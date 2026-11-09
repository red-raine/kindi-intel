"""tools/awp_ridge/probe.py — the ridge/AWP experiment, in the honest order.

The pasted Gemini AWP script is not AWP: it hooks whole layer outputs, not per-linear inputs, and it thresholds
a flattened Wanda mask with no reconstruction. This probe does the steps the branch's gates actually require,
on a real-weight pack, in this order:

  1. Collect the TRUE inputs of a linear from the C engine (`Engine.generate` steps the token and the scratch
     it uses for that linear is read back; the weights the linear multiplies are snapshotted before packing).
  2. Form the activation covariance C = X^T X + lambda I and its inverse — the inverse Hessian the OBS/GPTQ
     line and true AWP use to decide how the *surviving* weights move when a weight is removed.
  3. Apply the OBS-style constraint: prune by Wanda saliency, project the pruned coordinates to exactly zero,
     and let the closed-form ridge solution move the survivors to compensate.
  4. THEN ternarise, and only then measure. A "ridge" number taken on the fractional weights before rounding
     is a different claim; the engine only ever runs ternary codes.

The probe reports output-space error ||X(W' - W)|| for each arm. It does not run the paired NLL walk; if the
ridge arm does not beat naive ternarisation here, there is no walk to run. If it does, the packed variant is
built and the paired walk (`tools.nll --pack A --against B`) becomes the gate.

Usage:
  python -u tools/awp_ridge/probe.py B:/models/scratch/qwen-layer0-pack --expert 4 [--lambda 1.0] [--sparsity 0.3]
"""
import argparse
import ctypes
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from kindi.cpu import Engine

# scratch buffers of Model (see kindi/cpu.py _SCR and _fields_ around "red"/"pr"); grab the true linear inputs.
SCR = ["h", "xn", "xp", "y", "qkv", "zv", "cs", "qk", "gout", "outn", "m", "qn", "kr", "vv", "qr", "kro", "sc",
       "att", "ffg", "ffu", "T", "T2", "logits", "gb", "gg"]


def scratch(eng, name):
    a = getattr(eng.m, name)
    # PF is a POINTER(c_float); recover the numpy view through ctypes address arithmetic.
    addr = ctypes.cast(a, ctypes.c_void_p).value
    return np.ctypeslib.as_array((ctypes.c_float * eng.m.D).from_address(addr)).copy()


def snapshot_rank1_slice(eng, layer, expert, kind):
    """One expert's gate weight [768 x 2048] as fp16, read off the Engine's live array (not the dequantised pack).

    The snapshotted version is a copy; without it the numpy arms would be measuring the engine's own staged
    bytes, which is not the AWP object.
    """
    from kindi import packio
    bi = json.load(open(os.path.join(eng.pack, "bcodes_index.json")))["tensors"]
    dp = packio.load_scales(eng.pack)
    bc = np.memmap(os.path.join(eng.pack, "bcodes.bin"), dtype=np.uint8, mode="r")
    from kindi import quant
    rec = bi["blk.%d.moe_%s.e%d.weight" % (layer, kind, expert)]
    R, C = int(rec["rows"]), int(rec["ne0"])
    codes = np.asarray(bc[rec["bc_off"]:rec["bc_off"] + R * (C // 4)]).reshape(C // 4, R)   # store layout
    scales = np.asarray(packio.tensor_scales(dp, rec)).reshape(C // 128, R)                  # store layout
    return quant.dequantise(codes, scales, R, C)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack")
    ap.add_argument("--expert", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256, help="calibration tokens (one input row each)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.01,
                    help="relative damping: lam * mean(diag(X^T X)) (the GPTQ convention is 0.01)")
    ap.add_argument("--sparsity", type=float, default=0.30, help="fraction of weights pruned before reconstruction")
    a = ap.parse_args()

    eng = Engine(a.pack, ctx=64, log=lambda *x: None)
    # gate input x is the folded post-norm activation (xp after rmsfold, before the gate product): snapshot it per step.
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
    X = np.asarray(Xs, dtype=np.float64)     # (N, D) true gate inputs
    W = snapshot_rank1_slice(eng, 0, a.expert, "gate").astype(np.float64)   # (768, 2048) target weights
    print("collected %d true gate inputs, W %s" % (X.shape[0], W.shape))

    lam = max(a.lam, 0.0) * float(np.mean(np.sum(X * X, axis=0)))
    C = X.T @ X + lam * np.eye(X.shape[1])
    Ci = np.linalg.inv(C)
    Y = X @ W.T                                    # (N, 768) the fp16 output the ridge must preserve

    def relerr(Wm):
        return float(np.linalg.norm(X @ Wm.T - Y) / max(np.linalg.norm(Y), 1e-30))

    # arm 0: fp16 target (zero by construction) ----------------------------------
    fp_err = relerr(W)

    # arm 1: naive absmean ternarisation ----------------------------------------
    from kindi import quant
    _c, _s, Wq = quant.quantise(W.astype(np.float32), scheme="absmean")
    naive = relerr(Wq.astype(np.float64))

    # arm 2: Wanda-prune, OBS-style projection, then ternarise --------------------
    row_act = np.sqrt(np.mean(X * X, axis=0))                     # (D,) column RMS activity
    sal = np.abs(W) * row_act[None, :]
    thresh = np.quantile(sal, a.sparsity)
    mask = sal > thresh                                           # surviving coordinates
    # obs_closed_form: for each output row w, w'_P = 0 and w'_F = w_F - H^-1[F,P](H^-1[P,P])^-1 w_P
    Wr = W.copy()
    Fi = np.where(mask[0])[0]
    Pi = np.where(~mask[0])[0]
    # per output row it is the same H, so precompute the gain G = H^-1[F,P] (H^-1[P,P])^-1 once.
    Hpp_inv = np.linalg.inv(Ci[np.ix_(Pi, Pi)])
    G = Ci[np.ix_(Fi, Pi)] @ Hpp_inv                              # (F, P)
    Wr[:, Fi] = W[:, Fi] - W[:, Pi] @ G.T
    Wr[:, Pi] = 0.0
    _c, _s, Wr_q = quant.quantise(Wr.astype(np.float32), scheme="absmean")
    ridge = relerr(Wr_q.astype(np.float64))

    # arm 3: the pasted-script behaviour, for contrast: flat Wanda threshold, no reconstruction ------------
    Wt = np.where(mask, W, 0.0)
    _c, _s, Wt_q = quant.quantise(Wt.astype(np.float32), scheme="absmean")
    flat = relerr(Wt_q.astype(np.float64))

    print(json.dumps({
        "expert": a.expert, "tokens": int(X.shape[0]), "lambda": lam, "sparsity": a.sparsity,
        "fp16_output_error": fp_err, "naive_absmean_output_error": naive,
        "wanda_flat_then_ternarise_output_error": flat,
        "ridge_project_then_ternarise_output_error": ridge,
        "ridge_gain_vs_naive": naive - ridge,
    }, indent=2))
    eng.reset()


if __name__ == "__main__":
    main()
