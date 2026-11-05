"""Does an orthogonal preprocessing reduce what our ternary pack loses? Measured, not assumed.

Our pack stores 2 bits a weight: one fp16 scale a 128-weight group, three codes, round-to-nearest. "Smoothing" a
matrix with an orthogonal transform before rounding is the standard trick in the low-bit literature (QuaRot,
SpinQuant, DuQuant, QuIP#, and the GSR paper that prompted this probe) and the reported wins are large: GSR moves
Llama-2-7B from 20.29 to 11.59 WikiText-2 perplexity at 2-bit weights. But every one of those numbers is measured
inside a *different* quantiser - GPTQ with error feedback, per-channel or 4-8 bit activations, a learned rotation -
and the authors of the group-rotation line (GSR, GyRot, AMXFP4) each say the gain shrinks or reverses as the scale
group gets smaller. At our group of 128 with no error feedback, the combination is unmeasured. This measures it.

What is measured, per real matrix: the relative Frobenius error `||W - dequant(quant(W R))|| / ||W||` of the matrix
the pack would hold, and the worst single group's error. Nothing else. A weight error is a proxy for the NLL delta
(the pack's cost is a function of per-element error under the activation distribution), so a win here is a reason to
run a paired NLL walk, not a substitute for one - and this probe says so rather than quoting a perplexity it did not
measure.

What is NOT measured: end-to-end perplexity, and the activation-side cost. The cost is arithmetic and is reported
per token as a fraction of the GEMV it precedes: a 128-block Walsh transform on a 2048-wide activation is 14,336
adds against a 3.1M-flop GEMV, i.e. 0.5%, and if the transform is folded into the weights once at conversion time
it is zero per token. `fold` is the catch: BitNet v2 measured that folding the rotation into the weights *before*
ternarising is the worse order at 1.58 bits (20.83 vs 11.33 PPL), which is exactly the choice this probe prices.

  python -m tools.rotq.probe --selftest
  python -m tools.rotq.probe --gguf B:/models/Qwen3-Coder-30B-A3B-Instruct/Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf
"""
import argparse
import json
import sys

import numpy as np

GROUP = 128                 # the pack's scale group: one fp16 level a 128 weights (`kindi.convert`)


# --- the transforms ---------------------------------------------------------------------------------------------
# Every arm is an orthogonal map on the *input* dimension: W -> W R, with the engine applying R^T to the activation.
# That is a rewrite of the same function; what changes is the matrix the ternary codes have to represent.

def hadamard(n):
    """the Sylvester Hadamard matrix, normalised so H @ H.T = I (an orthogonal, symmetric transform)"""
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    if h.shape[0] != n:
        raise ValueError("%d is not a power of two" % n)
    return h / np.sqrt(n)


def sequency_order(n):
    """row indices of an n-point Hadamard sorted by sign changes - Walsh's sequency order

    Computed from the matrix rather than from a bit formula, because the formula's conventions differ between
    papers (natural, dyadic, bit-reversed) and a mistaken convention is invisible in the output: it still looks like
    a rotation. Sorting by the number of sign changes is the definition, so there is nothing to get wrong.
    """
    h = hadamard(n)
    changes = (np.abs(np.diff(np.sign(h), axis=1)) > 0).sum(axis=1)
    return np.argsort(changes, kind="stable")


def block_hadamard(n, g):
    """an n-point transform made of n/g dense Hadamard blocks of size g - a group-aligned rotation

    `g` is the quantiser's own group, so no block boundary is crossed: each group is rotated within itself. That is
    what keeps the per-group scale meaningful, and it is the arm GSR and DuQuant argue for.
    """
    if n % g:
        raise ValueError("%d does not divide into blocks of %d" % (n, g))
    return hadamard(g)


def apply_arm(name, w, rng):
    """the matrix as the pack would see it under one arm (orthogonality is asserted in `selftest`)"""
    out, d = w.shape
    if name == "none":
        return w
    if name == "hadamard":
        return w @ hadamard(d)
    if name == "walsh-sequency":
        return w @ hadamard(d)[sequency_order(d)]
    if name == "group-128":
        return (w.reshape(out, d // GROUP, GROUP) @ hadamard(GROUP)).reshape(out, d)
    if name == "group-128-sequency":
        # Sorting the transform's rows is a permutation *inside* each group, and a group-local quantiser is exactly
        # invariant to that: the 128 values stay in the same group, so they share the same scale and the same
        # multiset. It is kept as an arm precisely to pin that, not because it can differ. See `selftest`.
        h = hadamard(GROUP)[sequency_order(GROUP)]
        return (w.reshape(out, d // GROUP, GROUP) @ h).reshape(out, d)
    if name == "zigzag-128":
        # DuQuant's cheap half: transpose the (n_groups x GROUP) channel grid so that the same relative position
        # across all groups lands in one group, then rotate inside each new group. This is the arm that can move an
        # outlier out of the group that carries it - a permutation of *whole groups* cannot, it only relabels them.
        g = d // GROUP
        grid = w.reshape(out, g, GROUP).transpose(0, 2, 1).reshape(out, GROUP, g)
        h = hadamard(g) if g > 1 and (g & (g - 1)) == 0 else None
        if h is not None:
            grid = grid @ h
        return grid.reshape(out, d)
    if name == "orthogonal":
        q, r = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
        return w @ (q * np.sign(np.diag(r)))
    if name == "signed-perm":
        perm = rng.permutation(d)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), d)
        return w[:, perm] * signs
    raise KeyError(name)


def arms_for(d):
    """the arms whose transform is defined on a d-wide input

    A Hadamard needs a power of two. The expert matrices are 768 wide, where only the group-aligned arms, the
    random permutation and the dense QR are defined; skipping the others is correct, and reported - measuring a
    padded transform instead would be a different matrix.
    """
    pow2 = d & (d - 1) == 0
    out = []
    for arm in ARMS:
        if not pow2 and arm in ("hadamard", "walsh-sequency"):
            continue
        out.append(arm)
    return tuple(out)


ARMS = ("none", "signed-perm", "group-128", "group-128-sequency", "zigzag-128", "hadamard", "walsh-sequency",
        "orthogonal")

# adds a token of a d-wide activation, against the 2*out*d flops of the GEMV the matrix feeds. A folded transform is
# free at run time and is not counted here; this is the number for the activation-side variant.
def arm_cost(name, d):
    return {"none": 0, "signed-perm": 0, "group-128": d * 7, "group-128-sequency": d * 7,
            "zigzag-128": d * 7 + d, "hadamard": d * int(np.log2(d)) if d & (d - 1) == 0 else d * d,
            "walsh-sequency": d * int(np.log2(d)) if d & (d - 1) == 0 else d * d,
            "orthogonal": d * d}[name]


# --- the quantiser ----------------------------------------------------------------------------------------------

def quantise(w, scheme="absmean"):
    """the pack's ternary round: one scale a 128-group, codes in {-1,0,1}, and what it decodes back to

    `absmean` is PTQ1_0's convention (scale = mean |w| of the group). `lsq` keeps the codes and re-derives the scale
    as the least-squares optimum for them, which is the `--scheme lsq` arm of `kindi convert`. A group of zeros keeps
    a zero scale rather than dividing by it.
    """
    rows = w.shape[0]
    blocks = w.reshape(rows, -1, GROUP)
    s = np.abs(blocks).mean(axis=-1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    codes = np.clip(np.rint(blocks / s), -1.0, 1.0)
    if scheme == "lsq":
        num = (blocks * codes).sum(axis=-1, keepdims=True)
        den = (codes * codes).sum(axis=-1, keepdims=True)
        s = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    return (codes * s).reshape(rows, -1)


def error(w, rec):
    """relative Frobenius error of the reconstruction, and the worst group's relative error"""
    den = float(np.linalg.norm(w))
    if den == 0.0:
        return 0.0, 0.0
    rel = float(np.linalg.norm(w - rec)) / den
    a = w.reshape(w.shape[0], -1, GROUP)
    b = rec.reshape(rec.shape[0], -1, GROUP)
    per = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1), 1e-30)
    return rel, float(per.max())


def run_one(w, scheme="absmean", seed=0, arms=ARMS):
    """every arm's error on one matrix, in the order given"""
    out = []
    for arm in arms:
        rng = np.random.default_rng(seed)
        x = apply_arm(arm, np.asarray(w, dtype=np.float32), rng)
        rel, worst = error(x, quantise(x, scheme))
        out.append(dict(arm=arm, rel=rel, worst_group=worst, adds=arm_cost(arm, w.shape[1])))
    return out


# --- selftest ---------------------------------------------------------------------------------------------------

def selftest():
    """The transforms are orthogonal, the group arm keeps the group grid, and the metric can tell the arms apart.

    A probe whose arms are not orthogonal would be measuring a different function, and one whose metric cannot
    separate a smoothed matrix from an unsmoothed one would report a null that means nothing. Both are checked on
    synthetic matrices whose answer is known.
    """
    checks = [0]

    def check(name, ok, detail=""):
        checks[0] += 1
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    rng = np.random.default_rng(1)
    d = 256
    for arm in arms_for(d):
        r = apply_arm(arm, np.eye(d, dtype=np.float32), np.random.default_rng(7))
        check("%s is orthogonal" % arm, np.allclose(r @ r.T, np.eye(d), atol=1e-4),
              "%.2e" % float(np.abs(r @ r.T - np.eye(d)).max()))
    check("the 768-wide expert arms skip only the two global Hadamards",
          set(arms_for(768)) == set(ARMS) - {"hadamard", "walsh-sequency"}, str(arms_for(768)))
    # A permutation inside a group cannot change a group-local quantiser's error at all: same scale, same multiset.
    w = rng.standard_normal((32, 256)).astype(np.float32)
    a = run_one(w, arms=("group-128", "group-128-sequency"))
    check("a within-group permutation is exactly inert for a group-local quantiser",
          a[0]["rel"] == a[1]["rel"] and a[0]["worst_group"] == a[1]["worst_group"],
          "%r %r" % (a[0]["rel"], a[1]["rel"]))
    # ... while the zigzag, which moves channels between groups, is not: it must change the answer, or it is the
    # same no-op the whole-group permutation turned out to be.
    z = run_one(w, arms=("group-128", "zigzag-128"))
    check("the zigzag arm is not inert", z[0]["rel"] != z[1]["rel"], "%r %r" % (z[0]["rel"], z[1]["rel"]))
    # H@H.T = I, and the natural Hadamard is symmetric
    h = hadamard(d)
    check("hadamard inverse is its transpose", np.allclose(h @ h.T, np.eye(d), atol=1e-6))
    check("hadamard is symmetric", np.allclose(h, h.T, atol=1e-6))
    # sequency: row 0 is all ones, and sign changes are sorted
    seq = hadamard(256)[sequency_order(256)]
    changes = (np.abs(np.diff(np.sign(seq), axis=1)) > 0).sum(axis=1)
    check("sequency order starts at the constant row", np.allclose(seq[0], 1.0 / np.sqrt(256)))
    check("sequency order is monotone in sign changes", bool(np.all(np.diff(changes) >= 0)), str(changes[:8]))
    # a group-aligned rotation leaves each group's energy exactly as it was: that is what "no boundary crossed" means
    w = rng.standard_normal((32, 256)).astype(np.float32)
    g = (w.reshape(32, 2, 128) @ hadamard(128)).reshape(32, 256)
    e0 = (w.reshape(32, 2, 128) ** 2).sum(-1)
    e1 = (g.reshape(32, 2, 128) ** 2).sum(-1)
    check("group arm preserves each group's energy", np.allclose(e0, e1, rtol=1e-5),
          "%.2e" % float(np.abs(e0 - e1).max() / e0.max()))
    # ... while the global arm does not: it moves energy between groups, which is the mechanism under test
    e2 = ((w @ hadamard(256)).reshape(32, 2, 128) ** 2).sum(-1)
    check("global arm moves energy between groups", not np.allclose(e0, e2, rtol=1e-3))

    # A matrix the format could have produced is a fixed point of *lsq*, not of absmean: absmean divides by
    # mean|codes|, so a block containing zeros comes back with a scale multiplied by that mean. That is not a bug in
    # the probe, it is a property of the format worth pinning - it is why `--scheme lsq` is the scheme whose scale is
    # self-consistent with its own codes, and why `kindi refit` can re-derive a pack's scales exactly.
    codes = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float32), (8, 2, 128))
    s0 = np.abs(codes).mean(axis=-1, keepdims=True)
    exact = (codes * s0).reshape(8, 256)
    rel, worst = error(exact, quantise(exact, "lsq"))
    check("a format-produced matrix is an lsq fixed point", rel < 1e-6 and worst < 1e-6, "%.2e/%.2e" % (rel, worst))
    check("... and not an absmean one, because absmean divides by mean|codes|", error(exact, quantise(exact))[0] > 0.1)
    binary = (rng.choice(np.array([-1.0, 1.0], dtype=np.float32), (8, 2, 128))
              * rng.uniform(0.5, 2.0, (8, 2, 1))).reshape(8, 256)
    check("an all-nonzero block is exact under absmean, which is the Q1_0 case",
          error(binary, quantise(binary))[0] < 1e-6)
    near = exact + 0.01 * rng.standard_normal(exact.shape).astype(np.float32)
    far = exact + 0.5 * rng.standard_normal(exact.shape).astype(np.float32)
    check("the metric orders two reconstructions", error(near, quantise(near))[0] < error(far, quantise(far))[0])

    # An outlier, and the mechanism that actually moves it. A group-aligned rotation preserves each group's energy
    # exactly, which suggests it cannot help a block-local outlier - but it does, and the reason is clipping rather
    # than energy: the scale is a group mean, so an element 60x the block is coded as 1 and lost. Rotating within the
    # block spreads it over 128 elements whose largest is 60/sqrt(128) of it, so nothing is clipped away. Both arms
    # must show it, at different spreads, or the arms are measuring something other than smoothing.
    heavy = rng.standard_normal((16, 256)).astype(np.float32) * 0.05
    heavy[:, 7] *= 60.0
    rows = run_one(heavy, arms=("none", "group-128", "hadamard"))
    by = {r["arm"]: r["rel"] for r in rows}
    check("a global rotation helps a planted outlier a lot", by["hadamard"] < by["none"] * 0.5, str(by))
    check("a group rotation helps it too: the clip is spread, not the energy", by["group-128"] < by["none"] * 0.5,
          str(by))
    # Which of the two wins is data-dependent, and here the group arm does: spreading over 128 columns raises that
    # block's scale less than a global spread raises every block's. Do not assert an ordering - record it.
    check("the two arms land within a factor of two of each other", max(by["group-128"], by["hadamard"]) <
          2 * min(by["group-128"], by["hadamard"]), str(by))

    # the scales behave: lsq is never worse than absmean on the same codes
    w2 = rng.standard_normal((16, 256)).astype(np.float32)
    check("lsq fits at least as well as absmean",
          error(w2, quantise(w2, "lsq"))[0] <= error(w2, quantise(w2, "absmean"))[0] + 1e-6)
    # determinism: a seeded arm is the same arm
    a = apply_arm("orthogonal", w2, np.random.default_rng(3))
    b = apply_arm("orthogonal", w2, np.random.default_rng(3))
    check("a seeded transform is reproducible", np.array_equal(a, b))
    return checks[0]


# --- the real matrices ------------------------------------------------------------------------------------------

# one of each kind of matrix the pack holds, at both ends of the depth: an attention projection, the output
# projection, an expert gate (2048 wide) and an expert down (768 wide, the smallest in-dim the pack has)
DEFAULT = (("blk.0.attn_q.weight", 0), ("blk.0.attn_output.weight", 0), ("blk.0.ffn_gate_exps.weight", 0),
           ("blk.24.ffn_down_exps.weight", 0), ("blk.47.attn_q.weight", 0), ("blk.47.ffn_up_exps.weight", 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description="group rotations against this pack's ternary quantiser")
    ap.add_argument("--gguf", help="a GGUF to read matrices from")
    ap.add_argument("--tensor", action="append", default=None, help="NAME[:EXPERT], repeatable")
    ap.add_argument("--scheme", default="absmean", choices=("absmean", "lsq"))
    ap.add_argument("--json", help="write the table here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        print("%d checks passed" % selftest())
        return 0
    if not a.gguf:
        ap.error("--gguf is needed (or --selftest)")
    from tools.expert_merge.ggufmoe import Moepack
    m = Moepack(a.gguf)
    picks = []
    for spec in (a.tensor or ["%s:%d" % (n, e) for n, e in DEFAULT]):
        name, _, e = spec.partition(":")
        picks.append((name, int(e or 0)))
    rows = []
    for name, e in picks:
        if name not in m.info:
            print("skip %s: not in %s" % (name, a.gguf))
            continue
        w = m.expert_weight(name, e) if name.endswith("_exps.weight") else m.tensor(name)
        got = run_one(w, a.scheme, arms=arms_for(w.shape[1]))
        base = got[0]["rel"]
        rows.append(dict(tensor=name, expert=e, shape=list(w.shape), arms=got))
        print("\n%s%s  [%d x %d]  scheme %s" % (name, " expert %d" % e if name.endswith("_exps.weight") else "",
                                                w.shape[0], w.shape[1], a.scheme))
        print("  %-22s %9s %9s %7s  %8s  %7s" % ("arm", "rel err", "vs none", "worst", "adds/tok", "of GEMV"))
        for r in got:
            print("  %-22s %9.5f %8.1f%% %7.4f  %8d  %6.2f%%" %
                  (r["arm"], r["rel"], 100.0 * (r["rel"] / base - 1.0), r["worst_group"], r["adds"],
                   100.0 * r["adds"] / (2.0 * w.shape[0] * w.shape[1])))
    if rows:
        # Per matrix, not per list: an arm skipped on the 768-wide experts must not be averaged against a `none`
        # list of a different length. Each arm is compared only where both it and `none` were measured.
        print("\nrelative error against `none` on the same matrix (mean, and the range over matrices)")
        for arm in ARMS:
            deltas = []
            for r in rows:
                got = {x["arm"]: x["rel"] for x in r["arms"]}
                if arm in got and "none" in got:
                    deltas.append(got[arm] / got["none"] - 1.0)
            if not deltas:
                continue
            deltas = np.array(deltas)
            verdict = "better" if deltas.mean() < -0.005 else ("worse" if deltas.mean() > 0.005 else "level")
            print("  %-22s %+7.1f%%  [%+6.1f%% .. %+6.1f%%]  %s over %d matrices" %
                  (arm, 100.0 * deltas.mean(), 100.0 * deltas.min(), 100.0 * deltas.max(), verdict, len(deltas)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(gguf=a.gguf, scheme=a.scheme, group=GROUP, rows=rows), f, indent=1)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
