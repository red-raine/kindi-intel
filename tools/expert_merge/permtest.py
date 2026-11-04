"""Is a cluster of experts one function wearing a permuted hidden layer, or several similar functions?

The question decides what a merge means. Two experts whose hidden units are a relabelling of each other compute the
*same* function, so averaging them after alignment is exact and averaging them before it is not; two experts that
are merely similar compute different functions, and no relabelling makes the average equal to either.

The test is on weights, and it is deliberately draconian. If expert B is expert A with hidden unit i renamed to
pi(i) and optionally negated, then

    gate_B[pi(i)] = s_i * gate_A[i],   up_B[pi(i)] = s_i * up_A[i],   down_B[:, pi(i)] = down_A[:, i] * s_i

so the cross-correlation of the 768 gate rows is a signed permutation matrix: every row of A has exactly one
partner in B with |cos| = 1, the same partner shows up in gate, up and down, and the residual after applying pi
falls to zero - exactly zero here, because quantisation is *row-equivariant*: a row of 2048 Q2_K weights is the
same 84-byte blocks wherever it is stored, so a permutation cannot change the stored values.

That means the answer is not a threshold judgement on a similarity: either the rows match one-to-one at machine
precision, or they do not. What is left to report is how close a *non*-permutation gets - the best bijection's
residual against the identity's, and against an unconstrained orthogonal alignment, which is the most generous
relabelling there is.
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:                                            # pragma: no cover
    linear_sum_assignment = None


# --- weights -----------------------------------------------------------------------------------------------------

def load(pack, il, e):
    s = "blk.%d." % il
    return dict(gate=pack.expert_weight(s + "ffn_gate_exps.weight", e).astype(np.float32),
                up=pack.expert_weight(s + "ffn_up_exps.weight", e).astype(np.float32),
                down=pack.expert_weight(s + "ffn_down_exps.weight", e).astype(np.float32))


def permute(E, pi=None, sign=None):
    """apply a hidden-unit relabelling to an expert, in the same direction on all three matrices"""
    n = E["gate"].shape[0]
    pi = np.arange(n) if pi is None else np.asarray(pi)
    sign = np.ones(n, np.float32) if sign is None else np.asarray(sign, np.float32)
    G, U, D = np.empty_like(E["gate"]), np.empty_like(E["up"]), np.empty_like(E["down"])
    G[pi] = E["gate"] * sign[:, None]
    U[pi] = E["up"] * sign[:, None]
    D[:, pi] = E["down"] * sign[None, :]
    return dict(gate=G, up=U, down=D)


def _row_unit(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-30)


def _col_unit(M):
    n = np.linalg.norm(M, axis=0, keepdims=True)
    return M / np.maximum(n, 1e-30)


def match(A, B):
    """the bijection that best maps A's hidden units onto B's, using all three matrices as evidence

    Rows of gate/up and columns of down are the same object under the relabelling, so all three vote. The
    assignment maximises the sum of |cos| (a permutation may carry a sign, and the merge does not care which).
    """
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is needed for the assignment")
    ca = (_row_unit(A["gate"]) @ _row_unit(B["gate"]).T
          + _row_unit(A["up"]) @ _row_unit(B["up"]).T
          + _col_unit(A["down"]).T @ _col_unit(B["down"]))
    S = np.abs(ca) / 3.0
    r, c = linear_sum_assignment(-S)
    pi = np.empty_like(r)
    pi[r] = c                                              # unit i of A -> unit pi[i] of B
    best = S[r, c]
    return pi, best


def _best_sign(A, B, pi, best=None):
    """the sign of each matched pair, decided on the gate row (a negation shows up here as a negative dot)"""
    g = (_row_unit(A["gate"]) * _row_unit(B["gate"])[pi]).sum(axis=1)
    return np.where(g < 0, -1.0, 1.0)


def orthogonal(B, A):
    """B after the best orthogonal map from A - the most a relabelling is allowed to do"""
    out = {}
    for k in ("gate", "up", "down"):
        M, N = A[k], B[k]
        u, _s, vt = np.linalg.svd(M.T @ N, full_matrices=False)
        Q = u @ vt
        out[k] = M @ Q
    return out


def rel_error(B, A):
    num = sum(float(np.sum((B[k] - A[k]) ** 2)) for k in ("gate", "up", "down"))
    den = sum(float(np.sum(A[k] ** 2)) for k in ("gate", "up", "down"))
    return float(np.sqrt(num / max(den, 1e-30)))


# --- the test ----------------------------------------------------------------------------------------------------

def classify(pack, il_a, ea, il_b, eb, rng=None):
    rng = np.random.default_rng(0 if rng is None else rng)
    A, B = load(pack, il_a, ea), load(pack, il_b, eb)
    return classify_pair(A, B, rng=rng)


def classify_pair(A, B, rng=None):
    rng = np.random.default_rng(0 if rng is None else rng)
    pi, best = match(A, B)
    sign = _best_sign(A, B, pi, best)
    Ap = permute(A, pi, sign)
    n = A["gate"].shape[0]
    r_id = rel_error(B, A)
    r_perm = rel_error(B, Ap)
    r_orth = rel_error(B, orthogonal(B, A))
    r_rand = float(np.mean([rel_error(B, permute(A, rng.permutation(n), rng.choice([-1.0, 1.0], n)))
                            for _ in range(3)]))
    # how one-to-one the match is, independently per matrix
    cg = np.abs(_row_unit(A["gate"]) @ _row_unit(B["gate"]).T).max(axis=1)
    cu = np.abs(_row_unit(A["up"]) @ _row_unit(B["up"]).T).max(axis=1)
    bidir = float(np.mean(np.argmax(np.abs(_row_unit(A["gate"]) @ _row_unit(B["gate"]).T), axis=1)
                          == np.argmax(np.abs(_row_unit(A["up"]) @ _row_unit(B["up"]).T), axis=1)))
    frac = lambda v, t: float(np.mean(v > t))
    out = dict(r_id=r_id, r_perm=r_perm, r_orth=r_orth, r_rand=r_rand,
               row_cos_gate=float(np.median(cg)), row_cos_up=float(np.median(cu)),
               frac_gt_99=min(frac(cg, 0.99), frac(cu, 0.99)), frac_gt_90=min(frac(cg, 0.9), frac(cu, 0.9)),
               same_partner=bidir, gain=float(1.0 - r_perm / max(r_id, 1e-30)), pi=pi, sign=sign)
    out["verdict"] = verdict(out)
    return out


def verdict(m, tol=1e-6):
    """strict: a permutation leaves nothing behind once the relabelling is applied

    Quantisation is row-equivariant, so a permutation has to come back *exactly*, not approximately. The row-cos and
    bijection statistics are reported beside the verdict rather than gating it: a hidden unit whose row is all zeros
    (or a tie between two near-identical rows) makes their `argmax` arbitrary even when the weights are a perfect
    relabelling, which is a fact about that unit and not about the pair.
    """
    if m["r_perm"] < tol:
        return "identical" if m["r_id"] < tol else "permutation"
    if m["gain"] > 0.10:
        return "aligned-similar"          # a bijection buys something, but the pair is not one function
    return "merely-similar"


# --- what a merge costs, on the activations the router actually sent --------------------------------

def ffn(E, X):
    """expert output: down( silu(gate x) * up x ), batch over rows of X"""
    g = X @ E["gate"].T
    u = X @ E["up"].T
    a = g / (1.0 + np.exp(-g)) * u
    return a @ E["down"].T


def align_into(A, B):
    """B relabelled into A's hidden-unit frame, so the two are directly comparable (and averageable)

    `match` reports pi[i] = the unit of B that A's unit i corresponds to, which is the map *out* of A. Putting B
    into A's frame therefore needs the inverse permutation, with each pair's sign carried to the slot it lands in.
    """
    pi, best = match(A, B)
    s = _best_sign(A, B, pi, best)
    inv = np.empty_like(pi)
    inv[pi] = np.arange(pi.size)
    return permute(B, inv, s[inv]), pi


def merge(A, B, aligned_first=False):
    """average two experts, before or after putting B into A's hidden-unit frame"""
    if aligned_first:
        B, _pi = align_into(A, B)
    return {k: (A[k] + B[k]) / 2.0 for k in ("gate", "up", "down")}


def _swiglu(E):
    """the same expert under merge.py's keys, so the merge under test is the module's own"""
    return {"gate_proj.weight": E["gate"], "up_proj.weight": E["up"], "down_proj.weight": E["down"]}


def merge_cost(A, B, X, spec=True):
    """relative error of the merged expert against each parent, on real activation rows

    `spec` runs the alignment the specification actually uses - `merge.align_and_fuse`, gate rows only, plain
    average - and returns its mean aligned cosine, the number the pipeline's 0.85 gate is applied to. That way the
    merge being judged is the shipped one, not a second implementation of it.
    """
    yA, yB = ffn(A, X), ffn(B, X)
    rows = []
    for tag in ("raw", "aligned"):
        if tag == "raw":
            M, score = merge(A, B), float("nan")
        elif spec:
            from .merge import align_and_fuse
            fused, score = align_and_fuse([_swiglu(A), _swiglu(B)], [1.0, 1.0])
            M = dict(gate=fused["gate_proj.weight"], up=fused["up_proj.weight"],
                     down=fused["down_proj.weight"])
        else:
            M, score = merge(A, B, aligned_first=True), float("nan")
        yM = ffn(M, X)
        rows.append(dict(how=tag, cosine=score,
                         err_a=float(np.linalg.norm(yM - yA) / max(np.linalg.norm(yA), 1e-30)),
                         err_b=float(np.linalg.norm(yM - yB) / max(np.linalg.norm(yB), 1e-30)),
                         err_a_vs_b=float(np.linalg.norm(yB - yA) / max(np.linalg.norm(yA), 1e-30))))
    # what two *identical* copies would give, so a small number can be read against the floor
    yI = ffn(merge({k: A[k] for k in A}, {k: A[k] for k in A}), X)
    floor = float(np.linalg.norm(yI - yA) / max(np.linalg.norm(yA), 1e-30))
    for r in rows:
        r["floor"] = floor
    return rows


# --- self-test on planted answers ---------------------------------------------------------------------------------

def selftest(log=print):
    rng = np.random.default_rng(7)
    n, d, h = 64, 128, 96
    A = dict(gate=rng.standard_normal((h, d)).astype(np.float32),
             up=rng.standard_normal((h, d)).astype(np.float32),
             down=rng.standard_normal((d, h)).astype(np.float32))
    nchk = 0

    def check(name, cond, detail):
        nonlocal nchk
        nchk += 1
        if not cond:
            raise AssertionError("%s: %s" % (name, detail))
        log("  ok   %-58s %s" % (name, detail))

    # 1. a planted permutation must come back as one, bit for bit
    pi = rng.permutation(h)
    s = rng.choice([-1.0, 1.0], h)
    P = permute(A, pi, s)
    m = classify_pair(A, P)
    check("a real permutation is recovered exactly", m["verdict"] == "permutation" and m["r_perm"] < 1e-12,
          "verdict %s, residual after alignment %.1e" % (m["verdict"], m["r_perm"]))
    check("the recovered mapping is the planted one", bool(np.array_equal(m["pi"], pi)),
          "all %d units matched" % h)

    # 1b. an exact copy is identical, which is a different statement from a permutation
    ident = classify_pair(A, {k: A[k].copy() for k in A})
    check("an exact copy is reported as identical", ident["verdict"] == "identical",
          "verdict %s" % ident["verdict"])

    # 2. a copy with noise is similar, not permuted
    B2 = {k: (A[k] + 0.05 * rng.standard_normal(A[k].shape).astype(np.float32)) for k in A}
    m2 = classify_pair(A, B2)
    check("a noisy copy is not a permutation", m2["verdict"] != "permutation" and m2["r_perm"] > 1e-2,
          "verdict %s, residual %.3f (identity %.3f)" % (m2["verdict"], m2["r_perm"], m2["r_id"]))

    # 3. two unrelated experts are neither
    C = dict(gate=rng.standard_normal((h, d)).astype(np.float32),
             up=rng.standard_normal((h, d)).astype(np.float32),
             down=rng.standard_normal((d, h)).astype(np.float32))
    m3 = classify_pair(A, C)
    check("unrelated experts earn no alignment gain", m3["verdict"] == "merely-similar" and m3["gain"] < 0.1,
          "verdict %s, gain %.3f" % (m3["verdict"], m3["gain"]))

    # 4. merging: exact after alignment for a permutation, lossy either way for a noisy copy
    X = rng.standard_normal((32, d)).astype(np.float32)
    cost_perm = {r["how"]: r for r in merge_cost(A, P, X, spec=False)}
    check("merging a signed permutation is exact after alignment", cost_perm["aligned"]["err_a"] < 1e-12,
          "aligned err %.1e, raw err %.3f" % (cost_perm["aligned"]["err_a"], cost_perm["raw"]["err_a"]))
    cost_sim = {r["how"]: r for r in merge_cost(A, B2, X, spec=False)}
    check("merging similar experts is lossy with and without alignment",
          cost_sim["aligned"]["err_a"] > 1e-3 and cost_sim["aligned"]["err_a"] >= cost_sim["aligned"]["floor"],
          "aligned err %.3f, raw err %.3f, floor %.1e"
          % (cost_sim["aligned"]["err_a"], cost_sim["raw"]["err_a"], cost_sim["aligned"]["floor"]))

    # 5. the shipped merge's own statistic, on a permutation it can represent (no sign flips)
    plain = permute(A, pi)
    cost_spec = {r["how"]: r for r in merge_cost(A, plain, X)}
    check("the specified merge reports cosine 1.0 and recovers the anchor",
          abs(cost_spec["aligned"]["cosine"] - 1.0) < 1e-6 and cost_spec["aligned"]["err_a"] < 1e-5,
          "cosine %.6f, err %.1e (float32 rounding: the merge accumulates in float64)"
          % (cost_spec["aligned"]["cosine"], cost_spec["aligned"]["err_a"]))
    # ... and what it does when half the units are negated, which its signed-cosine cost cannot represent. The
    # assignment chases the positive matches instead, the cosine collapses, and the merged expert is not either
    # parent - the gate catches it, which is the gate doing its job.
    cost_signed = {r["how"]: r for r in merge_cost(A, P, X)}
    check("a signed permutation defeats the unsigned alignment, and the gate sees it",
          cost_signed["aligned"]["err_a"] > 0.05 and cost_signed["aligned"]["cosine"] < 0.85,
          "cosine %.3f (gate 0.85), merged output %.3f off the anchor"
          % (cost_signed["aligned"]["cosine"], cost_signed["aligned"]["err_a"]))
    log("\n  %d checks passed" % nchk)
    return nchk


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
