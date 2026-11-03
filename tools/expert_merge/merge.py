"""Check the expert-family merge as specified, and a version that keeps its invariants.

The specification hands us `align_and_fuse_expert_family`: permutation-align each expert's SwiGLU MLP to an anchor
expert by cosine similarity of gate_proj *weight rows*, then Fisher-weight-average the aligned tensors. Two questions
are answerable without a donor model:

  * does it recover an anchor from experts that are genuine permutations of it (the case the alignment exists for)?
  * what does it do when the family is not alignable at all (two independently trained experts)?

    python -m tools.expert_merge.merge

The answer to the first is yes, to float rounding. The answer to the second is that it returns a plausible-looking
average with no signal that anything was wrong - which is why the specification's own "assert cosine > 0.85 before
quantising" gate matters more than the merge itself.

Two defects found by running it, both fixed in `align_and_fuse`:

1. **The Fisher tensors the pipeline specifies cannot multiply the weights it merges.** Stage 0 accumulates
   `diag(X^T X)`, an input-channel second moment of shape `[d_model]`, but Stage 3 does `fisher * weight` on a
   `[d_ffn, d_model]` tensor. Element-wise Fisher merging needs a per-*weight* statistic of the same shape as the
   weight (or a scalar per expert); `diag(X) ` of the Gauss-Newton matrix for gate_proj is `[d_ffn, d_model]` only if
   you keep the input covariance's full form and project it: `F = diag(X^T G X)` with G the output-side curvature.
   The corrected function accepts either a same-shape tensor or a scalar, and refuses a wrong-shape one loudly.
2. **The permutation is applied before it is validated.** `linear_sum_assignment` is called on a possibly non-square
   cost matrix and `row_ind` is discarded; the assignment is only a permutation when the matrix is square, and there
   is no check that the resulting alignment is any good. The corrected function keeps `row_ind`, requires a square
   matrix, and returns the mean aligned cosine so a caller can gate on it.
"""
import numpy as np


def _unit_rows(w, eps):
    n = np.linalg.norm(w, axis=1, keepdims=True)
    return w / np.maximum(n, eps)


def align_and_fuse_expert_family(expert_weights, expert_fishers, eps=1e-6):
    """The specification's version, transcribed. Kept to be measured, not to be used.

    expert_weights, expert_fishers: lists of dict name -> ndarray, same keys in every dict.
    """
    base = expert_weights[0]
    keys = base.keys()
    aligned_w, aligned_f = [base], [expert_fishers[0]]
    d_ffn = base["gate_proj.weight"].shape[0]                       # computed, never used
    _ = d_ffn

    for i in range(1, len(expert_weights)):
        cur_w = {k: v.copy() for k, v in expert_weights[i].items()}
        cur_f = {k: v.copy() for k, v in expert_fishers[i].items()}

        wa = base["gate_proj.weight"]
        wt = cur_w["gate_proj.weight"]
        cost = -(wa @ wt.T) / (np.linalg.norm(wa, axis=1, keepdims=True)
                               * np.linalg.norm(wt, axis=1)[None, :] + eps)
        row_ind, col_ind = _assignment(cost)
        _ = row_ind                                                 # discarded: only valid when square
        perm = col_ind

        cur_w["gate_proj.weight"] = cur_w["gate_proj.weight"][perm, :]
        cur_f["gate_proj.weight"] = cur_f["gate_proj.weight"][perm, :]
        cur_w["up_proj.weight"] = cur_w["up_proj.weight"][perm, :]
        cur_f["up_proj.weight"] = cur_f["up_proj.weight"][perm, :]
        cur_w["down_proj.weight"] = cur_w["down_proj.weight"][:, perm]
        cur_f["down_proj.weight"] = cur_f["down_proj.weight"][:, perm]
        aligned_w.append(cur_w)
        aligned_f.append(cur_f)

    fused = {}
    for k in keys:
        num = np.zeros_like(base[k])
        den = np.zeros_like(base[k])
        for w, f in zip(aligned_w, aligned_f):
            fk = f[k] + eps
            num = num + fk * w[k]
            den = den + fk
        fused[k] = num / den
    return fused


def _assignment(cost):
    """linear_sum_assignment without a scipy import: this repo has no scipy dependency and the check is small."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:
        return _hungarian(cost)


def _hungarian(cost):
    """O(n^3) Jonker-Volgenant is overkill here; a dense Hungarian is enough for the check."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    assert n == m, "the specification's alignment assumes a square cost matrix"
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows = np.arange(1, n + 1)
    cols = np.zeros(n, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            cols[p[j] - 1] = j - 1
    return rows - 1, cols


def align_and_fuse(expert_weights, expert_fishers, eps=1e-6):
    """The same merge with its invariants kept.

    Differences from the specification:
      * a Fisher entry may be a scalar, a same-shape tensor, or a per-expert scalar; anything else raises, because
        `diag(X^T X)` from Stage 0 has shape [d_model] and cannot weight a [d_ffn, d_model] tensor;
      * every expert must carry the same keys, and gate/up/down must agree on d_ffn;
      * row_ind is kept, so the permutation is only applied when the matrix is square;
      * the mean aligned cosine comes back with the result, so a caller can gate on it.
    """
    base = expert_weights[0]
    keys = list(base.keys())
    for i, w in enumerate(expert_weights):
        if set(w.keys()) != set(keys):
            raise ValueError("expert %d has keys %s, anchor has %s" % (i, sorted(w.keys()), sorted(keys)))

    gate = base["gate_proj.weight"]
    d_ffn = gate.shape[0]
    for i, w in enumerate(expert_weights):
        if w["gate_proj.weight"].shape[0] != d_ffn or w["up_proj.weight"].shape[0] != d_ffn:
            raise ValueError("expert %d disagrees on d_ffn" % i)
        if w["down_proj.weight"].shape[1] != d_ffn:
            raise ValueError("expert %d down_proj does not have d_ffn columns" % i)

    aligned_w, aligned_f = [base], [_fisher(expert_fishers[0], keys, base, eps)]
    scores = []
    anchor_rows = _unit_rows(gate, eps)
    for i in range(1, len(expert_weights)):
        cur_w = dict(expert_weights[i])
        cur_f = _fisher(expert_fishers[i], keys, base, eps)
        rows = _unit_rows(cur_w["gate_proj.weight"], eps)
        cost = -(anchor_rows @ rows.T)
        if cost.shape[0] != cost.shape[1]:
            raise ValueError("expert %d: gate rows differ from the anchor, so no permutation aligns them" % i)
        row_ind, col_ind = _assignment(cost)
        if not np.array_equal(np.sort(col_ind), np.arange(d_ffn)):
            raise ValueError("expert %d: assignment is not a permutation" % i)
        scores.append(float(np.mean(-cost[row_ind, col_ind])))

        perm = col_ind
        w = dict(cur_w)
        w["gate_proj.weight"] = cur_w["gate_proj.weight"][perm, :]
        w["up_proj.weight"] = cur_w["up_proj.weight"][perm, :]
        w["down_proj.weight"] = cur_w["down_proj.weight"][:, perm]
        f = dict(cur_f)
        if f["gate_proj.weight"].ndim == 2:
            f["gate_proj.weight"] = f["gate_proj.weight"][perm, :]
            f["up_proj.weight"] = f["up_proj.weight"][perm, :]
            f["down_proj.weight"] = f["down_proj.weight"][:, perm]
        aligned_w.append(w)
        aligned_f.append(f)

    fused = {}
    for k in keys:
        num = np.zeros_like(np.asarray(base[k], dtype=np.float64))
        den = np.zeros_like(num)
        for w, f in zip(aligned_w, aligned_f):
            fk = f[k] + eps
            num = num + fk * w[k]
            den = den + fk
        fused[k] = num / den
    return fused, float(np.mean(scores)) if scores else 1.0


def _fisher(f, keys, base, eps):
    """normalise a Fisher spec: scalar, per-key scalar, or per-key tensor of the weight's shape"""
    out = {}
    if np.isscalar(f):
        return {k: f for k in keys}
    for k in keys:
        if k not in f:
            raise ValueError("Fisher is missing %s" % k)
        v = np.asarray(f[k], dtype=np.float64)
        if v.ndim == 0:
            out[k] = v
        elif v.shape == np.asarray(base[k]).shape:
            out[k] = v
        else:
            raise ValueError("Fisher for %s is shaped %s but the weight is %s: Stage 0's diag(X^T X) has the wrong "
                             "rank for element-wise merging" % (k, v.shape, np.asarray(base[k]).shape))
    return out


# --- the two cases the merge is supposed to distinguish ---------------------------------------------------------------

def _swiglu(rng, d_model, d_ffn):
    return {"gate_proj.weight": rng.normal(size=(d_ffn, d_model)),
            "up_proj.weight": rng.normal(size=(d_ffn, d_model)),
            "down_proj.weight": rng.normal(size=(d_model, d_ffn))}


def _permute(expert, perm):
    out = {}
    out["gate_proj.weight"] = expert["gate_proj.weight"][perm, :]
    out["up_proj.weight"] = expert["up_proj.weight"][perm, :]
    out["down_proj.weight"] = expert["down_proj.weight"][:, perm]
    return out


def check():
    rng = np.random.RandomState(11)
    d_model, d_ffn = 64, 128
    anchor = _swiglu(rng, d_model, d_ffn)
    fishers = [{k: np.abs(rng.normal(size=v.shape)) + 0.1 for k, v in anchor.items()}]

    # case A: the family really is the anchor, permuted - the alignment should undo it exactly
    family = [anchor]
    fam_f = list(fishers)
    for _ in range(2):
        perm = rng.permutation(d_ffn)
        family.append(_permute(anchor, perm))
        fam_f.append({k: np.abs(rng.normal(size=v.shape)) + 0.1 for k, v in anchor.items()})

    fused, score = align_and_fuse(family, fam_f)
    err_a = max(float(np.max(np.abs(fused[k] - anchor[k]))) for k in anchor)
    print("A. permuted copies of one expert (d_ffn=%d), 3 members" % d_ffn)
    print("   mean aligned cosine %.6f   max |fused - anchor| %.3e   %s"
          % (score, err_a, "recovers the anchor" if err_a < 1e-9 else "DOES NOT recover the anchor"))

    # the same family, but weighted by a Fisher of the wrong rank - the pipeline's own Stage 0 statistic
    bad = [{"gate_proj.weight": np.ones(d_model), "up_proj.weight": np.ones(d_model),
            "down_proj.weight": np.ones(d_model)}] * 3
    try:
        align_and_fuse(family, bad)
        print("   B. diag(X^T X) Fishers accepted - the shape check did not fire")
    except ValueError as e:
        print("   B. diag(X^T X) Fishers refused: %s" % e)

    # case B: unrelated experts. The specification's function returns a confident-looking average anyway.
    strangers = [anchor] + [_swiglu(rng, d_model, d_ffn) for _ in range(2)]
    str_f = [{k: np.abs(rng.normal(size=v.shape)) + 0.1 for k, v in anchor.items()} for _ in strangers]
    fused_s, score_s = align_and_fuse(strangers, str_f)
    err_s = max(float(np.max(np.abs(fused_s[k] - anchor[k]))) for k in anchor)
    spec_s = align_and_fuse_expert_family(strangers, str_f)
    err_spec = max(float(np.max(np.abs(spec_s[k] - anchor[k]))) for k in anchor)
    print("\nC. three unrelated experts (not alignable)")
    print("   mean aligned cosine %.6f  (compare case A: %.6f)" % (score_s, score))
    print("   max |fused - anchor| here %.3f, and %.3f from the specification's version" % (err_s, err_spec))
    print("   neither version raises; the only thing that separates A from C is the cosine it returns")

    # case D: the specification's arithmetic on case A, so the permutation logic itself is verified
    spec_a = align_and_fuse_expert_family(family, fam_f)
    err_spec_a = max(float(np.max(np.abs(spec_a[k] - anchor[k]))) for k in anchor)
    print("\nD. the specification's own function on case A: max |fused - anchor| %.3e  %s"
          % (err_spec_a, "permutation logic verified" if err_spec_a < 1e-9 else "BROKEN"))
    return 0 if (err_a < 1e-9 and err_spec_a < 1e-9) else 1


if __name__ == "__main__":
    raise SystemExit(check())
