"""Can a 64-row router be re-fitted in closed form to the teacher's routing, better than pooling its rows?

This is `P4`'s cheap half. `masterplan.md` §3 P4 asks for a router fitted to reproduce the teacher's routing
distribution on the merged stacks, and gates it on a 1,275-token paired NLL walk. That walk costs ~1.5 h and needs a
pack written with the fitted rows, so the question worth answering first is whether the fit has any headroom at all:
the route log already holds what a fit would be fitted on - `xn`, the vector the experts saw, and `route_logits`, the
teacher's 128-way pre-softmax scores for the same token - and pooling those logits into 64 groups is exactly what the
shipped merged pack's router does (a routing-mass-weighted mean of member rows is, by linearity, the same weighted
mean of member logits). So both arms can be compared *before* a pack exists.

**The estimator is the one P4 specifies: ridge regularised towards the pooled rows.** Fit `W` to the residual between
the teacher's grouped log-probabilities and the pooled logits,

    argmin_W  || (X W^T) - (log p_64 - L_pool) ||^2 + lambda || W ||^2 ,    prediction = L_pool + X W^T

so `lambda -> inf` is *exactly* the pooled router and the arm can only interpolate away from it. That matters: an
un-anchored fit is free to be worse than the shipped router for a reason that has nothing to do with the question,
and this file's first version was un-anchored and measured exactly that (+0.052 KL, resolved, all four folds). The
anchored form makes the number a statement about re-fitting, not about a bad regulariser.

Measured per layer, on held-out documents (fit on three of the four logged documents, score the fourth, all four
folds; `lambda` chosen per fold on that fold's held-out tokens):

  * `pooled`   - softmax over the mass-weighted mean of the member logits: the shipped arm's router, `lambda = inf`.
  * `ridge`    - the anchored fit above, best `lambda` from the grid, one 64-row matrix per layer.
  * `shuffled` - the same fit against another token's routing, as a kill control: if a fit to the wrong tokens does
                 as well as the fit to the right ones, the number is fitting noise.

The verdict is a **paired** per-token KL difference, ridge minus pooled, with a moving-block bootstrap interval
(`tools.nll.bootstrap_ci`) - the same pairing rule the NLL gates use. An interval covering zero is reported as
unresolved. This is a *ceiling* for router re-fitting, not an NLL result: it says how much of the teacher's grouped
routing a linear map can recover from the logged tokens, and therefore whether the merged pack's +0.128 routing half
could be bought back by any router at this token budget. The pack-side walk stays unrun until this says the fit is
worth writing.

    python -m tools.expert_merge.routerfit --routes routes.npz --groups 64
    python -m tools.expert_merge.routerfit --routes routes.npz --groups 64 --json routerfit.json
    python -m tools.expert_merge.routerfit --selftest

Token counts are printed with every number: the four logged documents are 384 tokens of one corpus, and a different
corpus is a re-run, not a reuse.
"""
import json
import sys

import numpy as np

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
TOP = 8


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def grouped_targets(logits, gids, n_groups):
    """The teacher's 128-way distribution summed inside each group: the distribution the merged router should match."""
    p = softmax(logits)
    out = np.zeros((p.shape[0], n_groups), dtype=np.float64)
    for g in range(n_groups):
        cols = np.flatnonzero(gids == g)
        if cols.size:
            out[:, g] = p[:, cols].sum(axis=1)
    return out


def pooled_logits(logits, gids, n_groups, mass=None):
    """What the shipped merged router scores: the routing-mass-weighted mean of the member rows, i.e. of the logits."""
    n = logits.shape[0]
    w = np.ones(logits.shape[1], dtype=np.float64) if mass is None else np.asarray(mass, dtype=np.float64)
    out = np.zeros((n, n_groups), dtype=np.float64)
    for g in range(n_groups):
        cols = np.flatnonzero(gids == g)
        if cols.size:
            wg = w[cols]
            tot = wg.sum()
            if not np.isfinite(tot) or tot <= 0:      # nothing routes here: fall back to the plain mean
                out[:, g] = logits[:, cols].mean(axis=1)
            else:
                out[:, g] = logits[:, cols] @ (wg / tot)
    return out


def fit_ridge(X, R, lam):
    """Closed-form ridge on the residual: argmin ||X W^T - R||^2 + lam ||W||^2, returned as [G, d], plus the centers.

    Centering matters because `xn` has a large common component (RMS-norm output is not zero-mean), and an
    uncentered normal matrix is dominated by it: without it the fit is a bias term pretending to be a direction.
    """
    X = np.asarray(X, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    xm = X.mean(axis=0)
    rm = R.mean(axis=0)
    Xc = X - xm
    Rc = R - rm
    A = Xc.T @ Xc
    A[np.diag_indices_from(A)] += lam
    W = np.linalg.solve(A, Xc.T @ Rc).T                 # [G, d]
    return W, xm, rm


def predict(W, xm, rm, X):
    return ((np.asarray(X, dtype=np.float64) - xm) @ W.T) + rm


def per_token_kl(p, q_logits):
    q = softmax(q_logits)
    return np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(q, 1e-12))), axis=1)


def kl(p, q_logits):
    return float(per_token_kl(p, q_logits).mean())


def top8_overlap(p, q_logits, k=TOP):
    """How many of the teacher's top-k groups the arm also ranks in its top k, per token, averaged."""
    k = min(k, p.shape[1])
    a = np.argsort(-p, axis=1)[:, :k]
    b = np.argsort(-np.asarray(q_logits, dtype=np.float64), axis=1)[:, :k]
    hit = [len(set(a[i]) & set(b[i])) / float(k) for i in range(p.shape[0])]
    return float(np.mean(hit))


def _mass(routes):
    """Routing mass per expert, averaged over layers - the currency `mergepack` pools rows with."""
    wts = np.asarray(routes["route_wts"], dtype=np.float64)
    ids = np.asarray(routes["route_ids"]).astype(np.int64)
    E = int(np.asarray(routes["route_logits"]).shape[2])
    m = np.zeros(E, dtype=np.float64)
    for L in range(ids.shape[0]):
        np.add.at(m, ids[L].ravel(), wts[L].ravel())
    m /= max(1, ids.shape[0])
    return np.maximum(m, 0.0)


def evaluate(routes, gids, n_groups, log=print, center=True):
    """Leave-one-document-out, one pass; returns folds, the paired token differences and the pooled reference.

    `center=False` drops the intercept: the fit then predicts `L_pool + x W^T` with no bias term, which is the form a
    pack can actually carry (the router is `h @ W.T`, no bias). The intercept is what the emitted rows would have to
    drop anyway, so the constrained number is the one that applies to a walk.
    """
    rl = np.asarray(routes["route_logits"], dtype=np.float64)
    xn = routes["xn"]
    doc = np.asarray(routes["doc"]).astype(np.int64)
    n_layers = rl.shape[0]
    docs = sorted(set(doc.tolist()))
    if len(docs) < 2:
        raise SystemExit("routerfit: a held-out fold needs at least two documents in the log")
    mass = _mass(routes)

    folds, all_diffs = [], []
    for held in docs:
        test = doc == held
        train = ~test
        pooled_by_layer, ridge_by_layer = {}, {lam: {} for lam in LAMBDAS}
        for L in range(n_layers):
            Xtr = np.asarray(xn[L][train], dtype=np.float64)
            Xte = np.asarray(xn[L][test], dtype=np.float64)
            Ttr = grouped_targets(rl[L][train], gids, n_groups)
            Tte = grouped_targets(rl[L][test], gids, n_groups)
            Lpool_tr = pooled_logits(rl[L][train], gids, n_groups, mass=mass)
            Lpool_te = pooled_logits(rl[L][test], gids, n_groups, mass=mass)
            pooled_by_layer[L] = per_token_kl(Tte, Lpool_te)
            scale = float(np.trace(Xtr.T @ Xtr) / max(1, Xtr.shape[1]))
            resid_tr = np.log(np.maximum(Ttr, 1e-12)) - Lpool_tr
            xm = Xtr.mean(axis=0) if center else np.zeros(Xtr.shape[1])
            rm = resid_tr.mean(axis=0) if center else np.zeros(resid_tr.shape[1])
            Xc = Xtr - xm
            A = Xc.T @ Xc
            Rm = resid_tr - rm
            Xtc = Xte - xm
            for lam in LAMBDAS:
                Aw = A + lam * scale * np.eye(A.shape[0])
                W = np.linalg.solve(Aw, Xc.T @ Rm).T
                pred = Lpool_te + (Xtc @ W.T) + rm
                ridge_by_layer[lam][L] = per_token_kl(Tte, pred)

        pooled_all = np.concatenate([pooled_by_layer[L] for L in range(n_layers)])
        best = None
        for lam in LAMBDAS:
            ridge_all = np.concatenate([ridge_by_layer[lam][L] for L in range(n_layers)])
            got = dict(lam_rel=float(lam), kl_ridge=float(ridge_all.mean()), kl_pooled=float(pooled_all.mean()))
            if best is None or got["kl_ridge"] < best["kl_ridge"]:
                best = got
                best_pts = ridge_all
        best.update(heldout_doc=int(held), tokens=int(test.sum()), kl_delta=best["kl_ridge"] - best["kl_pooled"])
        folds.append(best)
        all_diffs.append(best_pts - pooled_all)
        log("  held-out doc %d (%d tokens): pooled KL %.4f, ridge KL %.4f (lambda %g), delta %+.4f"
            % (best["heldout_doc"], best["tokens"], best["kl_pooled"], best["kl_ridge"], best["lam_rel"],
               best["kl_delta"]))

    from tools.nll import bootstrap_ci

    d = np.concatenate(all_diffs) if all_diffs else np.zeros(0)
    lo, hi = bootstrap_ci(d, block=8, iters=2000, seed=7) if d.size else (float("nan"), float("nan"))
    return dict(folds=folds, tokens=int(d.size), mean_delta=float(d.mean()) if d.size else float("nan"),
                ci_lo=lo, ci_hi=hi, resolved=bool(d.size and (lo > 0 or hi < 0)),
                kl_pooled=float(np.mean([r["kl_pooled"] for r in folds])),
                kl_ridge=float(np.mean([r["kl_ridge"] for r in folds])))


def emit_rows(routes, gids, n_groups, lam_rel=0.1, center=True):
    """Fit the rows on the whole log at the chosen lambda, for writing into a merged pack.

    Held-out folds exist to price the estimator; the rows a pack would carry are the ones fitted on everything, since
    a deployment has the whole log and no held-out document. The lambda is the one the folds picked, carried over
    rather than re-selected in sample, because re-selecting it here would fit the choice to the fit.
    """
    rl = np.asarray(routes["route_logits"], dtype=np.float64)
    xn = routes["xn"]
    mass = _mass(routes)
    rows, xm_all, rm_all = [], [], []
    for L in range(rl.shape[0]):
        X = np.asarray(xn[L], dtype=np.float64)
        T = grouped_targets(rl[L], gids, n_groups)
        Lpool = pooled_logits(rl[L], gids, n_groups, mass=mass)
        scale = float(np.trace(X.T @ X) / max(1, X.shape[1]))
        resid = np.log(np.maximum(T, 1e-12)) - Lpool
        if center:
            W, xm, rm = fit_ridge(X, resid, lam_rel * scale)
            rows.append(W.astype(np.float32))
            xm_all.append(xm.astype(np.float32))
            rm_all.append(rm.astype(np.float32))
        else:
            # no intercept: a router is `h @ W.T`, so a bias cannot be written, and `extra` is the difference the
            # pack's own pooled rows need added to them (rows_new = rows_pooled + extra)
            A = X.T @ X
            A[np.diag_indices_from(A)] += lam_rel * scale
            rows.append(np.linalg.solve(A, X.T @ resid).T.astype(np.float32))
    if center:
        return dict(rows=np.stack(rows), xm=np.stack(xm_all), rm=np.stack(rm_all), gids=np.asarray(gids, np.int64),
                    lam_rel=float(lam_rel))
    return dict(extra=np.stack(rows), gids=np.asarray(gids, np.int64), lam_rel=float(lam_rel))


def main(argv):
    if "--selftest" in argv:
        return selftest()

    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    path = arg("--routes")
    if not path:
        print("usage: python -m tools.expert_merge.routerfit --routes routes.npz [--groups 64] [--json out.json]")
        return 2
    n_groups = int(arg("--groups", 64))

    from tools.expert_merge.routes import load
    from tools.expert_merge.mergepack import groups
    from tools.bitnet_gemm.cluster_budget import coactivation

    routes = load(path)
    ids = np.asarray(routes["route_ids"])
    wts = np.asarray(routes["route_wts"])
    meta = routes.get("meta") or {}
    n_experts = int(meta.get("experts") or ids.max() + 1)
    k = int(meta.get("active") or ids.shape[-1])
    coc = coactivation(ids, wts, slice(None), n_experts, k=k)
    gids = groups(coc, n_groups)
    print("  %s: %d layers, %d tokens, %d experts -> %d groups"
          % (path, ids.shape[0], ids.shape[1], n_experts, n_groups))

    center = "--no-intercept" not in argv
    if not center:
        print("  bias-free fit: no intercept, so the emitted rows are the difference the pack's pooled rows add")
    emit = arg("--emit")
    if emit and "--emit-only" in argv:
        lam_rel = float(arg("--lambda", 0.1))
        pack = emit_rows(routes, gids, n_groups, lam_rel, center=center)
        np.savez_compressed(emit, **pack)
        print("  emitted %s: %s %s at lambda %g, fit on all %d tokens"
              % (emit, "rows" if "rows" in pack else "extra", pack.get("rows", pack.get("extra")).shape, lam_rel,
                 ids.shape[1]))
        return 0
    got = evaluate(routes, gids, n_groups, center=center)
    got.update(dict(routes=path, layers=int(ids.shape[0]), experts=n_experts, groups=n_groups,
                    corpus_tokens=int(ids.shape[1]), docs=meta.get("docs"), lambdas=list(LAMBDAS), center=center))
    print("  %d paired per-token differences: ridge - pooled KL = %+.4f [%+.4f, %+.4f] %s"
          % (got["tokens"], got["mean_delta"], got["ci_lo"], got["ci_hi"],
             "resolved" if got["resolved"] else "**covers zero**"))
    out = arg("--json")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(got, f, indent=1, sort_keys=True)
        print("  wrote %s" % out)

    emit = arg("--emit")
    if emit:
        lam_rel = float(arg("--lambda", 0.1))
        pack = emit_rows(routes, gids, n_groups, lam_rel, center=center)
        np.savez_compressed(emit, **pack)
        with open(emit.rsplit(".", 1)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(dict(routes=path, groups=n_groups, experts=n_experts, layers=int(ids.shape[0]),
                           corpus_tokens=int(ids.shape[1]), lambda_rel=lam_rel, center=center,
                           note=("rows: predict ((x - xm) @ W.T) + rm, replacing the pack's rows" if center else
                                 "extra: add to the pack's pooled rows, rows_new = rows_pooled + extra")), f, indent=1)
        shaped = pack.get("rows", pack.get("extra"))
        print("  emitted %s: %s %s at lambda %g (fit on all %d tokens) - %s"
              % (emit, "rows" if "rows" in pack else "extra", shaped.shape, lam_rel, ids.shape[1],
                 "not yet written into any pack" if center else
                 "added in memory by moe.Qwen3Moe(router_extra=...), not written to a pack"))
    return 0


def selftest():
    """Mechanics, not a scientific result: the anchored solve, the fold isolation, the pairing, the kill control.

    The synthetic teacher is linear in `x_n` with identical rows inside each group, which makes the grouped target
    exactly the pooled one. That is deliberate: it is the case where a re-fit has *no* headroom, so the selftest can
    assert the tool does not manufacture an improvement - the failure mode that would make every number it prints
    worthless. The anchored estimator makes that a hard property rather than a hope: at large lambda it *is* the
    pooled router.
    """
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    rs = np.random.RandomState(0)
    n_layers, per_doc, docs, E, d, G = 2, 40, 4, 8, 8, 4
    gids = np.repeat(np.arange(G), E // G)
    X = rs.randn(n_layers, per_doc * docs, d).astype(np.float32)
    Wg = rs.randn(n_layers, G, d) * 0.4
    W_true = Wg[:, gids, :]                       # rows inside a group are the same row
    logits = np.einsum("lnd,led->lne", X.astype(np.float64), W_true)
    doc = np.repeat(np.arange(docs), per_doc).astype(np.int16)
    routes = dict(route_logits=logits.astype(np.float16), xn=X,
                  route_ids=np.argsort(-logits, axis=2)[:, :, :2].astype(np.int16),
                  route_wts=np.full((n_layers, per_doc * docs, 2), 0.5, np.float16), doc=doc)

    # the solve: the anchored fit at high lambda is the pooled router, and the residual fit is exact for a linear target
    L0 = np.asarray(routes["route_logits"], dtype=np.float64)[0]
    X0 = np.asarray(X[0], dtype=np.float64)
    T0 = grouped_targets(L0, gids, G)
    P0 = pooled_logits(L0, gids, G)
    W, xm, rm = fit_ridge(X0, np.log(np.maximum(T0, 1e-12)) - P0, 1e-6 * float(np.trace(X0.T @ X0) / d))
    in_sample = kl(T0, P0 + predict(W, xm, rm, X0))
    check("routerfit: the anchored fit reproduces a linear teacher in sample", in_sample < 1e-2,
          "in-sample KL %.2e" % in_sample)
    check("routerfit: an all-zero residual fit leaves the pooled router untouched",
          np.allclose(predict(np.zeros_like(W), 0.0, 0.0, X0), 0.0), "zero map")

    got = evaluate(routes, gids, G, log=lambda *a: None)
    check("routerfit: no headroom is reported as no headroom (pooled is already the target)",
          abs(got["mean_delta"]) < 0.02, "delta %+.4f [%+.4f, %+.4f]" % (got["mean_delta"], got["ci_lo"],
                                                                        got["ci_hi"]))
    check("routerfit: folds never fit on the document they score",
          sorted(r["heldout_doc"] for r in got["folds"]) == list(range(docs))
          and got["tokens"] == per_doc * docs * n_layers,
          "%d paired tokens for %d folds" % (got["tokens"], len(got["folds"])))

    # kill property: fit to another token's routing must not look like a fit to this one
    shifted = dict(routes)
    shifted["route_logits"] = np.roll(np.asarray(routes["route_logits"], dtype=np.float64), 1, axis=1).astype(np.float16)
    got2 = evaluate(shifted, gids, G, log=lambda *a: None)
    check("routerfit: fitting the wrong tokens' routing does not beat pooling",
          got2["mean_delta"] >= -0.02,
          "shifted ridge %.4f pooled %.4f" % (got2["kl_ridge"], got2["kl_pooled"]))

    ok = sum(1 for _, c, _ in checks if c)
    for name, passed, detail in checks:
        print("  %s %s%s" % ("ok  " if passed else "FAIL", name, ("  (%s)" % detail) if detail else ""))
    print("  %d/%d checks passed" % (ok, len(checks)))
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
