"""P13, priced before it is built: can a rank-r residual represent what absorption threw away?

Absorption replaces a cold expert with its centroid, and P11 measured what that costs: absorb-96 is
**+0.2269 [+0.1549, +0.2937]** worse than prune-96 at the same 96 slabs. RAVE's fourth state keeps the
loss as a rank-4-8 factor (`W_e = W_c + U S Vᵀ`) instead of pretending it is zero. Serving that needs
the factors read when the group runs - a change to `kindi/cpu.py` and `kcpu.c`, not a flag - so before
anyone pays for it this asks the prior question on real data:

  **on the activations expert `e` actually saw, how much of its difference from its centroid does a
  rank-r factor reproduce?**

That is a pre-gate, not a verdict. It cannot produce an NLL delta, and it must not be quoted as one.
What it can do is kill the direction cheaply: if a rank-8 factor recovers a small fraction of the
deviation, no arrangement of the same bytes will close 0.227 nats, and the C kernel work is not worth
starting. If it recovers most of it, the walk becomes worth its hour.

Three numbers per (layer, absorbed expert, rank), all on the route log the gate already uses:

  * `energy`   - the fraction of ‖W_e - W_c‖² a rank-r truncation keeps. Weight space, cheap, blind to
    which inputs matter.
  * `operator` - the same thing on the vectors the router actually sent that expert:
    ‖(W_e-W_c)x - D_r x‖ / ‖(W_e-W_c)x‖, averaged over its tokens. Linear, per matrix.
  * `function` - the end-to-end one. The expert's FFN is rebuilt as centroid + rank-r factor on gate,
    up and down, and `1 - ‖FFN_e(x) - FFN_r(x)‖ / ‖FFN_e(x) - FFN_c(x)‖` says how much of the output
    difference the residual puts back. This is the number the NLL gap is made of; it is still not an
    NLL measurement.

  python -m tools.expert_merge.residual --pack PACK.gguf --routes ROUTES.npz --keep 96 --ranks 2,4,8 --json OUT.json
  python -m tools.expert_merge.residual --selftest
"""
import json
import os
import sys

import numpy as np

from .mergepack import assign_absorb

RANKS = (0, 2, 4, 8)


def _arg(argv, name, default=None):
    if name in argv:
        return argv[argv.index(name) + 1]
    return default


def low_rank(d, rank, rng=None, oversample=10, iters=2):
    """the best rank-`rank` approximation of `d`, via a randomized range finder

    Exact SVD of a 768x2048 slab is ~1-2 s and this runs on 32 experts x 3 matrices x 48 layers, so    the sketch is what makes the whole question affordable: `d @ omega` finds the dominant subspace at
    O(m n r) instead of O(m n min(m,n)); two power iterations close the low-bias the first pass has on a
    flat spectrum. `selftest` checks the result against `np.linalg.svd` on a known matrix.
    """
    d = np.asarray(d, dtype=np.float32)
    if rank <= 0:
        return np.zeros_like(d)
    r = min(int(rank), min(d.shape) - 1)
    rng = rng or np.random.RandomState(0)
    om = rng.standard_normal((d.shape[1], r + oversample)).astype(np.float32)
    q, _ = np.linalg.qr(d @ om)
    # power iterations: one sketch pass is biased *low* on a flat spectrum (measured: 0.149 vs 0.178 of the
    # energy at rank 2), and an underestimate here would be the pre-gate's whole answer turning on the sampler
    for _ in range(int(iters)):
        q, _ = np.linalg.qr(d @ (d.T @ q))
    b = q.T @ d
    ub, s, vt = np.linalg.svd(b, full_matrices=False)
    u = q @ ub
    return ((u[:, :r] * s[:r]) @ vt[:r]).astype(np.float32)


def ffn(x, wg, wu, wd):
    """the block's own arithmetic: silu(gate x) * (up x), then down"""
    g = x @ wg.T
    u = x @ wu.T
    h = (g / (1.0 + np.exp(-g))) * u
    return h @ wd.T


def cooc(ids, n_expert):
    """how often two experts were chosen by the same token, over the route log's own decisions"""
    c = np.zeros((n_expert, n_expert), dtype=np.float64)
    for row in ids:
        for i in row:
            c[i, row] += 1.0
    return c


def measure_layer(pack, il, ids, wts, xn, keep, cap, ranks, seed=0, log=None):
    """one layer: assign, then ask each rank how much of each absorbed expert's deviation it keeps"""
    n_expert = pack.n_expert
    mass = np.zeros(n_expert, dtype=np.float64)
    np.add.at(mass, ids.reshape(-1), wts.reshape(-1))
    gids = assign_absorb(cooc(ids, n_expert) + cooc(ids, n_expert).T, keep, mass, cap=cap)
    members = {}
    for e in range(n_expert):
        members.setdefault(int(gids[e]), []).append(e)
    centroids = dict((g, max(mem, key=lambda e: (mass[e], -e))) for g, mem in members.items())
    # only the experts that were actually absorbed, and the centroids they were absorbed into
    absorbed = [(e, centroids[int(gids[e])]) for e in range(n_expert) if e != centroids[int(gids[e])]]
    if log:
        log("  layer %d: %d absorbed into %d stacks" % (il, len(absorbed), keep))
    s = "blk.%d." % il
    names = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
    need = sorted({e for pair in absorbed for e in pair})
    W = dict((e, [pack.expert_weight(s + nm, e) for nm in names]) for e in need)
    # the tokens each expert saw: its own routing decisions, at the weight it was given
    toks = dict((e, np.flatnonzero((ids == e).any(axis=1))) for e in need)
    rng = np.random.RandomState(seed + il)
    rows = []
    for e, c in absorbed:
        sel = toks[e]
        if sel.size < 2:                                  # never routed here: nothing to judge it on
            continue
        x = np.asarray(xn[sel], dtype=np.float32)
        wc, we = W[c], W[e]
        y_c, y_e = ffn(x, *wc), ffn(x, *we)
        base = np.linalg.norm(y_e - y_c)
        if base < 1e-9:
            continue
        for r in ranks:
            dr = [low_rank(we[j] - wc[j], r, rng) for j in range(3)]
            y_r = ffn(x, *[wc[j] + dr[j] for j in range(3)])
            rec = dict(layer=il, expert=int(e), centroid=int(c), rank=int(r),
                       tokens=int(sel.size),
                       energy=float(sum(np.linalg.norm(dr[j]) ** 2 for j in range(3)) /
                                    max(1e-12, sum(np.linalg.norm(we[j] - wc[j]) ** 2 for j in range(3)))),
                       function=float(1.0 - np.linalg.norm(y_e - y_r) / base))
            # Each matrix is judged on the vector it is actually applied to: gate and up see `x`, down sees the
            # hidden activation `silu(gate x) * (up x)`. Feeding `x` to `down` is a shape error first (768 against
            # 2048) and a wrong measurement second, so the inputs are built per matrix rather than assumed.
            h_e = (lambda g, u: (g / (1.0 + np.exp(-g))) * u)(x @ we[0].T, x @ we[1].T)
            inp = (x, x, h_e)
            op_n, op_d = 0.0, 0.0
            for j in range(3):
                d = we[j] - wc[j]
                xj = inp[j]
                op_n += float(np.sum((xj @ d.T - xj @ dr[j].T) ** 2))
                op_d += float(np.sum((xj @ d.T) ** 2))
            rec["operator"] = float(1.0 - np.sqrt(op_n / max(1e-12, op_d)))
            rows.append(rec)
    return rows


def summarise(rows, ranks, log=print):
    """per rank: the median over (layer, expert) with a 95% bootstrap interval, and the token count behind it

    The unit of resampling is the absorbed expert, not the token: two experts in one layer share the same
    centroid and the same corpus, so resampling tokens would claim a precision the design does not have.
    """
    out = {}
    for r in ranks:
        sub = [x for x in rows if x["rank"] == r]
        if not sub:
            continue
        vals = np.array([x["function"] for x in sub], dtype=np.float64)
        rng = np.random.RandomState(13)
        boot = np.array([np.median(vals[rng.randint(0, vals.size, vals.size)]) for _ in range(2000)])
        out[str(r)] = dict(n_experts=int(vals.size), tokens=int(sum(x["tokens"] for x in sub)),
                           function_median=float(np.median(vals)),
                           function_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           operator_median=float(np.median([x["operator"] for x in sub])),
                           energy_median=float(np.median([x["energy"] for x in sub])))
        if log:
            log("  rank %2d: function recovered %.3f [%.3f, %.3f] over %d absorbed experts; operator %.3f, "
                "energy %.3f" % (r, out[str(r)]["function_median"], out[str(r)]["function_ci"][0],
                                 out[str(r)]["function_ci"][1], vals.size, out[str(r)]["operator_median"],
                                 out[str(r)]["energy_median"]))
    return out


def main(argv):
    if "--selftest" in argv:
        selftest()
        print("checks passed")
        return 0
    from .ggufmoe import Moepack
    from . import routes as routelog
    pack_path, rpath = _arg(argv, "--pack"), _arg(argv, "--routes")
    if not pack_path or not rpath:
        raise SystemExit("usage: python -m tools.expert_merge.residual --pack PACK.gguf --routes ROUTES.npz "
                         "[--keep 96] [--cap 8] [--ranks 0,2,4,8] [--layers N] [--json OUT.json]")
    keep = int(_arg(argv, "--keep", 96))
    cap = int(_arg(argv, "--cap", 8))
    ranks = tuple(int(x) for x in str(_arg(argv, "--ranks", "0,2,4,8")).split(","))
    max_layers = int(_arg(argv, "--layers", 0)) or 10 ** 9
    pack = Moepack(pack_path)
    got = routelog.load(rpath)
    ids, wts, xn = got["route_ids"], got["route_wts"], got["xn"]
    print("residual pre-gate: %d layers, %d tokens, %d experts -> %d stacks (cap %d), ranks %s"
          % (min(ids.shape[0], max_layers), ids.shape[1], pack.n_expert, keep, cap, ",".join(map(str, ranks))))
    rows = []
    for il in range(min(pack.n_layer, max_layers)):
        rows += measure_layer(pack, il, ids[il], wts[il], xn[il], keep, cap, ranks, log=print)
    summary = summarise(rows, ranks)
    payload = dict(pack=pack_path, routes=rpath, keep=keep, cap=cap, layers=min(pack.n_layer, max_layers),
                   tokens=int(ids.shape[1]), summary=summary, rows=rows,
                   note="a pre-gate on activations, not an NLL walk: no delta here is comparable to a gated arm")
    out = _arg(argv, "--json")
    if out:
        with open(out, "w") as f:
            json.dump(payload, f, indent=1)
        print("wrote %s" % out)
    return 0


def selftest():
    """that the sketch is the exact truncation, and that 'function recovered' means what it says

    Two claims, both falsifiable here: `low_rank` matches `np.linalg.svd` on a matrix whose rank is known,
    and the recovery statistic is 1.0 when the residual has the deviation's rank and 0.0 when it has none.
    A statistic that reads high on a residual that restores nothing would be the whole pre-gate.
    """
    rng = np.random.RandomState(5)
    a = rng.randn(48, 96) @ rng.randn(96, 64)                     # rank 48 at most, in a 128x96 frame
    d = np.zeros((128, 96), dtype=np.float32)
    d[:48, :64] = a
    _, s, _ = np.linalg.svd(d, full_matrices=False)
    for r in (2, 4, 8):
        got = low_rank(d, r, np.random.RandomState(0))
        exact = float((s[:r] ** 2).sum() / (s ** 2).sum())
        mine = float(np.linalg.norm(got) ** 2 / np.linalg.norm(d) ** 2)
        assert abs(mine - exact) < 2e-3, "rank %d sketch energy %.4f vs exact %.4f" % (r, mine, exact)
    assert float(np.linalg.norm(low_rank(d, 48, np.random.RandomState(0)) - d) / np.linalg.norm(d)) < 1e-3, \
        "a rank that spans the matrix must reconstruct it"

    # function: expert = centroid + a deviation of known rank, so recovery is 1 at that rank and 0 at rank 0
    x = rng.randn(32, 96).astype(np.float32)
    # gate and up are [ff, d], down is [d, ff] - the fixture's shapes are the block's, not three equal ones
    shapes = ((64, 96), (64, 96), (96, 64))
    wc = [rng.randn(*s).astype(np.float32) * 0.1 for s in shapes]
    dev = [np.outer(rng.randn(s[0]), rng.randn(s[1])).astype(np.float32) for s in shapes]
    we = [wc[j] + dev[j] for j in range(3)]
    base = np.linalg.norm(ffn(x, *we) - ffn(x, *wc))
    y0 = ffn(x, *wc)
    assert np.linalg.norm(ffn(x, *we) - y0) > 1e-6, "the fixture's deviation does nothing"
    dr = [low_rank(we[j] - wc[j], 1, np.random.RandomState(1)) for j in range(3)]
    rec1 = 1.0 - np.linalg.norm(ffn(x, *we) - ffn(x, *[wc[j] + dr[j] for j in range(3)])) / base
    assert rec1 > 0.95, "a rank-1 residual did not restore a rank-1 deviation (%.3f)" % rec1
    rec0 = 1.0 - np.linalg.norm(ffn(x, *we) - ffn(x, *[wc[j] + low_rank(we[j] - wc[j], 0) for j in range(3)])) / base
    assert abs(rec0) < 1e-6, "rank 0 must recover nothing (%.3f)" % rec0

    # the assignment is the one the absorb pack used: 96 stacks, 32 absorbed, non-uniform sizes
    n = 128
    ids = np.stack([np.arange(8) + (t % 120) for t in range(200)]).reshape(200, 8)
    wts = np.full((200, 8), 1.0 / 8, dtype=np.float32)
    mass = np.zeros(n)
    np.add.at(mass, ids.reshape(-1), wts.reshape(-1))
    g = assign_absorb(cooc(ids, n) + cooc(ids, n).T, 96, mass, cap=8)
    assert len(set(g.tolist())) == 96, "the map did not land on 96 stacks"
    sizes = np.bincount(g)
    assert sizes.max() <= 8 and (sizes > 1).sum() >= 1, "group sizes %s" % sizes
    return "residual: sketch == exact truncation; rank-1 fixture recovered 1.000, rank 0 recovered 0.000"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
