"""Which experts to drop, under an exact per-layer byte budget, solved - and then gated on NLL.

The question is the other half of `cluster_budget.py`. That one asks whether a *paging* schedule can afford the
experts a layer routes to; this one asks which experts a layer can simply not have. Both start from the same
captured router log (`tools.expert_merge.routes`), and both need an answer that is a choice under a budget rather
than a ranking, because an expert a layer keeps costs its full byte count and the budget is hard.

The value of an expert, as far as the log can say, is the **routing mass** it was applied with: the sum, over every
token in the log, of the renormalised top-k weight the router multiplied that expert's output by. Dropping an
expert hands that mass to the experts that remain, so mass is exactly the quantity the model loses at that layer -
and nothing in the log holds the rest of the quantity (how much the remaining experts disagree with the one that
was dropped), which is why this tool does not pretend the inventory is a loss estimate. The loss is measured
downstream: `--gate` runs teacher-forced NLL on the same tokens through the unpruned and the pruned router and
subtracts, token by token.

The solve is `kindi.rco.plan`, the same multiple-choice knapsack the pack converter uses to decide which tensors
stay f32: one choice per expert, `drop` (0 bytes, loses its mass) or `keep` (its bytes, loses nothing), total
within the budget, minimise the mass lost. That is an exact fit, not a metaphor - RCO's problem is "one format per
item, sum of bytes under a budget, minimise summed error", and pruning is that with two formats, one of which is
free. `weights=1` in the entries because RCO multiplies a per-weight error by the weight count to get a total, and
the quantity here is already a per-expert total.

Where the DP earns its keep, and where it does not, is stated by the tool rather than glossed: with equal-size
experts the optimum is the top-m by mass and the DP reproduces the sort exactly (the selftest asserts the set
equality); with per-expert sizes that differ - a pack whose layers were quantised to different types, a budget that
lands between multiples of the expert size - the sort is no longer optimal and the DP is better. Both cases are
tested; `solve()` reports which one it is on.

    python -m tools.expert_merge.prune --routes routes.npz --keep 96                 # 96 of 128 a layer
    python -m tools.expert_merge.prune --routes routes.npz --pack MODEL.gguf --frac 0.75
    python -m tools.expert_merge.prune --routes routes.npz --pack MODEL.gguf --frac 0.75 --gate --docs 2
    python -m tools.expert_merge.prune --selftest

`--keep N` prices every expert at one byte, so the budget is a count and no pack is needed. `--frac F` prices each
expert from the pack's own tensor table, so the budget is bytes and the answer is what fits.

The gate: pruning is applied as a mask on the router - the kept ids keep their logits, the rest go to -inf and the
top-k renormalisation runs as it always does - so the two arms differ only in which experts exist, with no
re-quantisation and no second routing rule. The result is a plan plus a per-token pairing; `--max-delta` is the
tolerance in nats/token, the mean difference decides, and a moving-block bootstrap interval over the same tokens
says how much of it could be noise. A pass whose interval straddles the tolerance is reported as marginal rather
than accepted on the point estimate.
"""
import json
import os
import sys

import numpy as np

from kindi import rco

NLL_TOL = 0.02          # nats/token: the default a plan has to stay inside
GATE_BLOCK = 8          # tokens a bootstrap resample moves together


# --- the statistics the log carries ---------------------------------------------------------------------------

def mass_per_expert(routes, n_expert=None):
    """[layers, experts] routing mass, and [layers, experts] how often each was selected.

    The mass is the sum of the renormalised top-k weight every selection was applied with, so a layer's row sums to
    the number of tokens in the log; `count` is the number of selections, which is a different shape of the same
    fact (8 slots a token) and is reported because an expert with high mass in one slot and low count elsewhere is
    a different bet from one that is simply routed to by everything.
    """
    wts = routes["route_wts"].astype(np.float64)
    ids = routes["route_ids"].astype(np.int64)
    L, T, K = wts.shape
    n = int(n_expert or ids.max() + 1)
    mass = np.zeros((L, n), dtype=np.float64)
    count = np.zeros((L, n), dtype=np.int64)
    for l in range(L):
        for s in range(K):
            np.add.at(mass[l], ids[l, :, s], wts[l, :, s])
            np.add.at(count[l], ids[l, :, s], 1)
    return mass, count


def expert_bytes(pack, n_layer=None):
    """[layers, experts] bytes one expert's three matrices occupy in this pack, from its tensor table.

    Read, not assumed: the gate/up/down stacks of a Qwen3-MoE layer need not share a quantisation type (this pack
    is Q2_K for gate and up and Q3_K for down), and a per-expert byte count is the divisor every budget question
    starts with. The stacks are `[in, out, n_expert]`, so a tensor's span divides by its expert count exactly.
    """
    layers = int(n_layer or pack.n_layer)
    out = np.zeros((layers, pack.n_expert), dtype=np.int64)
    for il in range(layers):
        s = "blk.%d." % il
        per = 0
        for nm in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            ne, ty, _off = pack.info[s + nm]
            n_expert = int(ne[2])
            span = pack._span(ty, int(np.prod(ne)))
            if span % n_expert:
                raise SystemExit("prune: %s spans %d bytes, not a whole number of %d experts"
                                 % (s + nm, span, n_expert))
            per += span // n_expert
        out[il] = per
    return out


# --- the solve ------------------------------------------------------------------------------------------------

def plan_layer(mass, sizes, budget, unit=1024, max_cells=40 << 20):
    """One layer: choose which experts to keep so the kept bytes fit `budget` and the dropped mass is minimal.

    `mass` is one layer's row from `mass_per_expert`; `sizes` its per-expert byte counts. `kindi.rco.plan` is asked
    for a two-format choice a expert - `drop` at 0 bytes with the expert's mass as its error, `keep` at its bytes
    with no error - which is the pruning problem written as RCO's.
    """
    mass = np.asarray(mass, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.int64)
    entries = [dict(name=str(e), weights=1.0, err={"drop": float(mass[e]), "keep": 0.0},
                    bytes={"drop": 0, "keep": int(sizes[e])})
               for e in range(mass.size)]
    p = rco.plan(entries, int(budget), formats=("drop", "keep"), unit=int(unit), max_cells=int(max_cells))
    keep = sorted(int(n) for n, f in p["plan"].items() if f == "keep")
    return dict(keep=keep, bytes=int(p["bytes"]), budget=int(p["budget"]), dropped_mass=float(p["error"]),
                method=p["method"], gap=float(p["gap"]))


def solve(routes, per_expert, frac=None, keep=None, unit=1024, min_keep=1, log=None):
    """Every layer, solved on its own budget: `frac` of the pack's expert bytes, or exactly `keep` experts.

    The budget is per layer and exact - each layer may spend its allowance and no more - so a layer that needs
    more experts than its allowance buys does not borrow from a layer that needs fewer. That is the constraint a
    paged or a size-capped serving layout actually has (layer L's experts are its own tensors), and it is why the
    answer is 48 independent knapsacks rather than one big one.
    """
    mass, _count = mass_per_expert(routes, per_expert.shape[1])
    L = mass.shape[0]
    if L != per_expert.shape[0]:
        raise SystemExit("prune: the log has %d layers and the pack prices %d" % (L, per_expert.shape[0]))
    if keep is None and frac is None:
        raise SystemExit("prune: give --keep N or --frac F")
    out = []
    for l in range(L):
        sizes = np.asarray(per_expert[l], dtype=np.int64)
        if keep is None:
            budget = int(frac * int(sizes.sum()))
        else:
            budget = int(keep)
        p = plan_layer(mass[l], sizes, budget, unit=unit)
        if len(p["keep"]) < int(min_keep):
            raise SystemExit("prune: layer %d: a budget of %d bytes keeps %d experts, below the floor of %d "
                             "(%d bytes an expert)" % (l, budget, len(p["keep"]), int(min_keep), int(sizes[0])))
        p["layer"] = l
        p["total_mass"] = float(mass[l].sum())
        out.append(p)
    if log:
        log("  solved %d layers: %s" % (L, ", ".join(sorted({"%s%s" % (p["method"], "" if p["gap"] == 0 else
                                                                       " (gap %.6g)" % p["gap"]) for p in out}))))
    return out


def summary(plans, per_expert, active=None, priced="bytes", log=print):
    """One block an operator can act on: what each layer kept, and what that costs in the pack's own bytes.

    The bytes come from the keeping ids and the pack's price list, not from the solver's `bytes` field: under `--keep`
    the solve priced an expert at one byte, so its spend is a count. Both are exact; only one of them is a size, and
    saying "GB" about a count is how a plan for 64 experts reads as a plan for 64 bytes.
    """
    kept = np.array([len(p["keep"]) for p in plans])
    spent = sum(int(per_expert[l][p["keep"]].sum()) for l, p in enumerate(plans))
    total = int(per_expert.sum())
    dropped = [1.0 - (p["total_mass"] - p["dropped_mass"]) / p["total_mass"] for p in plans]
    if priced == "count":
        log("  %d layers, %d experts a layer, priced as a count (no pack given)"
            % (len(plans), int(per_expert.shape[1])))
        log("  keep %d-%d of %d a layer: %d of %d expert slots (%.1f%%), dropped routing mass %.1f%%"
            % (kept.min(), kept.max(), int(per_expert.shape[1]), spent, total, 100.0 * spent / total,
               100.0 * float(np.mean(dropped))))
    else:
        log("  %d layers, %d experts a layer, %s bytes an expert"
            % (len(plans), int(per_expert.shape[1]), "{:,}".format(int(per_expert[0, 0]))))
        log("  keep %d-%d of %d a layer: %.2f GB of %.2f GB expert weights (%.1f%%), dropped routing mass %.1f%%"
            % (kept.min(), kept.max(), int(per_expert.shape[1]), spent / 2 ** 30, total / 2 ** 30,
               100.0 * spent / total, 100.0 * float(np.mean(dropped))))
    if active and int(kept.min()) < int(active):
        log("  %d layers keep fewer than the model's %d active experts, so they route to %d slots there"
            % (int((kept < int(active)).sum()), int(active), int(kept.min())))
    return {"kept": int(kept.sum()), "layers": len(plans), "pricing": priced,
            "bytes_kept" if priced == "bytes" else "slots_kept": int(spent),
            "bytes_all" if priced == "bytes" else "slots_all": int(total),
            "dropped_mass": float(np.mean(dropped)), "min_kept": int(kept.min()), "max_kept": int(kept.max())}


def top_m_by_mass(mass, m):
    """The rule the DP has to beat: keep the `m` experts with the most routing mass."""
    return sorted(int(i) for i in np.argsort(-np.asarray(mass, dtype=np.float64))[:m])


def as_plan(plans, pricing="bytes"):
    """The artifact: layer -> kept ids, plus what the solve spent. Small enough to commit, exact enough to apply.

    `pricing` says whether `bytes_per_layer`/`budget_per_layer` are bytes (`--frac`, priced from the pack) or expert
    counts (`--keep`, every expert at one byte). Both are exact; reading one as the other is a factor of 1.7 million.
    """
    return {"pricing": pricing, "keep_per_layer": [p["keep"] for p in plans],
            "bytes_per_layer": [p["bytes"] for p in plans],
            "budget_per_layer": [p["budget"] for p in plans],
            "dropped_mass_per_layer": [p["dropped_mass"] for p in plans],
            "method": sorted({p["method"] for p in plans}),
            "gap": float(max(p["gap"] for p in plans))}


# --- the gate: NLL on the same tokens, with and without the plan ----------------------------------------------

def nll_moe(model, ids, log=None):
    """Teacher-forced NLL of one sequence in one forward, and the per-token contributions.

    `forward` already returns the distribution at every position, so scoring token i is the log-softmax at i-1 for
    it: no head re-run, no windowing, and the pairing the gate needs is free - the same token index in both arms.
    """
    got = model.forward(np.asarray(ids, dtype=np.int64), log=log)
    if got.get("logits") is None:
        raise SystemExit("prune: the pack does not decode every layer, so it has no next-token distribution to gate")
    z = np.asarray(got["logits"], dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    lp = z - (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))
    targets = np.asarray(ids[1:], dtype=np.int64)
    per = -lp[:-1][np.arange(targets.size), targets]
    return dict(nll=float(per.mean()), tokens=int(per.size), per_token=per)


def gate(baseline, plans, max_delta=NLL_TOL, block=GATE_BLOCK, iters=2000, seed=7):
    """Verdict per plan: the mean per-token NLL change, its moving-block interval, and whether it fits the tolerance.

    Accept on the point estimate and say when the interval does not resolve it: a plan whose interval straddles the
    tolerance is `marginal`, not accepted because it happened to land under. The interval is the one thing that
    turns "the number went up a little" into a gate, because 190 tokens drawn from two documents are not 190
    independent observations.
    """
    from tools.nll import bootstrap_ci

    a = np.asarray(baseline["per_token"], dtype=np.float64)
    rows = []
    for p in plans:
        b = np.asarray(p["per_token"], dtype=np.float64)
        if b.size != a.size:
            raise SystemExit("prune: the two arms scored different tokens (%d vs %d)" % (b.size, a.size))
        d = b - a
        lo, hi = bootstrap_ci(d, block=block, iters=iters, seed=seed)
        verdict = dict(delta=float(d.mean()), ci_lo=lo, ci_hi=hi, tokens=int(d.size), max_delta=float(max_delta),
                       passed=bool(d.mean() <= max_delta),
                       marginal=bool(d.mean() <= max_delta and hi > max_delta))
        rows.append(verdict)
    return rows


# --- the CLI --------------------------------------------------------------------------------------------------

def _plans_from_spec(routes, per_expert, argv):
    """every `--keep` / `--frac` value becomes its own plan, so one process can gate a curve of them"""
    keeps = float_list(argv, "--keep", int)
    fracs = float_list(argv, "--frac", float)
    if not keeps and not fracs:
        raise SystemExit("prune: give --keep N[,N...] or --frac F[,F...] (or --selftest)")
    unit = int(_arg(argv, "--unit", 1024))
    plans = []
    for k in keeps:
        # one byte an expert: the budget is a count, and the solve is the same solve
        plans.append(("keep=%d" % k, solve(routes, np.ones_like(per_expert), keep=k, unit=1), "count"))
    for f in fracs:
        plans.append(("frac=%g" % f, solve(routes, per_expert, frac=f, unit=unit), "bytes"))
    return plans


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def float_list(argv, name, cast):
    raw = _arg(argv, name)
    if raw is None:
        return []
    return [cast(x) for x in str(raw).split(",") if x.strip()]


def main(argv):
    if "--selftest" in argv:
        selftest()
        print("checks passed")
        return 0

    path = _arg(argv, "--routes")
    if not path:
        print("usage: python -m tools.expert_merge.prune --routes ROUTES.npz --keep N[,N...] "
              "| --frac F[,F...] [--pack MODEL.gguf] [--json out.json] [--gate --docs 2 --tokens 96 "
              "--max-delta 0.02] [--selftest]")
        return 2
    from .routes import load

    got = load(path)
    meta = got.get("meta", {})
    n_expert = int(meta.get("experts", got["route_ids"].max() + 1))
    n_layer = int(got["route_wts"].shape[0])
    print("  %s: %d layers, %d tokens, %d experts, top-%d"
          % (os.path.basename(path), n_layer, got["route_wts"].shape[1], n_expert,
             got["route_wts"].shape[2]))

    pack_path = _arg(argv, "--pack")
    pack = None
    if pack_path:
        from .ggufmoe import Moepack

        pack = Moepack(pack_path)
        per_expert = expert_bytes(pack, n_layer=n_layer)
    else:
        # no pack, no bytes: one byte an expert, so a budget is a count and the solve is still exact
        per_expert = np.ones((n_layer, n_expert), dtype=np.int64)

    active = int(meta.get("active", 0)) or None
    named = _plans_from_spec(got, per_expert, argv)
    results = []
    for name, plans, pricing in named:
        print(" %s" % name)
        tot = summary(plans, per_expert, active=active, priced=pricing)
        results.append((name, plans, tot, pricing))

    model = None
    docs = []
    if "--gate" in argv:
        from .moe import Qwen3Moe, tokenizer
        from .routes import corpus

        if not pack:
            raise SystemExit("prune: --gate needs --pack: the NLL has to come from the model the plan would prune")
        # `--corpus` is the difference between a gate and a fit. The default is the log's own documents, which is
        # the right choice for a smoke check and the wrong one for a claim: the plan was chosen on those tokens, so
        # an NLL measured on them cannot fall much. Held-out text is one flag away, and the tool says which it used.
        held_out = _arg(argv, "--corpus")
        per_doc = int(_arg(argv, "--tokens", 96))
        ndocs = int(_arg(argv, "--docs", 2))
        tok = tokenizer(pack)
        if held_out:
            names = [n for n in held_out.split(",") if n.strip()]
            fit = set(d[0] for d in meta.get("docs", []))
            overlap = sorted(fit.intersection(names))
            print("  gate corpus: %s (held out)%s" % (", ".join(names),
                                                      "" if not overlap else
                                                      "; NOT held out: %s also fed the plan" % ", ".join(overlap)))
        else:
            names = [d[0] for d in meta.get("docs", [])] or None
            print("  gate corpus: the log's own documents (in-sample - the plan was chosen on these tokens)")
        docs = corpus(tok, names, per_doc=per_doc)[:ndocs] if names else corpus(
            tok, ("docs/CPU-PERF-2026.md", "tools/kl.py"), per_doc=per_doc)
        if not docs:
            raise SystemExit("prune: no corpus to gate on (the log's own documents are not in this tree)")
        print("    %s" % ", ".join("%s(%d tokens)" % (n, len(i)) for n, i in docs))
        model = Qwen3Moe(pack_path, log=lambda *a: None)
        model.set_prune(None)
        per_token, scored = [], 0
        for name, ids in docs:
            r = nll_moe(model, ids)
            per_token.append(r["per_token"])
            scored += r["tokens"]
            print("    %-28s nll %.4f over %d tokens" % (name, r["nll"], r["tokens"]))
        baseline = dict(per_token=np.concatenate(per_token), nll=float(np.concatenate(per_token).mean()),
                        tokens=scored)
        print("    baseline nll %.4f over %d tokens" % (baseline["nll"], baseline["tokens"]))

    tol = float(_arg(argv, "--max-delta", NLL_TOL))
    accepted = []
    payload = {"routes": path, "pack": pack_path, "n_layer": n_layer, "n_expert": n_expert,
               "tokens_logged": int(got["route_wts"].shape[1]), "plans": []}
    for name, plans, tot, pricing in results:
        plan = as_plan(plans, pricing=pricing)
        row = dict(spec=name, summary=tot, **plan)
        if model is not None:
            model.set_prune(plan["keep_per_layer"])
            per_token, scored = [], 0
            for dname, ids in docs:
                r = nll_moe(model, ids)
                per_token.append(r["per_token"])
                scored += r["tokens"]
            model.set_prune(None)
            v = gate(dict(per_token=baseline["per_token"]), [dict(per_token=np.concatenate(per_token))],
                     max_delta=tol)[0]
            row["gate"] = dict(v, nll=float(np.concatenate(per_token).mean()),
                               baseline_nll=baseline["nll"], max_delta=tol, docs=[[n, len(i)] for n, i in docs])
            if v["passed"]:
                accepted.append(name)
            verdict = "REJECTED"
            if v["passed"]:
                verdict = "ACCEPTED" if not v["marginal"] else \
                    "ACCEPTED, but the interval reaches %+.4f, so the tolerance is not excluded" % v["ci_hi"]
            print("    nll %.4f -> %.4f (%+.4f, 95%% CI %+.4f..%+.4f) %s (tolerance %+.3f)"
                  % (baseline["nll"], row["gate"]["nll"], v["delta"], v["ci_lo"], v["ci_hi"], verdict, tol))
        payload["plans"].append(row)

    out = _arg(argv, "--json")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)   # utf-8, no NaN: a plan has to be machine-readable
        print("  wrote %s" % out)
    if "--gate" in argv:
        if not accepted:
            print("prune: no plan stays inside %.3f nats/token; nothing to apply" % tol)
            return 1
        print("prune: accepted %s" % ", ".join(accepted))
    return 0


# --- the checks -----------------------------------------------------------------------------------------------

def selftest():
    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    rng = np.random.RandomState(11)

    # --- the statistics: a log whose answer is known
    ids = np.array([[[0, 1], [1, 2], [0, 2]]], dtype=np.int16)          # 1 layer, 3 tokens, top-2
    wts = np.array([[[0.5, 0.5], [0.25, 0.75], [0.4, 0.6]]], dtype=np.float16)
    m, c = mass_per_expert(dict(route_ids=ids, route_wts=wts), n_expert=3)
    check("prune: mass is the weight each expert was applied with",
          np.allclose(m[0], [0.9, 0.75, 1.35], atol=1e-3), str(m[0]))   # the log stores weights in fp16
    check("prune: counts are selections, not mass", list(c[0]) == [2, 2, 2], str(c[0]))
    check("prune: a layer's mass sums to its token count", abs(m[0].sum() - 3.0) < 1e-9, str(m[0].sum()))

    # --- the count path: equal prices, so the exact answer is the mass-sorted one
    mass = rng.permutation(np.linspace(1.0, 128.0, 128))
    sizes = np.full(128, 1668 * 1024, dtype=np.int64)
    p = plan_layer(mass, sizes, budget=96 * 1668 * 1024)
    check("prune: an even-size budget keeps exactly the experts asked for", len(p["keep"]) == 96, str(len(p["keep"])))
    check("prune: the solve is the exact table, not the fallback", p["method"] == "dp" and p["gap"] == 0.0,
          "%s gap %s" % (p["method"], p["gap"]))
    check("prune: with even sizes it reproduces the mass sort exactly - the DP adds nothing there, and says so",
          set(p["keep"]) == set(top_m_by_mass(mass, 96)))
    check("prune: the budget is a ceiling, not a target to exceed",
          p["bytes"] <= p["budget"] and p["bytes"] == 96 * 1668 * 1024, "%d vs %d" % (p["bytes"], p["budget"]))
    check("prune: the dropped mass is the mass of the experts left out",
          abs(p["dropped_mass"] - mass.sum() + mass[p["keep"]].sum()) < 1e-9,
          "%.6g" % p["dropped_mass"])

    # --- a budget between multiples of the expert size: the sort is no longer optimal, and the DP must beat it
    # the heaviest expert costs 4 of 6 KiB, so 'heaviest first' takes it and then nothing else fits; two lighter
    # experts spend the budget exactly and carry more mass together
    mass2 = np.array([10.0, 8.0, 7.0])
    sizes2 = np.array([4, 3, 3], dtype=np.int64)                          # in KiB
    q = plan_layer(mass2, sizes2, budget=6, unit=1)
    greedy = _greedy_keeps(mass2, sizes2, 6)
    check("prune: it fits", q["bytes"] <= 6, "%d" % q["bytes"])
    check("prune: with unequal sizes it beats keeping the heaviest experts that fit",
          q["dropped_mass"] < mass2.sum() - mass2[greedy].sum(),
          "kept %s dropped %.3g, the sort drops %.3g"
          % (q["keep"], q["dropped_mass"], mass2.sum() - mass2[greedy].sum()))
    check("prune: and it is still exact on that grid", q["gap"] == 0.0 and q["method"] == "dp", q["method"])

    # --- feasibility across the whole range, and the degenerate budgets handled explicitly
    for frac in (0.0, 0.25, 0.5, 0.9, 1.0):
        p = plan_layer(mass, sizes, budget=int(frac * sizes.sum()))
        check("prune: a %.0f%% budget is not exceeded" % (100 * frac), p["bytes"] <= int(frac * sizes.sum()))
        check("prune: a %.0f%% budget is fully used when the experts are equal" % (100 * frac),
              p["bytes"] == int(frac * sizes.sum()) // sizes[0] * sizes[0])
    p1 = plan_layer(mass, sizes, budget=sizes.sum())
    check("prune: the whole budget keeps everything", len(p1["keep"]) == 128 and p1["dropped_mass"] == 0.0)

    # an expert the router never chose is free to drop, and goes before anything that carries mass
    mass3 = mass.copy()
    mass3[:8] = 0.0
    p2 = plan_layer(mass3, sizes, budget=120 * sizes[0])
    check("prune: experts the log never routed to are dropped first", set(range(8)) <= set(range(128)) - set(p2["keep"]),
          str(p2["keep"][:10]))
    mass4 = np.zeros(4)
    p3 = plan_layer(mass4, np.ones(4, dtype=np.int64), budget=2)
    check("prune: with no mass anywhere it keeps nothing - every byte would buy nothing",
          p3["keep"] == [] and p3["bytes"] == 0, str(p3))

    # --- the mask is a router with fewer options, not a second routing rule
    from .moe import mask_logits, masked_route, softmax

    logits = rng.randn(6, 128).astype(np.float32)
    idx, w = masked_route(logits, 8)
    ref = np.argsort(-softmax(logits), axis=1)[:, :8]
    check("prune: with no mask the route is the model's own", np.array_equal(np.sort(idx, axis=1), np.sort(ref, axis=1)))
    keep = list(range(0, 64, 2))
    midx, mw = masked_route(mask_logits(logits, keep), 8)
    check("prune: a masked route picks only from the kept set", set(midx.ravel().tolist()) <= set(keep),
          str(sorted(set(midx.ravel().tolist()))[:8]))
    check("prune: and renormalises as it always does", np.allclose(mw.sum(axis=1), 1.0), str(mw.sum(axis=1)))
    fidx, fw = masked_route(mask_logits(logits, keep), 128)
    check("prune: asking for more slots than exist clamps to the kept count",
          fidx.shape[1] == len(keep) and np.allclose(fw.sum(axis=1), 1.0), str(fidx.shape))
    check("prune: the mask leaves the kept logits bit-identical",
          np.array_equal(mask_logits(logits, keep)[:, keep], logits[:, keep]))
    check("prune: and the dropped ones are -inf, not a large negative number",
          np.isneginf(mask_logits(logits, keep)).sum() == logits.shape[0] * (128 - len(keep)))

    # --- the verdict, without a model: the point estimate decides, and a straddling interval is called marginal
    base = dict(per_token=np.zeros(64))
    near = gate(base, [dict(per_token=np.full(64, 0.019))], max_delta=0.02)[0]
    check("prune: a plan inside the tolerance passes", near["passed"] and not near["marginal"], str(near))
    far = gate(base, [dict(per_token=np.full(64, 0.05))], max_delta=0.02)[0]
    check("prune: a plan outside it fails", not far["passed"], str(far))
    noisy = gate(base, [dict(per_token=np.full(32, 0.0).tolist() + np.full(32, 0.04).tolist())], max_delta=0.02)[0]
    check("prune: an interval that straddles the tolerance is marginal, not accepted", noisy["marginal"],
          str(noisy))
    return 0


def _greedy_keeps(mass, sizes, budget):
    """who the 'heaviest first that fits' rule keeps, as ids - the baseline the DP has to improve on"""
    order = np.argsort(-np.asarray(mass, dtype=np.float64))
    used, kept = 0, []
    for i in order:
        if used + int(sizes[i]) <= budget:
            kept.append(int(i))
            used += int(sizes[i])
    return sorted(kept)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
