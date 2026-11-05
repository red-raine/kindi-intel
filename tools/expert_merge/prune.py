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
    python -m tools.expert_merge.prune --routes routes.npz --keep 96 --protect-json superex.json --gate
    python -m tools.expert_merge.prune --routes routes.npz --pack MODEL.gguf --frac 0.75 --gate --docs 2
    python -m tools.expert_merge.prune --selftest

`--keep N` prices every expert at one byte, so the budget is a count and no pack is needed. `--frac F` prices each
expert from the pack's own tensor table, so the budget is bytes and the answer is what fits.

**`--protect` is a guard, not a preference.** Routing mass is the wrong currency for an expert that is not a small
linear map: `tools.expert_merge.superex` finds four experts that write 210-2,152x their own layer's median output while
carrying 0.28-1.4% of its routing mass, and the lowest of them ranks 92 of 128 by mass. A mass budget keeping 64 drops
it and reports the loss as a fraction of a percent of mass - correctly, and blind to the mechanism. So the flagged set
is forced into the keep set *before* the solve, priced in the same budget, and the tool says what the guard cost. If
the guard alone does not fit the budget the run stops: a protection that is silently dropped is the failure this flag
exists to prevent. Ids are `layer:expert`, 0-based, the same indexing `superex` and the plan JSON use.

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

def plan_layer(mass, sizes, budget, unit=1024, max_cells=40 << 20, protect=None):
    """One layer: choose which experts to keep so the kept bytes fit `budget` and the dropped mass is minimal.

    `mass` is one layer's row from `mass_per_expert`; `sizes` its per-expert byte counts. `kindi.rco.plan` is asked
    for a two-format choice a expert - `drop` at 0 bytes with the expert's mass as its error, `keep` at its bytes
    with no error - which is the pruning problem written as RCO's.

    `protect` is a set of expert ids this layer may not drop. They are taken out of the budget first and kept
    unconditionally, so the solve is the same knapsack over what is left, at a smaller budget - the guard spends
    bytes rather than overriding the price. That ordering is the point: with `protect` applied inside the DP's error
    term the result would depend on how the solver happened to break a tie.
    """
    mass = np.asarray(mass, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.int64)
    forced = sorted({int(e) for e in (protect or ())})
    if forced and (forced[0] < 0 or forced[-1] >= mass.size):
        raise SystemExit("prune: expert %d does not exist in a layer of %d" % (forced[-1], mass.size))
    forced_bytes = int(sizes[forced].sum()) if forced else 0
    if forced_bytes > int(budget):
        raise SystemExit("prune: the %d protected expert(s) cost %d, over the layer's budget of %d by %d - a "
                         "protection that does not fit is refused rather than dropped quietly"
                         % (len(forced), forced_bytes, int(budget), forced_bytes - int(budget)))
    rest = [e for e in range(mass.size) if e not in set(forced)]
    if not rest:
        return dict(keep=forced, bytes=forced_bytes, budget=int(budget), dropped_mass=0.0, method="protected",
                    gap=0.0, protected=forced, protected_bytes=forced_bytes)
    entries = [dict(name=str(e), weights=1.0, err={"drop": float(mass[e]), "keep": 0.0},
                    bytes={"drop": 0, "keep": int(sizes[e])})
               for e in rest]
    p = rco.plan(entries, int(budget) - forced_bytes, formats=("drop", "keep"), unit=int(unit),
                 max_cells=int(max_cells))
    keep = sorted(set(forced) | {int(n) for n, f in p["plan"].items() if f == "keep"})
    return dict(keep=keep, bytes=int(p["bytes"]) + forced_bytes, budget=int(budget),
                dropped_mass=float(p["error"]), method=p["method"], gap=float(p["gap"]),
                protected=forced, protected_bytes=forced_bytes)


def solve(routes, per_expert, frac=None, keep=None, unit=1024, min_keep=1, log=None, protect=None):
    """Every layer, solved on its own budget: `frac` of the pack's expert bytes, or exactly `keep` experts.

    The budget is per layer and exact - each layer may spend its allowance and no more - so a layer that needs
    more experts than its allowance buys does not borrow from a layer that needs fewer. That is the constraint a
    paged or a size-capped serving layout actually has (layer L's experts are its own tensors), and it is why the
    answer is 48 independent knapsacks rather than one big one.

    `protect` is `{layer: {expert, ...}}` - the guard from `superex`, forced into every layer's keep set.
    """
    mass, _count = mass_per_expert(routes, per_expert.shape[1])
    L = mass.shape[0]
    if L != per_expert.shape[0]:
        raise SystemExit("prune: the log has %d layers and the pack prices %d" % (L, per_expert.shape[0]))
    if keep is None and frac is None:
        raise SystemExit("prune: give --keep N or --frac F")
    protect = protect or {}
    bad = sorted(set(protect) - set(range(L)))
    if bad:
        raise SystemExit("prune: --protect names layer(s) %s and the log has %d" % (bad, L))
    out = []
    for l in range(L):
        sizes = np.asarray(per_expert[l], dtype=np.int64)
        if keep is None:
            budget = int(frac * int(sizes.sum()))
        else:
            budget = int(keep)
        p = plan_layer(mass[l], sizes, budget, unit=unit, protect=protect.get(l))
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


def summary(plans, per_expert, active=None, priced="bytes", sized=None, log=print):
    """One block an operator can act on: what each layer kept, and what that costs in the pack's own bytes.

    The bytes come from the keeping ids and the pack's price list, not from the solver's `bytes` field: under `--keep`
    the solve priced an expert at one byte, so its spend is a count. Both are exact; only one of them is a size, and
    saying "GB" about a count is how a plan for 64 experts reads as a plan for 64 bytes.
    """
    kept = np.array([len(p["keep"]) for p in plans])
    spent = sum(int(per_expert[l][p["keep"]].sum()) for l, p in enumerate(plans))
    total = int(per_expert.sum())
    dropped = [1.0 - (p["total_mass"] - p["dropped_mass"]) / p["total_mass"] for p in plans]
    if priced == "count" and not sized:
        log("  %d layers, %d experts a layer, priced as a count (no pack given)"
            % (len(plans), int(per_expert.shape[1])))
        log("  keep %d-%d of %d a layer: %d of %d expert slots (%.1f%%), dropped routing mass %.1f%%"
            % (kept.min(), kept.max(), int(per_expert.shape[1]), spent, total, 100.0 * spent / total,
               100.0 * float(np.mean(dropped))))
    elif priced == "count":
        # the solve spent a count, but the plan has a size: the pack was read, so report both rather than a count
        # labelled as a slot while the file it names is 208 MB a layer
        log("  %d layers, %d experts a layer, %s bytes an expert"
            % (len(plans), int(per_expert.shape[1]), "{:,}".format(int(per_expert[0, 0]))))
        log("  keep %d-%d of %d a layer: %.2f GB of %.2f GB expert weights (%.1f%%), dropped routing mass %.1f%%"
            % (kept.min(), kept.max(), int(per_expert.shape[1]), spent / 2 ** 30, total / 2 ** 30,
               100.0 * spent / total, 100.0 * float(np.mean(dropped))))
    else:
        log("  %d layers, %d experts a layer, %s bytes an expert"
            % (len(plans), int(per_expert.shape[1]), "{:,}".format(int(per_expert[0, 0]))))
        log("  keep %d-%d of %d a layer: %.2f GB of %.2f GB expert weights (%.1f%%), dropped routing mass %.1f%%"
            % (kept.min(), kept.max(), int(per_expert.shape[1]), spent / 2 ** 30, total / 2 ** 30,
               100.0 * spent / total, 100.0 * float(np.mean(dropped))))
    if active and int(kept.min()) < int(active):
        log("  %d layers keep fewer than the model's %d active experts, so they route to %d slots there"
            % (int((kept < int(active)).sum()), int(active), int(kept.min())))
    # both prices are recorded when both are known: `slots_*` is what the solve spent, `bytes_*` is what the plan
    # weighs in the pack. Under `--keep` they are a count and a size of the same plan - and a merged table that only
    # has the count reads "-" for the size, which is the wrong way round for a note about bytes.
    if sized is None:
        sized = priced == "bytes"
    out = {"kept": int(kept.sum()), "layers": len(plans), "pricing": priced,
           "dropped_mass": float(np.mean(dropped)), "min_kept": int(kept.min()), "max_kept": int(kept.max())}
    if sized:
        out.update({"bytes_kept": int(spent), "bytes_all": int(total)})
    if priced != "bytes":
        out.update({"slots_kept": int(spent), "slots_all": int(total)})
    return out


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
    per = _per_token(got["logits"], ids)
    return dict(nll=float(per.mean()), tokens=int(per.size), per_token=per)


def _per_token(logits, ids, rows=128):
    """Teacher-forced NLL a token: log-softmax of the distribution at i-1 read at token i.

    Only one number a row is needed - the target's log-probability - so the vocabulary is reduced a block of rows at
    a time. That matters once a gate batches its corpus: 1,800 tokens over a 152k vocabulary is 1.1 GB of logits, and
    promoting all of it to float64 for a log-sum-exp would double it for a difference no measurement can see.
    """
    targets = np.asarray(ids[1:], dtype=np.int64)
    n = int(targets.size)
    per = np.empty(n, dtype=np.float64)
    for a in range(0, n, int(rows)):
        b = min(a + int(rows), n)
        z = np.asarray(logits[a:b], dtype=np.float64)
        m = z.max(axis=1, keepdims=True)
        lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]
        per[a:b] = lse - z[np.arange(b - a), targets[a:b]]
    return per


def nll_moe_batch(model, docs, log=None):
    """The same per-token NLL for a list of `(name, ids)` documents in **one** forward.

    The gate used to walk the corpus a document at a time, which pays the dequantiser for a layer's experts once a
    document because that is the expensive half of the pass. Batched, the expert union is taken over the corpus and
    paid once; the tokens come back in the order they were given, so the arms stay paired token for token and the
    bootstrap in `gate` never sees the difference.
    """
    got = model.forward_docs([np.asarray(i, dtype=np.int64) for _, i in docs], log=log)
    if got["per_doc"][0].get("logits") is None:
        raise SystemExit("prune: the pack does not decode every layer, so it has no next-token distribution to gate")
    per_doc = [_per_token(g["logits"], i) for g, (_, i) in zip(got["per_doc"], docs)]
    per = np.concatenate(per_doc) if per_doc else np.zeros(0)
    return dict(per_token=per, tokens=int(per.size), nll=float(per.mean()) if per.size else float("nan"),
                per_doc=[float(p.mean()) if p.size else float("nan") for p in per_doc])


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

def _plans_from_spec(routes, per_expert, argv, protect=None):
    """every `--keep` / `--frac` value becomes its own plan, so one process can gate a curve of them"""
    keeps = float_list(argv, "--keep", int)
    fracs = float_list(argv, "--frac", float)
    if not keeps and not fracs:
        raise SystemExit("prune: give --keep N[,N...] or --frac F[,F...] (or --selftest)")
    unit = int(_arg(argv, "--unit", 1024))
    plans = []
    for k in keeps:
        # one byte an expert: the budget is a count, and the solve is the same solve
        plans.append(("keep=%d" % k, solve(routes, np.ones_like(per_expert), keep=k, unit=1, protect=protect),
                      "count"))
    for f in fracs:
        plans.append(("frac=%g" % f, solve(routes, per_expert, frac=f, unit=unit, protect=protect), "bytes"))
    return plans


def parse_protect(spec):
    """`1:68,2:92` -> {1: {68}, 2: {92}}; 0-based, the indexing `superex` and the plan JSON use"""
    out = {}
    for item in str(spec).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit("prune: --protect wants layer:expert pairs, not %r" % item)
        l, e = item.split(":", 1)
        out.setdefault(int(l), set()).add(int(e))
    return out


def load_protect(path, ratio=100.0):
    """Read a flagged set rather than a hand-typed list: a `superex` profile, a pair list, or {layer: [ids]}.

    The point of the flag is that the set is a measurement. Re-typing it into `--protect` is how a guard stops
    matching the run it came from, so the artifact is the accepted input.

    **A stored mask is not accepted as the set when the profile is there.** `superex` writes both its masks, and an
    artifact older than the relative threshold carries the global 10%-of-peak cut under the same key - 31 experts here,
    23 of them in layer 47 at 1.7x their own layer's median, and the layer-1 expert at 210x missing from it. So a
    file with `output` is re-thresholded by `superex.flagged` at `ratio` (100x the expert's own layer's median, the cut
    that separates the population cleanly), and a file that only carries a mask is used as given.
    """
    with open(path, encoding="utf-8") as f:
        got = json.load(f)
    if isinstance(got, dict) and got.get("output") is not None and got.get("layers") is not None:
        from .superex import flagged

        out = np.asarray(got["output"], dtype=np.float64)
        mask, _peak = flagged(out, float(got.get("frac", 0.10)), ratio=float(ratio))
        pairs = [[int(l), int(e)] for l, e in zip(*np.nonzero(mask))]
        stored = got.get("flagged")
        note = ""
        if stored is not None and len(stored) != len(pairs):
            note = "; its stored `flagged` mask is the %d-expert global cut, not this set" % len(stored)
        print("  guard: %d expert(s) above %.0fx their own layer's median in %s%s"
              % (len(pairs), float(ratio), os.path.basename(path), note))
    else:
        if isinstance(got, dict):
            pairs = got.get("flagged", got.get("protect"))
            if pairs is None and got and all(str(k).lstrip("-").isdigit() for k in got):
                return {int(k): {int(e) for e in v} for k, v in got.items()}
        elif isinstance(got, list):
            pairs = got
        else:
            pairs = None
        if pairs is None:
            raise SystemExit("prune: %s has no `output` profile, `flagged`/`protect` mask, or {layer: [ids]}" % path)
        print("  guard: %d stored pair(s) from %s, used as given" % (len(pairs), os.path.basename(path)))
    out = {}
    for pair in pairs:
        if len(pair) != 2:
            raise SystemExit("prune: %s: %r is not a (layer, expert) pair" % (path, pair))
        out.setdefault(int(pair[0]), set()).add(int(pair[1]))
    return out


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
              "| --frac F[,F...] [--pack MODEL.gguf] [--protect L:E,...] [--protect-json superex.json] "
              "[--json out.json] [--gate --docs 2 --tokens 96 --max-delta 0.02] [--selftest]")
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

    protect = {}
    for flag, loader in (("--protect", parse_protect), ("--protect-json", load_protect)):
        raw = _arg(argv, flag)
        if raw is None:
            continue
        for l, ids in loader(raw).items():
            protect.setdefault(l, set()).update(ids)
    if protect:
        pairs = [[l, e] for l in sorted(protect) for e in sorted(protect[l])]
        over = sorted(l for l in protect if l >= n_layer or max(protect[l]) >= n_expert)
        if over:
            raise SystemExit("prune: --protect names layer %d, and this log is %d layers of %d experts"
                             % (over[0], n_layer, n_expert))
        print("  guard: %d expert(s) forced into the keep set: %s"
              % (len(pairs), ", ".join("L%dE%d" % (l, e) for l, e in pairs)))
        if pack is not None:
            cost = int(sum(per_expert[l][sorted(ids)].sum() for l, ids in protect.items()))
            print("  guard costs %s bytes, %.2f%% of the expert weights"
                  % ("{:,}".format(cost), 100.0 * cost / float(per_expert.sum())))

    named = _plans_from_spec(got, per_expert, argv, protect=protect or None)
    results = []
    for name, plans, pricing in named:
        print(" %s" % name)
        tot = summary(plans, per_expert, active=active, priced=pricing, sized=pack is not None)
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
        r = nll_moe_batch(model, docs)
        per_token = [r["per_token"]]
        scored = r["tokens"]
        for (name, ids), d_nll in zip(docs, r["per_doc"]):
            print("    %-28s nll %.4f over %d tokens" % (name, d_nll, max(1, len(ids) - 1)))
        baseline = dict(per_token=np.concatenate(per_token), nll=float(np.concatenate(per_token).mean()),
                        tokens=scored)
        print("    baseline nll %.4f over %d tokens" % (baseline["nll"], baseline["tokens"]))

    tol = float(_arg(argv, "--max-delta", NLL_TOL))
    accepted = []
    payload = {"routes": path, "pack": pack_path, "n_layer": n_layer, "n_expert": n_expert,
               "tokens_logged": int(got["route_wts"].shape[1]),
               "protect": ([[l, e] for l in sorted(protect) for e in sorted(protect[l])] if protect else []),
               "plans": []}
    for name, plans, tot, pricing in results:
        plan = as_plan(plans, pricing=pricing)
        row = dict(spec=name, summary=tot, **plan)
        if model is not None:
            model.set_prune(plan["keep_per_layer"])
            r = nll_moe_batch(model, docs)
            per_token, scored = [r["per_token"]], r["tokens"]
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

    # --- the guard: a flagged expert is kept even when a mass budget would drop it, and that is the whole of P0
    massp = np.linspace(1.0, 128.0, 128)                  # expert 0 carries the least routing mass
    sizesp = np.full(128, 1668 * 1024, dtype=np.int64)
    plain = plan_layer(massp, sizesp, budget=96 * sizesp[0])
    check("prune: without the guard, the least-mass experts are the ones dropped", 0 not in plain["keep"])
    guarded = plan_layer(massp, sizesp, budget=96 * sizesp[0], protect=[0])
    check("prune: a protected expert survives the budget that would have dropped it", 0 in guarded["keep"],
          str(guarded["keep"][:5]))
    check("prune: the guard spends the budget rather than raising it",
          guarded["bytes"] <= guarded["budget"] and guarded["protected_bytes"] == int(sizesp[0]),
          "%d of %d" % (guarded["bytes"], guarded["budget"]))
    check("prune: and it takes a kept slot from the unprotected, so the count is unchanged",
          len(guarded["keep"]) == len(plain["keep"]) and len(guarded["keep"]) == 96, str(len(guarded["keep"])))
    # protecting the lightest expert displaces the heaviest one the plain plan kept - the guard is paid for in mass,
    # and the amount it pays is exactly the difference between those two, not a number the solver chose
    check("prune: the guard trades routing mass for the flagged expert, by exactly one slot's worth",
          guarded["dropped_mass"] > plain["dropped_mass"]
          and abs((guarded["dropped_mass"] - plain["dropped_mass"]) - (massp[32] - massp[0])) < 1e-6,
          "%.6g vs %.6g" % (guarded["dropped_mass"], plain["dropped_mass"]))
    top = plan_layer(massp, sizesp, budget=96 * sizesp[0], protect=[127])
    check("prune: protecting an expert the solve keeps anyway changes nothing",
          127 in top["keep"] and len(top["keep"]) == 96
          and abs(top["dropped_mass"] - plain["dropped_mass"]) < 1e-9, str(top["keep"][:4]))
    try:
        plan_layer(massp, sizesp, budget=sizesp[0], protect=[0, 1])
        check("prune: a guard that does not fit the budget is refused, not dropped quietly", False)
    except SystemExit as exc:
        check("prune: a guard that does not fit the budget is refused, not dropped quietly",
              "refused" in str(exc), str(exc))

    # --- the spec and the artifact: 0-based, the same indexing superex and the plan JSON use
    check("prune: --protect parses layer:expert pairs", parse_protect("1:68,2:92,1:114") == {1: {68, 114}, 2: {92}},
          str(parse_protect("1:68,2:92,1:114")))
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(flagged=[[2, 92], [1, 114], [3, 82], [1, 68]]), f)
        check("prune: --protect-json reads superex's flagged set",
              load_protect(tmp) == {1: {68, 114}, 2: {92}, 3: {82}}, str(load_protect(tmp)))
        # a profile is re-thresholded, not trusted: the stored mask here is a stale global cut that names an expert the
        # relative cut does not, and the relative cut names one the stored mask omits
        out = np.ones((2, 4))
        out[0, 2] = 500.0
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_TRUNC), "w", encoding="utf-8") as f:
            json.dump(dict(layers=[0, 1], output=out.tolist(), flagged=[[0, 2], [1, 0]]), f)
        check("prune: --protect-json derives the relative set from a profile instead of its stored mask",
              load_protect(tmp) == {0: {2}}, str(load_protect(tmp)))
    finally:
        os.unlink(tmp)
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
