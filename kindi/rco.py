"""kindi rco: which tensors to keep at full precision, under an exact byte budget.

A pack spends its size on codes, and every tensor that is held at f32 instead is 16 times larger than the same tensor
ternarised. Which tensors deserve that is a budget problem, not a preference: a pack has a size the machine can hold,
and the question is which tensors spend those bytes best.

This is the second half of RCO (ISTA-DASLab, arXiv 2605.00649), which the GSQ-RCO releases use to *assign per-tensor
quantization types* under exact budgets. The difference is the search: RCO optimises on a Riemannian manifold through
a differentiable model, and this module does not - it is given one measured error per tensor per format, and solves
the choice that follows exactly. That is the honest trade: the errors here are measured on the weights, not on the
model's output, and the selection between them is exact rather than approximate.

The problem as solved here is a multiple-choice knapsack: one format a tensor, total bytes within budget, minimise the
summed squared error. The dynamic program below is exact on the byte grid it is given (1 KiB units by default). A
problem too large for the table falls back to efficiency-ordered greedy plus a 1-swap pass, and reports the
fractional-knapsack upper bound and the gap, so a heuristic result says so instead of looking exact.

  entries = [dict(name="blk.0.attn_qkv.weight", weights=2048 * 2048,
                  err={"ternary": 3.1e-6, "f32": 0.0},
                  bytes={"ternary": 2048 * 2048 * 2.125 / 8, "f32": 2048 * 2048 * 4})]
  p = plan(entries, budget=900 << 20)          # {"plan": {name: fmt}, "bytes": ..., "gap": 0.0}
"""
import numpy as np

# bits a weight, as the pack actually stores them
FORMATS = {"ternary": 2.0 + 16.0 / 128.0, "f16": 16.0, "f32": 32.0}


def _bytes(weights, fmt):
    return int(weights * FORMATS[fmt] / 8.0) if fmt in FORMATS else int(weights * fmt / 8.0)


def plan(entries, budget, formats=("ternary", "f32"), unit=1024, max_cells=40 << 20):
    """Choose one format a tensor so the total size stays within `budget` and the summed error is minimal.

    `entries` are dicts with `name`, `weights`, `err` (per format, in squared-error units) and optionally `bytes`
    (per format; computed from `FORMATS` otherwise). Every tensor must have an error for every format offered. The
    return is a dict: `plan` (name -> format), `bytes`, `error`, `budget`, `method`, and `bound`/`gap` - the
    fractional-knapsack optimum and how far this solution is from it, both 0 for an exact solve.
    """
    if not entries:
        raise SystemExit("kindi rco: no tensors to plan")
    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        raise SystemExit("kindi rco: a tensor appears twice in the plan input")
    for e in entries:
        for f in formats:
            if f not in e["err"]:
                raise SystemExit("kindi rco: %s has no error for %s" % (e["name"], f))
        e["_b"] = {f: int(e.get("bytes", {}).get(f) or _bytes(e["weights"], f)) for f in formats}
    base = "ternary" if "ternary" in formats else formats[0]     # the cheapest format every tensor can take
    base_bytes = sum(e["_b"][base] for e in entries)
    if budget < base_bytes:
        raise SystemExit("kindi rco: a budget of %d bytes is below the %d-byte floor of %d tensors at %s "
                         "(%.1f MB short)" % (budget, base_bytes, len(entries), base, (base_bytes - budget) / 2 ** 20))
    # upgrades: a tensor's other formats, priced as extra bytes and saved error
    ups = []
    for e in entries:
        row = [(f, e["_b"][f] - e["_b"][base], e["err"][base] - e["err"][f]) for f in formats if f != base]
        row = [r for r in row if r[1] > 0 and r[2] > 0]           # a format that costs nothing or saves nothing is free
        ups.append(row)
    cap = (budget - base_bytes) // unit
    n = len(entries)
    cells = n * (cap + 1)
    if cells <= max_cells and cap > 0:
        # dp[j] = the most error (in summed squared-error units) that j units of budget can save
        dp = np.zeros(cap + 1, dtype=np.float64)
        take = np.full((n, cap + 1), -1, dtype=np.int8)           # which upgrade each tensor took, per capacity
        weights = np.array([e["weights"] for e in entries], dtype=np.float64)
        for i, row in enumerate(ups):
            if not row:
                continue
            best = dp.copy()
            bestc = np.full(cap + 1, -1, dtype=np.int8)
            for k, (f, db, de) in enumerate(row):
                u = max(1, db // unit)
                if u > cap:
                    continue
                cand = dp[:cap + 1 - u] + de * weights[i]         # summed error saved, in weight units
                better = cand > best[u:]
                best[u:][better] = cand[better]
                bestc[u:][better] = k
            dp = best
            take[i] = bestc
        # walk the table back
        j = cap
        chosen = {}
        for i in range(n - 1, -1, -1):
            k = int(take[i][j])
            if k >= 0:
                f, db, _de = ups[i][k]
                chosen[names[i]] = f
                j -= max(1, db // unit)
        used = sum(entries[i]["_b"][chosen.get(names[i], base)] for i in range(n))
        err = sum(entries[i]["weights"] * entries[i]["err"][chosen.get(names[i], base)] for i in range(n))
        return dict(plan={nm: chosen.get(nm, base) for nm in names}, bytes=int(used), error=float(err),
                    budget=int(budget), method="dp", bound=float(err), gap=0.0)

    # greedy by saved error a byte, then a 1-swap pass; reported with the fractional bound so the gap is visible
    order = []
    for i, row in enumerate(ups):
        for f, db, de in row:
            order.append((de * entries[i]["weights"] / db, i, f, db))
    order.sort(reverse=True)
    at = {nm: i for i, nm in enumerate(names)}
    chosen, used = {}, base_bytes
    for eff, i, f, db in order:
        if names[i] in chosen:
            continue
        if used + db <= budget:
            chosen[names[i]] = f
            used += db
    improved = True
    while improved:                                                # 1-swap: exchange one choice for a better one
        improved = False
        for i, nm in enumerate(names):
            if nm in chosen:
                continue
            for f, db, de in ups[i]:
                if used + db > budget:
                    continue
                gain = de * entries[i]["weights"]
                worst = None
                for other, of in chosen.items():
                    j = at[other]
                    for f2, db2, de2 in ups[j]:
                        if f2 != of:
                            continue
                        regain = de2 * entries[j]["weights"] - gain
                        if used - db2 + db <= budget and regain < 0 and (worst is None or regain < worst[0]):
                            worst = (regain, other, j, f2)
                if worst is not None:
                    _r, other, j, f2 = worst
                    used += db - entries[j]["_b"][chosen[other]] + entries[j]["_b"][base]
                    chosen.pop(other)
                    chosen[nm] = f
                    improved = True
                    break
            if improved:
                break
    used = sum(entries[i]["_b"][chosen.get(names[i], base)] for i in range(n))
    err = sum(entries[i]["weights"] * entries[i]["err"][chosen.get(names[i], base)] for i in range(n))
    # fractional knapsack over the same upgrades: an upper bound on what any choice can save
    room, save = budget - base_bytes, 0.0
    for eff, i, f, db in order:
        take = min(room, db)
        save += eff * take
        room -= take
        if room <= 0:
            break
    base_err = sum(entries[i]["weights"] * entries[i]["err"][base] for i in range(n))
    bound = base_err - save
    return dict(plan={nm: chosen.get(nm, base) for nm in names}, bytes=int(used), error=float(err),
                budget=int(budget), method="greedy", bound=float(bound), gap=float(max(0.0, err - bound)))


def summary(p, log=print):
    """One line an operator can act on."""
    fmt = {}
    for f in p["plan"].values():
        fmt[f] = fmt.get(f, 0) + 1
    log("rco: %s in %s, %.2f MB of %.2f MB (error %.6g%s)"
        % (", ".join("%d %s" % (v, k) for k, v in sorted(fmt.items())), p["method"], p["bytes"] / 2 ** 20,
           p["budget"] / 2 ** 20, p["error"],
           "" if p["gap"] == 0 else ", gap %.6g to the bound" % p["gap"]))
    return 0


def selftest():
    """Exactness where the table is small enough, feasibility everywhere, and the trade the plan is supposed to make."""
    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    def ent(name, w, et, ef=0.0):
        return dict(name=name, weights=w, err={"ternary": et, "f32": ef})

    # three tensors of very different sizes and sensitivities
    E = [ent("a", 100_000, 4e-6), ent("b", 1_000_000, 1e-6), ent("c", 10_000, 9e-5)]
    floor = sum(_bytes(e["weights"], "ternary") for e in E)
    full = sum(_bytes(e["weights"], "f32") for e in E)

    p0 = plan([dict(e) for e in E], floor)
    check("rco: a budget at the floor keeps everything ternary",
          set(p0["plan"].values()) == {"ternary"} and p0["bytes"] == floor, str(p0["plan"]))
    p1 = plan([dict(e) for e in E], full)
    check("rco: a budget for everything f32 upgrades everything",
          set(p1["plan"].values()) == {"f32"} and p1["bytes"] <= full, str(p1["plan"]))
    try:
        plan([dict(e) for e in E], floor - 1)
    except SystemExit:
        pass
    else:
        raise AssertionError("a budget below the floor must be refused, not silently exceeded")

    # the plan must never exceed the budget, at any budget in the range
    for frac in (0.0, 0.1, 0.3, 0.5, 0.9, 1.0):
        b = int(floor + (full - floor) * frac)
        p = plan([dict(e) for e in E], b)
        check("rco: the plan fits a %d-byte budget" % b, p["bytes"] <= b, "%d > %d" % (p["bytes"], b))
        check("rco: every tensor is assigned", set(p["plan"]) == {"a", "b", "c"})

    # the choice is about error a byte, not absolute error: the small tensor c has the largest error and the
    # cheapest upgrade, so a budget that can only upgrade one of them must take c, not the largest error per byte a
    # naive "biggest error first" rule would take
    budget = floor + (_bytes(E[2]["weights"], "f32") - _bytes(E[2]["weights"], "ternary"))
    p2 = plan([dict(e) for e in E], budget)
    check("rco: one upgrade goes to the tensor that spends the byte best", p2["plan"]["c"] == "f32",
          str(p2["plan"]))

    # the exact table and the fallback agree on a problem small enough for both, and the fallback reports its gap
    exact = plan([dict(e) for e in E], floor + (full - floor) // 2)
    check("rco: the exact solve reports no gap", exact["gap"] == 0.0 and exact["method"] == "dp")
    tight = plan([dict(e) for e in E], floor + (full - floor) // 2, max_cells=1)
    check("rco: the fallback is feasible too", tight["bytes"] <= tight["budget"] and tight["method"] == "greedy")
    check("rco: the fallback is never better than the bound", tight["error"] >= tight["bound"] - 1e-9,
          "error %.6g bound %.6g" % (tight["error"], tight["bound"]))
    check("rco: the fallback is never worse than the exact solve's error on this input",
          tight["error"] <= exact["error"] + 1e-9, "greedy %.6g exact %.6g" % (tight["error"], exact["error"]))
    return 0


if __name__ == "__main__":
    selftest()
    print("checks passed")
    raise SystemExit(0)
