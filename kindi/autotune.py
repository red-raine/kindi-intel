"""kindi autotune: the importer choosing its own quantiser, tensor by tensor, by measuring it.

`kindi qat-pack` takes one scale scheme for the whole checkpoint - `--scheme lsq` and every tensor is quantised
that way - and then, if there is a `--budget`, decides which tensors stay f32. The first of those two choices is
the one an operator cannot actually make: whether a tensor's weights are shaped like the ones `absmean` was
designed for, or heavy-tailed enough to want `gsq`, is a property of that tensor, it differs inside one model, and
it is measurable in the only unit that matters - the reconstruction error the kernel will produce.

So this module measures it, per tensor, and then hands the answer to the solver that already exists:

  measure   every tensor x every scheme -> the squared error `quantise()` actually returns, and the bytes
  choose    per tensor, the scheme with the least error
  allocate  `kindi.rco.plan` under the same byte budget, with the chosen scheme's error as the price of ternary

The per-tensor choice is free, and that is the whole reason it can be made without a budget: a ternary tensor is
`C/4` code bytes and `C/128` fp16 scales whatever scheme picked the codes, so `absmean`, `lsq` and `gsq` differ in
one number and not in one byte. Bytes are spent only by staying f32, and that decision is the knapsack `--budget`
already solves - the only change here is that it is now priced with the error the pack will really have, instead
of the error one global scheme happened to have.

  from . import autotune
  pol = autotune.tune(sd, budget=900 << 20)        # {"schemes": {name: "gsq"}, "formats": {...}, ...}
  pol["error"], pol["global"]["lsq"]["error"]      # what the tuned pack costs against the best single scheme

What this is not: the objective is still weight-space squared error, not the layer's output error - there is no
calibration set and no Hessian here, the same limit `kindi.gsq` states for itself. A per-tensor *group size* would
be the next lever and is not offered, because the pack layout stores one fp16 scale per 128 weights and the reader
would have to change with it. And a tensor's best scheme is chosen on the same weights the pack is written from,
so this is a fit and not a cross-validation: it says which quantiser represents these weights best, not which one
generalises to text the model was not trained on.
"""
import json
import os
import sys

import numpy as np

from .quant import GROUP, SCHEMES, _pad8, is_ternary, quantise

# the scheme a tie falls back to: `lsq` is the default the CLI already had, so an unbroken tie changes nothing
TIE = "lsq"


def measure(sd, names=None, group=GROUP, schemes=SCHEMES, log=None):
    """One row a tensor: the error each scheme produces on it, and what that costs in bytes.

    The error is the mean squared difference between the tensor and what `quantise()` decodes back - the pack's
    own arithmetic, fp16 scale included - over the padded matrix `kindi.quant._pad8` writes, so it is the same
    quantity `pack()` already feeds the budget solver and the two numbers are comparable.
    """
    schemes = tuple(schemes)
    for s in schemes:
        if s not in SCHEMES:
            raise ValueError("unknown scale scheme %r (have %s)" % (s, ", ".join(SCHEMES)))
    rows = []
    for k in (names if names is not None else [k for k in sorted(sd) if is_ternary(k, sd[k])]):
        a = np.ascontiguousarray(sd[k], dtype=np.float32)
        ap, n_rows = _pad8(a)
        err, cost = {}, {}
        for s in schemes:
            codes, sc, deq = quantise(ap, group, s)
            err[s] = float(((ap - deq) ** 2).mean())
            cost[s] = int(codes.size + 2 * sc.size)
        costs = set(cost.values())
        if len(costs) != 1:
            # if this ever stops being true the choice stops being free, and the budget solve below would be wrong
            raise AssertionError("schemes disagree on the bytes of %s: %s" % (k, cost))
        if log:
            log("  %-40s %s" % (k, " ".join("%s %.3e" % (s, err[s]) for s in schemes)))
        rows.append(dict(name=k, weights=int(ap.size), err=err,
                         bytes={"ternary": cost[schemes[0]], "f32": int(4 * ap.size)}))
    return rows


def _pick(row, schemes, within=0.0):
    """The scheme to use for one tensor: the cheapest whose error is within `within` of the best measured one.

    `within=0` is the strict minimum, ties to `lsq`. A tolerance is the other half of the decision, because the
    schemes do not cost the same to fit: `absmean` is one pass, `lsq` searches a 33-point grid and `gsq` runs its
    alternation to a fixed point, so a scheme that wins by 0.01% on a tensor costs ~50x what the one it beat costs.
    With a tolerance the cheap scheme is chosen unless the expensive one is measurably better than it, which is the
    trade a pack of a thousand tensors has to make.
    """
    order = sorted(schemes, key=lambda s: (row["err"][s], s != TIE, schemes.index(s)))
    best = order[0]
    if not within:
        return best
    ceiling = row["err"][best] * (1.0 + float(within))
    cheap_first = sorted(schemes, key=lambda s: schemes.index(s))       # SCHEMES is quoted cheap to expensive
    for s in cheap_first:
        if row["err"][s] <= ceiling:
            return s
    return best


def _total(rows, err_of):
    """error summed the way the budget solver sums it (per-weight error x weight count), and as a per-weight mean"""
    tot = sum(r["weights"] * err_of(r) for r in rows)
    return float(tot / max(1, sum(r["weights"] for r in rows)))


def _allocate(rows, err_key, budget):
    """one rco solve: ternary at `err_key`'s error a tensor, f32 at none, total bytes within `budget`"""
    from . import rco

    entries = [dict(name=r["name"], weights=r["weights"],
                    err={"ternary": r["err"][err_key] if isinstance(err_key, str) else err_key(r), "f32": 0.0},
                    bytes=dict(r["bytes"])) for r in rows]
    p = rco.plan(entries, int(budget), formats=("ternary", "f32"))
    return p


def tune(sd, budget=None, group=GROUP, schemes=SCHEMES, within=0.0, log=print):
    """Choose a scheme a tensor (free) and then a format a tensor (under `budget`), both by measurement.

    Returns `schemes` (name -> scheme), `formats` (name -> "ternary"/"f32"), the resulting `error` as a per-weight
    mean and `bytes`, and `global`: what each single scheme would have cost if the whole checkpoint had used it,
    solved through the same budget. The comparison is the point - it is the number that says whether the choice
    was worth measuring - so `global` is computed under the same budget and the same solver, not by holding the
    tuned plan's allocation fixed.
    """
    names = [k for k in sorted(sd) if is_ternary(k, sd[k])]
    if not names:
        raise SystemExit("kindi autotune: no 2-D weight matrices in this checkpoint, so there is nothing to tune")
    schemes = tuple(schemes)
    if log:
        log("  autotune: %d ternary tensors, schemes %s" % (len(names), ", ".join(schemes)))
    rows = measure(sd, names, group=group, schemes=schemes, log=log)
    picks = dict((r["name"], _pick(r, schemes, within)) for r in rows)
    by_name = dict((r["name"], r) for r in rows)

    def tuned_err(r):
        return r["err"][picks[r["name"]]]

    if budget is None:
        p = dict(plan=dict((r["name"], "ternary") for r in rows), error=_total(rows, tuned_err),
                 bytes=sum(r["bytes"]["ternary"] for r in rows), method="none", budget=None)
    else:
        p = _allocate(rows, tuned_err, budget)
    formats = p["plan"]

    glob = {}
    for s in schemes:
        if budget is None:
            g = dict(plan=dict((r["name"], "ternary") for r in rows), error=_total(rows, lambda r: r["err"][s]),
                     bytes=sum(r["bytes"]["ternary"] for r in rows), method="none", budget=None)
        else:
            g = _allocate(rows, s, budget)
        glob[s] = dict(error=float(g["error"]), bytes=int(g["bytes"]), method=g["method"],
                       f32=sum(1 for f in g["plan"].values() if f == "f32"), gap=float(g.get("gap", 0.0)))
    best_global = min(glob, key=lambda s: (glob[s]["error"], s))
    out = dict(schemes=picks, formats=formats, error=float(p["error"]), bytes=int(p["bytes"]),
               budget=None if budget is None else int(budget), method=p["method"], within=float(within),
               tensors=len(rows), schemes_tried=list(schemes), global_=glob, best_global=best_global,
               vs_best_global=float(p["error"] - glob[best_global]["error"]),
               vs_best_global_pct=float(100.0 * (p["error"] / glob[best_global]["error"] - 1.0))
               if glob[best_global]["error"] else 0.0,
               f32=[k for k, f in formats.items() if f == "f32"],
               counts=dict((s, sum(1 for v in picks.values() if v == s)) for s in schemes),
               table=[dict(name=r["name"], weights=r["weights"], err=r["err"],
                           scheme=picks[r["name"]], fmt=formats[r["name"]]) for r in rows])
    out["global"] = out.pop("global_")
    if log:
        log("  autotune: %s; %d f32 of %d"
            % (", ".join("%d %s" % (out["counts"][s], s) for s in schemes), len(out["f32"]), len(rows)))
        log("  autotune: %.6g per weight (%s), best single scheme %s at %.6g (%+.2f%%)"
            % (out["error"], "%.2f MB" % (out["bytes"] / 2 ** 20), best_global, glob[best_global]["error"],
               out["vs_best_global_pct"]))
    return out


def write(pol, out_dir):
    """the decision, written beside the pack so it can be read back rather than trusted

    `drift.verify_pack` and `kindi.quant.pack` both read this file when the scheme is "auto": the per-tensor map is
    the only record of what the pack's bytes mean when they are not all one scheme.
    """
    path = os.path.join(out_dir, "autotune.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict((k, v) for k, v in pol.items() if k != "table"), f, indent=1, sort_keys=True)
    return path


def load(pack_dir):
    """the policy a pack was written with: {name: scheme}, or None if it was written with one global scheme"""
    path = os.path.join(pack_dir, "autotune.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("schemes") or None


def policy_for(scheme, sd, out_dir=None):
    """{name: scheme} for a pack: the file's own policy, one global scheme, or None for "every tensor the same" """
    if isinstance(scheme, dict):
        return scheme
    if scheme == "auto":
        got = load(out_dir) if out_dir else None
        if got is None:
            raise SystemExit("kindi: scheme 'auto' but no autotune.json in %s" % (out_dir,))
        return got
    return None


def main(argv=None):
    """`python -m kindi.autotune CHECKPOINT.pt [--budget BYTES] [--json out.json]`: the policy, before the pack"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        selftest()
        print("checks passed")
        return 0
    if not argv or argv[0].startswith("-"):
        print("usage: python -m kindi.autotune CHECKPOINT.pt [--budget BYTES] [--schemes absmean,lsq,gsq] "
              "[--group 128] [--within 0.01] [--json out.json] [--selftest]\n"
              "  --within takes a cheaper scheme when it is within that fraction of the best error a tensor")
        return 2
    path = argv[0]
    budget = int(argv[argv.index("--budget") + 1]) if "--budget" in argv else None
    group = int(argv[argv.index("--group") + 1]) if "--group" in argv else GROUP
    schemes = tuple(argv[argv.index("--schemes") + 1].split(",")) if "--schemes" in argv else SCHEMES
    within = float(argv[argv.index("--within") + 1]) if "--within" in argv else 0.0
    from . import ptload

    sd = dict((k, ptload.widen(v)) for k, v in ptload.load(path).items() if v.dtype.kind in "fiu")
    pol = tune(sd, budget=budget, group=group, schemes=schemes, within=within, log=lambda *a: print(*a))
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pol, f, indent=1, sort_keys=True)
        print("  wrote %s" % out)
    return 0


def selftest():
    """The choice is free, it is the measured minimum, and the budget still binds.

    What is asserted is not that `gsq` wins - that is a measurement, and it is one `kindi.gsq` already makes - but
    that the policy picks the scheme with the least measured error a tensor, that the arithmetic the choice claims
    holds (equal bytes, a total no worse than any single scheme), and that determinism and the budget survive it.
    """
    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    rng = np.random.RandomState(7)
    sd = {}
    for i, (rows, cols) in enumerate(((48, 256), (32, 128), (8, 128))):
        sd["layers.%d.w.weight" % i] = (rng.randn(rows, cols) * 0.02).astype(np.float32)
    # heavy-tailed: what gsq's own note says it is 11% better on
    sd["layers.3.w.weight"] = (rng.standard_t(2.0, size=(40, 256)) * 0.02).astype(np.float32)
    sd["layers.0.norm.weight"] = np.ones(256, dtype=np.float32)          # not 2-D per `is_ternary`: untouched
    sd["layers.0.odd.weight"] = rng.randn(4, 100).astype(np.float32)     # width not a multiple of 128: untouched

    # the same tensor through each scheme at the same width must be the same size on disk: this is what makes the
    # per-tensor choice free, and `measure` raises if a scheme ever stops agreeing with the others
    sizes = {}
    for s in ("absmean", "lsq", "gsq"):
        codes, sc, deq = quantise(np.ascontiguousarray(sd["layers.0.w.weight"], dtype=np.float32), 128, s)
        sizes[s] = codes.size + 2 * sc.size
    check("autotune: a ternary tensor is C/4 code bytes plus C/128 fp16 scales whatever the scheme",
          len(set(sizes.values())) == 1, str(sizes))

    pol = tune(sd, budget=None, log=None)
    check("autotune: every ternary tensor gets a policy and a format",
          set(pol["schemes"]) == {"layers.%d.w.weight" % i for i in range(4)}, str(sorted(pol["schemes"])))
    check("autotune: float and ragged tensors are left out of the plan",
          "layers.0.norm.weight" not in pol["schemes"] and "layers.0.odd.weight" not in pol["schemes"])
    for row in pol["table"]:
        check("autotune: %s got the scheme with the least measured error" % row["name"],
              row["err"][row["scheme"]] == min(row["err"].values()), str(row["err"]))
        check("autotune: %s is ternary when there is no budget" % row["name"], row["fmt"] == "ternary")
    check("autotune: the tuned total is never worse than any single global scheme",
          all(pol["error"] <= pol["global"][s]["error"] + 1e-12 for s in pol["schemes_tried"]),
          str(pol["error"]) + " " + str(dict((s, pol["global"][s]["error"]) for s in pol["schemes_tried"])))
    check("autotune: and it is strictly better when two tensors disagree about their scheme",
          len(set(pol["schemes"].values())) < 2 or pol["error"] < min(pol["global"][s]["error"] for s in pol["schemes_tried"]),
          "%s vs %s" % (pol["error"], dict((s, pol["global"][s]["error"]) for s in pol["schemes_tried"])))
    check("autotune: the heavy-tailed tensor is the one gsq is for", pol["schemes"]["layers.3.w.weight"] == "gsq",
          pol["schemes"]["layers.3.w.weight"])

    again = tune(sd, budget=None, log=None)
    check("autotune: a second run gives the same policy and the same error",
          again["schemes"] == pol["schemes"] and again["error"] == pol["error"])

    # a tolerance trades error for fit cost: it must only ever choose something cheaper, and only within the
    # fraction it was given - that is the whole contract of the `within` knob
    loose = tune(sd, within=0.05, log=None)
    for row in loose["table"]:
        check("autotune: within() never picks a pricier scheme than the strict minimum on %s" % row["name"],
              SCHEMES.index(row["scheme"]) <= SCHEMES.index(pol["schemes"][row["name"]]),
              "%s vs %s" % (row["scheme"], pol["schemes"][row["name"]]))
    check("autotune: and the error it gives up stays inside the tolerance",
          loose["error"] <= pol["error"] * 1.05 + 1e-12,
          "%s vs %s" % (loose["error"], pol["error"]))

    # the budget still binds, and it buys the tensors rco says it buys
    floor = sum(r["bytes"]["ternary"] for r in measure(sd, log=None))
    full = sum(r["bytes"]["f32"] for r in measure(sd, log=None))
    for frac in (0.0, 0.35, 1.0):
        b = int(floor + (full - floor) * frac)
        p = tune(sd, budget=b, log=None)
        check("autotune: the tuned plan fits a %d-byte budget" % b, p["bytes"] <= b, "%d > %d" % (p["bytes"], b))
        check("autotune: every tensor is assigned at %d bytes" % b,
              set(p["formats"]) == set(p["schemes"]))
    up = tune(sd, budget=full, log=None)
    check("autotune: a budget for everything f32 spends it and has no error left",
          up["error"] == 0.0 and all(f == "f32" for f in up["formats"].values()), str(up["formats"]))
    check("autotune: and it is still no worse than any single scheme at that budget",
          all(up["error"] <= up["global"][s]["error"] + 1e-12 for s in up["schemes_tried"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
