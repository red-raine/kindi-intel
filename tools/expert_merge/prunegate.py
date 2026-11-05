"""One table out of a widened prune gate, and what each arm of it actually establishes.

`tools.expert_merge.prune --gate` scores one arm against a baseline it measured itself, and prints an interval. A
widened gate is the same tool run several times - a curve of keeps, split across processes because each pass is
hours of 48-layer forward - and the thing that has to hold when the pieces come back together is that they are
comparable: every shard scored the same documents, the same tokens, and a baseline that agrees bit for bit with
every other shard's. This module checks that and then says, per arm, which of four claims the interval supports:

  lossless        the whole interval is inside +/- tolerance: the arm costs nothing measurable, and the interval is
                  narrow enough that "nothing" is a claim and not a hope
  not worse       the upper end is inside tolerance but the lower end is not: the arm is certified no worse than the
                  tolerance allows, and may be better - pruning's mass argument predicts a small gain, and an
                  interval that reaches below -tolerance is consistent with one
  loss            the lower end is above tolerance: the arm is certified worse than the tolerance allows
  unresolved      the interval straddles a bound: nothing is established, at this corpus size

The tolerance is the gate's (`--max-delta`, 0.02 nats/token by default). Nothing here re-computes an interval: the
bootstrap is `prune`'s, on paired per-token deltas, and this only reads it. That is deliberate - an interval computed
from summary statistics would be a different estimator wearing the same number.

  python -m tools.expert_merge.prunegate --json out.json wide-112-96.json wide-80-64.json wide-48-32.json
  python -m tools.expert_merge.prunegate --selftest
"""
import json
import os

# the four claims, in the order a reader wants them: the strongest first
LOSS, UNRESOLVED, NOT_WORSE, LOSSLESS = "loss", "unresolved", "not worse", "lossless"


def classify(lo, hi, tol):
    """Which claim an interval supports against a symmetric tolerance. Bounds are inclusive: a bound exactly at the
    tolerance is inside it, and an interval that sits on the line is not evidence of a cost."""
    if lo > tol:
        return LOSS
    if lo >= -tol and hi <= tol:
        return LOSSLESS
    if hi <= tol:
        return NOT_WORSE
    return UNRESOLVED


def _gate_of(plan):
    g = plan.get("gate")
    if not g:
        raise SystemExit("prunegate: %s has no gate: run --gate, not just the plan"
                         % (plan.get("spec") or "a plan"))
    for k in ("delta", "ci_lo", "ci_hi", "max_delta", "tokens"):
        if k not in g:
            raise SystemExit("prunegate: a gate entry is missing %r" % k)
    return g


def merge(paths):
    """Read the shards into one curve: {keep: row}, after checking they are the same experiment.

    The baseline is the control every delta is measured against, so a shard whose baseline NLL differs scored other
    tokens (or another pack, or a damaged one) and its arms are not comparable with the rest. That is refused rather
    than averaged: a baseline that agrees to the last bit is the evidence that the shards ran the same forward.
    """
    if not paths:
        raise SystemExit("prunegate: give at least one prune --gate --json file")
    rows, docs, base_nll, tol = {}, None, None, None
    for path in paths:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        for plan in payload.get("plans", []):
            g = _gate_of(plan)
            if base_nll is None:
                base_nll = float(g.get("baseline_nll", float("nan")))
            elif abs(float(g.get("baseline_nll", float("nan"))) - base_nll) > 1e-9:
                raise SystemExit("prunegate: %s scored a baseline of %.6f, another shard scored %.6f - the shards "
                                 "are not the same experiment" % (os.path.basename(path),
                                                                 g.get("baseline_nll"), base_nll))
            got = [tuple(d) for d in g.get("docs", [])]
            if docs is None:
                docs = got
            elif got != docs:
                raise SystemExit("prunegate: %s scored %s, another shard scored %s"
                                 % (os.path.basename(path), got, docs))
            if tol is None:
                tol = float(g["max_delta"])
            keep = plan["spec"]
            if keep in rows:
                raise SystemExit("prunegate: %s appears in two shards" % keep)
            rows[keep] = dict(spec=keep, nll=float(g.get("nll", float("nan"))), delta=float(g["delta"]),
                              lo=float(g["ci_lo"]), hi=float(g["ci_hi"]), tokens=int(g["tokens"]),
                              tol=float(g["max_delta"]),
                              verdict=classify(float(g["ci_lo"]), float(g["ci_hi"]), float(g["max_delta"])),
                              summary=plan.get("summary", {}))
    order = sorted(rows.values(), key=_keep_of, reverse=True)
    return dict(tolerance=tol, baseline_nll=base_nll, docs=[list(d) for d in (docs or [])],
                tokens=order[0]["tokens"] if order else 0, arms=order)


def _keep_of(row):
    """the keep count an arm's spec names ("keep=96"), so the table reads 128 down to 32 whatever order it ran in"""
    try:
        return int(str(row["spec"]).split("=")[-1])
    except ValueError:
        return -1


def report(merged, log=print):
    """The table, plus the sentence the table exists to support: where the boundary is."""
    tol, base = merged["tolerance"], merged["baseline_nll"]
    log("  %d held-out tokens over %d documents: %s"
        % (merged["tokens"], len(merged["docs"]), ", ".join("%s(%d)" % (n, t) for n, t in merged["docs"])))
    log("  baseline %d experts: nll %.4f; tolerance +/-%.3f nats/token" % (128, base, tol))
    log("")
    log("  | keep | expert bytes | routing mass | nll | change | 95% interval | width | verdict |")
    log("  |---:|---:|---:|---:|---:|---|---:|---|")
    for r in merged["arms"]:
        s = r["summary"]
        log("  | %s | %s | %s | %.4f | %+.4f | %+.4f..%+.4f | %.4f | %s |"
            % (str(r["spec"]).split("=")[-1],
               s.get("bytes_human") or _mb(s.get("bytes_kept") or s.get("expert_bytes")),
               _pct(s.get("dropped_mass", s.get("mass_dropped"))), r["nll"], r["delta"], r["lo"], r["hi"],
               r["hi"] - r["lo"],
               r["verdict"]))
    log("")
    for v in (LOSSLESS, NOT_WORSE, LOSS, UNRESOLVED):
        got = [r["spec"].split("=")[-1] for r in merged["arms"] if r["verdict"] == v]
        if got:
            log("  %-11s %s" % (v + ":", ", ".join(got)))
    narrow = [r for r in merged["arms"] if (r["hi"] - r["lo"]) < 2 * tol]
    log("  narrower than the tolerance: %s"
        % (", ".join(r["spec"].split("=")[-1] for r in narrow) if narrow else "none at this corpus size"))
    return merged


def _mb(n):
    return "%.1f GB" % (float(n) / 1e9) if n else "-"


def _pct(x):
    return "-" if x is None else "%.1f%%" % (100.0 * float(x))


def main(argv):
    if "--selftest" in argv:
        selftest()
        print("checks passed")
        return 0
    json_out = argv[argv.index("--json") + 1] if "--json" in argv else None
    # the output path is a .json too, and reading it as a shard meant the tool could not write the file it was
    # asked for on the first run and could not read it on the second - the one argument that is never an input
    paths = [a for i, a in enumerate(argv) if a.endswith(".json") and i != (argv.index("--json") + 1
                                                                          if json_out else -1)]
    if not paths:
        print("usage: python -m tools.expert_merge.prunegate [--json OUT.json] SHARD.json [SHARD.json ...]")
        return 2
    merged = report(merge(paths))
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=1, sort_keys=True)
        print("  wrote %s" % json_out)
    return 0


def selftest():
    """The classification is the whole tool, so it is tested at every boundary, and the merge is tested to refuse
    shards that are not the same experiment."""
    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    t = 0.02
    check("prunegate: an interval inside the tolerance is lossless", classify(-0.01, 0.01, t) == LOSSLESS)
    check("prunegate: a bound exactly at the tolerance is still inside it", classify(-t, t, t) == LOSSLESS)
    check("prunegate: an interval that reaches below the tolerance but not above is not worse",
          classify(-0.10, 0.015, t) == NOT_WORSE)
    check("prunegate: an upper end exactly at the tolerance is not worse", classify(-0.10, t, t) == NOT_WORSE)
    check("prunegate: a lower end above the tolerance is a loss", classify(0.021, 0.2, t) == LOSS)
    check("prunegate: a lower end exactly at the tolerance is not a loss", classify(t, 0.2, t) != LOSS)
    check("prunegate: an interval that straddles the tolerance establishes nothing",
          classify(-0.05, 0.06, t) == UNRESOLVED)
    check("prunegate: so does one with no interval at all", classify(-0.5, 0.5, t) == UNRESOLVED)

    import tempfile
    def shard(baseline, keep, delta, lo, hi, docs=(("docs/a.md", 48), ("kindi/b.py", 48)), tokens=94, tol=t):
        plan = dict(spec="keep=%d" % keep, summary=dict(expert_bytes=0, mass_dropped=0.056),
                    gate=dict(delta=delta, ci_lo=lo, ci_hi=hi, tokens=tokens, max_delta=tol,
                              baseline_nll=baseline, nll=baseline + delta, docs=[list(d) for d in docs]))
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(dict(plans=[plan]), f)
        return path

    a = shard(4.9563, 96, -0.017, -0.06, 0.05)
    b = shard(4.9563, 64, -0.071, -0.22, 0.06)
    m = merge([a, b])
    check("prunegate: two shards merge into one curve, ordered by keep", [r["spec"] for r in m["arms"]]
          == ["keep=96", "keep=64"], str([r["spec"] for r in m["arms"]]))
    check("prunegate: the baseline is carried once", m["baseline_nll"] == 4.9563)
    check("prunegate: and both arms are unresolved at 94 tokens",
          all(r["verdict"] == UNRESOLVED for r in m["arms"]), str([r["verdict"] for r in m["arms"]]))

    c = shard(4.9, 80, 0.0, -0.01, 0.01)
    try:
        merge([a, c])
    except SystemExit as e:
        check("prunegate: a shard with another baseline is refused, not averaged", "not the same experiment" in str(e))
    else:
        raise AssertionError("a differing baseline must be refused")

    d = shard(4.9563, 80, 0.0, -0.01, 0.01, docs=(("docs/a.md", 48),))
    try:
        merge([a, d])
    except SystemExit as e:
        check("prunegate: a shard that scored other documents is refused", "scored" in str(e))
    else:
        raise AssertionError("a differing corpus must be refused")

    wide = shard(4.9563, 96, -0.017, -0.03, 0.005, tokens=768)
    m2 = merge([wide])
    check("prunegate: a wide corpus is what turns the same delta into a claim",
          m2["arms"][0]["verdict"] == NOT_WORSE, m2["arms"][0]["verdict"])
    try:
        merge([tempfile.mkstemp(suffix=".json")[1] and _no_gate()])
    except SystemExit as e:
        check("prunegate: a plan with no gate is refused rather than read as zero", "no gate" in str(e))
    else:
        raise AssertionError("a plan without a gate must be refused")
    return 0


def _no_gate():
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(dict(plans=[dict(spec="keep=96", summary={})]), f)
    return path


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
