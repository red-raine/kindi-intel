"""Score the pass's contradiction detection against the hand-labelled sample.

A label says two things about a case: which assignment classes are genuinely contradictory (by the names the code
uses), and what the case's three-valued verdict should be. Matching is by (file, set of names), so a contradiction
reported on the right class counts once however many lines feed it, and a contradiction reported on a different class
is a false positive rather than a lucky hit.

What the metrics say, and what they do not: precision/recall here describe this sample, which is small and written by
the same person who wrote the rules. The abstention count is the more portable number - UNKNOWN findings are locations
where the pass declined to guess, and every one of them is a place where a two-valued checker would have had to.
"""
import json
import os

from . import validity
from .tri import Tri, parse as parse_tri


def load_labels(case_dir):
    with open(os.path.join(case_dir, "labels.json"), encoding="utf-8") as f:
        return json.load(f)


def _expected(data):
    out = {}
    for case in data["cases"]:
        groups = {frozenset(g) for g in case.get("contradictions", [])}
        out[case["file"]] = (groups, parse_tri(case["tri"]))
    return out


def _case_tri(findings):
    if any(f.tri is Tri.FALSE for f in findings):
        return Tri.FALSE
    if any(f.tri is Tri.TRUE for f in findings):
        return Tri.TRUE
    return Tri.UNKNOWN


def score(case_dir, diags=(), log=print):
    data = load_labels(case_dir)
    expected = _expected(data)
    files = [os.path.join(case_dir, name) for name in sorted(expected)]
    result = validity.analyse_paths(files, diags)
    found = validity.by_file(result.findings)

    log("%-32s %-8s %-8s %-28s %s" % ("case", "label", "pass", "expected contradictions", "reported contradictions"))
    tp = fp = fn = agree = 0
    misses = []
    for name in sorted(expected):
        groups, want_tri = expected[name]
        findings = found.get(name, [])
        got_tri = _case_tri(findings)
        got = {frozenset(f.names) for f in findings if f.tri is Tri.FALSE}
        hit = groups & got
        tp += len(hit)
        fp += len(got - groups)
        fn += len(groups - hit)
        agree += int(got_tri == want_tri)
        for group in sorted(groups - got):
            misses.append("%s: %s" % (name, "+".join(sorted(group)) if group else "(empty class)"))
        log("%-32s %-8s %-8s %-28s %s%s"
            % (name, want_tri, got_tri,
               "; ".join("+".join(sorted(g)) for g in sorted(groups)) or "-",
               "; ".join("+".join(sorted(g)) for g in sorted(got)) or "-",
               "" if hit or not groups else "   MISS" if not (got & groups) else "   partly"))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    unknown = sum(1 for f in result.findings if f.tri is Tri.UNKNOWN)
    definite = sum(1 for f in result.findings if f.tri is not Tri.UNKNOWN)
    total = len(result.findings)

    log("")
    log("contradictions        %d found, %d expected, %d matched, %d missed, %d false"
        % (tp + fp, tp + fn, tp, fn, fp))
    log("precision %.3f   recall %.3f   f1 %.3f" % (precision, recall, f1))
    log("verdicts on %d assignment classes: %d definite, %d UNKNOWN (abstained on %.0f%%)"
        % (total, definite, unknown, 100.0 * unknown / total if total else 0.0))
    log("case verdict agreement %d/%d" % (agree, len(expected)))
    for note in result.notes:
        log("note: %s" % note)
    for miss in misses:
        log("missed: %s" % miss)
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn,
                unknown=unknown, definite=definite, agree=agree, cases=len(expected), misses=misses,
                findings=result.findings, notes=result.notes)
