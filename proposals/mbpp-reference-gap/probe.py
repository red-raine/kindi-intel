"""Prove the gap the payload closes: three planted records, audited against the old and the filled reference set.

A coverage number is a claim about ids; this is the same claim expressed as behaviour. Each probe record is a training
corpus row that quotes a benchmark problem verbatim - two from ids the deployed set never had (5 and 555) and one from
an id it does have (100, the control). The gate is then run twice, changing only the reference file:

    before  the two gap records are *kept*: 100 of 974 MBPP problems are invisible to a 13-gram gate
    after   both are rejected as contaminated, the control still is, and unrelated prose is still kept

    python probe.py --before B:/datasets/data-miner/decontam/reference_sets.json --after /tmp/reference_sets.filled.json

This drives `tools.datahub.gate.Gate` directly rather than the `audit` CLI, for one reason: the CLI rebuilds the
reference 13-gram index per run, and eight runs of a 1,100-record reference is eight rebuilds. The gate itself is the
same object the CLI uses, and each probe record is offered to it one at a time so a rejection is attributable to that
record and not to "the first one in the file".

`--after` is produced by `apply.py --out`; nothing here writes to the reference files.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from tools.datahub import gate, text                            # noqa: E402

PROSE = ("The delivery note describes a warehouse move scheduled for the following spring, and the writer spends three "
         "paragraphs on the loading bay doors before mentioning the inventory at all.")
PROBES = (("mbpp", "5", "prompt", "gap 1-10", ("kept", "contaminated")),
          ("mbpp", "555", "code", "gap 511-600", ("kept", "contaminated")),
          ("mbpp", "100", "code", "control (id present before)", ("contaminated", "contaminated")))


def load(path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def by_id(records, source, task_id):
    for rec in records:
        if rec.get("source") == source and rec.get("task_id") == "%s/%s" % (source, task_id):
            return rec
    return None


def reference_grams(path, n=13):
    """The index the gate builds: 13-grams of every reference record's code, prompt and canonical solution."""
    grams = set()
    for rec in load(path):
        if isinstance(rec, dict):
            for key in ("code", "prompt", "canonical_solution"):
                v = rec.get(key)
                if isinstance(v, str):
                    grams |= text.ngrams(v, n)
    return grams


def verdict(grams, body):
    """One record through a gate holding one reference index: 'kept', or the reason it was refused."""
    g = gate.Gate(gate.Config(min_chars=32))
    g.set_reference(grams, "humaneval+mbpp")
    rep = gate.Report("probe")
    rep.lines = 1
    data = g.offer({"text": body}, rep)
    if data is None:
        reasons = [r for r, c in rep.reasons.items() if c]
        return reasons[0] if reasons else "rejected"
    return "kept"


def main(argv):
    ap = argparse.ArgumentParser(description="audit planted benchmark quotations against two reference sets")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args(argv)

    before, after = load(args.before), load(args.after)
    # the gap records exist only in the filled set, so that is where a probe body can be read from; the point of the
    # probe is then to show `before` cannot see it
    for _s, tid, _f, _l, _w in PROBES:
        if tid in ("5", "555") and by_id(before, "mbpp", tid) is not None:
            raise SystemExit("probe: mbpp/%s is already in --before; this reference set has no such gap" % tid)

    rows = []
    for source, tid, field, label, want in PROBES:
        rec = by_id(after, source, tid)
        if rec is None:
            raise SystemExit("probe: %s/%s is not in --after" % (source, tid))
        rows.append((label, "%s/%s %s" % (source, tid, field), rec[field].strip(), want))
    rows.append(("unrelated prose", "-", PROSE, ("kept", "kept")))

    idx = {}
    for name, path in (("before", args.before), ("after", args.after)):
        if not os.path.exists(path):
            raise SystemExit("probe: no reference set at %s" % path)
        idx[name] = reference_grams(path)
        print("%-6s %-46s %d records, %d 13-grams" % (name, os.path.basename(path), len(load(path)), len(idx[name])))
    print()

    print("%-26s %-24s %-12s %-12s %s" % ("probe", "quotes", "before", "after", ""))
    failures = 0
    for label, origin, body, want in rows:
        got = (verdict(idx["before"], body), verdict(idx["after"], body))
        ok = got == want
        failures += 0 if ok else 1
        print("%-26s %-24s %-12s %-12s %s" % (label, origin, got[0], got[1],
                                              "" if ok else "UNEXPECTED (wanted %s -> %s)" % want))
    print("\n%d of %d probes behaved as the payload claims" % (len(rows) - failures, len(rows)))
    print("a gap record kept by `before` is the defect: the gate cannot see MBPP ids 1-10 or 511-600 in that file.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
