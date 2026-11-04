"""Coverage of a reference set: which benchmark task ids it actually holds, against the ids it claims.

Decontamination is only as good as the reference it compares against. A set holding 874 of MBPP's 974 problems cannot
see a leaked solution built on one of the other 100, and the gate then reports "no contamination" with exactly the
confidence it would have reported a clean corpus. Counting records does not answer that - 1,038 records could be 1,038
copies of one task - so this module reads the task ids, groups them into benchmark families, and compares each family
with the number of problems it claims.

The claim comes from `--expect FAMILY=N` (or `FAMILY=LO:HI`), and by default from the published sizes of the two
benchmarks the gate is built for: HumanEval numbers 0-163 and MBPP 1-974. A default claim applies to a family the set
actually mentions, so an MBPP-only reference is not failed for holding no HumanEval; an explicitly named family is
enforced whether or not the set holds any of it, because naming it is the claim.

Ids are grouped by the prefix before the slash, lowercased and cut at the first underscore, so `mbpp`, `mbpp_s` and
`mbpp_sanitized` are one id space: they are splits of one benchmark, and "974 of 974" is a statement about the union.
"""
import re

FIRST = {"humaneval": 0, "mbpp": 1}
SIZES = {"humaneval": 164, "mbpp": 974}
_ID = re.compile(r"^\s*([^\s/]+)\s*/\s*(\d+)\s*$")


def family(prefix):
    """the id space a task id's prefix belongs to: `mbpp`, `mbpp_s` and `mbpp_sanitized` are all `mbpp`"""
    return str(prefix).strip().lower().split("_")[0]


def ranges(values, limit=24):
    """[1, 2, 3, 10] -> ('1-3, 10', 2). A very scattered set is cut at `limit` ranges, with a count for the rest."""
    runs = []
    for i in sorted(values):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    fmt = lambda r: "%d" % r[0] if r[0] == r[1] else "%d-%d" % (r[0], r[1])       # noqa: E731
    text = ", ".join(fmt(r) for r in runs[:limit])
    if len(runs) > limit:
        text += ", +%d more" % (len(runs) - limit)
    return text, len(runs)


def ids(records):
    """-> (families: {family: set(ids)}, duplicates: {family: [id]}, unparsable: [task_id], sources: {family: {source: n}})

    A record with no `task_id`, or one that is not `prefix/number`, is unparsable: it cannot contribute to coverage,
    and a set containing it cannot be said to hold what it claims.
    """
    fam, dup, bad, src = {}, {}, [], {}
    for rec in records:
        tid = rec.get("task_id") if isinstance(rec, dict) else None
        m = _ID.match(tid) if isinstance(tid, str) else None
        if m is None:
            bad.append(tid)
            continue
        f, i = family(m.group(1)), int(m.group(2))
        have = fam.setdefault(f, set())
        if i in have:
            dup.setdefault(f, []).append(i)
        have.add(i)
        source = str(rec.get("source") or "?")
        src.setdefault(f, {})
        src[f][source] = src[f].get(source, 0) + 1
    return fam, dup, bad, src


def claims(specs=None):
    """-> (expected, named): `expected` maps family -> (lo, hi) inclusive; `named` is the families the caller stated"""
    expected = dict((f, (FIRST[f], FIRST[f] + SIZES[f] - 1)) for f in SIZES)
    named = set()
    for spec in specs or ():
        if "=" not in spec:
            raise SystemExit("datahub: --expect wants FAMILY=N or FAMILY=LO:HI (got %r)" % (spec,))
        name, value = spec.split("=", 1)
        f = family(name)
        try:
            if ":" in value:
                lo, hi = (int(x) for x in value.split(":", 1))
            else:
                lo = FIRST.get(f, 1)
                hi = lo + int(value) - 1
        except ValueError:
            raise SystemExit("datahub: --expect %r is not FAMILY=N or FAMILY=LO:HI" % (spec,))
        if hi < lo:
            raise SystemExit("datahub: --expect %s claims no ids at all" % (spec,))
        expected[f] = (lo, hi)
        named.add(f)
    return expected, named


class Coverage(object):
    def __init__(self, records, rows, unparsable):
        self.records = records
        self.rows = rows
        self.unparsable = unparsable

    @property
    def missing(self):
        return sum(r["missing"] for r in self.rows)

    @property
    def ok(self):
        return not self.unparsable and self.missing == 0

    def failure(self):
        """one line naming what is wrong, for a FAIL or a WARNING - empty families are not the only way to be short"""
        parts = ["%s: %d of %d ids missing (%s)" % (r["family"], r["missing"], r["claimed"], r["gaps"])
                 for r in self.rows if r["missing"]]
        if self.unparsable:
            parts.append("%d record(s) with no parsable task_id" % len(self.unparsable))
        return "; ".join(parts) or "complete"

    def report(self):
        out = ["reference coverage: %d record(s), %d id famil%s"
               % (self.records, len(self.rows), "y" if len(self.rows) == 1 else "ies")]
        for r in self.rows:
            state = ("MISSING %d of %d: %s" % (r["missing"], r["claimed"], r["gaps"]) if r["missing"] else
                     "complete" + (" (%d id(s) outside the claim)" % r["extra"] if r["extra"] else ""))
            out.append("  %-10s %4d-%-4d %4d present   %s" % (r["family"], r["lo"], r["hi"], r["present"], state))
            if r["sources"]:
                out.append("             from %s" % ", ".join("%s %d" % kv for kv in sorted(r["sources"].items())))
            if r["duplicates"]:
                out.append("             %d repeated id(s): %s" % (r["duplicates"], r["dup_ids"]))
        if self.unparsable:
            out.append("  %d record(s) have no parsable task_id (first: %r)"
                       % (len(self.unparsable), self.unparsable[0]))
        return "\n".join(out)

    def as_dict(self):
        return dict(ok=self.ok, records=self.records, missing=self.missing,
                    unparsable=[str(x) for x in self.unparsable[:20]], families=self.rows)


def check(records, expected=None, required=()):
    """Compare a reference set's ids with what it claims. Returns a `Coverage`.

    `required` names the families to enforce even when the set holds none of them, which is what makes an explicitly
    stated claim a claim rather than a default.
    """
    fam, dup, bad, src = ids(records)
    expected = dict(expected or {})
    rows = []
    for f in sorted(expected):
        if f not in fam and f not in required:
            continue
        lo, hi = expected[f]
        have = fam.get(f, set())
        inside = set(i for i in have if lo <= i <= hi)
        missing = [i for i in range(lo, hi + 1) if i not in inside]
        gaps, _ranges = ranges(missing)
        rows.append(dict(family=f, lo=lo, hi=hi, claimed=hi - lo + 1, present=len(inside), missing=len(missing),
                         gaps=gaps, extra=sum(1 for i in have if not lo <= i <= hi),
                         duplicates=len(dup.get(f, [])), dup_ids=ranges(dup.get(f, []))[0],
                         sources=dict(src.get(f, {}))))
    return Coverage(len(records), rows, bad)
