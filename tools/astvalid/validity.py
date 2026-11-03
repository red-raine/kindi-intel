"""The three-valued validity pass.

One module in, one finding per equivalence class of assignments out:

  FALSE    two definite claims in the class disagree, or pyright's own assignment diagnostic landed on it
  TRUE     the class has exactly one definite claim (or several that agree)
  UNKNOWN  the class exists but nothing definite is known about its type - an abstention, not a pass

A finding is a *class*, not a line: `a = b; b = 1; b = "s"` is one contradiction with three members, and `x = 1;
x = "s"` is a class with one member and two claims.
"""
import os
from dataclasses import dataclass, field

from . import facts as F
from .tri import Tri
from .unionfind import UnionFind, evidence, verdict


@dataclass
class Finding:
    file: str
    members: list
    claims: list
    tri: Tri
    reason: str
    evidence: list = field(default_factory=list)          # pyright messages that landed on this class
    path: str = ""
    support: int = 0                                      # claims that point at `tri`
    against: int = 0                                      # claims that point away from it
    unseen: int = 0                                       # assignment sites whose value stated nothing at all

    @property
    def confidence(self):
        """The pair's second half: how much of the evidence agrees, or None when there is no evidence to agree.

        None is not 0.0 and not 0.5. A class nobody claimed anything about is a different state of knowledge from one
        where the claims split evenly, and folding either into a single score is the collapse this keeps apart.
        """
        total = self.support + self.against
        return (self.support / total) if total and self.tri is not Tri.UNKNOWN else None

    @property
    def names(self):
        return sorted({F.display(m) for m in self.members})

    def claim_text(self):
        out = []
        for c in sorted(self.claims, key=lambda L: (L.line, L.col)):
            out.append("%s <- line %d (%s)" % (c.type, c.line, c.source))
        return ", ".join(out)


@dataclass
class Result:
    findings: list
    notes: list


def claimed(facts, uf, prefix=""):
    """root -> the definite claims on that class, with the guard rule applied.

    An `isinstance` guard describes the value that *arrived*. If the same location is then written, that write replaces
    the value rather than reading it again - `if isinstance(prompt, str): prompt = [prompt]` is normalisation, not a
    contradiction - so a guarded claim is dropped from any location the code reassigns.
    """
    written = set(facts.sites)
    out = {}
    for label in facts.labels:
        if label.source == "isinstance" and label.key in written:
            continue
        out.setdefault(uf.find(prefix + label.key), []).append(label)
    return out


def _place(diag, at, line_keys):
    """The assignment target a pyright diagnostic is about: exact position first, then the line's only target."""
    key = at.get((diag.line, diag.col))
    if key:
        return key, None
    candidates = line_keys.get(diag.line) or []
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, "%s:%d: pyright %s is not on an assignment target the pass tracks (%s)" % (
            diag.file, diag.line, diag.rule or "diagnostic", diag.message[:80])
    return None, "%s:%d: pyright %s could be about any of %s, not placed" % (
        diag.file, diag.line, diag.rule or "diagnostic", ", ".join(sorted(F.display(k) for k in candidates)))


def analyse_source(source, file="<string>", diags=()):
    notes = []
    try:
        facts = F.collect(source, file)
    except SyntaxError as e:
        return Result([], ["%s:%s: cannot parse (%s)" % (file, e.lineno, e.msg)])
    notes.extend(facts.notes)

    uf = UnionFind()
    for key in list(facts.sites) + [c.key for c in facts.labels] + [k for edge in facts.edges for k in edge]:
        uf.add(key)
    for a, b in facts.edges:
        uf.union(a, b)
    claims = claimed(facts, uf)
    poisoned = {}
    line_keys = {}
    for (line, _col), key in facts.at.items():
        line_keys.setdefault(line, []).append(key)

    for d in diags:
        if os.path.basename(d.file) != os.path.basename(file):
            continue
        if not d.is_conflict:
            continue
        key, note = _place(d, facts.at, line_keys)
        if note:
            notes.append(note)
            continue
        root = uf.find(key)
        poisoned.setdefault(root, []).append("%s: %s" % (d.rule or "pyright", d.message))
        if d.got:                                   # what pyright says the value actually is, for the aliases
            claims.setdefault(root, []).append(F.Label(key, d.got, d.line, d.col, "pyright"))

    findings = []
    for root, members in sorted(uf.classes().items(), key=lambda kv: min(kv[1])):
        cls = claims.get(root) or []
        bad = poisoned.get(root) or []
        if bad:
            tri = Tri.FALSE
            reason = "pyright: " + bad[0]
        else:
            tri = verdict(cls)
            if tri is Tri.FALSE:
                reason = "claims disagree: " + ", ".join("%s" % c.type for c in
                                                         sorted(cls, key=lambda L: (L.line, L.col)))
            elif tri is Tri.TRUE:
                reason = "one definite claim: " + cls[0].type
            else:
                reason = "abstained: no definite type claim reaches this class"
        _state, support, against = evidence(cls)
        if bad:
            against = max(against, len(cls))
        sites = sum(len(facts.sites.get(m, ())) for m in members)
        seen = sum(1 for c in cls if c.source in ("value", "annotation", "isinstance"))
        findings.append(Finding(file, members, cls, tri, reason, bad,
                                support=support, against=against, unseen=max(0, sites - seen)))
    return Result(findings, notes)


def analyse_file(path, diags=()):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        return Result([], ["%s: cannot read (%s)" % (path, e)])
    res = analyse_source(source, os.path.basename(path), diags)
    for finding in res.findings:                    # report the path as given, not just the basename
        finding.path = path
    return res


def analyse_paths(paths, diags=()):
    """Every .py under each path (a file, or a directory walked in sorted order)."""
    out, notes = [], []
    for p in paths:
        files = [p] if os.path.isfile(p) else sorted(
            os.path.join(root, name) for root, _dirs, names in os.walk(p)
            for name in names if name.endswith(".py"))
        for path in files:
            res = analyse_file(path, diags)
            out.extend(res.findings)
            notes.extend(res.notes)
    return Result(out, notes)


def by_file(findings):
    out = {}
    for f in findings:
        out.setdefault(f.file, []).append(f)
    return out
