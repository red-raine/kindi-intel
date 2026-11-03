"""The graph layer: node truth is local, consistency is global.

A per-file pass answers "is this location consistent with the claims made about it *here*". It cannot answer the
question the codebase actually asks, because a location can be fine in every file that mentions it and wrong once the
files are joined:

    # conf.py
    marker: int = 1
    # use.py
    from conf import marker
    marker = "id"

Each module is locally consistent - `conf.marker` has one claim, `use.marker` has one - and the pair is not, because the
import makes them one location with an `int` and a `str`. That contradiction has no line to point at: it lives on the
edge, not in a node. So it is reported as a graph finding, and `sat` goes UNSAT, while both nodes keep saying TRUE.

    sat = AND over every node's local truth, and FALSE for each cross-module conflict

`Tri` is already Kleene, so `AND` is the conjunction and the three answers fall out: SAT (every node decided true),
UNSAT (a node is false, or two nodes disagree), UNKNOWN (nothing false, but something abstained). UNKNOWN is not a
failure of the pass; it is the honest statement that some node was not decided, and it is why a two-valued checker
cannot say SAT about a real package.

Two deliberate limits:

* only `from m import n` joins modules - a module-level name and the name it is imported as. `import m` followed by
  `m.x = ...` is not joined, and attribute chains are not joined across modules at all (`@self.x` in two classes is two
  objects, and joining them would be a false positive factory);
* the graph is only as wide as the paths it was given. An import whose target was not analysed is counted as
  `unresolved`, never as agreement.
"""
import os
from dataclasses import dataclass, field

from . import facts as F
from . import validity
from .tri import AND, Tri
from .unionfind import UnionFind, evidence


@dataclass
class Edge:
    kind: str                       # "import" (the only kind that can cross a module boundary)
    a: str                          # global key
    b: str
    site: str                       # file:line that stated it, for the report


@dataclass
class Conflict:
    """One class that spans modules and does not hold together, with the edge that created the span."""

    members: list
    claims: list
    modules: list
    reason: str
    edges: list = field(default_factory=list)

    def describe(self):
        return "%s: %s" % (" | ".join(self.modules), self.reason)


@dataclass
class GraphResult:
    modules: dict                   # id -> path
    sat: Tri
    conflicts: list
    nodes: list                     # validity.Finding, per module, unchanged by the graph layer
    notes: list
    unresolved: int = 0

    def counts(self):
        tri = [n.tri for n in self.nodes]
        return dict(nodes=len(tri), true=sum(1 for t in tri if t is Tri.TRUE),
                    false=sum(1 for t in tri if t is Tri.FALSE),
                    unknown=sum(1 for t in tri if t is Tri.UNKNOWN),
                    conflicts=len(self.conflicts), unresolved=self.unresolved)

    def per_module(self):
        out = {}
        for n in self.nodes:
            out.setdefault(n.file, []).append(n.tri)
        return {k: _conjoin(v) for k, v in out.items()}


def _conjoin(tris):
    if not tris:
        return Tri.TRUE
    acc = tris[0]
    for t in tris[1:]:
        acc = AND(acc, t)
    return acc


def module_id(path):
    """The name Python would import the file as: `kindi/cache.py` -> `kindi.cache`, `kindi/__init__.py` -> `kindi`.

    Derived from the filesystem, by walking up while the directory is a package, rather than from the path the caller
    happened to pass - otherwise `from . import quant` resolves against a module id that only exists relative to wherever
    the command was pointed. A file outside any package keeps its bare name.
    """
    path = os.path.abspath(path)
    name = os.path.basename(path)[:-3] if path.endswith(".py") else os.path.basename(path)
    parts = [] if name == "__init__" else [name]
    d = os.path.dirname(path)
    while os.path.isfile(os.path.join(d, "__init__.py")):
        parts.insert(0, os.path.basename(d))
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return ".".join(parts) or "<root>"


def _target(module, imp):
    """The module id a `from ... import ...` names, or None if the dots cannot be resolved against `module`."""
    if imp.level == 0:
        return imp.module
    base = module.split(".")[:-1]                      # the package this module lives in
    for _ in range(imp.level - 1):
        if not base:
            return None                                # more dots than the module has parents
        base = base[:-1]
    if not imp.module:
        return ".".join(base) or None
    return ".".join(base + [imp.module]) if base else imp.module


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def analyse(paths, diags=()):
    """Walk the paths, collect every module's facts, join the ones the imports connect, and conjoin the truths."""
    files, notes = [], []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        else:
            files.extend(sorted(os.path.join(root, name) for root, _d, names in os.walk(p)
                                for name in names if name.endswith(".py")))
    root = os.path.commonpath([os.path.abspath(f) for f in files]) if files else "."
    if os.path.isfile(root):
        root = os.path.dirname(root)

    modules, facts_by_id = {}, {}
    for path in files:
        mid = module_id(path)
        if mid in modules and os.path.abspath(modules[mid]) != os.path.abspath(path):
            mid = mid + "@" + os.path.relpath(path, root)     # two files outside a package with the same stem
        modules[mid] = path
        try:
            facts_by_id[mid] = F.collect(_read(path), os.path.basename(path))
        except (SyntaxError, OSError) as e:
            notes.append("%s: cannot read (%s)" % (path, getattr(e, "msg", e)))

    uf = UnionFind()
    for mid, facts in facts_by_id.items():
        for key in list(facts.sites) + [c.key for c in facts.labels]:
            uf.add(mid + "|" + key)
        for a, b in facts.edges:
            uf.union(mid + "|" + a, mid + "|" + b)

    edges, unresolved = [], 0
    for mid, facts in facts_by_id.items():
        for imp in facts.imports:
            target = _target(mid, imp)
            if not target or target not in facts_by_id:
                unresolved += 1                    # the graph is only as wide as the paths it was given
                continue
            a, b = mid + "|mod::" + imp.alias, target + "|mod::" + imp.name
            uf.union(a, b)
            edges.append(Edge("import", a, b, "%s:%d" % (os.path.basename(modules[mid]), imp.line)))

    claims = {}                                 # global root -> [(module id, label)], so a claim keeps its origin
    for mid, facts in facts_by_id.items():
        for root_key, labels in validity.claimed(facts, uf, prefix=mid + "|").items():
            claims.setdefault(root_key, []).extend((mid, label) for label in labels)

    conflicts = []
    for root_key, members in uf.classes().items():
        mods = sorted({m.split("|")[0] for m in members})
        if len(mods) < 2:
            continue                            # a single-module class is the node layer's business, not the graph's
        cls = claims.get(root_key) or []
        tri, _support, _against = evidence([L for _mid, L in cls])
        if tri is not Tri.FALSE:
            continue
        touching = [e for e in edges if e.a in members and e.b in members]
        reason = "; ".join("%s.%s says %s" % (mid, F.display(L.key), L.type) for mid, L in cls)
        conflicts.append(Conflict(sorted(members), cls, mods, reason, touching))

    nodes = []
    for mid, path in sorted(modules.items()):
        result = validity.analyse_file(path, diags)
        for finding in result.findings:
            finding.file = mid                  # name modules, not paths, so a conflict reads module-to-module
        nodes.extend(result.findings)
        notes.extend(result.notes)

    sat = _conjoin([n.tri for n in nodes] + [Tri.FALSE] * len(conflicts))
    return GraphResult(modules, sat, conflicts, nodes, notes, unresolved)


def report(g, log=print):
    c = g.counts()
    log("graph: %s  (%d modules, %d nodes)" % (_verdict(g.sat), len(g.modules), c["nodes"]))
    log("  node truth   %d true, %d unknown, %d false" % (c["true"], c["unknown"], c["false"]))
    log("  cross-module %d conflict(s), %d import(s) not resolved (target not analysed)"
        % (c["conflicts"], c["unresolved"]))
    for mid, tri in sorted(g.per_module().items()):
        mine = [n for n in g.nodes if n.file == mid]
        unknown = sum(1 for n in mine if n.tri is Tri.UNKNOWN)
        log("  %-28s %-7s %d nodes, %d unknown" % (mid, _verdict(tri), len(mine), unknown))
    for conflict in g.conflicts:
        log("")
        log("  CONFLICT %s" % conflict.describe())
        for edge in conflict.edges:
            log("    joined by %s (%s)" % (edge.site, edge.kind))
        for mid, label in conflict.claims:
            log("    %-30s %s  (line %d)" % (mid + "." + F.display(label.key), label.type, label.line))
    for note in g.notes[:10]:
        log("note: %s" % note)


def _verdict(tri):
    return {Tri.TRUE: "SAT", Tri.FALSE: "UNSAT", Tri.UNKNOWN: "UNKNOWN"}[tri]
