"""Union-find over assignment sites.

The pass does not ask "does this line contradict that one"; it asks "do all the places this value can come from agree".
Assignments make that a union-find problem: `a = b` and `b = obj.n` put `a`, `b` and `obj.n` in one class, and the
class is contradictory only if the definite type claims attached to its members disagree.

Keys are the pass's own spelling of a location, not the source text:
  `mod::x` / `f::y`      a plain name, qualified by the enclosing function/class chain (so two functions with a local
                         `x` are two locations - the sample has a case for it)
  `@obj.n` / `@obj.a.b`  an attribute chain, deliberately *not* scope-qualified: the same object attribute is the same
                         location wherever it is written
"""
from .tri import Tri


class UnionFind(object):
    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)
        return x

    def find(self, x):
        p = self.parent
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:                     # path compression, iterative
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return ra

    def classes(self):
        """root -> sorted members"""
        out = {}
        for x in list(self.parent):
            out.setdefault(self.find(x), []).append(x)
        return {root: sorted(members) for root, members in out.items()}

    def of(self, keys):
        """the distinct classes that `keys` land in"""
        return {self.find(k) for k in keys}


def evidence(labels):
    """(state, support, against) for one class: the verdict *and* the counts behind it, never one collapsed number.

    A caller that only wants the verdict takes the first element; a caller reporting confidence must keep the pair,
    because `UNKNOWN` with no claims at all and `UNKNOWN` with four claims that say nothing definite are different
    states of knowledge, and a single scalar would make them the same.
    """
    tri = _verdict(labels)
    if not labels:
        return tri, 0, 0
    declared = [L for L in labels if L.source == "annotation"]
    if declared:
        base = min(declared, key=lambda L: (L.line, L.col)).type
        support = sum(1 for L in labels if assignable(L.type, base))
        return tri, support, len(labels) - support
    if tri is Tri.FALSE:                       # no declaration: support is the largest set that agrees pairwise
        return tri, _largest_agreeing(labels), len(labels) - _largest_agreeing(labels)
    return tri, len(labels), 0


def _largest_agreeing(labels):
    """The biggest subset of claims in which every pair is compatible - what a disagreement is measured against."""
    best = 0
    for i, L in enumerate(labels):
        n = 1 + sum(1 for j, M in enumerate(labels) if j != i and compatible(L.type, M.type))
        best = max(best, n)
    return best


def verdict(labels):
    """Three-valued verdict for one class from its definite type claims.

    Two rules, because Python has two:

    * an **annotation** declares a type, so every other claim must be assignable to it - one way. `x: int = "s"` is a
      contradiction; `x: float = 1` is not.
    * with no annotation, nothing is declared and the claims only have to be *compatible* - related either way round
      by the subtype lattice. `x = True; x = 1` is `bool` then `int` and compatible; `x = 1; x = "s"` is not.

    A class with no definite claim is UNKNOWN: the pass does not infer a type to have an opinion about.
    """
    return _verdict(labels)


def _verdict(labels):
    if not labels:
        return Tri.UNKNOWN
    declared = [L for L in labels if L.source == "annotation"]
    if declared:
        base = min(declared, key=lambda L: (L.line, L.col)).type
        return Tri.TRUE if all(assignable(L.type, base) for L in labels) else Tri.FALSE
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if not compatible(a.type, b.type):
                return Tri.FALSE
    return Tri.TRUE


def compatible(a, b):
    """Can one location hold both? True when either type is assignable to the other."""
    return assignable(a, b) or assignable(b, a)


def assignable(sub, sup):
    """Can a value of type `sub` be bound where `sup` is declared? (a deliberately small lattice)"""
    return all(any(_one(a, b) for b in _alternatives(sup)) for a in _alternatives(sub))


_SUBTYPES = {("bool", "int"), ("int", "float"), ("bool", "float")}


def _one(a, b):
    a, b = _base(a), _base(b)
    if "None" in (a, b):                      # `x = None` then `x = 1` is the Optional idiom, not a conflict
        return True
    if a == b or "Any" in (a, b) or "object" in (a, b):
        return True
    return (a, b) in _SUBTYPES


def _base(t):
    """`list[int]` -> `list`, `int | None` kept by the caller: generic arguments are ignored on purpose."""
    return str(t).split("[")[0].strip()


def _alternatives(t):
    """`Optional[int]` / `Union[int, str]` / `int | None` -> the alternatives a value may take."""
    t = str(t).replace("typing.", "").strip()
    if t.startswith("Optional[") and t.endswith("]"):
        return _alternatives(t[9:-1]) | {"None"}
    if t.startswith("Union[") and t.endswith("]"):
        out = set()
        for part in t[6:-1].split(","):
            out |= _alternatives(part)
        return out
    if "|" in t:
        out = set()
        for part in t.split("|"):
            out |= _alternatives(part)
        return out
    return {t}
