"""What the AST says about assignments, without following any value through a call.

Two kinds of fact come out of a module:

  edges   `lhs = rhs` where both sides are a name or an attribute chain: the union-find joins the two locations
  labels  a definite type claim about a location, from an annotation (a variable's or a parameter's), a literal (or a
          literal-shaped expression), a builtin constructor, or an `isinstance` narrowing

Everything else - a call result, an iteration variable, a subscript, a value produced by a function - is left with no
claim at all. That is deliberate: a missing claim becomes UNKNOWN, which the score counts, while a guessed claim would
become a contradiction or a false clean result.
"""
import ast
from dataclasses import dataclass


@dataclass
class Label:
    key: str
    type: str
    line: int
    col: int
    source: str                     # "annotation" | "value" | "isinstance" | "pyright"


@dataclass
class Import:
    """`from module import name as alias`: which local name stands for which module-level name elsewhere."""
    module: str
    level: int                      # dots before the module name: `from .cpu import x` is level 1
    name: str
    alias: str
    line: int


@dataclass
class Facts:
    file: str
    edges: list
    labels: list
    sites: dict                     # key -> [(line, col)] of its assignment targets
    at: dict                        # (line, col) -> key, to place a pyright diagnostic on a target
    notes: list
    imports: list = None            # [Import]; None only for a Facts built by hand


CONST_TYPES = {type(None): "None", bool: "bool", int: "int", float: "float", complex: "complex", str: "str",
               bytes: "bytes"}
CTORS = {"int", "str", "float", "bool", "bytes", "list", "dict", "set", "tuple", "complex"}


def display(key):
    """The name a report shows: scope prefix and attribute marker dropped."""
    return key.split("::")[-1].lstrip("@")


class Collector(ast.NodeVisitor):
    def __init__(self, file):
        self.file = file
        self.scopes = []
        self.edges = []
        self.labels = []
        self.sites = {}
        self.at = {}
        self.notes = []
        self.imports = []

    # ------------------------------------------------------------------ naming
    def scope(self):
        return ".".join(self.scopes) if self.scopes else "mod"

    def key(self, node):
        if isinstance(node, ast.Name):
            return self.scope() + "::" + node.id
        if isinstance(node, ast.Attribute):
            chain = self.chain(node)
            return "@" + chain if chain else None
        return None

    def chain(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self.chain(node.value)
            return base + "." + node.attr if base else None
        return None

    # ------------------------------------------------------------------ scopes
    def _function(self, node):
        self.scopes.append(node.name)
        args = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args) + list(node.args.kwonlyargs)
        for arg in args:                                  # `def f(x: int)` states x's type as plainly as `x: int =`
            text = type_text(arg.annotation) if arg.annotation is not None else None
            if text:
                key = self.scope() + "::" + arg.arg
                self.labels.append(Label(key, text, arg.lineno, arg.col_offset, "annotation"))
        for s in node.body:
            self.visit(s)
        self.scopes.pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_ClassDef(self, node):
        self.scopes.append(node.name)
        for s in node.body:
            self.visit(s)
        self.scopes.pop()

    def visit_Lambda(self, node):
        return                           # no assignment statements inside a lambda

    # ------------------------------------------------------------------ imports
    def visit_ImportFrom(self, node):
        for alias in node.names:                     # `from .cpu import is_expert as ie`
            self.imports.append(Import(node.module or "", node.level, alias.name,
                                       alias.asname or alias.name, node.lineno))

    def visit_Import(self, node):
        pass                                         # `import a.b`: no module-level name is bound to a tracked key

    # ------------------------------------------------------------------ assignments
    def site(self, key, node):
        here = (node.lineno, node.col_offset)
        self.sites.setdefault(key, [])
        if here not in self.sites[key]:
            self.sites[key].append(here)
        self.at.setdefault(here, key)

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self.bind(target, node.value)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)
        key = self.key(node.target)
        if key is None:
            return
        self.site(key, node.target)
        declared = type_text(node.annotation)
        if declared:
            self.labels.append(Label(key, declared, node.target.lineno, node.target.col_offset, "annotation"))
        if node.value is not None:
            self.bind(node.target, node.value)

    def bind(self, target, value):
        key = self.key(target)
        if key is not None:
            self.site(key, target)
            got = value_type(value)
            if got:
                self.labels.append(Label(key, got, target.lineno, target.col_offset, "value"))
            if isinstance(value, (ast.Name, ast.Attribute)):
                other = self.key(value)
                if other and other != key:
                    self.edges.append((key, other))
            return
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts):
            for a, b in zip(target.elts, value.elts):
                self.bind(a, b)
            return
        if isinstance(target, ast.Starred):
            return
        self.notes.append("%s:%d: target is not a name or attribute (%s), not tracked"
                          % (self.file, target.lineno, type(target).__name__))

    # ------------------------------------------------------------------ narrowing
    def visit_If(self, node):
        self.narrow(node.test)
        self.generic_visit(node)

    def visit_While(self, node):
        self.narrow(node.test)
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.narrow(node.test)
        self.generic_visit(node)

    def narrow(self, test, positive=True):
        """`isinstance(x, T)` claims T for x in the branch it guards; `not isinstance` claims nothing."""
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            self.narrow(test.operand, not positive)
            return
        if isinstance(test, ast.BoolOp):
            for v in test.values:                       # `isinstance(x, T) and ...`: the claim still holds
                if isinstance(v, ast.Call):
                    self.narrow(v, positive)
            return
        if not (positive and isinstance(test, ast.Call)):
            return
        if not (isinstance(test.func, ast.Name) and test.func.id == "isinstance" and len(test.args) == 2):
            return
        key = self.key(test.args[0])
        kind = test.args[1]
        if not key or not isinstance(kind, ast.Name):
            return
        if kind.id not in CTORS and not kind.id[:1].isupper():        # a builtin, or a class the source names
            return
        self.labels.append(Label(key, kind.id, test.lineno, test.col_offset, "isinstance"))


def type_text(node):
    try:
        return ast.unparse(node).replace("typing.", "")
    except Exception:                                   # noqa: BLE001 - an unparseable annotation is no claim
        return None


def value_type(node):
    """The type a literal-shaped expression states outright, or None."""
    if isinstance(node, ast.Constant):
        return CONST_TYPES.get(type(node.value))
    if isinstance(node, ast.List) or isinstance(node, ast.ListComp):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Dict) or isinstance(node, ast.DictComp):
        return "dict"
    if isinstance(node, ast.Set) or isinstance(node, ast.SetComp):
        return "set"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.IfExp):
        a, b = value_type(node.body), value_type(node.orelse)
        return a if a and a == b else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return value_type(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CTORS:
        return node.func.id
    return None


def collect(source, file="<string>"):
    tree = ast.parse(source, filename=file)
    c = Collector(file)
    for node in tree.body:
        c.visit(node)
    return Facts(file, c.edges, c.labels, c.sites, c.at, c.notes, c.imports)
