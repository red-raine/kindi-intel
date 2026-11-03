"""kindi's dev-tool front end for the validity pass.

  python -m tools.astvalid check kindi/cpu.py                 # three-valued verdicts per assignment class
  python -m tools.astvalid check tools/astvalid/cases --strict  # exit 1 if a contradiction is reported
  python -m tools.astvalid graph kindi                         # node truth + SAT/UNSAT/UNKNOWN over the modules
  python -m tools.astvalid score                               # precision/recall against the labelled sample
  python -m tools.astvalid selftest

`--pyright` selects the diagnostics: `auto` runs pyright if it is installed, `off` runs the pass alone, and a path to a
saved `pyright --outputjson` file replays a recorded run.
"""
import argparse
import os
import re
import sys

from . import graph, pyright, validity
from . import score as scoring
from .facts import Label
from .tri import AND, NOT, OR, Tri
from .unionfind import UnionFind, assignable, verdict

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")


def diagnostics(choice, paths, log, recorded=None):
    """(diagnostics, source description). Never raises: a broken pyright degrades to the pass alone."""
    if choice == "off":
        return [], "off"
    if choice != "auto":
        return pyright.load(choice), os.path.basename(choice)
    diags, err = pyright.run(paths)
    if not err:
        return diags, "live pyright"
    log("astvalid: %s" % err)
    if recorded and os.path.isfile(recorded):
        return pyright.load(recorded), "recorded " + os.path.basename(recorded)
    log("astvalid: continuing without pyright; UNKNOWN is the answer wherever a return type mattered")
    return [], "unavailable"


def _group_notes(notes):
    """The same gap shows up hundreds of times on real code: report each kind once, with its first site."""
    groups = {}
    for note in notes:
        m = re.match(r"^[^:]+:\d+: (.+)$", note) or re.match(r"^[^:]+:\d+\s+(\(.+)$", note)
        key = re.sub(r"\s*\(\d+ of \d+\)$", "", (m.group(1) if m else note))
        groups.setdefault(key, []).append(note)
    return groups


def cmd_check(a):
    diags, source = diagnostics(a.pyright, a.paths, print)
    result = validity.analyse_paths(a.paths, diags)
    print("pyright: %s (%d diagnostics)" % (source, len(diags)))
    current = None
    shown = [f for f in result.findings if a.all or f.tri is not Tri.UNKNOWN]
    for finding in shown:
        if finding.file != current:
            current = finding.file
            print("\n%s" % (finding.path or finding.file))
        print("  %-7s %-24s %s%s" % (finding.tri, ", ".join(finding.names), finding.claim_text(),
                                     ("  [%s]" % finding.evidence[0]) if finding.evidence else ""))
    bad = sum(1 for f in result.findings if f.tri is Tri.FALSE)
    unknown = sum(1 for f in result.findings if f.tri is Tri.UNKNOWN)
    print("\n%d classes: %d contradictory, %d clean, %d unknown" % (len(result.findings), bad,
                                                                    len(result.findings) - bad - unknown, unknown))
    for pattern, examples in _group_notes(result.notes).items():
        print("note: %d x %s (first: %s)" % (len(examples), pattern, examples[0]))
    if a.json:
        import json
        print(json.dumps([dict(file=f.file, names=f.names, tri=str(f.tri), reason=f.reason) for f in result.findings],
                         indent=2))
    return 1 if (a.strict and bad) else 0


def cmd_graph(a):
    diags, source = diagnostics(a.pyright, a.paths, print)
    g = graph.analyse(a.paths, diags)
    print("pyright: %s (%d diagnostics)\n" % (source, len(diags)))
    graph.report(g)
    return 1 if (a.strict and g.sat is Tri.FALSE) else 0


def cmd_score(a):
    recorded = a.recorded or os.path.join(a.cases, "pyright-report.json")
    diags, source = diagnostics(a.pyright, [a.cases], print, recorded)
    print("pyright: %s (%d diagnostics)\n" % (source, len(diags)))
    return 0 if scoring.score(a.cases, diags)["f1"] >= a.min_f1 else 1


def cmd_selftest(a):
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    ok("Kleene AND", [AND(Tri.TRUE, Tri.UNKNOWN), AND(Tri.FALSE, Tri.UNKNOWN), AND(Tri.TRUE, Tri.TRUE)]
       == [Tri.UNKNOWN, Tri.FALSE, Tri.TRUE])
    ok("Kleene OR", [OR(Tri.FALSE, Tri.UNKNOWN), OR(Tri.TRUE, Tri.UNKNOWN), OR(Tri.FALSE, Tri.FALSE)]
       == [Tri.UNKNOWN, Tri.TRUE, Tri.FALSE])
    ok("Kleene NOT", [NOT(Tri.TRUE), NOT(Tri.FALSE), NOT(Tri.UNKNOWN)] == [Tri.FALSE, Tri.TRUE, Tri.UNKNOWN])
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    ok("union-find one class of three", uf.find("a") == uf.find("c") and len(uf.classes()) == 1)
    ok("assignability lattice", [assignable("bool", "int"), assignable("int", "float"), assignable("float", "int"),
                                 assignable("str", "int"), assignable("int", "Optional[int]"),
                                 assignable("int", "int | None"), assignable("list", "list[int]"),
                                 assignable("int", "None")]
       == [True, True, False, False, True, True, True, True])

    def verdict_of(*types):
        return verdict([Label("mod::x", t, i + 1, 0, "value") for i, t in enumerate(types)])

    def annotated(base, *types):
        return verdict([Label("mod::x", base, 1, 0, "annotation")]
                       + [Label("mod::x", t, i + 2, 0, "value") for i, t in enumerate(types)])

    ok("verdict true on one claim", verdict_of("int") is Tri.TRUE)
    ok("verdict false on a conflict", verdict_of("int", "str") is Tri.FALSE)
    ok("verdict true on related numeric types", verdict_of("int", "float") is Tri.TRUE)
    ok("verdict true on bool and int", verdict_of("bool", "int") is Tri.TRUE)
    ok("verdict true on None then a value", verdict_of("None", "int") is Tri.TRUE)
    ok("verdict unknown with no claims", verdict([]) is Tri.UNKNOWN)
    ok("annotation is one-way: int then float ok", annotated("float", "int") is Tri.TRUE)
    ok("annotation is one-way: float then int is not", annotated("int", "float") is Tri.FALSE)

    res = validity.analyse_source("x = 1\nx = 's'\ny = 2\n")
    ok("AST: literal conflict found", [f.tri for f in res.findings if f.names == ["x"]] == [Tri.FALSE])
    ok("AST: clean class found", [f.tri for f in res.findings if f.names == ["y"]] == [Tri.TRUE])
    res = validity.analyse_source("def f():\n    x = 1\ndef g():\n    x = 's'\n")
    ok("AST: same name in two functions is two locations", all(f.tri is Tri.TRUE for f in res.findings))
    res = validity.analyse_source("a = b\nb = 1\nb = 's'\n")
    ok("AST: aliasing carries the contradiction", [f.tri for f in res.findings if "a" in f.names] == [Tri.FALSE])
    res = validity.analyse_source("x = None\nx = 1\n")
    ok("AST: None then int is the Optional idiom", [f.tri for f in res.findings if f.names == ["x"]] == [Tri.TRUE])
    res = validity.analyse_source("def f(x: int):\n    x = 's'\n")
    ok("AST: a parameter annotation is a local claim", [f.tri for f in res.findings if f.names == ["x"]] == [Tri.FALSE])
    res = validity.analyse_source("x: int = 1\nx = 2\n")
    ok("AST: annotation then literal agree", [f.tri for f in res.findings if f.names == ["x"]] == [Tri.TRUE])
    res = validity.analyse_source("x = get_int()\n")
    ok("AST: no local claim is UNKNOWN, not a guess", [f.tri for f in res.findings if f.names == ["x"]] == [Tri.UNKNOWN])
    diag = pyright.Diag(file="<string>", line=2, col=4, severity="error", rule="reportAssignmentType",
                        message='Type "str" is not assignable to declared type "int"', got="str", want="int")
    res = validity.analyse_source("x = get_int()\nx = 's'\n", diags=[diag])
    ok("pyright: a diagnostic makes the class false", [f.tri for f in res.findings if f.names == ["x"]] == [Tri.FALSE])

    # the graph layer: two modules that are each locally fine and jointly contradictory
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        def write(name, text):
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                f.write(text)

        write("conf.py", "marker: int = 1\n")
        write("use.py", "from conf import marker\nmarker = 'id'\n")
        g = graph.analyse([tmp])
        nodes = {n.file: n.tri for n in g.nodes}
        ok("graph: a cross-module conflict is one conflict", len(g.conflicts) == 1)
        ok("graph: both nodes stay locally true", nodes == {"conf": Tri.TRUE, "use": Tri.TRUE})
        ok("graph: the pair is UNSAT", g.sat is Tri.FALSE)
        ok("graph: the conflict names the edge", bool(g.conflicts and g.conflicts[0].edges))

        write("use.py", "from conf import marker\nmarker = 2\n")
        g = graph.analyse([tmp])
        ok("graph: agreeing modules are SAT", g.sat is Tri.TRUE and not g.conflicts)

        write("loose.py", "value = compute()\n")
        g = graph.analyse([tmp])
        ok("graph: one abstaining node makes the graph UNKNOWN", g.sat is Tri.UNKNOWN)
        ok("graph: an unresolved import is counted, not assumed", graph.analyse([os.path.join(tmp, "use.py")]).unresolved == 1)

    ok("confidence is None when nothing was claimed",
       validity.Finding("f", [], [], Tri.UNKNOWN, "", support=0, against=0).confidence is None)
    ok("confidence keeps the pair (2 support, 1 against)",
       validity.Finding("f", [], [], Tri.TRUE, "", support=2, against=1).confidence == 2 / 3)

    for name, passed in checks:
        print("%-4s %s" % ("ok" if passed else "FAIL", name))
    bad = [name for name, passed in checks if not passed]
    print("\n%d/%d checks passed" % (len(checks) - len(bad), len(checks)))
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m tools.astvalid", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("check", help="three-valued verdicts for the assignment classes under these paths")
    s.add_argument("paths", nargs="+")
    s.add_argument("--pyright", default="auto", help="auto | off | a saved --outputjson file")
    s.add_argument("--strict", action="store_true", help="exit 1 when a contradiction is reported")
    s.add_argument("--all", action="store_true", help="list the UNKNOWN classes too")
    s.add_argument("--json", action="store_true", help="also print the findings as JSON")
    s = sub.add_parser("graph", help="node truth per module plus one SAT/UNSAT/UNKNOWN for the graph")
    s.add_argument("paths", nargs="+")
    s.add_argument("--pyright", default="auto", help="auto | off | a saved --outputjson file")
    s.add_argument("--strict", action="store_true", help="exit 1 when the graph is UNSAT")
    s = sub.add_parser("score", help="score contradiction detection against the labelled sample")
    s.add_argument("--cases", default=CASES)
    s.add_argument("--pyright", default="auto", help="auto | off | a saved --outputjson file")
    s.add_argument("--recorded", default=None, help="saved pyright output to fall back on")
    s.add_argument("--min-f1", type=float, default=0.65, dest="min_f1")
    sub.add_parser("selftest", help="check the logic, the union-find and the AST pass in isolation")
    a = p.parse_args(argv)
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "graph":
        return cmd_graph(a)
    if a.cmd == "score":
        return cmd_score(a)
    if a.cmd == "selftest":
        return cmd_selftest(a)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
