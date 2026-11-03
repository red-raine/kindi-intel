# astvalid: a three-valued validity pass over Python assignments

`tools/astvalid` answers one question about a Python codebase, three ways: **are the assignments to this value
consistent?** TRUE when the evidence says yes, FALSE when it says no, UNKNOWN when the pass has nothing definite to say.
UNKNOWN is a first-class answer - it is what a checker should say instead of guessing - and the score reports how often
it abstains.

The answer comes in **two layers**, because they are two different questions. A *node* (one assignment class) is judged
by the claims made about it in its own module. The *graph* is the conjunction over nodes plus the constraints that
imports add between them, and it can be UNSAT where every node is TRUE. See "Two layers" below.

It is a review tool, not a type checker and not part of the engine: nothing under `kindi/` imports it.

```
python -m tools.astvalid check kindi/cpu.py                 # verdicts per assignment class
python -m tools.astvalid check kindi/cpu.py --all           # include the UNKNOWN classes
python -m tools.astvalid check tools/astvalid/cases --strict # exit 1 if a contradiction is reported
python -m tools.astvalid graph kindi                       # node truth + SAT/UNSAT/UNKNOWN over the modules
python -m tools.astvalid score                              # precision/recall over the labelled sample
python -m tools.astvalid selftest                            # the logic, the union-find, the rules, both layers
```

## How a verdict is reached

**The AST** (`facts.py`) states what can be seen without following a value through a call: assignment pairs, an
annotation (on a variable or a parameter), a literal or literal-shaped expression, a builtin constructor, and an
`isinstance` guard. Calls, iteration variables and subscripts state nothing, and are left as no claim.

**The union-find** (`unionfind.py`) makes `a = b` and `b = obj.n` one location, so a contradiction found on one alias is
reported on all of them. Plain names are qualified by their enclosing function or class (`mod::x`, `f::x`), so the same
local name in two functions is two locations; attribute chains (`@obj.n`) are deliberately not qualified, because the
same attribute is the same location wherever it is written.

**pyright** (`pyright.py`) supplies the one thing an AST cannot see: types declared away from the assignment - a class
body's `count: int`, a function's return type. A diagnostic from `reportAssignmentType` is a definite negative for the
class it lands on; one from `reportAttributeAccessIssue` counts only when it names both types (`"Literal['one']" is not
assignable to "int"`), because the other form of that rule ("Attribute 'n' is unknown") says nothing about types and is
how this kind of pass earns false positives. Diagnostics are read from a live run, or from a saved
`pyright --outputjson` file, and a type pyright names joins the union-find so it reaches the value's aliases.

A class is FALSE when:

- it has an **annotation** and some other claim is not assignable to it - one way, because an annotation declares;
- with no annotation, two claims of unrelated types reach it - any relationship counts, so `bool`/`int`/`float` agree,
  and `None` is the Optional idiom rather than a conflict;
- pyright reports an assignment-type conflict on it.

An `isinstance` guard describes the value that *arrived*, so it is dropped from a location the code subsequently
assigns: `if isinstance(prompt, str): prompt = [prompt]` is normalisation, not a contradiction. Measured on `kindi/`,
that rule is the difference between 6 false positives and none.

## Two layers: node truth, graph consistency

A per-file pass cannot see the contradiction that motivated this layer, because it is not in either file:

```python
# conf.py
marker: int = 1

# use.py
from conf import marker
marker = "id"
```

`conf.marker` has one claim and `use.marker` has one; both modules are locally consistent. The import makes them one
location holding an `int` and a `str`, and there is no line to point at - the contradiction is on the *edge*. So
`graph.py` reports it as a graph finding, and both nodes keep saying TRUE:

| layer | question | answer |
|---|---|---|
| node | are this location's claims consistent *here*? | `TRUE` / `FALSE` / `UNKNOWN` |
| graph | do all nodes and all import edges hold together? | `SAT` / `UNSAT` / `UNKNOWN` |

`sat` is the Kleene conjunction of every node's truth, with each cross-module conflict counted as a FALSE:
**SAT** when every node was decided true, **UNSAT** when some node is false or two modules disagree, **UNKNOWN**
when nothing is false but something abstained. That last case is the important one: a two-valued checker has to call a
half-decided codebase clean.

What the graph layer does and does not join:

- **`from m import n` only.** The imported name and the module-level name it names become one location. Module ids are
derived the way Python derives them - by walking up while the directory has an `__init__.py` - so `from . import quant`
resolves against the package and not against wherever the command was pointed.
- **`import m` then `m.x = ...` is not joined**, and neither are attribute chains across modules: `@self.x` in two
  classes is two objects, and joining them would be a false-positive factory.
- **An import whose target was not analysed is counted as `unresolved`**, never as agreement. The graph is only as wide
  as the paths it was given.

On `kindi/` (40 modules, 3022 nodes) the layer resolves 108 of 123 imports, leaves 15 unresolved to the standard library,
and finds **0 cross-module conflicts** - with the abstention below, that is absence of evidence rather than a clean bill.

## Confidence is a pair, not a number

A verdict is stored with the evidence behind it - `support`, `against`, and `unseen` (assignment sites whose value
stated nothing at all) - and `confidence` is derived: `support / (support + against)`, or **`None`** when there was
nothing to agree about. `None` is not `0.5`: a class nobody claimed anything about is a different state of knowledge
from one whose claims split evenly, and folding either into one score is exactly the collapse this avoids. The report
therefore keeps `(unknown, 0 claims)` distinguishable from `(unknown, 4 claims)`, and an UNKNOWN node's `unseen` count
says how much of the location the pass declined to look at.

## The labelled sample and what it measures

`cases/` holds 23 snippets with `labels.json`: per case, the assignment classes that are genuinely contradictory (by the
names the code writes) and the verdict a correct pass should reach.

Matching is by (file, set of names), so a contradiction reported on the right class counts once however many lines feed
it, and one reported on a different class is a false positive rather than a lucky hit.

```
python -m tools.astvalid score                   # live pyright, if installed
python -m tools.astvalid score --pyright off      # the AST and the union-find alone
python -m tools.astvalid score --pyright cases/pyright-report.json   # replay a recorded run
```

Measured on 2026-11-02, pyright 1.1.411, Python 3.12:

| arm | precision | recall | f1 | case verdicts | abstained |
|---|---:|---:|---:|---:|---:|
| AST + union-find + pyright | 1.000 | 0.900 | 0.947 | 22/23 | 12% of 32 classes |
| AST + union-find only | 1.000 | 0.700 | 0.824 | 20/23 | 12% |

The one miss in both arms is `c14`: dict-entry values are subscripts, which the pass does not track, and the label says
so. The two extra misses without pyright are the cases whose only declared type lives in a class body (`c11`, `c22`),
which is exactly what the pyright input is for. `cases/pyright-report.json` is a recording of that run, so the score
reproduces on a machine without pyright.

On this repository's own engine (`check kindi/`, pyright on) the pass reports **1 contradiction in 3022 assignment
classes**, with 2161 abstentions: a conditional `Graph = Replay` in `kindi/spec.py` behind a module-level `class Graph`,
which pyright also flags. The six false positives the first version reported on this code were all the
`isinstance`-guard normalisation above.

The same code through the two layers, pyright off so the pass stands alone:

| | |
|---|---:|
| modules | 40 |
| nodes | 3022 |
| nodes TRUE | 860 |
| nodes UNKNOWN | 2162 (**71.5%**) |
| nodes FALSE | 0 |
| imports resolved / unresolved | 108 / 15 |
| cross-module conflicts | 0 |
| `sat` | **UNKNOWN** |

The 71.5% is the number that matters for anything built on top of this: the node layer decides 28% of locations, so the
graph is UNKNOWN almost regardless of what the graph layer can find, and any claim that a package is consistent has to
be read as "nothing was proven inconsistent". With pyright on, the same file set is UNSAT - from that one local FALSE in
`spec.py`, not from a cross-module edge.

## Limits, stated rather than papered over

- **No flow sensitivity.** A class is judged as a whole; two claims in mutually exclusive branches disagree. That is the
  intended reading (a location that holds two kinds of value is the smell), but it is stricter than pyright, which
  re-narrows unannotated names - the sample is labelled to this stricter reading and says why.
- **Subscripts and unpacking a call result are not tracked**, so dict/list element rebinding is invisible.
- **Attributes are one location per name**, not per object: `a.n` and `b.n` in different objects are the same key.
- **No type inference.** A value produced by a call is UNKNOWN unless pyright supplies the type; the pass never guesses.
- **Class-body annotations do not become attribute claims** - `class C: n: int` plus `c.n = "x"` is found by pyright, not
  by the AST half.
- The sample is small and written together with the rules, so its precision is optimistic for unseen code. The
  abstention rate and the false positives found on real code are the more portable numbers.
- **The graph layer has no true positive on real code yet.** Its seven assertions in `selftest` build the two-module case
  above, plus an agreeing pair and an abstaining one, but `kindi/` produces no cross-module conflict. So it is proven to
  separate the layers and to stay quiet; it is not yet proven to find something a per-file pass would miss.

## What is deliberately not here

The three experiments this was split from are separate, and only the first is in this directory:

- **Validator** (this tool): AST -> claims -> union-find -> node truth -> graph `sat`. Its metric is contradiction
  precision/recall against hand labels, plus the abstention rate.
- **Fingerprint router**: not built. It needs an embedding model and a rotated-vector-to-trits projection, and the
  objective there (retrieval quality) shares no data or metric with this one.
- **Quantizer**: not here either - the 2-bit weight pack and its GEMM live in
  [tools/bitnet_gemm](../bitnet_gemm/README.md). That tool's codes are also two bits, but they are a *storage* format
  for weights, not a truth value; nothing in it is logic, and nothing here is a weight.

Mixing those three into one ternary type was the part of the original design that could not be built; keeping them as
three objects with three metrics is what makes each one falsifiable on its own.
