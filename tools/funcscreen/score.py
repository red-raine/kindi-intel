"""The functional screen: does the continuation parse, is it self-consistent, does it satisfy the declared facts?

Three layers, and they answer different questions, so the report keeps them apart rather than folding them into one
score (the same discipline `tools.astvalid` applies to its own verdicts):

  parse    `ast.parse` over prompt + continuation. Fluent nonsense at 2-3 bpw fails here first.
  astvalid `tools.astvalid.validity.analyse_source` over the same text: TRUE when every assignment class is
           consistent, FALSE when some class is contradictory, UNKNOWN when the pass abstains. UNKNOWN is reported
           as an abstention, never as a pass - a screen that counts its own silence as success cannot reject anything.
  exec     the completed function, and only that function node, is executed against the item's declared
           input/output facts. Trailing generated text past the function is not executed: what is scored is whether
           the function the prompt asked for does what was declared.

Arithmetic items are exact-match on the first number in the continuation.

Generation is greedy (`Engine.generate`), which is deterministic: there is no sampling seed to fix, and a re-run of
the same pack produces the same outputs. Outputs are stored in the run JSON so a later arm is compared against the
*same* generations, not a fresh sample.

    python -m tools.funcscreen --pack PACK --tokens 200 --out run.json
    python -m tools.funcscreen --compare a.json b.json
    python -m tools.funcscreen --selftest
"""
import ast
import json
import os
import re
import sys

import numpy as np

from tools.astvalid import validity
from tools.astvalid.tri import Tri
from tools.funcscreen.prompts import ITEMS, sha256
from tools.nll import bootstrap_ci

CODE = "code"
ARITH = "arith"
ITERS = 2000
SEED = 7

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _parse(text):
    try:
        return ast.parse(text), None
    except SyntaxError as e:
        return None, "%s:%s" % (type(e).__name__, e)


def _parse_prefix(text, fname):
    """the longest line-prefix of `text` that parses and defines `fname`, or (None, text, err)

    A free continuation does not stop at the end of the function it was asked for: on a real pack the first lines
    complete it correctly and the rest drifts into repeated definitions and a final truncated expression, which makes
    the whole 200 tokens unparseable even when the function itself was right (measured on bonsai-2b: 15 of 20 code
    items failed whole-text parse, most with a correct function in the first lines). Scoring the whole continuation
    would punish length, not wrongness. The longest parseable prefix is the completed program the model actually
    produced; everything after the first unparseable line is drift, and it is reported as truncated, not scored.
    """
    tree, err = _parse(text)
    if tree is not None and any(isinstance(n, ast.FunctionDef) and n.name == fname for n in tree.body):
        return tree, text, False
    lines = text.splitlines(keepends=True)
    for cut in range(len(lines) - 1, 0, -1):
        prefix = "".join(lines[:cut])
        tree, _e = _parse(prefix)
        if tree is not None and any(isinstance(n, ast.FunctionDef) and n.name == fname for n in tree.body):
            return tree, prefix, True
    return None, text, err or "no parseable prefix defines %s" % fname


def _astvalid_verdict(text):
    """FALSE if any assignment class is contradictory, TRUE if classes were decided and none was, else UNKNOWN"""
    result = validity.analyse_source(text)
    tris = [f.tri for f in result.findings]
    if any(t is Tri.FALSE for t in tris):
        return "false"
    if any(t is Tri.TRUE for t in tris):
        return "true"
    return "unknown"


def _exec_facts(tree, fname, facts):
    """execute the one function node named `fname` against its declared (args, expected) facts"""
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fname), None)
    if node is None:
        return False, "no function %s defined" % fname
    module = ast.Module(body=[node], type_ignores=[])
    try:
        code = compile(module, "<funcscreen>", "exec")
    except (SyntaxError, TypeError, ValueError) as e:
        return False, "compile: %s" % e
    namespace = {"__builtins__": {"abs": abs, "min": min, "max": max, "len": len, "range": range,
                                  "enumerate": enumerate, "str": str, "int": int, "float": float,
                                  "bool": bool, "list": list, "sum": sum}}
    try:
        exec(code, namespace)  # noqa: S102 - the screen's whole job is running the candidate function
    except Exception as e:  # noqa: BLE001 - any failure of the candidate is a screen failure, with its reason kept
        return False, "exec: %s: %s" % (type(e).__name__, e)
    fn = namespace.get(fname)
    if not callable(fn):
        return False, "function %s not callable" % fname
    for args, expected in facts:
        try:
            got = fn(*args)
        except Exception as e:  # noqa: BLE001
            return False, "call %r raised %s: %s" % (args, type(e).__name__, e)
        if got != expected:
            return False, "call %r gave %r, declared %r" % (args, got, expected)
    return True, ""


def score_code(prompt, continuation, fname, facts):
    """-> dict(parse_ok, astvalid, exec_ok, passed, detail). `passed` requires all three layers to pass."""
    text = prompt + continuation
    tree, scored_text, truncated = _parse_prefix(text, fname)
    if tree is None:
        return dict(parse_ok=False, astvalid="unknown", exec_ok=False, passed=False,
                    detail="parse: %s" % truncated)
    verdict = _astvalid_verdict(scored_text)
    exec_ok, detail = _exec_facts(tree, fname, facts)
    return dict(parse_ok=True, astvalid=verdict, exec_ok=exec_ok, truncated=bool(truncated),
                passed=bool(exec_ok and verdict != "false"), detail=detail)


def score_arith(continuation, expected):
    """exact-match on the first number the continuation produces"""
    m = _NUMBER.search(continuation or "")
    if not m:
        return dict(passed=False, got=None, detail="no number in continuation")
    got = m.group(0)
    try:
        ok = float(got) == float(expected)
    except ValueError:
        ok = got == expected
    return dict(passed=bool(ok), got=got, detail="" if ok else "got %s, expected %s" % (got, expected))


def _tok(pack):
    from kindi.tokenizer import Tok
    vocab = json.load(open(os.path.join(pack, "vocab.json"), encoding="utf-8"))
    merges = os.path.join(pack, "merges.txt")
    if not os.path.isfile(merges):
        raise SystemExit("tools.funcscreen: %s has no merges.txt" % pack)
    return Tok(vocab, merges)


def run(pack, items=ITEMS, ntokens=200, ctx=None, log=print):
    """generate greedily over every item and score it: one row per item, outputs stored"""
    from kindi.cpu import Engine
    tok = _tok(pack)
    tagged = [(it, tok.encode(it[2])) for it in items]
    ctx = int(ctx) if ctx else (max(len(ids) for _it, ids in tagged) + ntokens + 8)
    eng = Engine(pack, ctx=ctx, log=lambda *a: None)
    log("pack %s, %d items (%d code, %d arithmetic), %d greedy tokens each, ctx %d"
        % (pack, len(tagged), sum(1 for it, _ in tagged if it[1] == CODE),
           sum(1 for it, _ in tagged if it[1] == ARITH), ntokens, ctx))
    rows = []
    for k, (item, ids) in enumerate(tagged):
        pid, kind, prompt, fname, facts = item
        out_ids = eng.generate(list(ids), int(ntokens))
        continuation = tok.decode(list(out_ids))
        rec = {"id": pid, "kind": kind, "prompt": prompt, "output": continuation,
               "prompt_tokens": len(ids), "generated_tokens": len(out_ids)}
        if kind == CODE:
            rec.update(score_code(prompt, continuation, fname, facts))
        else:
            s = score_arith(continuation, str(facts))
            rec.update({"got": s["got"], "passed": s["passed"], "detail": s["detail"],
                        "expected": str(facts)})
        rows.append(rec)
        log("  %-10s %-6s %s  %s" % (pid, kind, "PASS" if rec["passed"] else "fail",
                                     rec.get("detail") or ""))
        _ = k
    body = {"pack": os.path.abspath(pack), "ntokens": int(ntokens), "ctx": ctx,
            "prompt_sha256": sha256(), "rows": rows, "summary": summarize(rows)}
    return body


def summarize(rows):
    """pass rates per kind and overall, plus the astvalid abstention count: what the screen could not judge"""
    out = {}
    for kind in (CODE, ARITH, "all"):
        sub = [r for r in rows if kind == "all" or r["kind"] == kind]
        out[kind] = {"items": len(sub), "passed": sum(1 for r in sub if r["passed"]),
                     "rate": (sum(1 for r in sub if r["passed"]) / len(sub)) if sub else 0.0}
    code_rows = [r for r in rows if r["kind"] == CODE]
    out["astvalid"] = {"false": sum(1 for r in code_rows if r.get("astvalid") == "false"),
                       "true": sum(1 for r in code_rows if r.get("astvalid") == "true"),
                       "unknown": sum(1 for r in code_rows if r.get("astvalid") == "unknown"),
                       "parse_failures": sum(1 for r in code_rows if not r.get("parse_ok"))}
    return out


def compare(a_rows, b_rows, log=print):
    """paired pass-rate difference (b - a) per kind with a prompt-level bootstrap interval

    One item is one observation, resampled whole: the units are the 40 items, not the tokens inside a continuation.
    """
    by_id = {r["id"]: r for r in b_rows}
    out = {}
    for kind in (CODE, ARITH, "all"):
        pairs = [(r, by_id[r["id"]]) for r in a_rows
                 if (kind == "all" or r["kind"] == kind) and r["id"] in by_id]
        d = np.array([(1.0 if b["passed"] else 0.0) - (1.0 if a["passed"] else 0.0) for a, b in pairs],
                     dtype=np.float64)
        lo, hi = bootstrap_ci(d, block=1, iters=ITERS, seed=SEED)
        out[kind] = {"items": int(d.size), "delta": float(d.mean()) if d.size else 0.0,
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "resolvable": bool(d.size and (lo > 0.0 or hi < 0.0)),
                     "bootstrap": {"block": 1, "iters": ITERS, "seed": SEED, "unit": "item"}}
        log("  %-6s %d items, pass-rate delta %+.4f, 95%% interval [%+.4f, %+.4f] -> %s"
            % (kind, d.size, out[kind]["delta"], lo, hi,
               "resolvable" if out[kind]["resolvable"] else "not resolvable on this set"))
    return out


PASS = []
TOTAL = []


def check(name, ok, detail=""):
    TOTAL.append(name)
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def selftest():
    del PASS[:]
    from tools.funcscreen.prompts import CODE as C, ARITH as A

    # the three layers on hand-written continuations, with answers known without an engine
    good = score_code("def double(x):\n    y = 2\n    return", " y * x\n", "double", [((3,), 6), ((0,), 0)])
    check("screen: a correct completion passes parse, astvalid and exec", good["passed"] and good["parse_ok"]
          and good["astvalid"] == "true" and good["exec_ok"], str(good))

    nonsense = score_code("def add(a, b):\n    return", " the sum of the two numbers is computed by\n", "add",
                          [((2, 3), 5)])
    check("screen: fluent prose after the return fails to parse (the 2-3 bpw failure mode)",
          not nonsense["passed"] and not nonsense["parse_ok"], str(nonsense))

    wrong = score_code("def add(a, b):\n    return", " a - b\n", "add", [((2, 3), 5)])
    check("screen: parseable code that does the wrong thing fails the declared facts, not the parser",
          not wrong["passed"] and wrong["parse_ok"] and not wrong["exec_ok"], str(wrong))

    contra = score_code("def f(x):\n    y = 1\n    y = 's'\n    return", " y\n", "f", [((0,), "s")])
    check("screen: a self-contradictory assignment class is reported false by astvalid",
          contra["astvalid"] == "false" and not contra["passed"], str(contra))

    drift = score_code("def add(a, b):\n    return", " a + b\n\nthen the model keeps going and finally\nbreaks off mid-expr a +", "add",
                       [((2, 3), 5)])
    check("screen: a correct function followed by drifting, truncated text still passes (scored on the prefix)",
          drift["passed"] and drift["parse_ok"] and drift["truncated"] and drift["exec_ok"], str(drift))

    abstain = score_code("def identity(x):\n    return", " x\n", "identity", [((4,), 4)])
    check("screen: astvalid abstains (unknown) instead of guessing, and the abstention is kept distinct",
          abstain["astvalid"] == "unknown" and abstain["passed"] and abstain["exec_ok"], str(abstain))

    check("screen: arithmetic exact-match accepts the right number and refuses the wrong one",
          score_arith(" 42\n", "42")["passed"] and not score_arith(" 41\n", "42")["passed"]
          and not score_arith(" no number here\n", "42")["passed"])

    # the kill property, in miniature: a corrupted arm must score strictly worse on the same items
    import random
    rng = random.Random(0)
    items = C[:4] + A[:4]
    # hand-written correct continuations for these 8, from their prompts
    correct = {"code-01": " a + b\n", "code-02": " n % 2 == 0\n",
               "code-03": " n * factorial(n - 1)\n", "code-04": " s[::-1]\n",
               "arith-01": " 5\n", "arith-02": " 72\n", "arith-03": " 42\n", "arith-04": " 72\n"}

    def score_all(texts):
        rows = []
        for it in items:
            pid, kind, prompt, fname, facts = it
            text = texts[pid]
            if kind == CODE:
                r = score_code(prompt, text, fname, facts)
            else:
                r = score_arith(text, str(facts))
            rows.append({"id": pid, "kind": kind, "passed": r["passed"]})
        return rows

    good_rows = score_all(correct)
    corrupted = {}
    for pid, text in correct.items():
        toks = text.split()
        rng.shuffle(toks)
        corrupted[pid] = " ".join(toks) + " " + text.strip().split()[-1] + "x\n" if toks else "??\n"
    bad_rows = score_all(corrupted)
    gs, bs = summarize(good_rows), summarize(bad_rows)
    check("screen (kill property): the correct arm passes and the shuffled arm fails on the same items",
          gs["all"]["passed"] > bs["all"]["passed"] and bs["all"]["passed"] < gs["all"]["passed"],
          "correct %d/%d, corrupted %d/%d" % (gs["all"]["passed"], gs["all"]["items"],
                                               bs["all"]["passed"], bs["all"]["items"]))

    same = compare(good_rows, good_rows, log=lambda *a: None)
    check("screen: a pack against itself has a zero delta and a zero-width interval",
          same["all"]["delta"] == 0.0 and same["all"]["ci_lo"] == 0.0 and same["all"]["ci_hi"] == 0.0)

    check("screen: the item set is 40 items, 20 code and 20 arithmetic, with unique ids and a stable digest",
          len(ITEMS) == 40 and len(C) == 20 and len(A) == 20
          and len({it[0] for it in ITEMS}) == 40 and len(sha256()) == 64,
          "sha256 %s" % sha256()[:12])

    # end to end on the synthetic fixture: plumbing, stored outputs, determinism. Random weights, so the pass
    # counts say nothing about a trained model - this checks the harness, not the model.
    import tempfile
    from kindi import synthpack
    tmp = tempfile.mkdtemp(prefix="funcscreen-")
    pack, _t, _f = synthpack.build(tmp, log=lambda *a: None, width=128, ffn=256, layers=4)
    subset = [C[0], C[1], A[0], A[1]]
    first = run(pack, items=subset, ntokens=24, log=lambda *a: None)
    second = run(pack, items=subset, ntokens=24, log=lambda *a: None)
    check("screen: the run stores every output and is deterministic (greedy, no sampling)",
          [r["output"] for r in first["rows"]] == [r["output"] for r in second["rows"]]
          and len(first["rows"]) == 4 and first["prompt_sha256"] == sha256(),
          "%d rows" % len(first["rows"]))

    print("\n%d/%d checks passed" % (len(PASS), len(TOTAL)))  # noqa: E501
    return 0 if len(PASS) == len(TOTAL) else 1


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if "--compare" in argv:
        argv = list(argv)
        a = json.load(open(argv[argv.index("--compare") + 1], encoding="utf-8"))
        b = json.load(open(argv[argv.index("--compare") + 2], encoding="utf-8"))
        if a.get("prompt_sha256") != b.get("prompt_sha256"):
            raise SystemExit("tools.funcscreen: the two runs used different item sets; the comparison is void")
        print("paired comparison over %d items (sha256 %s)" % (len(a["rows"]), a["prompt_sha256"][:12]))
        out = compare(a["rows"], b["rows"])
        if "--out" in argv:
            with open(_arg(argv, "--out"), "w", encoding="utf-8") as f:
                json.dump({"a": a.get("pack"), "b": b.get("pack"),
                           "prompt_sha256": a["prompt_sha256"], "comparison": out}, f, indent=1)
        return 0
    pack = _arg(argv, "--pack")
    if not pack:
        print(__doc__.strip())
        return 2
    body = run(pack, ntokens=int(_arg(argv, "--tokens", 200)),
               ctx=int(_arg(argv, "--ctx", 0)) or None)
    s = body["summary"]
    print("\ncode %d/%d passed, arithmetic %d/%d passed, overall %d/%d"
          % (s["code"]["passed"], s["code"]["items"], s["arith"]["passed"], s["arith"]["items"],
             s["all"]["passed"], s["all"]["items"]))
    print("astvalid over code: %d true, %d false, %d unknown (abstained); %d parse failures"
          % (s["astvalid"]["true"], s["astvalid"]["false"], s["astvalid"]["unknown"],
             s["astvalid"]["parse_failures"]))
    if "--out" in argv:
        with open(_arg(argv, "--out"), "w", encoding="utf-8") as f:
            json.dump(body, f, indent=1)
            f.write("\n")
        print("wrote %s" % _arg(argv, "--out"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
