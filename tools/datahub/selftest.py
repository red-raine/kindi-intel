"""Fixture-based checks for the data gate: every rejection reason fires on purpose, and shards round-trip.

    python -m tools.datahub.selftest
"""
import json
import os
import shutil
import tempfile

from . import coverage, gate, shard, text

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


LONG = "The quick brown fox jumps over the lazy dog while the compiler inlines a pathological branch."   # 92 chars


def t_text():
    check("clean: CRLF becomes LF", text.clean("a\r\nb") == "a\nb", repr(text.clean("a\r\nb")))
    check("clean: NUL and control bytes are removed", text.clean("a\x00b\x07c") == "abc",
          repr(text.clean("a\x00b\x07c")))
    check("clean: NFC composes", text.clean("e\u0301") == "\u00e9", repr(text.clean("e\u0301")))
    check("fold: case and whitespace collapse", text.fold("  A\t B\n\nC ") == "a b c", repr(text.fold("  A\t B\n\nC ")))
    check("fingerprint: equal under case and spacing", text.fingerprint("A b  C") == text.fingerprint("a  b c"))
    check("fingerprint: differs on content", text.fingerprint("a b c") != text.fingerprint("a b d"))
    check("ngrams: 13 words give one 13-gram", len(text.ngrams(" ".join(str(i) for i in range(13)), 13)) == 1,
          str(len(text.ngrams(" ".join(str(i) for i in range(13)), 13))))
    check("ngrams: 12 words give none", len(text.ngrams(" ".join(str(i) for i in range(12)), 13)) == 0)
    check("script: detects CJK", text.script("这是一个中文句子") == "cjk", text.script("这是一个中文句子"))
    check("script: detects latin", text.script("a plain english sentence") == "latin")


def t_gate_reasons():
    d = tempfile.mkdtemp(prefix="datahub-")
    try:
        p = os.path.join(d, "a.jsonl")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"text": LONG}) + "\n")             # accepted
            fh.write(json.dumps({"body": LONG}) + "\n")             # schema: no 'text'
            fh.write(json.dumps({"text": "too short"}) + "\n")      # short
            fh.write(json.dumps({"text": LONG}) + "\n")             # duplicate
            fh.write(json.dumps({"text": "x" * 400}) + "\n")        # long (max 200)
            fh.write("{not json\n")                                 # json
            fh.write(json.dumps([1, 2, 3]) + "\n")                  # json: not an object
        with open(p, "ab") as fb:
            fb.write(b"\xff\xfe broken\n")                           # encoding: not valid UTF-8
        g = gate.Gate(gate.Config(max_chars=200, decontam=False))
        rep = g.run(p)
        r = rep.as_dict()
        check("gate: one record accepted", r["accepted"] == 1, "accepted %d of %d" % (r["accepted"], r["lines"]))
        for reason, want in (("schema", 1), ("short", 1), ("duplicate", 1), ("long", 1), ("json", 2), ("encoding", 1)):
            check("gate: %s counted" % reason, r["reasons"][reason] == want,
                  "got %d, wanted %d" % (r["reasons"][reason], want))
    finally:
        shutil.rmtree(d)


def t_contamination():
    d = tempfile.mkdtemp(prefix="datahub-")
    try:
        ref = "def has_close_elements(numbers, threshold):\n    for i, a in enumerate(numbers):\n" \
              "        for j, b in enumerate(numbers):\n            if i != j and abs(a - b) < threshold:\n" \
              "                return True\n    return False\n"
        g = gate.Gate(gate.Config(decontam=True))
        g.set_reference(text.ngrams(ref, 13), "humaneval")
        rep = gate.Report("x")
        leak = {"text": "Here is a solution:\n" + ref + "\nThat should pass the tests and the hidden cases too."}
        check("decontam: a leaked reference solution is rejected", g.offer(leak, rep) is None,
              "reasons %s" % rep.reasons)
        rep2 = gate.Report("x")
        clean_rec = {"text": "Compose the following text about gardening in early spring and the tools involved. " * 3}
        check("decontam: unrelated text passes", g.offer(clean_rec, rep2) is not None)
    finally:
        shutil.rmtree(d)


def t_near_dup():
    d = tempfile.mkdtemp(prefix="datahub-")
    try:
        base = " ".join("word%d" % i for i in range(120))
        variant = base.replace("word50", "changed")
        g = gate.Gate(gate.Config(near_dup=True, decontam=False))
        rep = gate.Report("x")
        check("near-dup: the original is kept", g.offer({"text": base}, rep) is not None)
        check("near-dup: a one-word edit is rejected", g.offer({"text": variant}, rep) is None,
              "reasons %s" % rep.reasons)
        rep2 = gate.Report("x")
        other = " ".join("different%d" % i for i in range(120))
        check("near-dup: unrelated text is kept", g.offer({"text": other}, rep2) is not None)
    finally:
        shutil.rmtree(d)


def t_cross_source_dedup():
    d = tempfile.mkdtemp(prefix="datahub-")
    try:
        a = write_jsonl(os.path.join(d, "a.jsonl"), [{"text": LONG}])
        b = write_jsonl(os.path.join(d, "b.jsonl"), [{"text": LONG}])
        g = gate.Gate(gate.Config(decontam=False))
        ra, rb = g.run(a), g.run(b)
        check("dedup: the same text in a second file is caught", rb.accepted == 0 and rb.reasons["duplicate"] == 1,
              "second file: %d accepted, %s" % (rb.accepted, rb.reasons))
    finally:
        shutil.rmtree(d)


def t_shards():
    d = tempfile.mkdtemp(prefix="datahub-")
    try:
        chunks = [b"alpha" * 10, b"beta" * 10, b"gamma" * 10]
        with shard.ShardWriter(d, shard_bytes=80, prefix="t") as w:
            for c in chunks:
                w.write(c)
            mpath, man = w.close(dict(note="fixture"))
        ok, problems = shard.verify(d, prefix="t")
        check("shards: manifest verifies", ok and not problems, str(problems))
        check("shards: rolled over to more than one file", len(man["shards"]) > 1, "%d shards" % len(man["shards"]))
        check("shards: record count and bytes match", man["records"] == 3 and man["bytes"] == sum(len(c) for c in chunks),
              "%d records, %d bytes" % (man["records"], man["bytes"]))
        with open(os.path.join(d, "t.index.jsonl"), encoding="utf-8") as fh:
            idx = [json.loads(l) for l in fh]
        rebuilt = b"".join(shard.read(os.path.join(d, man["shards"][e["s"]]["name"]), e["o"], e["n"]) for e in idx)
        check("shards: index rebuilds the exact stream", rebuilt == b"".join(chunks), "%d bytes" % len(rebuilt))
        with open(os.path.join(d, man["shards"][0]["name"]), "r+b") as fh:
            fh.seek(0)
            b0 = fh.read(1)
            fh.seek(0)
            fh.write(bytes([b0[0] ^ 0xFF]))
        ok2, problems2 = shard.verify(d, prefix="t")
        check("shards: verify catches a flipped byte", not ok2 and problems2, str(problems2))
    finally:
        shutil.rmtree(d)


def t_empty_build():
    from . import __main__ as cli
    d = tempfile.mkdtemp(prefix="datahub-")
    try:
        p = write_jsonl(os.path.join(d, "tiny.jsonl"), [{"text": "a"}, {"text": "b"}])
        rc = cli.main(["build", p, "--out", os.path.join(d, "out"), "--no-decontam", "--min-chars", "10"])
        check("build: refusing when nothing survives returns 2", rc == 2, "exit %d" % rc)
        left = [f for f in os.listdir(os.path.join(d, "out"))] if os.path.isdir(os.path.join(d, "out")) else []
        check("build: no shards or manifest left behind on refusal", not left, "output dir holds %s" % left)
    finally:
        shutil.rmtree(d)


def t_cli_decontam_wiring():
    """End to end through the CLI, because loading a reference set is not the same as applying it.

    `audit` once loaded the references and still passed leaked solutions: the gate was constructed with
    `decontam=False` and the flag only controlled loading. The unit tests could not see that - only the CLI path could.
    """
    from . import __main__ as cli
    d = tempfile.mkdtemp(prefix="datahub-")
    try:
        # >= ngram words, because a reference shorter than the window has no n-grams to match against
        snippet = ("def leaked(a, b, c):\n    total = a + b + c\n    if total > 100:\n        return total // 2\n"
                   "    return total + 1\n")
        ref = os.path.join(d, "ref.json")
        with open(ref, "w", encoding="utf-8") as fh:
            json.dump([{"source": "humaneval", "task_id": "X/0", "code": snippet}], fh)
        src = write_jsonl(os.path.join(d, "in.jsonl"), [
            {"text": "A solution follows.\n" + snippet + "\nIt passes the tests we ran on it."},
            {"text": "Unrelated prose about the weather on the coast in late autumn and the ferry timetable. " * 2},
        ])
        report = os.path.join(d, "report.json")
        rc = cli.main(["audit", src, "--reference", ref, "--report", report, "--min-chars", "10"])
        with open(report, encoding="utf-8") as fh:
            filed = json.load(fh)["files"][0]
        check("cli: audit applies the reference it loads", filed["reasons"]["contaminated"] == 1,
              "contaminated %d, kept %d" % (filed["reasons"]["contaminated"], filed["accepted"]))
        check("cli: the clean record still passes", filed["accepted"] == 1, "kept %d" % filed["accepted"])
    finally:
        shutil.rmtree(d)


def _recs(fam, ids, source=None):
    return [{"source": source or fam.lower(), "task_id": "%s/%d" % (fam, i), "code": "pass"} for i in ids]


def t_coverage():
    """A reference set is only worth what it covers: ids, not records, and the published sizes as the claim."""
    full = (_recs("HumanEval", range(164)) + _recs("mbpp", range(1, 511)) +
            _recs("mbpp_s", range(511, 975), "mbpp_sanitized"))
    short = (_recs("HumanEval", range(164)) + _recs("mbpp", range(11, 511)) +
             _recs("mbpp_s", range(601, 975), "mbpp_sanitized"))
    expected, named = coverage.claims()
    cov = coverage.check(full, expected, named)
    mb = [r for r in cov.rows if r["family"] == "mbpp"][0]
    check("coverage: a complete set passes, its MBPP ids counted across two layers",
          cov.ok and mb["present"] == 974 and mb["sources"] == {"mbpp": 510, "mbpp_sanitized": 464},
          "%s, sources %s" % (cov.failure(), mb["sources"]))

    cov = coverage.check(short, expected, named)
    mb = [r for r in cov.rows if r["family"] == "mbpp"][0]
    check("coverage: the 100 missing MBPP ids are reported as ranges",
          not cov.ok and mb["missing"] == 100 and mb["gaps"] == "1-10, 511-600" and cov.missing == 100,
          cov.failure())
    check("coverage: the failure names the family and the count",
          cov.failure().startswith("mbpp: 100 of 974 ids missing (1-10, 511-600)"), cov.failure())

    dup = short + _recs("mbpp", [11, 11])
    cov = coverage.check(dup, expected, named)
    mb = [r for r in cov.rows if r["family"] == "mbpp"][0]
    check("coverage: a repeated id is counted but does not pad the coverage",
          mb["present"] == 874 and mb["missing"] == 100 and mb["duplicates"] == 2,
          "present %d, duplicates %d" % (mb["present"], mb["duplicates"]))

    cov = coverage.check(full + [{"task_id": "not-a-task"}], expected, named)
    check("coverage: a record with no parsable task_id fails the check",
          not cov.ok and len(cov.unparsable) == 1, cov.failure())

    cov = coverage.check(_recs("mbpp", range(1, 975)), expected, named)
    check("coverage: a default claim skips a family the set does not mention",
          cov.ok and [r["family"] for r in cov.rows] == ["mbpp"], str([r["family"] for r in cov.rows]))
    exp_h, named_h = coverage.claims(["humaneval=164"])
    cov = coverage.check(_recs("mbpp", range(1, 975)), exp_h, named_h)
    check("coverage: a family claimed but not present is a failure",
          not cov.ok and cov.rows[0]["family"] == "humaneval" and cov.rows[0]["present"] == 0, cov.failure())

    exp, named = coverage.claims()
    check("coverage: the default claims are the published sizes",
          exp == {"humaneval": (0, 163), "mbpp": (1, 974)} and not named, str(exp))
    exp, named = coverage.claims(["mbpp=974", "humaneval=0:163"])
    check("coverage: --expect takes a count and an explicit range",
          exp["mbpp"] == (1, 974) and exp["humaneval"] == (0, 163) and named == {"mbpp", "humaneval"}, str(exp))
    refused = True
    for bad in (["mbpp"], ["mbpp=many"]):
        try:
            coverage.claims(bad)
        except SystemExit:
            continue
        refused = False
    check("coverage: a malformed --expect is refused rather than guessed", refused)


def t_cli_reference_coverage():
    """The gate must refuse a reference it cannot trust, through the same CLI a caller uses."""
    from . import __main__ as cli
    d = tempfile.mkdtemp(prefix="datahub-")
    try:
        ref = os.path.join(d, "ref.json")
        with open(ref, "w", encoding="utf-8") as fh:
            json.dump([{"source": "humaneval", "task_id": "HumanEval/0",
                        "code": "def f(a, b):\n    return a + b\n"}], fh)
        src = write_jsonl(os.path.join(d, "in.jsonl"), [{"text": LONG}])
        base = ["audit", src, "--reference", ref, "--min-chars", "10"]
        rc = cli.main(base)
        check("cli: a reference short of its claim exits 2", rc == 2, "exit %d" % rc)
        rc = cli.main(base + ["--allow-short-reference"])
        check("cli: --allow-short-reference proceeds with the gap recorded", rc == 0, "exit %d" % rc)
        rc = cli.main(base + ["--expect", "humaneval=1"])
        check("cli: --expect restates the claim, and a set that meets it passes", rc == 0, "exit %d" % rc)

        out = os.path.join(d, "out")
        rc = cli.main(["build", src, "--out", out, "--reference", ref, "--min-chars", "10"])
        left = sorted(os.listdir(out)) if os.path.isdir(out) else []
        check("cli: build refuses a short reference and writes nothing", rc == 2 and not left,
              "exit %d, %s" % (rc, left))
        rc = cli.main(["build", src, "--out", out, "--reference", ref, "--min-chars", "10",
                       "--allow-short-reference"])
        with open(os.path.join(out, "shard.manifest.json"), encoding="utf-8") as fh:
            cov = json.load(fh)["gate"]["reference_coverage"]
        check("cli: an allowed gap is recorded in the manifest",
              rc == 0 and cov and cov["ok"] is False and cov["missing"] == 163,
              "exit %d, %s" % (rc, cov and (cov["ok"], cov["missing"])))
    finally:
        shutil.rmtree(d)


def selftest():
    del PASS[:]
    for t in (t_text, t_gate_reasons, t_contamination, t_near_dup, t_cross_source_dedup, t_shards, t_empty_build,
              t_cli_decontam_wiring, t_coverage, t_cli_reference_coverage):
        t()
    expected = 47
    ok = len(PASS) == expected
    print("\n%d checks passed%s" % (len(PASS), "" if ok else " of %d - a check did not run" % expected))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
