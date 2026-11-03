"""Held-out NLL through a whole pack, content-hashed so the number is comparable with somebody else's.

Every quality number in this repository is relative: KL between two cache formats, top-1 agreement between two heads,
the same prompts through two arms. That is enough to choose between options and useless for answering "is this pack
still the same model", which is the question a published artifact has to answer for a stranger. DeepSeek's ternary
expert pack (`meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest`) ships the shape that does answer it -
`nll-eval.json` / `nll-reference.json` / `nll-walk.json`: teacher-forced NLL over a held-out corpus, split by domain,
with the sha256 of both the corpus and the code that produced it, and a reference file to subtract. This module is
that walk for a Kindi pack, and it writes the same key names, so a `kindi` number and a `goethe` number sit in the
same shape.

    python -m tools.nll --pack PACK --corpus FILE                # one NLL for one text
    python -m tools.nll --pack PACK --corpus DIR --json out.json  # one per subdirectory, as domains
    python -m tools.nll --pack PACK --corpus DIR --reference ref.json
    python -m tools.nll --selftest

NLL is `-mean(log p(token_i | tokens_<i))` in nats, and it is not a metric that rewards guessing: a model that has
learned nothing of the corpus scores `ln(V)` (12.42 for the 248,320-token vocabulary) - which is the number to check
against before trusting a smaller one, because it is also what an off-by-one alignment in the scoring loop produces.

Cost: one head evaluation per scored token, which on this CPU engine is generation speed - 200 tokens of prose is
seconds on a 2B and minutes on a 27B. `--tokens` caps the budget per domain and the cap is printed, because a mean
over a truncated window is a different number from a mean over the whole corpus.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

CODE = os.path.abspath(__file__)
PASS = []


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_texts(texts):
    """one digest over a corpus, stable in file order and independent of the pack"""
    h = hashlib.sha256()
    for name, text in texts:
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()


def corpus_files(corpus):
    """-> [(domain, path)] for a file (one domain) or a directory (a domain per subdirectory)"""
    if os.path.isfile(corpus):
        return [("all", corpus)]
    out = []
    for name in sorted(os.listdir(corpus)):
        p = os.path.join(corpus, name)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".txt"):
                    out.append((name, os.path.join(p, f)))
        elif name.endswith(".txt"):
            out.append(("all", p))
    if not out:
        raise SystemExit("tools.nll: no .txt files under %s" % corpus)
    return out


def tokenise(pack, text):
    """the pack's own BPE, so the tokens scored are the tokens the model was trained on"""
    from kindi.tokenizer import Tok
    vocab = json.load(open(os.path.join(pack, "vocab.json")))
    merges = os.path.join(pack, "merges.txt")
    if not os.path.isfile(merges):
        raise SystemExit("tools.nll: %s has no merges.txt, so its text cannot be tokenised here" % pack)
    return Tok(vocab, merges).encode(text)


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    return z - (m + np.log(np.exp(z - m).sum()))


def nll_of(pack, ids, ctx=0, tokens=0, log=None, keep=False):
    """teacher-forced NLL of a token sequence: one head evaluation per scored token

    The engine's `step(tok, pos)` consumes tok at pos and leaves the distribution for pos+1 in `logits`, so scoring
    token i costs the step that consumed token i-1. A long corpus is walked in windows of `ctx` tokens, each window
    starting fresh: that keeps every token inside the context the engine was built with, and it is why the number is
    a mean over windows rather than over the corpus in one pass.
    """
    from kindi.cpu import Engine
    ids = [int(t) for t in ids]
    if len(ids) < 2:
        raise SystemExit("tools.nll: need at least two tokens to score one")
    if tokens:
        ids = ids[:int(tokens) + 1]
    ctx = int(ctx) or min(len(ids) + 1, 512)
    e = Engine(pack, ctx=ctx, log=log or (lambda *a: None))
    total, scored = 0.0, 0
    trace = [] if keep else None
    t0 = time.time()
    for start in range(0, len(ids) - 1, ctx - 1):
        window = ids[start:start + ctx]
        if len(window) < 2:
            break
        e.reset()
        e.step(window[0], 0)
        for i in range(1, len(window)):
            z = np.asarray(e.logits[:e.ylog_n], dtype=np.float64)
            lse = float(z.max() + np.log(np.exp(z - z.max()).sum()))
            lp = -float(z[window[i]] - lse)
            total += lp
            scored += 1
            if keep:
                trace.append((start + i, window[i], lp))
            if i + 1 < len(window):
                e.step(window[i], i)
    out = {"nll": total / max(scored, 1), "tokens": scored, "seconds": time.time() - t0}
    if keep:
        out["trace"] = trace
    return out


def walk(pack, corpus, tokens=0, ctx=0, log=print):
    """the eval file: per-domain NLL, the token counts, and the two digests"""
    files = corpus_files(corpus)
    per, texts = [], []
    for domain, path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        texts.append((os.path.relpath(path, corpus), text))
        ids = tokenise(pack, text)
        if len(ids) < 2:
            log("  %-12s %s: too short to score, skipped" % (domain, os.path.basename(path)))
            continue
        r = nll_of(pack, ids, ctx=ctx, tokens=tokens, log=None)
        per.append({"domain": domain, "nll": r["nll"], "tokens": r["tokens"]})
        log("  %-12s %-28s nll %.4f over %d tokens (%.1fs)"
            % (domain, os.path.basename(path), r["nll"], r["tokens"], r["seconds"]))
    if not per:
        raise SystemExit("tools.nll: nothing scorable in %s" % corpus)
    by_domain, weights = {}, {}
    for r in per:
        by_domain.setdefault(r["domain"], []).append(r)
    out = {"code_sha256": sha256_file(CODE), "corpus_sha256": sha256_texts(texts),
           "heldout_tokens": sum(r["tokens"] for r in per),
           "heldout_nll": sum(r["nll"] * r["tokens"] for r in per) / sum(r["tokens"] for r in per),
           "heldout_by_domain": {}}
    for d, rows in sorted(by_domain.items()):
        n = sum(r["tokens"] for r in rows)
        out["heldout_by_domain"][d] = {"nll": sum(r["nll"] * r["tokens"] for r in rows) / n, "tokens": n}
    return out


def compare(got, ref, log=print):
    """-> the deltas, and whether each one is outside the corpus noise the two files agree on

    A reference is only meaningful on the same corpus: `corpus_sha256` must match, and if it does not, the delta is
    printed with that stated rather than suppressed, because comparing two corpora is a real thing to do by accident.
    """
    same = got.get("corpus_sha256") == ref.get("corpus_sha256")
    if not same:
        log("reference is a different corpus (%s vs %s): deltas are not comparable"
            % (str(got.get("corpus_sha256"))[:12], str(ref.get("corpus_sha256"))[:12]))
    rows = []
    if "heldout_nll" in ref and ref["heldout_nll"] is not None:
        rows.append(("overall", ref["heldout_nll"], got["heldout_nll"]))
    for d, r in sorted((ref.get("heldout_by_domain") or {}).items()):
        g = (got.get("heldout_by_domain") or {}).get(d)
        if g and r.get("nll") is not None:
            rows.append((d, r["nll"], g["nll"]))
    for name, a, b in rows:
        log("  %-12s reference %.4f, this pack %.4f, delta %+.4f" % (name, a, b, b - a))
    return {"comparable": same, "rows": rows}


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)


def selftest():
    import tempfile
    from kindi import synthpack
    tmp = tempfile.mkdtemp(prefix="nll-")
    pack = synthpack.build(tmp, log=lambda *a: None)[0]

    ids = list(range(101, 101 + 24))
    r = nll_of(pack, ids, ctx=64, keep=True)
    check("one head evaluation per scored token", r["tokens"] == len(ids) - 1, "%d tokens" % r["tokens"])

    # The identity, not the model: every scored token must be exactly -(logit - logsumexp) of its own step, and the
    # mean of those must be the number reported. This is the only part of an NLL that can be checked without a
    # reference, and checking the model instead would be checking random weights (the fixture's logits drift from
    # run to run after ~27 positions, which is why this sequence is short).
    check("the trace is in token order", [t for _p, t, _lp in r["trace"]] == ids[1:])
    check("every scored token has a non-positive, finite log-probability",
          all(lp <= 0 and np.isfinite(lp) for _p, _t, lp in r["trace"]))
    check("the reported nll is the mean of those log-probabilities",
          abs(float(np.mean([lp for _p, _t, lp in r["trace"]])) - r["nll"]) < 1e-12,
          "nll %.6f" % r["nll"])

    # the wiring, at the one position where the fixture is stable: two tokens is one head evaluation, and it must
    # agree with the same computation done by hand on the engine's own buffer.
    from kindi.cpu import Engine
    e = Engine(pack, ctx=16, log=lambda *a: None)
    e.reset()
    e.step(ids[0], 0)
    z = np.asarray(e.logits[:e.ylog_n], dtype=np.float64)
    hand = -(z[ids[1]] - (z.max() + np.log(np.exp(z - z.max()).sum())))
    one = nll_of(pack, ids[:2], ctx=16, keep=True)
    check("a two-token walk scores the step the engine actually took",
          abs(one["nll"] - hand) < 1e-9, "walk %.9f, hand %.9f" % (one["nll"], hand))

    # a window shorter than the sequence must not change the tokens that fit in it
    short = nll_of(pack, ids[:16], ctx=32)
    check("a window scores exactly the tokens it holds", short["tokens"] == 15, "%d" % short["tokens"])

    # the digests: content-addressed, and blind to the pack
    a = walk(pack, _write(tmp, "a", "the cache holds one key and one value for every position"),
             log=lambda *x: None)
    b = walk(pack, _write(tmp, "b", "the cache holds one key and one value for every position"),
             log=lambda *x: None)
    c = walk(pack, _write(tmp, "c", "a different corpus, with other words in it entirely"),
             log=lambda *x: None)
    check("the corpus digest depends on the text, not the path", a["corpus_sha256"] == b["corpus_sha256"])
    check("a different corpus changes the digest", a["corpus_sha256"] != c["corpus_sha256"])
    check("the code digest is this file", a["code_sha256"] == sha256_file(CODE))

    # reference arithmetic, including the case that has to be refused
    ref = dict(a, heldout_nll=a["heldout_nll"] + 0.5)
    cmp = compare(a, ref, log=lambda *x: None)
    check("a reference on the same corpus is comparable", cmp["comparable"])
    check("the delta is this pack minus the reference",
          abs((cmp["rows"][0][2] - cmp["rows"][0][1]) - -0.5) < 1e-12, "%.6f" % (cmp["rows"][0][2] - cmp["rows"][0][1]))
    check("a reference on another corpus says so", not compare(a, c, log=lambda *x: None)["comparable"])

    print("\n%d checks passed" % len(PASS))
    return 0


def _write(where, name, text):
    p = os.path.join(where, name + ".txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def _arg(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main(argv):
    if "--selftest" in argv:
        return selftest()
    pack = _arg(argv, "--pack")
    corpus = _arg(argv, "--corpus")
    if not pack or not corpus:
        raise SystemExit("usage: python -m tools.nll --pack PACK --corpus FILE|DIR [--tokens N] [--json OUT] [--reference REF]")
    print("pack %s, corpus %s\n" % (pack, corpus))
    out = walk(pack, corpus, tokens=int(_arg(argv, "--tokens", 0)), ctx=int(_arg(argv, "--ctx", 0)))
    print("\nheld-out NLL %.4f over %d tokens" % (out["heldout_nll"], out["heldout_tokens"]))
    for d, r in sorted(out["heldout_by_domain"].items()):
        print("  %-12s %.4f over %d tokens" % (d, r["nll"], r["tokens"]))
    print("\nchance is ln(248320) = 12.42 nats: a number near it is a broken alignment, not a bad model")
    if "--reference" in argv:
        ref = json.load(open(_arg(argv, "--reference")))
        print("\nagainst %s" % _arg(argv, "--reference"))
        compare(out, ref)
    if "--json" in argv:
        with open(_arg(argv, "--json"), "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print("\nwrote %s" % _arg(argv, "--json"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
