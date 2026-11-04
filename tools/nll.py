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
    python -m tools.nll --pack A --against B --corpus DIR   # paired: the same tokens, two packs, one interval
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
import math
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


def nll_of(pack, ids, ctx=0, tokens=0, log=None, keep=False, prefill=0):
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
    if prefill and prefill >= ctx:
        raise SystemExit("tools.nll: --prefill %d needs a context longer than itself (ctx %d): the position has to fit "
                         "in one window, or the sequence restarts inside it" % (prefill, ctx))
    e = Engine(pack, ctx=ctx, log=log or (lambda *a: None))
    total, scored = 0.0, 0
    trace = [] if keep else None
    t0 = time.time()
    e.reset()
    for start in range(0, len(ids) - 1, ctx - 1):
        window = ids[start:start + ctx]
        if len(window) < 2:
            break
        if start:
            e.reset()
        e.step(window[0], 0)
        for i in range(1, len(window)):
            if start + i < prefill:
                # context to be consumed but not scored: `feed` runs the same forward pass without the head, which is
                # what makes a walk that starts past 8k positions affordable
                if i + 1 < len(window):
                    e.feed(window[i], i)
                continue
            z = np.asarray(e.logits[:e.ylog_n], dtype=np.float64)
            lse = float(z.max() + np.log(np.exp(z - z.max()).sum()))
            lp = -float(z[window[i]] - lse)
            total += lp
            scored += 1
            if keep:
                trace.append((start + i, window[i], lp))
            if i + 1 < len(window):
                e.step(window[i], i)
    # the vocabulary the walk scored against: its logarithm is chance, and a pack with fewer tokens than the 248,320
    # one must be told its own chance rather than that one
    out = {"nll": total / max(scored, 1), "tokens": scored, "seconds": time.time() - t0, "vocab": int(e.ylog_n)}
    if keep:
        out["trace"] = trace
    return out


def walk(pack, corpus, tokens=0, ctx=0, log=print, prefill=0):
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
        r = nll_of(pack, ids, ctx=ctx, tokens=tokens, log=None, prefill=prefill)
        per.append({"domain": domain, "nll": r["nll"], "tokens": r["tokens"], "vocab": r["vocab"]})
        log("  %-12s %-28s nll %.4f over %d tokens (%.1fs)"
            % (domain, os.path.basename(path), r["nll"], r["tokens"], r["seconds"]))
    if not per:
        raise SystemExit("tools.nll: nothing scorable in %s" % corpus)
    by_domain, weights = {}, {}
    for r in per:
        by_domain.setdefault(r["domain"], []).append(r)
    out = {"code_sha256": sha256_file(CODE), "corpus_sha256": sha256_texts(texts),
           "vocab": per[0]["vocab"], "heldout_tokens": sum(r["tokens"] for r in per),
           "heldout_nll": sum(r["nll"] * r["tokens"] for r in per) / sum(r["tokens"] for r in per),
           "heldout_by_domain": {}}
    for d, rows in sorted(by_domain.items()):
        n = sum(r["tokens"] for r in rows)
        out["heldout_by_domain"][d] = {"nll": sum(r["nll"] * r["tokens"] for r in rows) / n, "tokens": n}
    return out


def bootstrap_ci(d, block=32, iters=2000, seed=7, level=0.95):
    """percentile interval for the mean of `d`, resampling contiguous blocks rather than single tokens

    Per-token contributions are not independent: positions inside a window share activations, and a block whose scale
    was fitted badly moves every token after it in the same direction. A token-level standard error treats 4,000
    strongly correlated numbers as 4,000 observations and reports an interval far too narrow to mean anything. A
    moving-block bootstrap keeps the correlation inside each block (`block` tokens) and resamples blocks, so the width
    it reports is the width implied by the number of effectively independent stretches in the corpus. Seed fixed: the
    interval must be the same number on a re-run, or it is not a gate.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n == 0:
        return (float("nan"), float("nan"))
    block = max(1, min(int(block), n))
    rng = np.random.RandomState(seed)
    nb = max(1, n // block)
    off = np.arange(block)
    means = np.empty(iters)
    for i in range(iters):
        s = rng.randint(0, max(1, n - block + 1), size=nb)
        means[i] = d[np.clip((s[:, None] + off[None, :]).ravel(), 0, n - 1)].mean()
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100.0, (1 + level) / 2 * 100.0])
    return (float(lo), float(hi))


def _band_rows(ta, tb, bands, block):
    """the paired walk cut into position bands: each pack's own nats and the difference, per band

    A single number over a walk that starts at 7k and ends at 10k averages positions inside the context a model was
    trained for with positions far outside it, and those are the two numbers a rope change is supposed to separate.
    """
    rows = []
    for lo, hi in zip(bands, bands[1:]):
        pairs = [(x, y) for x, y in zip(ta, tb) if lo <= x[0] < hi]
        if not pairs:
            continue
        a = np.array([x[2] for x, _y in pairs], dtype=np.float64)
        b = np.array([y[2] for _x, y in pairs], dtype=np.float64)
        d = b - a
        clo, chi = bootstrap_ci(d, block=block)
        rows.append({"from": int(lo), "to": int(hi), "tokens": int(d.size),
                     "nll_a": float(a.mean()), "nll_b": float(b.mean()),
                     "delta": float(d.mean()), "ci_lo": clo, "ci_hi": chi})
    return rows


def paired(a_pack, b_pack, corpus, ctx=0, tokens=0, block=32, prefill=0, log=print, bands=None):
    """the same corpus through two packs, subtract token by token, and put an interval on the mean difference

    Both packs are walked over one tokenisation and one set of windows, so the difference at a token is the price of
    whatever differs between the packs at that token - here a block scale's encoding - with the corpus, the
    tokenisation and the arithmetic of everything else held fixed. `block` is the bootstrap block in tokens.

    `bands` is a list of position boundaries, e.g. `[7168, 8192, 9216, 10240]`: the same walk is then also reported
    band by band, so `[8192, ...]` answers what a change costs *past* a model's trained window rather than on average
    over a walk that is mostly inside it.
    """
    rows, pooled, tokens_seen, by_band = [], [], 0, []
    for domain, path in corpus_files(corpus):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        ids = tokenise(a_pack, text)
        if ids != tokenise(b_pack, text):
            raise SystemExit("tools.nll: %s and %s tokenise %s differently; pair packs of one vocabulary"
                             % (a_pack, b_pack, os.path.basename(path)))
        if len(ids) < 2:
            continue
        ra = nll_of(a_pack, ids, ctx=ctx, tokens=tokens, keep=True, prefill=prefill)
        rb = nll_of(b_pack, ids, ctx=ctx, tokens=tokens, keep=True, prefill=prefill)
        if [t[:2] for t in ra["trace"]] != [t[:2] for t in rb["trace"]]:
            raise SystemExit("tools.nll: the two walks scored different tokens in %s" % os.path.basename(path))
        d = np.array([y[2] - x[2] for x, y in zip(ra["trace"], rb["trace"])], dtype=np.float64)
        lo, hi = bootstrap_ci(d, block=block)
        rows.append({"domain": domain, "file": os.path.basename(path), "tokens": int(d.size),
                     "nll_a": ra["nll"], "nll_b": rb["nll"], "delta": float(d.mean()),
                     "ci_lo": lo, "ci_hi": hi, "favours_b": int((d < 0).sum()), "favours_a": int((d > 0).sum()),
                     "worst": float(d.max()), "best": float(d.min())})
        pooled.append(d)
        tokens_seen += d.size
        log("  %-10s %-24s a %.4f, b %.4f, delta %+.5f  [%+.5f, %+.5f]"
            % (domain, os.path.basename(path)[:24], ra["nll"], rb["nll"], d.mean(), lo, hi))
        if bands:
            brows = _band_rows(ra["trace"], rb["trace"], bands, block)
            rows[-1]["bands"] = brows
            for br in brows:
                log("    %5d-%-5d %5d tokens  a %.4f, b %.4f, delta %+.5f  [%+.5f, %+.5f]"
                    % (br["from"], br["to"], br["tokens"], br["nll_a"], br["nll_b"], br["delta"],
                       br["ci_lo"], br["ci_hi"]))
            by_band.extend((br["from"], br["to"],
                            np.array([y[2] - x[2] for x, y in zip(ra["trace"], rb["trace"])
                                      if br["from"] <= x[0] < br["to"]], dtype=np.float64)) for br in brows)
    if not rows:
        raise SystemExit("tools.nll: nothing scorable in %s" % corpus)
    alld = np.concatenate(pooled)
    lo, hi = bootstrap_ci(alld, block=block)
    out = {"a": os.path.abspath(a_pack), "b": os.path.abspath(b_pack), "block": int(block),
           "corpus_sha256": sha256_texts([(r["domain"], r["file"]) for r in rows]),
           "code_sha256": sha256_file(CODE), "tokens": tokens_seen, "rows": rows,
           "delta": float(alld.mean()), "ci_lo": lo, "ci_hi": hi,
           "favours_b": int((alld < 0).sum()), "favours_a": int((alld > 0).sum()),
           "excludes_zero": bool(lo > 0 or hi < 0)}
    if bands and by_band:
        # one row per boundary pair across every file in the corpus: the same positions, pooled
        summaries = []
        for lo, hi in [(int(a), int(b)) for a, b in zip(bands, bands[1:])]:
            d = np.concatenate([seg for a, b, seg in by_band if (a, b) == (lo, hi)])
            per = [row["bands"][i] for row in rows for i, br in enumerate(row.get("bands") or [])
                   if (br["from"], br["to"]) == (lo, hi)]
            if d.size == 0 or not per:
                continue
            n = d.size
            clo, chi = bootstrap_ci(d, block=block)
            summaries.append({"from": lo, "to": hi, "tokens": int(n),
                              "nll_a": sum(p["nll_a"] * p["tokens"] for p in per) / max(1, sum(p["tokens"] for p in per)),
                              "nll_b": sum(p["nll_b"] * p["tokens"] for p in per) / max(1, sum(p["tokens"] for p in per)),
                              "delta": float(d.mean()), "ci_lo": clo, "ci_hi": chi})
        out["bands"] = {"boundaries": [int(b) for b in bands], "rows": summaries}
        for s in summaries:
            log("  band %5d-%-5d %5d tokens  a %.4f, b %.4f, delta %+.5f  [%+.5f, %+.5f]"
                % (s["from"], s["to"], s["tokens"], s["nll_a"], s["nll_b"], s["delta"], s["ci_lo"], s["ci_hi"]))
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
    from kindi.cpu import Engine
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
    # the trace stores each token's own contribution to the NLL, -log p(token | prefix), which is a count of nats and
    # so positive; a negative one would be a probability above 1
    check("every scored token contributes a finite, non-negative number of nats",
          all(lp >= 0 and np.isfinite(lp) for _p, _t, lp in r["trace"]))
    check("the reported nll is the mean of those per-token contributions",
          abs(float(np.mean([lp for _p, _t, lp in r["trace"]])) - r["nll"]) < 1e-12,
          "nll %.6f" % r["nll"])

    # the wiring, at the one position where the fixture is stable: two tokens is one head evaluation, and it must
    # agree with the same computation done by hand on the engine's own buffer.
    e = Engine(pack, ctx=16, log=lambda *a: None)
    e.reset()
    e.step(ids[0], 0)
    z = np.asarray(e.logits[:e.ylog_n], dtype=np.float64)
    hand = -(z[ids[1]] - (z.max() + np.log(np.exp(z - z.max()).sum())))
    one = nll_of(pack, ids[:2], ctx=16, keep=True)
    check("a two-token walk scores the step the engine actually took",
          abs(one["nll"] - hand) < 1e-9, "walk %.9f, hand %.9f" % (one["nll"], hand))

    # Embeddings-in: a float row where the token id would go. `embed_row` hands back the row the lookup feeds, before
    # the fold, so injecting it must reproduce that token's logits bit for bit - a fold applied twice, skipped, or to a
    # row read at the wrong stride would all show up here. The default synthpack carries the Hadamard rotation (HB > 0),
    # so this is the fold path and not the identity the unfolded pack below takes.
    e = Engine(pack, ctx=16, log=lambda *a: None)
    row_x, row_y = e.embed_row(ids[0]), e.embed_row(ids[1])
    check("an embedding row is the model's own width, read without folding",
          row_x.shape == (e.m.D,) and e.m.fold_on == 1 and e.m.HB > 0,
          "shape %s, fold_on %d, block %d" % (row_x.shape, e.m.fold_on, e.m.HB))
    e.reset()
    e.step(ids[0], 0)
    by_id = e.logits[:e.ylog_n].copy()
    e.reset()
    e.step_emb(row_x, 0)
    by_row = e.logits[:e.ylog_n].copy()
    check("an injected row reproduces the token id's logits exactly", np.array_equal(by_id, by_row),
          "max |d| %.3g" % float(np.abs(by_id.astype(np.float64) - by_row).max()))
    e.reset()
    e.step_emb(row_y, 0)
    other = e.logits[:e.ylog_n].copy()
    check("a second row is a second input, not the first one cached",
          not np.array_equal(other, by_row), "max |d| %.3g" % float(np.abs(other - by_row).max()))
    # and in sequence: the same position, the same history, the same token
    e.reset()
    e.step(ids[0], 0)
    tok_id = e.step(ids[1], 1)
    l_id = e.logits[:e.ylog_n].copy()
    e.reset()
    e.step(ids[0], 0)
    tok_emb = e.step_emb(e.embed_row(ids[1]), 1)
    l_emb = e.logits[:e.ylog_n].copy()
    check("an injected row at position 1 takes the same step as the id",
          tok_id == tok_emb and np.array_equal(l_id, l_emb), "tokens %d and %d" % (tok_id, tok_emb))
    # a prefix of rows builds the same state as a prefix of ids (feed, not step: no head until the scored token)
    e.reset()
    e.feed(ids[0], 0)
    e.feed(ids[1], 1)
    ptok_id = e.step(ids[2], 2)
    p_id = e.logits[:e.ylog_n].copy()
    e.reset()
    e.feed_emb(row_x, 0)
    e.feed_emb(row_y, 1)
    ptok_emb = e.step_emb(e.embed_row(ids[2]), 2)
    check("a prefix fed as rows leaves the state a prefix of ids would",
          ptok_id == ptok_emb and np.array_equal(p_id, e.logits[:e.ylog_n]), "tokens %d and %d" % (ptok_id, ptok_emb))
    for bad, why in ((np.zeros(3, dtype=np.float32), "a row of the wrong width"),
                     (np.zeros((e.m.D, 2), dtype=np.float32), "a two-dimensional row")):
        try:
            e.step_emb(bad, 0)
        except ValueError:
            check("%s is refused" % why, True)
        else:
            check("%s is refused" % why, False)
    try:
        e.step_emb(row_x, e.ctx)
    except OverflowError:
        check("a position past the context is refused by step_emb, as by step", True)
    else:
        check("a position past the context is refused by step_emb, as by step", False)

    # a window shorter than the sequence must not change the tokens that fit in it
    short = nll_of(pack, ids[:16], ctx=32)
    check("a window scores exactly the tokens it holds", short["tokens"] == 15, "%d" % short["tokens"])

    # A pack of the shape the 1-bit Qwen3 files convert into: no Hadamard rotation in the weights, no per-head q gate,
    # and a vocabulary that is not a multiple of 8 (so the head's rows are padded past it). The engine used to divide by
    # the fold block size in embed() on every token, which a pack with no block size made fatal, and it read q at a
    # stride of 2*head_dim, which a dense q projection does not have: both are visible here as a load or a step that
    # does not come back.
    dpack = synthpack.build(tempfile.mkdtemp(prefix="nll-"), log=lambda *a: None,
                            dense=True, fold=False, vocab=130)[0]
    d = Engine(dpack, ctx=16, log=lambda *a: None)
    check("an unfolded dense pack loads with no fold and no fold block",
          d.m.fold_on == 0 and d.m.HB == 0, "fold_on %d, block %d" % (d.m.fold_on, d.m.HB))
    check("the q matrix's own rows set the per-head stride, and a dense q has none of the gate's half",
          d.m.qhd == d.m.HD, "%d of %d" % (d.m.qhd, d.m.HD))
    check("the softmax scale is 1/sqrt(head_dim)",
          abs(d.m.qscale - 1.0 / float(np.sqrt(d.m.HD))) < 1e-7, "%.6f" % d.m.qscale)
    check("the head scores the manifest's vocabulary, which need not be a multiple of 8",
          d.ylog_n == 130, "%d" % d.ylog_n)
    d.reset()
    d.step(ids[0], 0)
    z0 = np.asarray(d.logits[:130], dtype=np.float64)
    check("a token through the unfolded path comes back with finite logits inside the vocabulary",
          bool(np.isfinite(z0).all()) and 0 <= int(z0.argmax()) < 130, "argmax %d" % int(z0.argmax()))
    # embeddings-in on the unfolded shape: the row is the whole input, so there is no block size to divide by and the
    # injection path must not assume one
    drow = d.embed_row(ids[0])
    d.reset()
    d.step(ids[0], 0)
    du_id = d.logits[:130].copy()
    d.reset()
    d.step_emb(drow, 0)
    check("an unfolded pack takes an injected row through no fold at all",
          np.array_equal(du_id, d.logits[:130].copy()), "max |d| %.3g" % float(np.abs(du_id - d.logits[:130]).max()))

    # the prefill mode: the leading positions are consumed without the head, so a walk can start past the window the
    # model was trained on; the scored tokens must be exactly the ones after the prefix
    pf = nll_of(pack, ids, ctx=32, tokens=24, prefill=8)
    check("a prefilled walk scores exactly the tokens after the prefix", pf["tokens"] == 16, "%d tokens" % pf["tokens"])

    # rope scaling is read off the manifest (`kindi.rope`): a pack that asks for YaRN must give the engine the YaRN
    # table, not the plain one, or the file's own positional encoding is silently replaced by a different one
    from kindi import rope
    yarn = {"type": "yarn", "factor": 4.0, "original_context_length": 8192}
    rpack = synthpack.build(tempfile.mkdtemp(prefix="nll-"), log=lambda *a: None, rope_scaling=yarn)[0]
    reng = Engine(rpack, ctx=64, log=lambda *a: None)
    rh = reng.dims["rope_dim"] // 2
    got = np.ctypeslib.as_array(reng.m.cos_t, shape=(64, rh))
    check("a pack that asks for YaRN gets the YaRN table in the kernel",
          np.allclose(got, rope.tables(64, reng.dims["rope_dim"], reng.dims["rope_base"], yarn)[0], atol=1e-7)
          and not np.allclose(got, rope.tables(64, reng.dims["rope_dim"], reng.dims["rope_base"])[0], atol=1e-3))

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

    # the paired mode: the same tokens through two packs, subtracted token by token. Against itself every difference
    # must be exactly zero, so a sign error, a window misalignment or a stale buffer cannot pass as a small delta.
    paired_corpus = _write(tmp, "p", "a corpus long enough to fill one window of this fixture twice over")
    pr = paired(pack, pack, paired_corpus, ctx=24, log=lambda *x: None)
    check("a pack against itself differs by exactly zero on every token",
          pr["delta"] == 0.0 and pr["ci_lo"] == 0.0 and pr["ci_hi"] == 0.0 and pr["excludes_zero"] is False,
          "%g over %d tokens" % (pr["delta"], pr["tokens"]))

    # the position bands: a band is a slice of the same walk, so within one window the slices must add up to the walk
    # and a self-pair must be zero in each. A band that lost or double-counted a position would still average to
    # something plausible. The corpus is deliberately short enough for one window: a band is a range of absolute
    # positions, and a walk of several windows revisits the same positions, so a band would then cover one window's
    # worth of them rather than all of them.
    pb = paired(pack, pack, _write(tmp, "pb", "short window only"),
                ctx=24, log=lambda *x: None, bands=[0, 4, 8, 24])
    brows = (pb.get("bands") or {}).get("rows") or []
    check("the bands partition the scored tokens, and a self-pair is zero inside every band",
          len(brows) >= 2 and sum(r["tokens"] for r in brows) == pb["tokens"]
          and all(r["delta"] == 0.0 and r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0 for r in brows),
          "%d bands, %d of %d tokens" % (len(brows), sum(r["tokens"] for r in brows), pb["tokens"]))

    # the interval: it must contain the mean it is an interval for, and it must widen when the resampled unit is a block
    # rather than a token - correlated rows have fewer independent observations than they have rows, which is the whole
    # reason a token-level interval is not quoted for this gate
    rng = np.random.RandomState(1)
    iid = rng.normal(0.01, 1.0, 4096)
    ar, x = np.empty(4096), 0.0
    for i in range(4096):
        x = 0.9 * x + rng.normal(0.0, np.sqrt(1.0 - 0.81))
        ar[i] = x
    lo_i, hi_i = bootstrap_ci(iid, block=1)
    lo_t, hi_t = bootstrap_ci(ar, block=1)
    lo_b, hi_b = bootstrap_ci(ar, block=32)
    check("the interval contains the mean it was computed for",
          lo_i <= iid.mean() <= hi_i and lo_t <= ar.mean() <= hi_t and lo_b <= ar.mean() <= hi_b)
    check("a correlated series gets a wider interval when the block is the unit",
          (hi_b - lo_b) > 1.5 * (hi_t - lo_t), "%.4f against %.4f" % (hi_b - lo_b, hi_t - lo_t))

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
        raise SystemExit("usage: python -m tools.nll --pack PACK --corpus FILE|DIR [--tokens N] [--json OUT] [--reference REF]\n"
                         "       python -m tools.nll --pack A --against B --corpus DIR [--ctx N] [--block N] [--json OUT]\n"
                         "       python -m tools.nll --pack A --against B --corpus DIR --ctx 10240 --prefill 7168\n"
                         "                         --bands 7168,8192,9216,10240     # past-window bands of one walk")
    if "--against" in argv:
        b = _arg(argv, "--against")
        print("pack A %s\npack B %s, corpus %s, ctx %s, block %s tokens\n"
              % (pack, b, corpus, _arg(argv, "--ctx", "0"), _arg(argv, "--block", "32")))
        bands = [int(x) for x in _arg(argv, "--bands", "").split(",") if x.strip()] if "--bands" in argv else None
        pr = paired(pack, b, corpus, ctx=int(_arg(argv, "--ctx", 0)), tokens=int(_arg(argv, "--tokens", 0)),
                    block=int(_arg(argv, "--block", 32)), prefill=int(_arg(argv, "--prefill", 0)),
                    log=lambda *a: print(*a, flush=True), bands=bands)
        print("\nmean delta (B - A) %+.5f nats over %d tokens, 95%% block-bootstrap interval [%+.5f, %+.5f]"
              % (pr["delta"], pr["tokens"], pr["ci_lo"], pr["ci_hi"]))
        print("%s: B costs %s than A, and the interval %s zero"
              % ("clear" if pr["excludes_zero"] else "not clear",
                 "more" if pr["delta"] > 0 else "less", "excludes" if pr["excludes_zero"] else "covers"))
        print("tokens where B is closer: %d, where A is: %d" % (pr["favours_b"], pr["favours_a"]))
        if "--json" in argv:
            with open(_arg(argv, "--json"), "w") as f:
                json.dump(pr, f, indent=1, sort_keys=True)
            print("\nwrote %s" % _arg(argv, "--json"))
        return 0
    print("pack %s, corpus %s\n" % (pack, corpus))
    out = walk(pack, corpus, tokens=int(_arg(argv, "--tokens", 0)), ctx=int(_arg(argv, "--ctx", 0)),
               prefill=int(_arg(argv, "--prefill", 0)))
    print("\nheld-out NLL %.4f over %d tokens" % (out["heldout_nll"], out["heldout_tokens"]))
    for d, r in sorted(out["heldout_by_domain"].items()):
        print("  %-12s %.4f over %d tokens" % (d, r["nll"], r["tokens"]))
    print("\nchance is ln(%d) = %.2f nats: a number near it is a broken alignment, not a bad model"
          % (out["vocab"], math.log(out["vocab"])))
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
