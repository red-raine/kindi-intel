"""Is the 32-block scale's end-to-end gain over the head scale resolvable across the full 48 prompts?

`tools.kvq.measure` answers *how much* each V format costs in KL, and on the real pack it reports mode 4 (one scale a
head, 2.0625 bits a value) at 0.19804 nats against mode 6 (one scale per 32 values, 2.5 bits) at 0.15993 over three
40-token windows. Three windows are three numbers, not an interval, and the earlier arm table's confidence came from a
paired bootstrap over prompts that only a harness outside this tree could run. This module runs the same comparison
here: every one of the 48 prompts in `tools.kvq.prompts`, the 160 decisions that predict its 160 greedy continuation
tokens, both formats replayed over the **same** token stream, and a paired bootstrap over the 48 per-prompt
differences. The prompts are read by both arms and not scored, so the total is the same **7,680 decisions** (48 x 160)
the arm table quotes - a different protocol would not be a comparable number.

Pairing is the whole design. The token stream is the fp16 arm's own greedy walk, so at every decision both formats see
the identical context and their KLs differ only in the cache format - no sampling, no free-running divergence between
the arms, no prompt-level noise from the two arms seeing different text. The bootstrap resamples **prompts**, one
observation each, because 160 decisions from one prompt share a context and are not 160 pieces of evidence; a
token-level interval would be far too narrow to be a gate.

    python -m tools.kvq.sweep --pack ~/.kindi/packs/bonsai-2b            # the 48 prompts, prints the interval
    python -m tools.kvq.sweep --pack PACK --tokens 40 --out run.json    # a shorter run, with the JSON
    python -m tools.kvq.sweep --selftest                                # checks with known answers

The keys are held at q8_0 (`KINDI_CPU_KVQ_K=2`) throughout, as in `tools.kvq.measure`: the question is about the values.
"""
import json
import os
import sys

import numpy as np

from tools.kvq import measure
from tools.kvq.prompts import PROMPTS, by_category, sha256
from tools.nll import bootstrap_ci

MODES = (4, 6)
ITERS = 2000
SEED = 7


def stream(pack, ids, ntokens, ctx):
    """the fp16 arm's greedy walk: the prompt, then `ntokens` tokens the fp16 cache chose

    This stream is what every arm replays. Replaying rather than letting each arm generate its own continuation is
    what makes the per-prompt numbers paired: after the first token a free-running 2-bit arm is reading text it wrote
    itself, and the KL at decision 80 would then compare two different passages.
    """
    e = measure._engine(pack, 0, ctx=ctx)
    return [int(t) for t in ids] + [int(t) for t in e.generate(ids, ntokens)]


def kl_stream(base, arm, toks, keep_from=0):
    """KL(fp16 || arm) at the decisions of `toks` from index `keep_from` on, both engines in lockstep

    Both caches are consumed one token at a time at the same position, so the logits compared are always the two
    models' answers to the same prefix. `keep_from` is the index of the first decision returned: the decision after
    consuming `toks[k]` predicts `toks[k + 1]`, so passing `len(prompt) - 1` scores the 160 continuations and not the
    prompt, which both arms read as given text. Returns the per-decision KLs in nats and how many argmaxes agreed.
    """
    base.reset()
    arm.reset()
    n = base.ylog_n
    base.feed(int(toks[0]), 0)
    arm.feed(int(toks[0]), 0)
    kls, agree = [], 0
    for pos in range(1, len(toks)):
        base.step(int(toks[pos]), pos)
        arm.step(int(toks[pos]), pos)
        if pos - 1 < keep_from:                 # a prompt token: consumed, not scored
            continue
        za = np.array(base.logits[:n], dtype=np.float64)
        zb = np.array(arm.logits[:n], dtype=np.float64)
        za -= za.max()
        zb -= zb.max()
        la = za - np.log(np.exp(za).sum())
        lb = zb - np.log(np.exp(zb).sum())
        kls.append(float(np.sum(np.exp(la) * (la - lb))))
        agree += int(int(np.argmax(base.logits[:n])) == int(np.argmax(arm.logits[:n])))
    return kls, agree


def summarize(rows):
    """the per-prompt means, the paired difference and its interval, resampling prompts

    `block=1` is the iid bootstrap over prompts: one prompt is one observation, and the 160 decisions inside it are
    deliberately not counted as 160. The seed is fixed, so the interval is the same number on a re-run.
    """
    a = np.array([r["kl4"] for r in rows], dtype=np.float64)
    b = np.array([r["kl6"] for r in rows], dtype=np.float64)
    d = a - b
    lo, hi = bootstrap_ci(d, block=1, iters=ITERS, seed=SEED)
    return {"prompts": int(a.size), "mean4": float(a.mean()), "mean6": float(b.mean()),
            "delta": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "mode6_closer": int((d > 0).sum()), "resolvable": bool(lo > 0.0 or hi < 0.0),
            "bootstrap": {"block": 1, "iters": ITERS, "seed": SEED, "unit": "prompt"}}


def run(pack, items=PROMPTS, ntokens=160, log=print, out=None):
    """the sweep: every prompt through fp16, mode 4 and mode 6, then the paired interval

    One engine per format is built once and reset between prompts; the format decides which host buffers exist at
    construction (see `tools.kvq.measure._engine`), so it cannot be re-armed on a live engine.
    """
    tagged = [(pid, cat, measure.tokenise(pack, text, 4096)) for pid, cat, text in items]
    ctx = max(len(ids) for _p, _c, ids in tagged) + ntokens + 8
    eng = {m: measure._engine(pack, m, ctx=ctx) for m in (0,) + MODES}
    hd = int(getattr(eng[0], "dims", {}).get("head_dim", 64))
    log("pack %s, head_dim %d, K = q8_0, V = modes 4 and 6, %d prompts, %d greedy tokens each"
        % (pack, hd, len(tagged), ntokens))
    rows = []
    for k, (pid, cat, ids) in enumerate(tagged):
        toks = stream(pack, ids, ntokens, ctx)
        rec = {"id": pid, "category": cat, "prompt_tokens": len(ids), "decisions": ntokens}
        for m in MODES:
            kls, agree = kl_stream(eng[0], eng[m], toks, keep_from=len(ids) - 1)
            if len(kls) != ntokens:
                raise SystemExit("tools.kvq.sweep: %s scored %d decisions, not %d" % (pid, len(kls), ntokens))
            rec["kl%d" % m] = float(np.mean(kls))
            rec["kl%d_0" % m] = float(kls[0])
            rec["top1_%d" % m] = agree / float(len(kls))
        rows.append(rec)
        log("  %-14s %5d prompt + %d tokens  KL 4 %.5f  6 %.5f  delta %+.5f   [%d/%d]"
            % (pid, len(ids), ntokens, rec["kl4"], rec["kl6"], rec["kl4"] - rec["kl6"], k + 1, len(tagged)))
    return report(rows, pack, ntokens, hd, out=out, log=log)


def report(rows, pack, ntokens, hd, out=None, log=print):
    whole = summarize(rows)
    # step 0 is the one decision where every arm sees a context no arm wrote, so it is reported separately
    step0 = {"mean4": float(np.mean([r["kl4_0"] for r in rows])),
             "mean6": float(np.mean([r["kl6_0"] for r in rows]))}
    d0 = np.array([r["kl4_0"] - r["kl6_0"] for r in rows], dtype=np.float64)
    lo0, hi0 = bootstrap_ci(d0, block=1, iters=ITERS, seed=SEED)
    step0.update({"delta": float(d0.mean()), "ci_lo": float(lo0), "ci_hi": float(hi0),
                  "mode6_closer": int((d0 > 0).sum()), "resolvable": bool(lo0 > 0.0 or hi0 < 0.0)})

    decisions = sum(r["decisions"] for r in rows)
    bits = (measure.bits_per_value(4, hd), measure.bits_per_value(6, hd))
    log("")
    log("  %-14s %10s %10s %10s" % ("prompt", "mode 4 KL", "mode 6 KL", "delta"))
    for r in rows:
        log("  %-14s %10.5f %10.5f %+10.5f" % (r["id"], r["kl4"], r["kl6"], r["kl4"] - r["kl6"]))
    log("")
    log("%d prompts, %d decisions; bits a value %.4f (mode 4) against %.4f (mode 6)"
        % (whole["prompts"], decisions, bits[0], bits[1]))
    log("  mean KL      mode 4 %.5f   mode 6 %.5f nats" % (whole["mean4"], whole["mean6"]))
    log("  paired difference (4 - 6) %+.5f, 95%% interval [%+.5f, %+.5f] over prompts"
        % (whole["delta"], whole["ci_lo"], whole["ci_hi"]))
    log("  mode 6 closer on %d of %d prompts; the interval %s zero -> %s"
        % (whole["mode6_closer"], whole["prompts"],
           "excludes" if whole["resolvable"] else "includes",
           "resolvable" if whole["resolvable"] else "not resolvable on this set"))
    log("  at step 0 only: %.5f against %.5f, difference %+.5f [%+.5f, %+.5f]"
        % (step0["mean4"], step0["mean6"], step0["delta"], step0["ci_lo"], step0["ci_hi"]))
    body = {"pack": os.path.abspath(pack), "head_dim": hd, "ntokens": ntokens,
            "prompts": whole["prompts"], "decisions": decisions,
            "prompt_sha256": sha256(), "by_category": by_category(),
            "bits_per_value": {"4": bits[0], "6": bits[1]},
            "summary": whole, "step0": step0, "rows": rows}
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=1)
            fh.write("\n")
        log("  wrote %s" % out)
    return body


PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def selftest():
    del PASS[:]
    from kindi import synthpack
    import tempfile
    tmp = tempfile.mkdtemp(prefix="kvq-sweep-")
    pack, _t, _f = synthpack.build(tmp, log=lambda *a: None, width=128, ffn=256, layers=4)
    ctx, ntokens = 24, 6
    ids = [3, 11, 7, 19, 23, 5, 29, 2]
    eng = {m: measure._engine(pack, m, ctx=ctx) for m in (0,) + MODES}

    toks = stream(pack, ids, ntokens, ctx)
    check("sweep: the stream is the prompt followed by the requested number of tokens",
          toks[:len(ids)] == ids and len(toks) == len(ids) + ntokens, "%d tokens" % len(toks))

    exploited, agree = kl_stream(eng[0], eng[0], toks)
    check("sweep: two fp16 caches replay to KL 0 at every decision, argmax included",
          max(exploited) == 0.0 and agree == len(exploited), "%d decisions" % len(exploited))

    k4, _a4 = kl_stream(eng[0], eng[4], toks)
    k6, _a6 = kl_stream(eng[0], eng[6], toks)
    check("sweep: both 2-bit formats diverge from fp16 and from each other",
          max(k4) > 0.0 and max(k6) > 0.0 and not np.allclose(k4, k6),
          "max KL %.4f against %.4f" % (max(k4), max(k6)))

    # the inference, with answers known without an engine: zero difference must have a zero-width interval,
    # a constant one an interval above zero, and a sign-alternating one an interval that covers zero
    def rows_of(ds):
        return [{"kl4": 0.1 + d, "kl6": 0.1, "kl4_0": 0.1 + d, "kl6_0": 0.1, "id": str(i), "category": "x",
                 "prompt_tokens": 1, "decisions": 1} for i, d in enumerate(ds)]
    zero = summarize(rows_of([0.0] * 48))
    pos = summarize(rows_of([0.01] * 48))
    alt = summarize(rows_of([0.01, -0.01] * 24))
    check("sweep: the bootstrap says zero for no difference, above zero for a constant one, and covers zero when the"
          " prompt signs alternate",
          zero["ci_lo"] == 0.0 and zero["ci_hi"] == 0.0 and pos["ci_lo"] > 0.0
          and alt["ci_lo"] < 0.0 < alt["ci_hi"],
          "zero [%.4f, %.4f], constant [%.4f, %.4f], alternating [%.4f, %.4f]"
          % (zero["ci_lo"], zero["ci_hi"], pos["ci_lo"], pos["ci_hi"], alt["ci_lo"], alt["ci_hi"]))
    again = summarize(rows_of([0.01, -0.01] * 24))
    check("sweep: the interval is the same number on a re-run (seed fixed)",
          (again["ci_lo"], again["ci_hi"]) == (alt["ci_lo"], alt["ci_hi"]),
          "[%+.6f, %+.6f]" % (again["ci_lo"], again["ci_hi"]))

    # the slice is the protocol: the 160 decisions that predict the continuation, not the prompt both arms read
    full, _af = kl_stream(eng[0], eng[6], toks)
    gen, _ag = kl_stream(eng[0], eng[6], toks, keep_from=len(ids) - 1)
    check("sweep: keep_from scores exactly the continuation decisions, and they are the tail of the full stream",
          len(gen) == ntokens and np.allclose(gen, full[len(ids) - 1:]) and len(full) == len(toks) - 1,
          "%d of %d decisions" % (len(gen), len(full)))

    check("sweep: the prompt set is 48 prompts, four in each of twelve categories, with unique ids",
          len(PROMPTS) == 48 and set(by_category().values()) == {4} and len({p[0] for p in PROMPTS}) == 48
          and len(sha256()) == 64,
          "%d prompts, sha256 %s" % (len(PROMPTS), sha256()[:12]))

    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == 7 else 1


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main(argv):
    if "--selftest" in argv:
        return selftest()
    pack = _arg(argv, "--pack")
    if not pack:
        print(__doc__.strip())
        return 2
    out = _arg(argv, "--out")
    run(pack, ntokens=int(_arg(argv, "--tokens", 160)), out=out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
