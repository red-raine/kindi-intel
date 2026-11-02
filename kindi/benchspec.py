"""kindi bench-spec: speculative decode on real text, measured through the tokenizer.

The two arms are the shipped ones - `cpu.Engine.generate` (plain greedy) and `cpu.generate_spec` (n-gram draft, one
token per pass, verified by the two-token GEMV). The prompts are windows of a text file tokenised by the pack's own
tokenizer, so the numbers describe real text instead of synthetic token ids. The speculative arm must reproduce the
plain one token for token or the run fails.

Prefill is removed by cancellation rather than estimated: each arm also runs the same prompt with `--tokens 1`, and
the decode rate is (t(n) - t(1)) / (n - 1). At 48-token prompts prefill is a quarter of the time, so estimating it from
a separate read would swing the ratio by several points.

  kindi bench-spec README.md
  kindi bench-spec docs/BENCHMARKS.md --tokens 128 --prompts 8 --prompt-len 48
"""
import os
import time


def run(cfg, log=print, text=None, n=128, prompts=8, plen=48):
    if os.environ.get("KINDI_DEVICE", "gpu") != "cpu":
        raise SystemExit("kindi bench-spec measures the CPU engine's speculative path: set KINDI_DEVICE=cpu "
                         "(the GPU engines use KINDI_SPEC instead)")
    if not text:
        raise SystemExit("kindi bench-spec: give a text file to read prompts from (prose or code)")
    from .cpu import generate_spec
    from .engine import make_engine
    from .tokenizer import Tok

    pack = cfg["pack"]
    try:
        raw = open(text, encoding="utf-8", errors="replace").read()
    except OSError as e:
        raise SystemExit("kindi bench-spec: cannot read %s (%s)" % (text, e))
    eng = make_engine(pack, ctx=cfg["ctx"], log=log)
    tok = Tok(eng.vocab, os.path.join(pack, "merges.txt"))
    ids = tok.encode(raw)
    if len(ids) < plen + 8:
        raise SystemExit("kindi bench-spec: %s holds %d tokens, too few for %d-token prompts" % (text, len(ids), plen))
    windows = [ids[i:i + plen] for i in range(0, len(ids) - plen, plen * 6)][:prompts]
    log("bench-spec: %s, %d tokens of text, %d prompts of %d tokens, %d new tokens each"
        % (text, len(ids), len(windows), plen, n))

    acc = drf = 0
    one_plain = one_spec = t_plain = t_spec = 0.0
    same = 0
    for i, ids_p in enumerate(windows):
        # each arm at n and at 1: the difference cancels the prefill and the first token
        t0 = time.time(); eng.generate(ids_p, 1); p1 = time.time() - t0
        t0 = time.time(); generate_spec(eng, ids_p, 1, k=1, stats={}); s1 = time.time() - t0
        t0 = time.time(); plain = eng.generate(ids_p, n); pn = time.time() - t0
        t0 = time.time()
        st = {}
        spec = generate_spec(eng, ids_p, n, k=1, stats=st)
        sn = time.time() - t0
        one_plain += p1; one_spec += s1; t_plain += pn; t_spec += sn
        log("  prompt %d: plain %.2fs - %.2fs = %.2fs, spec %.2fs - %.2fs = %.2fs"
            % (i, pn, p1, pn - p1, sn, s1, sn - s1))

        acc += st["accepted"]
        drf += st["drafted"]
        eq = list(plain[:n]) == list(spec[:n])
        same += eq
        if not eq:
            log("  prompt %d: the speculative arm differs from plain greedy at token %d"
                % (i, next(k for k in range(n) if plain[k] != spec[k])))

    per_arm = (n - 1) * len(windows)                 # tokens in the decode-only part of each arm
    dec_plain = t_plain - one_plain
    dec_spec = t_spec - one_spec
    log("plain greedy         %6.2f tokens/s" % (per_arm / dec_plain))
    log("speculative (n-gram) %6.2f tokens/s  %.3fx" % (per_arm / dec_spec, dec_plain / dec_spec))
    log("draft acceptance     %d/%d = %.3f  (break-even is 0.35)" % (acc, drf, acc / max(1, drf)))
    log("identical to greedy  %d/%d prompts" % (same, len(windows)))
    if same != len(windows):
        log("kindi bench-spec: the speculative path changed the output - that is a bug, not a speed-up")
        return 1
    return 0
