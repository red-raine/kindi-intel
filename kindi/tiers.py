"""kindi verify-tier: are the kernel tiers actually the same computation?

The CPU engine has three GEMV implementations over one pack: scalar C (tier 0, no intrinsics), the AVX2 pair-table
kernel (tier 1) and the int8-activation prototype (tier 2, expected to differ). `Engine._pick_tier` runs a one-token
probe at load and refuses a tier that moves the logits; this command is the full statement of the same claim - the
whole model, several prompts, logits, greedy tokens, recurrent state and the two-token pass - plus what each tier
costs.

  kindi verify-tier                     # on the configured pack
  kindi verify-tier --tokens 32 --prompts 3

Exit code is 1 if a tier that is supposed to be exact is not. The int8 tier is reported and not counted as a failure:
the point of that path was to find out whether int8 activations keep the output, and the answer recorded in
docs/CPU-PERF-2026.md is that they do not.
"""
import time

import numpy as np


def _state(eng):
    return [a.copy() for a in eng.M] + [a.copy() for a in eng.conv]


def _restore(eng, st):
    for a, b in zip(eng.M + eng.conv, st):
        a[...] = b


def _same_state(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def _two_token(eng, ids):
    """(tokens, state, logits) of one two-token pass against two single steps, from the same starting state"""
    eng.reset()
    for i, t in enumerate(ids[:-1]):
        eng.feed(t, i)
    pos, tok = len(ids) - 1, ids[-1]
    st0 = _state(eng)
    a = eng.step2(tok, tok, pos)
    st_a, lg_a = _state(eng), eng.logits.copy()      # after the pass: m holds token 0's logits
    _restore(eng, st0)
    t0 = eng.step(tok, pos)
    lg_first = eng.logits.copy()
    t1 = eng.step(tok, pos + 1)
    return a, (t0, t1), _same_state(st_a, _state(eng)), np.array_equal(lg_a, lg_first)


def run(cfg, log=print, tokens=24, prompts=3):
    from .engine import make_engine
    eng = make_engine(cfg["pack"], ctx=max(cfg["ctx"], 256), log=lambda *a: None)
    tiers = eng.avail_tiers()
    log("kindi verify-tier: %s, tiers %s" % (cfg["pack"], ", ".join(n for _b, n in tiers)))
    vocab = max(2, int(eng.m.head.R))
    seqs = [[(i * 37 + 11 * p) % vocab for i in range(8 + 4 * p)] for p in range(prompts)]

    ref = {}
    for bit, name in tiers:
        eng.set_tier(bit)
        ref[name] = [eng.generate(s, tokens) for s in seqs]

    base, ok = ref["scalar"], True
    log("\n  tier      prompts identical to scalar   speed")
    for bit, name in tiers:
        same = sum(1 for a, b in zip(base, ref[name]) if a == b)
        eng.set_tier(bit)
        t0 = time.time()
        eng.generate(seqs[0], tokens)
        dt = (time.time() - t0) / tokens
        log("  %-9s %3d/%d%s                       %7.3f ms/token"
            % (name, same, len(base), "   DIFFERENT" if same != len(base) else "", dt * 1000))
        if name != "q8" and same != len(base):
            ok = False

    log("\n  two-token pass vs two single steps:")
    for bit, name in tiers:
        eng.set_tier(bit)
        two, one, same_state, same_logits = _two_token(eng, seqs[0])
        log("    %-9s tokens %s vs %s   state %s   logits %s"
            % (name, two, one, "equal" if same_state else "DIFFER", "equal" if same_logits else "DIFFER"))
        if name != "q8" and (two != one or not same_state or not same_logits):
            ok = False
    log("\n  %s" % ("every exact tier agrees" if ok else "A TIER THAT SHOULD BE EXACT IS NOT"))
    return 0 if ok else 1
