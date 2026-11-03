"""kindi spec-check: does the serving loop's two-token pass leave the state where a single step would?

A greedy request can draft one token, verify both with `kc_step2`, and end up with a token in the state that it never
emits: the request can stop on the first of the two (EOS, a stop string, `max_tokens`, a client that went away).
`Runner.spec_stop` puts the recurrent state back to what one `step` would have left, and this is the test for it. It
drives the same `Runner` methods the server's loop calls - `spec_begin`, `spec_next`, `spec_stop` - against the
synthetic pack, with the drafter scripted from a plain greedy run so each path is taken on purpose rather than when
the model happens to agree with the n-gram:

  * a stop on the token of an accepted pass, by token id and by stop string - the case that has to discard;
  * a stop one token later, after the queued token was used - nothing to discard, and the state must already match;
  * a run whose drafts are all rejected, which must leave exactly the state plain stepping does.

The fixture's vocabulary cannot emit the model's real EOS ids, so the stop is a token the run does produce: the branch
in the loop (`if tok in EOS`) is the same one either way.

  kindi spec-check                  # build a fixture and check
  kindi spec-check --pack DIR       # a real pack instead of the fixture
"""

import os
import tempfile

import numpy as np


def _snap(eng, nkv):
    """the state one `step` leaves: the recurrent state, and the K/V the fed tokens wrote.

    The logits are not in here: kc_step2 leaves the first token's logits in the engine, and the second token's in its
    own buffer, so after a pass that keeps both tokens the engine's logits are one token behind plain stepping - by
    design, and nothing reads them again (greedy sampling takes the pass's token, not a distribution)."""
    return ([a.copy() for a in eng.M] + [a.copy() for a in eng.conv],
            [k[:nkv].copy() for k in eng.Kc], [v[:nkv].copy() for v in eng.Vc])


def _same(a, b):
    if len(a[0]) != len(b[0]) or len(a[1]) != len(b[1]) or len(a[2]) != len(b[2]):
        return False
    return all(np.array_equal(x, y) for x, y in zip(a[0] + a[1] + a[2], b[0] + b[1] + b[2]))


class Scripted(object):
    """A drafter that proposes what the plain run produced, so acceptance is deterministic.

    The n-gram drafter is driven by the model, and a fixture whose weights are noise would accept almost never: the
    pass would then never be the thing under test. `wrong` proposes a different token every time, for the rejected
    path."""

    def __init__(self, seq, vocab, wrong=False):
        self.seq, self.vocab, self.wrong, self.fed = list(seq), int(vocab), wrong, 0

    def push(self, _tok):
        self.fed += 1

    def propose(self, _k):
        # the loop feeds `tok` before asking, so `fed` is already the index of the token that follows it
        if self.fed >= len(self.seq):
            return []
        t = self.seq[self.fed]
        return [(t + 1) % self.vocab if self.wrong else t]


def _plain(eng, runner, prompt, n):
    """greedy through `Runner.step`: the tokens, and the state at each of them (states[j] = after prompt + seq[:j])"""
    runner.tokens = []
    eng.reset()
    nxt = None
    for t in prompt:
        nxt = runner.step(t)
    seq, states = [nxt], [_snap(eng, len(runner.tokens))]
    while len(seq) < n:
        nxt = runner.step(nxt)
        seq.append(nxt)
        states.append(_snap(eng, len(runner.tokens)))
    return seq, states


def _served(eng, runner, tokz, prompt, seq, stop_at, how, wrong=False):
    """One request through the loop the server runs: emit, test for the stop, and discard when it stops.

    `how` is "eos" (the stop is the token id seq[stop_at]) or "stop" (the stop is that token's text, as a stop string
    does). Returns the counters, the tokens emitted, and whether the state and Runner.tokens match the plain run."""
    runner.tokens = []
    eng.reset()
    nxt = None
    for t in prompt:
        nxt = runner.step(t)                      # the server's prepare(): the prompt, then one step
    ok_seq = nxt == seq[0]
    runner.spec_begin(True, drafter=Scripted(seq, eng.m.head.R, wrong=wrong))
    text = ""
    stop_text = tokz.decode_bytes(seq[stop_at]).decode("utf-8", "replace") if how == "stop" else None
    emitted, finish = [], None
    tok = nxt
    for j in range(len(seq)):
        if how == "eos" and tok == seq[stop_at]:
            finish = "EOS (token %d)" % stop_at
            break
        emitted.append(tok)
        text += tokz.decode_bytes(tok).decode("utf-8", "replace")
        if stop_text is not None and stop_text and stop_text in text:
            finish = "stop string %r" % stop_text
            break
        tok = runner.spec_next(tok)
    st = runner.spec_stop()
    return st, finish, emitted, ok_seq


def _server_case(cfg, pack, log):
    """The same greedy request twice through the server's own loop (server.Kindi.generate), speculation off and on.

    The output, the tokens and the state after the request must be identical. The fixture's n-gram drafts are wrong
    tokens, so this covers the loop with the pass wired in and the rejected path inside it - the discard is what the
    cases above do, with the drafter scripted."""
    from .server import Kindi
    rc = dict(cfg)
    rc.update(pack=pack, ctx="512", state_dir=os.path.join(os.path.dirname(pack.rstrip(os.sep)), "server-state"))
    K = Kindi(rc, log=lambda *a: None)
    req = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 6}

    def one(spec, stop):
        os.environ["KINDI_CPU_SPEC"] = spec
        K.eng.reset()
        K.runner.tokens = []
        text, finish = "", None
        for kind, val in K.generate(dict(req, stop=stop)):
            if kind == "text":
                text += val
            else:
                finish = val["finish"]
        return text, finish, list(K.runner.tokens), _snap(K.eng, len(K.runner.tokens))

    plain = one("0", [])
    # a stop string only a later piece can match (every piece of this fixture's text is "t<id>")
    stop = [plain[0][-4:]] if len(plain[0]) >= 4 else []
    a = one("0", stop)
    b = one("1", stop)
    same = (a[0] == b[0] and a[2] == b[2] and _same(a[3], b[3]))
    log("  %-32s %-24s %-6s %-8s %s" % ("server loop, spec off vs on", "%r / %s" % (a[0], a[1]),
                                          "equal" if _same(a[3], b[3]) else "DIFFER",
                                          "equal" if a[2] == b[2] else "DIFFER",
                                          "speculative" if os.environ.get("KINDI_CPU_SPEC") else "-"))
    if not same:
        log("      off: %r %s\n      on:  %r %s" % (a[0], a[2], b[0], b[2]))
    return same


def run(cfg, log=print, pack=None, tokens=8):
    if os.environ.get("KINDI_DEVICE", "gpu") != "cpu":
        raise SystemExit("kindi spec-check drives the CPU engine's two-token pass: set KINDI_DEVICE=cpu")
    from . import synthpack
    from .cache import Runner
    from .engine import make_engine
    from .tokenizer import Tok

    os.environ["KINDI_CPU_SPEC"] = "1"            # the switch the server obeys; this command exists to test the path
    tmp = tempfile.mkdtemp(prefix="kindi-spec-check-")
    if pack is None:
        pack = os.path.join(tmp, "pack")
        synthpack.build(pack, log=lambda *a: None)
        log("spec-check: a fixture pack at %s" % pack)
    eng = make_engine(pack, ctx=128, log=lambda *a: None)
    tokz = Tok(eng.vocab, os.path.join(pack, "merges.txt"))
    rc = dict(cfg)
    rc.update(state_dir=os.path.join(tmp, "state"), checkpoint_every=4096, park_max=0, prefix_max=0, prefix_min=4096)
    runner = Runner(eng, rc, "spec-check", lambda *a: None)
    prompt = [11, 5, 11, 5, 3, 9, 3, 9]
    tokens = max(4, tokens)
    seq, states = _plain(eng, runner, prompt, tokens)
    log("spec-check: %s, %d greedy tokens from the fixture: %s" % (pack, tokens, " ".join(map(str, seq))))

    def text_of(i):
        return tokz.decode_bytes(seq[i]).decode("utf-8", "replace")

    # the same two stop decisions the loop makes on the token of an accepted pass (the second one in the pass), and
    # then the same two one token later; a stop string only works when its text is not in the token before it, and a
    # stop by id only means what it says when that id does not occur earlier in the run. A real pack repeats tokens -
    # the 2B's greedy run starts `198 3 16 15 15 59 4 3`, so a stop at 7 fired at 1 - which a fixture of noise does
    # not, and that is how this went unnoticed until spec-check ran against a pack.
    fresh = [i for i in range(2, len(seq)) if seq[i] not in seq[:i]]
    cases = [("accepted draft, stop id", 1, "eos", False)]
    if text_of(1) and text_of(1) not in text_of(0):
        cases.append(("accepted draft, stop string", 1, "stop", False))
    else:
        log("spec-check: the first two tokens of the fixture decode alike; no stop-string case")
    if 2 in fresh:
        cases.append(("accepted draft, stop one later", 2, "eos", False))
    else:
        log("spec-check: token 2 repeats an earlier token; no one-token-later case")
    if fresh:
        cases.append(("every draft rejected", max(fresh), "eos", True))
    else:
        log("spec-check: every token after the first repeats an earlier one; no rejected-run case")
    ok = True
    log("\n  %-32s %-24s %-6s %-8s %s" % ("case", "stopped", "state", "tokens", "passes"))
    for name, at, kind, wrong in cases:
        st, finish, emitted, ok_seq = _served(eng, runner, tokz, prompt, seq, at, kind, wrong=wrong)
        # EOS is tested before the token is emitted; a stop string after it (see the loop in server.py)
        want = seq[:at] if kind == "eos" else seq[:at + 1]
        same_state = _same(_snap(eng, len(runner.tokens)), states[at])
        same_tokens = runner.tokens == prompt + seq[:at]
        used = ("accepted %d" % st["accepted"]) if not wrong else ("rejected %d" % st["rejected"])
        good = ok_seq and emitted == want and same_state and same_tokens and (st["passes"] > 0) and \
            ((st["rejected"] > 0) if wrong else (st["accepted"] > 0))
        log("  %-32s %-24s %-6s %-8s %s%s" % (name, finish or "-", "equal" if same_state else "DIFFER",
                                              "equal" if same_tokens else "DIFFER", used, "" if good else "   FAILED"))
        if not good:
            ok = False
            if emitted != want:
                log("      the loop emitted %s, the plain run %s" % (emitted, want))
            if not same_tokens:
                log("      Runner.tokens is %s, expected %s" % (runner.tokens, prompt + seq[:at]))
    log("")
    ok = _server_case(cfg, pack, log) and ok
    log("\n  %s" % ("the serving loop's pass is transparent, stops included" if ok else
                   "THE TWO-TOKEN PASS CHANGED THE STATE OR THE TOKENS"))
    return 0 if ok else 1
