"""Capture the V vectors the cache quantiser sees, so the codec question can be asked on real values.

`tools.kvq.measure` compares the 2-bit V formats against an fp16 cache by running the whole engine and scoring the
logits, which answers what a codec costs the model. It cannot answer what a codec costs a *vector*, and on
`kindi.synthpack` - everything this host had until a real pack was converted - it answers the model question on random
weights, which is the wrong distribution: that is why the module's end-to-end ordering flipped between two fixtures.

This closes that gap from both ends at once. `kcpu.c` grows a hook (`KINDI_CPU_VCAP`) that writes every V vector as the
cache is about to quantise it - the fp32 `m->vv` of each (layer, position, head), before any codec touches it - and this
module reads such a file and scores the codecs on it. The hook does not depend on `KINDI_CPU_KVQ`, so one capture is
the input to every format at once, including ones added later:

    python -m tools.kvq.capture                          # capture from the synthetic pack, then score
    python -m tools.kvq.capture --cap FILE               # score a capture from another host, no engine
    python -m tools.kvq.capture --cap FILE --per-layer   # and per layer, where the drift lives
    python -m tools.kvq.capture --selftest

From a real pack, either point this module at one and let it drive the engine:

    python -m tools.kvq.capture --pack PACK --tokens 256 --per-layer
    python -m tools.kvq.capture --pack PACK --text FILE --out /tmp/v.bin

or let the capture ride along with any CPU run, with nothing to write, and score the file afterwards:

    KINDI_DEVICE=cpu KINDI_CPU_VCAP=/tmp/v.bin kindi --pack PACK bench-spec TEXT   # or serve, spec-check, ...
    python -m tools.kvq.capture --cap /tmp/v.bin --per-layer

Run against `prism-ml/Ternary-Bonsai-2-2B` at 198 tokens of prose, the capture is 2376 V vectors over 6 attention
layers, and the block scale is 5.7% closer at the codec (0.4072 -> 0.3841 mean relL2), winning 2374 of the 2376 vectors;
per layer the gain runs 2.5% to 8.2%, and the middle layers carry the worst absolute error (0.483 relL2 at layer 19).
End to end, `tools.kvq.measure --pack` on the same pack keeps the ordering: 0.15993 against 0.19804 nats.

The file is a 20-byte header (magic, layers, attention layers, kv heads, head_dim) followed by records of four int32
(layer, position, head) and `head_dim` float32. Records are self-describing because they are written from inside a
parallel loop, where the order is the scheduler's and not the model's; a truncated file is a prefix and loads fine.
"""
import ctypes
import json
import os
import sys
import tempfile

import numpy as np

from .measure import bits_per_value, q2_block, q2_head, rel_l2

MAGIC = 0x4B564331
PASS = []

# prose, so the activations are the ones a model actually produces - the codec question is about the values a layer
# writes into its cache, and a list of random token ids is not that, however real the weights are
DEFAULT_TEXT = (
    "The cache holds one key and one value for every position the model has read. When a layer writes a value it "
    "knows nothing about the ones before it: the projection is a matrix product over the residual stream, so the "
    "range within a single head is whatever that token happened to produce. Some positions are dominated by one "
    "outlier coordinate; others are quiet all the way across. A format that spends one scale on the whole head has "
    "to cover the loudest coordinate it will ever see, and every quiet value in that head rounds against a grid "
    "sized for something it is not. A format that spends a scale on each thirty-two values tracks the range where "
    "it actually lives, at the price of two more bytes for every block it writes. Whether that is a good trade is "
    "not a question about codecs in the abstract. It is a question about how much the range moves within a head in "
    "a trained model, which is why the answer needs vectors from one rather than from a fixture with random weights."
)


def tokenise(pack, text, n):
    """the first `n` tokens of `text` in the pack's own tokeniser"""
    from kindi.tokenizer import Tok
    vocab = json.load(open(os.path.join(pack, "vocab.json")))
    merges = os.path.join(pack, "merges.txt")
    if not os.path.isfile(merges):
        raise SystemExit("tools.kvq.capture: %s has no merges.txt, so its text cannot be tokenised here" % pack)
    return Tok(vocab, merges).encode(text)[:int(n)]


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def _engine(pack, kvq_v=0, ctx=64, capture=None, limit=None):
    """an engine whose V cache is in `kvq_v`, optionally writing every V vector to `capture`

    The capture is armed through `kc_vcap_begin` and armed *after* construction on purpose: `Engine.__init__` probes the
    kernel tiers by stepping a token per tier, and those steps write the V cache too. Armed earlier, the file would open
    with a handful of probe records at position 0 that no prompt produced.

    It is armed explicitly rather than with `KINDI_CPU_VCAP` because that variable is invisible to the MSVC build: that
    DLL links the CRT statically, so its getenv reads an environment snapshot taken when the library loads, and Python
    has already set this variable by then. The variable still works when it is in the process environment at launch.
    """
    os.environ["KINDI_CPU_KVQ_K"] = "0"
    os.environ["KINDI_CPU_KVQ_V"] = str(int(kvq_v))
    from kindi.cpu import Engine
    e = Engine(pack, ctx=ctx, log=lambda *a: None)
    if not hasattr(e.lib, "kc_vcap_begin"):
        raise SystemExit("tools.kvq.capture: this kcpu build has no V-capture hook")
    if capture:
        e.lib.kc_vcap_begin.argtypes = [ctypes.POINTER(type(e.m)), ctypes.c_char_p, ctypes.c_longlong]
        e.lib.kc_vcap_begin.restype = ctypes.c_int
        e.lib.kc_vcap_close.argtypes = []
        rc = e.lib.kc_vcap_begin(ctypes.byref(e.m), os.fsencode(capture), int(limit or 0))
        if rc != 0:
            raise SystemExit("tools.kvq.capture: could not open the capture %s (rc %d)" % (capture, rc))
    return e


def capture(pack, ids, path, ctx=None, kvq_v=0, limit=None):
    """run `ids` through the engine and write the V vectors to `path`; returns the path and the engine"""
    e = _engine(pack, kvq_v=kvq_v, ctx=ctx or (len(ids) + 8), capture=path, limit=limit)
    try:
        for i, t in enumerate(ids):
            e.feed(int(t), i)          # feed, not step: the logits are not the question here
    finally:
        e.lib.kc_vcap_close()          # the file stays open across steps, so the host closes it
    return path, e


def load(path):
    """read a capture: header, then records sorted by (layer, position, head)"""
    with open(path, "rb") as f:
        head = np.fromfile(f, dtype=np.int32, count=5)
        if head.size != 5 or int(head[0]) != MAGIC:
            raise SystemExit("tools.kvq.capture: %s is not a V capture (header %r)" % (path, head.tolist()))
        nl, natt, nhkv, hd = (int(v) for v in head[1:])
        body = np.fromfile(f, dtype=np.uint32)
    stride = 4 + hd
    n = body.size // stride
    body = body[:n * stride].reshape(n, stride)
    idx = body[:, :4].astype(np.int32)
    vec = np.ascontiguousarray(body[:, 4:]).view(np.float32)
    order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))
    return dict(nl=nl, natt=natt, nhkv=nhkv, hd=hd, idx=idx[order], vec=vec[order], path=path)


def write_capture(path, nl, natt, nhkv, hd, idx, vec):
    """write a capture the way kcpu.c does - the format a file from another host has to match, and what the loader is
    checked against without running an engine"""
    idx = np.asarray(idx, np.int32).reshape(-1, 4)
    vec = np.ascontiguousarray(np.asarray(vec, np.float32).reshape(len(idx), hd))
    with open(path, "wb") as f:
        np.array([MAGIC, nl, natt, nhkv, hd], np.int32).tofile(f)
        np.hstack([idx.view(np.uint32), vec.view(np.uint32)]).astype(np.uint32).tofile(f)
    return path


def score(vecs, block=32):
    """mean relative L2 of the two V codecs over a stack of vectors: (scale per head, scale per 32-value block)"""
    vecs = np.asarray(vecs, np.float64)
    head = np.array([rel_l2(v, q2_head(v)) for v in vecs])
    blk = np.array([rel_l2(v, q2_block(v, block)) for v in vecs])
    return head, blk


def report(cap, per_layer=False, note=""):
    v, idx = cap["vec"], cap["idx"]
    if not len(v):
        raise SystemExit("tools.kvq.capture: %s holds no records" % cap["path"])
    hd = cap["hd"]
    head, blk = score(v)
    tokens = int(idx[:, 1].max()) + 1
    print("capture %s: %d layers (%d attention), %d kv heads, head_dim %d"
          % (cap["path"], cap["nl"], cap["natt"], cap["nhkv"], hd))
    print("  %d V vectors over %d positions%s\n" % (len(v), tokens, (", " + note) if note else ""))
    print("  codec on the captured values, mean relative L2 over %d vectors" % len(v))
    print("  %-30s %7s %8s" % ("", "bits", "relL2"))
    print("  %-30s %7.2f %8.4f" % ("2-bit, one scale a head", bits_per_value(4, hd), head.mean()))
    print("  %-30s %7.2f %8.4f" % ("2-bit, one scale a 32-block", bits_per_value(6, hd), blk.mean()))
    gain = (head.mean() - blk.mean()) / max(head.mean(), 1e-30)
    print("  gain from blocking: %.1f%% (%s)" % (100 * gain,
                                                 "blocking closer" if gain > 0 else "blocking no closer"))
    worse = int((blk > head).sum())
    print("  per vector: block closer on %d of %d, head closer on %d" % (len(v) - worse, len(v), worse))
    if per_layer:
        print("\n  per layer (mean relL2)")
        print("  %6s %10s %10s %8s" % ("layer", "scale/head", "scale/32", "vectors"))
        for il in sorted(set(int(x) for x in idx[:, 0])):
            m = idx[:, 0] == il
            print("  %6d %10.4f %10.4f %8d" % (il, head[m].mean(), blk[m].mean(), int(m.sum())))
    print("\nRead this as the codec question on the values themselves. Whether it matters for the model is the other")
    print("measurement (`python -m tools.kvq.measure`), and it needs the same real pack to be worth running.")
    return head, blk


def selftest():
    del PASS[:]
    from kindi import synthpack
    tmp = tempfile.mkdtemp(prefix="vcap-")
    pack, _t, _f = synthpack.build(tmp, log=lambda *a: None, width=128, ffn=256, layers=2)
    ids = [3, 11, 7, 19, 23, 5]
    rng = np.random.RandomState(0)
    path = os.path.join(tmp, "v.bin")
    _p, eng = capture(pack, ids, path, ctx=16)
    cap = load(path)
    want = len(ids) * cap["natt"] * cap["nhkv"]
    ok_head = (cap["nl"], cap["natt"], cap["nhkv"], cap["hd"]) == (eng.m.NL, 1, eng.m.NHKV, eng.m.HD)
    check("capture: the header is the engine's shape and every position, head and attention layer is there",
          ok_head and len(cap["vec"]) == want,
          "%d layers (%d attention), %d kv heads, head_dim %d, %d of %d vectors"
          % (cap["nl"], cap["natt"], cap["nhkv"], cap["hd"], len(cap["vec"]), want))
    # the hook must write the vector the cache is built from: the fp16 cache is that value rounded, so a record has to
    # agree with it to fp16 precision. layers=2 with an interval of 2 leaves one attention layer, so Vc[0] is its cache.
    worst, where = 0.0, None
    hd, nhkv = cap["hd"], cap["nhkv"]
    flat = eng.Vc[0]                                     # [ctx][NHKV][HD], held flat as fp16 bits
    for r, (il, pos, kv, _hd) in enumerate(cap["idx"]):
        o = (int(pos) * nhkv + int(kv)) * hd
        cached = flat[o:o + hd].view(np.float16).astype(np.float64)
        got = np.asarray(cap["vec"][r], np.float64)
        e = float(np.linalg.norm(got - cached) / max(np.linalg.norm(cached), 1e-30))
        if e > worst:
            worst, where = e, (int(il), int(pos), int(kv))
    check("capture: a record is the vector the fp16 cache holds, at layer/pos/head %r" % (where,), worst < 1e-3,
          "worst relL2 %.1e over %d vectors (fp16 precision is 1.0e-03)" % (worst, len(cap["vec"])))
    # the hook is before the codec, so the cache format must not change what is written at position 0 - later positions
    # are the model reacting to a different cache, which is the other measurement's business
    p6 = os.path.join(tmp, "v6.bin")
    capture(pack, ids, p6, ctx=16, kvq_v=6)
    c6 = load(p6)
    m = c6["idx"][:, 1] == 0
    same = np.array_equal(c6["vec"][m], cap["vec"][cap["idx"][:, 1] == 0])
    check("capture: the format does not change the captured value (position 0 identical under mode 6)", same,
          "%d vectors compared" % int(m.sum()))
    # a bounded capture loads as a prefix, which is what a long run needs
    p3 = os.path.join(tmp, "v3.bin")
    capture(pack, ids, p3, ctx=16, limit=5)
    check("capture: KINDI_CPU_VCAP_MAX bounds the file and the loader takes the prefix", len(load(p3)["vec"]) == 5,
          "5 of %d" % want)
    # the path a real pack takes: a file produced elsewhere, with the same layout
    hd = cap["hd"]
    idx = np.array([[1, p, kv, hd] for p in range(4) for kv in range(2)], np.int32)
    vec = rng.randn(len(idx), hd) * np.repeat(rng.uniform(0.25, 4.0, hd // 32), 32)   # a drifting head
    foreign = write_capture(os.path.join(tmp, "foreign.bin"), 2, 1, 2, hd, idx, vec)
    fc = load(foreign)
    fh, fb = score(fc["vec"])
    check("capture: a file written by another host loads and scores on its own vectors",
          len(fc["vec"]) == len(idx) and np.allclose(fc["vec"], vec, atol=1e-6) and np.all(np.isfinite(fh)),
          "mean relL2 head %.4f, block %.4f over %d vectors" % (fh.mean(), fb.mean(), len(fc["vec"])))
    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == 5 else 1


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main(argv):
    if "--selftest" in argv:
        return selftest()
    per_layer = "--per-layer" in argv
    if "--cap" in argv:
        report(load(_arg(argv, "--cap")), per_layer=per_layer)
        return 0
    if "--pack" in argv:
        pack = _arg(argv, "--pack")
        text = open(_arg(argv, "--text"), encoding="utf-8").read() if "--text" in argv else DEFAULT_TEXT
        ids = tokenise(pack, text, int(_arg(argv, "--tokens", 256)))
        out = _arg(argv, "--out") or os.path.join(tempfile.mkdtemp(prefix="vcap-"), "v.bin")
        capture(pack, ids, out, ctx=len(ids) + 8)
        report(load(out), per_layer=per_layer, note="%s, %d tokens of text" % (pack, len(ids)))
        return 0
    from kindi import synthpack
    tmp = tempfile.mkdtemp(prefix="vcap-")
    pack, _t, _f = synthpack.build(tmp, log=lambda *a: None, width=256, ffn=512, layers=6)
    ids = list(range(5, 5 + 24))
    path = os.path.join(tmp, "v.bin")
    capture(pack, ids, path, ctx=48)
    report(load(path), per_layer=per_layer,
           note="synthetic pack: random weights, so the absolute numbers are not the model's")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
