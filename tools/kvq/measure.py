"""Does a per-32-block scale rescue 2 bits on the K/V values?

`kcpu.c` stores the cache in one of six formats. Mode 4 spends 2 bits a value on four levels {-3,-1,+1,+3} with **one
scale per (position, head)**; mode 6 spends the same 2 bits a value on the same four levels with **one scale per 32
values**, which costs 2.5 bits a value instead of 2.0625. The question this module answers is whether that scale is what
limits the values, by running the engine twice over the same tokens - once with an fp16 cache, once with a quantised one -
and comparing the logits the model produces:

    python -m tools.kvq.measure            # the table: KL, top-1 agreement, bits a value
    python -m tools.kvq.measure --selftest # checks with known answers

The keys are held at q8_0 (`KINDI_CPU_KVQ_K=2`) in every row so the only thing changing is the **values**, which is the
question asked. Divergence is measured against the fp16 cache on the same weights: KL(P_fp16 || P_quant) in nats, and how
often the greedy token agrees.

The model is the synthetic fixture (`kindi.synthpack`), because there is no Prism pack on this host. Its weights are
random, so the numbers say how much this codec costs the *arithmetic*, not what it costs a trained 2B. That limit is
stated here rather than hidden: a trained model's V vectors are the ones the note's harness measured on 372 real cache
vectors, and the same code would need a pack to repeat that.
"""
import ctypes
import os
import sys
import tempfile

import numpy as np

LT = np.array([-3.0, -1.0, 1.0, 3.0])


def bits_per_value(mode, hd):
    """Bits a value, including the scale: one fp16 a head (2 bits) or one fp16 per 32 values (2.5)."""
    side = {0: 16.0, 1: 2.0, 3: 4.0, 4: 2.0, 5: 4.0, 6: 2.0}[mode]
    return side + (16.0 / hd if mode not in (6,) else 16.0 / 32.0)


FORMATS = {0: "fp16", 1: "ternary, 2 bits, scale a head", 3: "4-level, 4 bits, scale a head",
           4: "4-level, 2 bits, scale a head", 5: "4-level twice, 4 bits, scale a head",
           6: "4-level, 2 bits, scale a 32-block"}


def _fit(v, starts):
    """the four-start least-squares scale of q2_pack, in numpy: returns (level index per value, scale)"""
    best = (None, 1e-8, np.inf)
    for s0 in starts:
        a = s0 if s0 > 0 else 1e-8
        for _ in range(10):
            b = 2.0 * a
            k = np.where(v >= b, 3, np.where(v > 0, 2, np.where(v > -b, 1, 0)))
            den = float((LT[k] ** 2).sum())
            na = float((v * LT[k]).sum() / den) if den > 0 else a
            if not na > 0:
                break
            a = na
        b = 2.0 * a
        k = np.where(v >= b, 3, np.where(v > 0, 2, np.where(v > -b, 1, 0)))
        sse = float(((v - a * LT[k]) ** 2).sum())
        if sse < best[2]:
            best = (k, a, sse)
    return best[0], best[1]


def q2_head(v):
    """mode 4: one scale for the whole vector"""
    v = np.asarray(v, np.float64)
    am, mn = float(np.abs(v).max()), float(np.abs(v).sum())
    k, a = _fit(v, [am / 6.0, am / 9.0, am / 12.0, 0.5 * mn / len(v)])
    return a * LT[k]


def q2_block(v, block=32):
    """mode 6: one scale per block"""
    v = np.asarray(v, np.float64)
    out = np.empty_like(v)
    for i in range(0, len(v), block):
        x = v[i:i + block]
        am, mn = float(np.abs(x).max()), float(np.abs(x).sum())
        k, a = _fit(x, [am / 6.0, am / 9.0, am / 12.0, 0.5 * mn / len(x)])
        out[i:i + block] = a * LT[k]
    return out


def rel_l2(orig, rec):
    return float(np.linalg.norm(orig - rec) / max(np.linalg.norm(orig), 1e-30))


def codec_curve(trials=200, hd=64, seed=0):
    """The reconstruct-ability of the two codecs on two shapes of vector, which is where the answer lives.

    - `flat`: one scale for the head is right, because every block has the same range. This is what a random-weight
      model's V projections look like.
    - `drift`: the range varies across the head by a factor of four from one 32-block to the next - a trained model's
      values can look like this, and it is the case blocking is supposed to fix.
    """
    rng = np.random.RandomState(seed)
    out = {}
    for name, make in (("flat", lambda: rng.randn(hd)),
                       ("drift", lambda: rng.randn(hd) * np.repeat(rng.uniform(0.25, 4.0, hd // 32), 32))):
        head, block = [], []
        for _ in range(trials):
            v = make()
            head.append(rel_l2(v, q2_head(v)))
            block.append(rel_l2(v, q2_block(v)))
        out[name] = (float(np.mean(head)), float(np.mean(block)))
    return out

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def _engine(pack, kvq_v, ctx=96):
    """one engine with the V side forced to `kvq_v`, over a pack that was already built"""
    os.environ["KINDI_CPU_KVQ_K"] = "2"
    os.environ["KINDI_CPU_KVQ_V"] = str(kvq_v)
    from kindi.cpu import Engine
    e = Engine(pack, ctx=ctx, log=lambda *a: None)
    if e.kvq_v != kvq_v:
        raise SystemExit("the engine refused V format %d (got %d)" % (kvq_v, e.kvq_v))
    return e


def logits_trace(pack, ids, kvq_v, ctx=96):
    """logits after every token of `ids`, in the engine's float buffer"""
    e = _engine(pack, kvq_v, ctx=ctx)
    out = []
    for i, t in enumerate(ids[:-1]):
        e.feed(int(t), i)
        e.step(int(ids[i + 1]), i + 1)          # step fills self.logits for the next token
        out.append(np.array(e.logits[:e.ylog_n], dtype=np.float64))
    return np.array(out)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compare(a, b):
    """KL(a || b) in nats, mean over positions, and how often the greedy token agrees"""
    pa, pb = softmax(a), softmax(b)
    kl = float(np.mean(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30)), axis=1)))
    agree = float(np.mean(a.argmax(axis=1) == b.argmax(axis=1)))
    return kl, agree


def run(pack, ids, hd=64):
    base = logits_trace(pack, ids, 0)
    rows = []
    for mode in sorted(FORMATS):
        if mode == 0:
            rows.append(dict(mode=mode, name=FORMATS[mode], bits=16.0, kl=0.0, agree=1.0))
            continue
        got = logits_trace(pack, ids, mode)
        kl, agree = compare(base, got)
        rows.append(dict(mode=mode, name=FORMATS[mode], bits=bits_per_value(mode, hd), kl=kl, agree=agree))
    return rows


def selftest():
    del PASS[:]
    from kindi import synthpack
    tmp = tempfile.mkdtemp(prefix="kvq-")
    pack, _t, _f = synthpack.build(tmp, log=lambda *a: None, width=128, ffn=256, layers=4)
    ids = [3, 11, 7, 19, 23, 5, 29, 2, 17, 13]
    base = logits_trace(pack, ids, 0)
    same = logits_trace(pack, ids, 0)
    kl0, agree0 = compare(base, same)
    check("kvq: the fp16 cache reproduces itself (KL 0, identical argmax)", kl0 == 0.0 and agree0 == 1.0,
          "KL %.3g, agree %.2f" % (kl0, agree0))
    got = logits_trace(pack, ids, 4)
    kl4, agree4 = compare(base, got)
    check("kvq: quantising the values to 2 bits changes the logits", kl4 > 0.0,
          "KL %.3g, top-1 %.2f" % (kl4, agree4))
    got6 = logits_trace(pack, ids, 6)
    kl6, agree6 = compare(base, got6)
    check("kvq: the block-scale format runs and is a different cache from mode 4", kl6 > 0.0 and not np.array_equal(got, got6),
          "KL %.3g, top-1 %.2f" % (kl6, agree6))
    # the bit accounting is arithmetic, not a measurement: 2 bits a value plus one fp16 scale per 32 values
    check("kvq: a 32-block scale costs 2.5 bits a value, a head scale 2.25 at head_dim 64",
          abs(bits_per_value(6, 64) - 2.5) < 1e-9 and abs(bits_per_value(4, 64) - 2.25) < 1e-9,
          "block %.3f, head %.3f" % (bits_per_value(6, 64), bits_per_value(4, 64)))
    # the mechanism: a block scale reconstructs closer, and the gain is larger when the range varies across the head
    curve = codec_curve(trials=120)
    gain = {k: (v[0] - v[1]) / v[0] for k, v in curve.items()}
    check("kvq: a block scale reconstructs closer, and by more when the range drifts",
          all(v[1] < v[0] for v in curve.values()) and gain["drift"] > gain["flat"],
          "drift %.4f -> %.4f (%.1f%%), flat %.4f -> %.4f (%.1f%%)"
          % (curve["drift"][0], curve["drift"][1], 100 * gain["drift"],
             curve["flat"][0], curve["flat"][1], 100 * gain["flat"]))

    # End to end is reported, not asserted: on a random-weight model the ordering of mode 4 and mode 6 KL flips with the
    # fixture's size (it did between width 256 x 6 layers and width 128 x 4), so an assertion here would be pinning noise.
    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == 5 else 1


def report():
    from kindi import synthpack
    tmp = tempfile.mkdtemp(prefix="kvq-")
    pack, _t, _f = synthpack.build(tmp, log=lambda *a: None, width=256, ffn=512, layers=6)
    ids = list(range(5, 5 + 40))
    hd = int(getattr(_engine(pack, 0), "dims", {}).get("head_dim", 64))
    rows = run(pack, ids, hd=hd)
    print("synthetic fixture (6 layers, width 256, random weights); K = q8_0 everywhere, V varies\n")
    print("  %-34s %7s %9s %8s" % ("V format", "bits", "KL", "top-1"))
    for r in rows:
        print("  %-34s %7.2f %9.4f %7.0f%%" % (r["name"], r["bits"], r["kl"], 100 * r["agree"]))
    print("\nThe comparison of interest is mode 4 against mode 6: the same 2-bit alphabet, one scale a head or one a 32-value")
    print("block. If blocking rescues 2 bits, mode 6 is closer to fp16 than mode 4 while staying nearer 2 bits than 4.")

    curve = codec_curve()
    print("\nand at the codec itself, mean relative L2 error over 200 vectors (%d-value head):" % hd)
    print("  %-8s %12s %12s %9s" % ("vector", "scale/head", "scale/32", "gain"))
    for name in ("flat", "drift"):
        h, b = curve[name]
        print("  %-8s %12.4f %12.4f %8.1f%%" % (name, h, b, 100 * (h - b) / h))
    print("\n`flat` is what random weights give - every block has about the same range; `drift` varies the range")
    print("four-fold across the head, the case blocking is meant to fix. A block scale reconstructs closer in both, by")
    print("more the more the range drifts, and that part is stable across fixtures.")

    small = synthpack.build(tempfile.mkdtemp(prefix="kvq-"), log=lambda *a: None, width=128, ffn=256, layers=4)[0]
    print("\nthe same comparison on a second fixture, because the logit ranking is not stable across sizes:")
    for label, pk in (("width 256, 6 layers", pack), ("width 128, 4 layers", small)):
        kls = []
        for start in (5, 100):
            ids2 = list(range(start, start + 40))
            b2 = logits_trace(pk, ids2, 0)
            kls.append((compare(b2, logits_trace(pk, ids2, 4))[0], compare(b2, logits_trace(pk, ids2, 6))[0]))
        h, b = float(np.mean([k[0] for k in kls])), float(np.mean([k[1] for k in kls]))
        print("  %-20s scale/head KL %.5f, scale/32 KL %.5f -> %s"
              % (label, h, b, "32-block closer" if b < h else "head scale closer"))
    print("\nThat is the honest state of the end-to-end question: on random weights the sign flips with the fixture, so it")
    print("cannot be settled without a trained pack, where the V vectors have the structure the note's harness measured.")
    return 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    return report()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
