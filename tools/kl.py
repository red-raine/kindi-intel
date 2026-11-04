"""Price what a re-fitted ternary build buys: what changed in the file, and what that did to the next-token distribution.

`Ingot-XXIV-27B` is Prism's Ternary Bonsai 2 27B with "the same ternary codes, with the rest of its weights re-tuned ...
to bring it closer to Qwen3.8-27B". That is a claim about bytes and a claim about behaviour, and the two need different
measurements - which is why this module has two subcommands:

    blocks   the two PTQ1_0 GGUFs block by block: are the {-1,0,+1} codes identical, and how far did each 128-weight
             group's fp16 scale move? No model is loaded and no token is generated, so this is exact and cheap, and it
             answers "what did the miner change" rather than "did it help".
    logits   the two converted packs on the same tokens: KL in both directions, per position, plus top-1 agreement and
             each model's own NLL. This is what "closer to the teacher" has to show up in, and here it is measured
             without the teacher - as the distance between the two ternary builds.

The block layout is Prism's PTQ1_0: 28 bytes per 128-weight group, 26 of packed base-3 trits and an fp16 scale. Tensors
are compared as raw blocks, so the comparison is of the file, not of a re-encoding of it.

    python -m tools.kl blocks --a Bonsai.gguf --b Ingot.gguf [--json out.json]
    python -m tools.kl logits --a PACK --b PACK [--text FILE] [--tokens 96] [--json out.json]
    python -m tools.kl --selftest

One caveat that belongs in the numbers rather than the prose: a KL between two quants of the same teacher is not a
quality ranking. Ingot's README prices itself at KL(teacher || model) 0.08276 against the king's 0.08409; what this
measures is how far Ingot moved the *Bonsai* distribution, which is the miner's lever. A smaller distance from Bonsai is
not automatically better, and a larger one is not automatically worse - it is only the size of the change.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from .nll import tokenise

PTQ1_0 = 143
BLOCK = 28
PROSE = ("The instrument had been rebuilt twice before anyone thought to ask what it measured. On the third morning "
         "the technician wrote the first honest line in the log: the needle moves, the paper does not, and the reading "
         "means nothing until the two are connected. She spent the rest of the week on that connection, and the "
         "notebook from those days is the only part of the archive that anyone still cites.")

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


# --- blocks -----------------------------------------------------------------------------------------------------------

def compare_blocks(ra, rb):
    """Two stacks of 28-byte PTQ1_0 blocks -> what differs between them.

    Codes are compared as the 26 bytes they are stored in, which is exact: identical bytes are identical trits under any
    decoding, and the trit count is only decoded for the blocks that actually differ. Scales are the fp16 pair at
    [26:28]; a zero scale has no ratio and is counted separately rather than divided by.
    """
    from kindi.convert import trits_from_raw                 # imported here so `logits` does not need it

    same = np.all(ra[:, :26] == rb[:, :26], axis=1)
    differing = ~same
    trits = 0
    if differing.any():
        trits = int((trits_from_raw(ra[differing]) != trits_from_raw(rb[differing])).sum())
    sa = np.ascontiguousarray(ra[:, 26:28]).view(np.float16).reshape(-1).astype(np.float32)
    sb = np.ascontiguousarray(rb[:, 26:28]).view(np.float16).reshape(-1).astype(np.float32)
    same_bits = (ra[:, 26:28] == rb[:, 26:28]).all(axis=1)
    usable = (sa != 0) & np.isfinite(sa) & np.isfinite(sb)
    ratio = np.where(usable, sb / np.where(usable, sa, 1.0), np.nan)
    return dict(blocks=int(ra.shape[0]), code_diff=int(differing.sum()), trit_diff=trits,
                scale_same=int(same_bits.sum()), scale_usable=int(usable.sum()),
                up=int(np.nansum(ratio > 1.01)), down=int(np.nansum(ratio < 0.99)),
                same_1pct=int(np.nansum(np.abs(ratio - 1.0) <= 0.01)),
                ratio=ratio, zero_scale=int((sa == 0).sum()))


def blocks(a_path, b_path, log=print, chunk=4096):
    """Walk both GGUFs' PTQ1_0 tensors and accumulate `compare_blocks` over every group."""
    from kindi.gguf import GGUF

    ga, gb = GGUF(a_path), GGUF(b_path)
    ta = dict((t[0], t) for t in ga.tensors)
    tb = dict((t[0], t) for t in gb.tensors)
    shared = [n for n in ta if n in tb and ta[n][2] == PTQ1_0 and tb[n][2] == PTQ1_0]
    log("  %d tensors in a, %d in b, %d PTQ1_0 in both" % (len(ta), len(tb), len(shared)))
    fa = np.memmap(a_path, dtype=np.uint8, mode="r")
    fb = np.memmap(b_path, dtype=np.uint8, mode="r")
    tot = dict(blocks=0, code_diff=0, trit_diff=0, scale_same=0, scale_usable=0, up=0, down=0,
               same_1pct=0, zero_scale=0, weights=0, ratios=[])
    rows = []
    for name in sorted(shared):
        ne, off = ta[name][1], ta[name][3]
        ne0 = int(ne[0])
        groups = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        bpr = ne0 // 128
        n = groups * bpr
        base_a, base_b = ga.data_start + off, gb.data_start + tb[name][3]
        acc = dict(blocks=0, code_diff=0, trit_diff=0, scale_same=0, scale_usable=0, up=0, down=0,
                   same_1pct=0, zero_scale=0)
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            ra = np.asarray(fa[base_a + c0 * BLOCK: base_a + c1 * BLOCK]).reshape(-1, BLOCK)
            rb = np.asarray(fb[base_b + c0 * BLOCK: base_b + c1 * BLOCK]).reshape(-1, BLOCK)
            s = compare_blocks(ra, rb)
            for k in acc:
                acc[k] += s[k]
            if len(tot["ratios"]) < 200000:                   # a subsample for percentiles, counts stay exact
                r = s["ratio"]
                tot["ratios"].extend(r[np.isfinite(r)][: max(0, 200000 - len(tot["ratios"]))].tolist())
        for k in ("blocks", "code_diff", "trit_diff", "scale_same", "scale_usable", "up", "down",
                  "same_1pct", "zero_scale"):
            tot[k] += acc[k]
        tot["weights"] += n * 128
        rows.append(dict(name=name, weights=n * 128, **acc))
    ratios = np.asarray(tot.pop("ratios"), dtype=np.float32)
    tot["ratio_median"] = float(np.median(ratios)) if ratios.size else float("nan")
    tot["ratio_p5"] = float(np.percentile(ratios, 5)) if ratios.size else float("nan")
    tot["ratio_p95"] = float(np.percentile(ratios, 95)) if ratios.size else float("nan")
    tot["ratio_sample"] = int(ratios.size)
    rows.sort(key=lambda r: -r["trit_diff"])
    return tot, rows


# --- logits -----------------------------------------------------------------------------------------------------------

def log_softmax_rows(z):
    m = z.max(axis=1, keepdims=True)
    return z - (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))


def logits_of(pack, ids, ctx=None, log=None):
    """The distribution for every token but the first, plus the targets it is scored against.

    One pass, one head evaluation per scored token: the engine's `step(tok, pos)` leaves the distribution for pos+1, so
    the loop stores the buffer before stepping the next token. `ctx` must hold the sequence; a longer one is refused by
    the engine rather than silently wrapped.
    """
    from kindi.cpu import Engine

    ids = [int(t) for t in ids]
    ctx = int(ctx or (len(ids) + 8))
    if len(ids) > ctx:
        raise SystemExit("tools.kl: %d tokens do not fit in a context of %d" % (len(ids), ctx))
    e = Engine(pack, ctx=ctx, log=log or (lambda *a: None))
    t0 = time.time()
    e.reset()
    e.step(ids[0], 0)
    zs, targets = [], []
    for i in range(1, len(ids)):
        zs.append(np.array(e.logits[:e.ylog_n], dtype=np.float32))
        targets.append(ids[i])
        if i + 1 < len(ids):
            e.step(ids[i], i)
    return np.stack(zs), np.asarray(targets), time.time() - t0


def metrics(za, zb, targets):
    """KL both ways, per position, top-1 agreement, and each model's own NLL on the same targets."""
    la, lb = log_softmax_rows(za.astype(np.float64)), log_softmax_rows(zb.astype(np.float64))
    pa, pb = np.exp(la), np.exp(lb)
    kl_ab = np.sum(pa * (la - lb), axis=1)
    kl_ba = np.sum(pb * (lb - la), axis=1)
    idx = np.arange(len(targets))
    nll_a = float(-la[idx, targets].mean())
    nll_b = float(-lb[idx, targets].mean())
    return dict(tokens=int(len(targets)), kl_ab=float(kl_ab.mean()), kl_ba=float(kl_ba.mean()),
                kl_ab_max=float(kl_ab.max()), kl_ab_max_at=int(kl_ab.argmax()), kl_ab_median=float(np.median(kl_ab)),
                agree=float((za.argmax(1) == zb.argmax(1)).mean()),
                nll_a=nll_a, nll_b=nll_b, nll_delta=nll_a - nll_b,
                per_position=kl_ab.tolist())


# --- cli --------------------------------------------------------------------------------------------------------------

def cmd_blocks(args):
    t0 = time.time()
    tot, rows = blocks(args.a, args.b)
    print("\n%-34s %10s %12s %12s %10s" % ("tensor", "weights", "code blocks", "trits differ", "scales chg"))
    for r in rows[:12]:
        if r["weights"] and (r["code_diff"] or r["scale_same"] != r["blocks"]):
            print("%-34s %10d %12d %12d %10d"
                  % (r["name"][:34], r["weights"], r["code_diff"], r["trit_diff"], r["blocks"] - r["scale_same"]))
    w = max(tot["weights"], 1)
    print("\ntotals over %d PTQ1_0 weights in %d groups of 128" % (tot["weights"], tot["blocks"]))
    print("  trits differing     %12d  %8.6f%% of weights" % (tot["trit_diff"], 100.0 * tot["trit_diff"] / w))
    print("  groups with a code diff  %8d  %8.4f%%" % (tot["code_diff"], 100.0 * tot["code_diff"] / max(tot["blocks"], 1)))
    print("  scales changed      %12d  %8.4f%% of groups (%.4f%% bit-identical)"
          % (tot["blocks"] - tot["scale_same"], 100.0 * (tot["blocks"] - tot["scale_same"]) / max(tot["blocks"], 1),
             100.0 * tot["scale_same"] / max(tot["blocks"], 1)))
    print("  ratio b/a           median %.5f  p5 %.5f  p95 %.5f  (>1: %d, <1: %d, within 1%%: %d)"
          % (tot["ratio_median"], tot["ratio_p5"], tot["ratio_p95"], tot["up"], tot["down"], tot["same_1pct"]))
    print("  %.1fs" % (time.time() - t0))
    if args.json:
        out = dict(a=args.a, b=args.b, **dict((k, v) for k, v in tot.items() if not isinstance(v, list)),
                   top_tensors=rows[:20])
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print("report: %s" % args.json)
    return 0


def cmd_logits(args):
    text = PROSE
    if args.text:
        with open(args.text, encoding="utf-8", errors="replace") as f:
            text = f.read()
    ids = tokenise(args.a, text)[: int(args.tokens) + 1]
    if len(ids) < 3:
        raise SystemExit("tools.kl: %d tokens is too few to compare distributions" % len(ids))
    v_a = json.load(open(os.path.join(args.a, "vocab.json")))
    v_b = json.load(open(os.path.join(args.b, "vocab.json")))
    same_vocab = v_a == v_b
    print("tokens: %d (the pack's own tokeniser); vocabularies %s"
          % (len(ids), "identical" if same_vocab else "DIFFER (%d vs %d) - the KL is not comparable"
             % (len(v_a), len(v_b))))
    za, ta, sa = logits_of(args.a, ids, args.ctx)
    zb, tb, sb = logits_of(args.b, ids, args.ctx)
    if not np.array_equal(ta, tb) or za.shape != zb.shape:
        raise SystemExit("tools.kl: the two runs have different shapes (%r vs %r) or targets" % (za.shape, zb.shape))
    m = metrics(za, zb, ta)
    print("\na=%s\nb=%s" % (args.a, args.b))
    print("  KL(a || b)  %.6f nats   median %.6f   max %.6f at position %d"
          % (m["kl_ab"], m["kl_ab_median"], m["kl_ab_max"], m["kl_ab_max_at"]))
    print("  KL(b || a)  %.6f nats" % m["kl_ba"])
    print("  top-1 agreement %.4f" % m["agree"])
    print("  NLL a %.6f   b %.6f   b - a %+.6f nats/token" % (m["nll_a"], m["nll_b"], -m["nll_delta"]))
    print("  engine time %.1fs + %.1fs" % (sa, sb))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(a=args.a, b=args.b, ids=ids, same_vocab=same_vocab, **m), f, indent=2)
        print("report: %s" % args.json)
    return 0


def selftest():
    import tempfile
    from kindi import synthpack

    del PASS[:]
    # -- blocks: the comparison is exact, and it counts both directions of scale change
    rng = np.random.RandomState(0)
    base = rng.randint(0, 256, size=(64, BLOCK)).astype(np.uint8)
    base[:, 26:28] = np.ascontiguousarray(np.float16([0.5]).view(np.uint8)).reshape(1, 2)
    same = compare_blocks(base, base.copy())
    check("blocks: identical files differ in nothing", same["code_diff"] == 0 and same["trit_diff"] == 0
          and same["scale_same"] == 64 and same["same_1pct"] == 64)
    changed = base.copy()
    changed[7, 0] ^= 0x01
    one = compare_blocks(base, changed)
    check("blocks: one flipped code byte is one group and a handful of trits",
          one["code_diff"] == 1 and 1 <= one["trit_diff"] <= 128,
          "%d trits in 1 of 64 groups" % one["trit_diff"])
    scaled = base.copy()
    half = np.ascontiguousarray(np.float16([0.25]).view(np.uint8)).reshape(1, 2)
    scaled[:10, 26:28] = half
    s = compare_blocks(base, scaled)
    moved = s["ratio"][np.abs(s["ratio"] - 1.0) > 1e-6]          # the ten that moved, not the fifty-four that did not
    check("blocks: a halved scale is a down-ratio and no code change",
          s["code_diff"] == 0 and s["down"] == 10 and moved.size == 10 and np.allclose(moved, 0.5, atol=1e-6),
          "down %d, all ten at ratio %.4f" % (s["down"], float(moved.mean())))

    # -- logits: two packs of the same shape, so the wiring can be checked without a 27B
    tmp = tempfile.mkdtemp(prefix="kl-")
    a = synthpack.build(os.path.join(tmp, "a"), seed=11, log=lambda *x: None)[0]
    b = synthpack.build(os.path.join(tmp, "b"), seed=12, log=lambda *x: None)[0]
    ids = list(range(101, 113))
    za, ta, _ = logits_of(a, ids, ctx=32)
    zb, tb, _ = logits_of(b, ids, ctx=32)
    self_m = metrics(za, za, ta)
    check("logits: comparing a pack with itself is KL 0 and agreement 1",
          abs(self_m["kl_ab"]) < 1e-12 and self_m["agree"] == 1.0 and self_m["nll_delta"] == 0.0,
          "KL %.1e, nll %.6f" % (self_m["kl_ab"], self_m["nll_a"]))
    m = metrics(za, zb, ta)
    check("logits: two different packs have a positive, finite, symmetric pair of KLs",
          m["kl_ab"] > 0 and np.isfinite(m["kl_ab"]) and np.isfinite(m["kl_ba"]),
          "KL(a||b) %.4f, KL(b||a) %.4f, agreement %.3f" % (m["kl_ab"], m["kl_ba"], m["agree"]))
    check("logits: one row per scored token, and the NLL is on the same targets",
          za.shape[0] == len(ids) - 1 and np.array_equal(ta, np.asarray(ids[1:])))
    check("logits: a sequence longer than the context is refused, not wrapped",
          _refused(lambda: logits_of(a, list(range(100, 120)), ctx=8)))

    # -- the tokeniser, because `cmd_logits` scores the tokens the pack's own vocab produces, and a pack `convert`
    # writes has a plain GGUF token list: no 248044 added-token region to build the special-token split from.
    from kindi.tokenizer import Tok
    text = "The cache holds one key and one value for every position."
    merges = os.path.join(a, "merges.txt")
    names = ["t%d" % i for i in range(128)]
    by_shape = [Tok(v, merges).encode(text) for v in (dict((s, i) for i, s in enumerate(names)), names,
                                                      [[s, -1.0] for s in names])]
    check("logits: a dict, a list or a list of pairs of tokens all encode to the same ids, with no added-token "
          "region read as special tokens",
          by_shape[0] == by_shape[1] == by_shape[2] == tokenise(a, text) and len(by_shape[1]) >= 11
          and Tok(names, merges).split_re is None,
          "%d ids, all three shapes agree" % len(by_shape[1]))
    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == 8 else 1


def _refused(fn):
    try:
        fn()
        return False
    except SystemExit:
        return True


def main(argv):
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(prog="tools.kl", description="what changed between two ternary builds, in bytes and in logits")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("blocks", help="compare two PTQ1_0 GGUFs block by block")
    b.add_argument("--a", required=True)
    b.add_argument("--b", required=True)
    b.add_argument("--json")
    l = sub.add_parser("logits", help="KL between two converted packs on the same tokens")
    l.add_argument("--a", required=True)
    l.add_argument("--b", required=True)
    l.add_argument("--text")
    l.add_argument("--tokens", type=int, default=96)
    l.add_argument("--ctx", type=int, default=0)
    l.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "blocks":
        return cmd_blocks(a)
    if a.cmd == "logits":
        return cmd_logits(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
