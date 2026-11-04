"""Walk one pack out to a long position and score the tail, writing each scored token as it lands.

Why not `tools.nll --prefill`: a 9k-token window at 1.7B on this CPU engine is about an hour of forward passes (attention
is O(position) a token, so the cost is quadratic in the prefill), and a run that is interrupted has to leave behind what
it already measured. `tools.nll` keeps its trace in memory and writes the file at the end; this appends one JSON object
per scored token and flushes, and prints a progress line every `--progress` positions so the rate is visible while it
runs. The two files it writes are then joined by `join` below, which is the measurement.

    python -u scripts/rope-longctx.py walk --pack PACK --out plain.jsonl --ctx 9216 --prefill 7680
    python -u scripts/rope-longctx.py walk --pack PACK --out yarn.jsonl  --ctx 9216 --prefill 7680
    python scripts/rope-longctx.py join --a plain.jsonl --b yarn.jsonl --bands 7680,8192,9216 --json out.json

`--offset` starts the window later in the corpus and numbers its positions from there, which is the cheap way to
measure *at* a deep position: 512 tokens fed at 11.7k and 512 scored at 12.3k costs a thousand forward passes, where
a genuine 12.3k-token context costs twelve thousand. The difference it measures is at positions no longer inside the
trained window with a ~0.5k history rather than the full 12k; that is a narrower claim than a real 12k context, and
the numbers are only quoted that way - and it ranks factors differently from the full-context walk, because with a few
hundred tokens of history compressing the apparent distances between them helps for reasons that are not extrapolation
(see CPU-PERF-2026.md).

`--rope-factor F --rope-window N` rebuilds the kernel's table with a factor the manifest does not name (`F = 1` is the
plain table), which is how "what if the file had asked for 1.25x instead of 4x" is measured without a pack per factor.
The arrays are overwritten in place and the first value is read back, so a patch that did not land prints a different
mscale instead of quietly reporting a null result.

`a` is the reference arm and `b` the one being tested: the delta is `b - a` in nats, so a negative number means `b` puts
more probability on the held-out tokens.
"""
import ctypes
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORPUS = "B:/models/corpora/scale-gate/prose/prose.txt"


def _arg(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def walk(argv):
    from kindi.cpu import Engine
    from kindi import rope
    from tools.nll import tokenise

    pack, out = _arg(argv, "--pack"), _arg(argv, "--out")
    corpus = _arg(argv, "--corpus", CORPUS)
    ctx, prefill = int(_arg(argv, "--ctx", "9216")), int(_arg(argv, "--prefill", "7680"))
    offset = int(_arg(argv, "--offset", "0"))
    progress = int(_arg(argv, "--progress", "256"))
    if not pack or not out:
        raise SystemExit("walk needs --pack and --out")
    if prefill >= ctx or prefill < offset:
        raise SystemExit("--prefill must sit between --offset and --ctx, or nothing is left to score")

    with open(corpus, encoding="utf-8", errors="replace") as f:
        allids = tokenise(pack, f.read())
    if len(allids) < ctx:
        raise SystemExit("%s tokenises to %d tokens, short of a %d-token window" % (corpus, len(allids), ctx))
    ids = allids[offset:ctx]
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    first = max(prefill, offset + 1)          # the token at `offset` sets the position and is never scored
    print("pack %s\n  rope %s\n  corpus %s, positions %d..%d\n  %d tokens, feeding %d, scoring %d\n"
          % (pack, man.get("rope_scaling"), corpus, offset, offset + len(ids) - 1, len(ids),
             first - offset, max(0, offset + len(ids) - first)), flush=True)

    e = Engine(pack, ctx=ctx, log=lambda *a: None)
    # `--rope-factor F` rebuilds the table with a factor the manifest does not name, which is how "what if the file
    # had asked for 1.25x instead of 4x" gets measured without a pack per factor. F = 1 is the plain table. The kernel's
    # arrays are overwritten in place and read back, so a patch that did not land fails here rather than as a null result.
    factor = _arg(argv, "--rope-factor")
    if factor is not None:
        sc = dict(man.get("rope_scaling") or {"type": "yarn", "original_context_length": ctx})
        sc["factor"] = float(factor)
        if sc["factor"] <= 1.0:
            sc = {"type": "none"}
        if _arg(argv, "--rope-window"):
            sc["original_context_length"] = int(_arg(argv, "--rope-window"))
        cos, sin = rope.tables(ctx, man["dimensions"]["rope_dim"], man["dimensions"]["rope_base"], sc)
        half = man["dimensions"]["rope_dim"] // 2
        for field, tab in (("cos_t", cos), ("sin_t", sin)):
            buf = np.ctypeslib.as_array(getattr(e.m, field), shape=(ctx, half))
            buf[:] = tab
        print("  table replaced: factor %s, window %s -> row 0 of cos %.6f (manifest: %s)"
              % (sc.get("factor", 1.0), sc.get("original_context_length"),
                 float(getattr(e.m, "cos_t")[0]), man.get("rope_scaling")), flush=True)
    t0 = time.time()
    e.reset()
    e.step(ids[0], offset)
    scored = 0
    with open(out, "w", encoding="utf-8") as f:
        for i in range(1, len(ids)):
            pos = offset + i
            if pos < prefill:
                if i + 1 < len(ids):
                    e.feed(ids[i], pos)
            else:
                z = np.asarray(e.logits[:e.ylog_n], dtype=np.float64)
                lp = float(z.max() + np.log(np.exp(z - z.max()).sum()) - z[ids[i]])
                f.write(json.dumps({"pos": pos, "tok": int(ids[i]), "lp": lp}) + "\n")
                scored += 1
                if i + 1 < len(ids):
                    e.step(ids[i], pos)
            if i % progress == 0:
                f.flush()
                el = time.time() - t0
                eta = el / i * (len(ids) - 1 - i)
                print("  pos %5d/%d  %.1f min, %.3f s/token, eta %.1f min, %d scored"
                      % (pos, offset + len(ids) - 1, el / 60.0, el / i, eta / 60.0, scored), flush=True)
    print("\nwrote %s: %d scored tokens in %.1f min" % (out, scored, (time.time() - t0) / 60.0), flush=True)
    return 0


def join(argv):
    from tools.nll import bootstrap_ci

    a_path, b_path = _arg(argv, "--a"), _arg(argv, "--b")
    bands = [int(x) for x in _arg(argv, "--bands", "7680,8192,9216").split(",")]
    block = int(_arg(argv, "--block", "32"))

    def load(path):
        rows = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[r["pos"]] = (int(r["tok"]), float(r["lp"]))
        return rows

    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("join: the two walks scored no position in common")
    if any(a[p][0] != b[p][0] for p in common):
        raise SystemExit("join: the walks scored different tokens at the same position; they are not pair-able")

    rows = []
    for lo, hi in zip(bands, bands[1:]):
        pos = [p for p in common if lo <= p < hi]
        if not pos:
            continue
        la = np.array([a[p][1] for p in pos])
        lb = np.array([b[p][1] for p in pos])
        d = lb - la
        clo, chi = bootstrap_ci(d, block=block)
        rows.append({"from": lo, "to": hi, "tokens": int(d.size), "nll_a": float(la.mean()),
                     "nll_b": float(lb.mean()), "delta": float(d.mean()), "ci_lo": clo, "ci_hi": chi,
                     "favours_b": int((d < 0).sum()), "favours_a": int((d > 0).sum()),
                     "worst": float(d.max()), "best": float(d.min())})
        print("  %5d-%-5d %5d tokens  a %.4f, b %.4f, delta %+.5f  95%% [%+.5f, %+.5f]  b closer on %d, a on %d"
              % (lo, hi, d.size, la.mean(), lb.mean(), d.mean(), clo, chi, int((d < 0).sum()), int((d > 0).sum())),
              flush=True)
    out = {"a": os.path.abspath(a_path), "b": os.path.abspath(b_path), "block": block,
           "band_boundaries": bands, "positions": [int(common[0]), int(common[-1])], "rows": rows}
    if "--json" in argv:
        with open(_arg(argv, "--json"), "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print("\nwrote %s" % _arg(argv, "--json"))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit(walk(sys.argv[2:]) if cmd == "walk" else join(sys.argv[2:]) if cmd == "join" else
             (_ for _ in ()).throw(SystemExit(__doc__)))
