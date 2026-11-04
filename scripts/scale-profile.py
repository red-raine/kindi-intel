"""Where in a window does a format's NLL cost sit: a per-position profile of the paired delta.

`scripts/rope-longctx.py` (and `tools/nll --prefill`) can say what a format costs on average over a walk. That average
hides the shape, and the shape is the finding: a cost that is flat in position is per-token quantisation noise, while a
cost that grows with position is an error written into a recurrent state and carried forward - the two want different
fixes, and the second one means the number gets worse as the context gets longer. The 2B packs differ only in how their
block scales are stored (`bonsai-2b-v1` keeps fp16, `bonsai-2b-v2` keeps an 8-bit index into a per-tensor table, with
identical codes), which makes them the clean pair for this.

    python -u scripts/rope-longctx.py walk --pack PACK_A --out a.jsonl --ctx 2048 --prefill 0
    python -u scripts/rope-longctx.py walk --pack PACK_B --out b.jsonl --ctx 2048 --prefill 0
    python scripts/scale-profile.py join --a a.jsonl --b b.jsonl --bins 64 --block 32 --json out.json

The delta is `b - a` in nats a token, so positive means `b` is worse at that position. The bins are the shape, the
quarters are the accumulation in four numbers, and the slope answers the actual question with an interval: nats per
1,000 positions, block-bootstrapped over `--block`-token blocks because neighbouring positions share activations and a
badly encoded block moves every position after it in the same direction.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _arg(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def load(path):
    """-> {position: (token, nats)}, the per-token trace a walk writes"""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[int(r["pos"])] = (int(r["tok"]), float(r["lp"]))
    return rows


def _slope(x, y):
    """least-squares slope of y on x"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    den = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - ym)).sum() / den) if den else float("nan")


def slope_ci(x, y, block, iters=2000, seed=7, level=0.95):
    """block bootstrap of the slope: resample contiguous blocks, refit, take the percentiles

    Resampling single positions would treat a walk as thousands of independent observations; it is not, and a
    position-level interval on a drift-shaped signal comes out absurdly narrow.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    block = max(1, min(int(block), n))
    nb = max(1, n // block)
    rng = np.random.RandomState(seed)
    off = np.arange(block)
    out = np.empty(iters)
    for i in range(iters):
        s = rng.randint(0, max(1, n - block + 1), size=nb)
        idx = np.clip((s[:, None] + off[None, :]).ravel(), 0, n - 1)
        out[i] = _slope(x[idx], y[idx])
    lo, hi = np.percentile(out, [(1 - level) / 2 * 100.0, (1 + level) / 2 * 100.0])
    return float(lo), float(hi)


def join(argv):
    from tools.nll import bootstrap_ci

    a_path, b_path = _arg(argv, "--a"), _arg(argv, "--b")
    bins, block = int(_arg(argv, "--bins", "64")), int(_arg(argv, "--block", "32"))
    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    if len(common) < 8:
        raise SystemExit("scale-profile: the two walks share only %d positions" % len(common))
    if any(a[p][0] != b[p][0] for p in common):
        raise SystemExit("scale-profile: the walks scored different tokens at the same position")

    pos = np.asarray(common, dtype=np.float64)
    la = np.array([a[p][1] for p in common])
    lb = np.array([b[p][1] for p in common])
    d = lb - la

    print("  %d scored positions %d..%d, b - a in nats\n" % (d.size, common[0], common[-1]))
    print("  position bin     tokens     nll a     nll b      delta")
    rows = []
    start = common[0]
    while start <= common[-1]:
        stop = start + bins
        sel = [i for i, p in enumerate(common) if start <= p < stop]
        if sel:
            dd = d[sel]
            lo, hi = bootstrap_ci(dd, block=max(4, block // 2))
            rows.append({"from": start, "to": min(stop, common[-1] + 1), "tokens": int(dd.size),
                         "nll_a": float(la[sel].mean()), "nll_b": float(lb[sel].mean()),
                         "delta": float(dd.mean()), "ci_lo": lo, "ci_hi": hi})
            print("  %5d-%-5d   %6d   %8.4f  %8.4f   %+.5f  [%+.5f, %+.5f]"
                  % (rows[-1]["from"], rows[-1]["to"], dd.size, rows[-1]["nll_a"], rows[-1]["nll_b"],
                     dd.mean(), lo, hi))
        start = stop

    n = d.size
    q = max(1, n // 4)
    quarters = []
    print("\n  quarters (the accumulation in four numbers)")
    for i in range(4):
        idx = np.arange(i * q, (i + 1) * q if i < 3 else n)
        dd = d[idx]
        lo, hi = bootstrap_ci(dd, block=block)
        quarters.append({"from": int(pos[idx[0]]), "to": int(pos[idx[-1]]) + 1, "tokens": int(dd.size),
                         "delta": float(dd.mean()), "ci_lo": lo, "ci_hi": hi})
        print("  %5d-%-5d   %6d   %+.5f  [%+.5f, %+.5f]"
              % (quarters[-1]["from"], quarters[-1]["to"], dd.size, dd.mean(), lo, hi))

    s = _slope(pos, d)
    slo, shi = slope_ci(pos, d, block)
    print("\n  slope %+.5f nats per 1,000 positions, 95%% [%+.5f, %+.5f]" % (s * 1000.0, slo * 1000.0, shi * 1000.0))
    print("  %s that the cost %s with position" % ("clear" if (slo > 0 or shi < 0) else "not clear",
                                                   "rises" if s > 0 else "falls"))
    tot_lo, tot_hi = bootstrap_ci(d, block=block)
    print("  over the walk: %+.5f [%+.5f, %+.5f], a closer at %d positions, b at %d"
          % (d.mean(), tot_lo, tot_hi, int((d > 0).sum()), int((d < 0).sum())))

    out = {"a": os.path.abspath(a_path), "b": os.path.abspath(b_path), "bins": bins, "block": block,
           "positions": [common[0], common[-1]], "tokens": int(n), "delta": float(d.mean()),
           "ci": [tot_lo, tot_hi], "slope_per_1000": s * 1000.0, "slope_ci": [slo * 1000.0, shi * 1000.0],
           "slope_excludes_zero": bool(slo > 0 or shi < 0), "bins_table": rows, "quarters": quarters}
    if "--json" in argv:
        with open(_arg(argv, "--json"), "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print("\nwrote %s" % _arg(argv, "--json"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "join":
        raise SystemExit(__doc__)
    sys.exit(join(sys.argv[2:]))
