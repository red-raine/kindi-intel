"""The multi-probe index as a built object: construct once, query many, and time both arms on the same queries.

`probe.py` counts candidates and models the cost as `candidates x D + p`. A count is not a clock: gathering cells means a
`searchsorted` per probed key, a concatenation and a sort of the pool, and on small N that overhead can be larger than the
distance work it saves. This module builds the index the way a router would - keys sorted once, then a query per position
- and reports measured wall-clock per query beside the full 20-NN arm, so the saving is a measurement rather than an
inference from the candidate count.

    python -m tools.fingerprint.index               # build, time, and compare against the full router
    python -m tools.fingerprint.index --selftest    # checks with known answers

The identity holds at `p=0, radius=0`: one cell holding every training position, so both arms must return exactly the
same predictions. Every other setting is an approximation of that router, and the timing only means something because the
zero setting proves the two code paths agree.
"""
import sys
import time

import numpy as np

from ..bench import compare
from . import cell as C
from . import probe as P
from . import router as R

TOPK = R.TOPK


class Index(object):
    """Sorted `p`-bit keys over the training codes, plus the order permutation they were built under."""

    def __init__(self, ref_bits, p, seed=0):
        self.p = p
        self.dim = ref_bits.shape[1]
        self.order = C.bit_order(self.dim, seed)
        keys = C.key_of(ref_bits, self.order, p)
        perm = np.argsort(keys, kind="stable")
        self.keys = keys[perm]
        self.rows = perm                              # candidate id -> row of ref_bits
        self.n = len(perm)

    def _probe_keys(self, q_bits, radius):
        qkey = int(C.key_of(q_bits[None, :], self.order, self.p)[0])
        n = P.probe_count(self.p, radius)
        return np.fromiter(P.probe_keys(qkey, self.p, radius), np.int64, count=n)

    def candidates(self, q_bits, radius, budget):
        """Training rows in probe order, cut at the budget by whole cells. Returns (rows, scanned).

        One `searchsorted` over all probe keys, not one per key: at radius 3 and `p=8` that is 93 numpy calls turned into a
        single vectorised one, which is the difference between the per-query cost being dominated by the gather or by the
        interpreter. `candidates_loop` is the same search written the obvious way, and a check asserts the two agree
        exactly - an optimisation that changes which cells are visited is a different index, not a faster one.
        """
        keys = self._probe_keys(q_bits, radius)
        lo = np.searchsorted(self.keys, keys, "left")
        hi = np.searchsorted(self.keys, keys, "right")
        counts = hi - lo
        if budget is not None:
            cum = np.cumsum(counts)
            take = int(np.searchsorted(cum, budget, "left")) + 1     # cells whose start is still under budget
            lo, counts = lo[:take], counts[:take]
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, np.int64), 0
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        idx = np.repeat(lo - starts, counts) + np.arange(total)       # position inside each cell's slice
        rows = self.rows[idx]
        return np.sort(rows), total

    def candidates_loop(self, q_bits, radius, budget):
        """The reference form: a `searchsorted` per probe key, cells concatenated in order. Same answer, more calls."""
        out = []
        scanned = 0
        for key in self._probe_keys(q_bits, radius):
            if budget is not None and out and scanned >= budget:
                break
            lo = int(np.searchsorted(self.keys, key, "left"))
            hi = int(np.searchsorted(self.keys, key, "right"))
            if hi > lo:
                out.append(self.rows[lo:hi])
                scanned += hi - lo
        if not out:
            return np.zeros(0, np.int64), 0
        rows = np.concatenate(out)
        return np.sort(rows), len(rows)


def answer(bits, ref_bits, ref_cls, q, rows, k=TOPK):
    """Top classes for one query, given the candidate rows: distance, then votes, then the top five bins."""
    if len(rows) == 0:
        return np.zeros(0, np.int64)
    sim = R.similarity(bits[q:q + 1], ref_bits[rows])[0]
    votes = R.votes_for(sim[None, :], ref_cls[rows], min(k, len(rows)), int(ref_cls.max()) + 1)
    return R.predict(votes)[0]


def timed_fold(corpus, layer, p, radius, budget, repeats=3, seed=0, stats=None):
    """One leave-one-document-out pass with both arms timed on identical queries.

    Returns per-position records of (full prediction, index prediction, candidates scanned) and the **median** seconds per
    query for each arm. `repeats` takes the best of several rounds per query, because a single sample of a
    tens-of-microseconds call is mostly scheduler noise; `stats` collects the per-query sample vectors so the caller can
    report the spread too, rather than a median with nothing beside it.
    """
    bits = corpus.layer(layer)
    recs, t_full, t_idx, build_s = [], [], [], []
    for d in np.unique(corpus.doc_of):
        test = np.nonzero(corpus.doc_of == d)[0]
        train = np.nonzero(corpus.doc_of != d)[0]
        ref_bits, ref_cls = bits[train], corpus.cls[train]
        t0 = time.perf_counter()
        index = Index(ref_bits, p, seed)
        build_s.append(time.perf_counter() - t0)
        for q in test:
            best_full = best_idx = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                full = answer(bits, ref_bits, ref_cls, q, np.arange(len(ref_bits)))
                best_full = min(best_full, time.perf_counter() - t0)
                t0 = time.perf_counter()
                rows, scanned = index.candidates(bits[q], radius, budget)
                got = answer(bits, ref_bits, ref_cls, q, rows)
                best_idx = min(best_idx, time.perf_counter() - t0)
            t_full.append(best_full)
            t_idx.append(best_idx)
            recs.append((q, full, got, scanned))
    if stats is not None:
        stats["full"] = t_full
        stats["idx"] = t_idx
        stats["build"] = build_s
    return recs, float(np.median(t_full)), float(np.median(t_idx)), float(np.median(build_s))


def hits(recs, corpus, k=TOPK):
    """hit1/hit5 for both arms over the recorded queries, plus how often they agree exactly."""
    n = h1f = h5f = h1i = h5i = same = 0
    for q, full, got, _ in recs:
        n += 1
        cls, top5 = int(corpus.cls[q]), set(corpus.top5[q].tolist())
        h1f += int(cls in set(full.tolist()))
        h5f += int(bool(top5 & set(full.tolist())))
        h1i += int(cls in set(got.tolist()))
        h5i += int(bool(top5 & set(got.tolist())))
        same += int(list(full) == list(got))
    return dict(n=n, full_hit1=h1f / n, idx_hit1=h1i / n, full_hit5=h5f / n, idx_hit5=h5i / n,
                identical=same / n, mean_scan=float(np.mean([r[3] for r in recs])))


# --- tests ------------------------------------------------------------------------------------------------------------

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def t_zero_setting_is_the_router():
    """p=0, radius 0, no budget: one cell, every training row, so the two arms must agree exactly."""
    c = R.synth_corpus(docs=6, positions=10, dim=64, nclass=12, seed=71, layer_signal=(0.9,))
    recs, _, _, _ = timed_fold(c, 0, 0, 0, None, repeats=1)
    s = hits(recs, c)
    check("index: p=0 radius 0 reproduces the full router on every query", s["identical"] == 1.0,
          "%.0f%% identical, hit1 %.3f" % (100 * s["identical"], s["full_hit1"]))


def t_budget_is_whole_cells():
    """The budget cuts by cells, never mid-cell, so a scan may overshoot by at most one cell's worth."""
    c = R.synth_corpus(docs=6, positions=10, dim=64, nclass=12, seed=72, layer_signal=(0.9,))
    recs, _, _, _ = timed_fold(c, 0, 6, 3, 16, repeats=1)
    worst = max(r[3] for r in recs)
    check("index: a budgeted scan stops by whole cells", worst <= 16 + 10, "worst %d candidates" % worst)


def t_build_is_deterministic():
    c = R.synth_corpus(docs=5, positions=8, dim=48, nclass=8, seed=73, layer_signal=(0.9,))
    bits = c.layer(0)
    a, b = Index(bits, 8, seed=3), Index(bits, 8, seed=3)
    check("index: the same bits and seed build the same keys",
          np.array_equal(a.keys, b.keys) and np.array_equal(a.rows, b.rows))


def t_probe_finds_a_bit_flip():
    """A concrete neighbour that a single cell cannot see: flip one probe bit of the query, and radius 1 must find it.

    The width is 32 and `p=32` so *every* coordinate is probed: at `p=8` the key reads a permutation's first eight columns,
    and a fixture that writes only the first eight leaves the rest zero, so all three rows share one key and the check
    silently measures nothing (it reported 3 candidates at radius 0 and read as a failure of the probe).
    """
    bits = np.zeros((3, 32), np.uint8)
    bits[0] = [1, 0, 1, 1, 0, 0, 1, 0] * 4
    bits[1] = bits[0].copy()
    bits[1, 11] = 1 - bits[1, 11]                      # differs from row 0 in exactly one coordinate
    bits[2] = [1, 1, 1, 0, 1, 1, 0, 1] * 4
    index = Index(bits, 32, seed=0)
    exact, n0 = index.candidates(bits[0], 0, None)
    near, n1 = index.candidates(bits[0], 1, None)
    check("index: radius 0 cannot see a one-bit neighbour, radius 1 can",
          n0 == 1 and n1 == 2 and set(near.tolist()) == {0, 1},
          "radius 0 -> %d, radius 1 -> %d" % (n0, n1))


def t_vectorised_gather_matches_the_loop():
    """The fast gather must visit the same cells as the obvious one - at every radius and budget, on real codes."""
    c = R.synth_corpus(docs=8, positions=12, dim=192, nclass=20, seed=76, layer_signal=(0.5, 0.9))
    bits = c.layer(1)
    index = Index(bits, 8, seed=0)
    bad = []
    for q in range(0, len(bits), 7):
        for radius in (0, 1, 2, 3):
            for budget in (None, 8, 40):
                fast, nf = index.candidates(bits[q], radius, budget)
                slow, ns = index.candidates_loop(bits[q], radius, budget)
                if nf != ns or not np.array_equal(fast, slow):
                    bad.append((q, radius, budget, nf, ns))
    check("index: the vectorised gather agrees with the loop exactly", not bad, "%d disagreements" % len(bad))


def t_candidate_ids_are_training_rows():
    """The regression cell.py's identity test caught: ids index the training subset, never the full corpus."""
    c = R.synth_corpus(docs=6, positions=10, dim=64, nclass=10, seed=74, layer_signal=(0.9,))
    bits = c.layer(0)
    train = np.nonzero(c.doc_of != 0)[0]
    index = Index(bits[train], 6, seed=0)
    rows, _ = index.candidates(bits[np.nonzero(c.doc_of == 0)[0][0]], 2, None)
    check("index: candidate ids stay inside the training rows", len(rows) == 0 or int(rows.max()) < len(train),
          "max id %s of %d training rows" % (int(rows.max()) if len(rows) else "-", len(train)))


def t_budget_is_a_count_not_a_fraction():
    """The mistake that produced "22.6x faster" at the note's size: a fixed budget covers a shrinking share of a growing corpus.

    Same `p`, radius and budget, four times the positions. The absolute scan stays near the budget while its share of the
    corpus falls by more than half - which is why the small-corpus retention does not follow the setting to a bigger one.
    """
    shares = {}
    for docs in (24, 96):                         # 384 and 1,536 positions: both large enough to fill the budget
        c = R.synth_corpus(docs=docs, positions=16, dim=64, nclass=40, seed=77, layer_signal=(0.5, 0.9))
        bits = c.layer(1)
        index = Index(bits, 8, seed=0)
        scans = [index.candidates(bits[q], 3, 96)[1] for q in range(0, len(bits), 5)]
        shares[len(bits)] = (float(np.mean(scans)), float(np.mean(scans)) / len(bits))
    (n_small, (scan_small, frac_small)), (n_big, (scan_big, frac_big)) = sorted(shares.items())
    check("index: the same budget covers a smaller share as the corpus grows",
          frac_big < 0.6 * frac_small and scan_big < 2 * scan_small,
          "%d positions: %d scanned (%.0f%%); %d positions: %d scanned (%.0f%%)"
          % (n_small, scan_small, 100 * frac_small, n_big, scan_big, 100 * frac_big))


def t_the_crossover_is_at_the_real_width():
    """Where the index starts to pay, measured on both sides instead of assumed.

    A probe costs a fixed ~93 `searchsorted` calls and a gather; the full arm costs one matmul of `N x D`. At 192 bits the
    fixed cost wins and the index is several times *slower*; at the 2,048 bits the fingerprint actually has, it is faster.
    Asserting only the favourable side would have hidden the regime in which this whole idea loses.
    """
    narrow = R.synth_corpus(docs=10, positions=16, dim=192, nclass=40, seed=75, layer_signal=(0.5, 0.92))
    middle = R.synth_corpus(docs=10, positions=16, dim=1024, nclass=40, seed=75, layer_signal=(0.5, 0.92))
    wide = R.synth_corpus(docs=10, positions=16, dim=2048, nclass=40, seed=75, layer_signal=(0.5, 0.92))
    _, tf_n, ti_n, _ = timed_fold(narrow, 1, 8, 3, 96, repeats=2)
    _, tf_m, ti_m, _ = timed_fold(middle, 1, 8, 3, 96, repeats=2)
    _, tf_w, ti_w, _ = timed_fold(wide, 1, 8, 3, 96, repeats=2)
    check("index: at 192 bits the full 20-NN is still the cheaper arm", tf_n < ti_n,
          "full %.1f us vs index %.1f us" % (1e6 * tf_n, 1e6 * ti_n))
    check("index: at 1024 bits the vectorised gather already pays", ti_m < tf_m,
          "index %.1f us vs full %.1f us (%.1fx)" % (1e6 * ti_m, 1e6 * tf_m, tf_m / ti_m))
    check("index: at 2048 bits the built index answers faster than the full 20-NN", ti_w < tf_w,
          "index %.1f us vs full %.1f us (%.1fx)" % (1e6 * ti_w, 1e6 * tf_w, tf_w / ti_w))


def selftest():
    del PASS[:]
    for t in (t_zero_setting_is_the_router, t_budget_is_whole_cells, t_build_is_deterministic,
              t_probe_finds_a_bit_flip, t_vectorised_gather_matches_the_loop, t_candidate_ids_are_training_rows,
              t_budget_is_a_count_not_a_fraction, t_the_crossover_is_at_the_real_width):
        t()
    expected = 10
    ok = len(PASS) == expected
    print("\n%d checks passed%s" % (len(PASS), "" if ok else " of %d - a check did not run" % expected))
    return 0 if ok else 1


def report(corpus, layer):
    ref, _ = C.full_reference(corpus, layer)
    print("corpus: %d positions, %d documents, %d bits, %d classes; full 20-NN hit1 %.4f"
          % (len(corpus.cls), len(np.unique(corpus.doc_of)), corpus.bits.shape[2], corpus.nclass, ref))
    print("\n  %-14s %8s %10s %10s %8s %8s %9s" % ("setting", "hit1", "us/query", "full us", "scan%", "build ms",
                                                   "identical"))
    for p, radius, budget in ((0, 0, None), (8, 2, 96), (8, 3, 96), (8, 3, None)):
        stats = {}
        recs, t_full, t_idx, build = timed_fold(corpus, layer, p, radius, budget, repeats=3, stats=stats)
        s = hits(recs, corpus)
        def spread(v):
            return float(np.std(v) / max(np.mean(v), 1e-12))
        print("  p%-2d r%d b%-6s %8.4f %8.1f %10.1f %8.1f %8.1f %9.0f%%"
              % (p, radius, "all" if budget is None else budget, s["idx_hit1"], 1e6 * t_idx, 1e6 * t_full,
                 100 * s["mean_scan"] / len(corpus.cls), 1e3 * build, 100 * s["identical"]))
        if p or radius or budget is not None:
            c2 = compare(dict(median=t_full, rel=spread(stats["full"])), dict(median=t_idx, rel=spread(stats["idx"])),
                         "full", "index")
            print("  %-14s %8s spread %.0f%% full / %.0f%% index -> %s"
                  % ("", "", 100 * spread(stats["full"]), 100 * spread(stats["idx"]), c2["text"]))
    print("\nThe p=0 row is the router itself, so its 'speedup' is 1.0 by construction: it exists to show the timed arm\n"
          "computes the same predictions. Anything above it is an approximation, and cheap only counts if it holds up.")
    return 0


def scale_probe(positions=3456, dim=2048, queries=60, seed=61, settings=None):
    """The same comparison at the note's corpus size (48 documents x 72 positions), sampled rather than exhaustive.

    This exists because the sweep's answer **does not transfer unchanged**. A budget is a count, and the sweep held it at 96
    while the corpus grew three positions at a time: at N=176 that is 55% of everything, at N=3,384 it is 3%, so the first
    scale run reported "22.6x faster" with hit1 fallen from 0.986 to 0.306. Sweeping the budget here shows the real curve:
    the retention the sweep promised costs about a third of a scan at this size, not a quarter.
    """
    if settings is None:
        settings = ((8, 3, 96), (8, 3, 512), (8, 3, 1024), (8, 3, 2048), (12, 3, 1024))
    c = R.synth_corpus(docs=48, positions=positions // 48, dim=dim, nclass=751, seed=seed,
                       layer_signal=(0.4, 0.6, 0.8))
    bits = c.layer(2)
    test = np.nonzero(c.doc_of == 0)[0]
    train = np.nonzero(c.doc_of != 0)[0]
    ref_bits, ref_cls = bits[train], c.cls[train]
    picks = np.random.RandomState(0).choice(test, size=min(queries, len(test)), replace=False)
    every = np.arange(len(ref_bits))
    tf, full_hit = [], {}
    for q in picks:
        t0 = time.perf_counter()
        full = answer(bits, ref_bits, ref_cls, q, every)
        tf.append(time.perf_counter() - t0)
        full_hit[q] = c.cls[q] in set(full.tolist())
    ref_rate = float(np.mean(list(full_hit.values())))
    rows = []
    for p, radius, budget in settings:
        idx = Index(ref_bits, p, seed=0)
        hit, scan, t = [], [], []
        for q in picks:
            t0 = time.perf_counter()
            cand, n = idx.candidates(bits[q], radius, budget)
            pred = answer(bits, ref_bits, ref_cls, q, cand)
            t.append(time.perf_counter() - t0)
            scan.append(n)
            hit.append(c.cls[q] in set(pred.tolist()))
        rows.append(dict(probe=p, radius=radius, budget=budget, hit1=float(np.mean(hit)),
                         retention=float(np.mean(hit) / max(ref_rate, 1e-12)),
                         scan=float(np.mean(scan)), scan_frac=float(np.mean(scan) / len(train)),
                         us=1e6 * float(np.mean(t)), us_frac=float(np.mean(tf) / np.mean(t))))
    return dict(n=len(train), queries=len(picks), full_us=1e6 * float(np.mean(tf)), full_hit1=ref_rate, rows=rows)


def main(argv):
    if "--scale" in argv:
        s = scale_probe()
        print("at the note's corpus size: %d training rows, 2,048 bits, %d queries from one held-out document"
              % (s["n"], s["queries"]))
        print("full 20-NN over everything: %.0f us/query, hit1 %.3f\n" % (s["full_us"], s["full_hit1"]))
        print("  %-14s %7s %9s %7s %9s %8s" % ("setting", "hit1", "retain", "scan%", "us/query", "speedup"))
        for r in s["rows"]:
            print("  p%-2d r%d b%-6d %7.3f %9.3f %6.0f%% %9.0f %7.2fx"
                  % (r["probe"], r["radius"], r["budget"], r["hit1"], r["retention"], 100 * r["scan_frac"],
                     r["us"], r["us_frac"]))
        print("\nThe budget is a count, not a fraction: at 96 candidates the index visits 3%% of this corpus, so the retention\n"
              "it showed on a 176-row corpus does not survive the growth. The %d queries are a reading, not an interval."
              % s["queries"])
        return 0
    if "--corpus" in argv:
        c = R.load_corpus(argv[argv.index("--corpus") + 1])
        if c is None:
            print("no fingerprint corpus there")
            return 2
        return report(c, c.layers[-1])
    # The finding is a crossover, so the default report is the table that shows it: same setting, four code widths.
    print("multi-probe index vs the full 20-NN, same queries, best of 3 rounds (p=8, radius 3, budget 96)\n")
    print("  %-18s %6s %10s %10s %8s %8s" % ("corpus", "N", "full us", "index us", "ratio", "scan%"))
    for dim, pos in ((192, 16), (1024, 16), (2048, 16), (2048, 32)):
        c = R.synth_corpus(docs=12, positions=pos, dim=dim, nclass=40, seed=61, layer_signal=(0.5, 0.92))
        stats = {}
        recs, t_full, t_idx, build = timed_fold(c, 1, 8, 3, 96, repeats=3, stats=stats)
        s = hits(recs, c)
        n_train = len(c.cls) - pos
        print("  %-18s %6d %10.1f %10.1f %8.2fx %7.0f%%"
              % ("%d bits, %d pos" % (dim, pos), n_train, 1e6 * t_full, 1e6 * t_idx, t_idx / t_full,
                 100 * s["mean_scan"] / n_train))
        sp = (float(np.std(stats["full"]) / max(np.mean(stats["full"]), 1e-12)),
              float(np.std(stats["idx"]) / max(np.mean(stats["idx"]), 1e-12)))
        verdict = compare(dict(median=t_full, rel=sp[0]), dict(median=t_idx, rel=sp[1]), "full", "index")
        print("  %-18s %6s hit1 %.3f (index) vs %.3f (full), %.0f%% of predictions identical, build %.1f ms"
              % ("", "", s["idx_hit1"], s["full_hit1"], 100 * s["identical"], 1e3 * build))
        print("  %-18s %6s spread %.0f%% full / %.0f%% index -> %s"
              % ("", "", 100 * sp[0], 100 * sp[1], verdict["text"]))
    print("\nBelow the crossover the probe's fixed cost (~93 key lookups and a gather) is larger than the matmul it replaces;\n"
          "above it the saved multiply-adds win, and the saving grows with N because the full arm's cost is O(N x D).")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main(sys.argv[1:]))
