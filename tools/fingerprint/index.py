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

    def candidates(self, q_bits, radius, budget):
        """Training rows in probe order, cut at the budget by whole cells. Returns (rows, scanned)."""
        qkey = C.key_of(q_bits[None, :], self.order, self.p)[0]
        out = []
        for key in P.probe_keys(qkey, self.p, radius):
            if budget is not None and len(out) and sum(len(c) for c in out) >= budget:
                break
            lo = int(np.searchsorted(self.keys, key, "left"))
            hi = int(np.searchsorted(self.keys, key, "right"))
            if hi > lo:
                out.append(self.rows[lo:hi])
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


def timed_fold(corpus, layer, p, radius, budget, repeats=3, seed=0):
    """One leave-one-document-out pass with both arms timed on identical queries.

    Returns per-position records of (full prediction, index prediction, candidates scanned) and the mean seconds per
    query for each arm. `repeats` takes the best of several rounds, because a 40 microsecond measurement on a shared
    machine is mostly scheduler noise otherwise.
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
    return recs, float(np.mean(t_full)), float(np.mean(t_idx)), float(np.mean(build_s))


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


def t_candidate_ids_are_training_rows():
    """The regression cell.py's identity test caught: ids index the training subset, never the full corpus."""
    c = R.synth_corpus(docs=6, positions=10, dim=64, nclass=10, seed=74, layer_signal=(0.9,))
    bits = c.layer(0)
    train = np.nonzero(c.doc_of != 0)[0]
    index = Index(bits[train], 6, seed=0)
    rows, _ = index.candidates(bits[np.nonzero(c.doc_of == 0)[0][0]], 2, None)
    check("index: candidate ids stay inside the training rows", len(rows) == 0 or int(rows.max()) < len(train),
          "max id %s of %d training rows" % (int(rows.max()) if len(rows) else "-", len(train)))


def t_the_crossover_is_at_the_real_width():
    """Where the index starts to pay, measured on both sides instead of assumed.

    A probe costs a fixed ~93 `searchsorted` calls and a gather; the full arm costs one matmul of `N x D`. At 192 bits the
    fixed cost wins and the index is several times *slower*; at the 2,048 bits the fingerprint actually has, it is faster.
    Asserting only the favourable side would have hidden the regime in which this whole idea loses.
    """
    narrow = R.synth_corpus(docs=10, positions=16, dim=192, nclass=40, seed=75, layer_signal=(0.5, 0.92))
    wide = R.synth_corpus(docs=10, positions=16, dim=2048, nclass=40, seed=75, layer_signal=(0.5, 0.92))
    _, tf_n, ti_n, _ = timed_fold(narrow, 1, 8, 3, 96, repeats=2)
    _, tf_w, ti_w, _ = timed_fold(wide, 1, 8, 3, 96, repeats=2)
    check("index: at 192 bits the full 20-NN is still the cheaper arm", tf_n < ti_n,
          "full %.1f us vs index %.1f us" % (1e6 * tf_n, 1e6 * ti_n))
    check("index: at 2048 bits the built index answers faster than the full 20-NN", ti_w < tf_w,
          "index %.1f us vs full %.1f us (%.1fx)" % (1e6 * ti_w, 1e6 * tf_w, tf_w / ti_w))


def selftest():
    del PASS[:]
    for t in (t_zero_setting_is_the_router, t_budget_is_whole_cells, t_build_is_deterministic,
              t_probe_finds_a_bit_flip, t_candidate_ids_are_training_rows, t_the_crossover_is_at_the_real_width):
        t()
    expected = 7
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
        recs, t_full, t_idx, build = timed_fold(corpus, layer, p, radius, budget, repeats=3)
        s = hits(recs, corpus)
        print("  p%-2d r%d b%-6s %8.4f %10.1f %10.1f %8.1f %8.1f %9.0f%%"
              % (p, radius, "all" if budget is None else budget, s["idx_hit1"], 1e6 * t_idx, 1e6 * t_full,
                 100 * s["mean_scan"] / len(corpus.cls), 1e3 * build, 100 * s["identical"]))
    print("\nThe p=0 row is the router itself, so its 'speedup' is 1.0 by construction: it exists to show the timed arm\n"
          "computes the same predictions. Anything above it is an approximation, and cheap only counts if it holds up.")
    return 0


def main(argv):
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
        recs, t_full, t_idx, build = timed_fold(c, 1, 8, 3, 96, repeats=3)
        s = hits(recs, c)
        n_train = len(c.cls) - pos
        print("  %-18s %6d %10.1f %10.1f %8.2fx %7.0f%%"
              % ("%d bits, %d pos" % (dim, pos), n_train, 1e6 * t_full, 1e6 * t_idx, t_idx / t_full,
                 100 * s["mean_scan"] / n_train))
        print("  %-18s %6s hit1 %.3f (index) vs %.3f (full), %.0f%% of predictions identical, build %.1f ms"
              % ("", "", s["idx_hit1"], s["full_hit1"], 100 * s["identical"], 1e3 * build))
    print("\nBelow the crossover the probe's fixed cost (~93 key lookups and a gather) is larger than the matmul it replaces;\n"
          "above it the saved multiply-adds win, and the saving grows with N because the full arm's cost is O(N x D).")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main(sys.argv[1:]))
