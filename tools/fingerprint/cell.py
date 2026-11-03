"""Cell-plus-NN: hash to a cell on `p` probe bits, then run the NN inside it, and sweep the cost.

The note's negative result is that compressing the fingerprint costs more than it saves: a table indexed by 8 coordinates,
or a codebook at k = 64/256/1024, lands at 0.29-0.35 hit1 against 0.46-0.54 for 20-NN over the whole code. This module
measures the shape of that trade rather than repeating the verdict, and it answers one question with a rule instead of an
opinion: **what is the cheapest (probe bits, candidate budget) that keeps 95% of the full-router hit rate?**

Definitions, because the numbers are meaningless without them:

* **cell** = the set of training positions whose first `p` bits (under one fixed random permutation of the coordinates)
  are identical to the query's. Nothing about the cell looks at the query's other 2,048 - p bits.
* **candidate budget B** = the most candidates the exact NN is allowed to scan. A cell larger than B is *split*, not
  truncated: the key is extended in steps of 4 bits until the bucket fits, or 32 bits is reached. Randomly dropping
  candidates would make the miss rate an artifact of insertion order; splitting makes it a property of the bits.
* **cost** = mean candidates scanned x D bits, plus p bits for the key. Full 20-NN costs N x D + 0.

Splitting has a price of its own, and it is the interesting part: the deeper the split, the fewer true neighbours remain
in the cell, so hit rate falls as the budget tightens. The sweep reports that curve, and `select()` picks the cheapest
point on it that keeps 95% of the full router's hit1.

    python -m tools.fingerprint.cell            # sweep + the selected setting
    python -m tools.fingerprint.cell --selftest
"""
import json
import os
import sys

import numpy as np

from . import router as R

PROBES = (0, 4, 8, 12, 16, 24, 32)
BUDGETS = (16, 32, 64, 128, 256, None)
MAX_KEY_BITS = 32                      # beyond int64 packing; a real index would hash, and this one would overflow


def bit_order(dim, seed=0):
    """One fixed permutation of the coordinates, so every probe count uses a nested prefix of the same bits."""
    return np.random.RandomState(seed).permutation(dim)


def key_of(bits, order, count):
    """Pack `count` bits into one int64. Returns zeros for count 0, which is a single cell holding everything."""
    if count <= 0:
        return np.zeros(len(bits), np.int64)
    cols = order[:count]
    weights = np.int64(1) << np.arange(count, dtype=np.int64)
    return bits[:, cols].astype(np.int64) @ weights


def level_counts(probe):
    """The nested key lengths tried for one probe setting: `probe` first, then +4 bits until the cap."""
    counts = [probe]
    c = probe + 4
    while c <= MAX_KEY_BITS:
        counts.append(c)
        c += 4
    return counts


class CellIndex(object):
    """Sorted keys per level, so a query's bucket is two searchsorted calls."""

    def __init__(self, ref_bits, probe, order, step=4):
        self.counts = [c for c in level_counts(probe) if c <= ref_bits.shape[1]]
        self.maps = []
        for c in self.counts:
            k = key_of(ref_bits, order, c)
            order_idx = np.argsort(k, kind="stable")
            self.maps.append((k[order_idx], order_idx))

    def bounds(self, q_bits, order):
        """Per query, the (lo, hi) slice of each level's bucket. Computed once, reused for every budget."""
        out = []
        for row in range(q_bits.shape[0]):
            per_level = []
            for c, (ks, _) in zip(self.counts, self.maps):
                qk = key_of(q_bits[row:row + 1], order, c)[0]
                per_level.append((int(np.searchsorted(ks, qk, "left")), int(np.searchsorted(ks, qk, "right"))))
            out.append(per_level)
        return out

    def candidates(self, per_query_bounds, budget):
        """Pick the deepest bucket that fits the budget. Returns (index arrays, overflow count)."""
        cands, overflow = [], 0
        for per_level in per_query_bounds:
            chosen = None
            for (lo, hi), (_, idx) in zip(per_level, self.maps):
                if hi > lo and (budget is None or hi - lo <= budget):
                    chosen = idx[lo:hi]
                    break
            if chosen is None:                      # even 32 bits does not fit: take what fits, and count it
                lo, hi = per_level[-1]
                _, idx = self.maps[-1]
                chosen = idx[lo:min(hi, lo + (budget or len(idx)))] if hi > lo else idx[:budget or len(idx)]
                overflow += 1
            cands.append(chosen)
        return cands, overflow


def score_fold(q_bits, ref_bits, ref_cls, cls, top5, test, cands, k=20):
    """hit1/hit5 using 20-NN restricted to each query's candidates.

    `ref_*` are the training positions the cell index was built over, so a candidate id is a position in *those* arrays.
    Indexing the full corpus's labels with it silently votes with another document's classes - which is what the identity
    test caught on its first run.
    """
    h1 = h5 = 0
    nclass = int(cls.max()) + 1
    for row, t in enumerate(test):
        pool = np.sort(cands[row]) if len(cands[row]) else cands[row]
        if len(pool) == 0:
            continue
        sim = R.similarity(q_bits[t:t + 1], ref_bits[pool])[0]
        pred = set(R.predict(R.votes_for(sim[None, :], ref_cls[pool], min(k, len(pool)), nclass))[0].tolist())
        h1 += int(cls[t] in pred)
        h5 += int(len(set(top5[t].tolist()) & pred) > 0)
    return h1, h5, len(test)


def leave_one_document_out(corpus, layer, probe, budget, seed=0, step=4):
    """Per-document hits and the mean number of candidates scanned, for one (probe, budget)."""
    bits = corpus.layer(layer)
    order = bit_order(bits.shape[1], seed)
    out = {}
    for d in np.unique(corpus.doc_of):
        test = np.nonzero(corpus.doc_of == d)[0]
        train = np.nonzero(corpus.doc_of != d)[0]
        ref_bits, ref_cls = bits[train], corpus.cls[train]
        index = CellIndex(ref_bits, probe, order, step=step)
        bounds = index.bounds(bits[test], order)
        cands, _ = index.candidates(bounds, budget)
        h1, h5, n = score_fold(bits, ref_bits, ref_cls, corpus.cls, corpus.top5, test, cands)
        out[int(d)] = (h1, h5, n, float(np.mean([len(c) for c in cands])) if cands else 0.0)
    return out


def full_reference(corpus, layer, k=20):
    """The router this is being measured against: 20-NN over all N candidates."""
    per = R.leave_one_document_out(corpus, layer, k)
    return sum(v[0] for v in per.values()) / sum(v[2] for v in per.values()), per


def sweep(corpus, layer, probes=PROBES, budgets=BUDGETS, k=20, seed=0):
    """Every (probe, budget), scored leave-one-document-out, with cost and retention against the full router."""
    ref_rate, ref_per = full_reference(corpus, layer, k)
    n_train = len(corpus.cls) - int(np.bincount(corpus.doc_of).min())
    rows = []
    for p in probes:
        for b in budgets:
            per = leave_one_document_out(corpus, layer, p, b, seed=seed)
            h1 = sum(v[0] for v in per.values()) / sum(v[2] for v in per.values())
            h5 = sum(v[1] for v in per.values()) / sum(v[2] for v in per.values())
            scan = float(np.mean([v[3] for v in per.values()]))
            rows.append(dict(probe=p, budget=b, hit1=round(h1, 4), hit5=round(h5, 4),
                             retention=round(h1 / max(ref_rate, 1e-12), 4),
                             scan=round(scan, 1), cost_bits=round(scan * corpus.bits.shape[2] + p),
                             full_scan=n_train))
    return ref_rate, rows


def select(rows, keep=0.95):
    """The cheapest row that keeps `keep` of the full router's hit1. Ties break toward smaller probe, then budget."""
    ok = [r for r in rows if r["retention"] >= keep]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["cost_bits"], r["probe"], r["budget"] or 10 ** 9))


def nested_selection(corpus, layer, keep=0.95, k=20, seed=0, probes=PROBES, budgets=BUDGETS):
    """Choose (probe, budget) for each document on the *other* documents, then score the document with it.

    The sweep's cheapest row is chosen on all documents at once; this is the version that would survive review, because a
    document's settings were never chosen by looking at its own score.
    """
    docs = [int(d) for d in np.unique(corpus.doc_of)]
    picked = {}
    for d in docs:
        # a corpus restricted to the other documents, with its own class labels, for the inner selection
        others = [x for x in docs if x != d]
        mask = np.isin(corpus.doc_of, others)
        sub = R.Corpus(corpus.bits[:, mask], corpus.cls[mask], corpus.top5[mask],
                       _reindex(corpus.doc_of[mask]), corpus.prev[mask], layers=corpus.layers)
        sub.docs = others
        rate, rows = sweep(sub, layer, probes=probes, budgets=budgets, k=k, seed=seed)
        best = select(rows, keep)
        if best is None:
            best = dict(probe=0, budget=None, hit1=1.0)
        per = leave_one_document_out(_single_doc(corpus, d, others[0]), layer, best["probe"], best["budget"], seed=seed)
        h1, h5, n, scan = per[d]
        picked[d] = dict(probe=best["probe"], budget=best["budget"], hit1=h1, hit5=h5, n=n, scan=scan,
                         inner=best["hit1"])
    return picked


def _reindex(doc_of):
    uniq, inv = np.unique(doc_of, return_inverse=True)
    return inv.astype(np.int64)


def _single_doc(corpus, doc, other):
    """A corpus holding `doc` as the fold and `other` as a stand-in training label, so folds stay well defined."""
    mask = corpus.doc_of == doc
    bits = corpus.bits[:, mask]
    doc_of = np.zeros(int(mask.sum()), np.int64)
    return R.Corpus(bits, corpus.cls[mask], corpus.top5[mask], doc_of, corpus.prev[mask], layers=corpus.layers)


# --- tests ------------------------------------------------------------------------------------------------------------

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def t_single_cell_is_full():
    """probe=0, no budget is one cell holding every training position: it must equal the full router exactly."""
    c = R.synth_corpus(docs=6, positions=12, dim=96, nclass=12, seed=11, layer_signal=(0.85,))
    cell = leave_one_document_out(c, 0, probe=0, budget=None)
    full = R.leave_one_document_out(c, 0, 20)
    same_hits = all(cell[d][0] == full[d][0] and cell[d][2] == full[d][2] for d in full)
    check("identity: probe 0 with no budget equals the full 20-NN", same_hits,
          "cell %d vs full %d" % (sum(v[0] for v in cell.values()), sum(v[0] for v in full.values())))
    check("identity: and it scans every candidate", max(v[3] for v in cell.values()) > 0,
          "mean scanned %.1f" % float(np.mean([v[3] for v in cell.values()])))


def t_splitting_respects_budget():
    """No query may scan more candidates than the budget says, because a cell that overflows is split."""
    c = R.synth_corpus(docs=6, positions=12, dim=96, nclass=12, seed=12, layer_signal=(0.85,))
    per = leave_one_document_out(c, 0, probe=0, budget=8)
    worst = max(v[3] for v in per.values())
    check("budget: a split cell never exceeds the candidate budget", worst <= 8.0, "mean scanned %.1f" % worst)


def t_cost_is_monotone():
    """Larger budget, or fewer probe bits, can only scan more candidates - never fewer."""
    c = R.synth_corpus(docs=6, positions=12, dim=96, nclass=12, seed=13, layer_signal=(0.85,))
    scans = []
    for b in (8, 16, 32, 64):
        per = leave_one_document_out(c, 0, probe=0, budget=b)
        scans.append(float(np.mean([v[3] for v in per.values()])))
    check("cost: increasing the budget never lowers the scan", all(scans[i] <= scans[i + 1] + 1e-9
                                                                  for i in range(len(scans) - 1)),
          "scans %s" % [round(s, 1) for s in scans])


def t_probe_regimes():
    """Deeper probes do not lower the hit rate when the bits carry the class, and cannot rescue it when they do not.

    The first version of this test asserted that more probe bits always lose neighbours. It failed at 0.588 against 0.250,
    and the measurement was right: on informative bits a deep cell is a set of near-identical codes, so the NN inside it
    is a *better* neighbourhood than 4 random bits over the whole corpus. Fragmentation only costs when the bits are noise.
    """
    info = R.synth_corpus(docs=10, positions=16, dim=192, nclass=40, seed=14, layer_signal=(0.5, 0.9))
    h = {}
    for p in (0, 24):
        per = leave_one_document_out(info, 1, probe=p, budget=64)
        h[p] = sum(v[0] for v in per.values()) / sum(v[2] for v in per.values())
    check("probe: on informative bits a deep cell beats a shallow one", h[24] >= h[0] - 0.02,
          "probe 0 %.3f, probe 24 %.3f" % (h[0], h[24]))
    noise = R.synth_corpus(docs=10, positions=16, dim=192, nclass=40, seed=14, layer_signal=(0.5,))
    hn = {}
    for p in (0, 24):
        per = leave_one_document_out(noise, 0, probe=p, budget=64)
        hn[p] = sum(v[0] for v in per.values()) / sum(v[2] for v in per.values())
    check("probe: on noise a deep cell cannot rescue the router", abs(hn[24] - hn[0]) < 0.1 and hn[24] < 0.35,
          "probe 0 %.3f, probe 24 %.3f, chance %.3f" % (hn[0], hn[24], R.TOPK / noise.nclass))


def t_selection_rule():
    """The rule returns the cheapest setting that keeps the bar - and None when the grid cannot, out loud.

    The first version asserted that a 95% setting always exists. It does not: on this corpus the whole cell grid loses
    more than 5% of the full router's hit rate, which is the note's own conclusion arriving from the measurement rather
    than from a summary of it.
    """
    c = R.synth_corpus(docs=10, positions=16, dim=192, nclass=40, seed=15, layer_signal=(0.5, 0.9))
    ref, rows = sweep(c, 1, probes=(0, 8, 16, 24, 32), budgets=(8, 16, 32, 64, 128, None))
    best = select(rows)
    check("select: with a full-cell option available a 95% setting is found",
          best is not None and best["retention"] >= 0.95,
          "none found" if best is None else "probe %d, budget %s, retention %.3f" % (best["probe"], best["budget"],
                                                                                   best["retention"]))
    cheaper = [r for r in rows if r["retention"] >= 0.95 and best and r["cost_bits"] < best["cost_bits"]]
    check("select: nothing cheaper also keeps 95%", not cheaper, "%d cheaper rows" % len(cheaper))
    if best:
        check("select: the chosen setting scans fewer candidates than the full index",
              best["scan"] <= best["full_scan"], "scan %.1f of %d" % (best["scan"], best["full_scan"]))
    tight_ref, tight = sweep(c, 1, probes=(0, 8, 16, 24, 32), budgets=(8, 16, 32))
    check("select: a grid that cannot reach 95% returns None rather than the best available", select(tight) is None,
          "best retention in the tight grid %.3f" % max(r["retention"] for r in tight))
    strict = select(rows, keep=2.0)
    check("select: an impossible bar returns nothing", strict is None)
    loose = select(rows, keep=0.0)
    check("select: with no retention floor it returns the cheapest row",
          loose is not None and loose["cost_bits"] == min(r["cost_bits"] for r in rows),
          "cost %.0f" % (loose or {}).get("cost_bits", -1))


def selftest():
    del PASS[:]
    for t in (t_single_cell_is_full, t_splitting_respects_budget, t_cost_is_monotone, t_probe_regimes,
              t_selection_rule):
        t()
    expected = 12
    ok = len(PASS) == expected
    print("\n%d checks passed%s" % (len(PASS), "" if ok else " of %d - a check did not run" % expected))
    return 0 if ok else 1


def report(corpus, layer, note=""):
    ref, rows = sweep(corpus, layer)
    best = select(rows)
    print("corpus: %d positions, %d documents, %d bits, %d classes%s"
          % (len(corpus.cls), len(np.unique(corpus.doc_of)), corpus.bits.shape[2], corpus.nclass, note))
    print("layer %d, reference (20-NN over everything): hit1 %.4f" % (layer, ref))
    print("\n  %-6s %-8s %8s %8s %8s %8s" % ("probe", "budget", "hit1", "retain", "scan", "cost"))
    for r in rows:
        mark = " <-" if best and (r["probe"], r["budget"]) == (best["probe"], best["budget"]) else ""
        print("  %-6d %-8s %8.4f %8.3f %8.1f %8.0f%s"
              % (r["probe"], "-" if r["budget"] is None else r["budget"], r["hit1"], r["retention"], r["scan"],
                 r["cost_bits"], mark))
    if best:
        print("\ncheapest >=95%%: probe %d, budget %s - hit1 %.4f (%.1f%% of full), scans %.1f of %d candidates"
              % (best["probe"], best["budget"], best["hit1"], 100 * best["retention"], best["scan"], best["full_scan"]))
    else:
        print("\nno setting kept 95% of the full hit rate")
    return best


def main(argv):
    if "--corpus" in argv:
        path = argv[argv.index("--corpus") + 1]
        c = R.load_corpus(path)
        if c is None:
            print("no fingerprint corpus at %s" % path)
            return 2
        return 0 if report(c, c.layers[-1]) else 0
    c = R.synth_corpus(docs=10, positions=16, dim=192, nclass=40, seed=21, layer_signal=(0.5, 0.72, 0.9))
    report(c, 2, " (synthetic: layer 2 holds the signal)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main(sys.argv[1:]))
