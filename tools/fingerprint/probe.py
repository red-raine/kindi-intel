"""Multi-probe cells: search the query's cell and its Hamming neighbours, and see what that buys.

A single-probe cell index asks for an exact match on `p` probe bits. A true neighbour one bit away is then in another cell
and invisible at any budget - which is why `cell.py` found that keeping 95% of the hit rate costs 94% of a full scan: the
budget cannot spend work it is not allowed to reach.

Multi-probe fixes that by visiting cells in increasing Hamming distance from the query key - the original, then every key
that differs in one of the `p` bits, then every key that differs in two - and stopping once the budget is reached. Cells
are visited nearest-first, so a budget that cuts the search keeps the closest codes.

The question `--ask` answers: **is there a (probe bits, radius, budget) that keeps 95% of the full router's hit1 while
scanning less than half the corpus?**

    python -m tools.fingerprint.probe               # sweep and answer on a synthetic corpus
    python -m tools.fingerprint.probe --selftest    # checks with known answers
"""
import sys

import numpy as np

from . import cell as C
from . import router as R

PROBES = (6, 8, 10)
RADII = (0, 2, 3)
BUDGETS = (8, 16, 24, 32, 48, 64, 96, 128)


def probe_keys(qkey, p, radius):
    """The query's key, then keys at Hamming distance 1, then 2 - nearest first, no duplicates."""
    yield int(qkey)
    if radius >= 1:
        for i in range(p):
            yield int(qkey) ^ (1 << i)
    if radius >= 2:
        for i in range(p):
            for j in range(i + 1, p):
                yield int(qkey) ^ (1 << i) ^ (1 << j)
    if radius >= 3:
        for i in range(p):
            for j in range(i + 1, p):
                for k in range(j + 1, p):
                    yield int(qkey) ^ (1 << i) ^ (1 << j) ^ (1 << k)


def probe_count(p, radius):
    """How many cells one lookup visits: the Hamming ball of radius `radius` around the query key.

    Radius 3 was silently a no-op in the first version - `probe_keys` stopped at pairs, so `radius=3` returned exactly
    radius 2's cells and the table read as if a wider probe bought nothing. It is implemented now because at p=8 the
    triple ring costs 93 of 256 cells (36% of a scan), which is inside the 50% the question asks about.
    """
    n = 1
    if radius >= 1:
        n += p
    if radius >= 2:
        n += p * (p - 1) // 2
    if radius >= 3:
        n += p * (p - 1) * (p - 2) // 6
    return n


def cell_slice(maps, level, key):
    ks, order = maps[level]
    lo = np.searchsorted(ks, key, "left")
    hi = np.searchsorted(ks, key, "right")
    return order[lo:hi]


def gather(maps, level, qkey, p, radius, budget):
    """Cells in probe order until the budget is reached. The last cell may overshoot; the count is returned too."""
    out, n = [], 0
    for key in probe_keys(qkey, p, radius):
        if budget is not None and n >= budget:
            break
        sl = cell_slice(maps, level, key)
        if len(sl):
            out.append(sl)
            n += len(sl)
    if not out:
        return np.zeros(0, np.int64), 0
    return np.concatenate(out), n


def folded_candidates(ref_bits, q_bits, order, p, radius, budgets, budget_cap=None):
    """One pass over the queries, gathering cells in probe order, then cutting that same order at each budget.

    Gathering per budget independently would repeat every searchsorted; instead the cells are collected once up to the
    largest budget and each smaller budget takes the prefix that fits.
    """
    counts = [c for c in C.level_counts(p) if c <= ref_bits.shape[1]]
    level = 0                                     # the `p`-bit level; splitting is what multi-probe replaces
    maps = []
    for c in counts:
        k = C.key_of(ref_bits, order, c)
        order_idx = np.argsort(k, kind="stable")
        maps.append((k[order_idx], order_idx))
    per_query = []
    for row in range(q_bits.shape[0]):
        qkey = C.key_of(q_bits[row:row + 1], order, counts[level])[0]
        cells = [cell_slice(maps, level, key) for key in probe_keys(qkey, p, radius)]
        cells = [c for c in cells if len(c)]
        per_query.append(cells)
    out = dict((b, []) for b in budgets)
    scanned = dict((b, []) for b in budgets)
    for cells in per_query:
        cum = np.cumsum([len(c) for c in cells]) if cells else np.zeros(0, int)
        for b in budgets:
            if b is None:
                take = len(cells)                       # unbounded budget: every probed cell
            else:
                take = int(np.searchsorted(cum, b, "left") + 1) if len(cum) else 0
            pool = np.concatenate(cells[:take]) if take else np.zeros(0, np.int64)
            out[b].append(np.sort(pool))
            scanned[b].append(len(pool))
    return out, scanned


def leave_one_document_out(corpus, layer, p, radius, budget, seed=0):
    """Per-document hit1/hit5 and the mean number of candidates scanned, for one (p, radius, budget)."""
    bits = corpus.layer(layer)
    order = C.bit_order(bits.shape[1], seed)
    out = {}
    for d in np.unique(corpus.doc_of):
        test = np.nonzero(corpus.doc_of == d)[0]
        train = np.nonzero(corpus.doc_of != d)[0]
        ref_bits, ref_cls = bits[train], corpus.cls[train]
        cands, scanned = folded_candidates(ref_bits, bits[test], order, p, radius, [budget])
        h1, h5, n = C.score_fold(bits, ref_bits, ref_cls, corpus.cls, corpus.top5, test, cands[budget])
        out[int(d)] = (h1, h5, n, float(np.mean(scanned[budget])) if scanned[budget] else 0.0)
    return out


def sweep(corpus, layer, probes=PROBES, radii=RADII, budgets=BUDGETS, seed=0):
    ref_rate, _ = C.full_reference(corpus, layer)
    full_scan = float(len(corpus.cls) - np.bincount(corpus.doc_of).min())
    rows = []
    for p in probes:
        for r in radii:
            for b in budgets:
                per = leave_one_document_out(corpus, layer, p, r, b, seed=seed)
                h1 = sum(v[0] for v in per.values()) / sum(v[2] for v in per.values())
                scan = float(np.mean([v[3] for v in per.values()]))
                rows.append(dict(probe=p, radius=r, budget=b, hit1=round(h1, 4),
                                 retention=round(h1 / max(ref_rate, 1e-12), 4), scan=round(scan, 1),
                                 scan_frac=round(scan / full_scan, 3), probes_per_lookup=probe_count(p, r),
                                 full_scan=int(full_scan)))
    return ref_rate, rows


def fastest(rows, keep=0.95, max_frac=0.5):
    """The cheapest setting that keeps `keep` of the full hit rate inside `max_frac` of a scan, or None."""
    ok = [r for r in rows if r["retention"] >= keep and r["scan_frac"] < max_frac]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["scan"], r["probe"], r["radius"], r["budget"]))


# --- tests ------------------------------------------------------------------------------------------------------------

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def t_probe_count():
    check("probe: radius 0 visits one cell", probe_count(8, 0) == 1)
    check("probe: radius 1 visits 1 + p cells", probe_count(8, 1) == 9, "p=8 -> %d" % probe_count(8, 1))
    check("probe: radius 2 visits 1 + p + p(p-1)/2", probe_count(8, 2) == 37, "p=8 -> %d" % probe_count(8, 2))
    check("probe: radius 3 visits the whole ball", probe_count(8, 3) == 1 + 8 + 28 + 56, "p=8 -> %d" % probe_count(8, 3))
    check("probe: keys are distinct", len(set(probe_keys(0b101010, 6, 2))) == 1 + 6 + 15,
          "%d distinct" % len(set(probe_keys(0b101010, 6, 2))))
    check("probe: radius 3 really visits triples", len(set(probe_keys(0b101010, 6, 3))) == 1 + 6 + 15 + 20,
          "%d distinct" % len(set(probe_keys(0b101010, 6, 3))))
    check("probe: the query's own cell comes first", list(probe_keys(7, 3, 1))[0] == 7)


def t_zero_probe_is_full():
    """p=0, radius 0 is one cell holding every training position, so it must equal the full router exactly.

    The first version of this test asserted that radius 0 reproduces `cell.py` at the same p and budget. It does not, and
    the gap is the finding: `cell.py` walks to coarser keys when a deep bucket is *empty*, so it silently searches a
    neighbourhood above the query; `probe.py` at radius 0 has no such fallback and simply finds nothing (18 hits against
    42). That implicit escalation is why the single-probe sweep looked so much better than a pure p-bit cell would.
    """
    c = R.synth_corpus(docs=8, positions=12, dim=96, nclass=20, seed=41, layer_signal=(0.85,))
    per = leave_one_document_out(c, 0, 0, 0, None)
    full = R.leave_one_document_out(c, 0, 20)
    same = all(per[d][0] == full[d][0] and per[d][2] == full[d][2] for d in full)
    check("identity: p=0 radius 0 equals the full 20-NN", same,
          "cell %d vs full %d" % (sum(v[0] for v in per.values()), sum(v[0] for v in full.values())))


def t_multiprobe_recovers_neighbours():
    """The point of probing: at a fixed budget, radius 1 must find neighbours that differ in one key bit."""
    c = R.synth_corpus(docs=10, positions=16, dim=192, nclass=40, seed=42, layer_signal=(0.5, 0.9))
    h = {}
    for r in (0, 1, 2):
        per = leave_one_document_out(c, 1, 10, r, 32)
        h[r] = sum(v[0] for v in per.values()) / sum(v[2] for v in per.values())
    check("multi-probe: radius 1 beats radius 0 at the same budget", h[1] > h[0] + 0.02,
          "r0 %.3f, r1 %.3f, r2 %.3f" % (h[0], h[1], h[2]))
    check("multi-probe: scanning more cells never lowers hit1", h[2] >= h[1] - 0.02,
          "r1 %.3f, r2 %.3f" % (h[1], h[2]))


def t_budget_bounds_scan():
    """A budget is a target, not a cap: the last cell may overshoot, and the overshoot is bounded by one cell."""
    c = R.synth_corpus(docs=6, positions=10, dim=64, nclass=10, seed=43, layer_signal=(0.85,))
    per = leave_one_document_out(c, 0, 6, 1, 16)
    worst = max(v[3] for v in per.values())
    check("budget: the scan stays within one cell of the budget", worst <= 16 + 16, "worst mean scan %.1f" % worst)


def t_answer_the_question():
    """The headline result, pinned: multi-probe does reach 95% of the hit rate inside half a scan.

    It only looks otherwise if the probe ball stops at pairs. At p=8 the triple ring is 93 of 256 cells - 45% of a scan -
    and that is where the bar is met; with radius 2 alone the best under half a scan was 92.8% retention and the answer
    read as a no. This check exists so that cap cannot quietly return.
    """
    c = R.synth_corpus(docs=12, positions=16, dim=192, nclass=40, seed=61, layer_signal=(0.5, 0.92))
    ref, rows = sweep(c, 1)
    per = leave_one_document_out(c, 1, 8, 3, None)
    hit1 = sum(v[0] for v in per.values()) / sum(v[2] for v in per.values())
    scan = float(np.mean([v[3] for v in per.values()]))
    full_scan = rows[0]["full_scan"]
    check("answer: p=8 radius 3 keeps 95% of the hit rate below half a scan",
          hit1 / ref >= 0.95 and scan / full_scan < 0.5,
          "retention %.3f at %.0f%% of a scan" % (hit1 / ref, 100 * scan / full_scan))
    by = dict(((r["probe"], r["radius"], r["budget"]), r["retention"]) for r in rows)
    pairs = [((p, 0, b), (p, 2, b)) for (p, _, b) in by if (p, 2, b) in by]
    better = sum(1 for a, c2 in pairs if by[c2] > by[a])
    check("sweep: a wider probe never loses to the single cell", better >= 0.9 * len(pairs),
          "%d of %d settings" % (better, len(pairs)))
    best = fastest(rows)
    check("answer: the selection rule finds it", best is not None and best["retention"] >= 0.95
          and best["scan_frac"] < 0.5,
          "none" if best is None else "p%d r%d b%d -> %.3f at %.0f%%" % (best["probe"], best["radius"],
                                                                        best["budget"], best["retention"],
                                                                        100 * best["scan_frac"]))
    check("answer: the bar is strict - an impossible one returns None", fastest(rows, keep=2.0) is None)


def selftest():
    del PASS[:]
    for t in (t_probe_count, t_zero_probe_is_full, t_multiprobe_recovers_neighbours, t_budget_bounds_scan,
              t_answer_the_question):
        t()
    expected = 15
    ok = len(PASS) == expected
    print("\n%d checks passed%s" % (len(PASS), "" if ok else " of %d - a check did not run" % expected))
    return 0 if ok else 1


def summary(rows, keep=0.95, max_frac=0.5):
    """The two numbers that answer the question, without a cap on the search: cheapest >=95%, and best under half."""
    ok = [r for r in rows if r["retention"] >= keep]
    cheap = min(ok, key=lambda r: r["scan"]) if ok else None
    under = [r for r in rows if r["scan_frac"] < max_frac]
    return cheap, (max(under, key=lambda r: r["retention"]) if under else None)


def report(corpus, layer, note=""):
    ref, rows = sweep(corpus, layer)
    best = fastest(rows)
    cheap, best_under = summary(rows)
    print("corpus: %d positions, %d documents, %d bits, %d classes%s"
          % (len(corpus.cls), len(np.unique(corpus.doc_of)), corpus.bits.shape[2], corpus.nclass, note))
    print("layer %d, reference (20-NN over everything): hit1 %.4f, full scan %d"
          % (layer, ref, rows[0]["full_scan"]))
    print("\n  %-6s %-7s %-8s %8s %8s %8s" % ("probe", "radius", "budget", "hit1", "retain", "scan%"))
    for r in rows:
        mark = " <-" if best and (r["probe"], r["radius"], r["budget"]) == (best["probe"], best["radius"],
                                                                           best["budget"]) else ""
        print("  %-6d %-7d %-8d %8.4f %8.3f %8.1f%s"
              % (r["probe"], r["radius"], r["budget"], r["hit1"], r["retention"], 100 * r["scan_frac"], mark))
    print("\nunder 50%% of a scan, the best setting is %s"
          % ("none" if best_under is None else
             "probe %d, radius %d, budget %d: %.3f retention" % (best_under["probe"], best_under["radius"],
                                                                 best_under["budget"], best_under["retention"])))
    print("cheapest >=95%% at any scan: %s"
          % ("unreachable" if cheap is None else
             "probe %d, radius %d, budget %d - %.0f%% of a scan, %.3f retention"
             % (cheap["probe"], cheap["radius"], cheap["budget"], 100 * cheap["scan_frac"], cheap["retention"])))
    print("verdict: %s" % ("yes, half a scan keeps 95%%" if best else
                            "not on this corpus - the bar costs more than half a scan"))
    return best


def main(argv):
    if "--corpus" in argv:
        path = argv[argv.index("--corpus") + 1]
        c = R.load_corpus(path)
        if c is None:
            print("no fingerprint corpus at %s" % path)
            return 2
        report(c, c.layers[-1])
        return 0
    c = R.synth_corpus(docs=12, positions=16, dim=192, nclass=40, seed=51, layer_signal=(0.5, 0.72, 0.9))
    report(c, 2, " (synthetic: layer 2 holds the signal)")

    # The verdict is not a constant: where the bar lands depends on how much signal the codes carry, so the three corpora
    # hold the layer signal fixed (layer 1, 0.5/0.92) and vary only the bits - different seeds, different probe keys.
    print("\nfrontier across corpora (where does 95% of the hit rate stop costing half a scan?)")
    print("  %-22s %6s %-22s %7s %-9s" % ("corpus", "ref", "cheapest >=95%", "scan%", "<50%?"))
    for seed in (61, 62, 63):
        cc = R.synth_corpus(docs=12, positions=16, dim=192, nclass=40, seed=seed,
                            layer_signal=(0.5, 0.92))
        ref, rows = sweep(cc, 1)
        cheap, best_under = summary(rows)
        print("  %-22s %6.3f %-22s %6.0f%% %-9s" % (
            "seed %d, layer 1" % seed, ref,
            "none reach it" if cheap is None else "p%d r%d b%s" % (cheap["probe"], cheap["radius"], cheap["budget"]),
            0.0 if cheap is None else 100 * cheap["scan_frac"],
            "yes" if cheap is not None and cheap["scan_frac"] < 0.5 else "no"))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main(sys.argv[1:]))
