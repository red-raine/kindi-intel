"""The 20-NN fingerprint router, as a component: index, leave-one-document-out scoring, and tuning.

`docs/ROUTING-FINGERPRINT.md` reports that the sign code of the folded residual stream predicts the model's own next-token
class well enough to be worth routing on: 20-NN over all 2,048 bits, 0.5361 hit1 at layer 23 against 0.1772 for a global
majority table. That result came from `~/kt_route`, which is not in this tree and whose corpus (`fp_bits_centred.npy` and
friends) no longer exists - only the scripts survived, in a temp directory. So the numbers cannot be re-derived here.

What can be built here is the thing that produced them: an index over the whole code, folds that never let a document
into its own neighbour set, the layer and neighbour count chosen on documents other than the one being scored, and
document-level intervals. This module is that, with the protocol taken from `route_router.py` / `route_score.py`:

  * the last position of each document is dropped (its class has no observable future)
  * the class is the model's argmax token, gathered into contiguous indices; `top5` is the model's own five best tokens
  * `hit1` = the argmax class is among the router's predicted top-5 classes
  * `hit5` = one of the model's top-5 tokens is among the router's predicted top-5 classes
  * neighbours are voted into a class histogram, and the prediction is the five largest bins
  * intervals bootstrap over **documents**, because positions inside one document are not independent

Tuning is nested on purpose. Picking `(layer, k)` by scoring every document and then reporting the winner's score is how
a router reports a number nobody can reproduce; here each held-out document's `(layer, k)` is chosen by running
leave-one-document-out **on the other documents only**, so the score reported for that document was never used to choose
its settings.

    python -m tools.fingerprint.router                 # synthetic corpus, known answers, tuning and baselines
    python -m tools.fingerprint.router --corpus DIR    # a real `~/kt_route/corpus`-style directory
"""
import json
import os
import sys

import numpy as np

LAYERS = (0, 3, 9, 15, 18, 21, 23)
KS = (1, 5, 10, 20, 40, 80)
TOPK = 5


# --- corpus -----------------------------------------------------------------------------------------------------------

class Corpus(object):
    """Packed sign bits per layer, plus the labels needed to score a prediction.

    bits: (layers, positions, dim) in {0, 1}; prev: previous token id per position, -1 at a document's first position;
    cls: contiguous class index of the model's argmax; top5: the model's own five best token classes.
    """

    def __init__(self, bits, cls, top5, doc_of, prev, layers=LAYERS):
        self.bits = bits
        self.cls = cls
        self.top5 = top5
        self.doc_of = doc_of
        self.prev = prev
        self.layers = tuple(layers)
        self.nclass = int(cls.max()) + 1 if len(cls) else 0

    def layer(self, name):
        return self.bits[self.layers.index(name)]


def load_corpus(path, keep_last=False):
    """Read a `~/kt_route/corpus` directory: fp_bits_centred.npy, argmax.npy, top5.npy, ids.npy, meta.json.

    Returns None when the directory does not hold that layout, so a caller can fall back to synthesis rather than fail.
    """
    need = ("fp_bits_centred.npy", "argmax.npy", "top5.npy", "meta.json")
    if not all(os.path.exists(os.path.join(path, f)) for f in need):
        return None
    meta = json.load(open(os.path.join(path, "meta.json")))
    ndoc, tlen = meta["NDOC"], meta["TLEN"]
    packed = np.load(os.path.join(path, "fp_bits_centred.npy"))
    nl = packed.shape[0]
    keep = np.ones((ndoc, tlen), bool)
    if not keep_last:
        keep[:, -1] = False
    sel = np.nonzero(keep.reshape(-1))[0]
    bits = np.stack([np.unpackbits(packed[i], axis=1)[sel] for i in range(nl)])
    arg = np.load(os.path.join(path, "argmax.npy")).reshape(-1)[sel]
    top5_raw = np.load(os.path.join(path, "top5.npy")).reshape(-1, TOPK)[sel]
    ids = np.load(os.path.join(path, "ids.npy")).reshape(-1)[sel] if os.path.exists(
        os.path.join(path, "ids.npy")) else np.zeros(len(sel), np.int64)

    # one shared vocabulary for argmax and top-5, so a token id means the same class in both arrays
    uniq, inv = np.unique(np.concatenate([arg, top5_raw.reshape(-1)]), return_inverse=True)
    cls = inv[:len(arg)].astype(np.int32)
    top5 = inv[len(arg):].reshape(-1, TOPK).astype(np.int32)
    pos_of = np.tile(np.arange(tlen), ndoc)[sel]
    doc_of = np.repeat(np.arange(ndoc), tlen)[sel]
    prev = np.where(pos_of > 0, ids[np.maximum(np.arange(len(sel)) - 1, 0)], -1)
    layers = tuple(meta.get("LAYERS", LAYERS))[:nl]
    return Corpus(bits, cls, top5, doc_of, prev, layers=layers)


def synth_corpus(docs=12, positions=20, dim=256, nclass=8, seed=0, layer_signal=(0.5, 0.75, 0.95)):
    """A corpus whose answer is known: class determines the bits, with document-level drift and per-position noise.

    `layer_signal[l]` is the probability that a bit agrees with the class prototype in layer `l`. **0.5 is the noise
    point**, not 0.0: a coordinate that always disagrees carries the class as well as one that always agrees, so 0.0 is a
    perfectly informative (inverted) layer. Tuning should find the highest value, whatever the order of the layers.
    Documents draw classes from their own topic distribution and share a document-level bit mask, which is what makes
    leave-one-document-out the right fold.
    """
    rs = np.random.RandomState(seed)
    nl = len(layer_signal)
    proto = rs.rand(nclass, dim) < 0.5
    bits = np.zeros((nl, docs * positions, dim), np.uint8)
    cls = np.zeros(docs * positions, np.int32)
    doc_of = np.repeat(np.arange(docs), positions)
    top5 = np.zeros((docs * positions, TOPK), np.int32)
    topics = [rs.choice(nclass, size=max(2, nclass // 2), replace=False) for _ in range(docs)]
    for d in range(docs):
        drift = rs.rand(dim) < 0.1                       # bits this document flips everywhere
        for p in range(positions):
            i = d * positions + p
            c = int(rs.choice(topics[d]))
            cls[i] = c
            others = [x for x in rs.permutation(nclass) if x != c][:TOPK - 1]
            row = [c] + [int(x) for x in others]
            top5[i] = (row + [c] * TOPK)[:TOPK]          # pad when there are fewer classes than TOPK
            for l, agree in enumerate(layer_signal):
                bit = proto[c].astype(np.uint8).copy()
                flip = rs.rand(dim) > agree
                bit[flip] ^= 1
                bit ^= drift
                bits[l, i] = bit
    prev = np.where(np.tile(np.arange(positions), docs) > 0, cls[np.maximum(np.arange(len(cls)) - 1, 0)], -1)
    return Corpus(bits, cls, top5, doc_of, prev, layers=tuple(range(nl)))


# --- the index --------------------------------------------------------------------------------------------------------

def similarity(a, b):
    """Agreement score over the whole code, one matmul: D - 2 * Hamming, so higher is closer.

    The codes are cast to {\u22121, +1} first, deliberately. On {0, 1} codes the plain dot product is *not* the agreement
    count - it is inflated by however many 1s a row carries, so a dense row would look closer to everything than a sparse
    one, and the ranking would follow density rather than similarity. On {\u22121, +1} codes dot = D - 2 * Hamming, which is
    monotone in the distance the original harness measured.
    """
    return (a.astype(np.float32) * 2 - 1) @ (b.astype(np.float32) * 2 - 1).T


_sim = similarity


def votes_for(sim, train_cls, k, nclass):
    """Ballot counts over the k nearest training positions, for every test row."""
    k = min(k, sim.shape[1])
    nb = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    votes = np.zeros((sim.shape[0], nclass), np.int32)
    labels = train_cls[nb]
    for row in range(votes.shape[0]):
        np.add.at(votes[row], labels[row], 1)
    return votes


def predict(votes, top=TOPK):
    """Top classes per row, ties broken by class index so the output is deterministic."""
    return np.argsort(-votes, axis=1, kind="stable")[:, :top]


def score_fold(bits, cls, top5, train, test, k):
    """hit1 and hit5 over the test positions of one fold, with the index built from `train` only."""
    sim = _sim(bits[test], bits[train])
    pred = predict(votes_for(sim, cls[train], k, int(cls.max()) + 1))
    hit1 = np.array([int(cls[test[i]] in pred[i]) for i in range(len(test))])
    hit5 = np.array([int(len(set(top5[test[i]].tolist()) & set(pred[i].tolist())) > 0) for i in range(len(test))])
    return hit1, hit5, pred


def leave_one_document_out(corpus, layer, k):
    """Per-document hit1/hit5. The held-out document contributes no prototypes: that is the whole point of the fold."""
    bits = corpus.layer(layer)
    out = {}
    for d in np.unique(corpus.doc_of):
        test = np.nonzero(corpus.doc_of == d)[0]
        train = np.nonzero(corpus.doc_of != d)[0]
        h1, h5, _ = score_fold(bits, corpus.cls, corpus.top5, train, test, k)
        out[int(d)] = (int(h1.sum()), int(h5.sum()), len(test))
    return out


def tune(corpus, layers=None, ks=None, score="hit1"):
    """Nested selection: for each held-out document, choose (layer, k) on the *other* documents, then score it.

    Returns per-document records: the chosen layer and k, that document's hits under them, and the settings' inner score.
    """
    layers = tuple(layers or corpus.layers)
    ks = tuple(ks or KS)
    docs = [int(d) for d in np.unique(corpus.doc_of)]
    sims = dict((l, _sim(corpus.layer(l), corpus.layer(l))) for l in layers)
    picks = {}

    def inner_score(l, k, allowed_docs, hold):
        """Leave-one-document-out over `allowed_docs`, scored on their positions only, excluding `hold` entirely."""
        pos = np.nonzero(np.isin(corpus.doc_of, allowed_docs))[0]
        total = [0, 0, 0]
        for d in allowed_docs:
            if d == hold:
                continue
            test = np.nonzero(corpus.doc_of == d)[0]
            train = pos[corpus.doc_of[pos] != d]
            if not len(train) or not len(test):
                continue
            sub = sims[l][np.ix_(test, train)]
            pred = predict(votes_for(sub, corpus.cls[train], k, corpus.nclass))
            h1 = sum(int(corpus.cls[t] in pred[i]) for i, t in enumerate(test))
            h5 = sum(int(len(set(corpus.top5[t].tolist()) & set(pred[i].tolist())) > 0) for i, t in enumerate(test))
            total[0] += h1
            total[1] += h5
            total[2] += len(test)
        num = total[0] if score == "hit1" else total[1]
        return num / max(total[2], 1)

    for d in docs:
        others = [x for x in docs if x != d]
        best, best_score = None, -1.0
        for l in layers:
            for k in ks:
                s = inner_score(l, k, others, d)
                if s > best_score + 1e-12:
                    best, best_score = (l, k), s
        test = np.nonzero(corpus.doc_of == d)[0]
        train = np.nonzero(corpus.doc_of != d)[0]
        h1, h5, _ = score_fold(corpus.layer(best[0]), corpus.cls, corpus.top5, train, test, best[1])
        picks[d] = dict(layer=best[0], k=best[1], hit1=int(h1.sum()), hit5=int(h5.sum()), n=len(test),
                        inner=round(best_score, 4))
    return picks


def bootstrap(picks, field="hit5", draws=2000, seed=0):
    """Document-level bootstrap: resample documents, not positions, because positions within one are correlated."""
    docs = sorted(picks)
    num = np.array([picks[d][field] for d in docs], float)
    den = np.array([picks[d]["n"] for d in docs], float)
    point = float(num.sum() / den.sum())
    rs = np.random.RandomState(seed)
    stats = np.empty(draws)
    for i in range(draws):
        idx = rs.randint(0, len(docs), len(docs))
        stats[i] = num[idx].sum() / den[idx].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


# --- baselines --------------------------------------------------------------------------------------------------------

def majority_baseline(corpus):
    out = {}
    for d in np.unique(corpus.doc_of):
        train = np.nonzero(corpus.doc_of != d)[0]
        test = np.nonzero(corpus.doc_of == d)[0]
        top = set(np.bincount(corpus.cls[train], minlength=corpus.nclass).argsort()[-TOPK:].tolist())
        out[int(d)] = (sum(int(corpus.cls[t] in top) for t in test),
                       sum(int(len(set(corpus.top5[t].tolist()) & top) > 0) for t in test), len(test))
    return out


def bigram_baseline(corpus):
    """Predict from the previous token alone, falling back to the fold's majority where the context is unseen."""
    out = {}
    for d in np.unique(corpus.doc_of):
        train = np.nonzero(corpus.doc_of != d)[0]
        test = np.nonzero(corpus.doc_of == d)[0]
        table = {}
        for i in train:
            table.setdefault(int(corpus.prev[i]), np.zeros(corpus.nclass, np.int32))[corpus.cls[i]] += 1
        glob = np.bincount(corpus.cls[train], minlength=corpus.nclass)
        h1 = h5 = 0
        for t in test:
            counts = table.get(int(corpus.prev[t]), glob)
            pred = set(np.argsort(-counts, kind="stable")[:TOPK].tolist())
            h1 += int(corpus.cls[t] in pred)
            h5 += int(len(set(corpus.top5[t].tolist()) & pred) > 0)
        out[int(d)] = (h1, h5, len(test))
    return out


def summarise(per_doc, name, field="hit5"):
    point, lo, hi = bootstrap(dict((d, dict(hit5=v[1], n=v[2], hit1=v[0])) for d, v in per_doc.items()), field)
    num = sum(v[0] if field == "hit1" else v[1] for v in per_doc.values())
    den = sum(v[2] for v in per_doc.values())
    return "%d/%d = %.4f [%.4f, %.4f]" % (num, den, point, lo, hi)


# --- tests ------------------------------------------------------------------------------------------------------------

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def t_distance():
    rs = np.random.RandomState(0)
    a = (rs.rand(9, 64) < 0.5).astype(np.uint8)
    got = _sim(a, a)
    brute = np.array([[int((a[i] == a[j]).sum()) for j in range(9)] for i in range(9)])
    recovered = (got + 64) / 2                                   # dot on ±1 codes = D - 2*Hamming
    check("distance: the score recovers brute-force agreement", np.array_equal(recovered, brute),
          "max |diff| %.1f" % float(np.abs(recovered - brute).max()))
    b = 1 - a[0]
    check("distance: a complement sits at the far end", _sim(a[:1], b[None])[0, 0] == -64,
          "score %d, so agreement %d of 64" % (int(_sim(a[:1], b[None])[0, 0]), int((_sim(a[:1], b[None])[0, 0] + 64) / 2)))


def t_no_leak():
    c = synth_corpus(docs=6, positions=8, dim=64, nclass=6, seed=1, layer_signal=(0.9,))
    overlaps, covered = [], 0
    for d in np.unique(c.doc_of):
        train = np.nonzero(c.doc_of != d)[0]
        test = np.nonzero(c.doc_of == d)[0]
        overlaps.append(len(set(train.tolist()) & set(test.tolist())))
        covered += len(test)
    check("folds: a held-out document contributes no neighbours", max(overlaps) == 0, "max overlap %d" % max(overlaps))
    check("folds: every position is in exactly one test fold", covered == len(c.cls),
          "%d positions, %d covered" % (len(c.cls), covered))


def t_recovers_planted_layer():
    """The layer choice must find the layer the signal was planted in, without being told which it is."""
    c = synth_corpus(docs=12, positions=16, dim=192, nclass=40, seed=2, layer_signal=(0.5, 0.65, 0.92))
    picks = tune(c, layers=(0, 1, 2), ks=(5, 20, 40))
    chosen = [p["layer"] for p in picks.values()]
    check("tuning: picks the layer the signal is in", chosen.count(2) >= len(chosen) - 1,
          "chosen layers %s" % sorted(set(chosen)))
    tuned = sum(p["hit1"] for p in picks.values()) / sum(p["n"] for p in picks.values())
    noise_per = leave_one_document_out(c, 0, 20)
    noise = sum(v[0] for v in noise_per.values()) / sum(v[2] for v in noise_per.values())
    check("tuning: tuned hit1 beats the noise layer by a wide margin", tuned > noise + 0.25,
          "tuned %.3f against layer 0 %.3f (chance ~%.3f)" % (tuned, noise, TOPK / c.nclass))


def t_neighbour_count():
    c = synth_corpus(docs=12, positions=16, dim=192, nclass=40, seed=3, layer_signal=(0.7,))
    rates = {}
    for k in (1, 5, 20, 80):
        per = leave_one_document_out(c, 0, k)
        rates[k] = sum(v[1] for v in per.values()) / sum(v[2] for v in per.values())
    check("k: one neighbour is worse than a vote", rates[20] > rates[1],
          "k=1 %.3f, k=20 %.3f" % (rates[1], rates[20]))
    check("k: a vote over almost everything is worse than 20", rates[20] > rates[80],
          "k=20 %.3f, k=80 %.3f" % (rates[20], rates[80]))


def t_prediction_shape():
    c = synth_corpus(docs=4, positions=6, dim=32, nclass=6, seed=4, layer_signal=(0.9,))
    train = np.nonzero(c.doc_of != 0)[0]
    test = np.nonzero(c.doc_of == 0)[0]
    h1, h5, pred = score_fold(c.layer(0), c.cls, c.top5, train, test, 5)
    check("predict: five distinct classes per row", pred.shape[1] == TOPK and all(len(set(r)) == TOPK for r in pred),
          "shape %s" % (pred.shape,))
    check("predict: hit5 is never below hit1", h5.sum() >= h1.sum(), "hit1 %d, hit5 %d" % (h1.sum(), h5.sum()))


def t_bootstrap():
    picks = dict((d, dict(hit1=5, hit5=5, n=10)) for d in range(8))
    point, lo, hi = bootstrap(picks)
    check("bootstrap: identical documents give a zero-width interval", abs(hi - lo) < 1e-12 and abs(point - 0.5) < 1e-12,
          "%.4f [%.4f, %.4f]" % (point, lo, hi))
    picks2 = dict((d, dict(hit1=d % 2, hit5=(d % 3) * 4, n=10)) for d in range(20))
    p2, l2, h2 = bootstrap(picks2)
    check("bootstrap: the interval brackets the point estimate", l2 <= p2 <= h2, "%.4f [%.4f, %.4f]" % (p2, l2, h2))


def t_baselines_lose():
    """On a corpus whose class is a function of the bits, a router that ignores the bits must lose."""
    c = synth_corpus(docs=12, positions=16, dim=192, nclass=40, seed=5, layer_signal=(0.5, 0.92))
    nn = leave_one_document_out(c, 1, 20)
    maj = majority_baseline(c)
    big = bigram_baseline(c)
    nn1 = sum(v[0] for v in nn.values()) / sum(v[2] for v in nn.values())
    maj1 = sum(v[0] for v in maj.values()) / sum(v[2] for v in maj.values())
    big1 = sum(v[0] for v in big.values()) / sum(v[2] for v in big.values())
    check("baselines: 20-NN beats the global majority table", nn1 > maj1 + 0.1,
          "20-NN %.3f, majority %.3f" % (nn1, maj1))
    check("baselines: 20-NN beats the previous-token table", nn1 > big1 + 0.1,
          "20-NN %.3f, bigram %.3f" % (nn1, big1))


def t_real_layout():
    """The loader must read a `~/kt_route/corpus`-style directory, since that is what reproduces a published number."""
    import tempfile, shutil
    d = tempfile.mkdtemp(prefix="fp-")
    try:
        ndoc, tlen, dim, nl = 3, 4, 8, 2
        rs = np.random.RandomState(0)
        bits = (rs.rand(nl, ndoc * tlen, dim) < 0.5).astype(np.uint8)
        np.save(os.path.join(d, "fp_bits_centred.npy"), np.packbits(bits, axis=2))
        np.save(os.path.join(d, "argmax.npy"), rs.randint(0, 5, ndoc * tlen).reshape(ndoc, tlen))
        np.save(os.path.join(d, "top5.npy"), rs.randint(0, 5, (ndoc * tlen, TOPK)))
        np.save(os.path.join(d, "ids.npy"), rs.randint(0, 5, ndoc * tlen))
        json.dump(dict(NDOC=ndoc, TLEN=tlen, D=dim, NL=nl, LAYERS=list(range(nl))),
                  open(os.path.join(d, "meta.json"), "w"))
        c = load_corpus(d)
        check("loader: reads the original layout", c is not None and c.bits.shape == (nl, ndoc * (tlen - 1), dim),
              "shape %s" % (None if c is None else (c.bits.shape,)))
        check("loader: drops the last position of each document", c is not None and len(c.cls) == ndoc * (tlen - 1),
              "positions %d" % (0 if c is None else len(c.cls)))
        check("loader: returns None for a directory without the layout", load_corpus(os.path.join(d, "nope")) is None)
    finally:
        shutil.rmtree(d)


def t_missing_corpus_is_an_error():
    """`--corpus` with a path that does not hold one must exit 2, not quietly score the synthetic corpus instead.

    The fallback used to be silent, so a typo printed planted-signal numbers (0.9875 hit1) under whatever directory the
    user named. The other three fingerprint modules already refused; this pins the fourth.
    """
    rc = main(["--corpus", os.path.join(os.path.dirname(os.path.abspath(__file__)), "no-such-corpus")])
    check("cli: a `--corpus` path with no corpus exits 2", rc == 2, "exit %r" % (rc,))


def selftest():
    del PASS[:]
    for t in (t_distance, t_no_leak, t_recovers_planted_layer, t_neighbour_count, t_prediction_shape, t_bootstrap,
              t_baselines_lose, t_real_layout, t_missing_corpus_is_an_error):
        t()
    expected = 18
    ok = len(PASS) == expected
    print("\n%d checks passed%s" % (len(PASS), "" if ok else " of %d - a check did not run" % expected))
    return 0 if ok else 1


def report(corpus, note=""):
    print("corpus: %d positions, %d documents, %d bits, %d classes%s"
          % (len(corpus.cls), len(np.unique(corpus.doc_of)), corpus.bits.shape[2], corpus.nclass, note))
    picks = tune(corpus)
    layers = [p["layer"] for p in picks.values()]
    ks = [p["k"] for p in picks.values()]
    print("tuned per document: layer %s, k %s" % (sorted(set(layers)), sorted(set(ks))))
    print("  %-26s %s" % ("tuned (layer, k on held-out docs)", summarise(dict((d, (p["hit1"], p["hit5"], p["n"]))
                                                                           for d, p in picks.items()), "tuned")))
    for l in corpus.layers:
        for k in (1, 20):
            per = leave_one_document_out(corpus, l, k)
            print("  %-26s %s" % ("layer %d, %d-NN" % (l, k), summarise(per, "x")))
    print("  %-26s %s" % ("global majority", summarise(majority_baseline(corpus), "x")))
    print("  %-26s %s" % ("previous-token table", summarise(bigram_baseline(corpus), "x")))
    return 0


def main(argv):
    path = None
    if "--corpus" in argv:
        path = argv[argv.index("--corpus") + 1]
    c = load_corpus(path) if path else None
    if path and c is None:
        # Not a fallback: a mistyped path would otherwise print planted-signal numbers under a real corpus's name, which is
        # exactly the kind of number nobody can reproduce. The other three fingerprint modules already exit 2 here.
        print("no fingerprint corpus at %s" % path)
        return 2
    if c is None:
        c = synth_corpus(docs=10, positions=16, dim=128, nclass=40, seed=7, layer_signal=(0.5, 0.65, 0.92))
        return report(c, " (synthetic: layer 2 holds the signal, layer 0 is at the 0.5 noise point)")
    return report(c)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main(sys.argv[1:]))
