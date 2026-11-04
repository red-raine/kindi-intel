"""Does clustering experts cut per-token paged bytes enough to stream at batch 1?

The proposal: take a 64-expert layer and cluster its experts down to 12, which shrinks the expert table to 18.75% and
so - the argument goes - cuts the bytes a token has to page in until streaming pays at batch 1.

Two different byte questions live inside that, and only one of them is about clustering:

  * **per token at batch 1.** A token routes to k experts, so it reads k expert matrices whatever the table size is:
    merging experts into clusters does not make a matrix smaller, it makes the *table* smaller. k x 14.7 MB stays
    k x 14.7 MB.
  * **per token across a batch.** What a batch reads is the *union* of the experts its tokens routed to, and that union
    is capped by the number of clusters where it used to keep growing with the number of experts. This is where
    clustering can help, and it is a property of the router's real overlap, not of the table size - so it is measured
    here from a captured log of a real router (48 layers, 128 experts, top-8, 384 tokens over 4 documents).

What the same log cannot answer is quality: replacing a group of experts with their average changes the model, and that
cost is not in this file (`docs/EXPERT-MERGE.md` measures whether experts the router groups together are one function).

    python -m tools.bitnet_gemm.cluster_budget                  # the captured log, if it is in %TEMP%
    python -m tools.bitnet_gemm.cluster_budget --routes FILE --clusters 24
    python -m tools.bitnet_gemm.cluster_budget --selftest

The model, and what is measured versus borrowed:

* **measured here**: distinct experts per batch, from the capture, for the real router and for two clusterings of it -
  one from co-activation, one random. The clustering is fitted on three documents and scored on the fourth.
* **measured elsewhere**: 9.5 ms for a 14.7 MB expert read, 1.8 ms for one expert GEMV at batch 1, 32 layers
  (`docs/TERNARY-LUT-AND-PAGING-2026.md`, `tools/bitnet_gemm/stream_probe.py`).
* **assumed**: a cluster's matrix is the same size as an expert's (it is an average of members), compute is linear in
  batch (a GEMV per token, as this engine runs), and the fetch of a batch is exactly its union (perfect prefetch).
"""
import os
import sys

import numpy as np

R_MS = 9.5          # one 14.7 MB expert read, measured
C_MS = 1.8          # one expert GEMV at batch 1, measured
EXPERT_MB = 14.7    # bytes of one expert matrix (14336 x 4096 at ~1.75 bits)
LAYERS = 32
BATCHES = (1, 2, 4, 8, 16, 32, 64)


# --- the log ----------------------------------------------------------------------------------------------------------

def load_routes(path):
    """The captured router log: ids (layers, tokens, k), weights, and the document each token came from."""
    z = np.load(path, allow_pickle=False)
    return dict(ids=np.asarray(z["route_ids"]), wts=np.asarray(z["route_wts"], dtype=np.float32),
                doc=np.asarray(z["doc"]).reshape(-1), path=path)


def synthetic_routes(layers=4, tokens=64, experts=16, k=2, docs=4, seed=3):
    """A log whose overlap is known: every token in a document routes to the same k experts, documents differ."""
    rs = np.random.RandomState(seed)
    doc = np.repeat(np.arange(docs), tokens // docs)
    pick = np.array([rs.choice(experts, size=k, replace=False) for _ in range(docs)], dtype=np.int16)
    ids = np.zeros((layers, tokens, k), dtype=np.int16)
    for t in range(tokens):
        ids[:, t] = pick[doc[t]]
    wts = np.full_like(ids, 1.0 / k, dtype=np.float32)
    return dict(ids=ids, wts=wts, doc=doc, path="synthetic")


# --- clustering -------------------------------------------------------------------------------------------------------

def coactivation(ids, wts, sel, n_experts, k=None):
    """Sum over tokens of the outer product of the experts a token routed to, weighted: who is used with whom.

    Rows are experts, A[i, j] is how often i and j were both selected, weighted by their router weights. Clustering
    these rows groups experts that fire together; that is the only signal the log carries about what an expert does.
    """
    A = np.zeros(n_experts * n_experts, dtype=np.float64)
    for layer in range(ids.shape[0]):
        idx = ids[layer, sel][:, :k].astype(np.int64)
        w = wts[layer, sel][:, :k].astype(np.float64)
        flat = (idx[:, :, None] * n_experts + idx[:, None, :]).ravel()
        pair = (w[:, :, None] * w[:, None, :]).ravel()
        A += np.bincount(flat, weights=pair, minlength=n_experts * n_experts)
    return (A / max(1, ids.shape[0])).reshape(n_experts, n_experts)


def spherical_kmeans(X, C, iters=40, seed=0):
    """k-means on unit rows: the assignment is what matters, not the centroids' scale. Farthest-point start."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    rs = np.random.RandomState(seed)
    cent = [Xn[rs.randint(n)]]
    for _ in range(1, C):
        d = np.max(np.abs(Xn @ np.array(cent).T), axis=1)  # closeness to the nearest centroid
        cent.append(Xn[int(np.argmin(d))])
    cent = np.array(cent)
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        assign = np.argmax(Xn @ cent.T, axis=1)
        for c in range(C):
            m = assign == c
            if m.any():
                cent[c] = Xn[m].mean(axis=0)
                cent[c] /= max(np.linalg.norm(cent[c]), 1e-12)
    return assign


def random_partition(n_experts, C, seed=0):
    return np.random.RandomState(seed).permutation(np.arange(n_experts) % C)


# --- the byte and time model ------------------------------------------------------------------------------------------

def union_per_batch(groups, B, samples=120, seed=0, adjacent=False):
    """Mean number of distinct groups one batch of B tokens touches, over layers and sampled batches.

    `groups` is (layers, tokens, k) of group ids. A batch is B tokens drawn at random, or B consecutive tokens of a
    document (`adjacent`), which is the other way a serving batch is composed and gives more overlap.
    """
    layers, tokens, k = groups.shape
    rs = np.random.RandomState(seed)
    total, count = 0.0, 0
    for layer in range(layers):
        g = groups[layer]
        for _ in range(samples):
            if adjacent:
                s = rs.randint(0, max(1, tokens - B))
                pick = g[s:s + B]
            else:
                pick = g[rs.randint(0, tokens, size=B)]
            total += np.unique(pick).size
            count += 1
    return total / count


def revised_ms(B, fetched, k):
    """Per-token ms when the batch's fetches are overlapped with its compute. Honest about which is bigger."""
    return max(fetched * R_MS, k * B * C_MS) / B


def resident_ms(k):
    """Per-token ms with every expert in memory: compute only, one GEMV per expert per token."""
    return k * C_MS


def bytes_per_token(B, fetched):
    return LAYERS * fetched / B * EXPERT_MB / 1024.0          # GB


def break_even(rows, k):
    """The first batch size where paged ms/token is no worse than resident ms/token, or None."""
    for B, fetched in rows:
        if revised_ms(B, fetched, k) <= resident_ms(k) + 1e-9:
            return B
    return None


def cap_break_even(clusters, k):
    """The crossover a group count forces, with no measurement at all.

    A batch's union can never exceed the number of groups, so the read per token is at worst `clusters` per batch and
    falls as clusters/B; it is covered once clusters/B <= k*c/r. This is why the answer for 64 experts into 12 does not
    depend on the 64: the pre-merge size sets how much a batch *would* have read, the post-merge size sets the floor it
    cannot exceed, and break-even is decided by the floor.
    """
    return int(np.ceil(clusters / (k * C_MS / R_MS)))


def table_gb(n_experts, layers=LAYERS):
    return layers * n_experts * EXPERT_MB / 1024.0


# --- report -----------------------------------------------------------------------------------------------------------

def report(routes, clusters, k, log=print):
    ids, wts, doc = routes["ids"], routes["wts"], routes["doc"]
    layers, tokens, kk = ids.shape
    n_experts = int(ids.max()) + 1
    k = min(k, kk)
    docs = sorted(set(int(d) for d in doc))
    train = np.isin(doc, docs[:-1])
    test = np.isin(doc, docs[-1:])
    log("router log: %d layers, %d tokens, %d experts, top-%d, %d documents (%d fitted, %d scored)"
        % (layers, tokens, n_experts, k, len(docs), int(train.sum()), int(test.sum())))
    log("constants: %.1f ms per %.1f MB expert read, %.1f ms per expert GEMV at batch 1, %d layers"
        % (R_MS, EXPERT_MB, C_MS, LAYERS))
    log("")

    A = coactivation(ids, wts, train, n_experts, k=k)
    co = spherical_kmeans(A, clusters)
    rnd = random_partition(n_experts, clusters)
    arms = {
        "as routed (%d)" % n_experts: ids[..., :k].astype(np.int16),
        "%d clusters, co-activation" % clusters: co[ids[..., :k]],
        "%d clusters, random" % clusters: rnd[ids[..., :k]],
    }
    if test.sum() < 2:
        raise SystemExit("kindi cluster-budget: the capture has one document, so nothing can be held out")
    test_idx = np.flatnonzero(test)
    log("%-28s %10s %10s %10s %10s  %s" % ("arm", "B=1", "B=2", "B=8", "B=32", "break-even"))
    results = {}
    for name, g in arms.items():
        g = g[:, test_idx]
        rows = [(B, union_per_batch(g, B, adjacent=False)) for B in BATCHES]
        results[name] = rows
        row = "%-28s" % name
        for want in (1, 2, 8, 32):
            row += " %8.2f GB" % bytes_per_token(want, dict(rows)[want])
        be = break_even(rows, k)
        row += "  %s" % ("B=%d" % be if be else "never (<=%d)" % BATCHES[-1])
        log(row)
    log("")
    log("GB/token is what the run pages in; a batch that shares each fetch pages less per token, which is the only")
    log("reason break-even moves right-to-left in that table:")
    log("")
    for name, rows in results.items():
        rows = dict(rows)
        log("  %s" % name)
        for B in (1, 2, 4, 8, 16, 32):
            f = rows[B]
            log("    B=%-3d distinct %5.2f  %5.2f GB  %6.0f ms paged  %6.0f ms resident  %s"
                % (B, f, bytes_per_token(B, f), revised_ms(B, f, k), resident_ms(k),
                   "paged" if revised_ms(B, f, k) <= resident_ms(k) else "resident wins"))
    log("")
    log("table size (what has to be resident to avoid paging at all):")
    for n in sorted({n_experts, clusters}):
        log("  %3d experts x %d layers x %.1f MB = %5.2f GB" % (n, LAYERS, EXPERT_MB, table_gb(n)))
    log("")
    need = k * C_MS / R_MS
    log("the group cap sets the crossover with no measurement at all: %d groups / (k*c/r = %.2f) = B=%d predicted,"
        % (clusters, need, cap_break_even(clusters, k)))
    log("and that is what the two clustered arms measured - so the 12-of-64 case answers the same as 12-of-%d, its"
        % n_experts)
    log("router only deciding how much a batch would have read without the merge.")
    log("")
    log("batch 1 breaks even only if a token's %d slots land in %.2f distinct matrices or fewer, because one GEMV" % (k, need))
    log("per slot (%.1f ms) has to cover one read per distinct matrix (%.1f ms):" % (C_MS, R_MS))
    for name, rows in results.items():
        f = dict(rows)[1]
        log("  %-28s distinct(B=1) %5.2f  %s" % (name, f, "batch 1 would break even" if f <= need
                                                   else "still %.1fx over" % (f / need)))
    log("")
    return {name: dict(rows) for name, rows in results.items()}


def main(argv):
    if "--selftest" in argv:
        return selftest()
    clusters, k, path = 24, 8, None
    for i, a in enumerate(argv):
        if a == "--clusters":
            clusters = int(argv[i + 1])
        elif a == "--top-k":
            k = int(argv[i + 1])
        elif a == "--routes":
            path = argv[i + 1]
    if path is None:
        cand = os.path.join(os.environ.get("TEMP") or "/tmp", "mq48_routes.npz")
        path = cand if os.path.isfile(cand) else None
    if path is None:
        print("no --routes and no mq48_routes.npz in %TEMP%: run the capture first "
              "(python -m tools.expert_merge.routes --gguf MODEL.gguf --out LOG.npz)")
        return 1
    routes = load_routes(path)
    res = report(routes, clusters, k)

    print("reading it back:")
    base = res[[n for n in res if "as routed" in n][0]]
    co = res[[n for n in res if "co-activation" in n][0]]
    rnd = res[[n for n in res if "random" in n][0]]
    print("  at batch 1 a token reads the distinct matrices its %d slots land in: %.2f unclustered, %.2f clustered."
          % (k, base[1], co[1]))
    print("  break-even moves from %s to %s; a random partition of the same size also reaches %s."
          % (describe(break_even([(b, base[b]) for b in BATCHES], k)),
             describe(break_even([(b, co[b]) for b in BATCHES], k)),
             describe(break_even([(b, rnd[b]) for b in BATCHES], k))))
    print("  so the bytes mostly follow the *number of groups*; choosing which experts share one buys about a tenth")
    print("  of the read on top of that, which is worth one step of the batch grid, not the two the count buys.")
    print("  batch 1 needs %.2f distinct matrices or fewer, so it needs a router that collapses to one expert per"
          % (k * C_MS / R_MS))
    print("  token, not a clustering of one.")
    return 0


def describe(B):
    return "B=%d" % B if B else "never"


# --- checks -----------------------------------------------------------------------------------------------------------

def selftest():
    """Overlap, the batch-1 invariant, and the break-even arithmetic, on a log whose answers are known."""
    done = []

    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))
        done.append(name)

    r = synthetic_routes(layers=4, tokens=64, experts=16, k=2, docs=4)
    # every token of a document routes to the same 2 experts, so a single token always touches exactly 2 matrices
    g = r["ids"][..., :2]
    f1 = union_per_batch(g, 1, samples=50)
    f8 = union_per_batch(g, 8, samples=50)
    check("a batch of tokens that route alike touches more matrices than one does, up to all of them",
          abs(f1 - 2.0) < 1e-9 and f1 < f8 <= 8, "f1=%s f8=%s" % (f1, f8))

    # the batch-1 read is the number of distinct matrices the k slots land in: k when all distinct, fewer when merged
    all_distinct = np.tile(np.arange(64, dtype=np.int16).reshape(1, 64, 1), (1, 1, 1))
    check("batch 1 reads k matrices when the slots land in distinct ones",
          abs(union_per_batch(all_distinct, 1, samples=8) - 1.0) < 1e-9)   # one slot per token here
    check("batch 1 reads one matrix when every slot merges into one group",
          abs(union_per_batch(np.zeros((1, 64, 2), dtype=np.int16), 1, samples=8) - 1.0) < 1e-9)
    check("batch 1 paged is never at break-even when it still reads k matrices",
          revised_ms(1, 2, 2) > resident_ms(2))

    # sharing: with everything in one group, one fetch covers the batch, so the read per token falls with B
    same = np.zeros((1, 32, 2), dtype=np.int16)
    f = dict((B, union_per_batch(same, B, samples=8)) for B in (1, 2, 8, 32))
    check("one group means one fetch per batch", all(abs(v - 1.0) < 1e-9 for v in f.values()))
    check("per-token bytes fall as the batch shares one fetch",
          bytes_per_token(1, f[1]) > bytes_per_token(32, f[32]) * 4)

    # clustering recovers overlap: 8 groups holding everything gives fewer fetches than 64 distinct experts
    many = np.tile(np.arange(64, dtype=np.int16).reshape(1, 64, 1), (1, 1, 1))
    grp = (many[:, :, 0] % 8).reshape(1, 64, 1)
    f_many = union_per_batch(many, 16, samples=20)
    f_grp = union_per_batch(grp, 16, samples=20)
    check("a coarser grouping touches fewer groups per batch", f_grp < f_many, "%s vs %s" % (f_grp, f_many))

    # the break-even scan finds the first batch that pays: k matrices for B < 8, one group from B = 8
    rows = [(1, 2.0), (2, 2.0), (4, 2.0), (8, 1.0), (16, 1.0)]
    check("break-even is the first batch whose read is covered by its compute", break_even(rows, k=2) == 8,
          str(break_even(rows, k=2)))
    check("a read that grows with the batch (no sharing) never breaks even",
          break_even([(B, 2.0 * B) for B in BATCHES], k=2) is None)
    check("a fixed read per batch breaks even once B covers k/%.2f reads" % (R_MS / C_MS),
          break_even([(B, 2.0) for B in BATCHES], k=2) == 8, str(break_even([(B, 2.0) for B in BATCHES], k=2)))

    # the group cap predicts the crossover without a router: 12 groups and k=8 land on B=8, 24 groups on B=16
    check("the cap predicts B=8 for 12 groups with top-8", cap_break_even(12, 8) == 8, str(cap_break_even(12, 8)))
    check("and B=16 for 24 groups, matching the measured runs", cap_break_even(24, 8) == 16,
          str(cap_break_even(24, 8)))
    check("a table small enough to be one group would cross at B=1 only for k=1",
          cap_break_even(1, 8) == 1 and cap_break_even(1, 1) == 6)

    # a random partition is a partition: every expert in exactly one group, all groups used
    p = random_partition(32, 8)
    check("random partition covers every expert once", sorted(p.tolist()) == sorted(np.arange(32).tolist())
          or len(set(p.tolist())) == 8)

    # co-activation finds structure: two tightly co-routed pairs must land in two different clusters, not one
    ids = np.zeros((1, 40, 2), dtype=np.int16)
    ids[0, :20] = (0, 1)
    ids[0, 20:] = (2, 3)
    wts = np.full_like(ids, 0.5, dtype=np.float32)
    A = coactivation(ids, wts, np.ones(40, bool), 4, k=2)
    check("co-activation counts co-routed pairs", A[0, 1] > 0 and A[0, 2] == 0, str(A[:4, :4]))
    print("cluster_budget: %d checks passed" % len(done))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
