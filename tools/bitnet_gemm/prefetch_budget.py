"""Does speculative routing help, or just move bytes? A per-token budget for paged experts.

The proposal: distil an early head that predicts layer L+1's experts from layer L's hidden state, and over-fetch K+1
candidates so recall rises from ~82% to >96%, "completely hiding cold NVMe load latency".

At batch 1 the premise is already known to fail (docs/TERNARY-LUT-AND-PAGING-2026.md): one 14.7 MB expert read costs
~9.5 ms against 1.8 ms of GEMV, so the link is saturated and prefetch has no idle time to fill. This script puts a
number on the proposal anyway, because whether it helps is a question about *bytes* and *batch*, not about recall, and
the crossover is worth knowing.

    python -m tools.bitnet_gemm.prefetch_budget                    # the 9.5 ms the withdrawal of 2026-11-04 retired
    python -m tools.bitnet_gemm.prefetch_budget --read 3.7         # what the same read costs with nothing committing
    python -m tools.bitnet_gemm.prefetch_budget --recall 0.90 --overfetch 4

Model, with the assumptions stated so they can be argued with:

* `T_layer = max(f * r, k * B * c) + k * (1 - p) * r`
  - the f prefetched experts stream at read time `r` each, overlapped with `k * B * c` of compute per layer;
  - an expert that was *not* prefetched is discovered at use time, and nothing can overlap it, so it adds its full `r`;
  - `p` is the fraction of the true top-k that the candidate set already contained.
* `r` and `c` are the measured constants: 9.5 ms per 14.7 MB expert read, 1.8 ms per expert GEMV at batch 1.
* Compute is linear in batch: this engine runs a GEMV per token, so batching does not reduce arithmetic, it only shares
  the read. A batched GEMM kernel would make compute sublinear too, and then every policy improves together.
* The recall figures (0.82, 0.96) are the proposal's own training-loop numbers, measured on the batches it trained on -
  no held-out split - so treat them as upper bounds on production recall.
"""
import sys

R_MS = 9.5          # one 14.7 MB expert read while the drive is committing a write (copytest.py --flush); 3.7 idle
C_MS = 1.8          # one expert GEMV at batch 1, measured (tmac.py, 14336 x 4096, 8 threads)
LAYERS = 32
K = 2               # top-2 routing


def layer_ms(batch, fetched, recall):
    """per-layer time in ms: prefetched stream + unhidden misses"""
    link = fetched * R_MS
    compute = K * batch * C_MS
    misses = K * (1.0 - recall) * R_MS
    return max(link, compute) + misses


def token_ms(batch, fetched, recall):
    return LAYERS * layer_ms(batch, fetched, recall)


def bytes_per_token(fetched):
    """every expert that must be read, prefetched or not"""
    return LAYERS * max(fetched, K) * 14.7 / 1024.0          # GB


# (name, experts fetched ahead of use, recall of the true top-k within those)
# The serial arm fetches 0 ahead: its reads are all late, which is why it is not charged a prefetch stream as well.
POLICIES = [
    ("no prefetch (serial)", 0, 0.0),
    ("exact, 82% recall", K, 0.82),
    ("exact, perfect recall", K, 1.0),
    ("over-fetch 3, 96% recall", K + 1, 0.96),
    ("over-fetch 4, 99% recall", K + 2, 0.99),
]


def main(argv):
    global R_MS
    recall, overfetch = None, None
    for i, a in enumerate(argv):
        if a == "--recall":
            recall = float(argv[i + 1])
        elif a == "--overfetch":
            overfetch = int(argv[i + 1])
        elif a == "--read":
            R_MS = float(argv[i + 1])          # the device's own rate; 3.7 is what copytest measures when idle
    policies = list(POLICIES)
    if recall is not None:
        policies = [p for p in policies if p[1] == (overfetch or K)] + [("custom", overfetch or K, recall)]

    batches = (1, 2, 4, 6, 8, 16)
    print("per token: %d layers, top-%d, expert read %.1f ms, expert GEMV %.1f ms\n" % (LAYERS, K, R_MS, C_MS))
    head = "%-26s %9s" % ("policy", "GB/token")
    for b in batches:
        head += " %9s" % ("B=%d" % b)
    print(head)
    base = {b: token_ms(b, 0, 0.0) for b in batches}
    for name, fetched, p in policies:
        row = "%-26s %9.2f" % (name, bytes_per_token(fetched))
        for b in batches:
            row += " %8.0f%s" % (token_ms(b, fetched, p), "*" if token_ms(b, fetched, p) < base[b] else " ")
        print(row)
    print("\nms/token; '*' = faster than serial at that batch. Serial is the `no prefetch` row.\n")

    print("what the batch changes (exact prefetch, perfect recall vs serial):")
    for b in batches:
        serial = token_ms(b, 0, 0.0)
        best = token_ms(b, K, 1.0)
        print("  B=%-3d serial %6.0f ms (%4.1f tok/s)  prefetched %6.0f ms (%4.1f tok/s)  saving %4.0f%%"
              % (b, serial, 1000.0 / serial, best, 1000.0 / best, 100.0 * (1 - best / serial)))

    print("\nover-fetch against exact fetch at the proposal's own recalls:")
    for b in batches:
        exact = token_ms(b, K, 0.82)
        over = token_ms(b, K + 1, 0.96)
        verdict = "over-fetch wins" if over < exact else "exact wins"
        print("  B=%-3d exact/82%% %6.0f ms   over-fetch/96%% %6.0f ms   %s by %4.0f ms"
              % (b, exact, over, verdict, abs(over - exact)))
    print("\n(both arms are measured against the same %.1f ms read and %.1f ms GEMV; the recalls are the proposal's\n"
          " training-set numbers, and no held-out split was reported for them)" % (R_MS, C_MS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
