"""Run the batched expert-paging pipeline instead of modelling it: is B=8 where streaming beats resident RAM?

The paging note predicted that at 5-6 tokens one expert's ~9 ms read sits under the batch's compute, so by B=8 a
double-buffered stream should cost no more than keeping every expert in memory. That prediction has only ever been
arithmetic from two rates. This executes it: a reader thread fills the next expert's 14.7 MB while the production pair-
table kernel multiplies the current one into the batch's tokens, and the same batch is timed resident (preloaded, no
reads) and streamed (reads overlapped).

    python -m tools.bitnet_gemm.batch_stream                 # the four cases, at the machine's thread count
    python -m tools.bitnet_gemm.batch_stream --batches 8:35  # one batch size and one fetch count
    python -m tools.bitnet_gemm.batch_stream --pool 8 --reps 3

What the numbers mean, and what they do not:

* **measured here**: the wall time of a batch, with real 14.7 MB reads from a real file and real GEMVs through
  `kc_rows8_gemv` (the same `rows8` the packs decode through), for the expert shape the budget assumes -
  14336 x 4096, 58.7 M weights.
* **taken as input**: how many experts a batch must fetch. Those come from `cluster_budget`'s measurement of the
  captured router; `--fetched` is that number for the batch size, because a pipeline cannot invent its own routing.
  Pass `--fetched B*k` to model "the batch shares nothing".
* **not measured**: the device. Reads are answered from somewhere behind the guest's page cache, as the note's own
  probe found, so the read rate here is the host's, not the medium's. The tool reports the per-read time it actually
  got beside the budget's 9.5 ms so the difference is visible rather than assumed.
"""
import ctypes
import os
import sys
import threading
import time

import numpy as np

from kindi import cpu
from tools.bitnet_gemm import libcompare

ROWS, COLS = 14336, 4096                     # one expert: 58.7 M weights, 14.7 MB packed
IN_BYTES = COLS * 4                          # an activation vector, fp32
# (batch, experts fetched for that batch). The fetched counts are cluster_budget's measured unions, rounded:
# 128 experts as routed gives 8.00 at B=1 and 35.06 at B=8; 12 clusters give 5.24 and 9.57. 2 x batch is "the batch
# shares every fetch", which is the assumption the note's B=8 prediction was built on.
CASES = [(1, 8), (8, 35), (8, 10), (8, 16), (16, 48)]


def pool_path():
    root = os.environ.get("KINDI_STREAM_DIR")
    if not root:
        root = "B:/models/scratch" if os.path.isdir("B:/models") else os.path.join(os.path.expanduser("~"),
                                                                                    ".cache", "kindi")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "expert_pool_%dx%d.bin" % (ROWS, COLS))


def build_pool(path, experts, log=print):
    """Write `experts` packed expert matrices (pair codes then scales) so the reads are real bytes off a real file."""
    rec = None
    if os.path.isfile(path):
        with open(path, "rb") as f:
            head = np.fromfile(f, dtype=np.int64, count=3)
        if len(head) == 3 and head[0] == experts and head[1] == ROWS and head[2] == COLS:
            return int(head[0])
    log("  building a %d-expert pool (%.0f MB) at %s" % (experts, experts * ROWS * COLS / 4 / 2 ** 20, path))
    rs = np.random.RandomState(7)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        np.array([experts, ROWS, COLS], dtype=np.int64).tofile(f)
        for _ in range(experts):
            w = rs.randint(-1, 2, size=(ROWS, COLS), dtype=np.int8)
            W, S = libcompare.pack_pair(w)
            f.write(W.tobytes())
            f.write(S.tobytes())
            del w, W, S
        rec = [experts]
    if os.path.exists(path):
        os.remove(path)                       # a half-written pool from an interrupted run would otherwise block the
    os.replace(tmp, path)                     # replace, which Windows refuses when the target is open elsewhere
    return experts


class Slots(object):
    """Two expert-sized slots and a reader thread that fills the next one while the caller multiplies the current.

    The reader gets its own descriptor and does seek+read rather than `os.pread`: pread is Unix-only, and on Windows a
    silent AttributeError in a daemon thread leaves the caller waiting on a slot that will never arrive.
    """

    def __init__(self, path, off0, per, experts, reps):
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self.off0, self.per = off0, per
        self.order = [i % experts for i in range(reps)]
        self.buf = [bytearray(per), bytearray(per)]
        self.ready = threading.Semaphore(0)
        self.free = threading.Semaphore(2)
        self.next_i = 0
        self.read_ms = []
        self.stop = False
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop:
            self.free.acquire()
            i = self.next_i
            self.next_i += 1
            if i >= len(self.order):
                self.ready.release()
                return
            slot = i % 2
            t0 = time.perf_counter()
            n = 0
            os.lseek(self.fd, self.off0 + self.order[i] * self.per, os.SEEK_SET)
            while n < self.per:
                got = os.read(self.fd, self.per - n)
                if not got:
                    raise IOError("kindi batch_stream: the expert pool is shorter than its header says")
                self.buf[slot][n:n + len(got)] = got
                n += len(got)
            self.read_ms.append((time.perf_counter() - t0) * 1e3)
            self.ready.release()

    def close(self):
        self.stop = True
        try:
            os.close(self.fd)
        except OSError:
            pass


def gemv(lib, W, S, x, y, threads):
    lib.kc_rows8_gemv(libcompare._u(W), libcompare._f(S), ROWS, COLS, libcompare._f(x), libcompare._f(y),
                      0, ROWS // 8, threads)


def as_mat(raw):
    """one expert's bytes -> the pair codes and scales the kernel reads, without a copy"""
    nw = ROWS * (COLS // 4)
    W = np.frombuffer(raw, dtype=np.uint32, count=nw // 4)
    S = np.frombuffer(raw, dtype=np.float32, count=(len(raw) - nw) // 4, offset=nw)
    return W, S


def main(argv):
    def arg(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    experts = int(arg("--pool", 32))
    threads = int(arg("--threads", os.cpu_count() or 8))
    reps = int(arg("--reps", 3))
    k = int(arg("--top-k", 8))
    cases = CASES
    if "--batches" in argv:
        cases = []
        for spec in arg("--batches", "").split(","):
            b, f = spec.split(":")
            cases.append((int(b), int(f)))

    lib, so = libcompare.lib()
    print("libkcpu: %s (tiers %s), %d threads" % (os.path.basename(so),
                                                  "avx2" if lib.kc_tier_mask() & 2 else "scalar", threads))
    path = pool_path()
    build_pool(path, experts)
    per = ROWS * (COLS // 4) + (ROWS // 8) * (COLS // 128) * 8 * 4
    print("expert %d x %d: %.1f MB packed, %d in the pool" % (ROWS, COLS, per / 2 ** 20, experts))
    print("")
    print("  %5s %8s %9s %12s %12s %7s  %s" % ("batch", "fetched", "read ms", "streamed ms/t", "resident ms/t",
                                               "ratio", "verdict"))
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        off0 = 24                                   # past the header
        xs = [np.random.RandomState(11 + i).randn(COLS).astype(np.float32) for i in range(max(b for b, _ in cases))]
        y = np.zeros(ROWS, dtype=np.float32)
        read_ms, gemv_ms = [], float("nan")
        for B, fetched in cases:
            fetched = max(1, fetched)          # not clamped to B*k: a batch's union is smaller than its multiplies
            # every token does k expert multiplies; an expert is multiplied by the tokens that routed to it, so the
            # batch's B*k multiplies are shared out over the fetched experts
            assign = [[] for _ in range(fetched)]
            for j in range(B * k):
                assign[j % fetched].append(j)

            resident = []
            for e in range(fetched):
                os.lseek(fd, off0 + (e % experts) * per, os.SEEK_SET)
                resident.append(os.read(fd, per))
            t_res = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                for e, mine in enumerate(assign):
                    W, S = as_mat(resident[e])
                    for j in mine:
                        gemv(lib, W, S, xs[j // k], y, threads)
                t_res = min(t_res, time.perf_counter() - t0)
            gemv_ms = t_res / (B * k) * 1e3

            t_str = float("inf")
            read_ms = []
            for _ in range(reps):
                sl = Slots(path, off0, per, experts, fetched)
                sl.th.start()
                t0 = time.perf_counter()
                for e, mine in enumerate(assign):
                    sl.ready.acquire()                       # the reader has this expert's bytes in slot e % 2
                    W, S = as_mat(sl.buf[e % 2])
                    for j in mine:
                        gemv(lib, W, S, xs[j // k], y, threads)
                    sl.free.release()                        # that slot can be refilled
                sl.stop = True
                sl.th.join(timeout=60)
                t_str = min(t_str, time.perf_counter() - t0)
                read_ms = sl.read_ms
                sl.close()
            r = float(np.median(read_ms)) if read_ms else float("nan")
            ratio = t_str / t_res
            verdict = "streaming" if t_str <= t_res else "resident"
            print("  %5d %8d %9.1f %12.1f %12.1f %7.2f  %s wins" % (B, fetched, r, t_str / B * 1e3, t_res / B * 1e3,
                                                                        ratio, verdict))
    finally:
        os.close(fd)
    print("")
    print("  ratio is streamed / resident, so below 1.00 streaming wins. It can only fall under 1 when the batch's")
    print("  fetches overlap its compute: fetched * read <= batch * k * gemv, i.e. fetched <= batch * k * c / r.")
    print("  With this host's %.1f ms read and %.2f ms GEMV that threshold is %.2f x batch, and the crossing batch is where"
          " the routed union flattens below it." % (read_ms[-1] if read_ms else float("nan"), gemv_ms, gemv_ms / read_ms[-1]
                                                     if read_ms else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
