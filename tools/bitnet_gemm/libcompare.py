"""One matrix, one activation vector, two decoders, one process - pinned and paired.

The paging note claimed the T-MAC LUT kernel "matches the shipped pair table at parity, so a repack of the weight layout
would buy nothing". That claim was made across sources: one number came from `tools/bitnet_gemm/tmac.py` and the other
from `kindi.cpu`, in different processes, on different matrices, with the packing done twice. A difference between them
could be the matrix, the pack layout, or the driver, and no amount of care in either file rules those out. This settles
it the only way it can be settled - both kernels in one process, one matrix, one activation vector, one timing loop.

    python -m tools.bitnet_gemm.libcompare                          # 2048 x 1920 and 14336 x 4096
    python -m tools.bitnet_gemm.libcompare --rounds 15 --runs 3 --threads 8

What it builds: `kindi/kcpu.c` into the same library `kindi.cpu` uses (MSVC here, gcc elsewhere), and `tmac.cpp` as a
torch extension. The pair-table arm runs `rows8` - the kernel the packs actually decode through - via `kc_rows8_gemv`,
a harness entry that takes the engine's own packed bytes so no model is needed. The LUT arm runs `tmac_gemv_avx2`.

Fairness, stated because it is easy to get wrong: both arms get the same ternary weights and the *same int8 activation
vector*, with the activation scale carried the way each kernel expects it (as floats for `rows8`, as `scale_a` for the
LUT), so the only difference left is how each kernel reads four weights at a time. Both arms get an explicit thread
count at every row - the first version of this file left the LUT's count unset, so its "one thread" row was really an
all-core LUT call against a one-thread pair table, and the 0.68x it printed was that mistake and not a kernel result.

Timing method is the one in `tmac.py`, for the same reason: the process is pinned to physical cores, the two arms run
back to back inside every round, and the number reported is the mean of the per-round paired ratios with a 95% bootstrap
interval. A minimum would report the quietest draw of a distribution whose spread is the question.
"""
import ctypes
import os
import sys

import numpy as np

from kindi import cbuild, cpu
from tools.bitnet_gemm import tmac

SHAPES = [(2048, 1920), (14336, 4096)]        # a projection GEMV, and one MoE expert slab


def lib():
    """build (or reuse) libkcpu and return it with the harness entry's signature declared"""
    src = os.path.join(os.path.dirname(os.path.abspath(cpu.__file__)), "kcpu.c")
    cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    os.makedirs(cache, exist_ok=True)
    so = os.environ.get("KINDI_CPU_LIB") or cbuild.build(src, cache)
    L = ctypes.CDLL(so)
    L.kc_init()
    L.kc_tier_mask.restype = ctypes.c_int
    P = ctypes.POINTER
    L.kc_rows8_gemv.argtypes = [P(ctypes.c_ubyte), P(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                                P(ctypes.c_float), P(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.kc_rows8_gemv.restype = ctypes.c_int
    return L, so


def pack_pair(w):
    """(R, C) int8 in {-1, 0, 1} -> the engine's pair codes [R/8][C/128][64] u32 and scales [R/8][C/128][8] float32.

    The bytes come from `kindi.cpu.pair_bytes`, the same function `kindi convert` stages with, so this arm reads the
    real layout rather than a harness-specific one. The scales are 1.0 because the matrix is already ternary - nothing
    is being quantised here, so both kernels see exactly the weights handed in.
    """
    R, C = w.shape
    t = (w.astype(np.int16) + 1).astype(np.uint8)                      # -1/0/+1 -> the pack's 0/1/2 trits
    tr = t.reshape(R, C // 4, 4)
    codes = (tr[:, :, 0] + 3 * tr[:, :, 1] + 9 * tr[:, :, 2] + 27 * tr[:, :, 3]).astype(np.uint8)
    sc = np.ones((C // 128, R), dtype=np.float16)
    Wb, Sb = cpu.pair_bytes(np.ascontiguousarray(codes.T), sc, R, C)
    W = np.frombuffer(Wb, dtype=np.uint32).reshape(R // 8, C // 128, 64)
    S = np.frombuffer(Sb, dtype=np.float32).reshape(R // 8, C // 128, 8)
    return np.ascontiguousarray(W), np.ascontiguousarray(S)


def _f(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _u(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))


def rows8_gemv(L, W, S, x, threads=1):
    R, C = int(W.shape[0] * 8), int(W.shape[2] * 2 * W.shape[1])
    y = np.zeros(R, dtype=np.float32)
    rc = L.kc_rows8_gemv(_u(W), _f(S), R, C, _f(x), _f(y), 0, R // 8, threads)
    if rc != R // 8:
        raise SystemExit("kindi libcompare: kc_rows8_gemv refused the matrix (returned %d)" % rc)
    return y


def fixture(R, C, seed):
    """the two arms' inputs, built once: a repack inside a timing loop is not the kernel and dominated it once"""
    import torch

    g = np.random.default_rng(seed)
    w = g.integers(-1, 2, size=(R, C), dtype=np.int8)
    a_int = np.clip(np.rint(g.standard_normal(C) * 40.0), -127, 127).astype(np.int8)
    scale_a = 1.0 / 37.0
    W, S = pack_pair(w)
    return dict(w=w, a_int=a_int, scale_a=scale_a, W=W, S=S,
                x=a_int.astype(np.float32) * np.float32(scale_a),
                a_t=torch.from_numpy(a_int),
                lut_packed=tmac.pack_for_tmac(torch.from_numpy(w)),
                ones=torch.ones(R))


def correctness(L, mod, fx, R, C):
    """both kernels against an exact integer reference, and against each other"""
    a_int, w, scale_a = fx["a_int"], fx["w"], fx["scale_a"]
    y_pair = rows8_gemv(L, fx["W"], fx["S"], fx["x"], threads=1)
    y_lut = mod.tmac_gemv_avx2(fx["a_t"], fx["lut_packed"], scale_a, fx["ones"]).numpy()
    ref = (a_int.astype(np.int64) @ w.astype(np.int64).T).astype(np.float64) * scale_a
    scale = float(np.abs(ref).max())
    dp = float(np.abs(y_pair - ref).max())
    dl = float(np.abs(y_lut - ref).max())
    dpn = float(np.abs(y_pair - y_lut).max())
    print("\n%d x %d ternary, one int8 activation vector (scale %.4f), %d rows' results each\n" % (R, C, scale_a, R))
    print("  pair table (rows8)  max |y - exact| %.3e  (%.2e of the largest |y|)" % (dp, dp / scale))
    print("  LUT (vpshufb)       max |y - exact| %.3e  (%.2e of the largest |y|)" % (dl, dl / scale))
    print("  the two against each other           %.3e  (%.2e of the largest |y|)" % (dpn, dpn / scale))
    return dpn / scale


def compare(mod, L, fx, R, C, threads, rounds, runs):
    """interleaved paired walk: every arm every round, ratio and interval against the pair table"""
    mod.set_threads(threads)
    arms = [("pair table (rows8)", lambda: rows8_gemv(L, fx["W"], fx["S"], fx["x"], threads)),
            ("LUT (vpshufb)", lambda: mod.tmac_gemv_avx2(fx["a_t"], fx["lut_packed"], fx["scale_a"], fx["ones"]))]
    pooled, per_run = {}, {name: [] for name, _ in arms}
    for _ in range(runs):
        t = tmac.paired(arms, rounds)
        b = np.asarray(t[arms[0][0]], dtype=np.float64)
        for name, _fn in arms:
            pooled.setdefault(name, []).extend(list(t[name]))
            per_run[name].append(float(np.mean(b / np.asarray(t[name], dtype=np.float64))))
    rows = tmac.summarise(pooled)
    print("    %-22s %9s %6s %9s %-15s %-16s %s"
          % ("arm", "median ms", "CV", "vs pair", "95% CI", "per run", "verdict"))
    for r in rows:
        print("    %-22s %9.3f %5.1f%% %8.2fx [%5.2f, %5.2f] %-16s %s   %.1f G weights/s"
              % (r["arm"], r["ms"], r["cv"], r["speedup"], r["lo"], r["hi"],
                 " ".join("%.2f" % s for s in per_run[r["arm"]]), r["verdict"],
                 R * C / (r["ms"] / 1e3) / 1e9))
    return rows


def main(argv):
    def arg(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    rounds = int(arg("--rounds", 15))
    runs = int(arg("--runs", 3))
    threads = int(arg("--threads", 8))
    seed = int(arg("--seed", 0))
    if "--rows" in argv or "--cols" in argv:
        shapes = [(int(arg("--rows", 2048)), int(arg("--cols", 1920)))]
    else:
        shapes = SHAPES
    for R, C in shapes:
        if R % 8 or C % 128:
            raise SystemExit("kindi libcompare: the pair layout needs rows %% 8 == 0 and cols %% 128 == 0 (%d x %d)"
                             % (R, C))

    import torch
    torch.set_num_threads(1)                     # rows8 is already OpenMP; torch must not add a pool of its own

    L, so = lib()
    mask = L.kc_tier_mask()
    print("libkcpu: %s (tiers %s)" % (so, "avx2" if mask & 2 else "scalar"))
    mod = tmac.build()

    worst = 0.0
    for R, C in shapes:
        fx = fixture(R, C, seed)
        worst = max(worst, correctness(L, mod, fx, R, C))

        print("\n  %d x %d, %d interleaved rounds x %d runs, paired against the pair table:"
              % (R, C, rounds, runs))
        for thr in sorted({1, threads}):
            info = tmac.pin_affinity(thr)
            print("\n    threads %d - %s" % (thr, tmac.affinity_line(info)))
            compare(mod, L, fx, R, C, thr, rounds, runs)
        print()

    verdict = ("the two kernels agree to %.2e of the largest output - accumulation order, not a layout difference"
               % worst if worst < 1e-6 else
               "the two kernels differ by %.2e of the largest output, which is more than float rounding" % worst)
    print("  %s" % verdict)
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
