"""Push one matrix through the pair-table kernel and the LUT kernel in a single process, and report both.

The paging note claimed the T-MAC LUT kernel "matches the shipped pair table at parity, so a repack of the weight layout
would buy nothing". That claim was made across sources: one number came from `tools/bitnet_gemm/tmac.py` and the other
from `kindi.cpu`, in different processes, on different matrices, with the packing done twice. A difference between them
could be the matrix, the pack layout, or the driver, and no amount of care in either file rules those out. This settles
it the only way it can be settled - both kernels in one process, one matrix, one activation vector, one timing loop.

    python -m tools.bitnet_gemm.libcompare
    python -m tools.bitnet_gemm.libcompare --rows 2048 --cols 1920 --reps 9 --threads 8

What it builds: `kindi/kcpu.c` into the same library `kindi.cpu` uses (MSVC here, gcc elsewhere), and `tmac.cpp` as a
torch extension. The pair-table arm runs `rows8` - the kernel the packs actually decode through - via `kc_rows8_gemv`,
a harness entry that takes the engine's own packed bytes so no model is needed. The LUT arm runs `tmac_gemv_avx2`.

Fairness, stated because it is easy to get wrong: both arms get the same ternary weights and the *same int8 activation
vector*, with the activation scale carried the way each kernel expects it (as floats for `rows8`, as `scale_a` for the
LUT), so the only difference left is how each kernel reads four weights at a time. Timing is best-of-N for each arm at
one thread, and the pair-table arm additionally at the engine's thread count, since that is how it is really called.
"""
import ctypes
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from kindi import cbuild, cpu

ROWS, COLS = 2048, 1920


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


def msvc_env():
    """Put cl.exe on this process's PATH.

    `kindi.cbuild` calls cl.exe through vcvarsall itself, but torch's cpp_extension shells out to `where cl`, so the
    LUT arm needs the environment imported rather than a wrapped command. Skipped off Windows and when cl is already
    reachable.
    """
    if os.name != "nt" or shutil.which("cl"):
        return
    vc = cbuild.find_vcvarsall()
    if not vc:
        raise SystemExit("kindi libcompare: no Visual Studio VC toolset to build the LUT extension")
    out = subprocess.run('call "%s" x64 >nul && set' % vc, shell=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v


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


def best_of(fn, reps):
    fn()                                                              # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv):
    def arg(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    R = int(arg("--rows", ROWS))
    C = int(arg("--cols", COLS))
    reps = int(arg("--reps", 9))
    threads = int(arg("--threads", os.cpu_count() or 8))
    seed = int(arg("--seed", 0))
    if R % 8 or C % 128:
        raise SystemExit("kindi libcompare: the pair layout needs rows % 8 == 0 and cols % 128 == 0 (%d x %d)"
                         % (R, C))

    import torch
    from tools.bitnet_gemm import tmac

    L, so = lib()
    mask = L.kc_tier_mask()
    print("libkcpu: %s (tiers %s)" % (so, "avx2" if mask & 2 else "scalar"))

    g = np.random.default_rng(seed)
    w = g.integers(-1, 2, size=(R, C), dtype=np.int8)
    a_int = np.clip(np.rint(g.standard_normal(C) * 40.0), -127, 127).astype(np.int8)
    scale_a = 1.0 / 37.0

    W, S = pack_pair(w)
    x = a_int.astype(np.float32) * np.float32(scale_a)                # rows8 takes floats
    y_pair = rows8_gemv(L, W, S, x, threads=1)

    msvc_env()
    mod = tmac.build()
    a_t = torch.from_numpy(a_int)
    packed = tmac.pack_for_tmac(torch.from_numpy(w))                 # once: a repack inside the timing loop is not
    ones = torch.ones(R)                                             # the kernel's cost, and it dominated it here
    y_lut = mod.tmac_gemv_avx2(a_t, packed, scale_a, ones).numpy()

    ref = (a_int.astype(np.int64) @ w.astype(np.int64).T).astype(np.float64) * scale_a
    dp = float(np.abs(y_pair - ref).max())
    dl = float(np.abs(y_lut - ref).max())
    dpn = float(np.abs(y_pair - y_lut).max())
    scale = float(np.abs(ref).max())
    print("\n%d x %d ternary, one int8 activation vector (scale %.4f), %d rows' results each\n" % (R, C, scale_a, R))
    print("  pair table (rows8)  max |y - exact| %.3e  (%.2e of the largest |y|)" % (dp, dp / scale))
    print("  LUT (vpshufb)       max |y - exact| %.3e  (%.2e of the largest |y|)" % (dl, dl / scale))
    print("  the two against each other           %.3e  (%.2e of the largest |y|)" % (dpn, dpn / scale))

    t_pair = best_of(lambda: rows8_gemv(L, W, S, x, threads=1), reps)
    t_lut = best_of(lambda: mod.tmac_gemv_avx2(a_t, packed, scale_a, ones), reps)
    print("\n  %4.2f M weights a call, best of %d\n" % (R * C / 1e6, reps))
    for name, t in (("pair table, 1 thread", t_pair), ("LUT, 1 thread", t_lut)):
        print("  %-22s %7.3f ms  %6.1f G weights/s" % (name, t * 1e3, R * C / t / 1e9))
    t_mt = best_of(lambda: rows8_gemv(L, W, S, x, threads=threads), reps)
    print("  %-22s %7.3f ms  %6.1f G weights/s  (%d threads)" % ("pair table, engine", t_mt * 1e3,
                                                                 R * C / t_mt / 1e9, threads))
    print("\n  ratio at one thread: LUT / pair table = %.2fx" % (t_lut / t_pair))

    verdict = ("the two kernels agree to %.2e of the largest output, so the parity claim holds within float rounding"
               % (dpn / scale) if dpn / scale < 1e-6 else
               "the two kernels differ by %.2e of the largest output, which is more than float rounding" % (dpn / scale))
    print("\n  %s" % verdict)
    return 0 if dpn / scale < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
