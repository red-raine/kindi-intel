"""Build and exercise the corrected T-MAC LUT GEMV, and measure it against the same machine's other decoders.

    python -m tools.bitnet_gemm.tmac            # build (once), validate, benchmark
    python -m tools.bitnet_gemm.tmac --rebuild

The kernel is `tmac.cpp`. It is the design note's idea - precompute the 16 partial sums of two activations, fetch them
with `vpshufb` - with int16 tables instead of int8, because a two-weight table of int8 activations holds up to 254 and
wraps. What that costs is one extra shuffle per table; what it buys is that int8 activations stay int8.
"""
import os
import sys
import time

import numpy as np
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))


def _flags():
    if os.name == "nt":
        return ["/O2", "/arch:AVX2", "/openmp", "/std:c++17"]
    return ["-O3", "-mavx2", "-ffast-math", "-std=c++17"]


def build(rebuild=False):
    return load(name="tmac_bitnet_avx2", sources=[os.path.join(HERE, "tmac.cpp")],
                extra_cflags=_flags(), extra_ldflags=[] if os.name == "nt" else ["-fopenmp"],
                verbose=rebuild, with_cuda=False, is_python_module=True)


def pack_for_tmac(w):
    """(N, K) int8 in {-1, 0, 1} -> (K // 4, N) uint8: the byte-per-neuron-per-group layout the kernel streams."""
    n, k = w.shape
    assert k % 4 == 0
    codes = torch.zeros_like(w, dtype=torch.uint8)
    codes[w == 1] = 1
    codes[w == -1] = 2
    packed = codes[:, 0::4] | (codes[:, 1::4] << 2) | (codes[:, 2::4] << 4) | (codes[:, 3::4] << 6)
    return packed.t().contiguous()


def reference(a_q, w, scale_a, scale_w):
    """int8 activations times ternary weights, exactly: the answer both arms must reproduce."""
    return (a_q.to(torch.int32) @ w.to(torch.int32).t()).to(torch.float32) * scale_a * scale_w.unsqueeze(0)


def validate(mod, n=2048, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    a = torch.clamp((torch.randn(k, generator=g) * 40).round(), -127, 127).to(torch.int8)
    scale_a = 1.0 / 37.0
    sw = torch.rand(n, generator=g) + 0.5
    out = mod.tmac_gemv_avx2(a, pack_for_tmac(w), scale_a, sw)
    want = reference(a.unsqueeze(0), w, scale_a, sw)[0]
    diff = float((out - want).abs().max())
    return diff, w, a


def bench(mod, n, k, threads=None, reps=9):
    g = torch.Generator().manual_seed(1)
    w = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    packed = pack_for_tmac(w)
    a = torch.clamp((torch.randn(k, generator=g) * 40).round(), -127, 127).to(torch.int8)
    sw = torch.ones(n)
    mod.tmac_gemv_avx2(a, packed, 1.0, sw)                     # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        mod.tmac_gemv_avx2(a, packed, 1.0, sw)
        best = min(best, time.perf_counter() - t0)
    return best


def validate_tiled(mod, n=2048, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    a = torch.clamp((torch.randn(k, generator=g) * 40).round(), -127, 127).to(torch.int8)
    scale_a = 1.0 / 37.0
    sw = torch.rand(n, generator=g) + 0.5
    want = reference(a.unsqueeze(0), w, scale_a, sw)[0]
    worst = 0.0
    for n_blk, k_grp in ((0, 0), (1024, 128), (1024, 0), (32, 128)):
        out = mod.tmac_gemv_avx2_tiled(a, pack_for_tmac(w), scale_a, sw, n_blk, k_grp)
        worst = max(worst, float((out - want).abs().max()))
    return worst


def bench_tiled(mod, n, k, n_blk, k_grp, reps=5):
    g = torch.Generator().manual_seed(1)
    w = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    packed = pack_for_tmac(w)
    a = torch.clamp((torch.randn(k, generator=g) * 40).round(), -127, 127).to(torch.int8)
    sw = torch.ones(n)
    mod.tmac_gemv_avx2_tiled(a, packed, 1.0, sw, n_blk, k_grp)      # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        mod.tmac_gemv_avx2_tiled(a, packed, 1.0, sw, n_blk, k_grp)
        best = min(best, time.perf_counter() - t0)
    return best


def tiling_bench(mod):
    """Does cache blocking pay for itself on a real MoE expert width?

    The blocked arm parallelises over N tiles and rebuilds the tables once per tile (a table is only useful to the
    neurons that built it), so the question is whether keeping the tables hot - and the accumulators contiguous -
    buys back that rebuild. Tile count has to exceed the thread count or the arm is partly idle, so the second shape
    is where the tiling is expected to stop paying.
    """
    print("\nMoE expert GEMV, K=4096, 8 threads: does the note's tiling pay?")
    print("  %-30s %10s %10s %12s" % ("shape / blocking", "ms", "vs untiled", "packed GB/s"))
    for n in (14336, 2048):
        k = 4096
        nbytes = n * k / 4
        base = min(bench(mod, n, k) * 1e3 for _ in range(3))        # same min-of-3 as the tiled arms
        rows = [("untiled (kernel)", None)]
        if n == 14336:
            rows += [("1024 x 512 (the note)", (1024, 128)), ("1024 x all K", (1024, 0)),
                     ("512 x 512", (512, 128)), ("2048 x 1024", (2048, 256))]
        else:
            rows += [("1024 x 512", (1024, 128)), ("512 x 512", (512, 128)), ("2048 x 1024", (2048, 256))]
        for label, cfg in rows:
            if cfg is None:
                ms = base
            else:
                ms = min(bench_tiled(mod, n, k, cfg[0], cfg[1]) * 1e3 for _ in range(3))
            print("  %-30s %10.3f %9.2fx %12.2f"
                  % ("N=%d %s" % (n, label), ms, base / ms, nbytes / (ms / 1e3) / 1e9))
        print()


def main(argv):
    torch.set_num_threads(1)
    mod = build("--rebuild" in argv)
    diff, _w, _a = validate(mod)
    print("validation: max |LUT - exact int8 reference| = %.3e  %s" % (diff, "PASS" if diff == 0 else "FAIL"))
    print("\nGEMV, weights decoded from 2-bit codes inside the kernel (0.25 bytes a weight)")
    print("  %-12s %4s %10s %12s %10s" % ("N x K", "thr", "ms", "weights/s", "packed GB/s"))
    for threads in (1, 8):
        if hasattr(mod, "set_threads"):
            mod.set_threads(threads)
        for n, k in ((1024, 1024), (2048, 1920), (4096, 4096)):
            ms = bench(mod, n, k) * 1e3
            nbytes = n * k / 4
            print("  %-12s %4d %10.4f %11.1fM %10.2f"
                  % ("%dx%d" % (n, k), threads, ms, n * k / ms / 1e3, nbytes / (ms / 1e3) / 1e9))
    if hasattr(mod, "tmac_gemv_avx2_tiled"):
        worst = validate_tiled(mod)
        print("\ntiled arm: max |tiled - exact int8 reference| = %.3e over 4 blockings  %s"
              % (worst, "PASS" if worst == 0 else "FAIL"))
        diff = max(diff, worst)
        mod.set_threads(8)
        tiling_bench(mod)
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
