"""Tests for the 2-bit ternary GEMM: packing, the tiled reference, the scales, and what the kernel can prove here.

    python -m tools.bitnet_gemm.test_bitnet           # or: python tools/bitnet_gemm/test_bitnet.py
    python -m tools.bitnet_gemm.test_bitnet --bench   # also time the unpack against a dense float GEMM

The Triton kernel is compared against the reference only when a device exists for it. Everything else - the packing,
the field order, the tile nest the kernel mirrors, the epilogue - runs on the host, which is why the kernel's arithmetic
is testable at all on a machine with no Intel GPU.
"""
import sys
import time

import numpy as np

from . import kernel, packing, reference

PASS = []


def check(name, fn):
    fn()
    PASS.append(name)
    print("ok   %s" % name)


def ternary(rng, n, k):
    return rng.integers(-1, 2, size=(n, k)).astype(np.int8)


def test_roundtrip():
    rng = np.random.default_rng(0)
    for n, k in ((1, 4), (3, 8), (17, 64), (64, 256)):
        w = ternary(rng, n, k)
        packed = packing.encode(w)
        assert packed.shape == (n, k // 4), packed.shape
        assert np.array_equal(packing.decode(packed), w), "pack/decode is not a round trip for %dx%d" % (n, k)
    # every code is exercised, including the unused one, which must decode to 0
    all_codes = np.array([[0 | (1 << 2) | (2 << 4) | (3 << 6)]], dtype=np.uint8)
    assert np.array_equal(packing.decode(all_codes), [[0, 1, -1, 0]]), packing.decode(all_codes)


def test_encode_rejects():
    for bad, why in ((np.array([[2, 0, 0, 0]]), "value outside the ternary set"),
                     (np.array([[-1, 0]]), "K not a multiple of four")):
        try:
            packing.encode(bad)
        except ValueError:
            continue
        raise AssertionError("encode accepted %s" % why)


def test_word_view():
    """The 32-bit view must not reorder fields: little-endian words, byte 0 in the low bits."""
    rng = np.random.default_rng(6)
    w = ternary(rng, 5, 64)
    packed = packing.encode(w)
    words = packing.as_words(packed)
    assert words.dtype == np.uint32 and words.shape == (5, 4)
    assert np.array_equal(packing.from_words(words), packed)
    assert np.array_equal(reference.tile_unpack_words(words), w)
    # a word whose top byte is all ones is negative as int32; the field extraction must still be right
    top = packing.encode(np.array([[-1] * 16], dtype=np.int8))
    as_signed = packing.as_words(top).view(np.int32)
    assert as_signed[0, 0] < 0, as_signed
    assert np.array_equal(reference.tile_unpack_words(as_signed.view(np.uint32)), [[-1] * 16])


def test_compile_report():
    """Compile for a target without a device: the frontend, the matrix path and the load width."""
    if kernel.triton is None:
        print("skip compile report (triton is not installed)")
        return
    for block in ((32, 64, 64), (64, 64, 128), (16, 32, 32)):
        r = kernel.compile_report(block=block)
        assert not r["problems"], "%s: %s" % (block, r["problems"])
        assert r["mma"] > 0
        print("     block %-14s mma %-3d loads %s shared %d" % (block, r["mma"], r["loads"], r["shared"]))


def test_tile_order():
    """The unpack's reshape must line the fields up with K, not reverse or interleave them."""
    w = np.array([[1, -1, 0, 1, -1, 0, 0, 1]], dtype=np.int8)          # K = 8, two bytes
    packed = packing.encode(w)
    assert np.array_equal(reference.tile_unpack(packed), w), reference.tile_unpack(packed)
    assert np.array_equal(reference.tile_unpack(packed[:, :1]), w[:, :4])


def test_tiled_matches_dense():
    rng = np.random.default_rng(1)
    for (m, n, k, bm, bn, bk) in ((37, 53, 128, 32, 64, 64), (16, 16, 64, 16, 16, 64), (100, 31, 256, 32, 16, 32)):
        a = rng.integers(-127, 128, size=(m, k)).astype(np.int8)
        w = ternary(rng, n, k)
        packed = packing.encode(w)
        want = reference.gemm_int(a, w)
        got = reference.gemm_tiled(a, packed, block_m=bm, block_n=bn, block_k=bk)
        assert got.shape == want.shape
        assert np.array_equal(got, want), "tiled differs from dense at (%d, %d, %d)" % (m, n, k)


def test_float_path_and_scales():
    rng = np.random.default_rng(2)
    m, n, k = 33, 47, 128
    a = rng.standard_normal((m, k)).astype(np.float32)
    w = ternary(rng, n, k)
    packed = packing.encode(w)
    base = reference.gemm_dense(a, w)
    assert np.array_equal(reference.gemm_packed(a, packed), base)
    sa = rng.random(m).astype(np.float32) + 0.5
    sw = rng.random(n).astype(np.float32) + 0.5
    got = reference.gemm_packed(a, packed, sa, sw)
    assert np.allclose(got, base * sa[:, None] * sw[None, :], rtol=0, atol=2e-4), "per-row/column scales"
    assert np.allclose(reference.gemm_packed(a, packed, 0.25, sw), base * 0.25 * sw[None, :], rtol=0, atol=2e-4), \
        "per-tensor activation scale"


def test_torch_mirror():
    try:
        import torch
    except ImportError:
        print("skip torch mirror (torch is not installed)")
        return
    rng = np.random.default_rng(3)
    w = ternary(rng, 12, 64)
    packed = packing.encode(w)
    t = torch.from_numpy(w.astype(np.int8))
    assert np.array_equal(packing.torch_encode(t).numpy(), packed)
    assert np.array_equal(packing.torch_decode(torch.from_numpy(packed)).numpy(), w)


def test_kernel_or_its_gate():
    """On a machine with a device, compare the kernel with the reference; otherwise the gate must explain itself."""
    env = kernel.environment()
    if env["reason"]:
        try:
            kernel.gemm(np.zeros((16, 64), np.float16), np.zeros((16, 16), np.uint8))
        except RuntimeError as e:
            assert env["reason"] in str(e), str(e)
            print("skip triton kernel: %s" % env["reason"])
            return
        raise AssertionError("gemm() did not refuse without a backend")
    import torch
    rng = np.random.default_rng(4)
    m, n, k = 64, 64, 64
    a = rng.standard_normal((m, k)).astype(np.float16)
    w = ternary(rng, n, k)
    packed = packing.encode(w)
    dev = env["device"]
    ta = torch.from_numpy(a).to(dev)
    tp = torch.from_numpy(packed).to(dev)
    got = kernel.gemm(ta, tp).float().cpu().numpy()
    want = reference.gemm_packed(a.astype(np.float32), packed)
    assert np.allclose(got, want, rtol=0, atol=5e-2), "kernel differs: max abs %.3g" % np.abs(got - want).max()
    # the same call with weights already in word form must agree with the byte form
    tw = torch.from_numpy(packing.as_words(packed)).to(dev)
    got2 = kernel.gemm(ta, tw).float().cpu().numpy()
    assert np.array_equal(got, got2), "byte-form and word-form weights disagree"


def bench(sizes=((256, 1024, 1024), (1024, 1024, 1024), (2048, 2048, 2048))):
    """What the on-the-fly unpack costs next to the multiply, on this CPU.

    Both arms call the same BLAS GEMM on fp32 weights. The only difference is that one arm pays the unpack first, so the
    delta is the unpack and nothing else. Reported against the weight traffic (0.25 bytes a weight packed, 4 bytes a
    weight as fp32), which is what makes the unpack cheap or dear.
    """
    import os

    import numpy as _np
    print("\nhost decode cost (numpy %s, %d threads, %s) - the engine's AVX2 kernel and the Arc are both elsewhere" %
          (_np.__version__, os.cpu_count(), "Gen9-class AVX2" if _np.core._multiarray_umath.__cpu_features__.get("AVX2", 0) else "?"))
    print("  %-16s %8s %8s %8s %9s %9s %8s" %
          ("MxNxK", "shift", "table", "memcpy", "gemm", "table/gemm", "w MB"))
    rng = _np.random.default_rng(5)
    for m, n, k in sizes:
        a = rng.standard_normal((m, k)).astype(_np.float32)
        w = ternary(rng, n, k)
        packed = packing.encode(w)
        wf = w.astype(_np.float32)
        best = dict(shift=float("inf"), table=float("inf"), copy=float("inf"), gemm=float("inf"))
        for _ in range(3):
            t0 = time.perf_counter()
            want = a @ wf.T
            best["gemm"] = min(best["gemm"], time.perf_counter() - t0)
            t0 = time.perf_counter()
            shift = packing.decode(packed).astype(_np.float32)
            best["shift"] = min(best["shift"], time.perf_counter() - t0)
            t0 = time.perf_counter()
            table = packing.decode_lut(packed).astype(_np.float32)
            best["table"] = min(best["table"], time.perf_counter() - t0)
            t0 = time.perf_counter()
            floor = wf.copy()                                   # the same bytes written, nothing decoded
            best["copy"] = min(best["copy"], time.perf_counter() - t0)
            assert _np.array_equal(shift, wf), "shift decode disagrees with the weights"
            assert _np.array_equal(table, wf), "table decode disagrees with the shift decode"
            assert _np.array_equal(floor, wf)
            assert _np.array_equal(a @ table.T, want), "the decoded weights do not give the fp32 answer"
        print("  %-16s %7.2fms %7.2fms %7.2fms %8.2fms %8.2fx %8.1f"
              % ("%dx%dx%d" % (m, n, k), best["shift"] * 1e3, best["table"] * 1e3, best["copy"] * 1e3,
                 best["gemm"] * 1e3, best["table"] / best["gemm"], wf.nbytes / 2 ** 20))
    print("  shift/table is the cost of the arithmetic decode against the byte table; copy is the floor for")
    print("  any pass that writes fp32 weights out. These are numpy passes, not the AVX2 GEMV in kindi/kcpu.c")


def main(argv):
    check("pack/decode round trip", test_roundtrip)
    check("encode rejects bad input", test_encode_rejects)
    check("32-bit word view", test_word_view)
    check("kernel compiles to the matrix path", test_compile_report)
    check("unpack field order matches K", test_tile_order)
    check("tiled GEMM equals dense", test_tiled_matches_dense)
    check("float path and scales", test_float_path_and_scales)
    check("torch mirror", test_torch_mirror)
    check("triton kernel (or its gate)", test_kernel_or_its_gate)
    if "--bench" in argv:
        bench()
    print("\n%d checks passed" % len(PASS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
