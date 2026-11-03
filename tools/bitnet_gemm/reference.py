"""Ground truth for the kernel: the same unpack, tiling and dot, written in numpy.

`gemm_int` is the exact statement of what the kernel computes - activations and ternary weights multiplied and summed
in integers, where tiling cannot change the answer, so a tiled result that differs from the dense one is a bug and not
rounding. `gemm_packed` adds the scales and the float conversion that the epilogue does.

`tile_unpack` is a line-for-line transcription of the unpack inside the Triton kernel. The kernel cannot be built on a
machine without an XPU or CUDA device, so this transcription is how its arithmetic gets tested: same shifts, same
mask, same reshape order.
"""
import numpy as np

from .packing import MASK, decode


def tile_unpack(packed_tile):
    """(BLOCK_N, BLOCK_K // 4) uint8 -> (BLOCK_N, BLOCK_K) int32 in {-1, 0, 1}: the kernel's unpack."""
    p = np.asarray(packed_tile).astype(np.int32)
    shifts = np.array([0, 2, 4, 6])[None, None, :]
    codes = (p[:, :, None] >> shifts) & MASK          # (N, K/4, 4), field order low to high
    values = (codes & 1) - ((codes >> 1) & 1)         # 0 -> 0, 1 -> +1, 2 -> -1, 3 -> 0
    return values.reshape(p.shape[0], p.shape[1] * 4)


def tile_unpack_words(word_tile):
    """(BLOCK_N, BLOCK_K // 16) uint32 -> (BLOCK_N, BLOCK_K) int32: the unpack the kernel actually runs."""
    p = np.asarray(word_tile).astype(np.uint32)
    shifts = np.arange(16, dtype=np.uint32)[None, None, :] * 2
    codes = (p[:, :, None] >> shifts) & MASK
    values = (codes & 1).astype(np.int32) - ((codes >> 1) & 1).astype(np.int32)
    return values.reshape(p.shape[0], p.shape[1] * 16)


def gemm_int(a, w):
    """(M, K) int activations times (N, K) ternary weights -> (M, N) int32."""
    return np.asarray(a).astype(np.int32) @ np.asarray(w).astype(np.int32).T


def gemm_tiled(a, packed, block_m=32, block_n=64, block_k=64, out=None):
    """The kernel's loop nest, on the host: dense int32 answer, tile by tile, loading weight words."""
    from .packing import as_words, from_words
    a = np.asarray(a).astype(np.int32)
    words = as_words(packed)
    M, K = a.shape
    N = words.shape[0]
    if K % 16 or block_k % 16 or K % block_k:
        raise ValueError("K must be a multiple of 16 and of block_k")
    c = np.zeros((M, N), np.int32) if out is None else out
    for m0 in range(0, M, block_m):
        rows = min(block_m, M - m0)
        for n0 in range(0, N, block_n):
            cols = min(block_n, N - n0)
            acc = np.zeros((rows, cols), np.int32)
            for k0 in range(0, K, block_k):
                w_tile = tile_unpack_words(words[n0:n0 + cols, k0 // 16:(k0 + block_k) // 16])
                acc += a[m0:m0 + rows, k0:k0 + block_k] @ w_tile.T
            c[m0:m0 + rows, n0:n0 + cols] = acc
    return c


def gemm_packed(a, packed, scale_a=None, scale_w=None):
    """The full statement, including the epilogue: C = (A @ W^T) * scale_a[:, None] * scale_w[None, :]."""
    w = decode(packed)
    out = np.asarray(a).astype(np.float32) @ w.astype(np.float32).T
    return apply_scales(out, scale_a, scale_w)


def gemm_dense(a, w, scale_a=None, scale_w=None):
    """The same with an unpacked ternary matrix, for the round-trip test."""
    out = np.asarray(a).astype(np.float32) @ np.asarray(w).astype(np.float32).T
    return apply_scales(out, scale_a, scale_w)


def apply_scales(out, scale_a=None, scale_w=None):
    out = np.asarray(out, np.float32)
    if scale_a is not None:
        sa = np.asarray(scale_a, np.float32).reshape(-1)
        if sa.size == 1:
            out = out * sa[0]
        else:
            out = out * sa[:, None]
    if scale_w is not None:
        sw = np.asarray(scale_w, np.float32).reshape(-1)
        if sw.size == 1:
            out = out * sw[0]
        else:
            out = out * sw[None, :]
    return out
