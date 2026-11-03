"""2-bit ternary packing: four weights per byte, in K order.

    code 0 -> 0
    code 1 -> +1
    code 2 -> -1
    code 3 -> unused, and decodes to 0

Two bits per weight is the smallest fixed-width code that holds three values, so the weight stream is a quarter of an
int8 tensor and a sixteenth of fp16. The code is *not* two's complement (that would put -1 at 3): the decoder below is
`(code & 1) - (code >> 1)`, which is two integer ops per weight and no branching, and it is the same expression the
Triton kernel uses.

`decode_lut` is the same map through a 256-entry byte table - the arm the host comparison in `test_bitnet.py` times
against the shift form.

The packing order is the one that matters for unpacking: the four weights of byte `k // 4` are the weights `k, k+1,
k+2, k+3`, low field first. A decoder therefore recovers them by shifting 0, 2, 4, 6 and can lay them back out with a
plain row-major reshape - no scatter, no per-lane permute.

`as_words` views four consecutive bytes as one uint32, so the kernel can load sixteen weights in a 32-bit load and
unpack them by shifting 0, 2, ..., 30. The field order is unchanged by that view: little-endian, byte 0 in bits 0-7.
"""
import numpy as np

MASK = 0b11
UNUSED = 3


def encode(w):
    """(-1, 0, 1) weights, shape (N, K), K a multiple of 4 -> uint8 codes (N, K // 4)."""
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError("weights must be 2-D (rows, K), got shape %r" % (w.shape,))
    if w.shape[1] % 4:
        raise ValueError("K must be a multiple of 4 to pack four weights per byte, got %d" % w.shape[1])
    bad = ~np.isin(w, (-1, 0, 1))
    if bad.any():
        raise ValueError("%d weights are outside {-1, 0, 1}; ternary only" % int(bad.sum()))
    codes = np.zeros(w.shape, dtype=np.uint8)
    codes[w == 1] = 1
    codes[w == -1] = 2
    codes = codes.reshape(w.shape[0], w.shape[1] // 4, 4)
    return (codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)).astype(np.uint8)


def decode(packed):
    """uint8 codes (N, K // 4) -> int8 weights (N, K) in {-1, 0, 1}."""
    p = np.asarray(packed).astype(np.int32)
    if p.ndim != 2:
        raise ValueError("packed weights must be 2-D (rows, K // 4), got shape %r" % (p.shape,))
    shifts = np.array([0, 2, 4, 6])[None, None, :]
    codes = (p[:, :, None] >> shifts) & MASK
    values = (codes & 1) - ((codes >> 1) & 1)
    return values.astype(np.int8).reshape(p.shape[0], p.shape[1] * 4)


def as_words(packed):
    """uint8 codes (N, K // 4) -> uint32 words (N, K // 16), little-endian, four bytes per word.

    The kernel loads these instead of bytes. A byte-granular load compiles to one `ld.global.b8` per byte per thread
    (measured: 8 of them per thread per K step for a 64x64 tile), which is four times the load instructions the same
    bytes need. K must be a multiple of 16 for a row to be whole words.
    """
    p = np.ascontiguousarray(np.asarray(packed).astype(np.uint8))
    if p.ndim != 2 or p.shape[1] % 4:
        raise ValueError("packed weights must be 2-D with K // 4 a multiple of 4 (K a multiple of 16), got %r"
                         % (p.shape,))
    return p.view(np.uint32).reshape(p.shape[0], p.shape[1] // 4)


def from_words(words):
    """uint32 words (N, K // 16) -> uint8 codes (N, K // 4)."""
    w = np.ascontiguousarray(np.asarray(words).astype(np.uint32))
    if w.ndim != 2:
        raise ValueError("words must be 2-D (rows, K // 16), got %r" % (w.shape,))
    return w.view(np.uint8).reshape(w.shape[0], w.shape[1] * 4)


def torch_as_words(packed, torch=None):
    """torch mirror of as_words(): uint8 (N, K // 4) -> int32 (N, K // 16), four packed bytes per word."""
    torch = torch or _torch()
    p = packed.contiguous()
    if p.dim() != 2 or p.shape[1] % 4:
        raise ValueError("K must be a multiple of 16, got packed shape %r" % (tuple(p.shape),))
    return p.view(torch.int32).contiguous()


# One row per possible byte: the four weights it holds, in K order. 256 bytes of table, built once at import.
def _build_table():
    table = np.zeros((256, 4), dtype=np.int8)
    for b in range(256):
        for i in range(4):
            code = (b >> (2 * i)) & 3
            table[b, i] = (code & 1) - (code >> 1)
    return table


TABLE = _build_table()


def decode_lut(packed):
    """decode() through a 256-entry table: one gather, no per-field arithmetic.

    Same answer as `decode` (there is a test for it), and on the host it is the fast path - an index by byte instead of
    four shifts, four masks and a subtract per weight, all of which materialise int32 temporaries.
    """
    p = np.ascontiguousarray(np.asarray(packed).astype(np.uint8))
    if p.ndim != 2:
        raise ValueError("packed weights must be 2-D (rows, K // 4), got shape %r" % (p.shape,))
    return TABLE[p].reshape(p.shape[0], p.shape[1] * 4)


def encode_bits(w):
    """The same packing as two bit planes, (bit0, bit1) per field - what a kernel loads if it prefers bit ops."""
    w = np.asarray(w)
    lo = (w == 1).astype(np.uint8)
    hi = (w == -1).astype(np.uint8)
    return lo, hi


def torch_encode(w, torch=None):
    """torch mirror of encode(): int8 tensor in {-1, 0, 1} -> uint8 (N, K // 4)."""
    torch = torch or _torch()
    if w.dim() != 2 or w.shape[1] % 4:
        raise ValueError("weights must be 2-D with K a multiple of 4, got %r" % (tuple(w.shape),))
    if bool((w.abs() > 1).any()):
        raise ValueError("weights are outside {-1, 0, 1}; ternary only")
    codes = torch.zeros(w.shape, dtype=torch.int32, device=w.device)
    codes = torch.where(w == 1, torch.ones_like(codes), codes)
    codes = torch.where(w == -1, torch.full_like(codes, 2), codes)
    codes = codes.reshape(w.shape[0], w.shape[1] // 4, 4)
    packed = codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)
    return packed.to(torch.uint8)


def torch_decode(packed, torch=None):
    """torch mirror of decode(): uint8 (N, K // 4) -> int8 (N, K)."""
    torch = torch or _torch()
    p = packed.to(torch.int32)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=p.device)
    codes = (p[:, :, None] >> shifts[None, None, :]) & MASK
    values = (codes & 1) - ((codes >> 1) & 1)
    return values.reshape(p.shape[0], p.shape[1] * 4).to(torch.int8)


def _torch():
    try:
        import torch
    except ImportError as e:                      # noqa: BLE001 - the numpy path must stay usable without torch
        raise RuntimeError("torch is not installed; the numpy path (encode/decode) needs none") from e
    return torch
