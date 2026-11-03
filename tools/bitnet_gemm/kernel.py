"""Triton ternary GEMM: packed 2-bit weights, unpacked in registers, multiplied on the matrix engine.

The kernel is the transcription of `reference.gemm_tiled`, which is the version that can be tested without an Intel
GPU or a Triton CPU backend: same tile nest, same shifts, same reshape order. What differs on hardware is only where
the multiply happens - `tl.dot` on the tile of unpacked weights, which lands on XMX on an Xe iGPU and on the tensor
cores of an NVIDIA card.

Design notes that differ from the first draft of this kernel:

* no `tl.tensor_scatter` - Triton has no such primitive. The sixteen weights of a word are recovered by shifting 0, 2,
  ..., 30 on a third axis and then collapsing it with a row-major `tl.reshape`, which is what makes the field order
  fall out for free;
* no `tl.where` chain - `(code & 1) - (code >> 1)` is the whole decoder, two integer ops per weight instead of up to
  eight selects;
* the packed tile is loaded once per tile and never re-read per field: the shift happens on the register tile
  (BLOCK_N, BLOCK_K // 16, 16), which is 4x smaller in memory than the unpacked (BLOCK_N, BLOCK_K) it produces;
* weight words, not weight bytes. Loading the byte form compiles to one `ld.global.b8` per byte per thread - measured
  8 of them per thread per K step for a 64x64 tile - where the same bytes as words compile to 32-bit loads. The
  decoder is unchanged by the view; only the load width is;
* K is required to be a multiple of BLOCK_K, so the inner loop needs no K mask; M and N are masked, which is where
  edge tiles are.

`compile_report()` builds the kernel for a target without a device present, which is the only check of the code above
that a machine without an Intel GPU can run.
"""
import collections
import re

try:                                                 # importing this module must work without triton installed
    import triton
    import triton.language as tl
except ImportError:                                  # pragma: no cover - exercised only without triton
    triton = None
    tl = None

DOT_MIN = 16                                        # tl.dot needs every dimension at least 16
WORD = 16                                           # ternary values per 32-bit word
DEFAULTS = {"xpu": (32, 64, 64), "cuda": (64, 64, 64), "cpu": (16, 32, 32)}


if triton is not None:

    @triton.jit
    def _ternary_gemm_kernel(a_ptr, w_ptr, c_ptr, sa_ptr, sw_ptr,
                             M, N, K,
                             stride_am, stride_ak, stride_wn, stride_wk,
                             stride_cm, stride_cn,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             HAS_SA: tl.constexpr, HAS_SW: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=offs_m[:, None] < M, other=0.0)

            # a word holds 16 K values, so BLOCK_K // 16 of them cover this step
            offs_kw = k0 // 16 + tl.arange(0, BLOCK_K // 16)
            words = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_kw[None, :] * stride_wk,
                            mask=offs_n[:, None] < N, other=0)
            shifts = tl.arange(0, 16) * 2
            codes = (words[:, :, None] >> shifts[None, None, :]) & 3
            values = (codes & 1) - ((codes >> 1) & 1)                    # 0, +1, -1
            w = tl.reshape(values, (BLOCK_N, BLOCK_K)).to(tl.float16)

            acc = tl.dot(a.to(tl.float16), tl.trans(w), acc)

        if HAS_SA:
            sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=1.0)
            acc = acc * sa[:, None]
        if HAS_SW:
            sw = tl.load(sw_ptr + offs_n, mask=offs_n < N, other=1.0)
            acc = acc * sw[None, :]

        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                 acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def environment():
    """What this machine has, in the words the error message uses."""
    try:
        import torch
    except ImportError:
        return dict(torch=None, triton=getattr(triton, "__version__", None), device=None, reason="torch is not installed")
    if triton is None:
        return dict(torch=torch.__version__, triton=None, device=None, reason="triton is not installed")
    if torch.xpu.is_available():
        return dict(torch=torch.__version__, triton=triton.__version__, device="xpu",
                    name=torch.xpu.get_device_name(0), reason=None)
    if torch.cuda.is_available():
        return dict(torch=torch.__version__, triton=triton.__version__, device="cuda",
                    name=torch.cuda.get_device_name(0), reason=None)
    return dict(torch=torch.__version__, triton=triton.__version__, device=None,
                reason="no XPU or CUDA device: this kernel needs one, and Triton has no CPU backend in this install")


def _words(w_packed, torch):
    """uint8 (N, K // 4) or int32 (N, K // 16) -> the int32 view the kernel reads."""
    if w_packed.dim() != 2:
        raise ValueError("w_packed is (N, K // 4) as bytes or (N, K // 16) as words")
    if w_packed.dtype in (torch.uint8, torch.int8):
        if w_packed.shape[1] % 4:
            raise ValueError("K must be a multiple of 16 to load weight words, got %d packed bytes per row"
                             % w_packed.shape[1])
        return w_packed.contiguous().view(torch.int32)
    if w_packed.dtype in (torch.int32, torch.uint32):
        return w_packed
    raise TypeError("w_packed must be uint8/int8 bytes or int32 words, got %s" % w_packed.dtype)


def gemm(a, w_packed, scale_a=None, scale_w=None, out=None, block=None):
    """C = (A @ W^T) * scale_a[:, None] * scale_w[None, :] on the device `a` lives on.

    `w_packed` is uint8 (N, K // 4) or int32 (N, K // 16); `a` is (M, K) fp16/fp32; `scale_a` (M,) and `scale_w` (N,)
    are optional fp32. Raises RuntimeError with the reason when the machine has no backend - use
    `reference.gemm_packed` there, which computes the same result on the host.
    """
    env = environment()
    if env["reason"]:
        raise RuntimeError("bitnet_gemm: %s. The numpy reference (reference.gemm_packed) computes the same result "
                           "on the host." % env["reason"])
    import torch
    if not isinstance(a, torch.Tensor) or not isinstance(w_packed, torch.Tensor):
        raise TypeError("bitnet_gemm: a and w_packed must be torch tensors")
    if a.dim() != 2:
        raise ValueError("a is (M, K)")
    words = _words(w_packed, torch)
    M, K = a.shape
    N = words.shape[0]
    if words.shape[1] * WORD != K:
        raise ValueError("w_packed covers %d K values per row for K = %d" % (words.shape[1] * WORD, K))
    if a.dtype not in (torch.float16, torch.float32):
        raise TypeError("a must be float16 or float32, got %s" % a.dtype)
    bm, bn, bk = block or DEFAULTS[env["device"]]
    if min(bm, bn, bk) < DOT_MIN or bk % WORD or K % bk:
        raise ValueError("block (%d, %d, %d): each dimension needs >= %d, BLOCK_K a multiple of %d and K a multiple "
                         "of BLOCK_K" % (bm, bn, bk, DOT_MIN, WORD))
    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    _ternary_gemm_kernel[grid](
        a, words, out,
        scale_a if scale_a is not None else a,
        scale_w if scale_w is not None else a,
        M, N, K,
        a.stride(0), a.stride(1),
        words.stride(0), words.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        HAS_SA=scale_a is not None, HAS_SW=scale_w is not None,
    )
    return out


def gemm_numpy(a, w_packed, scale_a=None, scale_w=None, block=None):
    """The same call on the host, so a caller can pick a path without branching on the device itself."""
    from .reference import gemm_packed
    return gemm_packed(a, w_packed, scale_a, scale_w)


def compile_report(target=("cuda", 80, 32), block=(32, 64, 64), scales=True):
    """Compile the kernel for `target` and report what came out - no device, no execution.

    This is the check that runs where the kernel cannot: it proves the frontend accepts the tile shapes and the reshape,
    that `tl.dot` became a matrix instruction, and at what width the weight stream is loaded. Returns a dict with the
    instruction counts plus `problems`, a list of things that would make the kernel wrong or slow.
    """
    if triton is None:
        raise RuntimeError("triton is not installed")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    sig = {"a_ptr": "*fp16", "w_ptr": "*i32", "c_ptr": "*fp16", "sa_ptr": "*fp32", "sw_ptr": "*fp32",
           "M": "i32", "N": "i32", "K": "i32",
           "stride_am": "i32", "stride_ak": "i32", "stride_wn": "i32", "stride_wk": "i32",
           "stride_cm": "i32", "stride_cn": "i32"}
    consts = {"BLOCK_M": block[0], "BLOCK_N": block[1], "BLOCK_K": block[2],
              "HAS_SA": bool(scales), "HAS_SW": bool(scales)}
    k = triton.compile(ASTSource(_ternary_gemm_kernel, sig, consts, None), target=GPUTarget(*target))
    ptx = k.asm["ptx"]
    counts = collections.Counter()
    for line in ptx.splitlines():
        line = line.strip()
        m = re.match(r"([a-z][a-z0-9]*(?:\.[a-z0-9:%-]+)*)\s", line)
        if m and not line.startswith(("//", ".", "$", "@%")):
            counts[m.group(1)] += 1
    mma = sum(n for op, n in counts.items() if op.startswith(("mma.sync", "wmma.")))
    loads = collections.Counter(re.findall(r"ld\.global\.[a-z0-9.]+", ptx))
    problems = []
    if not mma:
        problems.append("no matrix instruction in the PTX: tl.dot did not reach the tensor path")
    if loads.get("ld.global.b8") or loads.get("ld.global.u8"):
        problems.append("byte-wide weight loads: the weight stream is not being read as words")
    return dict(target=target, block=tuple(block), shared=k.metadata.shared, num_warps=k.metadata.num_warps,
                mma=mma, loads=dict(loads), stores=dict(collections.Counter(re.findall(r"st\.global\.[a-z0-9.]+", ptx))),
                instructions=dict(counts), ptx=ptx, problems=problems)
