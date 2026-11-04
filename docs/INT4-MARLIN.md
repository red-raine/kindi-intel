# INT4 tensor-core products: is Marlin usable in the engine?

[Marlin](https://github.com/IST-DASLab/marlin) is Elias Frantar's FP16xINT4 matmul kernel, and it is the strongest
published answer to the problem the Gemma 4 prompt path has on a 4-bit pack: keep the weights 4-bit all the way to the
tensor core instead of expanding them first. This page is what is left of that idea after reading the source against
`kindi/gemma4_tc.cu`. No speed number appears here: the machine this was written on has no NVIDIA card and no `nvcc`
(`nvidia-smi`, `nvcc` and `cupy` are all absent, and the installed `torch` is CPU-only), so nothing below is a
measurement. It is a compatibility result, and a test plan.

**Verdict.** Marlin cannot be used as a library here (blockers 1 and 2 below). One thing inside it - weights reshuffled
offline into fragment order, so the tensor-core operand is formed by shifting a word rather than by expanding digits
into shared memory - applies directly to the two types this engine has no tensor-core prompt GEMM for, and is worth
porting. Blockers 3 and 4 are why that port is a new kernel rather than a build flag.

## What Marlin requires, from its source

| requirement | where | value |
|---|---|---|
| licence | `LICENSE` | Apache 2.0, so vendoring a derived kernel is allowed |
| build | `setup.py` | a torch CUDA extension (`cpp_extension.CUDAExtension`) over a host `.cpp` and the kernel `.cu`; `install_requires=["numpy","torch"]` |
| toolchain | `README.md` | CUDA >= 11.8 with a matching `nvcc`, `torch >= 2.0`, compute capability >= 8.0 (Ampere/Ada; "not yet optimized for Hopper") |
| weights | `marlin/__init__.py` | symmetric int4 only: packed as `round(w/s) + 8`, clamped to `0..15`; `B` is `(k/16, n*16/8)` ints, which is exactly `k*n/2` bytes |
| groupsize | `Layer.__init__` | `-1` or `128` only; `in_features % 128 == 0`, `out_features % 256 == 0` |
| activations | `mul()` | fp16 in, fp16 out, fp32 accumulate |
| kernel | `marlin_cuda_kernel.cu` | `mma.sync m16n8k16 .f32.f16.f16.f32`, `ldmatrix.x4`, `cp.async.cg` with an `L2::evict_first` policy for the weights, `lop3` dequant straight into B fragments, several warps per output tile, striped SM partition with a global workspace reduction |

Two consequences of that table. First, the kernel file really is self-contained device code - the torch coupling is in
the host wrapper and the Python - so the device part is portable in principle. Second, the *format* is narrow: no
zero-points, no per-32 sub-block scales, no minima. Marlin's whole B-side path is `dequant(int q)` turning eight
nibbles into four `half2` fragments with a magic-number `lop3` trick, and there is no slot in it for a minimum.

## What the engine requires

The Gemma 4 kernels are compiled at load time by NVRTC through CuPy (`kindi/gemma4.py`, `_ptx_module`), from
`gemma4.cu` with `gemma4_tc.cu` appended, with the PTX version and target patched to `sm_80` because the image's NVRTC
is CUDA 10.2. Kernels are `extern "C"` and take raw pointers, ints and floats; nothing in the package imports torch at
runtime, and there is no nvcc anywhere in that path.

The prompt GEMM (`g4_tcmm`, `gemma4_tc.cu:90`) is int8 `mma.sync m16n8k32` (or `m16n8k16`, or a paired k64 form), 64
rows by 64 tokens per block, and a token's activations are quantised per 128-column group so that the 128-term dot
product against the weights is an *exact integer*, scaled afterwards by `x_scale * w_scale`. For the asymmetric
K-quants the minimum is folded in by subtracting `scale * min * sum(x)` (`XSUM`). That exactness is what lets a GPU
product be compared bit-for-bit against the CPU path, which is why the sanitising question for any new kernel here is
not "how fast" but "does the integer sum match".

The types that get the tensor-core GEMM are `8, 10, 11, 13, 14, 23, 123, 111, 114, 108` (`gemma4.py:136` in
`qkv_batch`, `gemma4.py:162` in `linear_batch`). **`12` (Q4_K) and `2` (Q4_0) are absent from both lists.** They fall
through to `k_g4_quant_mm4` - the dp4a int8 GEMM - so the most widely shipped 4-bit GGUF formats are the only ones
that do not reach a matrix core on the prompt path.

## Blockers

1. **Build path.** Marlin is a torch extension built by nvcc against ATen. This engine has no torch and no nvcc at
   runtime; it hands NVRTC a single source string and gets `extern "C"` kernels back. Adopting it means extracting the
   device template, dropping the host dispatch and `<iostream>`, and writing `extern "C"` instantiations per type and
   tile shape - the way `G4_TCMM` already instantiates the current kernel. That is a port, not a dependency.
2. **Numerics.** fp16 activations and fp16 fragment dequant are the two things the int8 scheme exists to avoid: fp16
   digits would give up the exact integer dot, and Marlin has no zero-point at all, so Q4_K's 6-bit minima would need
   the `XSUM` correction added anyway. On the cards this engine targets the mma type is not the reason to look at
   Marlin either: on GeForce Ampere and Ada, int8 tensor-core throughput is twice fp16 (and the RTX 30 cards are
   rate-limited to half rate in 16-bit, which is why the installer offers `gpu-unlock`), so Marlin's "4x over fp16" is
   measured against a baseline this path does not use - on an A10, a datacenter card at full fp16 rate.
3. **Formats.** Symmetric int4 with groups of 128 or all of k. The engine's packs are GGUF K-quants: 256-value
   super-blocks, per-32 sub-block 6-bit scales, and a minimum per sub-block. The kernel would have to grow the scale
   and min handling that `g4_tcmm` already has, at which point only the fragment layout is inherited.
4. **Coverage.** sm_80+ CUDA only - the same ceiling `tcmm` already has - so the port buys nothing on Kepler, Maxwell,
   Pascal or ROCm, and those paths would keep the dp4a GEMM.

## What is worth taking

The staging cost, which is the part of Marlin that applies unchanged to an int8 scheme:

- `g4_tcmm` stages weights as int8 in shared memory: `W[64][17]` uint4 is 17,408 bytes per 256-column band, holding
  64 rows x 256 columns = 16,384 bytes of digits. The same tile stored as Q4_K blocks (144 bytes per 256 weights,
  `_QUANTS` in `gemma4.py`) is 9,216 bytes, and as Q2_K (84 bytes) it is 5,376 - so the staging is 1.8x the packed
  size for Q4_K and 3.0x for Q2_K, and the nibble-to-int8 unpack is re-run by every token tile that touches the same
  rows (`ceil(nt/64)` times per prompt).
- Marlin stores the weights at exactly `k*n/2` bytes and forms the fragment from a shifted word in registers, so
  there is nothing expanded to stage and nothing to re-unpack. For int8 digits the same trick is cheaper than for
  fp16: eight nibbles become two registers, and the exact-integer dot and the per-run scale/min correction stay as
  they are. Note the ceiling here is 1.8x, not Marlin's 4x: a Q4_K pack carries 4.5 bits a weight (scales and
  minima included) where Marlin's format carries 4, so most of the ratio is already spent before the kernel starts.

That is the port: a repack kernel that lays Q4_K (and Q4_0) weights out in fragment order once at load - the pattern
`k_g4_iq4_repack` and `k_g4_q8_repack` already follow - and a `g4_tcmm` variant for types 2 and 12 that consumes it.
The permuted fragment order is the part that needs care; it is also the part that is checkable on paper, since the
permutation must be an involution and the digits must come out in the same order the mma expects.

Note what would still be missing after that port: `g4_tcmm` already double-buffers with `cp.async` (`tc.cu:167`,
`tca.cu:32`) but with the default L1-cached policy, while Marlin marks the weight stream `L2::evict_first` precisely
because those bytes are read once and should not displace the activations. That is a one-line change to the copies in
a kernel that already exists, and it is a separate experiment from the repack.

## The test plan, when there is a card

Nothing in this note has been run on hardware. In order, on an sm_80+ machine with a matching nvcc:

1. **The repack.** Round-trip Q4_K and Q4_0 rows through the fragment-order layout and back; the digits must return
   unchanged, and the integer dot of a repacked tile against a quantised activation row must equal the expanded-int8
   result exactly.
2. **The gate that already exists.** `_tcmm_ok` (`gemma4.py:198`) compares a candidate GEMM against the dp4a GEMM over
   a full tile, including the check that no row is left unwritten. Run it for types 2 and 12 exactly as it runs for
   Q8_0, and keep the type out of the list until it passes.
3. **Then the speed.** The RTX 3080 profile in [CARDS.md](CARDS.md) - prompt reading 1,093 tokens/s at 512 and 855 at
   22K - is the number to beat, at both lengths, because the staging saving grows with the token tile count and the
   L2 policy only matters once the prompt is long enough to stream weights out of cache.

Until 1 and 2 are green, the honest statement is the one at the top: the library does not fit, the layout idea does,
and its value is an unmeasured 1.8x (Q4_K) to 3.0x (Q2_K) reduction in the weight staging of two formats that
currently miss the matrix cores entirely.
