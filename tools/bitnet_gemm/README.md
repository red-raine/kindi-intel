# BitNet b1.58 GEMM: 2-bit weights, unpacked on the fly

A GEMM whose weights are stored four to a byte (2 bits each) and unpacked in registers before the multiply, plus the
host reference that decides whether the kernel is right. The measurement record for this tool - the LUT and its
tiling, and what an expert read costs - is [docs/TERNARY-LUT-AND-PAGING-2026.md](../../docs/TERNARY-LUT-AND-PAGING-2026.md).
Nothing under `kindi/` imports this: it is a separate tool
because the engine's CPU path (`kindi/cpu.py`, `kindi/kcpu.c`) already has its own packing and its own GEMV, and this
measures a different design.

    python -m tools.bitnet_gemm.test_bitnet           # 9 checks, no GPU needed
    python -m tools.bitnet_gemm.test_bitnet --bench   # plus the host decode cost

| file | what it is |
|---|---|
| `packing.py` | the code: 0→0, 1→+1, 2→−1, 3 unused→0, four per byte in K order, and a 32-bit word view for loading |
| `reference.py` | numpy ground truth: the exact integer GEMM, the tile nest, and the kernel's unpack |
| `kernel.py` | the Triton kernel, the device gate, and `compile_report()` |
| `test_bitnet.py` | the checks and the host benchmark |
| `tmac.cpp`, `tmac.py` | the T-MAC LUT GEMV and its cache-blocked variant, built and measured (below) |
| `stream_probe.py` | what a cold expert read costs on this device, unbuffered - the ceiling on any paging pipeline |
| `copytest.py` | whether a read rate is the medium, the path or a cache: one file through `ReadFile` unbuffered, `ReadFile` buffered, `CopyFileExW` and 4 KB random, plus the read-during-commit curve that retired the 9 ms figure |

## What was verified, and how

The machine this was built on has **no XPU, no CUDA, no IPEX, and no Triton CPU backend** (`triton 3.6.0`, backends
`amd` and `nvidia`; `torch 2.11.0+cpu`). So the kernel is never executed here. What is checked instead:

1. **The arithmetic, in numpy.** `reference.gemm_tiled` is the kernel's loop nest on the host - same tile order, same
   shifts, same reshape - and it is asserted equal to the dense integer GEMM for several tile shapes, including ragged
   edges (37x53x128, 100x31x256). Integer accumulation means tiling cannot change the answer, so equality is exact, not
   approximate.
2. **The unpack, field by field.** `reference.tile_unpack_words` is the kernel's `(code & 1) - (code >> 1)` on a
   16-wide shift axis; it is asserted equal to the weights that were packed, including a word whose top byte is all
   ones (negative as int32) - the case where a sign-extending shift would go wrong if the mask were missing.
3. **The codegen, by compiling.** `kernel.compile_report()` builds the kernel through the whole Triton pipeline for a
   target with no device attached and reads the PTX. This is what proves `tl.dot` became a matrix instruction and at
   what width the weights are loaded:

   | BLOCK_M,N,K | `mma.sync` | weight loads | A loads | shared |
   |---|---:|---|---|---:|
   | 32, 64, 64 | 16 | 8 × `ld.global.b32` | 16 × `ld.global.b16` | 14336 B |
   | 64, 64, 128 | 64 | 12 × `ld.global.b32` | 64 × `ld.global.b16` | 36864 B |
   | 16, 32, 32 | 2 | 4 × `ld.global.b32` | 4 × `ld.global.b16` | 3584 B |

   `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` is the tensor path on sm_80 and the equivalent of XMX on an Xe
   iGPU, so the multiply is not falling back to FMA. Shared memory is 3.5-36 KB, far below any 512 KB SLM budget, so
   these tiles are not limited by it.
4. **The gate, on the machines that cannot run it.** `gemm()` raises with the reason rather than a driver error, and a
   test asserts the message - so a caller on a CPU box gets told to use `reference.gemm_packed` instead of a stack
   trace from a launcher.

**Not verified: the numbers on a device.** No XPU and no CUDA device exist here, so the fp16 accumulate path, the
`mma` result and the epilogue's `acc.to(tl.float16)` truncation have never been executed. The first real run has to
compare `gemm()` against `reference.gemm_packed()` on the Arc (tolerance a few 1e-2 absolute at K = 1024 in fp16); the
test for that is already written and skips here.

## Four corrections to the draft this started from

* **`tl.tensor_scatter` does not exist.** Triton's frontend is SSA: a tile is not written into, it is rebuilt. That line
  is where the draft's unpack stops - `AttributeError: module 'triton.language' has no attribute 'tensor_scatter'`,
  reproduced by compiling the draft body. The replacement here shifts on a third axis and collapses it:
  `(N, K//4, 4) --reshape--> (N, K)`, which is C-order and needs no scatter, no permute and no second pass. `tl.interleave`
  also exists and would work (three nested calls); reshape is one op and was verified against numpy here.
* **The power-of-two `tl.where` chain is unnecessary.** `0 → 0, 1 → +1, 2 → −1` is `(code & 1) - (code >> 1)`: two
  integer ops, no selects, and it handles code 3 as 0 without a branch. That is what the PTX shows (`bfe`, `and`, `sub`).
* **Byte-granular weight loads give up the bandwidth win.** With `uint8` weights the compiler emitted 8 ×
  `ld.global.b8` per thread per K step - four times the load instructions for the same bytes, and no 32-bit transaction.
  Loading the stream as `int32` words (16 weights per word, shifts 0, 2, …, 30) turns that into `ld.global.b32`; the
  test now fails if a byte-wide weight load ever comes back.
* **`pack_ternary_weights` returns `int8` while the kernel declares `uint8`.** The code 2 in the top field is `0xC0`,
  which is negative as `int8`; it survives `& 3`, but every stride, view and comparison downstream then means something
  different. The packer here stays `uint8`, and the kernel reads signed words where the mask makes the sign irrelevant.

## The CPU half: on the target machine, the unpack is not free

This box is Gen9-class Intel (family 6, model 158), AVX2, **no AVX-512, no VNNI, no AMX** - the i9 the branch is about.
Per-weight unpack costs there, measured by timing the same fp32 BLAS GEMM with and without the unpack in front of it
(numpy 2.4.4, 16 threads; `--bench`):

| MxNxK | shift decode | table decode (256-entry) | memcpy floor | GEMM | table / GEMM |
|---|---:|---:|---:|---:|---:|
| 256x1024x1024 | 22.31 ms | 5.72 ms | 0.97 ms | 4.27 ms | 1.34x |
| 1024x1024x1024 | 22.43 ms | 5.75 ms | 1.14 ms | 7.33 ms | 0.78x |
| 2048x2048x2048 | 88.63 ms | 24.36 ms | 4.14 ms | 47.81 ms | 0.51x |

Two things fall out. **A byte table beats the shifts by 3.6-3.9x on the host**, which is why `decode_lut` exists and is
what the verification path uses; and **expanding 2-bit weights to fp32 and then calling a float GEMM costs half a GEMM
again at the sizes that matter** (24.4 ms of unpack against 47.8 ms of multiply at 2048³). On a chip with no VNNI and
no AMX, unpack-on-the-fly is a tax you pay every weight, every token. That is the opposite of the GPU case, where the
unpack is a few integer ops against an MMA that runs 16 MACs per instruction per lane.

So the shape this tool measures is the right one for the Arc and the wrong one for the i9. The i9's answer is already
in the engine and does not expand anything: `kindi/kcpu.c` keeps the pack's own 4-trit codes and reads them through a
pair table with `vpermps`, one permute per 8 rows × 2 columns. The element-wise LUT family was measured against that
table in [docs/CPU-PERF-2026.md](../../docs/CPU-PERF-2026.md) and lost by **3.4-4.0x** (an 81-entry `vpgatherdd` gather,
0.59-0.72 ms against the pair table's 0.18 ms on Coffee Lake) - the gather, not the table, is what costs. Note the two
results are not in conflict: the numpy table decode here is one gather over bytes against four int32 passes with 16 MB
temporaries, while the C comparison is a gather against a single `vpermps` that needs no table lookup per weight.

## The T-MAC LUT, built: exact, and no faster than what is already here

The design note's other kernel replaces the multiply with `vpshufb` table lookups: precompute the 16 partial sums of two
activations, fetch them by weight code. The idea is sound and it is now built ([tmac.cpp](tmac.cpp),
toolchain: MSVC 19.4x + ninja, AVX2, OpenMP) - but not as written, for two reasons that matter:

**The table cannot be `int8`.** A two-weight table holds `v0 + v1` where each term is up to 127 in magnitude, so entries
reach +-254 and `static_cast<int8_t>` wraps. Measured on realistic int8-quantised activations (scaled so the max is
127): **4.5% of activation pairs produce a table with at least one wrong entry**, and the wrap is 256 - so the note's own
`assert max_diff < 1e-3` cannot pass. The fix is int16 tables held as two byte planes, which is two shuffles per table
instead of one. That doubling is exactly the margin the LUT was supposed to win by.

**The build flags are GCC-only.** `-mavx2 -mfma -fopenmp -march=native` cannot build on the MSVC toolchain this
repository ships for Windows (`/arch:AVX2`, `/openmp`). `tmac.py` picks the flags per platform.

Corrected, it is bit-exact against the integer reference (`max |LUT - exact| = 0.0`) and it runs at:

| N x K | threads | ms | weights/s | packed stream |
|---|---:|---:|---:|---:|
| 1024 x 1024 | 1 | 0.119 | 8.8 G | 2.20 GB/s |
| 2048 x 1920 | 1 | 0.422 | 9.3 G | 2.33 GB/s |
| 4096 x 4096 | 8 | 0.648 | 25.9 G | 6.47 GB/s |
| 2048 x 1920 | 8 | **0.189** | 20.8 G | 5.20 GB/s |

Against the pair table already in the engine (`docs/CPU-PERF-2026.md`: **0.18 ms** for 2048 x 1920 ternary weights,
single- and 8-threaded alike): **0.189 ms**. That is parity, not a win - and a cross-source one, since that figure came
from another machine instance of the same Coffee Lake class. The honest reading is that the LUT and the pair table cost
the same here, so there is no reason to switch: the note's single-shuffle version would be ~2x fewer ALU ops per group and
would also be wrong, because int8 activations do not fit a single-shuffle int8 table.

A like-for-like number needs `libkcpu` built here and the same matrix run through `rows8`; this harness compares against a
documented figure, not a measurement taken in the same process.

## The cache-blocked T-MAC: built, exact, and it earns nothing

The next note adds tiling: an `N_blk` tile of outputs with the accumulators spilled to a per-tile buffer, a `K_blk`
tile so the tables fit L1, and `_mm_prefetch` on the weight stream. It is now the second entry point,
`tmac_gemv_avx2_tiled(a, w, scale_a, scale_w, n_blk, k_grp_blk)` in [tmac.cpp](tmac.cpp). It is **bit-exact** against the
integer reference (`max |tiled - exact| = 0.0` over four blockings), and its arithmetic is the unchanged one: the tables
are still built in `int8` and still wrap, so the same 4.5% of activation pairs are still wrong. Tiling does not touch
that bug.

Building it needs an MSVC environment (`vcvars64.bat`); `torch.utils.cpp_extension.load` fails before it compiles if `cl`
is not on `PATH`, which is the normal state of a shell here.

Same process, 8 threads, K = 4096, min of 15 runs per arm, 3 runs of the whole table:

| N | arm | ms (3 runs) | vs untiled |
|---|---|---:|---:|
| 14336 | untiled kernel | 1.38, 1.70, 2.12, 2.14, 2.28, 2.59 | 1.00x |
| 14336 | 1024 x 512 (the note's tiles) | 1.35, 1.52, 1.70, 1.80, 1.96, 1.96 | median ~1.13x |
| 14336 | 1024 x all K | 1.69, 2.08, 2.36, 2.64 | 1.00x |
| 2048 | untiled kernel | 0.34, 0.34, 0.34, 0.34, 0.39, 0.40 | 1.00x |
| 2048 | 1024 x 512 | 0.46, 0.47, 0.50, 0.51, 0.59 | **0.58-0.75x** |

The wide shape is the honest answer: **the tiled arm's best run and the untiled arm's worst run overlap**, so on this
machine the tiling cannot be separated from noise at N = 14336 (the per-call working set is 14.7 MB, which makes the
baseline itself swing 1.38-2.59 ms; the 2048 shape fits and is stable to +-15%). The narrow shape is unambiguous: tiling
**loses 25-42%**.

Three things explain it, and they matter more than the numbers:

* **The kernel is not bandwidth-bound, so there is nothing for blocking to recover.** At 25-27 G weights/s the packed
  stream moves 6.2-6.9 GB/s - this i9 streams several times that. The limiter is `vpshufb` and widening throughput per
  group of 32 neurons, so reordering the stream changes nothing. A GEMV reads each weight exactly once: there is no
  reuse for a tile to create, which is why the same blocking that is standard for a batched GEMM buys nothing here.
* **The premise that motivated the K tile is off by 4x once the table is correct.** A single-shuffle int8 table is
  32 bytes a group (the note's 32 KB at K = 4096); the two byte planes a correct table needs are 128 bytes a group, i.e.
  **131 KB at K = 4096**. That still fits this core's 256 KB L2 comfortably, which is why nothing stalls - and why the
  note's own K_blk = 512 (16 KB) is safe by accident rather than by its arithmetic.
* **`N_blk` is a parallelism knob, not a cache knob.** The tiled arm parallelises over N tiles, so at N = 2048 the note's
  1024-wide tile gives 2 tiles for 8 threads and half the machine idles - exactly the 0.6-0.75x measured.

Two concrete defects in that note besides the arithmetic: `_mm_prefetch(reinterpret_cast<const char*>(w_grp + 4 * N),
_MM_HINT_T0)` is `w_grp + 4 * N` **bytes**, and a group is N bytes, so it prefetches 4 groups - 57 KB, or **896 cache
lines**, at N = 14336 - while its comment says "4 lines ahead", and 57 KB cannot stay in a 32 KB L1. And the tiled loop
reads `local_acc[local_idx + 24 .. +31]` unconditionally, so it needs `N % 32 == 0`; the note checks neither `N % 32`
nor the vector's end, so an N not divisible by 32 reads past the buffer.

### The NVMe pipeline in the same note does not run

`StreamedMoEExpertLoader` is pseudocode that would fail if executed: `torch.from_file(..., size=...)` with a literal
ellipsis, no `dtype` (default `float32` for a stream of 2-bit codes), and `from_file` memory-maps - so "prefetch" is not a
read and the faults land inside the kernel, on the thread you wanted to keep computing. A thread per step against a
`Queue(maxsize=2)` blocks in `put` under the GIL, `top-2` routing returns its experts in gate order rather than request
order so `assert cur_id == active_expert_id` fails whenever the next token flips them, and overlapping one
sub-millisecond GEMV cannot hide a 14.7 MB NVMe read - the prefetch has to be issued during the *previous token's whole
decode*, into a double buffer sized for the union of the next token's top-k experts, with the pack streamed as packed.

## Paging experts from disk: 3.3-4.0 ms a read, and the 9 ms number was a read racing a write

The next note streams cold experts with `io_uring` + `O_DIRECT` + `mlock`, double-buffered against the GEMV. That is a
Linux mechanism and this box is Windows, so it is not built here; but the number it rests on is measurable, and it took
two probes and three wrong answers to measure. `stream_probe.py` reads with `FILE_FLAG_NO_BUFFERING` (the POSIX arm uses
`O_DIRECT`), and the flag is enforced - an unaligned `ReadFile` under it fails with `ERROR_INVALID_PARAMETER` (87),
checked rather than assumed. `copytest.py` is the second opinion: one file, the same bytes through `ReadFile` unbuffered,
`ReadFile` buffered, `CopyFileExW`, and 4 KB random, so a rate can be attributed to the medium, the path or the cache.

| arm | C: | B: | E: (USB spindle) |
|---|---:|---:|---:|
| 14.7 MB unbuffered, whole file | 3.10-3.47 GB/s (3.3-4.0 ms) | 2.64 | 0.14 |
| buffered `ReadFile`, cold / warm | 2.80 / 3.29 | 3.29 / 2.95 | 0.12 / 3.88-4.03 |
| `CopyFileExW`, one-way share | 1.95 (3.89 both ways) | 1.48 (2.97) | 0.11 |
| 4 KB random | 3.9-5.7 k IOPS | 24.8 k IOPS | 110 IOPS |
| buffered write: into cache, then commit | 1.84, then 0.068 GB/s | - | - |

**The 1.6 GB/s and the ~9 ms read came from reading a file the probe had just written.** A drive with a write-back cache
takes writes at 1.84 GB/s and commits them at 68 MB/s, so 2 GB takes 30 s to reach the medium - and for those 30 s,
reads of the file being committed run at 0.14 GB/s. Measured on a timer: 3.80 GB/s at 0.1 s, **0.14 GB/s from 4 to 30 s**,
3.68 GB/s at 37 s. With the drive idle the same 14.7 MB reads run at 3.1-3.5 GB/s.

Two corrections came with that. The B: arm is **not** a spinning disk - 24,820 random 4 KB IOPS over 33.9 GB against the
USB spindle's 110 - so its 2.6-3.3 GB/s are its own, and the "a 4 TB HDD cannot do 3 GB/s" argument that retired the old
table was sound in form and wrong about the hardware. The control that does the work is E:, 0.12-0.14 GB/s cold against
3.9-4.0 warm. And a **copy cannot settle a read rate**: it is read *and* write on one medium, so its one-way share is
bounded by the slower side and is a floor on the device, never a ceiling on reads. Measured copies ran 0.37-2.32 GB/s on
C: in one afternoon, tracking commit state rather than the drive.

The arithmetic that decides when streaming is worth building: the kernel digests 58.7 M weights in 1.8 ms = **32.6 G
weights/s**, while 3.7 ms of read is 15.9 G weights/s. Compute still outruns the read, by 2x instead of 5x, so at batch 1
the read still cannot be hidden and the fix is still **more tokens per weight read** - but it takes 2 tokens rather than
5-6 to make one expert's read free, and a pipeline that hides one read behind one multiply has a 33% ceiling rather than
17%. In a *running* pipeline the read costs 19-25 ms, not 3.7, because the reader shares bandwidth with the GEMV: that is
what `batch_stream.py` measures, and it is why streaming is still a batched design whose batch-8 arm loses to residency.
At batch 1 the expert set has to fit in RAM.

As code, the note needs five things before it runs correctly: (1) `wait_completion` returns whichever CQE arrives and
the loop pairs it with `cur_slot`, which only holds while exactly one read is outstanding - set the constructor's
`queue_depth=4` and `scale_w_dict[completed_tag]` gets the wrong scales for the wrong weights, silently; (2) the
destructor calls `io_uring_unregister_buffers` and `free` with requests possibly still in flight, so the DMA can write
into freed pages at shutdown; (3) `get_tensor_view` returns a non-owning tensor over a buffer the pipeline immediately
reuses, so any caller that keeps `w_packed` past the swap - the obvious "same expert as last token" optimization - reads
being-overwritten bytes; (4) `expert_size` offsets are computed from an unpadded width, so `exp * expert_size` is not
4096-aligned unless `K/4 * N` happens to be (14336 x 4096 is; N = 11008 is not) and `submit_read` throws; (5) `mlock` on
top of `io_uring_register_buffers` pins the same pages twice against `RLIMIT_MEMLOCK` for no benefit - registration
already pins them.

## Calling it

```python
from tools.bitnet_gemm import kernel, packing, reference

w = weights                                             # (N, K) int8 in {-1, 0, 1}
packed = packing.encode(w)                              # (N, K // 4) uint8, 0.25 bytes a weight
words = packing.as_words(packed)                        # (N, K // 16) uint32 - what the kernel loads

c = kernel.gemm(a, words, scale_a, scale_w)             # on xpu/cuda; raises with the reason elsewhere
c = reference.gemm_packed(a, packed, scale_a, scale_w)  # the same on the host, fp32
```

`K` must be a multiple of `BLOCK_K`, and every block dimension at least 16 (`tl.dot`'s minimum). `DEFAULTS` carries a
tile per device (`xpu` 32x64x64, `cuda` 64x64x64, `cpu` 16x32x32); the `cpu` entry is a placeholder for a Triton build
that has a CPU backend, which this install does not, and on this evidence the CPU should not use this kernel shape at
all.
