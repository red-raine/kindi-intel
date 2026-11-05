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

Corrected, it is bit-exact against the integer reference (`max |LUT - exact| = 0.0`) and it runs at (median of
interleaved rounds, pinned to the physical cores - a minimum of the same rounds sits 30-40% below this, and that floor
is what this table used to quote):

| N x K | threads | median ms | CV | weights/s | packed stream |
|---|---:|---:|---:|---:|---:|
| 1024 x 1024 | 1 | 0.100 | 5.8% | 10.5 G | 2.63 GB/s |
| 2048 x 1920 | 1 | 0.406 | 8.8% | 9.7 G | 2.42 GB/s |
| 4096 x 4096 | 1 | 1.764 | 46.3% | 9.5 G | 2.38 GB/s |
| 1024 x 1024 | 8 | 0.243 | 24.8% | 4.3 G | 1.08 GB/s |
| 2048 x 1920 | 8 | 0.275 | 9.8% | 14.3 G | 3.58 GB/s |
| 4096 x 4096 | 8 | 1.024 | 28.0% | 16.4 G | 4.10 GB/s |

Against the pair table already in the engine, this harness compares a documented figure, so the number that counts is
from [libcompare.py](libcompare.py): both kernels in one process, one matrix, one activation vector, pinned and paired.
**The pair table wins**: 1.5x at 2048 x 1920 and **2.0-2.9x** at 14336 x 4096, at one thread as well as eight, with
45 paired rounds an arm and the direction reproduced in two runs. The kernels agree to 2.2e-07 of their largest output,
so that is speed and not correctness. The earlier "parity" reading came from comparing this harness's figure, taken as a
minimum of nine, against a documented one - and there is nothing here to pay for a repack.

The single-shuffle version the note proposes would be ~2x fewer ALU ops per group and would also be wrong, because int8
activations do not fit a single-shuffle int8 table. That argument is about the LUT family, not about which kernel to
ship: the pair table is ahead on both counts.

## The cache-blocked T-MAC: built, exact, and 1.55x faster at expert width

The next note adds tiling: an `N_blk` tile of outputs with the accumulators spilled to a per-tile buffer, a `K_blk`
tile so the tables fit L1, and `_mm_prefetch` on the weight stream. It is now the second entry point,
`tmac_gemv_avx2_tiled(a, w, scale_a, scale_w, n_blk, k_grp_blk)` in [tmac.cpp](tmac.cpp). It is **bit-exact** against the
integer reference (`max |tiled - exact| = 0.0` over four blockings), and its arithmetic is the unchanged one: the tables
are still built in `int8` and still wrap, so the same 4.5% of activation pairs are still wrong. Tiling does not touch
that bug.

Building it needs an MSVC environment (`vcvars64.bat`); `torch.utils.cpp_extension.load` fails before it compiles if `cl`
is not on `PATH`, which is the normal state of a shell here - `tmac.py` now calls `kindi.cbuild.find_vcvarsall` and
imports that environment, so the documented command runs from Git Bash.

`python -m tools.bitnet_gemm.tmac` measures it properly: pinned to the 8 physical cores, the arms interleaved inside
each round, and a **paired** ratio with a 95% bootstrap interval instead of a min-of-N. On this host, K = 4096, 15 rounds
x 3 runs:

| N | arm | median ms | speedup | 95% CI | per run |
|---|---|---:|---:|---|---|
| 14336 | untiled kernel | 5.15 | 1.00x | | |
| 14336 | 1024 x 512 (the note's tiles) | **3.30** | **1.55x** | 1.45-1.66 | 1.59 1.70 1.37 |
| 14336 | 512 x 512 | 3.82 | 1.37x | 1.29-1.46 | 1.44 1.49 1.19 |
| 2048 | untiled kernel | 0.63 | 1.00x | | |
| 2048 | 1024 x 512 | 0.65 | 0.91x | 0.84-0.98 | 0.93 0.91 0.89 |
| 2048 | 512 x 512 | **0.58** | **1.10x** | 1.03-1.18 | 1.12 1.08 1.11 |
| 2048 | 2048 x 1024 | 1.15 | 0.55x | 0.52-0.57 | 0.56 0.53 0.55 |

The note's tiles win in every run at the expert width and sit inside the noise floor at N = 2048 (0.91x). The 45% loss
in that table belongs to the single-tile arm - 2048 x 1024 at N = 2048 is one tile for eight threads, and seven of them
idle. `512 x 512` is the blocking that pays at both widths, so it is the better default. These milliseconds are not
comparable with the ranges this file used to quote: those were unpinned min-of-N, these are pinned medians paired
within a round. `docs/TERNARY-LUT-AND-PAGING-2026.md` has the full method and the retractions.

**A mechanism I had for why this would not work is withdrawn too**: that the kernel is not bandwidth-bound, so there is
nothing for blocking to recover. The stream does move only 2.9-4.4 GB/s against several times that in DRAM headroom, and
the limiter is on the compute path (`vpshufb`, int16 widening, a GEMV that reads each weight once) - but the tiles still
recover 35% of the time, so the compute path was losing something to layout and occupancy that the theory did not model.
What the two shapes do say is consistent with two facts that survive:

* **The premise that motivated the K tile is off by 4x once the table is correct.** A single-shuffle int8 table is
  32 bytes a group (the note's 32 KB at K = 4096); the two byte planes a correct table needs are 128 bytes a group, i.e.
  **131 KB at K = 4096** - so the note's own K_blk = 512 (16 KB) is safe by accident rather than by its arithmetic. At
  16 KB a slice is L1-resident, which is the one part of the argument that does hold up.
* **`N_blk` splits the work across threads, and that dominates.** One tile for 8 threads is 0.55x (seven of them idle),
  two tiles is 0.91x, and 14 tiles is 1.55x. So the note's tile width is right at expert width and wrong below it, and
  what decides is tiles-per-thread, not the tile shape - except that `512 x 512` (28 and 4 tiles) wins at both.

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
