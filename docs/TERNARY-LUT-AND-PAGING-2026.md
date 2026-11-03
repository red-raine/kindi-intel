# Ternary LUT kernels and expert paging on this host

Two things are measured here, both about the same question: on an i9 with AVX2 and no VNNI or AMX, what does a
ternary weight actually cost, first when the *activation* is quantised to int8 and looked up through a table, and
second when the weight has to arrive from disk before it can be multiplied. Both were proposed to this branch as
ready-to-use kernels; neither was usable as written, and the numbers below are what settled each one.

**Two claims I made earlier in this branch are wrong, and both are flagged where they appear**: that the cache-blocked
LUT beat the untiled kernel by "1.58x", and that 1.67 GB/s was "the device". Neither survives a second measurement. The
corrections are collected in [Two corrections](#two-corrections) rather than folded quietly into the text.

The harness is [tools/bitnet_gemm](../tools/bitnet_gemm/README.md): `tmac.cpp` (the kernels), `tmac.py` (build,
validation, timings), `stream_probe.py` (the disk). Nothing under `kindi/` imports it; this is a measurement tool, not a
shipped path.

Host: Intel i9-9980HK (Coffee Lake, family 6 model 158), AVX2 + FMA3 + F16C, **no AVX-512, no VNNI, no AMX**, 8 cores /
16 threads, 64 GB RAM, Windows under Git Bash. One other caveat applies to everything in the second half: this host's
block devices are virtualised (see [The disk claims](#the-disk-claims-are-the-least-trustworthy-numbers-here)).

## The activation code: int8 in, int16 table out

The proposal was the T-MAC shape: precompute the sixteen partial sums of two int8 activation codes, fetch them by weight
code with `vpshufb`, and skip the multiply entirely. The arithmetic does not fit in a byte. A two-weight table holds
`v0 + v1` where each term is up to 127 in magnitude, so entries reach ±254 and `static_cast<int8_t>` wraps by 256 -
**4.5% of activation pairs, on int8-quantised activations scaled so the max is 127, produce a table with at least one
wrong entry**, and the proposal's own `assert max_diff < 1e-3` cannot pass.

Held as int16 and split into two byte planes, so one `vpshufb` still fetches each plane, the kernel is **bit-exact**
against the integer reference: `max |LUT - exact| = 0.0`, not a tolerance. What that costs is a second shuffle per
table, which is precisely the margin the single-shuffle version was going to win by.

```bash
# Windows: needs an MSVC environment, or torch's builder fails before it compiles
cmd //c 'call vcvars64.bat && python -m tools.bitnet_gemm.tmac --rebuild'
python -m tools.bitnet_gemm.tmac            # once built
```

Weights decoded from 2-bit codes inside the kernel, 0.25 bytes a weight, best of 9:

| N x K | threads | ms | weights/s | packed stream |
|---|---:|---:|---:|---:|
| 1024 x 1024 | 1 | 0.115-0.125 | 8.4-9.1 G | 2.1-2.3 GB/s |
| 2048 x 1920 | 1 | 0.379-0.437 | 9.0-10.4 G | 2.2-2.6 GB/s |
| 4096 x 4096 | 1 | 1.57-2.22 | 7.6-10.6 G | 1.9-2.7 GB/s |
| 1024 x 1024 | 8 | 0.066-0.068 | 15.5-15.8 G | 3.9 GB/s |
| 2048 x 1920 | 8 | 0.170-0.190 | 20.8-23.0 G | 5.2-5.8 GB/s |
| 4096 x 4096 | 8 | 0.61-0.68 | 25.1-27.3 G | 6.2-6.9 GB/s |
| 14336 x 4096 | 8 | 1.38-2.59 | 23-43 G | 5.7-10.6 GB/s |

The ranges are between runs, not within one: each figure is already a best-of-9 inside its process. **Even the
cache-resident rows move by up to 40%** - the 4096 x 4096 single-thread case measured 1.57-1.58 ms in three runs and
2.22 ms in a fourth, on this machine, with nothing changed. That is why the tiling section below reports a spread and
refuses to rank arms that overlap, and it is the reason a one-sample comparison was able to produce a 1.58x that did not
exist.

**Against the pair table already in the engine** ([CPU-PERF-2026.md](CPU-PERF-2026.md) documents 0.18 ms for the same
2048 x 1920 ternary GEMV, single- and 8-threaded alike): the LUT does **0.17-0.19 ms**. Parity, not a win - and a
cross-source comparison, since that figure came from another machine of the same Coffee Lake class, not from this
process. The honest reading is that the LUT family and the engine's `vpermps` pair table cost the same here, so there is
no reason to switch: the cheaper single-shuffle version would be ~2x fewer ALU ops per group and would also be wrong.

## The cache-blocked variant: correct, and it earns nothing

The follow-up proposal added `N_blk x K_blk` blocking, accumulators spilled to a per-tile buffer, and `_mm_prefetch` on
the weight stream. It is implemented as `tmac_gemv_avx2_tiled(a, w, scale_a, scale_w, n_blk, k_grp_blk)` and it is
bit-exact over four blockings (`max |tiled - exact| = 0.0`), with the same wrapped int8 table as above - blocking does
not touch that bug.

```bash
python -m tools.bitnet_gemm.tmac       # prints the tiling table after validating both arms
```

K = 4096, 8 threads, min of 15 runs per arm, three runs of the whole table:

| N | arm | ms across runs | verdict |
|---|---|---:|---|
| 14336 | untiled kernel | 1.38, 1.70, 2.12, 2.14, 2.28, 2.59 | baseline |
| 14336 | 1024 x 512 (the proposal's tiles) | 1.35, 1.52, 1.70, 1.80, 1.96, 1.96 | not separable from noise |
| 14336 | 1024 x all K | 1.69, 2.08, 2.36, 2.64 | ~same as untiled |
| 2048 | untiled kernel | 0.34, 0.34, 0.34, 0.34, 0.39, 0.40 | baseline |
| 2048 | 1024 x 512 | 0.46, 0.47, 0.50, 0.51, 0.59 | **loses 25-42%** |

**Corrected: my first run reported 1.58x for the tiled arm.** That was one sample against one sample on a machine whose
untiled baseline moves 1.38-2.59 ms run to run. With the baseline measured under the same min-of-three as the other
arms, the wide-shape difference disappears into the noise and the narrow shape is a clear loss.

Three mechanisms explain it, and they matter more than the numbers:

- **The kernel is not bandwidth-bound, so blocking has nothing to recover.** At 25-27 G weights/s the packed stream
  moves only 6.2-6.9 GB/s. The limiter is `vpshufb` and int16 widening per group of 32 neurons. A GEMV reads each weight
  exactly once - there is no reuse for a tile to create, which is why the blocking that is standard for a batched GEMM
  buys nothing here.
- **The premise behind the K tile is off by 4x once the table is correct.** A single-shuffle int8 table is 32 bytes a
  group (the proposal's 32 KB at K = 4096); the two byte planes a *correct* table needs are 128 bytes a group, i.e.
  **131 KB at K = 4096**. That still fits this core's 256 KB L2, which is why nothing stalls - and the proposal's
  K_blk = 512 (16 KB) is safe by accident rather than by its own arithmetic.
- **`N_blk` is a parallelism knob, not a cache knob.** The tiled arm parallelises over N tiles, so at N = 2048 a 1024-wide
  tile gives two tiles for eight threads and half the machine idles - the 0.58-0.75x measured.

Where the blocking does show a consistent direction is the **K tile alone**, at fixed tile count: 1024 x 512 against
1024 x all K is 1.52-1.96 ms against 1.69-2.64 ms, better in every run where both were measured. That is a ~10% lever at
wide N, and it is not worth a repack on its own.

Two defects in the proposal beyond the arithmetic, both cheap to check and both real: `_mm_prefetch(w_grp + 4 * N)` is
in **bytes** where a group is N bytes, so it prefetches 4 groups - 57 KB, or **896 cache lines**, at N = 14336 - while
its comment says "4 lines ahead", and 57 KB cannot survive a 32 KB L1. And the accumulator loop reads
`local_acc[local_idx + 24 .. +31]` unconditionally, so it silently requires `N % 32 == 0`; the proposal checks neither.

## The disk: an expert read costs ~9 ms, against 1.8 ms of GEMV

The third proposal streamed cold expert weights with `io_uring` + `O_DIRECT` + `mlock`, double-buffered against the
GEMV. `io_uring` does not exist on this host, so the mechanism is not built here; what is measurable is the number it
rests on. `stream_probe.py` reads with `FILE_FLAG_NO_BUFFERING` (the POSIX arm uses `O_DIRECT`), and the flag is
**enforced, checked rather than assumed**: an unaligned `ReadFile` under it fails with `ERROR_INVALID_PARAMETER` (87).

```bash
python -m tools.bitnet_gemm.stream_probe                 # C:, 1 GB file, 256 MB window, 3 passes
python -m tools.bitnet_gemm.stream_probe --threads 4     # four reads in flight (queue depth)
python -m tools.bitnet_gemm.stream_probe --path B:/      # control: SATA HDD
python -m tools.bitnet_gemm.stream_probe --path E:/      # control: USB
```

| read | C: (KIOXIA NVMe) | B: (WDC 4 TB HDD) | E: (Seagate USB) |
|---|---:|---:|---:|
| 14.7 MB, one call | 9.25-9.99 ms, 1.5-1.6 GB/s | 4.3 ms, 3.6 GB/s | 175 ms, 0.09 GB/s |
| 200 MB, one call | 3.3 GB/s | 2.5-3.0 GB/s | 32 MB/s |
| write 2 GB | 69 MB/s | 57 MB/s | - |
| 14.7 MB at queue depth 4 / 8 | 1.37 / 1.65 GB/s | - | - |

**Corrected: I first reported 1.67 GB/s as "the device", and that is not what it measures.** The control row is what
exposes it - a 4 TB spinning disk does not read at 2.5-3 GB/s. Reads of freshly written data are being answered from
somewhere outside the guest's page cache (the 23x read/write asymmetry on both disks points the same way), so no arm of
this probe observes the medium itself.

Two artifacts of my own probe had to be fixed before it gave even these numbers, and both produced confident nonsense
first: a file smaller than the drive's own cache reads from that cache (the *same HDD* reported 4 GB/s), and a repeated
1 MB block compresses to nothing under NTFS compression, so the read measured the decompressor. The current probe uses
incompressible data, a discarded first pass, steady-state passes over a 256 MB window, and reports the controls beside
the result.

The number that does not depend on knowing the medium is the **expert-sized read: ~9 ms for 14.7 MB**, flat at queue
depths 1, 4 and 8. Against 1.8 ms of GEMV, a pipeline that hides one expert read behind one expert multiply has a
ceiling of `1.8 / (1.8 + 9) = 17%`, per expert, per layer, per token. Reading 200 MB at a time does reach 3.3 GB/s, so
the per-call cost amortises with transfer size - the same "more work per read" conclusion from the other direction.

The arithmetic that decides when streaming is worth building: the kernel digests 58.7 M weights in 1.8 ms = **32.6 G
weights/s**, while 1.6 GB/s of packed weights is 6.4 G weights/s. Compute outruns the read by ~5x at expert granularity,
so at batch 1 the read can never be hidden. The fix is not a faster I/O path, it is **more tokens per weight read**: at
5-6 tokens one expert's 9 ms read sits under 5-6 x 1.8 ms of compute and the pipeline is level. Streaming experts is a
*batched* inference design; at batch 1 the expert set has to fit in RAM.

### Prefetch depth cannot fix this, and the numbers say why

The next proposal attacked the same problem from the scheduling side: predict layer L+1's experts from layer L's
hidden state, predict the next token from an n-gram table, and submit `io_uring` reads early so the transfer overlaps
compute. Its latency budget assumes a cold expert read is **15-35 us**.

That unit is off by three orders of magnitude, and the arithmetic is not close. 14.7 MB in 18-25 us is **590-820 GB/s**,
which is 100x a PCIe Gen4 x4 link (~8 GB/s) and 500x what this host measures. At line rate 14.7 MB cannot arrive in
less than 1.8 ms; measured, it is 9.5 ms. To have a 35 us transfer at 1.63 GB/s the expert would have to be **57 KB** -
at which size the design is correct, and which is a different model from this one.

The deeper problem is that prefetching hides *latency*, and there is no latency left to hide once the link is
saturated: the steady state is total bytes divided by bandwidth, which scheduling cannot change. Measured over one
token's FFN expert stream (32 layers x 14.7 MB, 470 MB), at the two depths the proposal would use:

| read pattern | throughput | 470 MB takes |
|---|---:|---:|
| depth 1 | 1.63 GB/s | 0.28 s |
| depth 4 (four readers in flight) | 1.37 GB/s | 0.33 s |

Deeper pipelining is **slightly worse**, and both are far above the 115 ms of GEMV compute they would feed. At top-2 the
stream is ~940 MB per token, so the floor is **0.58 s/token** however the reads are scheduled, predicted or reordered.

That also disposes of "mispredictions cost minimal CPU overhead beyond PCIe bus occupancy": on a bandwidth-bound stream
a wasted read costs 9.5 ms of *link time* and directly displaces a useful one. And a prediction path that maps a
speculated token to an expert as `token_id % num_experts` is not a prediction - it is a hash, with an expected hit rate
of 1/num_experts.

What does move the floor is changing the bytes or the work per byte, which is what the earlier sections already found:
a **batch of 5-6 tokens** puts one expert's 9.5 ms under its own compute, and **smaller experts** (clustering 64 experts
into 12, or a rank cut) divide the read by five and drop the break-even batch to about one. Both are model-side changes;
no amount of queue depth substitutes for them.

### The disk claims are the least trustworthy numbers here

Every figure in that section is what a process sees when it asks for these bytes, not a device characteristic. The HDD
control proves an intermediate cache exists that `FILE_FLAG_NO_BUFFERING` does not bypass, and I cannot reach it from
inside the guest to characterise it. Anyone re-running this on bare metal should expect the NVMe row to move.

## Two corrections

1. **"Cache blocking wins 1.58x."** Withdrawn: one sample against one sample, on a baseline that moves ±45% run to run.
   Re-measured with matched sampling, the wide shape is indistinguishable from the untiled kernel and the narrow shape
   loses 25-42%. The K tile alone is still worth ~10% at wide N; the N tile is a parallelism knob.
2. **"1.67 GB/s is the device."** Withdrawn: it is the throughput of 14.7 MB unbuffered reads on this host, behind a
   cache the flag does not bypass. The 4 TB HDD reporting 3 GB/s is the proof. What survives is the expert-sized read
   cost, and the ceiling derived from it (17% at batch 1, level at 5-6 tokens).

Neither correction changes the operating recommendation, which is the useful part: **do not port either kernel.** The
LUT matches the shipped pair table at parity, so a repack of the weight layout would buy nothing; paging is a batched
design and this engine decodes at batch 1.

## What would settle the rest

- **The comparison against the pair table needs to happen in one process.** Build `libkcpu`, push the same 2048 x 1920
  ternary matrix through `rows8` and through `tmac_gemv_avx2`, and report both from the same binary. Everything cross-
  source above is weaker than that.
- **The device needs a bare-metal run.** The same probe on a machine without a virtual block layer would say whether
  1.6 GB/s at 14.7 MB and 3.3 GB/s at 200 MB is the KIOXIA or the host.
- **The batched-regime prediction is untested.** The 5-6 token break-even is arithmetic from two measured rates, not a
  pipeline; a real double-buffered run at batch 8 would confirm or kill it.
