# Ternary LUT kernels and expert paging on this host

Two things are measured here, both about the same question: on an i9 with AVX2 and no VNNI or AMX, what does a
ternary weight actually cost, first when the *activation* is quantised to int8 and looked up through a table, and
second when the weight has to arrive from disk before it can be multiplied. Both were proposed to this branch as
ready-to-use kernels; neither was usable as written, and the numbers below are what settled each one.

**Four claims I made earlier in this branch are wrong, and all are flagged where they appear**: that 1.67 GB/s was
"the device", that clustering brings the streaming crossover to batch 1, that a spinning disk was reading at 3 GB/s, and
that the LUT kernel matched the shipped pair table. None of them survives a second measurement. The corrections are
collected in [Seven corrections](#seven-corrections) rather than folded quietly into the text.

A fourth claim - that the cache-blocked LUT beats the untiled kernel by 1.58x - was withdrawn here and is now
**reinstated with a proper interval**, at 1.80x: the original number came from one sample, the withdrawal from an
unpinned min-of-N, and both were measured with the SMT siblings counted as cores.
[That section](#the-cache-blocked-variant-correct-and-at-the-expert-width-it-is-180x-faster) says what was wrong with
both measurements, because getting the wrong answer twice is the part worth recording.

**And one of those corrections was itself wrong**, which is the most useful thing in the note: the answer to "that
spindle cannot read at 3 GB/s" was not "so B: is flash". B: *is* the 4 TB spindle, the 3 GB/s readings there were the
page cache, and the 1.66 GB/s figure under test turns out to be a real rate of the *other* drive - the NVMe at C:, which
is not where any pack lives. [Which volume, and what it actually is](#which-volume-and-what-it-is) has the hardware
identity, the three arms that agree, and the numbers a streaming design has to use instead.

The harness is [tools/bitnet_gemm](../tools/bitnet_gemm/README.md): `tmac.cpp` (the kernels), `tmac.py` (build,
validation, timings), `stream_probe.py` and `copytest.py` (the disk - the second exists because the first measured its
own file mid-write). Nothing under `kindi/` imports it; this is a measurement tool, not a shipped path.

Host: Intel i9-9980HK (Coffee Lake, family 6 model 158), AVX2 + FMA3 + F16C, **no AVX-512, no VNNI, no AMX**, 8 cores /
16 threads, 64 GB RAM, Windows under Git Bash. One other caveat applies to everything in the second half: this host's
block devices are virtualised (see [The disk claims, resolved](#the-disk-claims-resolved)).

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

Weights decoded from 2-bit codes inside the kernel, 0.25 bytes a weight. Pinned to the 8 physical cores, arms
interleaved, per-call time = median of the rounds:

| N x K | threads | median ms | CV | weights/s | packed stream |
|---|---:|---:|---:|---:|---:|
| 1024 x 1024 | 1 | 0.100 | 5.8% | 10.5 G | 2.63 GB/s |
| 2048 x 1920 | 1 | 0.406 | 8.8% | 9.7 G | 2.42 GB/s |
| 4096 x 4096 | 1 | 1.764 | 46.3% | 9.5 G | 2.38 GB/s |
| 1024 x 1024 | 8 | 0.243 | 24.8% | 4.3 G | 1.08 GB/s |
| 2048 x 1920 | 8 | 0.275 | 9.8% | 14.3 G | 3.58 GB/s |
| 4096 x 4096 | 8 | 1.024 | 28.0% | 16.4 G | 4.10 GB/s |
| 14336 x 4096 | 8 | 5.146 | 21.7% | 11.4 G | 2.87 GB/s |

The **CV column is the finding**. On this host a single call's time varies by 6-46% round to round even with nothing
else running, so a "best of 9" is a floor on the distribution rather than an estimate of it: the same 4096 x 4096 row
that read 0.61-0.68 ms as a best-of-9 reads **1.024 ms** as a median. Both are true and neither is wrong - they answer
different questions - but only the median is what a token costs and only the median can be given an interval. This is
what the earlier version of this table reported, and the two-hour disagreement about whether the tiling helped was one
person reading a floor as a centre.

The 8-thread 1024 x 1024 row being **slower than one thread** (0.243 against 0.100 ms) is OpenMP's launch and barrier
cost on a 1 M-weight matrix, and it is why every arm in the tiling table below is given a 14.7 MB expert to chew on.

**Against the pair table already in the engine**: this paragraph used to claim parity from a comparison across
sources, and that comparison is withdrawn - see [The two kernels, in one process](#the-two-kernels-in-one-process-the-pair-table-wins-everywhere),
where both decoders run in one process on one matrix. In one process the pair table is 1.7-2.2x faster at 2048 x 1920
and 2.8-2.9x faster at the expert width. The single-shuffle version would be ~2x fewer ALU ops a group and would also be
wrong; it is not the reason to skip a repack.

## The cache-blocked variant: correct, and at the expert width it is 1.80x faster

The follow-up proposal added `N_blk x K_blk` blocking, accumulators spilled to a per-tile buffer, and `_mm_prefetch` on
the weight stream. It is implemented as `tmac_gemv_avx2_tiled(a, w, scale_a, scale_w, n_blk, k_grp_blk)` and it is
bit-exact over four blockings (`max |tiled - exact| = 0.0`), with the same wrapped int8 table as above - blocking does
not touch that bug.

```bash
python -m tools.bitnet_gemm.tmac --json B:/models/scratch/tmac_tiling.json   # validate both arms, then pair them
```

**This section has now been wrong in both directions, so the method is part of the result.** It first reported 1.58x for
the tiled arm from one sample against one sample; then withdrew that to "not separable from noise" on an unpinned arm
whose baseline moved 1.38-2.59 ms run to run; and both of those measurements were taken as the *minimum* of their
samples - the luckiest draw of the distribution whose width was the question. What follows pins the process to the
physical cores, interleaves the arms inside every round, and reports a **paired** ratio with a bootstrap interval, the
same statistic the NLL gates use.

Method: K = 4096, 8 OpenMP threads, process affinity **`0x5555`** - one thread on each of the 8 **physical** cores,
SMT siblings excluded - 15 interleaved rounds per run with the arm order rotated each round, 3 runs of the whole table.
Ratio = mean of the per-round paired ratios, CI = 95% bootstrap over the pooled rounds (10k resamples, seeded); `per
run` is each run's own mean ratio, which is the column that says whether the answer comes back. The +/-2% band is this
host's noise floor - a ratio inside it is not called a result.

> **The mask in this table was `0xff` until 2026-11-05, and that is 8 logical processors, not 8 cores.** This machine is
> 8C/16T with adjacent SMT siblings (core 0 = `0x3`), so `0xff` was four cores' worth of threads; the topology call in
> `tmac.py` was misparsing `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` (DWORD relationship and size, read as two WORDs)
> and silently falling back to one mask per logical processor. With the parse fixed the mask is `0x5555` and every row
> below is re-measured. The effect is not cosmetic: the tiling win grows from 1.55x to **1.80x** and, at the narrow
> shape, the 512 x 512 arm stops being separable at all.

| N | arm | median ms | speedup | 95% CI | per run | verdict | packed stream |
|---|---|---:|---:|---|---|---|---:|
| 14336 | untiled kernel | 3.89 | 1.00x | | | baseline | 3.8 GB/s |
| 14336 | **1024 x 512 (the proposal's tiles)** | **2.21** | **1.80x** | **1.71-1.91** | 1.91 1.76 1.74 | **faster** | 6.6 GB/s |
| 14336 | 512 x 512 | 2.60 | 1.49x | 1.34-1.62 | 1.55 1.46 1.44 | faster | 5.7 GB/s |
| 14336 | 1024 x all K | 2.76 | 1.40x | 1.27-1.53 | 1.50 1.34 1.36 | faster | 5.3 GB/s |
| 14336 | 2048 x 1024 | 3.89 | 1.22x | 1.08-1.40 | 1.30 1.23 1.14 | faster | 3.8 GB/s |
| 2048 | untiled kernel | 0.48 | 1.00x | | | baseline | 4.3 GB/s |
| 2048 | 1024 x 512 | 0.53 | 0.98x | 0.79-1.25 | 0.77 1.11 1.05 | indistinguishable | 4.0 GB/s |
| 2048 | 512 x 512 | 0.55 | 1.02x | 0.85-1.29 | 0.74 1.19 1.14 | indistinguishable | 3.8 GB/s |
| 2048 | 1024 x all K | 0.76 | 0.75x | 0.63-0.92 | 0.59 0.82 0.83 | slower | 2.7 GB/s |
| 2048 | 2048 x 1024 | 0.95 | 0.53x | 0.46-0.63 | 0.40 0.60 0.59 | slower | 2.2 GB/s |

The direction is stable and the size is not: the note's tiles beat the unblocked kernel in **all three runs**
(1.74-1.91), so this is a finding rather than a lucky run, but the pooled interval (1.71-1.91) is narrower than the
per-run spread, which means the size is "between 1.7x and 1.9x" and not a figure to quote to two decimals. The baseline
row is the same arm as the 14336 x 4096 row of the throughput table above, which is the check that the two tables are
one measurement and not two.

What the shape dependence says, and it is the useful part:

- **`N_blk` splits the work across threads, and that is most of the win.** One tile for eight threads - 2048 x 1024 at
  N = 2048 - loses 47% (0.53x, all three runs): seven threads idle while one does the whole GEMV. Two tiles sit inside
  the noise floor at 0.98x, and 7 tiles (2048 x 1024 at N = 14336) wins 1.22x. The same 1024-wide tile at N = 14336 is
  14 tiles for 8 threads and wins 1.80x. So the note's tile size is right *at expert width* and wrong below it, and
  what tracks the win is roughly how many tiles there are for eight threads - not the tile shape.
- **The 512 x 512 claim does not survive the corrected mask.** It was 1.37x at wide N and 1.10x at narrow; it is
  **1.49x** and **1.02x [0.85, 1.29]** now, so at the narrow width nothing is separable and there is no blocking that
  pays at both widths. At the width we run, the note's own 1024 x 512 is the best arm - 2.21 ms against 2.60 for
  512 x 512 - so its tile choice was right for the shape that matters, and the earlier preference for 512 was an
  artefact of four-core pinning.
- **The K block adds about a third, on top of the N tile.** At N = 14336, 1024 x all K is 1.40x and 1024 x 512 is
  1.80x: holding the scheduling fixed, blocking K so a 512-group slice of the two-plane table (16 KB, inside L1) is
  reused by a whole tile before eviction is worth a **further 29%**. That splits the total saving roughly two thirds
  N tile and one third K block - which reverses the split reported under the old mask (60/40 the other way), and it is
  the reason the size is quoted as a range: the components move between runs more than the total does.

**The repack is still not justified, and this is the number that decides it.** The tiled LUT reaches 2.21 ms at
14336 x 4096; the shipped pair table, measured the same way in the same process **and under the same corrected mask**,
is **1.81 ms** ([below](#the-two-kernels-in-one-process-the-pair-table-wins-everywhere)). So the best LUT arm is
**1.22x** slower than the kernel the engine already runs, and the tiling win is real but internal to a family that
loses. Tiling the pair table is not the same experiment - `rows8` already walks two row blocks together with no
per-tile rebuild - but it is the one worth running next, and it is cheaper than a pack-format change.

**Withdrawn:** the previous text explained the (then null) result with "the kernel is not bandwidth-bound, so blocking
has nothing to recover", and asserted that a correct 131 KB table "is why nothing stalls". Both were fitted to a
difference that was noise, and the first is contradicted here: blocking recovers 35% of the time at the width we run.
What is still true is the arithmetic behind it - a correct two-plane table is 128 bytes a group, **131 KB at K = 4096**,
not the proposal's 32 KB - and that the stream moves only 2.9-4.4 GB/s, far under this machine's memory ceiling, so
the limiter is on the compute path and what the tiles recover is table-slice reuse and thread occupancy, not DRAM
traffic. The proposal's `K_blk = 512` (16 KB) is safe by the right arithmetic for the wrong reason: it is 4x smaller
than the table it is supposed to tile.

Still open: **which** of the tiled arm's three simultaneous differences does the work - dynamic scheduling across
tiles, the per-tile table rebuild, or the blocked accumulator - because all three arrive together in this table. The
one-factor arm (the untiled kernel with only its schedule changed) is the experiment that would attribute it, and it is
cheap now that the harness exists. The repack is worth doing at expert widths without waiting for that answer.

**One downstream number is now known to be a floor.** The paging sections below, and `stream_probe.py`,
`prefetch_budget.py` and `cluster_budget.py`, hide a 14.7 MB read behind **1.8 ms** of expert GEMV. That 1.8 ms is the
best-of-N figure this file used before pinning; the pinned median for the same call is **3.3 ms**, and the tiled arm is
what a paging pipeline would actually run. The direction of the error is safe - a floor on compute makes a double
buffer look *worse* than it is (ceiling `3.3 / (3.3 + 3.7) = 47%` at the idle read rate, not 33%) - so none of the
"streaming does not pay here" verdicts reverses on it, and none of them is re-derived above. But the ceilings, the
break-even batch counts and the 0.58 s/token figure all carry that constant, and they need re-running before anything
is built on them. Choosing C_MS is also a per-architecture decision now: `tmac.py --threads 8` prints the machine's.

Two defects in the proposal beyond the arithmetic, both cheap to check and both real: `_mm_prefetch(w_grp + 4 * N)` is
in **bytes** where a group is N bytes, so it prefetches 4 groups - 57 KB, or **896 cache lines**, at N = 14336 - while
its comment says "4 lines ahead", and 57 KB cannot survive a 32 KB L1. And the accumulator loop reads
`local_acc[local_idx + 24 .. +31]` unconditionally, so it silently requires `N % 32 == 0`; the proposal checks neither.
Neither defect is what makes the arm slow or fast, and neither is fixed by the measurement above.

## The two kernels, in one process: the pair table wins everywhere

Parity between the LUT and the shipped pair table was the weakest claim in this note, because it was made across
sources: the LUT's number from `tools/bitnet_gemm/tmac.py`, the pair table's from `kindi.cpu`, in different processes,
on different matrices, with each side packing the weights its own way. That cannot separate the kernels from their
drivers, and it is why the second half of the claim - that a repack would buy nothing - was inference rather than
measurement.

```bash
python -m tools.bitnet_gemm.libcompare        # --rows / --cols / --rounds / --runs / --threads
```

`kindi/kcpu.c` exposes `kc_rows8_gemv`, a harness entry that runs the production `rows8` on caller-packed bytes, so
both kernels load into one process: `libkcpu` through ctypes (built here with MSVC, `/arch:AVX2`), and
`tmac_gemv_avx2` through its torch extension. Same ternary matrix, same int8 activation vector, same timing loop, each
arm given an explicit thread count, 15 interleaved rounds x 2 runs per row, on mask `0x5555` (one thread on each of 8
physical cores) and mask `0x1` for the single-thread rows.

**The first version of this harness was not a comparison and its numbers are replaced, not adjusted.** It left the
LUT's OpenMP thread count at whatever the process default was, so the row it labelled "LUT, 1 thread" (0.219 ms) was
an all-core call, and the conclusion drawn from it - "at one thread the LUT is 1.8x faster than the pair table" - was
a label, not a kernel result. Both arms now get an explicit thread count on every row.

| shape | arm | 1 thread, ms | 8 threads, ms | vs pair, 1 thr | vs pair, 8 thr | G weights/s, 8 thr / 1 thr |
|---|---|---:|---:|---:|---:|---:|
| 2048 x 1920 | pair table (`rows8`) | 0.80 | 0.19 | 1.00x | 1.00x | 21 / 4.9 |
| 2048 x 1920 | LUT (`vpshufb`) | 1.37 | 0.42 | **0.56x** | **0.51x** | 9.4 / 2.9 |
| 14336 x 4096 | pair table (`rows8`) | 9.90 | 1.81 | 1.00x | 1.00x | 32.4 / 5.9 |
| 14336 x 4096 | LUT (`vpshufb`) | 27.6 | 5.26 | **0.36x** | **0.36x** | 11.2 / 2.1 |

Every ratio is the paired mean with a 95% bootstrap interval, and in all four rows the interval excludes 1.0 and both
runs agree: 0.56 [0.51, 0.62] twice at 2048 one thread, 0.51 [0.45, 0.57] at 2048 eight (0.52, 0.50 per run), 0.36
[0.35, 0.37] twice at 14336 one, 0.36 [0.32, 0.39] at 14336 eight (0.40, 0.31 per run). The expert-width row is the
stable one - the LUT's own median repeats (27.6 at one thread) and the ratio does not move - while the 8-thread row
carries a 97% CV on the LUT arm, which is this host's background load and not the kernel.

**So the parity claim is dead, in the direction that needs saying: the shipped kernel is faster.** At the width we run
- 14336 x 4096, one expert - the pair table is **2.8x** the LUT at one thread and 2.9x at eight (the eight-thread
figure understates it: the LUT arm's per-run ratios were 0.40 and 0.31 against a pair table whose own median moved
between runs). At 2048 x 1920 it is 1.7-2.2x. One mechanism is in the kernel's own comment: `rows8` walks two row
blocks together, so a group's table is loaded once for 16 rows instead of once for 8, which halves the table traffic and
gives the scheduler two independent accumulator chains to overlap with the shuffle latency. The LUT has one chain per
group. The rest of the gap is not attributed here - the tile experiment above shows how much a layout change inside the
LUT is worth (1.40x from the N tile, 1.29x more from the K block), and the same experiment has not been run on
`rows8`, whose per-thread working set is a 16-row strip of 16 KB with register accumulators and no tile to block.

**What survives is the correctness half, and it agrees to float rounding.** The arms differ by **2.2e-07** of the
largest output on both shapes (each is 5.9e-08 and 2.3e-07 from the exact integer answer), which is accumulation
order, not a layout difference. A LUT kernel would not change an answer; it would only make it later.

Caveats: the arms carry the activation scale the way each kernel expects it (floats into `rows8`, `scale_a` into the
LUT), which is the interface, not a difference in work; and these are one host's AVX2 tier - the picture is expected
to invert where `vpshufb` timing is good and widening is cheap, which is the Apple-silicon result in the literature,
and has not been measured here.

## The disk: an expert read is 3.3-4.0 ms, and the 9 ms figure was a read racing a write

The third proposal streamed cold expert weights with `io_uring` + `O_DIRECT` + `mlock`, double-buffered against the
GEMV. `io_uring` does not exist on this host, so the mechanism is not built here; what is measurable is the number it
rests on. `stream_probe.py` read 14.7 MB in 9.25-9.99 ms on C:, and every downstream figure in this note - the 17%
ceiling, the 5-6 token break-even, the 0.58 s/token floor - is that one number in arithmetic. It is wrong, and what
showed it wrong was a second path to the same bytes: a file copy.

`copytest.py` writes one file of incompressible data and measures it with the same kernel calls a read or a copy really
uses - `ReadFile` unbuffered, `ReadFile` buffered (the path `CopyFile` takes), `CopyFileExW`, and 4 KB random as the
control that identifies the medium rather than a cache:

```bash
python -m tools.bitnet_gemm.copytest --path C:/ --size 2048 --copy 1024 --flush 2048 --rand 300
python -m tools.bitnet_gemm.copytest --path B:/ --src B:/models/Mixtral-8x22B-v0.1.Q8_0-00001-of-00005.gguf --offset 16384 --size 2048
python -m tools.bitnet_gemm.copytest --path E:/ --src E:/Windows.iso --size 2048
```

| arm | C: | B: | E: (USB spindle) |
|---|---:|---:|---:|
| 14.7 MB unbuffered, whole file | 3.10-3.47 GB/s (3.3-4.0 ms a read) | 2.64 | 0.14 |
| 14.7 MB unbuffered, best single | 4.61 | 4.21 | 0.14 |
| buffered `ReadFile`, cold | 2.80-3.33 | 3.29 | 0.12 |
| buffered `ReadFile`, warm | 3.29-3.39 | 2.95 | 3.88-4.03 |
| `CopyFileExW`, one-way share | 1.95 (3.89 both ways) | 1.48 (2.97) | 0.11 |
| 4 KB random | 3.9-5.7 k IOPS | **24.8 k IOPS** | 110 IOPS |
| buffered write: into cache, then commit | 1.84, then 0.068 GB/s | - | - |

**On C: it is not the medium.** The same volume, the same 14.7 MB unbuffered reads, when nothing else is happening:
3.1-3.5 GB/s, 3.3-4.0 ms a read. The 1.6 GB/s figure was not [that drive's](#which-volume-and-what-it-is) speed.

> **Re-measured 2026-11-05, and this section was reading the wrong volume.** `stream_probe` writes to `%TEMP%`, which is
> on **C:**, while every pack and GGUF this branch ships lives on **B:**. They are not the same device and they differ by
> 7x. The `2.64` and `3.29` GB/s rows below are B:, and they are cache: see [Which volume](#which-volume-and-what-it-is).

**It is not the Windows I/O path either.** The buffered path a copy uses reaches 2.80-3.33 GB/s on C: and 3.29 on B: -
above the claim, not below it. On the genuinely slow device the fast arm is the *warm* buffered one (3.88-4.03 against
0.12 cold), so the path has headroom to spare; what it lacks is a device that can fill it.

**What it is: a read competing with the drive committing a write.** The `--flush` arm writes 2 GB without fsync and
reads it back on a timer, and the shape is unmistakable:

| when | rate |
|---|---:|
| 0.1 s after the write | 3.80 GB/s |
| 4-30 s after the write | **0.14 GB/s** |
| 37 s after the write | 3.68 GB/s |

A drive with a write-back cache takes writes at 1.84 GB/s and commits them at 68 MB/s, so 2 GB takes ~30 s to reach the
medium - and for those 30 s reads of the file being committed run at 0.14 GB/s, which is 10.5 ms per 14.7 MB. That is
the 9 ms number. `stream_probe` writes its own test file and reads it immediately, and the same code reading a file
written hours earlier reaches 3.1-3.9 GB/s, so what it reports is a file that has just been written rather than a
device. One honest gap: the commit reproduces the collapse on demand, but it does not account for every low reading -
`stream_probe` reported 1.61-1.65 GB/s on a file it had fsync'd, and `copytest` reported 3.44 on a file of the same age
and volume minutes earlier. The mechanism below is measured; that residual 2x between two probes on the same bytes is
not, and until it is, neither probe's single number should be quoted as a device rate.

**The copy does not vindicate 1.67 either, and this is what the question was for.** A copy is read *and* write on one
medium, so its one-way share is bounded by the slower side, and on this host the slower side is the commit: measured
copies ran 0.37-2.32 GB/s on C: and 1.48 on B:, a 6x spread in one afternoon that tracks cache and commit state rather
than the device. A copy is therefore a floor on the device - it proves the medium sustained at least that much in both
directions at once - and never a ceiling on reads. Quoting one as a read rate is this note's original mistake with the
sign flipped.

**The label on the control was right the first time, and the correction to it was wrong.** B: *is* a 4 TB SATA
spindle (WD4005FZBX), and the 24,820 IOPS that were used to call it flash are not reproducible and cannot be the device.
The evidence and the three arms that agree today are in [Which volume, and what it actually is](#which-volume-and-what-it-is).
E: is the USB Seagate, and its 0.12-0.14 GB/s cold against 3.88-4.03 warm is still the arm that shows what a cache
adds - it was never the only one.

So the rate to build on is the device's own, and only when the drive is idle. It was written as **3.3-4.0 ms per 14.7
MB**, which is a C: number, and C: is not where the weights are. Per volume, at the device's own cold rate:

| volume | what it is | cold unbuffered read | 14.7 MB expert | double-buffer ceiling |
|---|---|---:|---:|---:|
| **C:** | KIOXIA XG8 NVMe, 2 TB | 1.65-3.33 GB/s | 4.4-8.9 ms | 17-29% |
| **B:** | WD4005FZBX, 4 TB SATA HDD | **0.23 GB/s** | **64 ms** | **2.7%** |

On the volume every pack actually sits on, the expert read is **64 ms**, not 3.3-4.0, and the double-buffer ceiling is
**2.7%**, not 33%. That does not change the batched conclusion (which rests on the 19-25 ms measured under a running
gemm), and it makes model placement - packs on the NVMe, or resident - the first question any streaming design here
has to answer. It does **not** move the batch-8 result below, because that result never used this
rate: `batch_stream.py` times the read *while* sixteen threads of GEMV are running, and gets 19-25 ms - 5x the idle
device cost - because the reader and the kernel share memory bandwidth. Those are two different costs, and the pipeline
is the one that decides the design. The arithmetic above is unchanged in form: compute is 32.6 G weights/s and 3.7 ms of
read is 15.9 G weights/s, so compute still outruns the read, by 2x instead of 5x, and batch 1 still cannot hide it.

### Prefetch depth cannot fix this, and the numbers say why

The next proposal attacked the same problem from the scheduling side: predict layer L+1's experts from layer L's
hidden state, predict the next token from an n-gram table, and submit `io_uring` reads early so the transfer overlaps
compute. Its latency budget assumes a cold expert read is **15-35 us**.

That unit is off by three orders of magnitude, and the arithmetic is not close. 14.7 MB in 18-25 us is **590-820 GB/s**,
which is 100x a PCIe Gen4 x4 link (~8 GB/s) and 150-200x what this host measures. At line rate 14.7 MB cannot arrive in
less than 1.8 ms; measured, it is 3.3-4.0 ms with the drive idle (9.5 ms while it is committing a write, which is the
figure this section was written against). To have a 35 us transfer at 4 GB/s the expert would have to be **140 KB** -
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

Both rows of that table are low for the same reason the 9.5 ms was: `stream_probe` reads a file it has just written, so
its numbers are commit-bound. At the idle device rate the same 470 MB takes 0.12 s instead of 0.28 s, and the depth
comparison does not survive either - deeper pipelining on an idle drive is what fills a link that has spare capacity.
What the table does still show is that scheduling cannot change the floor once the link is saturated, and the pipeline
in the B=8 section shows the link *is* saturated in practice: 19-25 ms per expert read under a running batch.

That also disposes of "mispredictions cost minimal CPU overhead beyond PCIe bus occupancy": on a bandwidth-bound stream
a wasted read costs 3.3-4.0 ms of *link time* on an idle drive and 19-25 ms under a running batch, and either way it
directly displaces a useful one. And a prediction path that maps a
speculated token to an expert as `token_id % num_experts` is not a prediction - it is a hash, with an expected hit rate
of 1/num_experts.

### The distillation objective is sound; the over-fetch is not

The follow-up proposed training the early heads properly - KL against the temperature-softened teacher router plus a
Top-K margin-ranking term - and then raising prefetch recall from ~82% to >96% by fetching K+1 candidates, on the
argument that the extra expert is "a negligible margin" of bus utilisation. The first half is reasonable. The second
half is where it breaks, and it is arithmetic rather than opinion:

```bash
python -m tools.bitnet_gemm.prefetch_budget        # --recall / --overfetch to test other numbers
```

| policy | GB/token | B=1 | B=4 | B=6 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|---:|
| no prefetch (serial) | 0.92 | 723 | 1069 | 1299 | 1530 | 2451 |
| exact fetch, 82% recall | 0.92 | 717 | 717 | 801 | 1031 | 1953 |
| exact fetch, perfect recall | 0.92 | **608** | **608** | **691** | 922 | 1843 |
| over-fetch 3, 96% recall | 1.38 | 936 | 936 | 936 | 946 | 1868 |
| over-fetch 4, 99% recall | 1.84 | 1222 | 1222 | 1222 | 1222 | 1849 |

ms per token, from the measured 9.5 ms read and 1.8 ms GEMV. At batch 1 the over-fetch is **29% slower than no prefetch
at all** - it fetches 1.38 GB a token instead of 0.92 and the link has no spare time to absorb that. Raising recall by
fetching more is counterproductive on a bandwidth-bound stream; raising it *at the same fetch count* is the only useful
direction, and that is what a better head does. Over-fetch only overtakes the 82% exact arm at **B = 8**, and never
overtakes exact fetch with perfect recall at any batch in the table.

Every millisecond above is 9.5 ms of read, which is the commit-bound figure. Re-run at the device's own 3.7 ms
(`prefetch_budget.py --read 3.7`) the table halves and the crossover moves left rather than disappearing: at B=1
over-fetch still loses (365 against 279 ms, 31% worse, because it fetches 1.38 GB against 0.92), it starts winning at
**B = 4** instead of B = 8, and the best batch for exact prefetch is **B = 2** (49% saved) instead of B = 4-6. The
qualitative answer - over-fetching cannot buy recall on a saturated link, only a better head at the same fetch count
can - is unchanged, because it rests on the bytes, not on the rate.

Note also what the columns say about the whole idea: prefetching is worth most at **B = 4-6** (43-47%), precisely where
compute rises to meet the link - the same batching conclusion as above, reached from the scheduling side.

### One layer of lookahead is all these heads can reach

The runtime integration submits L+1's speculative reads from L's pre-attention normalisation. That is the correct
place for the *signal* - a head consumes a hidden state of its own layer - and it is also the ceiling: at batch 1 a
layer's compute is top-2 GEMV = **2 x 1.8 = 3.6 ms**, while an expert read is **3.3-4.0 ms** on an idle drive. One
layer of lookahead therefore covers **most of it** - 97% at 3.7 ms - which is the opposite of what this section said
when the read was the commit-bound 9.5 ms and one layer could hide only 38%. Both figures are right about different
states of the drive, and neither describes the case the design actually runs in: `batch_stream.py` measures a read at
**19-25 ms** while the batch's GEMV is running, and at that cost one layer of lookahead hides 15-19% again. The
two-rate answer is the one to carry: the read is cheap when nothing else is happening and expensive when the thing it
is meant to hide behind is happening, which is why the pipeline and not the isolated rate settles this.

Going deeper is not a training problem. A head maps layer L's state to layer L+1's router, so predicting L+2 would need
L+1's hidden state, which does not exist until L+1's attention has run. Chaining predictions through a *fabricated*
state is a different model, and its recall would compound (0.82^k for k layers). The integer router itself costs
~4-5 us per layer at 64 experts x 4096, i.e. ~160 us per token - 0.05% of a 352 ms token - so the head is not the
constraint and neither is its cost. What the head cannot do is make the read cheaper: at batch 1 it can only overlap
one layer's 3.6 ms against it, and in a running pipeline the read is 19-25 ms rather than 3.7. At batch 4-6 the same
layer has 14-22 ms of compute, which covers the idle read outright and a good part of the contended one; that is the
regime where the policy becomes worth building.

Three defects in the training harness as written, beyond the model: the recall it prints is computed on the **training
batches it is fitting**, with no held-out split, so 82% and 96% are upper bounds; the recall metric counts hits against a
K+1 candidate set and divides by K, so over-fetching inflates the score by construction - the very comparison being drawn;
and the two runtime sketches disagree about what feeds the head - one uses L's post-attention output, the other L's
pre-attention normalisation - so whichever is trained, the other integration is out of distribution. Post-attention is
the better predictor (less drift to bridge) and should be the one written down.

What does move the floor is changing the bytes or the work per byte, which is what the earlier sections already found:
a **batch of 2 tokens** puts one expert's 3.7 ms read under its own compute (5-6 is what the commit-bound 9.5 ms
needed, and it is the number this section was written with), and a **rank cut** divides the read itself.
Clustering 64 experts into 12 is the case that does not work that way - it divides the *table*, not the read a token
issues - and it is measured in the next section.

### Clustering 64 experts into 12 divides the table, not the read

The proposal: merge each group of experts into one averaged expert, taking 64 down to 12, so the expert table is 18.75%
of its size and - the claim - the break-even batch falls to about one. The two halves of that are different questions.
Merging does shrink the table, by definition. It does not shrink an expert: a cluster is still one 14.7 MB matrix, and
a token still routes to k of them. What changes per *token* is only that two of its slots may now land in the same
cluster, and what changes per *batch* is that the union of a batch's experts is capped by the number of clusters.

Both are properties of the router's real overlap, so they are measured from a captured log of one: 48 layers, 128
experts, top-8, 384 tokens over 4 documents, three documents fitting the clustering and the fourth scoring it.

```bash
python -m tools.bitnet_gemm.cluster_budget --clusters 12      # the proposal's shape, on a 128-expert log
python -m tools.bitnet_gemm.cluster_budget --clusters 24      # the same 18.75% ratio
python -m tools.bitnet_gemm.cluster_budget --selftest
```

| arm | distinct(B=1) | GB/token B=1 | ms/token B=1 | GB/token B=8 | ms/token B=8 | break-even |
|---|---:|---:|---:|---:|---:|---:|
| as routed, 128 experts | 8.00 | 3.67 | 76 | 2.01 | 42 | B=64 |
| 12 clusters, co-activation | 5.24 | 2.41 | 50 | 0.55 | 14 | **B=8** |
| 12 clusters, random partition | 6.13 | 2.82 | 58 | 0.66 | 14 | **B=8** |

resident arm for comparison: 14 ms/token at every batch (8 slots x 1.8 ms, no read at all). Table size at 32 layers:
58.8 GB as shipped, **5.5 GB at 12 clusters**.

These columns use the commit-bound 9.5 ms read. At the device's own 3.7 ms (`cluster_budget.py --read 3.7`) every
crossover moves left and the ordering is preserved: as routed **B=16**, 12 clusters by co-activation **B=2**, the same
12 at random **B=4**. That is a *better* outcome for clustering quality than the table below says - chosen groups are
worth two steps of the batch grid rather than zero - and it does not change the conclusion, because the batch-1 arm is
still over either way and the residency route is the one that reaches batch 1 outright.

What the columns say:

- **Batch 1 improves, and not for the reason given.** 5.24 distinct matrices instead of 8 is a 35% cut in the read, and
  it is real - but it is exactly the tokens whose k experts were merged into one matrix, i.e. the tokens that now compute
  fewer expert functions. It is the quality cost of the merge showing up as a bandwidth saving, not a free lunch.
- **Batch 1 still does not break even.** One token's compute is 8 x 1.8 = 14.4 ms and a read is 9.5 ms commit-bound
  (3.7 ms idle), so break-even at B=1 needs the slots in **1.52 distinct matrices or fewer** (3.89 at the idle rate). At
  12 clusters a token touches 5.24 - 3.5x over on the first reading, 1.35x on the second, and over on both.
- **The crossover moves from B=64 to B=8, and the number of groups is what moves it.** A batch's union cannot exceed the
  cluster count, so with 12 clusters the read per token falls as 12/B; it drops under the k/5.28 compute ratio at B=8.
  That is arithmetic on the group count: `cap_break_even(12, 8) = 8`, and `cap_break_even(24, 8) = 16`, which is what
  the 24-cluster run measured. A **random** partition of the same 12 reaches B=8 as well, which is the control that
  matters: if the crossover were the win being claimed, clustering quality would not be needed for it.
- **Picking the groups buys about a sixth of the bytes, not a step of batch.** Co-activation beats the random partition
  at every batch (5.24 vs 6.13 distinct at B=1; 9.57 vs 11.50 at B=8, which is 0.55 vs 0.66 GB) and never changes the
  crossover. Worth having, not worth a design.
- **The merge is not free, and the log prices the exposure but not the loss.** At 12 clusters **59.5% of a token's
  routing weight** lands in a group it shares with another slot, i.e. is computed by an averaged matrix instead of the
  expert the router chose. A random partition averages 43.5% and saves fewer bytes - so the better clustering merges
  *more* of the weight, because the experts it groups are the ones that carry weight together. Whether the average is a
  good stand-in for both members is a question about the weights, and this log has none of them.
- **The honest route to batch 1 is residency, not paging.** 5.5 GB at 12 clusters fits in VRAM on a 16 GB card, and then
  there is nothing to page and no break-even to reach: batch 1 runs at the 14 ms compute floor. "Break-even at one"
  comes true through the table fitting in memory - which is what merging experts actually delivers - not through
  streaming the merged table off disk.

Caveat on the shape, and what it does and does not cost the answer: the model in the proposal has 64 experts and the
capture has 128. The **crossover is exact anyway** - 12 groups cap the read at 12 per batch whatever the layer held
before the merge, so 64 into 12 breaks even at B=8 just as 12-of-128 does; the pre-merge 64 only sets the baseline
(how much an *unmerged* batch would have read, B=64 here). What 128 makes approximate is the byte figure at small
batch: 12-of-128 (9.4%) is a coarser merge than 12-of-64 (18.75%) and so collides slots more, which means 5.24 distinct
matrices at B=1 is the favourable end, not the likely one - the 24-cluster run matching the ratio gives 6.46 and
break-even B=16. What is *not* measured here at all is quality: whether the experts the router groups together are one
function is `docs/EXPERT-MERGE.md`'s question, and it has to be paid before any of this table can be spent.

### B=8, run as a pipeline instead of predicted

Two sections above predicted this from two rates: "at 5-6 tokens one expert's 9 ms read sits under 5-6 x 1.8 ms of
compute and the pipeline is level", so by B=8 streaming experts should cost no more than keeping them in RAM. That was
arithmetic. `tools/bitnet_gemm/batch_stream.py` executes it - a reader thread fills the next expert's bytes while the
production `rows8` multiplies the current one into the batch, and the identical batch is timed resident (preloaded, no
reads at all). Experts are the shape the budget assumes, 14336 x 4096 = 58.7 M weights, 15.8 MB packed.

```bash
python -m tools.bitnet_gemm.batch_stream          # --batches B:fetched, --pool, --reps, --threads
```

| batch | experts fetched | read, ms | streamed ms/token | resident ms/token | ratio |
|---|---:|---:|---:|---:|---:|
| 1 | 8 (as routed) | 19.2 | 148.5 | 13.9 | 10.65 |
| 8 | 35 (as routed) | 19.4 | 85.7 | 12.2 | 7.01 |
| 8 | 10 (12 clusters) | 21.4 | 27.5 | 11.1 | 2.47 |
| 8 | 8 (whole batch shares) | 19.8 | 22.1 | 9.5 | 2.33 |
| 16 | 8 (whole batch shares) | 21.1 | 14.7 | 11.2 | 1.31 |
| 32 | 11 (12 clusters) | 20.8 | 14.2 | 9.3 | 1.52 |
| 64 | 86 (as routed) | 24.5 | 30.5 | 10.7 | 2.86 |

**Streaming does not beat residency at B=8 in any arm, and the best B=8 arm is still 2.3x behind.** Two inputs to the
prediction do not survive the pipeline:

- **A read costs ~20 ms while the batch is computing, not the 3.3-4.0 ms it costs on an idle drive.** Read alone, the
  same 14.7 MB comes back in 3.3-4.0 ms (above), and the 9.5 ms the budget was written against is what it costs while
  the drive is committing a write. Here the reader shares memory bandwidth with 16 threads of GEMV and the same 15.8 MB
  takes 19-25 ms. Nothing about the medium changed; the compute is what happened to it. A pipeline budget that adds a
  rate measured in isolation to work measured in isolation will be optimistic by exactly this factor.
- **A batch of 8 does not route to one expert's worth of weights.** The captured router fetches **35 experts for 8
tokens** - 4.4 per token, not 1 - so the sharing the prediction leans on is not there at this batch size. Only a
  12-cluster table, whose union is capped by construction, gets near it (10 experts for 8 tokens).

The parity condition is `fetched <= batch * k * c / r`, and it can be evaluated for whatever the host measures: with
c = 1.4 ms and r = 20 ms that is `0.56 * batch`, so even a batch that fetches a constant 8 experts crosses at B = 14,
not B = 8 - and it arrives there only slightly ahead (1.31x behind, in fact, because the reader still steals
bandwidth from the compute it is supposed to be hiding behind).

So the batched-regime prediction is **falsified as stated** on this machine: at B=8 streaming is 2.3x to 7x worse than
residency, and residency stays the design at every batch measured. What would rescue it is not a better pipeline but
the two model-side changes: a table small enough to be resident (12 clusters, 5.5 GB), and experts small enough that
k of them cost what k multiplies cost.

Caveats: the expert pool is 32 matrices, so a batch fetching more than 32 re-reads experts that are already in the page
cache; and the GEMVs run at the machine's 16 threads, so the streamed arm's timing includes the reader's contention,
which is the point of the measurement rather than an artifact of it.The third caveat this gained last time - that the host's reads come from behind a cache nothing could bypass - was
withdrawn because "the drive reads at 3.1-3.5 GB/s on its own". **It is back, and it applies to B: specifically.** B: is a
4 TB SATA HDD that reads at 0.23 GB/s cold and does 100 IOPS random; the 3.1-3.5 GB/s readings there are the page cache,
and a volume counter run beside one of them shows 38-90 MB/s on the wire ([Which volume](#which-volume-and-what-it-is)).
The 0.14 GB/s and 9 ms figures were reads racing a write commit - that part stands - but so does the caveat, for a
different reason than the one it was withdrawn for.

### The disk claims, resolved

These were the least trustworthy numbers in the note, and the two rounds of measurement above are what settled them. A
warm cache and a running write commit both depress the read rate, and the second one is what the original 1.6 GB/s and
9.5 ms were: a probe reading the file it had just written. With the drive idle, C: does 3.1-3.5 GB/s and an expert read
there is 3.3-4.0 ms; the USB spindle in the control does 0.12-0.14 GB/s cold against 3.9-4.0 warm, which is the same
effect at 32x and on a device whose cold rate is unambiguous.

This paragraph was written when B: was believed to be flash. It is not, so the same sentences on B: describe the page
cache: buffered reads there report 3.4-3.5 GB/s while `\LogicalDisk(B:)\Disk Read Bytes/sec` reports 38-90 MB/s for the
same interval. The volume, not the state, is the difference.

### Which volume, and what it actually is

Re-measured 2026-11-05 after the question "is 1.66 GB/s this disk, or the I/O path?", and the answer is neither of the
things the note assumed, because the note had been comparing two different disks.

```powershell
Get-PhysicalDisk | Select DeviceId,FriendlyName,MediaType,BusType,Size
Get-Partition | Where DriveLetter -in 'B','C','E' | Select DriveLetter,DiskNumber
```

| volume | disk | device | spec read |
|---|---|---|---|
| **C:** | 1 | KIOXIA KXG8AZNV2T04, 2 TB | NVMe, PCIe 4.0, flash | ~6 GB/s |
| **B:** | 0 | WDC WD4005FZBX, 4 TB | **SATA, HDD, 7200 rpm** | ~0.23 GB/s |
| **E:** | 2 | Seagate BUP Portable, 5 TB | USB, HDD | ~0.14 GB/s |

Measured today, same box, drive idle, each arm cold (a region nothing had read, window larger than the drive cache):

| arm | B: (HDD) | C: (NVMe) |
|---|---:|---:|
| unbuffered sequential, 14.7 MB reads | **0.23 GB/s** | 1.65-3.33 GB/s |
| direct copy, one-way share | **0.025 GB/s** | 1.2-2.4 GB/s |
| direct copy, read+write | 0.05 GB/s | 2.4-4.7 GB/s |
| 4 KB random over 35 GB | **100 IOPS** | - |
| buffered read (the copy path) | 3.4-3.5 GB/s **from cache** | 2.9-3.6 GB/s |

```bash
# B: is the volume the packs live on, and it is a spindle
python -m tools.bitnet_gemm.copytest --path B:/ --src B:/models/llama/Mixtral-8x22B-v0.1.Q8_0-00002-of-00005.gguf \
    --offset 20000 --size 2048 --copy 1 --passes 2 --rand 500
# C: the NVMe, with the copy arm run on settled data instead of a slice it just wrote
python -m tools.bitnet_gemm.copytest --path C:/ --src B:/models/retired/Bonsai-8B-Q1_0.gguf \
    --size 1152 --copy 1024 --settle 45 --passes 2
```

Three arms agree on B: and all three say the same thing: **0.23 GB/s sequential, 100 IOPS random** is a 7200 rpm disk,
and no reading of this volume can be 1.66 GB/s let alone 3.3. The buffered arm disagrees only because it is not touching
the disk - instrumenting `\LogicalDisk(B:)` during it shows 38-90 MB/s on the wire while the pass claims 2.97 GB/s.

So, answering the question directly: **1.66 GB/s is a real sequential read rate of this machine's NVMe (C:), not the I/O
path** - it reproduces at 1.62 GB/s from a settled file, and a direct copy on that volume sustains 1.2-2.4 GB/s one-way
while doing both directions. It is *also* not the number any streaming design here should use, because the weights are on
B:, where the device does **0.23 GB/s** and a 14.7 MB expert costs **64 ms**, not 3.3-4.0.

`--settle` was added to `copytest` for this: in `--src` mode the copy arm used to copy a slice the tool had just written,
so it measured that slice's commit - the exact artifact the tool exists to name.

## Seven corrections

1. **"Cache blocking wins 1.58x."** Withdrawn, then **reinstated at 1.80x [1.71, 1.91]** - and the withdrawal is worth
   more than the number. The original was one sample against one sample; the withdrawal used an unpinned min-of-N whose
   baseline moved 45% run to run, and *both* were pinned to eight logical processors while claiming eight cores. With
   the topology parse fixed (`0x5555`, one thread a core) and the arms interleaved, the note's tiles beat the untiled
   kernel in all three runs (1.91 1.76 1.74) at N = 14336, and nothing is separable at N = 2048. The earlier "512 x 512
   pays at both widths" does not survive either - 1.49x wide, 1.02x narrow. The "kernel is not bandwidth-bound so
   blocking cannot help" mechanism that explained the withdrawal is also withdrawn: the limiter is on the compute path,
   but the tiles recover 44% of the time.
2. **"1.67 GB/s is the device."** Withdrawn, and now placed rather than merely doubted: it is what a read costs while
   the drive is committing its own fresh write. Measured directly, reads of a 2 GB file run at 0.14 GB/s for the 30 s
   its commit takes, then return to 3.68 GB/s; with the drive idle the same 14.7 MB reads run at 3.1-3.5 GB/s. The
   expert read is therefore **3.3-4.0 ms**, not 9.5, and the ceiling derived from it is 33% at batch 1 and level at 2
   tokens rather than 17% and 5-6 - except in a running pipeline, where 19-25 ms per read is measured and the batched
   conclusion stands on that instead.
3. **"Clustering 64 experts into 12 divides the read by five and drops the break-even batch to about one."** Half wrong.
   It divides the table by five, to 5.5 GB, which is a residency win and does reach batch 1 that way. The read is
   divided by 1.5 at best (8 -> 5.24 distinct matrices at batch 1), and the streaming crossover lands at B=8 (B=2 at
   the idle read rate), not B=1 - and a random partition of the same size lands close behind (B=4).
4. **"A 4 TB spinning disk cannot read at 2.5-3 GB/s, so this table is not a device measurement."** The argument was
   right and so was its premise. Its *refutation* was wrong: B: is not flash. It is disk 0 in
   `Get-PhysicalDisk`/`Get-Partition`, a **WD4005FZBX 4 TB SATA HDD**, which reads at **0.23 GB/s** cold and does **100
   random 4 KB IOPS** over 35.4 GB - the 24,820 IOPS recorded here are not reproducible and were page-cache hits. The
   2.6-3.3 GB/s B: rows in the table above are therefore cache too, and the whole table's B: column has to be read as an
   upper bound on nothing. Detail and the arms that agree today: [Which volume](#which-volume-and-what-it-is). The
   conclusion that the old numbers were not device numbers survives, for a different reason - they were reads racing a
   write commit - but it now applies to *this section's own* corrections as well.
5. **The 1.66 GB/s figure is a genuine device number - for the other drive.** Reproduced today at 1.62 GB/s on C:, and
   confirmed by a direct copy there (1.2-2.4 GB/s one-way, 2.4-4.7 GB/s read+write). It is not the I/O path and it is
   not a cache: it is what this NVMe does. It is also irrelevant to a pack stored on B:, which moves 0.23 GB/s.
6. **"The LUT matches the shipped pair table at parity, so a repack would buy nothing."** True conclusion, false
   reason, and the false reason mattered: the comparison was made across documents, and its "LUT" side was a best-of-9
   minimum whose thread count was never set. In one process, pinned and paired, the pair table is **1.7-2.2x** the LUT
   at 2048 x 1920 and **2.8-2.9x** at 14336 x 4096. No repack: the best tiled LUT arm still trails the untiled pair
   table by 1.22x.
7. **"Pinned to eight physical cores."** False when it was written, in both of this note's harnesses. This machine is
   8C/16T with adjacent siblings (core 0 = `0x3`), and `_core_masks()` was reading `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`
   as two WORD headers where the API has two DWORDs, so the size field read 0, the loop broke on the first record, and
   it fell back to one mask per logical processor without saying so. `pin_affinity(8)` pinned logicals 0-7, which are
   **four cores with both their SMT threads**, and every table in this note before 2026-11-05 says "8 physical cores" on
   numbers taken that way. The fix is two offsets and a selftest invariant (every logical appears in exactly one core's
   mask). Re-measured, the tiling win went 1.55x -> **1.80x**, the narrow-width 512 x 512 win disappeared, and the LUT
   fell further behind the pair table. The direction of every earlier conclusion held; the sizes did not.

None of these change the operating recommendation, which is the useful part: **do not port either kernel.** The pair
table is faster than the LUT on this host by 1.7-2.9x, so a repack would buy a slow kernel in a new layout; the LUT's
one advantage - fewer bytes touched per group by a single shuffle - is also the version that overflows an int8 table.
Paging is a batched design and this engine decodes at batch 1.

## What would settle the rest

- **The comparison against the pair table now happens in one process** (`tools/bitnet_gemm/libcompare.py`, above):
  both kernels on one matrix, both given an explicit thread count, agree to 2.2e-07, and the pair table is faster at
  every setting (0.36-0.56x for the LUT). What is still open is whether the pair table's own `rows8` gains anything
  from the kind of tiling that took 1.80x off the LUT - that is the same harness with a different pair of arms. Its
  structure says probably not: a thread's working set is a 16-row strip (16 KB) it sweeps once with the accumulators
  in registers, so there is no K tile to block and no reuse for one to create.
- **Thread scaling could not be measured here, and that is a limitation, not a result.** `libcompare --sweep` walks
  1, 2, 4, 6, 8 threads on the physical cores and then 8 against 16 on both SMT siblings, every thread count an arm
  inside the same rounds. Two attempts gave non-monotone curves (6 threads beating 8) with per-arm CV of 15-45%, while
  this host sat at 47% background load from an indexer, two IDEs and a dev server. Pairing cannot fix that: thread
  counts contend with the background differently from each other, so the arms are not exchangeable. A quiet host is
  required, and until then the honest statement is that the 1-thread, 8-thread and 16-thread numbers above are three
  points on a curve whose shape is unknown.
- **The device has been measured, and the answer was not "bare metal".** `copytest.py` (above) separates the three
  things the old probe conflated - medium, path, cache state - and the depressed rate turned out to be a read racing
  the drive's commit of a write, reproducible on demand (3.80 -> 0.14 -> 3.68 GB/s). What a bare-metal run would add
  now is only the write-commit rate on a drive without a large volatile cache; the read side no longer needs it.
- **The batched-regime prediction has been run and it did not hold** (`tools/bitnet_gemm/batch_stream.py`, above):
  at B=8 streaming is 2.3-7x worse than residency, because a read costs ~20 ms under a computing batch and the router
  fetches 4.4 experts per token at that size. What remains open is a pipeline whose reader does not contend with the
  compute it feeds - a separate I/O engine, or a drive whose uncontended read stays near its 3.7 ms while 16 threads
  of GEMV run. Neither exists here, so the design question stays open rather than answered in streaming's favour.

## P6 — the memory side, measured (2026-11-08)

*Method first, as the plan orders it.* The two claims that matter are what the shipped engine already does, so both
were measured on the shipped reader, with process counters from `GetProcessMemoryInfo` (total page faults, soft and
hard together — Windows exposes no per-process hard-fault counter through that API) and `GetProcessIoCounters`
(explicit read operations — **memory-mapped file reads do not appear in it**, so a zero there is not proof of zero
disk traffic; the fault count is the honest instrument for the mapped path).

**Resident engine, Bonsai-2B pack.** Batch 1, fixed 64-token prompt, 40 generated tokens, three runs, pack on C:
(the NVMe at 1.63 GB/s unbuffered from `stream_probe`; B:'s 0.23 GB/s HDD figure above still governs scratch packs).

| | value |
|---|---|
| load time | 30.4 s (conversion-CPU dominated; placement moves it by ~2 s on the HDD) |
| kept arrays after load | 637 MB, of which converted pair codes + fp32 scales 528.8 MB |
| peak / steady working set | 1.17 GB during load / 667 MB in decode |
| decode speed (prefill included in each call) | 10.28–10.50 tok/s |
| disk read operations / bytes in decode | **+0 / +0** in all three runs |
| page faults in decode | **+0–1 per 40-token run** |
| bytes streamed per token from RAM | 528.8 MB (every weight, every token — the GEMV stream in `docs/CPU-PERF-2026.md`) |
| bytes read per token from disk | **0** |

Read granularity at load is one memmap slice per tensor: codes `R*(C/4)` bytes and fp16 scales `R*(C/128)*2`
bytes, read exactly once, then converted to anonymous arrays the kernel streams from DRAM. There is no per-token
disk read to coalesce, and total faults at zero bound major faults at zero. The literature's `VirtualLock` win
(+26%) assumes file-backed weights under eviction pressure; this engine's kept sets are anonymous RAM on a 64 GB
box, so pinning has nothing to fix. **No pinning, no read-ahead: nothing is left to improve on the resident path.**

**Paged MoE reader** (synthetic fixture: 1024 width, 512 expert width, 12 experts, 2 active, 6 layers, cache 2;
no converted real MoE pack exists on this host — the Qwen arms are GGUF-only):

| | value |
|---|---|
| bytes per expert, raw (codes + fp16 scales) | 208,896 |
| bytes per expert, staged (pcodes form) | 221,184 |
| slices/views per miss | 6 (3 matrices × codes+scales), convert path or pcodes path |
| expert matrices contiguous in `bcodes.bin`? | **no** — offsets 917504, 1708032, 131072 for one expert's gate/up/down |
| pcodes staging: speed / faults per token / misses per token | 55.5 tok/s / **302** / 13.3 (resident 261.6 tok/s, 0 faults; synthetic sizes, not comparable in absolute terms) |
| conversion staging: speed / faults per token | 4.5 tok/s / **15,711** — the reason `pcodes.bin` exists |
| tokens | **identical** across resident, pcodes-paged and conversion-paged (token-identity gate holds) |

One pread per expert would need the pack to store an expert contiguously; today it does not, and coalescing cuts
syscalls and faults, not bytes (the same 221 KB is read either way). That is a convert-side layout change to be made
when a real MoE pack exists to validate it against — not before. **Deferred, with the current read sizes now on
record.**

**Threads = physical cores.** Same Bonsai-2B run, separate processes, i9-9980HK (8C/16T), tokens bit-identical:

| threads | tokens/s |
|---:|---:|
| 8 (physical) | 10.44 |
| 16 (logical, the current default) | 10.36 |

Single run each, so the 0.8% gap is inside the noise this note has been bitten by before — the honest statement is
**no measurable difference**, not a win for 8. The default stays at the logical count: a tie does not justify
changing it for prefill-heavy or larger models where SMT may still pay.

**P6 verdict: measured, engine unchanged.** Resident decode already has zero disk reads and zero faults per token;
the paged reader's read sizes are now measured and its coalescing fix is deferred until a converted MoE pack exists.
