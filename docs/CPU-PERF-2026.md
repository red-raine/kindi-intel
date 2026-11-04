# CPU performance plan (AVX2, 2026)

A roadmap for the CPU engine, built from the 2024-2026 literature rather than from llama.cpp-era advice, and
triaged against what this engine actually is. Every claim is either a measurement on this host or carries a URL to
the paper it came from.

## Target and current state

This host is an Intel i9-9980HK: Coffee Lake, **AVX2 + FMA3 + F16C + BMI2, no AVX-512, no VNNI, no AMX**, 8 physical
cores, dual-channel DDR4-2667 at **~25 GB/s** measured achievable read bandwidth (42.7 GB/s peak).

Bonsai-2 is already ternary: `{-1,0,+1}` at ~1.7 bits/param, 506 MB on disk. One token streams ~0.506 GB through the
GEMV, so the machine's bandwidth ceiling is **25 / 0.506 ≈ 49 tokens/s** and we measure **23-29 tokens/s** (8 threads,
warm). The GEMV is 88-92% of a token.

That means:

- The engine moves ~0.43 GB per token at 19 GB/s of GEMV throughput, so "49 tokens/s is the bandwidth ceiling" is an
  **upper bound that the kernel never approaches for a bandwidth reason**. See below: bandwidth is not the mechanism.
- The kernel is **not memory-bound at all**. It is issue-bound at IPC 1.6-1.8. Byte-reduction is therefore not a
  speed lever, and every cache and prefetch item in the triage is closed.

## What the hardware counters say (measured Nov 2026)

There is no `perf` in this WSL image and installing one needs root, so `~/kt/perf/gb.c` opens the PMU directly with
`perf_event_open` - one event set per OpenMP thread, summed across the team. It `#include`s `kindi/kcpu.c`, so the
kernel under test is byte-identical to the shipped one, and drives `rows8` exactly as `gemv` does (pair tables,
barrier, row-block split, barrier). `perf_event_paranoid` is 2, which permits self-measurement of cycles,
instructions, L1D and dTLB; the generic `LL` event is refused on this kernel.

Faithfulness check: one 1.536 GMAC pass takes 85 ms on one thread and 21-23 ms on eight, which is the rate the real
engine implies for one token's GEMV (~39 ms for 2.1 GMAC), so the counters describe the production kernel.

| working set | GB/s | IPC |
|---|---:|---:|
| 8 MB (L3-resident) | 18.2 | 1.71 |
| 16 MB | 18.8 | 1.69 |
| 49 MB | 18.2 | 1.57 |
| 123 MB | 17.6 | 1.57 |
| 384 MB (DRAM-resident) | 19.0 | 1.76 |

**Throughput does not depend on the working set.** A set that fits in L3 runs at the same 18 GB/s as one 48x larger
and served from DRAM. A pure streaming read of the same 432 MB reaches **26.0 GB/s** on this host, so the GEMV sits
at 73% of what the memory system can deliver - and because making the data cache-resident changes nothing, that 73%
is not a memory limit. It is the rate at which the instruction stream can be issued.

**Confirmed on the real pack, not only a synthetic buffer.** `~/kt/perf/real.py` loads `~/.kindi/packs/bonsai-2b`
through the shipped `Engine`, so the pair-code repack is the production one, collects all 151 distinct matrices and
hands their real pointers to the same harness: 470 MB of pair codes + 59 MB of fp32 scales = **529 MB and 1.88 G
parameters per token**. A 13 MB subset runs at 18.6 GB/s and the full 529 MB at 18.3, IPC 1.56-1.72, 0.17-0.19
vpermps per cycle - the same numbers as the synthetic run to within noise.

In its real context the GEMV is slower. `kc_get_times` puts it at **34.47 ms of a 40.1 ms token (85.9%)**, which is
15.3 GB/s, against 28.9 ms for the same 151 matrices run back to back. About **19% of the GEMV is lost to running it
interleaved** with the fold, GDN and attention stages instead of as one unbroken stream. Still far under the 26 GB/s
ceiling, so the verdict stands; but the isolated harness is an upper bound, and part of that 19% is the per-call
boundary that batching would remove too.

One attribution bug, found while reconciling the two: `TS(cat)` closes the *current* interval into `kc_times[t_cur]`
and only then switches to `cat`, so `gemv()`'s `TS(prev)` at its end puts its own duration in category 2. The head's
full-vocab GEMV - a tied 248320 x 2048 tensor, 127 MB of the 529 - therefore lands in stage 2, not stage 6. Stage 6's
0.38 ms is just the argmax scan. Read the timers with that in mind.

**Per core the kernel converts 3.2 MACs per cycle** (1.536 GMAC, 22.7 ms, 8 cores, 2.6 GHz effective). A dense fp32
FMA loop sustains 16 on this core, and AVX-VNNI would do 64. That gap is the whole opportunity.

Where the cycles are *not* going, each measured rather than assumed:

| suspect | measurement | verdict |
|---|---|---|
| memory bandwidth | 8 MB and 384 MB working sets run at the same GB/s | **not the limit** |
| pair-table L1 thrash | pinning the table to one 2 KB group: L1D misses 25.7 M -> 7.1 M (72% fewer), **0% faster** | **not the limit**; L2 absorbs it |
| L1D misses generally | 26 M per pass at 8 threads, 17% of 152 M loads | not the limit |
| port 5 (`vpermps`) | 0.20 per cycle against a cap of 1.0 | 20% utilised |
| dTLB / page walks | 106 k per pass = 1.13 per 4 KB page, i.e. one per page - already optimal for 4 KB | ~1% |
| 4-ahead prefetch | removing it: 21.9 against 21.5 ms median, ranges overlap | inside the noise |
| thread binding | 9 interleaved reps x 9 configs: every median between 21.1 and 24.2 ms | inside the noise |

One thread reaches IPC 2.84, so IPC 1.6-1.8 at eight threads is not an architectural ceiling either - it is simply
what this loop achieves when eight cores run it.

**Implication: the only lever left is fewer instructions per weight word.** 16 MACs cost about 8.6 instructions: three
decode the weight word (`vpbroadcastd`, `vpsrlvd`, `vpand`) and five serve one lookup. A second token batch shares
that decode exactly as the two-row-block unroll already shares it between row blocks, which drops the estimate from
0.54 to about 0.41 instructions per MAC - a predicted **~1.3x** on the GEMV, testable with the same harness. That
reopens plan item #4, for a different reason than the plan gave: not bandwidth, but issue slots.

The three binding and prefetch conclusions above were each *single-run* wins of 1.3-1.4x before interleaving, and
none survived it. That is the strongest argument yet for the interleaved protocol; on this host a cold or contended
run can invent a 40% speedup out of nothing.

## Measurement protocol (read this before trusting a number)

Run-to-run spread on this host is **about ±12% when the host is quiet, and far worse when it is not**: the same
binary measured 26.1 tokens/s in one session and 19.7 in the next, and one interleaved run spanned 14.5 to 27.0. The
first run after a rebuild is also **~30% slower than the third**: a cold build measured 17.0 and 19.6 tokens/s, the
same binaries 23.6, 26.2 and 29.2 warm.

So: **discard the first run, take three or more, compare medians, and never compare a cold run against warm
figures.** Better still, interleave: build both variants into separate shared objects and alternate them inside one
process, so drift in clock, cache and host load lands on both sides. A single cold sample had suggested that mirror
consolidation was a 23% regression; the interleaved A/B puts it at 0.55x, so single-sample numbers here understate
effects as often as they invent them. Also check that `unattended-upgrades` is not running: it produced one
of the contaminated runs quoted above, with no other sign that the machine was busy.

## The twenty techniques, triaged

`AVX2` column = does it work on plain AVX2/FMA with no AVX-512 or VNNI.

| # | Technique | AVX2 | Verdict for this engine |
|---|---|---|---|
| 1 | int16 LUT + pack/unpack (`TL1_1`/`TL2_1`) | yes | **Rejected, measured.** The element-wise form the pack allows (81 entries, `vpgatherdd`) is **3.4-4.0x slower** than the pair table; the gather is the cost. See "The element-wise LUT" below. |
| 2 | Element-wise `3^g` index (T-MAC) | partial | **Blocked, and moot.** `3^5 = 243` needs an 8-bit index, and the 81-entry form of the same idea already loses 3.4-4.0x to the pair table on a *smaller* table. No pack change can rescue it. |
| 3 | Vector LUT, 1 lookup → N tokens | yes | **Enables #2, #4, #5.** Pays only when N > 1, which we do not have yet. |
| 4 | Batched GEMV, N tokens per weight pass | yes | **Measured: 1.35x at N=2, bit-exact, and now wired into decode** via n-gram speculative verification. 8.92 -> 5.92 instructions per lookup, IPC 1.64 -> 1.49. Not a bandwidth win at all - it is an instruction win, which is why it works. End to end +1.18x at 0.86 draft acceptance, 1.48x ceiling. See below. |
| 5 | n-gram / prompt-lookup speculation | yes | **Top priority, with #4.** Draftless; no second model. |
| 6 | Cache-aware LUT tiling | yes | 2-row-block sharing is done and is worth saving instructions, not bytes. Column tiling is **de-prioritised**: making the table L1-resident is worth 0-4%, which will not pay for a pack-format change. |
| 7 | Mirror consolidation (`T[7-c] = -T[c]`) | yes | **Rejected, measured.** 0.63x and 0.55x, 0/9 wins in each warm interleaved run. The 16 KB table halves the table traffic and still loses, because the mirror index and the sign flip cost about five extra uops per weight word. See below. |
| 8 | Zero-density encoding (BITCOS, `2-z` bpw) | no | **Rejected.** Needs AVX-VNNI-INT8 (Alder Lake+). The paper measures Bonsai 1.7B at 39.9% zeros → **1.00×** vs five-trit, and warns that an 8-core bandwidth-rich client sees no gain. |
| 9 | Repack 1.7 bpw → byte-aligned 2.0 bpw | yes | **Low value.** The general case pays for per-weight bit unpacking; we unpack once per *word* (one `vpsrlvd` per 8 rows). |
| 10 | Prefetch distance tuning | yes | **Closed.** 4 ahead against none is a wash in interleaved medians (21.5 vs 21.9 ms). The earlier 2/8/4 comparison was single-shot and its 28.6-vs-19.7 spread does not survive; do not re-tune this. |
| 11 | 2 MB huge pages on the weight mapping | yes | **Measured, ~1%.** 106 k dTLB walks per pass is 1.13 per 4 KB page - already one per page, the best 4 KB pages can do - against 473 M cycles. `HugePages_Total` is 0 on this image, so it would need THP. Not worth the change. |
| 12 | Non-temporal loads for the weight stream | yes | Low value. The stream is already read-once; no published win. |
| 13 | OpenMP decomposition, physical-core thread count | yes | **Done.** 8 threads, documented in `CARDS.md`. |
| 14 | `-march=coffee-lake -mtune=coffee-lake -flto` | — | Check the current flags; cheap. `-march=native` already resolves to exactly this. |
| 15 | Trim the 8-12% non-GEMV | yes | Minor, but it is where the free percentage is on a 2B model. |
| 16 | AVX-512 / VNNI / AMX kernels | **no** | Not available on this CPU. Do not port them. |
| 17 | `I2_S` integer dot (`maddubs`/`madd`) | yes | **Rejected on exactness.** Lossless only for models trained with BitNet's per-tensor int8 activations; Bonsai is PTQ'd, so this would not be token-exact. |
| 18 | int16 LUT to keep exactness | yes | A constraint on any of the above, not an optimisation: int8 table quantisation is lossy. |
| 19 | MoE expert paging / hot-expert caching | yes | **Works, now cheap to stage, still off by default.** `convert --paircodes` writes the staged form (`pcodes.bin`), so a miss is two memcpys: 0.05-0.3 ms an expert against 8-12 ms rebuilding it, and a cache of 6/12 reads 23 ms/token against 187-251. What is left is 69 Python callbacks a token at ~0.3 ms - stage in C before paging a model that needs it. |
| 20 | Ternary MoE (MoTE and follow-ups) | no (GPU kernels) | Design input, not a kernel: **keep a dense shared expert and ternarise only the routed ones.** |

Bonus — **KV cache quantisation**: at ctx 1024 our KV is ~12 MB against 506 MB of weights, and attention is ~2% of a
token. Measured elsewhere, q8 KV is +0.7% decode at 6K but **−36.8% at 110K** from dequantisation. Irrelevant at our
contexts and actively harmful at long ones. Keep f16.

## Priority order

1. **Batched GEMV (N tokens per weight pass) + n-gram speculation (#4, #5).** This is the only lever with a
   published order-of-magnitude win (Vec-LUT reports 4.2× prefill and 3.2× parallel decode) and it is the one the
   measurements here point at. Speculative decoding already exists in `spec.py` but **only for the CUDA tensor-core
   plan engine** (`KINDI_TC=1`); the CPU engine has none.
2. ~~Fair warm A/B of mirror consolidation (#7)~~ — **done and rejected** at 0.55x. It is the most useful negative
   result so far: the kernel is issue-limited, not table-limited, which rules out the whole family of
   fewer-bytes-more-ops trades that #9 and #12 belong to.
3. **Stop guessing about the stall.** `perf stat` the GEMV for `LLC-load-misses`, `dTLB-load-misses` and IPC before
   the next kernel rewrite. We know we are at ~50% of bandwidth; we do not know whether that is L1 latency, TLB, or
   port-5 `vpermps` pressure, and the three want different fixes.
4. ~~Vector LUT (#3)~~ - **closed by measurement**: the element-wise table it exists to unlock is 3.4-4.0x slower than
   the pair table even in its cheapest form (81 entries, one gather per 8 rows x 4 columns), so the N-token page it
   would make possible starts from too far behind.

Done since this list was written: the kernel tiers (scalar / AVX2 / int8) with an exactness gate and a native Windows
build, the MoE FFN with a NumPy-validated reference, and expert paging - the last two measured above. Paging's
staging cost is solved by the pair-code pack; its remaining cost is one Python callback per selected expert.

### Could `vpshufb` replace `vpermps`: the pre-broadcast table, measured

`vpermps` is the one instruction in the batched GEMV that lands on port 5 for every weight word, so a byte-shuffle
substitute looks attractive - same port, but if it were cheaper per lookup the bottleneck would move. It cannot be,
for a reason visible in the ISA: `vpshufb` reaches only the sixteen bytes of its own 128-bit lane, so an 8-float table
is applied a half at a time - two shuffles and a blend where `vpermps` needs one instruction.

The first arm (`gab2_shuf.c`) broadcast each half into both lanes on every lookup, four `vbroadcasti128` per weight
word across the two tables, and measured 0.62x. The obvious reading was that the broadcasts were the cost, so
`gab2_shuf2.c` stores the halves *already duplicated* (32 bytes each, `{f0..f3,f0..f3}` and `{f4..f7,f4..f7}`), making
a lookup two plain 32-byte loads at the price of a 64 KB pair table per token instead of 32 KB. Both arms are
bit-exact against the two-pass reference at every size (`max|sep-shuf2| = 0.000e+00`), so this is purely a layout
question.

9 interleaved reps, 8 threads, medians, arm order rotated per rep:

| working set | b2 (ms) | shuf, broadcast (ms) | **shuf2, pre-broadcast (ms)** | b2/shuf2 | ins/word, shuf2 vs b2 | port 5/word, shuf2 vs b2 |
|---|---:|---:|---:|---:|---:|---:|
| 0.59 MB | 0.076 | 0.127 | 0.124 | 0.61x | 3.43 / 1.39 | 0.92 / 0.39 |
| 1.18 MB | 0.155 | 0.248 | 0.243 | 0.64x | 3.43 / 1.39 | 0.92 / 0.40 |
| 2.36 MB | 0.191 | 0.450 | 0.438 | 0.44x | 3.43 / 1.39 | 0.92 / 0.39 |
| 4.72 MB | 1.079 | 1.301 | 3.845 | 0.28x | 4.08 / 1.39 | 0.97 / 0.39 |
| 9.44 MB | 1.174 | 3.083 | 2.343 | 0.50x | 3.76 / 1.43 | 0.98 / 0.39 |
| 18.87 MB | 2.081 | 4.005 | 3.695 | 0.56x | 3.58 / 1.46 | 0.95 / 0.40 |
| 37.75 MB | 3.793 | 7.743 | 12.765 | 0.30x | 3.51 / 1.45 | 0.94 / 0.40 |
| 150.99 MB | 12.674 | 34.743 | 29.131 | 0.44x | 3.45 / 1.41 | 0.93 / 0.40 |
| 432.00 MB | 36.580 | 75.639 | 77.166 | 0.47x | 3.43 / 1.40 | 0.93 / 0.39 |

**No - and the layout was never the reason.** Median over the nine sizes is **0.49x** for the pre-broadcast form and
0.48x for the broadcast form, and the best single size reaches 0.64x: at no size does either beat `vpermps`, in L2 or
in DRAM. Removing the broadcasts changes the median by one percent, and the counters show why - **port 5 per weight
word is 0.92-0.98 in both shuf arms against 0.39 for `b2`, unchanged by the layout change.** `vbroadcasti128` with a
memory operand is a load uop, not a port-5 uop, so those four broadcasts were never occupying the resource the arm was
trying to relieve. What remains is 2.4-2.9x the instructions (`two vpshufb` + `vpblendd` + the extract, against one
`vpermps`), and on this kernel instruction count is throughput: the 2.5x instruction ratio and the 0.49x time ratio are
the same fact stated two ways.

Two sizes (4.72 MB, 37.75 MB) put the pre-broadcast arm far behind the broadcast one - 3.0x and 1.6x - which is what
doubling the table per token can do when it stops fitting; those are also the rows with the widest rep spread here, so
the honest summary is "never better, sometimes worse", not "equal".

The only form that gets to one shuffle per eight floats puts **16-bit** entries in one lane, so eight pair sums fit in
sixteen bytes and a `vcvtph2ps` widens them: two instructions and fp16 pair sums, which cannot pass the tier's
bit-exact gate against the fp32 reference. For an *exact* fp32 pair-sum lookup, `vpermps` is the instruction, and that
closes the byte-shuffle route for good.

## A structural limit worth recording

With `vpermps` (8-lane) and 4-bit pair codes, **one lookup covers 8 rows × 2 columns = 16 weights, and that is the
ceiling for a single token.** Encoding 4 trits per field would need 7 bits × 8 rows = 56 bits, more than the 32-bit
word, so fewer rows per lookup — no net gain. Widening the index beyond 8 entries is the only way past 16 weights per
lookup, and it requires the vector-LUT form, which requires N > 1 to pay. So **further N=1 kernel polish has a low
ceiling**; the multi-token path is where the remaining 2× lives.

## What the mirror A/B says about the table

Mirror consolidation was the cheap way to ask whether the 32 KB pair table costs anything. It halves the table and
loses 40-45% of the throughput, so the answer is not "the table is too big". More evidence, from a deliberately wrong
build that pins the table pointer to one 2 KB group (so every thread reads the same hot table all token long): median
1.09x against the real kernel, with a spread from 0.60 to 1.16. That is inside this host's noise, so at most the
whole table footprint is worth about 9%, and possibly nothing.

Both results point the same way: **the GEMV is limited by instructions and stalls, not by the bytes the table moves.**
Adding roughly five uops per weight word costs 40%, and removing three quarters of the table's footprint buys 0-9%.
Any future trade of "fewer bytes for more instructions" (#9 bit-packing, #12 non-temporal loads) starts out losing.

## Batched GEMV: measured 1.35x at N=2

`rows8_b2` in `~/kt/perf/gbb.c` runs two activations through one pass over the weights. Per weight word the decode
(`vpbroadcastd`, `vpsrlvd`, `vpand`) and the zero test (`vpcmpeqd`) depend only on the codes, so both are computed once
and reused by both tokens; each token keeps its own accumulator chain and its own add order over g and p. It is
**bit-exact** against two separate `rows8` passes - 0 mismatched floats over the whole model - and both pair tables are
rebuilt per call, as production would.

Interleaved A/B, 7 reps, 8 threads, all 151 real matrices of the Bonsai-2 pack:

| | ms / 2 tokens | ms / token | instructions per vpermps | IPC |
|---|---:|---:|---:|---:|
| two `rows8` passes (today) | 58.7 | 29.4 | 8.92 | 1.64 |
| one `rows8_b2` pass | 43.6 | **21.8** | **5.92** | 1.49 |

**1.35x per token against a 1.37x prediction**, driven by 8.92 -> 5.92 instructions per lookup. IPC *falls*
(1.64 -> 1.49), so the whole win is the instruction count - which is exactly what the issue-bound diagnosis predicts.

It also goes a long way to settling the bandwidth question. Mode 1 reads each weight byte once for two tokens, so
**265 MB per token instead of 529**, and is 1.35x faster. A bandwidth-bound kernel would have gained much more from
halving the traffic and nothing at all from anything else. At 12.1 GB/s of real DRAM it is further from the 26 GB/s
ceiling than the unbatched kernel is. What that run could not show is whether the 1.35x **floor** survives when there is
no DRAM traffic at all, so it was re-run that way.

### Re-run with a cache-resident working set: the floor is instructions, not bandwidth

`gab2.c` takes the same two arms - two `rows8` passes against one `rows8_b2` pass - and makes the working set an
argument, from 0.59 MB (L2) to 432 MB (DRAM). Both kernels and the `Mat` layout come straight from `kindi/kcpu.c` by
inclusion, the batched arm is checked bit-identical to the two passes before anything is timed
(`max|sep-b2| = 0.000e+00`), and the arm order is reversed on alternate reps.

| working set | sep (2 passes) | b2 (1 pass) | ratio | instruction ratio |
|---|---:|---:|---:|---:|
| 2.36 MB (L2/L3) | 0.225 ms | 0.167 ms | **1.345x** | 1.436x |
| 9.44 MB (L3; the L3 is 16 MB) | 0.853 ms | 0.634 ms | **1.346x** | 1.522x |
| 432 MB (DRAM) | 39.18 ms | 26.75 ms | 1.465x | 1.521x |

A repeat of the same three sizes gave 1.333x, 1.401x and 1.493x, the same ordering, so read the absolute ratios as
+-0.05 and the ordering between cache and DRAM as the finding.

A 180x reduction in working set does not move the floor: **1.345x cache-resident against 1.351x on the whole 529 MB
model.** A bandwidth explanation would have to change between those two regimes, and the value a true bandwidth limit
would produce is the 2.000x that two passes over the same bytes cost - nothing measured here is near it. The
instruction ratio is 1.44-1.52 in every configuration, and achieved bandwidth was 18.7-19.7 GB/s at these three sizes
against a 26 GB/s ceiling, so none of them came close to saturating memory. (The 0.59 MB row of the full sweep reads
15.8 GB/s, but a pass there is 66 us and the OpenMP region cost is part of it - that row is not a bandwidth
statement.)

Memory is not irrelevant, and the DRAM row shows where it matters: 1.465x against 1.346x cache-resident, so halving
the byte traffic is worth about 9 points of ratio once the stream is real. That is a contributor, not the floor. The
cache-resident rows sit *below* their own instruction ratio (1.345 against 1.436-1.522), which says that with data
supply removed something else limits the batched kernel. The first reading of this was ILP - two accumulator chains and
two table streams per weight word - so it was tested by varying the chain count on the same sweep; the answer is below,
and it is not ILP.

**Now wired up, and measured end to end.** Items #4 and #5 turned out to be one change: the kernel needs two tokens,
and only speculative verification produces them. That is implemented - `kc_step2` in `kindi/kcpu.c` runs two tokens
through the whole forward pass on one weight stream, and `kindi.cpu.generate_spec` drives it from an n-gram
drafter. The details and the tables are in [BENCHMARKS.md](BENCHMARKS.md#two-token-speculative-decode-on-the-cpu-i9-9980hk-8-threads);
the short version:

- **Bit-exact.** The pass matches two sequential `kc_step` calls to `0.000e+00` on the logits and leaves `M`, conv, K
  and V `array_equal`. Greedy output is identical token for token.
- **A two-token pass costs 1.351x one token**, so the ceiling is 1.48x at perfect draft acceptance, and break-even is
  an acceptance rate of 0.35. Measured +1.18x at 0.86 acceptance and +1.53x at 1.00.
- The ceiling is 1.48x and not 2x because only the products batch. The RMS folds, the GDN recurrence, the attention
  bodies and the embedding still run per token, and they double.
- **In the serving loop, off by default, and measured there.** `Runner.spec_begin/spec_next/spec_stop` put the pass in
  `server.py`'s token loop for greedy requests; a request that stops on the token of an accepted pass rolls back with
  `kc_state_load`, so the recurrent state and `Runner.tokens` end at what one `step` leaves. `KINDI_CPU_SPEC=1`
  enables it, and `kindi spec-check` is the test (accepted draft stopped on by token id and by stop string, stopped one
  token later, all drafts rejected, and the server's own loop with the pass off and on). End to end through the server
  on the real pack it is 0.93-1.02x at 269-273-token prompts and 1.26-1.37x at 65-token prompts. The pass costs ~1.5x
  a step on this host today rather than 1.351x - `bench-spec` re-run the same day gives 1.05x at 0.764 acceptance, and
  `step` against `step2` gives 1.49-1.57x - so break-even acceptance is 0.5. `BENCHMARKS.md` has both tables. A
  thinking budget closes the thinking mid-stream with tokens the loop feeds one at a time - those requests do not
  speculate.

The earlier reading of this - that batched verification is impossible because 18 of 24 layers are recurrent - was
wrong, and the mistake is worth recording. The recurrence constrains the *state*, not the *products*: a layer can take
both tokens through its GEMV at once and then step `M` and `conv` in token order, which is exactly how `spec.py`
already does it on the GPU ("the GDN steps in token order inside one launch"). What the recurrence does rule out is a
long verify batch with no per-token state work, not batching itself.

### The cache-resident shortfall is not ILP: measured

`~/kt/perf/gab2_ilp.c` keeps `gab2.c`'s harness, sizes and bit-exactness check and adds arms that differ only in how many
independent accumulator chains the same per-word instruction mix is spread over. Every arm is verified against two
`rows8` passes (`max|sep-arm| = 0.0e+00`) before anything is timed; the 4-chain arm is a timing probe only, because
collapsing the even/odd halves changes the summation order.

| arm | chains | instructions per pass (2.36 MB) | 2.36 MB (L2) | 9.44 MB (L3) | 37.7 MB | 432 MB (DRAM) |
|---|---:|---:|---:|---:|---:|---:|
| two `rows8` passes | - | 1.14 M | 0.192 ms | 0.774 ms | 3.355 ms | 38.30 ms |
| `rows8_b2` (production) | 8 | 0.76 M | 0.152 ms | 0.605 ms | 2.556 ms | 26.93 ms |
| 4 row blocks in flight | 16 | 0.86 M | 0.152 ms | 0.607 ms | 2.462 ms | 25.80 ms |
| two group windows | 8 | 0.76 M | 0.152 ms | 0.591 ms | 2.522 ms | 27.25 ms |
| one accumulator per row (probe) | 4 | 0.80 M | **0.144 ms** | **0.576 ms** | 2.391 ms | 25.69 ms |

**Adding chains changes nothing, and removing them does not hurt: the axis is flat to slightly negative.** 8 to 16
chains is 0.0% and +0.3% at the two cache-resident sizes and 4% faster at the two larger ones, on an arm that also halves
the table loads per word and retires 13% more instructions - sign and size both inside the run-to-run spread, so there is
no effect to report. Opening a second group window is -2% to +1%. The 4-chain probe keeps the production loop and instruction mix
(single accumulator per token and row block instead of an even/odd pair) and is **5% faster at every size**: 0.144 against
0.152 ms cache-resident, 25.69 against 26.93 ms at DRAM. If the kernel were waiting on chain latency, that arm would be
the slow one. It is not, so the cache-resident shortfall against the instruction ratio is not ILP and not the load count
- what is left per weight word is the work itself, two `vpermps` per word, one per token per row block, on the one port
that issues them.

The 4-chain form is not adopted: it sums the column pairs in a different order, so it would no longer be bit-identical to
two single-token passes, and that exactness is what makes `kc_step2` safe to drop into a token loop. Five percent does not
buy that. The measurement is what matters here: the production kernel is at the top of its chain-count curve, not below
it, so no amount of extra parallelism is available to recover.

Residency buys the two-token kernel very little either: 13.8 GB/s of weight stream at 2.36 MB and 9.44 MB, 13.1 at
37.7 MB, 14.3 at 432 MB. The cache-resident case is not faster per byte than the DRAM one, and both are around half of
the 26 GB/s this host can stream, so no row here is waiting on data supply.

(`./gab2_ilp <threads> <iters> <row blocks> <reps>`, 15 reps each here: `8 80 512 15`, `8 30 2048 15`, `8 12 8192 15`,
`8 5 93750 15`, after a first pass at 9 reps gave the same ordering. This session's sep/b2 ratios are 1.28x, 1.29x,
1.32x and 1.43x, against 1.345x / 1.346x / 1.465x earlier - the absolute times move with the host, the ordering does
not.)

## The vector LUT: one lookup, N tokens

The element-wise LUT above failed because it changed the *selection*: an 81-entry table needs a gather, and a gather is
several uops where `vpermps` is one. The vector LUT changes the other end - it keeps the pair table and the `vpermps`,
and makes the thing a lookup returns wider:

```
  rows8_b2  one token a table     TT[p][c] = fp32 (8 rows x 2 columns)     16 weights / vpermps

  vector LUT  the batch in the entry
            TT[p][c] = int16 A | int16 B << 16   in one int32 lane
                      |         |
                      |         +-- token B's partial sum for code c
                      +------------ token A's partial sum for code c

            8 codes = 8 int32 = one vpermd source, so one lookup returns
            8 rows x 2 columns x 2 tokens = 32 weights / vpermd
            and at N = 4 two tables (A,B) and (C,D) return 64 over 2 lookups
```

The int16 half is the price and it is not optional: two tokens fit in one 32-bit lane only because the sums are int16,
so this is an int8-activation kernel - the trade `rows8_q8` already makes, one lookup wider. int16 is enough for one
128-column group exactly: the worst case is 128 activations of 127 against all-+1 weights, 16256 against 32767, a
measured 2.0x headroom (the selftest builds that pack and checks it). The table is built per group into 2 KiB of stack
and consumed by every row block before the next group replaces it, which is the tiling the Vec-LUT paper
([arXiv 2512.06443](https://arxiv.org/abs/2512.06443)) reaches from the other end: its entry is an N-element int16
vector chosen by a scalar index, this one is a per-row int16 vector chosen by `vpermd`, and both win by making a
lookup's output wider than one token.

`kindi/kcpu.c` (`vlut2`, `vlut4`, exported as `kc_vlut_gemv`; `kc_b2_gemv` is the same `rows8_b2` for the other arm),
measured by `python -m tools.vlut.measure` - both kernels single-threaded on the same pack and the same tokens, table
build inside each arm's timing, ratios from interleaved A/B (median of per-round ratios, IQR around it):

| matrix | N | `rows8_b2` (min) | vector LUT (min) | LUT time / b2 | GMAC/s b2 -> LUT |
|---|---:|---:|---:|---:|---:|
| 6144 x 2048 (12.6 M weights) | 2 | 1.20 ms | 0.95 ms | **0.88x [0.85-0.94]** | 21.0 -> 26.4 |
| 6144 x 2048 | 4 | 2.41 ms | 1.24 ms | **0.58x [0.56-0.62]** | 20.8 -> 40.6 |
| 1024 x 2048 (2.1 M) | 2 | 0.22 ms | 0.18 ms | **0.85x [0.81-0.86]** | 19.3 -> 23.1 |
| 1024 x 2048 | 4 | 0.38 ms | 0.23 ms | **0.55x [0.52-0.57]** | 22.0 -> 36.7 |
| 256 x 512 (0.13 M) | 2 | 0.022 ms | 0.012 ms | **0.53x [0.52-0.54]** | 11.3 -> 21.8 |
| 256 x 512 | 4 | 0.040 ms | 0.020 ms | **0.49x [0.46-0.49]** | 15.0 -> 31.2 |

Read the N = 2 rows for the lookup width - both arms make one pass over the weights there, so the ratio is the
`vpermd` returning two tokens instead of one. Read N = 4 for batching: `rows8_b2` walks the weights twice, the LUT once.

What the numbers say, including the part that is not a win:

- **The lookup width pays at every size here, and most where everything is in cache.** 1.9x at 0.13 M weights, 1.18x
  at 2.1 M, 1.14x at 12.6 M (3 MB of pair codes, past this host's L2) - this run's spreads were tight (2-8%).
- **The N = 2 size of that win does not survive repeats, only its sign.** Four runs of the same code on this host gave
  0.53-0.55 at 0.13 M, 0.70-0.85 at 2.1 M and 0.88-0.95 at 12.6 M, and one of them put the large-matrix row inside its
  own noise ([0.87-1.08]). The LUT was faster at N = 2 in every run and faster still at N = 4 in every run; how much is ±0.1 and
  the large-matrix column should be read as "the lookup is no longer the thing being waited on". 32 weights a lookup
  buys 14-18% once the weight stream shares the budget, not 2x, because at N = 2 this kernel is front-end-bound on the
  decode and the table stream rather than on `vpermd` alone.
- **At N = 4 it is 1.7x, and half of that is not the lookup.** `rows8_b2` at N = 4 is two passes (2.41 ms against
  1.20), the LUT is one, so the shared weight stream and the shared decode are part of the win. 24.3 GMAC/s at N = 2
  against 40.6 at N = 4 is what a wider batch buys even at a fixed lookup width - and it is the number that matters
  for a prefill, where the batch is real.
- **It is not exact.** Against the fp32 kernel on the same tokens the int8 activations cost 6.4-6.6e-3 relative L2, and
  against an int64 evaluation of its own quantised input the kernel is good to 5e-8 - so the kernel is exact and the
  quantiser is the error. That is `rows8_q8`'s trade, and the exactness gate in this repo would refuse it as the
  default tier for the same reason it refuses q8.

It is **not wired into the forward pass**. Doing that needs int8 activations for two or four tokens inside `gemv2` (a
second `Q`/`QS` pair), and it changes the numerics of the two-token pass, which is the path speculative decode leans
on for bit-exactness with a single-token run. The measurement surface is there (`kc_vlut_gemv`), the decision the gate
would have to make is separate, and this section is the number that decision needs.

## Column tiling is blocked on the pack layout

Column-outer tiling would make the table L1-resident, but each thread would then read 256 B every 4 KB of the weight
matrix. That destroys DRAM row-buffer locality, so it likely loses more than it gains. The hot-table measurement
above weakens the case further: making the table resident at all is worth single-digit percent, which will not pay
for a pack-format change.

The fix is the one T-MAC uses: permute the weights so the **column group is the outer dimension**,
`[group][row block][64 words]`, which makes column-outer access sequential again. That is a pack-format change —
`kindi/pack` in Python, the layout notes in `docs/HOW-IT-WORKS.md`, and the kernel together — not a kernel tweak.
Worth doing only alongside #4.

## MoE and Qwen support: scoping

Neither exists in the CPU engine today, and the economics decide whether they are worth building.

- **Qwen support** needs, beyond what the 2B path has: GQA with `NHKV < NH` (the attention math already carries
  `NHKV`, so this is mostly plumbing), and the usual RoPE/SwiGLU that Bonsai's hybrid GDN layers already prove out.
- **MoE support** needs three things: a router matmul + softmax + top-k, an expert-indexed weight table, and a gather
  of the selected experts' rows per token.

The uncomfortable arithmetic, which is why MoE is *not* first on the list:

| model | active params | bytes/token at 1.7 bpw | vs Bonsai-2 (0.43 GB) |
|---|---|---:|---|
| Bonsai-2 dense | 2B | 0.43 GB | — |
| Qwen3-30B-A3B | ~3B | ~0.64 GB | **1.5× worse** |
| a 20B-A1B style MoE | ~1B | ~0.21 GB | ~2× better, but ~4.3 GB resident |

So "add MoE" is only a speed feature for a **low-active-parameter** MoE whose resident set still fits. And for a
ternary engine specifically, MoTE's lesson applies: ternarise the routed experts, keep the shared expert dense, or
pre-trained knowledge is destroyed.

Truly reported CPU-only MoE decode numbers were not found — the published systems (Apex, KTransformers) are GPU+CPU
hybrids that treat the CPU as the slow path. Treat a CPU-only MoE throughput claim as unverified until measured here.

### What landed, and what it costs

The MoE FFN is in `kcpu.c` and in the engine: a router GEMV, a softmax over the experts, the `nact` highest with their
weights renormalised, and 3 matrices per expert (gate, up, down) in the same ternary form as a dense FFN. Two details
the pack forced:

- **Every Mat is padded to a multiple of 8 rows**, so the router's output buffer has to hold the padded count, not the
  expert count. This was a heap overflow until `kc_moe_check`'s fixture found it (ASan, 4 experts against an 8-row
  buffer).
- **The expert's own width is not `NFF`.** `moe_expert` takes the width from the matrix (`experts[3e].R`) and folds
  with the expert-width sign vector (`L->moe.sF`), so an expert can be narrower than the dense FFN, as real MoE models
  have it.

Correctness is checked against a NumPy implementation of the same block (`kindi moe-check`), on the synthetic fixture,
router and expert arithmetic both: **1.9e-7 to 5.6e-7 relative** output difference over 3 layers x 2 trials with 4
experts and 2 active, and 2.4e-7 to 4.4e-7 with 8 experts and 3 active. The same test caught the second bug: the
first expert's fold overwrote the block input, so the second expert was reading the first expert's intermediate. Each
expert now starts from a fresh fold of `xn` - one FWHT per expert, which is nothing against that expert's products.

One limitation is structural rather than a bug: **a MoE layer gives up the two-token pass.** The two tokens can route
to different experts, so `kc_step2` runs each token's experts in turn. `verify-tier` shows the two-token path still
matches two single steps, i.e. it is correct and not faster on those layers.

### Paged experts: the pair-code pack makes staging a memcpy

An expert is a `Mat` like any other, so paging it needs no kernel change: the layer's expert table can point into a
staging buffer, and a fetch hook refills that buffer before the expert's products run (`moe_fetch`, wired from
`Engine.enable_paging`). The pool is a per-layer cache of `k` experts in pair-code form, loaded on demand from the
pack's own bytes.

Measured first on the machine that wrote the rest of this file, on a fixture of 6 layers, width 1024, 12 experts of
width 512 (2 active), 6 tokens after a 2-token warm-up, 8 threads: **1.5 ms/token with all 12 resident, 140.2 with a
cache of 6, 174.9 with 4, 742.1 with 2** - every size producing the resident engine's tokens exactly. The mechanism
was right and the staging was not: one expert's three matrices took 7 ms to rebuild through the NumPy pair-code
conversion, and a token that missed two experts in six layers paid for twelve of them. The conclusion was that **a
paged expert needs its pair codes on disk**, and that is now a pack variant rather than a note.

`kindi convert --paircodes` (and `synthpack(paircodes=True)`) writes `pcodes.bin`: for each routed expert matrix, the
GEMV words and scales `pair_mat` would build, so a miss is two memcpys instead of a conversion. It costs 0.22 MB for
the fixture's 256 x 1024 expert - 0.28 bytes a weight against the 0.266 its ternary codes and scales already take, so
1.06x the pack's expert bytes - and covers the routed matrices only, so a dense pack pays nothing. `Engine.enable_paging` maps it when the file is present, and checks its length against
every expert it would stage, so a stale file is refused rather than half-filled.

Re-measured on the CPU VM in this document, median of three runs, same fixture and 6 tokens after the same warm-up:

| experts resident | convert ms/token | memcpy ms/token | hook only | no fetch | misses/token | expert code memory |
|---|---:|---:|---:|---:|---:|---:|
| all 12 | **5.1** | - | - | - | - | 14.3 MB |
| cache 6 | 251.2 | **23.0** | 30.6 | 3.8 | 13.2 | 7.6 MB |
| cache 4 | 303.7 | **24.7** | 22.9 | 4.1 | 15.8 | 5.1 MB |
| cache 2 | 408.3 | **30.6** | 30.2 | 7.3 | 20.8 | 2.5 MB |

The absolutes are not comparable to the paragraph above - this VM is slower than the machine that produced 1.5 - so
read the columns against each other. The spread is wide: across five runs the memcpy column spanned 19-31 ms and the
hook column 15-36. Two of the columns are controls that stage nothing at all: `hook only` keeps the fetcher and empties
it, `no fetch` clears `moe_fetch` so the pool's slots are never refilled. What they show is that **the copy has stopped
being the cost**: the memcpy and hook columns swap order between runs (23.0 against 30.6 in the table, 21.8 against
24.2 and 19.3 against 17.9 in others), and 13-21 misses of a 0.2 MB expert at the 0.05-0.3 ms a copy reproduces at add
up to 1-6 ms of the 15-19 ms paging costs above the resident path.

The rest is the crossing: **69 `moe_fetch` calls a token, ~0.3 ms each.** It is a Python function called from inside
the OpenMP region, so every selected expert costs a GIL acquire and a thread wake - and the conversion column pays for
that same crossing before it pays 12 ms a conversion. What keeps experts resident is no longer staging; it is the
callback. Removing it means staging in C (a `kc_stage(layer, expert)` that reads `pcodes.bin` itself), which hands the
kernel a file handle and the slot pointers - an ABI change worth making when a model needs paging, not before.

So the default stays `expert_resident=None` (all resident), and the honest summary is that the variant turned paging
from ~50x the resident cost into ~5x. One caveat: only the MoE fixture exercises it. Bonsai has no experts and no
shipped GGUF produces `pcodes.bin`, so the first real MoE pack is where 0.28 bytes a weight and the copy throughput
get their first real test.

## The element-wise LUT: measured 3.4-4.0x slower than the pair table

The triage above says two things at once: the pair table is issue-bound but cheap (one `vpermps` per 8 rows x 2
columns), and the element-wise family (#1 int16 LUT, #2 `3^g` index) is blocked because `vpermps` selects 8 lanes.
What was never measured is whether an element-wise table that *can* work on AVX2 - a gather from an 81-entry table,
one lookup per 8 rows x 4 columns - actually beats the pair table. It does not.

`~/kt/perf/lut81.c` runs both arms over identical weights: the pack's 4-trit encoding, the same 128-wide groups, the
same per-row scales, `rows8` verbatim for the pair arm, and for the element-wise arm a table rebuilt from the pair
table (index `cA + 9*cB` for a slot's two words) and one `vpgatherdd` per 8 rows per 4 columns. 2048 x 1920 ternary
weights, both arms agreeing with a scalar reference to 1.5e-7 (the LUT arm sums its slots in slot order, so it is equal
to rounding, not bit-equal).

| arm | ms | GB/s | table per call |
|---|---:|---:|---:|
| pair (`rows8`, `vpermps`) | **0.18** | 10.9-11.2 | 7,680 floats (30 KB) |
| element-wise 81-entry (`vpgatherdd`) | 0.59-0.72 | 2.7-3.3 | 38,880 floats (155 KB) |

**3.4x to 4.0x slower**, single-threaded and 8-threaded alike, with both arms' table builds inside their timing. The
reason is the gather: on Coffee Lake a 8-element `vpgatherdd` is a multi-uop instruction with a latency measured in
cycles per element, against a `vpermps` that is one. The extra table footprint (155 KB against 30 KB, so L2 rather than
L1) and 5x the build work compound it.

That settles the family, and settles `3^5` without building it: a 243-entry table needs a different pack (5 trits per
byte, the GGUF's 26-byte block layout) *and* more table per column, so it would be strictly worse than the 81-entry
form measured here. The 4-trit pack's 81 entries are the most favourable case the gather could have had.

What it does not settle is a vector LUT, which is a different family: it keeps the pair table and the `vpermps` and
makes the *output* wider instead of the index, so nothing is gathered. That one is measured further down - see
[the vector LUT](#the-vector-lut-one-lookup-n-tokens): 1.7x at N = 4, 1.1-1.9x at N = 2.

## Kernel tiers, and the native Windows build

`kcpu.c` now carries three GEMV implementations and a gate that picks the fastest one which reproduces the others:

- **tier 0, scalar C** - no intrinsics at all, written to the same operation order as the vector kernel (same table
  entries, the same even/odd partial sums, and `fmaf` because the vector path uses `vfmadd`). That is what makes an
  exact comparison meaningful instead of approximate, and it is the fallback for a build without AVX2.
- **tier 1, AVX2** - the production pair-table kernel.
- **tier 2, int8 activations** - the prototype, expected to differ.

`Engine._pick_tier` runs a one-token probe at load and refuses a tier that moves the logits; `KINDI_CPU_TIER`
(`auto`/`scalar`/`avx2`/`q8`) overrides it. `kindi verify-tier` is the full statement of the claim - 3 prompts, 24
tokens, greedy tokens, recurrent state and the two-token pass:

| tier | prompts identical to scalar | two-token vs two steps | ms/token (fixture) |
|---|---:|---|---:|
| scalar | 3/3 | tokens, state and logits equal | 0.757 |
| avx2 | 3/3 | tokens, state and logits equal | **0.234** |
| q8 | 1/3 | state and logits differ | 0.240 |

The scalar tier is bit-exact with AVX2 on the first try, including the two-token pass, which is the useful part: a
fallback that agrees bit for bit is a reference implementation, and the int8 tier is now rejected by measurement
rather than by argument.

**The Windows build.** `kindi/cbuild.py` builds `kcpu.dll` with MSVC (`/O2 /arch:AVX2 /openmp /LD`) out of a Visual
Studio install, and `kindi.cpu` loads it exactly as it loads `libkcpu.so`. Three things had to change for the compiler:
OpenMP loop variables declared outside the `for` (MSVC's classic `/openmp` is C89-style), `__declspec(dllexport)` on
the entry points, and the integer fp16 conversion instead of `_cvtsh_ss`, which MSVC does not link. The same fixture on
the same host through both builds:

| build | scalar | avx2 | q8 | tiers agree |
|---|---:|---:|---:|---:|
| gcc (WSL) | 0.757 | 0.234 | 0.240 ms/token | yes |
| MSVC (Windows) | 1.080 | 0.449 | 0.777 ms/token | yes |

Within each build the tiers are bit-exact. **Across the two builds they are not**: the same 12 greedy tokens, the same
argmax, but 0 of 128 logits bit-identical, worst relative difference 7.9e-5. That is the C runtime, not the kernel -
different `expf`/`log1pf` in glibc and the MSVC CRT, plus the fp16 conversion - and it is worth recording as the honest
version of "bit-exact": exact within one build, token-exact across builds. The MSVC numbers are also slower, on a
tiny fixture where fixed costs dominate (16 threads on 3 layers).

## An AVX-VNNI / int8 dispatch path: scoping

Not buildable end to end on this host - Coffee Lake has no VNNI - so `vpdpbusd` itself still cannot run. But the half of
the design that the kill criterion actually rests on is **plain AVX2**: the 2-bit to int8 expansion is mask, shift and
`vpshufb`, and it can be counted here with the same hardware counters used everywhere else in this document. It has now
been counted; the result is below.

Three facts shape the plan, checked against the code and the ISA:

- `paircodes()` in `kindi/cpu.py` builds the kernel layout **at load time**, and `convert.py` has never known about it.
  A second layout is therefore a new Python repack plus an ISA check, **not** a `pack_format` change; the shipped pack
  stays as it is.
- There is no dispatch today: one AVX2 path, no `__builtin_cpu_supports`, no multiversioning.
- The kernel is deliberately exact - "Every pair sum is formed once in fp32; no activation quantisation". VNNI needs
  int8 activations, so the token-exactness guarantee (40/40 against llama.cpp) is what is actually at risk.

Byte arithmetic rules out the obvious form. int8 weights are one byte, 4x the 2-bit pack: 529 MB becomes 2.1 GB, or
81 ms of DRAM at 26 GB/s against today's 40 ms token. **Weights must stay at 2 bits on disk and be expanded to int8 in
registers.** That expansion is the entire question.

One trick makes the cheap ISA sufficient: `vpdpbusd` is u8 x s8, and the signed x signed `vpdpbssd` of AVX-VNNI-INT8
only arrived with Sierra Forest, not Alder Lake. But store weights as `w+1` in {0,1,2} and
`sum (w+1)x = sum wx + sum x`, so the correction is one scalar per token subtracted from every output. Plain AVX-VNNI
is then enough.

| step | work | risk |
|---|---|---|
| ISA detect + dispatch (`__builtin_cpu_supports("avxvnni")`, `KINDI_CPU_ISA` override) | ~0.5 day | none |
| second load-time repack, with a Python reference to diff against | 1-2 days | none |
| `rows_vnni`: expand 2-bit to `{0,1,2}` via `vpshufb`, `vpdpbusd`, int32 accumulate, requantise | 3-5 days | instruction cost now measured (3.31); port 5 on real silicon is the remainder |
| int8 activation quantisation per 128-block on top of the fold | 2-3 days | changes numerics |
| token-exactness against llama.cpp | **measured: PASSES 40/40, and 640/640 over 160 tokens** | no longer a risk; see below |

## The kill criterion, measured

`rows_vnni` was built as far as this host can run it: the real expansion, with the int8 dot stood in by
`vpmaddubsw` + `vpmaddwd` + `vpaddd` (three instructions where VNNI needs one). Subtracting that difference gives the
VNNI projection. Harness: `~/kt/perf/gbx.c`, 21 interleaved reps, medians, net of a loads-and-loop baseline.

Two checks run before any timing, and both pass:

- **Layout.** 128 weights per 32-byte input, every one exact against a scalar reference.
- **The `w+1` identity.** `sum (w+1)x - sum x` reproduces the signed product exactly, so plain AVX-VNNI really is enough.

The pack's atom matters here and it works in the design's favour. `paircodes()` stores one **4-bit code per unit-sum
pair** - "0..7 = which of the 8 nonzero (w0, w1) pairs, 8 = both zero; 2 bits per weight" ([kcpu.c:73](../kindi/kcpu.c#L73))
- so a nibble already carries two 2-bit weights and a 16-entry `vpshufb` table resolves both halves without any separate
unpacking step. Two tables (first weight, second weight) per nibble position, no extra instruction.

| per 128 weights | instructions | note |
|---|---:|---|
| loads + loop (baseline) | 2.25 | subtracted from everything below |
| expansion, net | 10.25 | 8 expected: 1 load, 2 `vpand`, 1 `vpsrlw`, 4 `vpshufb` |
| expansion + AVX2 dot, net | 21.25 | 19 expected; the 2.25 excess is compiler overhead |

| per 32 MACs | value |
|---|---:|
| **VNNI projection** | **3.31** |
| theoretical floor (if the compiler emitted exactly 19) | 2.75 |
| `rows8` today (8.92 per `vpermps`, 16 MACs each) | 17.84 |
| ratio | **5.39x fewer instructions** |

**Kill criterion `<= 4 instructions per 32 MACs`: MET, at 3.31.** The instruction half of the gate this document set
before writing any of the kernel is cleared with about 17% of margin, and it is cleared by the AVX2 part alone, which
means it held on the machine that cannot even execute the instruction it is scoping.

A supporting observation, measured on the same buffer: the expansion-plus-dot loop runs at **2.31 cycles per 16 MACs**
(IPC 1.26). `rows8` today needs about 5.44 for the same 16 MACs - 8.92 instructions at the IPC 1.64 measured earlier on
this host, which is a derivation rather than a fresh measurement. Even carrying the three-instruction dot stand-in, the
shape is already ~2.4x ahead.

**Port 5, the one thing still unresolved.** `vpshufb` and `vpdpbusd` are both `p5` uops on Alder Lake. Per 128 weights
that is 4 + 4 = 8 port-5 uops, exactly one per 16 MACs - the *same* port-5 cost as today's single `vpermps` per 16 MACs.
So the expansion does not add port pressure per MAC, and `rows8` currently uses only ~18% of port 5, which is why it is
issue-bound. A raw `UOPS_DISPATCHED_PORT.PORT_5` event opens under WSL2 but reads 0, so this stays analytic: real silicon
must confirm it, but the arithmetic says port 5 becomes co-binding rather than the new bottleneck.

It also composes with batching rather than competing with it: the expansion is per weight word, so it is shared by
every token in the batch, and its cost per token halves at N=2. An int8 path would want the batched structure anyway.

**Kill criteria.** The instruction-count gate is cleared above. The exactness gate - "if `kindi verify` cannot be made to
pass 40/40 within about two days of numerics work, stop" - passes on its own four prompts and **fails on a wider corpus**:
48 prompts cross it 13 times, always at decisions the fp32 model itself had left open (measurements in the next section).
By its literal terms the gate is cleared; as a claim about token-exactness in general it is not. Port 5 still needs real
Alder Lake.

## int8 activations: the exactness gate, measured

The open question was whether quantising activations to int8 destroys the token-exactness guarantee. It was settled on
this host, **without SDE**, because SDE is not installed and would not have helped: `vpdpbusd` is an integer
multiply-accumulate, and `vpmaddubsw` + `vpmaddwd` + `vpaddd` produce the *identical* integers. SDE emulates the
instruction; it does not change the value of a dot product. Only the **instruction count** needs VNNI silicon, and that
was already counted. So the numerics can be prototyped and diffed against llama.cpp on plain AVX2, which is what was
done.

`KINDI_CPU_Q8=1` (in `kcpu.c`, off by default, prototype only) routes every GEMV through `rows8_q8`: the folded input is
quantised to int8 symmetrically per group of 128 columns (`scale = max|x|/127`, `KINDI_CPU_Q8G` to change the group
width), each weight is used as its signed ternary value, products accumulate in int32, and each group total is rescaled by
`activation scale * row scale`. That is exactly the arithmetic a `vpdpbusd` kernel would produce; only the instructions
differ.

| check | result |
|---|---|
| `kindi verify`, fp32 control | PASS 4/4, 160/160 tokens |
| `kindi verify`, int8 activations | **PASS 4/4, 160/160 tokens** |
| 160 tokens per prompt, both paths | **640/640 tokens identical**, no divergence |
| per-logit drift, int8 vs fp32 | median 0.0115, p99 0.050, max 0.219 |
| per-logit drift, 8-prompt recheck (below) | median 0.0107, p99 0.047 |
| argmax agreement at 40 tokens | 160/160 positions |
| margin eaten by quantisation | median 0.1%, p90 2.3%, **max 73%** |

The gate passes on its own corpus, and with headroom rather than by a single lucky token: the token sequences are
identical for 160 tokens on all four prompts. But four prompts was the wrong size to draw the conclusion from, and the
wider sweep below crosses it. **This is robustness, not exactness.** The int8 arithmetic is not bit-identical to the fp32
path and never will be: logits move by up to 0.22, and the median |delta| of 0.0115 is comparable to the width of a narrow
top-1/top-2 gap.

Harness: `~/kt/perf/q8/` (`kq8_wide.py` runs the sweep below, `kq8_cmp.py` and `kq8_flips.py` compare it; one engine per
process, because two 506 MB packs in one process corrupt the heap). The gate itself is reproducible without it:
`KINDI_CPU_Q8=1 kindi verify`.

The margin column is the honest measure of fragility, and on 640 decisions it looked comfortable: the quantisation ate a
median of 0.1% of the winner's lead, at worst **73%** of it, and 3 of 640 positions lost more than half. A flip needs
>100%. That 73% turned out to be the largest survivor of a corpus that never got close enough, not a bound.

### The wider sweep: 13 of 48 prompts diverge

48 prompts - factual answers, code in four languages, maths, translation, JSON, SQL, shell, enumeration, sentiment and
extraction, style imitation, open-ended story text - at 160 greedy tokens each, so **7,680 decisions per arm**.

| quantity | value |
|---|---|
| prompts whose token stream diverges | **13 of 48 (27%)** |
| aligned decisions compared | 6,371 |
| margin eaten, aligned decisions | median 0.02%, p90 1.6%, p99 16.3%, p99.9 64.5%, max **76%** |
| flips at a decision above the corpus p1 fp32 margin (0.061) | **0 of 13** |
| fp32 margin at a flip | median 0.016, max 0.057 |
| four flip prompts x 24 tokens, each arm run twice | identical per arm, so the divergence is arithmetic, not a race |

Every one of the 13 flips happened where the fp32 model itself was undecided: below the 1st percentile of its own margins
(0.061), against quantisation noise of 0.0115 median and 0.05 at p99. Above roughly a 0.06 lead, nothing flipped anywhere
in the sweep. Two shapes of divergence, both read off the raw token ids:

- the runner-up is promoted for one token and the model rejoins the fp32 stream immediately - `List three uses of a hash
  table:` gives fp32 `13 2972 6169` and int8 `13 220 2972 6169`;
- the streams part for good - `Once upon a time, in a village by the sea, there lived` parts at token 10 and stays apart.

So **the gate as written is cleared, and the property it stands for is not.** `kindi verify`'s four prompts are confident
ones, which is why the int8 path passes them 40/40; a real workload meets near-ties constantly, and there 27% of 160-token
generations diverge. VNNI cannot become a silent default on the strength of the 40/40 result. It needs either a guard -
keep the fp32 path for decisions whose margin is under the drift, which is affordable because the head is a single GEMV
and the cheapest place to check - or a tighter quantisation scheme. The next section scores the second option against the
same 7,680 decisions.

### Finer schemes: per-64 and per-32 groups, and a per-channel input scale

`KINDI_CPU_Q8G` sets the activation group width (128 columns by default) and `KINDI_CPU_Q8C=1` switches the int8 path for
a calibrated per-channel scale applied to the ordinary fp32 GEMV, so each arm changes one thing. Every arm below is the
full 48 prompts x 160 tokens, compared against an fp32 arm that came out bit-identical to the one the earlier study used,
and the per-128 arm reproduces that study's int8 arm bit for bit - the same 7,680-token stream and the same logits
(`d96ba5fc5104db88`), since the group form at 128 columns is the old code with the group width made a parameter. That is
what makes the widths comparable at all.

| arm (scale per ...) | prompts diverging | differing steps | flips above the p1 margin | margin eaten: median / p99 / max | logit drift: median / p99 |
|---|---|---|---|---|---|
| fp32 | - | - | - | - | 0 / 0 |
| 128 columns (control) | 13 of 48 | 1136 | 1 | 0.02% / 16.3% / 76% | 0.0107 / 0.047 |
| 64 columns | 12 of 48 | 1199 | 1 | 0.00% / 15.1% / 96% | 0.0103 / 0.060 |
| 32 columns | 16 of 48 | 1149 | 0 | 0.00% / 14.2% / 98% | 0.0094 / 0.056 |
| per channel, calibrated | 38 of 48 | 4305 | 32 | 0.30% / 56.7% / 98% | 0.0689 / 0.451 |

**The near-tie flips do not disappear. The group width barely moves the drift, and the drift is what sets the count.**
128 to 32 columns is a 4x finer grid for a 12% smaller error: the scale is `max|x|/127` over the group, and the folded
input has nearly the same magnitude in every column, so the largest of 32 samples is only about a quarter smaller than
the largest of 128. The flips land at fp32 margins of the order of the drift and nowhere else - median margin at a flip
0.0159 (128 columns), 0.0125 (64), 0.0094 (32), against drifts of 0.0107, 0.0103 and 0.0094 - so cutting the step by 12%
moves the number of prompts that hit such a decision by one or two in either direction, which is the noise band of a
13-of-48 measurement, not an improvement. The census says the same thing from the other side: in the fp32 arm, 17 of
7,680 decisions - spread over 15 of the 48 prompts - have a top-1/top-2 margin under 0.01, the scale of the drift, and
the 128-column arm diverges on 13 prompts.

**A per-channel scale is worse, by 6.5x, and for a reason that is not fixable by calibration.** The step is fixed for a
column by the largest value it ever showed during calibration (6,300 tokens of prose and code outside this sweep), so a
token whose values are below the average of the calibration corpus is rounded on a grid that is too coarse for it. The
dynamic group form has no such problem: its step follows the token in hand. Measured, per-channel drift is 0.0689 median
against 0.0107, and 32 of its 38 flips land on decisions where fp32's own lead was 0.05 or more - it flips confident
decisions, which the group form does at most once in the whole sweep (fp32 margin at its worst flip: 1.72, against 0.057
for the group arms).

One implementation detail is worth recording because it looked like a scheme result at first: the first per-channel arm
keyed its table by GEMV width alone, and five call sites share the 2048-wide table (GDN qkv/gate, ssm_out, attention
qkv, attention o, dense up) whose calibrated maxima range from 5.07 to 9.08, so the smallest input was quantised on the
grid of the largest. That version measured 2.85% median margin eaten and 40 of 48 prompts diverging. Keying by call site
plus width - `kc_cal_*`, calibrated to six entries - gives 0.30% and 38 of 48. The key matters by 10x; the conclusion does
not change.

Finer groups are not free either, if only slightly: the three arms are the same 5 threads on the same 7,680 tokens, and
they took 1,785 s (128 columns), 1,817 s (64) and 1,846 s (32) - 1.7% and 3.4% more scale multiplications per row block,
for no gain on the output.

**What this rules out.** The flip count tracks the quantisation step, not the granularity of the scale, and no choice of
max-based group gets the step below the model's own near-ties: a 12% smaller step leaves the same 13 to 16 divergent
prompts. int8 activations therefore need the margin guard, or more activation bits - 16-bit activations against the same
ternary weights would put the step 256x lower - and a smaller group is neither. The harness for all of it is
`~/kt/perf/q8/` (`kq8_arms.sh` runs the group arms, `kq8_pc2.sh` calibrates and runs the per-channel one, `kq8_summ.py`
scores an arm against fp32 and `kq8_drift.py` dumps the logits the drift column is computed from).

### The margin guard: implemented, measured, and not enough

The section above ends "int8 activations therefore need the margin guard". It is now implemented, and it does not
work. `KINDI_CPU_Q8GUARD=<gap>` (in `kcpu.c`, off by default) turns a decision into a fallback when the int8 logits'
own top-1/top-2 gap is under `<gap>`: the head is then recomputed with the production fp32 kernel from the same folded
input and the decision taken from those logits. The head is the only place a decision can be re-derived cheaply -
anywhere else the state the token came from is gone, so a fallback there would mean replaying the sequence - and
`kc_guard_stats()` reports how many decisions were examined and how many fell back.

| arm | fallbacks | prompts diverging | differing steps | flips above p1 |
|---|---:|---:|---:|---:|
| int8 per-128 (the arm above) | 0 | 13 of 48 | 1,136 | 1 |
| **int8 + guard at 0.05** (the measured drift) | **77 of 7,680** | **14 of 48** | 1,219 | 1 |
| int8 + guard on every decision | 7,680 of 7,680 | 14 of 48 | 1,219 | 1 |

The guard is not neutral: it is **one prompt worse than doing nothing**, and the arm that falls back always produces a
stream identical to the one that falls back on 1% of decisions. That second fact is the useful one - it is the ceiling
of every head-only guard, and it says the fallback changes nothing that a near-tie did not already touch. The flips are
not made at the head: at a decision whose fp32 margin is 0.006 to 0.016 the residual stream reaching the head is already
int8-drifted, so an exact head working from a drifted input lands on the int8 answer, not the fp32 one. Making the last
GEMV exact cannot recover a decision that was lost 24 layers earlier.

The cost is not what stops it. Measured alone on 6 threads, one fallback is **+10.0 ms on an 87.4 ms int8 token** (head
248,320 x 2,048, 508.6 M parameters, 27% of the model), so at the 1% trigger rate the guard costs 0.1 ms/token, about
0.1%. With the guard off
`kindi verify` is unchanged at both tiers (4/4 prompts, 160/160 tokens, `KINDI_CPU_TIER=q8` included), and the fp32 arm
still reproduces the earlier sweep token for token.

So the fix has to be at the state, not the decision: replay a token from an fp32 checkpoint of the recurrent state when
its margin is small (cost set by how often the checkpoint is refreshed, not by how often the guard fires), keep the
last few layers in fp32 so the head sees an undrifted stream, or move to 16-bit activations, whose step is 256x finer -
the option the finer schemes above already point at.

### The exact tail: partial exactness does not recover the decisions either

The guard failed because the decision had already been lost upstream. `KINDI_CPU_Q8TAIL=<k>` asks the converse
directly: the last `k` blocks **and** the head run the fp32 kernel while everything before them stays int8, so the head
sees a stream drifted only by the prefix. `k = 0` is the plain int8 path and `k = NL` is plain fp32, and both endpoints
reproduce their arms token for token (3/3 prompts each), which is what makes the switch itself trustworthy.

| exact tail | prompts diverging | differing steps | flips above p1 |
|---|---:|---:|---:|
| 0 - plain int8 | 13 of 48 | 1,136 | 1 |
| 1 block + head | 12 of 48 | 834 | 1 |
| 4 blocks + head | **10 of 48** | 907 | 1 |
| 8 blocks + head | **10 of 48** | 896 | 1 |
| 12 blocks + head | **10 of 48** | 872 | 2 |
| 24 (= fp32) | 0 | 0 | 0 |

**The curve plateaus, and that is the result.** Making a sixth of the stack exact is worth one prompt and making a
third exact is worth three; doubling that to half the stack is worth nothing more, and the eight blocks between `k = 4`
and `k = 12` contribute not one prompt. The remaining ten are decided by the drift already present in the prefix the
tail cannot reach. The step count tells the same story less coarsely: 1,136 -> 834 with a single exact block, then flat
within 40, so the tail's real effect is a 25% fall in differing positions rather than a fix. The plateau is measured
from `k = 4` to `k = 12`; `k = 24` is exact because it is fp32, and nothing between 12 and 24 was run, so whether the
count falls back to zero somewhere in that gap or only at the end is open.

Cost, in this prototype, runs the wrong way to be a constraint: the elapsed time *falls* as the tail grows (1,942 s at
one block, 1,890 s at four, 1,848 s at eight, 1,725 s at twelve for the same 7,680 tokens), because this int8 path
measures 87.4 ms/token against 45.6 for fp32 and replacing an int8 layer with an fp32 one is a saving. That is the
prototype's slowness, not a property of the design: a real VNNI kernel would pay about `k/NL` of the GEMV time at
roughly a third of the int8 throughput, so four exact blocks would cost of order 10% and buy three prompts.

Both cheap shapes of guard are therefore closed. What is left is changing the error rather than its location: 16-bit
activations, whose step is 256x finer than int8's, or a state-level design that keeps an fp32 copy of the recurrent
state and replays a token from it when the margin is small.

The prototype is **not** a performance result and must not be read as one: it runs at 157 ms per token against 36 ms for
the fp32 path, 4.3x slower, because it uses `vpmulld` per column pair and scalar int8 indexing instead of the byte-parallel
`vpshufb` + `vpdpbusd` form, and it still reads the row-blocked pair layout rather than a column-major int8 repack. Its
only job was to produce the numbers above. Speed still needs the real kernel and real silicon.

**Hardware.** Correctness does not need the silicon, as the measurement above shows - and it turned out not to need SDE
either. Intel SDE emulates AVX-VNNI, AVX-512-VNNI and AMX-INT8 (slow, 10-100x) but it is not installed here, and for the
numerics it is unnecessary: integer results do not depend on which integer instruction computes them. SDE would still be
the right tool to validate the *encoding* and the dispatch path before buying hardware. Throughput needs real parts.

| target | hardware | note |
|---|---|---|
| AVX-VNNI 256-bit | Alder Lake / Raptor Lake, 12th-14th gen Core | cheapest real target, and the most likely Kindi hardware |
| AVX-512 VNNI | Ice Lake, Tiger Lake, Cascade Lake, Sapphire Rapids | mutually exclusive with the above: 12th-14th gen disables AVX-512 |
| AMX-INT8 | Sapphire Rapids / Emerald Rapids Xeon | cloud instances (c7i/m7i, GCP C3) are the cheap route |

Strategic read: this is weeks of focused work, the numerics now say the exactness guarantee does not survive a wide
corpus, and it pays only on hardware this host is not. The batched GEMV is measured at 1.35x and runs on the machine in front of us, so it goes
first. The VNNI path is the branch's reason to exist on 2024+ Intel, and SDE is how to build it correctly before any
silicon is bought.

## K/V-cache quantisation: q8_0 costs nothing, the small alphabets do not save the model

The K/V cache is a storage format, not an arithmetic one, so `KINDI_CPU_KVQ` (in `kcpu.c`, off by default, prototype
only) selects it per run and every path still materialises a head's 256 values into fp32 before the same dot products.
`KINDI_CPU_KVQ_K` and `KINDI_CPU_KVQ_V` select it a side at a time. `0` is fp16; `1` is ternary at 2 bits with one
scale per (position, head); `2` is q8_0's 32-value int8 blocks with an fp16 scale of `amax/127`; `3` is 4-bit, 16
levels with a fitted per-head scale; `4` is 2 bits on four levels `{-3,-1,1,3}a`; `5` is 4 bits as two passes of that
same 4-level code, levels `(2m+1)a/4` for m in -8..7. Modes 1, 3, 4 and 5 fit their scale by least squares to the code
it induces, from four starts, and keep the lowest error. Same 48 prompts, 160 greedy tokens, 7,680 decisions, and the
fp16 arm reproduces the stored `wide_fp32` arm token for token, so the switch is inert at 0.

| arm | K | V | bits/val | cache MiB | vs fp16 | prompts diverging | median KL at step 0 | max KL | top-1 at step 0 |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | fp16 | fp16 | 16.00 | 12.00 | 1.00x | - | 0 | 0 | 48/48 |
| q8 K, fp16 V | q8_0 | fp16 | 12.25 | 9.19 | 1.31x | 4 of 48 | 0.00000 | 0.0001 | 48/48 |
| q8_0 | q8_0 | q8_0 | 8.50 | 6.38 | 1.88x | 6 of 48 | 0.00002 | 0.0002 | 48/48 |
| ternary K, fp16 V | ternary | fp16 | 9.03 | 6.77 | 1.77x | 46 of 48 | 0.02136 | 0.2448 | 39/48 |
| fp16 K, ternary V | fp16 | ternary | 9.03 | 6.77 | 1.77x | 48 of 48 | 0.07000 | 0.3646 | 40/48 |
| q8 K, 4-bit V | q8_0 | 4-bit | 6.28 | 4.71 | 2.55x | 40 of 48 | **0.01308** | 0.0704 | 43/48 |
| q8 K, 4-level x2 V | q8_0 | 4-level x2 | 6.28 | 4.71 | 2.55x | 42 of 48 | 0.01446 | 0.0689 | 44/48 |
| q8 K, 2-bit V | q8_0 | 2-bit | 5.28 | 3.96 | 3.03x | 48 of 48 | 0.07439 | 0.4069 | 39/48 |
| 2-bit K, q8 V | 2-bit | q8_0 | 5.28 | 3.96 | 3.03x | 47 of 48 | 0.02187 | 0.1935 | 38/48 |
| 4-bit | 4-bit | 4-bit | 4.06 | 3.05 | 3.94x | 43 of 48 | **0.01433** | 0.0890 | 43/48 |
| 4-level x2 | 4-level x2 | 4-level x2 | 4.06 | 3.05 | 3.94x | 45 of 48 | 0.01896 | 0.0807 | 44/48 |
| ternary | ternary | ternary | 2.06 | 1.55 | 7.77x | 48 of 48 | 0.09135 | 0.3528 | 36/48 |
| 2-bit | 2-bit | 2-bit | 2.06 | 1.55 | 7.77x | 47 of 48 | 0.08116 | 0.4184 | 39/48 |

KL is over the full 248,320-token vocabulary between the two step-0 distributions, the one point where every arm sees
an identical context. Bits per value include the per-head scale (16/256 = 0.0625) and average the two sides.

**q8_0 is free.** Six prompts leave the fp16 stream and every one of them at a step where fp16's own lead was below
its p1 margin (0.0509); the median margin eaten at the aligned steps is 0.005%, the step-0 distribution moves by 2e-5
nats, and the argmax never changes. That is a better trade than int8 *activations* (13 of 48) because it touches six
layers' cache instead of every GEMV in the model.

**The values are the side that needs the bits.** Both one-sided arms exist at 2 bits and they are not close: K at 2
bits with V in fp16 costs 0.0214, V at 2 bits with K in fp16 costs 0.0700, and 46 to 48 of 48 streams leave fp16
either way. Moving the keys to q8_0 barely helps a K-side arm (2-bit K with q8 V: 0.0219), while V at 2 bits stays at
0.071 to 0.074 however the keys are stored. Keys only decide the ordering of 1,024 positions; values decide what gets
mixed.

**At 2 bits the alphabet was wrong, and fixing it does not rescue the rung.** Ternary spends its two bits on three
levels and drives 56% of codes to zero. Four levels at the same 2 bits reconstruct 16% closer on K and 8% closer on V
(relL2 0.425 / 0.422 against 0.506 / 0.460, over 744 cache vectors taken from three prompts at every position), and
that is exact: the shipped packer reproduces a NumPy four-level search code for code on 190,464 codes. End to end the
median step-0 KL moves 0.0914 -> 0.0812 and the step-0 top-1 goes 36 -> 39 of 48, but a paired bootstrap over the 48
prompts puts the mean difference at -0.014 with a [-0.036, +0.009] interval: it includes zero. Both 2-bit arms are
still a different model (47-48 of 48 streams leave fp16, KL four thousand times q8_0's), and 2 bits against 4 bits is
+0.066 [0.045, 0.088], which does exclude zero. The rung's deficit is the bit budget, not the codebook.

**At 4 bits the lever is the scale, not the alphabet.** The shipped 16-level code used `amax/7`, and a head's maximum
is a poor estimate of the step the rest of the vector wants: the least-squares scale is 18% closer on K and 19% on V
(relL2 0.1730 -> 0.1419 and 0.1825 -> 0.1481). Composing the 2-bit four-level alphabet twice into 4 bits - sixteen
even levels, `(2m+1)a/4` - reaches 0.1422 / 0.1660: level with the fitted 16-level code on K and 12% worse on V, so
the four-level code extended to four bits buys nothing the scale fit does not. End to end the fit takes the V-side
arm from 0.0139 to 0.0131 (mean -0.0028 [-0.0087, +0.0014]: not resolvable) and the both-sides arm from 0.0199 to
0.0143 (mean -0.0097 [-0.0180, -0.0034]: resolvable), and the two-pass form is worse on both sides
(+0.0038 [+0.0005, +0.0073]). What the fit does fix is the tail: max KL over the 48 prompts falls 0.162 -> 0.070 on
the V-side arm, because one outlier in a head used to squeeze every other value in it.

**Is 4-bit V usable?** Not in the q8_0 sense: 0.0131 nats against 2e-5, 40 of 48 streams diverge, and 5 of 48 first
tokens change. The first-token count moved the wrong way with the fit (2 of 48 under `amax/7`, 5 of 48 under the fit)
while every KL statistic improved, which is the kind of thing three prompts can do; the medians and the tail are the
measurements with an interval behind them. What 4-bit V is, is 6.28 bits and 2.55x, 46 of 48 top-1 at step 0 - a real
trade at long context and not a free one.

**Four starts, chosen on the objective.** The assignment map is piecewise constant in the scale, so iterating from one
start is bistable: a one-ulp difference in the input can settle in a different basin, and one position-0 head measured
0.63 and 0.79 relL2 from the two basins. Every fitted mode therefore iterates four starts (fractions of `amax` and the
mean) and keeps the lowest error, ~10k compares per cache write against ~10^8 multiply-adds per token.

**Is it worth any of the memory?** At ctx 1024 the whole cache is 12 MiB against 506 MB of weights, so q8_0 saves
5.6 MiB, 1.1% of the pack, and attention is 0.9% of a token, so nothing shows in throughput. At ctx 8192 the cache is
96 MiB against the same weights: 51 MiB at q8_0 (45 MiB saved, 9% of the pack), 24 MiB at 4-bit (72 MiB), 12 MiB at
2 bits (84 MiB, 16%). **q8_0 K/V is a long-context memory knob, not a speed or quality one**; 4-bit is the smallest
format that keeps the model, and the 2-bit rungs buy the last 12 MiB at the price of a different model.

**A scale per 32 values narrows the 2-bit V rung, and there is now a real pack to say so on.** Mode `6` is the same
four-level alphabet with one fp16 scale per **32-value block** instead of per head: 8 bytes of codes and 2 bytes of
scale a block, so 2.5 bits a value against 2.0625 at a 256-value head - 21% more cache than mode 4 for no extra level.
The pack the question needed is on this host now: `prism-ml/Ternary-Bonsai-2-2B`'s PTQ1_0 file (429 MB, sha256
checked) converted to a 506 MB pack at `~/.kindi/packs/bonsai-2b`. Measured three ways:

- **At the codec, on the values the model writes.** 2376 V vectors captured from that pack (198 tokens of prose
  through the trained 2B, 6 attention layers): mean relL2 **0.4072 -> 0.3841, 5.7% closer, and the block scale wins on
  2374 of the 2376 vectors.** The unanimity the random fixture never had. It is not uniform - 2.5% better at layer 23,
  8.2% at layer 19 - and the middle layers are the hardest to encode at all (relL2 0.448 / 0.483 at layers 15 / 19), so
  the range moves most exactly where the alphabet is already short.
- **End to end, the same sign, and in every window.** KL against the fp16 cache with K at q8_0 and only V varying,
  three 40-token windows: mode 4 **0.19804** nats mean against mode 6 **0.15993**, 19% lower KL for 21% more cache, with
  mode 6 closer in all three (0.264/0.234, 0.116/0.093, 0.214/0.154). The fixture used to flip this ranking between two
  sizes (0.2628 vs 0.2960 one way, 0.0128 vs 0.0107 the other) - that was the fixture talking, and trained values do not
  flip it.
- **It does not turn 2 bits into 4.** The best 2-bit V rung is still 8.4x the KL of 4-bit V (0.15993 against 0.01901),
  and 4-bit V costs 4.06 bits against 2.50. Mode 6 is the right pick *if* a 2-bit V is the constraint and the extra 21%
  of cache is available; it is not a substitute for the 4-bit rung.

What the variant did turn up is a bug in the selector: `kc_kv_quant2` packed its choice as `K*4 + V`, which stopped
being injective once `V` could reach 4 - K at 2 bits with V at 4 bits gave 8, the same code as K at 4 bits with V in
fp16 - so a refused setting could have compared equal to a wanted one. It is `K*8 + V` now, on both sides.

**How those values were got.** `kcpu.c` writes out every V vector as the cache is about to quantise it:
`KINDI_CPU_VCAP=<file>` (or `kc_vcap_begin`, for a host that sets things after the library loads) records the fp32
`m->vv` of each (layer, position, head) before any codec touches it. The value does not depend on `KINDI_CPU_KVQ`, so
one capture is the input to modes 1, 3, 4, 5 and 6 at once, and to anything added later. It can ride along with any run
(`KINDI_CPU_VCAP=/tmp/v.bin kindi --pack PACK bench-spec TEXT`) or be driven directly by
`python -m tools.kvq.capture --pack PACK --tokens 198 --per-layer`, which tokenises the text with that pack's own
`vocab.json`/`merges.txt` so the activations are a model's rather than random token ids; `--cap FILE --per-layer` scores
a capture with no engine and no pack at all. The synthetic pack still scores the expected nothing - 2.5% over 144
vectors, block closer on 125 - and the five `--selftest` checks pin what is not noise: the header is the engine's shape,
a record is the vector the fp16 cache holds to fp16 precision, the cache format does not change the captured value, a
truncated file loads as a prefix, and a capture written by another host loads and scores.

One platform note, because it bit this hook first: the Windows build links the CRT statically (`/MT`), so `getenv` inside
`kcpu.dll` reads an environment snapshot taken when the library loads. A variable Python sets with `os.environ` after
that is invisible to the C side - the shell works, a test does not - which is why the capture also has an explicit entry
point. The same applies to any other C-side `getenv` knob (`KINDI_CPU_Q8`, `KINDI_CPU_Q8G`, `KINDI_CPU_DUMP`, ...): set
them in the environment that launches the process.

Harness: `~/kt/perf/kvq/` (`kqkv_arms.sh`, `kqkv_asym.sh`, `kqkv_q4.sh`, `kqkv_q2.sh` and `kqkv_q45.sh` run the arms,
`kqkv_wide.py` is the sweep, `kvq_report.py` scores divergence, KL and cost, `kvq_detail.py` the divergence-step
distribution, `q2_floor.py` and `q45_floor.py` the alphabet floors, `q2_check.py` the position-0 round trip).
## Positional encoding: what the retired files ask for, and what it does past 8k

The two retired Bonsai Q1_0 GGUFs do not ask for the plain rotary table. Both declare `qwen3.rope.scaling` as
**yarn, factor 4**, and they differ in the window they scale from: `original_context_length` is **8192 in the 1.7B and
16384 in the 8B**. The converter records that block verbatim in `kindi_pack.json` as `rope_scaling`, `kindi.rope`
turns it into the table the kernels read, and every engine takes the table from there - the CPU one, the CUDA one and
the two-card plan engine - so a pack cannot be served with a different positional encoding from the file it came from.
A manifest with no `rope_scaling` (every other pack here) produces the table it always did, bit for bit.

`kindi.rope` is llama.cpp's arithmetic and not a re-derivation: `rope_yarn_ramp` and `ggml_rope_yarn_corr_dims` decide,
per rotary dimension pair, whether the angle is interpolated (divided by the factor) or extrapolated (left alone), and
`mscale = 1 + 0.1 ln(factor)` multiplies both tables. For the 1.7B the crossover is dimensions 17 to 34 of 64; for the
8B, 20 to 37 - the same `mscale` of 1.138629, because the factor is the same. The kernel needed no change: the table is
data it already reads. The one thing that did need re-deriving was the 8B *pack*: it was converted before the manifest
carried the key, so its block was rebuilt from its own GGUF (`kindi.convert.rope_scaling`) without touching a weight.

**Protocol.** A walk is one tokenisation and one window, scored one forward pass a token, so the comparison has to be
paired: the same tokens at the same positions through the two tables, subtracted token by token, with a moving-block
bootstrap (32-token blocks, 2,000 resamples, fixed seed) for the interval - positions inside a window share
activations, so a token-level interval here would be far too narrow to be a gate. `scripts/rope-longctx.py` does the
walk and the join, writes each scored token as it lands, and its 112-token window reproduces `tools/nll`'s own NLL to
the last digit (3.1648 both ways), which is the check that the two paths compute the same quantity.

**At 9-10k, on a real 7,680-token history, the two tables are indistinguishable.** Corpus: this machine's prose corpus
(21,223 tokens, sha256 in the JSONs), positions 7,680 to 10,239, 2,560 scored tokens, 61 minutes an arm:

| positions | tokens | plain RoPE | YaRN f4 from 8192 | delta (yarn - plain) | 95% interval |
|---|---:|---:|---:|---:|---|
| 7,680-8,192 (inside) | 512 | 1.2410 | 1.2094 | -0.0316 | [-0.0773, +0.0195] |
| 8,192-8,704 | 512 | 1.2223 | 1.2573 | +0.0350 | [-0.0066, +0.0860] |
| 8,704-9,216 | 512 | 1.3936 | 1.4403 | +0.0466 | [-0.0099, +0.1026] |
| 9,216-9,728 | 512 | 1.2977 | 1.3231 | +0.0254 | [-0.0442, +0.0828] |
| 9,728-10,240 | 512 | 1.1315 | 1.0867 | -0.0448 | [-0.0996, -0.0047] |
| **whole walk** | **2,560** | **1.2572** | **1.2633** | **+0.0061** | **[-0.0203, +0.0315]** |

The per-band levels move more between bands than between tables, because they are different text as well as
different positions - which is why the paired delta and not the level is the number to read. The delta changes sign
once in each direction and the whole-walk interval covers zero: on this corpus, at 1.25x the window, applying the
file's own YaRN is neither a gain nor a loss.

**At 1.5x it is a gain**, and a large one. The same two tables, scored at positions 12,288-12,799 over a 256-token
history (the cheap way to reach a deep position: 1,536 forward passes instead of 12,800, 11 minutes an arm):

| positions | tokens | plain RoPE | YaRN f4 from 8192 | delta | 95% interval | tokens where yarn is closer |
|---|---:|---:|---:|---:|---|---:|
| 12,288-12,416 | 128 | 3.9236 | 3.6020 | -0.3215 | [-0.5100, -0.0331] | 84 of 128 |
| 12,416-12,544 | 128 | 2.4171 | 2.1652 | -0.2520 | [-0.4163, -0.1155] | 87 of 128 |
| 12,544-12,672 | 128 | 2.4754 | 2.1519 | -0.3235 | [-0.6570, -0.2161] | 71 of 128 |
| 12,672-12,800 | 128 | 2.1419 | 2.0557 | -0.0862 | [-0.2359, +0.0720] | 68 of 128 |
| **whole walk** | **512** | **2.7395** | **2.4937** | **-0.2458** | **[-0.3544, -0.1276]** | **310 of 512** |

**What that pair of results says.** Factor 4 is a fourfold stretch: it is the amount of interpolation that makes 32,768
positions look like 8,192, and 8,192 is the number the file trained on. At 9k the window needs roughly 1.2x, so YaRN
is applying four times the compression the position calls for, and the two errors - plain RoPE extrapolating a little,
YaRN interpolating a lot - happen to cost the same. At 12.3k plain RoPE is far outside what the training positions
covered and the same table is worth a quarter of a nat, about 9% of the loss, on 310 of 512 tokens. So "apply the
YaRN the file asks for" is not a win at the boundary of the window; it is a win where the file was designed to be used,
and at 9k it is at worst free. For 9-10k the parameter to want is somewhere between 1.25x and 2x, not 4x, and the
section below measures the range: both ends beat plain RoPE and both beat the file's own factor 4.

**Does the factor matter, and should it match the depth?** The obvious follow-up - and the thing
[LongRoPE](https://arxiv.org/abs/2402.13753) does properly, with per-dimension factors found by search and a re-fit at
8k so the original window still works - is whether the factor should match the depth being served.
`scripts/rope-longctx.py --rope-factor F --rope-window 8192` rebuilds the kernel's table with a factor the manifest
does not name (and reads it back), so one pack answers that.

**Yes: 1.25 and 2 both beat plain, and the file's own 4 does not.** Same corpus, same 2,560-token walk, only the table
changed, so every row is paired token by token against plain RoPE:

| positions | plain | f 1.25 | f 2 | f 4 |
|---|---:|---:|---:|---:|
| 7,680-8,192 (inside) | 1.2410 | -0.0189 [-0.0412, +0.0000] | -0.0489 [-0.0946, -0.0091] | -0.0316 [-0.0773, +0.0195] |
| 8,192-8,704 | 1.2223 | -0.0211 [-0.0348, -0.0085] | -0.0238 [-0.0560, +0.0044] | +0.0350 [-0.0066, +0.0860] |
| 8,704-9,216 | 1.3936 | -0.0063 [-0.0266, +0.0116] | -0.0014 [-0.0335, +0.0346] | +0.0466 [-0.0099, +0.1026] |
| 9,216-9,728 | 1.2977 | -0.0061 [-0.0310, +0.0114] | -0.0122 [-0.0577, +0.0187] | +0.0254 [-0.0442, +0.0828] |
| 9,728-10,240 | 1.1315 | -0.0490 [-0.0891, -0.0206] | -0.0676 [-0.1218, -0.0327] | -0.0448 [-0.0996, -0.0047] |
| **whole walk** | **1.2572** | **-0.0203 [-0.0325, -0.0103]** | **-0.0308 [-0.0506, -0.0139]** | **+0.0061 [-0.0203, +0.0315]** |

Four arms, three verdicts. Factors 1.25 and 2 are negative in every band and their whole-walk intervals exclude zero;
factor 4's covers zero, so the table the file declares is the one that buys nothing measurable here. Head to head both
are real wins against 4 - 2 by 0.0369 [0.0215, 0.0536], 1.25 by 0.0264 [0.0040, 0.0498] - and 1.25 against 2 is -0.0105
[-0.0230, +0.0010], which is a tie. So "which factor wins" has a range for an answer, not a point: **anything from
about 1.25x to 2x of the 8,192 window beats both plain RoPE and the 4x the file declares, and this walk cannot separate
the ends of that range.** 1.25x-2x is what 9-10k positions actually ask for; 4x is sized for 32,768. It is not a free
change either way - the file says 4, and serving 1.25 or 2 means `kindi convert MODEL.gguf --rope-factor 2`, which is a
decision to disagree with the checkpoint rather than to implement it. (The flag writes the manifest's `rope_scaling`
block and keeps the file's own block beside it as `rope_scaling_source`, so a pack built this way says which file it
disagrees with.)

**The cheap depth probe, for what it is worth.** With a 256-token history (512 scored tokens, ~8-12 min an arm) the
same comparisons are:

| depth | plain | f 1.25 | f 2 | f 4 |
|---|---:|---:|---:|---:|
| 8,960-9,472 | 2.1588 | 2.1260 (-0.033 [-0.061, +0.002]) | 2.0524 (-0.106 [-0.168, -0.036]) | - |
| 12,288-12,799 | 2.7395 | 2.6098 (-0.130 [-0.178, -0.065]) | 2.4632 (-0.276 [-0.384, -0.168]) | 2.4937 (-0.246 [-0.354, -0.128]) |

The effect of the encoding grows with the distance from the trained window - the same factor 1.25 is worth 0.033 at 9k
and 0.130 at 12.3k - which is what a positional encoding running out of range looks like, and it is why the best factor
is not one number: at 12.3k factor 1.25 trails 2 and 4 by 0.13-0.15, while at 9-10k it is level with 2 and ahead of 4.
The probe also ranks factor 2 above plain at 9k, which the full walk confirms, so it preserves the ordering there; what
it does not do is size the effect, which it overstates by 3.4x (0.106 against 0.031). Two cautions follow. It is an instrument for direction, not
magnitude, and at 12.3k it puts factors 2 and 4 level within their intervals, so a factor choice between them cannot be
settled from it. And a protocol that helps everything by compressing distances among a few hundred tokens is not
measuring extrapolation, which is why the full-context walk stays the gate. That is also why LongRoPE's search stage
cannot be approximated by trying a handful of scalars - it searches per dimension, and its second stage re-fits the
model at 8k precisely because a fixed factor, whatever it is, perturbs the window the model was trained on.

**Limits.** One corpus and one model: the probes above are the 1.7B, and the 8B's block is recorded and applied but
not walked, because its cost law on this host puts one 16k arm at hours. The deep point carries a 256-token history, so
its *levels* are not comparable with the 8-10k walk's - only the plain/yarn difference within it is. A genuine 16k
context is 2.6 hours an arm at the measured `0.024 + 6.7e-5 x position` seconds a token, and was not run. The
intervals are bootstrap intervals over one corpus's positions, not over corpora.**Harness.** `python -u scripts/rope-longctx.py walk --pack PACK --out W.jsonl [--ctx N --prefill P --offset O]
[--rope-factor F --rope-window N]` for an arm, `... join --a plain.jsonl --b yarn.jsonl --bands a,b,c` for the table.
The walker's factor is for the values no pack was built at (1.25); a factor a pack should keep is `kindi convert
--rope-factor F`, which is the same block the walker overwrites.
JSONs beside the packs: `rope-factor-sweep-1.7b-10k.json` (all four factors on the 9-10k walk, per band),
`rope-f2-1.7b-10k.json` (factor 2 alone) and `rope-yarn-1.7b-deep.json` (the deep point).

## Evaluated and rejected (advice from outside this repo)

Three generic "how to speed up CPU inference" lists were triaged against this engine. Almost all of it was written for
PyTorch or llama.cpp deployments of dense or MoE models in the 8B-70B range; this engine is hand-written C over a
ternary 2B hybrid model that is already at 2 bits. Recorded so the same lists are not re-triaged.

| claim | why it does not apply here |
|---|---|
| TorchScript, `torch.compile`, IPEX, BetterTransformer, `torch.no_grad`, DataLoader workers | there is no PyTorch on the path: `kindi/cpu.py` imports `ctypes`, `json`, `os`, `time`, `numpy` and calls C |
| int8 / FP4 / mixed-precision quantisation, EXL3 mixed bitrate, MXFP4/NVFP4 | every one *raises* the bitrate. The weights are already ternary at 2 bits packed, with no lower rung to step to |
| "make the model fit in the cache hierarchy" | 529 MB of weights against 16 MB of L3. Only a ~60 M-parameter model at 2 bits would fit; that is not a Kindi model |
| operator fusion | already fused: one C function per stage inside a single OpenMP region. The only seams left are the ~425 barriers per token, about 0.5% |
| MoE: expert sparsity, expert paging and cache compression, FP8 router, "quantise the router not the experts" | the Bonsai catalog is dense. No router, no experts, nothing to page |
| multi-token prediction | Bonsai has no MTP heads |
| KV-cache quantisation, FP8 KV, infinite KV, kv-less attention | now measured above: q8_0 halves the cache and only moves near-ties, ternary destroys it at any grouping. At ctx 1024 the cache is 12 MB against 529 MB of weights and attention is 0.35 ms of a 40.1 ms token (0.9%), and three of four layers are GDN holding no KV cache at all - so the saving is real only for long context. FP8, infinite KV and kv-less attention are still unread |
| NUMA pinning, socket-local KV | single socket, dual-channel, no node to bind to |
| ZipNN exponent coding, shared exponents, entropy-coded checkpoints | these shrink *storage*. The runtime streams 2-bit pair codes and would decompress on the fly, which adds instructions - and 5 extra instructions per weight word measured 0.6x in the mirror A/B. Bytes are not the limit |
| L3-residency tiling, cache-aware weight layouts | measured: pinning the pair table into 2 KB is worth 0-4%, and the working-set sweep is flat from 8 MB to 384 MB |
| `GOMP_CPU_AFFINITY`, `KMP_AFFINITY`, core pinning "significantly improves throughput" | measured over 9 configurations x 9 interleaved reps: every median between 21.1 and 24.2 ms |
| model simplification / pruning | not a runtime setting; that is a different model |

What survived, and where it lives:

- **Speculative decoding** and **n-gram / prompt-lookup speculation** are the same change as batching (items #4 and
  #5), and the batched kernel is now measured at 1.35x. The one idea from those lists that became work.
- **Hybrid SSM** is not a future optimisation; Bonsai already is one.
- **SpinQuant/QuaRot-style rotation** against activation outliers is already in the engine: `fold()` applies a
  randomized Hadamard transform with the pack's sign vectors before every GEMV.
- **AVX-VNNI / AMX dispatch** is scoped above and stays open; it is the only item rejected by *hardware* rather than
  by measurement.

Not evaluated: the infinite-KV, kv-less, TurboQuant and GEMQ papers were cited but not read. On the profile above they
would have to beat 0.9% of a token to matter for throughput, which makes them a long-context and quality question
rather than a speed one - but that is a judgement from the profile, not from reading them.

## Sources

- T-MAC, EuroSys'25 — <https://arxiv.org/abs/2407.00088> (full text: <https://arxiv.org/html/2407.00088v2>)
- bitnet.cpp, ACL 2025 — <https://arxiv.org/html/2502.11880v1>
- 1-bit AI Infra (bitnet.cpp tech report) — <https://arxiv.org/html/2410.16144v2>
- Vec-LUT, MobiSys'26 — <https://arxiv.org/abs/2512.06443> (code: <https://github.com/Cipherxzc/vlut.cpp>)
- BITCOS, Intel 2026 — <https://arxiv.org/html/2609.16338v1>
- BitNet b1.58 — <https://arxiv.org/abs/2402.17764>
- MoTE — <https://arxiv.org/html/2506.14435v2>
- Apex (MoE CPU/GPU assignment) — <https://arxiv.org/html/2610.23006v1>
- llama.cpp speculation docs — <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- llama.cpp measured q8/q4 KV figures — <https://github.com/ggml-org/llama.cpp/discussions/20969>

## Provenance of the numbers

Every tokens-per-second figure in this document was measured on this host. Every speedup quoted from a paper is
labelled with the hardware it was measured on (M2 Ultra, i7-13700H, Snapdragon, Xeon); **none of them has a
Coffee Lake / DDR4-2667 counterpart in the literature**, so treat published speedups as upper bounds with unknown
transfer.
