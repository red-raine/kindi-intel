# Benchmarks

Each table compares Kindi and llama.cpp on the same card, with the same model file and the same context. "Read" is
the prompt reading speed: an N-token prompt divided by its read time. "Write" is the generation speed at that depth.
All figures are tokens per second, from one stream, greedy, with Kindi's exact head. When only one engine runs at a
context, the table says why the other does not ("out of memory", "crashed", "not measured").

Unless a table says otherwise, the method is the same for the two engines. They get the same token ids. The prompt
is read in segments up to each N, and the read time of N tokens is the sum of the segments. At each N, 32 tokens are
generated. Project page: [amirhusain.com/kindi](https://amirhusain.com/kindi/).

## Kindi 0.3.0: Nemotron, the CPU engine and the prompt cache

The tables in this section are new in 0.3.0. The Bonsai tables after them are from Kindi 0.2.0. We measured the
0.3.0 engine against them again on the K1200 cards: the Bonsai rows are within 0.5%.

### NVIDIA Nemotron 3.5 Bolt E4B (UD-Q3_K_XL, sha256 3f8032b9…)

The same GGUF for both engines, f16 K/V, the same token ids, greedy, both engines warmed first.

| Hardware | Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---|---:|---:|---:|---:|
| GeForce GTX 1080 8 GB | 512 | 742.5 | 1087.9 | 55.10 | 56.42 |
| GeForce GTX 1080 8 GB | 8,192 | 922.2 | 896.6 | 51.11 | 53.05 |
| GeForce GTX 1080 8 GB | 16,384 | 837.8 | 730.7 | 47.76 | 50.35 |
| GeForce GTX 1080 8 GB | 24,000 | 746.9 | 626.5 | 45.20 | 48.22 |
| Quadro K1200 4 GB (ctx 8,448) | 512 | 66.8 | 17.6 | 10.41 | 8.27 |
| Quadro K1200 4 GB (ctx 8,448) | 8,000 | 73.9 | 12.0 | 9.07 | 7.67 |
| 2× Quadro K1200, split (ctx 8,448) | 512 | 122.5 | 15.3 | 17.15 | 7.44 |
| 2× Quadro K1200, split (ctx 8,448) | 8,000 | 124.2 | 19.0 | 16.08 | 7.37 |
| GeForce RTX 3080, 190 W, one chat request | — | 2,614 | 2,630 | 123.1 | 88.0 |

Builds: GTX 1080: Kindi 0.3.0-dev 849840a, llama.cpp CUDA b11804 (c7fc509), 2026-10-25. K1200: Kindi 0.3.0-dev
(5350b87 + the server fix), llama.cpp Vulkan 1b29621 (llama.cpp splits by layers on two cards), 2026-10-24. RTX 3080:
Kindi 0.3.0-dev b340cbe, gpu-unlock loaded, 2026-10-24. On the GTX 1080, llama.cpp reads a 512-token prompt faster
and writes 2-7% faster; Kindi reads longer prompts faster.

### The CPU engine (Ternary-Bonsai-2, AVX2)

Read: a ~2,000-token prompt with no cache reuse. Write: 256 tokens. Median of 3 runs, each engine at its best thread
count. llama.cpp: Prism ML's fork, CPU build.

| CPU | Model | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---|---:|---:|---:|---:|
| AMD Ryzen 9 7940HS (8 cores) | 2B | 90.4 | 60.5 | 64.8 | 23.6 |
| AMD Ryzen 9 7940HS | 4B | 37.8 | 25.7 | 29.1 | 16.5 |
| Intel Core i7-8700 (6 cores) | 2B | 42.50 | 33.45 | 28.92 | 18.12 |
| Intel Core i7-8700 | 4B | 17.02 | 12.97 | 12.61 | 7.75 |

### The CPU engine on an i9-9980HK (8 cores / 16 threads, AVX2 only)

One machine, `OMP_NUM_THREADS` swept, the 2B model, decode from position 100, 10 tokens each. "gemv" is the share of
the token spent in the ternary GEMV, from the CPU profiler (`kc_get_times`; see CARDS.md). "GB/s" is tokens/s x
0.506 GB, the bytes of the 2B pack streamed per token, against the 24.99 GB/s a plain 8-thread read of 512 MB reaches
on this host.

| threads | ms/token | tokens/s | gemv share | GB/s |
|---|---:|---:|---:|---:|
| 1 | 140 | 7.1 | 92% | 3.6 |
| 2 | 79 | 12.7 | 92% | 6.4 |
| 4 | 53 | 19.0 | 90% | 9.6 |
| 8 | 39 | **25.4** | 88% | 12.9 |
| 16 | — | not usable, see below | — | — |

Repeat runs of the 8-thread row landed between 24.5 and 28.6 tokens/s, so treat the last digit as noise.

**Thread count is limited by the 16-thread cliff, not by cores.** An empty `#pragma omp barrier` costs 0.3-0.6 us at 4
and at 8 threads on this host (three repeats), but 3.3-4.1 ms at 16, and one token is roughly 425 barriers. 16
threads therefore measured 0.24 tokens/s before the change below, about 90x slower than 8. That cliff is the
hypervisor co-scheduling 16 vCPUs on 8 physical cores: native Windows measures 203 us there instead. A hand-rolled
sense-reversing barrier is no faster than libgomp's (0.25-0.6 us at 4 and 8 threads), so replacing the barrier buys
nothing - size `OMP_NUM_THREADS` to the physical cores and stop there.

A single earlier reading of 24.9 us at 8 threads was an outlier under host load; treat the <1 us figure as the
repeatable one.

Against the 0.3.0 engine on the same box: 6.9 tokens/s (its best of 1/2/4/8/16 threads) to 25.4, and the `kindi verify`
workload from 141.7 s to 12.7 s. Three changes to `rows8`, each keeping the arithmetic and its order:

1. It walks two row blocks per table pass, so a group's table part is read once for 16 rows instead of once for 8.
2. The zero-code mask is one compare instead of two shifts.
3. It prefetches the weight group four ahead. Two ahead gives 25.7 tokens/s and eight ahead 19.7, so four is the knee;
   the prefetch is why the 8-thread row moved off 44% of this host's 25 GB/s.

`kindi verify` still matches llama.cpp on all four prompts after all three.

### Mirror consolidation, measured and rejected

The eight nonzero pair sums are ordered so that code `c` and code `7-c` are exact negations, so the table can hold
only codes 0-3 (16 KB instead of 32 KB) and recover the rest by flipping a sign. Both builds came from the same
source with only `pair_tables` and the inner loop changed, linked with distinct symbol names, and the two libraries
alternated inside one process, warm, 24 decode tokens per run, order reversed every rep:

| run | current | mirror | mirror/current, median | mirror wins |
|---|---:|---:|---:|---:|
| 1 | 26.09 | 16.11 | 0.63 | 0/9 |
| 2 | 23.08 | 12.77 | 0.55 | 0/9 |

Both runs output the same tokens as the current kernel on three 32-token prompts, so the table is right and the
trade is simply bad: the mirror index (`min(c, 7-c)`) and the sign mask cost about five extra instructions per
weight word, and that is worth more than the bytes saved. In run 1 the two distributions did not overlap at all
(mirror max 19.11, current min 23.25).

Two implementations of the mirror were wrong before the exactness check caught them: `cmpgt` returns all ones, so
the "negation" was a bitwise complement; and `(7-c) & 3` maps code 0 to lane 3. Neither is visible in a timing run.

A second, deliberately wrong build pins the table pointer to a single 2 KB group, so every step re-reads the same hot
table: 1.09x median against the real kernel, with a 0.60-1.16 spread. Making the table resident is worth at most
single-digit percent, which is why the plan's column-tiling item is not worth a pack-format change.

### What the counters say about the GEMV (i9-9980HK, 8 threads)

`~/kt/perf/gb.c` opens the PMU with `perf_event_open` - there is no `perf` binary on this image and installing one needs
root - and drives `rows8` exactly as `gemv` does, over 432 MB of packed weights. One 1.536 GMAC pass takes 85 ms on one
thread and 21-23 ms on eight, which is the rate the real engine implies for a token's GEMV, so these numbers describe
the shipped kernel.

| working set | GB/s | IPC | L1D load miss | vpermps/cycle |
|---|---:|---:|---:|---:|
| 8 MB (L3-resident) | 18.2 | 1.71 | 17.9% | - |
| 49 MB | 18.2 | 1.57 | 16.3% | - |
| 123 MB | 17.6 | 1.57 | 17.1% | - |
| 384 MB (DRAM-resident) | 19.0 | 1.76 | 17.9% | 0.20 |

Throughput is flat across a 48x change in working set, so the GEMV is **not memory-bound**: a pure streaming read of
the same bytes reaches 26.0 GB/s on this host. Per core it converts **3.2 MACs per cycle**, against 16 for a dense
fp32 FMA loop. What that closes:

- Pinning the pair table into 2 KB of L1 cuts L1D misses from 25.7 M to 7.1 M - 72% of all of them - and is **0%
  faster**. The table costs cache, not throughput; L2 absorbs it.
- dTLB takes 106 k walks per pass, 1.13 per 4 KB page. That is already one per page, the best 4 KB pages can do, so
  huge pages are worth about 1%.
- Removing the 4-ahead prefetch is a wash (21.9 against 21.5 ms median).
- Thread binding is a wash: nine interleaved reps over nine configurations put every median between 21.1 and 24.2 ms.
  Single runs had shown `OMP_PLACES=cores OMP_PROC_BIND=spread` at 1.43x over unbound, and that was noise.

The GEMV is issue-bound at IPC 1.6-1.8, so the lever left is fewer instructions per weight word, not fewer bytes.
The batched-token estimate that follows from this is in `docs/CPU-PERF-2026.md`.

Confirmed on the real pack (`~/kt/perf/real.py`), not only a synthetic buffer: the 151 distinct matrices of
`~/.kindi/packs/bonsai-2b`, repacked by the shipped `Engine`, total **470 MB of pair codes + 59 MB of scales = 529 MB
and 1.88 G parameters per token**. A 13 MB subset runs at 18.6 GB/s and the whole 529 MB at 18.3, IPC 1.56-1.72,
which is the synthetic result to within noise. In situ the engine's own timers put the GEMV at **34.47 ms of a
40.1 ms token (85.9%, 15.3 GB/s)**, so roughly 19% is lost to running it interleaved with the other stages rather
than as one stream. That is still nowhere near the 26 GB/s ceiling, so nothing above changes - but the isolated
figure is an upper bound.

Two-token batching works. `rows8_b2` shares each weight word's decode and zero-mask between two activations and is
bit-exact against two separate `rows8` passes (0 mismatched floats over the whole model). Interleaved A/B, 7 reps,
8 threads, all 151 real matrices:

| | ms / 2 tokens | ms / token | instructions per vpermps | IPC |
|---|---:|---:|---:|---:|
| two `rows8` passes (today) | 58.7 | 29.4 | 8.92 | 1.64 |
| one `rows8_b2` pass | 43.6 | **21.8** | **5.92** | 1.49 |

**1.35x per token**, against a 1.37x prediction, from 8.92 -> 5.92 instructions per lookup. IPC falls, so the win is
the instruction count and not the memory traffic - and the batched pass reads each weight byte once for two tokens
(265 MB/token instead of 529) yet is only 1.35x faster, which is the final proof that bandwidth was never the limit.
It is not wired into decode, because decode is serial and there is no second token to batch with until speculative
verification or a second sequence exists.

### The prompt cache, GTX 1080

Seconds for each request (prompt and reply), `examples/prompt_cache_bench.py`. Kindi with its cache off and on; llama.cpp
(b11804) with its cache on. The replies with Kindi's cache off and on were the same in all 12 requests of each model.

| Job, request | Nemotron, Kindi off | Nemotron, Kindi on | Nemotron, llama.cpp on | 27B, Kindi off | 27B, Kindi on |
|---|---:|---:|---:|---:|---:|
| Agent, turn 2 | 2.76 | 0.67 | 1.29 | 15.33 | 2.52 |
| Agent, turn 3 | 2.96 | 0.86 | 1.30 | 15.65 | 2.93 |
| Regenerate | 2.69 | 0.43 | 4.93 | 15.47 | 1.85 |
| Document, question 2 | 11.03 | 0.70 | 3.16 | 49.42 | 14.53 |
| Document, question 3 | 10.71 | 0.42 | 3.10 | 47.19 | 11.46 |
| Two agents, A turn 2 | 2.58 | 0.58 | 1.40 | 14.68 | 2.61 |
| Two agents, B turn 2 | 2.57 | 0.53 | 1.44 | 14.80 | 2.79 |

### Radeon 780M

2026-10-25. Kindi 0.3.0 development build (b5230a3), `HSA_OVERRIDE_GFX_VERSION=11.0.2`, `amdgpu.cwsr_enable=0`.
The same token ids for both engines, greedy, one pass, 32 write tokens at each depth, both engines warmed first.
K/V: q8_0 for the 27B, f16 for Nemotron. llama.cpp: `-ngl 99 -fa on --parallel 1 -b 2048 -ub 512 -c 24576`; the
Prism fork (ROCm build) for the 27B, upstream b10700 (ROCm build) for Nemotron. Read / write tokens/s:

| Model | Context | Kindi | llama.cpp ROCm |
|---|---:|---:|---:|
| 27B | 512 | 103.7 / 10.61 | 60.5 / 10.33 |
| 27B | 8,192 | 95.6 / 9.99 | 58.2 / 8.52 |
| 27B | 24,000 | 80.8 / 8.75 | 54.5 / 6.13 |
| Nemotron (experimental) | 512 | 185.9 / 19.11 | 669.5 / 24.02 |
| Nemotron (experimental) | 8,192 | 391.9 / 20.03 | 598.7 / 21.36 |
| Nemotron (experimental) | 24,000 | 239.4 / 17.01 | 476.6 / 19.09 |

With the 27B, Kindi reads 1.5 to 1.7 times as fast at each context and writes 1.03, 1.17 and 1.43 times as fast.
Kindi's Nemotron support is experimental, and we are still optimizing it; on this chip llama.cpp is faster with it.

## Quadro K6000, 12 GB

2026-10-25. Quadro K6000 12 GB (Kepler GK110, sm_35) passed through to a KVM virtual machine on a dual Xeon E5-2673 v4 host, NVIDIA driver 470.256.02. The same model files and the same token ids for both engines, greedy, 32 tokens of write at each depth. Kindi 0.3.0 development build (597d2d8) in the CUDA 10.2 image, q8_0 K/V for the 27B, f16 for Nemotron. llama.cpp: Vulkan builds (the Prism fork for the 27B, upstream 1b29621 for Nemotron), `-ngl 99 -fa on`. Read / write tokens/s:

| Model | Context | Kindi | llama.cpp Vulkan |
|---|---:|---:|---:|
| 27B | 512 | 36.2 / 20.42 | 7.0 / 2.07 |
| 27B | 8,192 | 30.2 / 16.36 | 6.9 / 2.15 |
| 27B | 24,000 | 26.2 / 13.09 | did not finish in 1 hour |
| Nemotron (experimental) | 512 | 187.5 / 20.05 | 90.7 / 10.70 |
| Nemotron (experimental) | 8,192 | 182.4 / 19.08 | 33.8 / 9.80 |
| Nemotron (experimental) | 24,000 | 163.6 / 17.40 | 15.6 / 8.56 |

`kindi verify`: the 27B 4 of 4 token-exact; Nemotron 4 of 4 (the first 16 to 40 of 40 tokens agree). With the 27B,
Kindi reads 4 to 5 times as fast and writes 7 to 10 times as fast. Both engines were warmed first. The tuning settings that we tried (chunk size,
tile, split, blocking sync, batch) changed the speed by 3% or less.

## Quadro K420, 1 GB · Ternary-Bonsai-2-2B

2026-10-23. Kindi 0.2.0, ctx 24,576, f16 K/V. llama.cpp: the prebuilt Vulkan release,
`-ngl 99 -fa on -c 24640 -b 2048 -ub 512`, f16 K/V. Model file sha256 36061ad4….

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| 512 | 33.3 | 1.2 | 20.8 | 1.65 |
| 8,192 | 29.7 | crashed (Vulkan ErrorDeviceLost after 17 minutes) | 18.6 | crashed |
| 24,000 | 24.2 | not measured (it crashed at 8,192) | 15.3 | not measured |
| 32,768 | Kindi: does not fit (its maximum on 1 GB is 24,576) | loads; not measured | does not fit | not measured |
| 65,536 | does not fit | does not load (an 805 MB allocation fails) | does not fit | does not load |

## Quadro K1200, 4 GB, one card · 2B, 4B and 9B

2026-10-23 (Kindi) and 2026-10-22 (llama.cpp). Kindi 0.2.0, ctx 24,576, f16 K/V, chunked prompt reading. llama.cpp: Prism ML's fork,
Vulkan build, `-ngl 99 -fa on -c 24640 -b 2048 -ub 512 --cache-ram 0`, f16 K/V. Each model had a 1-hour limit.

| Model | Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---|---:|---:|---:|---:|
| 2B | 512 | 194.5 | 26.7 | 81.0 | 20.8 |
| 2B | 8,192 | 178.7 | 28.1 | 73.3 | 21.9 |
| 2B | 24,000 | 152.3 | 26.5 | 58.8 | 18.6 |
| 4B | 512 | 77.4 | 11.1 | 39.2 | 8.7 |
| 4B | 8,192 | 70.2 | 10.8 | 34.3 | 8.1 |
| 4B | 24,000 | 58.9 | 10.2 | 26.4 | 6.9 |
| 9B | 512 | 41.7 | 5.8 | 23.4 | 5.9 |
| 9B | 8,192 | 39.2 | 5.7 | 21.5 | 5.6 |
| 9B | 24,000 | 35.6 | not finished (the 1-hour limit ended during this prompt) | 18.1 | not measured |

The largest context on the card:

| Model | Kindi | llama.cpp |
|---|---|---|
| 2B | 131,072 (all that was asked) | loads at 65,536 (larger was not tried) |
| 4B | 86,272 | loads at 65,536 (larger was not tried) |
| 9B | 47,360 | loads at 65,536 (Kindi does not fit there) |

## Two Quadro K1200 cards · Ternary-Bonsai-2-27B

2026-10-23 (Kindi) and 2026-10-21/22 (llama.cpp). Kindi 0.2.0, the tensor-parallel engine, q8_0 K/V. llama.cpp: Prism ML's fork, Vulkan
build, `-ngl 99 -sm layer -fa 1 -ctk q8_0 -ctv q8_0`, with its default layer split (an even split does not load).

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| short (Kindi: generation at position 128; llama.cpp: below position 150) | - | - | 12.6 | 1.69 |
| 512 (the same 512 token ids) | 19.5 | 1.63 | - | - |
| 1,000 | not measured | does not run (a 508 MB Vulkan allocation fails) | not measured | does not run |

Kindi's 512-token read is the best of 3 from an empty state. llama.cpp's was measured on 2026-10-22 with llama-server
`-ngl 99 -sm layer -fa on -ctk q8_0 -ctv q8_0 -c 1024`, with the token ids sent to `/completion`.

At 1.3 tokens/s, a 24,000-token prompt takes about 5 hours on llama.cpp, so there are no deeper rows.

## GeForce GTX 1080, 8 GB · Ternary-Bonsai-2-27B

2026-10-22 (Kindi) and 2026-10-21 (llama.cpp). Kindi 0.2.0, one-card plan engine, 5-trit weights, int8 chunked prompt reading, ctx
65,536, q8_0 K/V. llama.cpp: Prism ML's fork, CUDA build for sm_61, `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 2048
-ub 512 -c 49280`. Model file sha256 53107f53….

| Context | Kindi read | llama.cpp read | Kindi write | llama.cpp write |
|---|---:|---:|---:|---:|
| 512 | 254.0 | 98.1 | 27.9 | 19.4 |
| 8,192 | 227.1 | 96.2 | 25.7 | 18.1 |
| 32,768 | 169.1 | 84.9 | 19.7 | 14.3 |
| 49,152 | 144.6 | 78.6 | 17.2 | 12.6 |
| 64,000 | 127.8 | out of memory (it does not load at 64K) | 15.4 | out of memory |

## GeForce RTX 3080, 10 GB · Ternary-Bonsai-2-27B

2026-10-22, one physical card (cluster GPU 7), both engines. The cluster cards have a power limit of 190 W; 380 W is
the default of the card. Kindi 0.2.0 (the 190 W columns; the 380 W columns: see below), tensor-core engine,
`KINDI_TC=1 KINDI_FD=1 KINDI_ATT_TC=1`, exact head, q8_0 K/V. llama.cpp: Prism ML's fork 1180918, CUDA 12.4 build
for sm_86. llama-bench `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -r 1`: read pp512 / 3000 / 12000 / 22000 from an empty
cache, write tg32 at a depth of 1K / 32K / 64K / 90K. No CPU limit (`--cpus 4.0`).

| Context | Kindi read 190 W | llama.cpp read 190 W | Kindi read 380 W | llama.cpp read 380 W |
|---|---:|---:|---:|---:|
| 512 | 1,093 | 936 | 1,361 | 1,217 |
| 3,000 | 1,029 | 995 | 1,282 | 1,265 |
| 12,000 | 944 | 926 | 1,171 | 1,170 |
| 22,000 | 855 | 855 | 1,050 | 1,074 |

In the 380 W columns, Kindi is the dev build release-tcspec6 (2026-10-22), whose exact path is the same code as 0.2.0.

| Depth | Kindi write 190 W | llama.cpp write 190 W | Kindi write 380 W | llama.cpp write 380 W |
|---|---:|---:|---:|---:|
| 1,000 | 63.2 | 52.2 | 66.4 | 56.5 |
| 32,768 | 56.3 | 41.0 | 60.1 | 47.0 |
| 65,536 | 49.8 | 28.0 | 54.7 | 37.9 |
| 90,000 | 44.0 | 23.5 | 51.2 | 32.9 |

In the 380 W columns, Kindi is the dev build release-tcspec6 (2026-10-22), whose exact path is the same code as 0.2.0.

Kindi uses about 0.02-0.05 CPU cores per stream (0.2.0); llama.cpp about 0.77, and both stay within 5% on one core
(`--cpus 1.0`, llama.cpp `-t 1 -tb 1`; release-tcspec6 run).

Agent prompts: 30 greedy and 20 sampled (temperature 0.7) pi prompts of 1,100 to 10,500 tokens, 128 tokens out,
through the servers (llama-server `-fa on -ctk q8_0 -ctv q8_0 -c 81920 -np 1 --cache-ram 0`). Write speed, greedy /
sampled:

| Engine | 190 W | 380 W |
|---|---:|---:|
| llama.cpp | 48.7 / 54.9 | 53.2 / 60.8 |
| Kindi, exact | 61.8 / 57.7 | 66.9 / 58.6 |
| Kindi + the 2B draft model (`KINDI_SPEC=draft2b`) | 99.9 / 91.3 | 107.9 / 90.9 |
| llama.cpp + the 2B draft model (`-md`) | does not load | does not load |

In the 380 W column, Kindi is the dev build release-tcspec6 (2026-10-22): its exact path is the same code as 0.2.0, and its draft-model row is from before 0.2.0's draft-model fix.

With the draft model, Kindi's greedy output is the same as without it (30 of 30 prompts). llama.cpp 1180918 does
not load the 2B as a draft model for the 27B: the two files have different Hadamard metadata.

## Why llama.cpp runs with Vulkan on the Kepler and Maxwell cards

The K420 has compute capability 3.0 (sm_30). CUDA 11 and later cannot compile code for it, and current llama.cpp
does not build with CUDA 10.2, the last toolkit for this card. The Vulkan backend runs on it with all layers on the
GPU, so the Vulkan build is the fair comparison. On the K1200 (sm_50) host, only the Vulkan build was available.

## Other published results

We looked for a published measurement of a modern language model on a Quadro K420, and on any Kepler GPU of the
sm_30 generation (GT 640/730, K420/K600). We found none. To our knowledge, Kindi is the fastest way to run a
current-generation LLM on this card. We have not had it independently verified. If you know of a faster result,
please tell us.

## Two-token speculative decode on the CPU (i9-9980HK, 8 threads)

The batched pair-code GEMV is wired into a real decode path rather than measured in isolation. `kc_step2` runs two
tokens through the whole forward pass with one weight stream (`rows8_b2`), and `kindi.cpu.generate_spec` drives it
from an n-gram (prompt-lookup) drafter.

**Exactness.** The two-token pass is bit-identical to two `kc_step` calls: `max|dlogit| = 0.000e+00` at every position
tested, and the recurrent state after the pass (every layer's `M`, every layer's conv, the fp16 K and V caches) is
`array_equal` to the state two sequential calls leave. `generate_spec` therefore reproduces plain greedy decoding
token for token, which is asserted in every run below.

The shipped single-token path was re-gated against llama.cpp after the `gdn`/`attn` split that made the pass possible:
`kindi --pack ~/.kindi/packs/bonsai-2b verify` with `KINDI_DEVICE=cpu` gives **4/4 prompts PASS, 40/40 tokens each,
"ALL PASS: token-exact with llama.cpp"**. Note what that gate does and does not cover: `verify` drives `generate`,
which is `feed`/`step`, so it proves the refactor left the ordinary decode untouched, but it never calls `kc_step2`.
The speculative path's guarantee rests on the bit-exactness above, which is why that was checked to the logit and to
the recurrent state rather than only to the token.

**Pass cost.** Two tokens in one pass cost 1.35x one token, so a pass buys 1.35 tokens' worth of time and returns
1 + acceptance tokens. That makes the break-even acceptance **0.35** and the ceiling at perfect acceptance 1.48x.

| | ms | note |
|---|---:|---|
| one-token `kc_step` | 42.13 | 23.7 tok/s |
| two-token `kc_step2` | 56.92 | 1.351x the pass |

The ceiling is 1.48x, not 2x, because only the products batch: the RMS folds, the GDN recurrence, the attention
bodies and the embedding still run per token, and they double. With the GEMV at ~86% of a token and the pair kernel
at 1.35x, `2/(0.86/1.35 + 0.14*2)` is exactly the 1.35 measured.

**End to end, decode loop only, medians of 3, prefill excluded:**

| context | plain | spec | | acceptance | identical |
|---|---:|---:|---:|---:|---|
| repetitive tokens | 18.75 tok/s | 22.07 tok/s | **+1.177x** | 0.86 | yes |
| degenerate (random prompt) | 23.55 tok/s | 36.09 tok/s | **+1.533x** | 1.00 | yes |

The two rows bracket what the mechanism does: the gain is the acceptance rate, nothing else. The random-prompt row
is not a good model reading, it is a degenerate one - the model emits a fixed token, the drafter predicts that same
token from the suffix index, and acceptance is 1.00, at the ceiling. The repetitive-prompt row is the more realistic
case at 0.86. Neither row is natural text, so treat the range as a bracket and not as a claim about prose.

One measurement mistake worth recording: the first run of this benchmark timed the prefill along with the decode
loop, which diluted the ratio to +1.11x. Both arms paid the same 47-token prefill, so the *shape* was right but the
number was not. The table above times only the decode loop.

**On real text, through the tokenizer.** Both rows above are synthetic token ids, which the paragraph above is careful
to say. `kindi bench-spec TEXTFILE` runs the same two arms over prompts cut from a real file and tokenised by the pack's
own tokenizer, so the prompts are prose or code rather than token ids. 128 new tokens per prompt, 48- and 256-token
prompts (the prompt is the drafter's history), at 8 and 4 prompts per cell:

| corpus | prompt | acceptance | plain | spec | | identical |
|---|---:|---:|---:|---:|---:|---:|
| prose (`README.md`) | 48 | 0.764 (381/499) | 18.2-22.4 tok/s | 23.1-25.1 tok/s | **+1.27x** (1.12-1.35) | 8/8 |
| prose (`README.md`) | 256 | 0.645 (169/262) | 14.4 tok/s | 20.6 tok/s | +1.44x | 4/4 |
| code (`kindi/cache.py`) | 48 | 0.753 (387/514) | 18.6 tok/s | 20.5 tok/s | +1.10x | 8/8 |
| code (`kindi/cache.py`) | 256 | 0.719 (200/278) | 20.7 tok/s | 25.8 tok/s | +1.24x | 4/4 |

Every generation was token-identical to plain greedy. Acceptance on natural text is **0.65-0.76, i.e. 1.9-2.2x the 0.35
break-even**, on prose and code alike, so this is not a code-only win. The +1.27x median is also what the pass cost
predicts: a pass returns 1 + acceptance tokens for 1.351x the cost of one token, so `(1 + 0.76)/1.351 = 1.30`, and the
cells bracket 1.10x to 1.44x around it. Read the per-cell ratios as +-0.15 and the acceptance figures as exact - they
are deterministic for the same text.

The earlier version of this table read +1.11x to +1.35x at 0.75-0.87 acceptance, from a harness that estimated the
prefill with a separate timed read and subtracted it. At 48-token prompts prefill is a quarter of the arm's time, so
that subtraction moved the ratio by up to 0.2x. `bench-spec` removes prefill by cancellation instead - each arm also
runs the same prompt at `--tokens 1`, giving `(t(n) - t(1)) / (n - 1)` - and those are the numbers above. The old
figures are superseded, not merely noisier.

One sample had to be thrown away: the first `code 256` run read **2.594x**, which is above the 1.48x ceiling and so
impossible. Per-prompt timings showed the cause was the host, not the kernel - the plain arm decoded the same 128
tokens in 3.76 s on one prompt and 7.92 s on the next. The re-run of that cell gives 1.24x. This VM's plain-arm rate
has swung 14-25 tok/s within one session; that is why the ratios move and why a single cell is not evidence.

**Decision: wired into the serving loop, off by default (`KINDI_CPU_SPEC=1` turns it on).** The acceptance data
removes the numerical doubt - real text is 1.9-2.2x the break-even, and the one-token draft did not change the output
in any generation run here. What blocked the default was not decode economics but the stop path: the server's token
loop handles EOS, stop strings, `</think>` boundaries and tool-call parsing one token at a time, so a two-token pass
has a second token that may have to be thrown away.

That is now implemented. `Runner.spec_begin/spec_next/spec_stop` draft through `kc_step2` and keep the loop's shape
(one `nxt` per emitted token); a request that stops on the token of an accepted pass calls `kc_state_load`, so the
recurrent state and `Runner.tokens` end up exactly where one `step` would have left them. `kindi spec-check` is the
test: a stop by token id, a stop by stop string, a stop one token later, a run whose drafts are all rejected, and the
same request through `server.Kindi.generate` with the pass off and on. Removing the rollback from `spec_stop` makes
the two stop cases fail while the tokens still match - the failure mode below, now caught.

It stays off by default for two reasons, both about evidence rather than mechanism. The economics above were measured
on this host and should be re-read on the machine that serves. And two loop paths still feed tokens of their own: a
thinking budget, and the loop stop that sets one, close the thinking mid-stream - so speculation is skipped for those
requests rather than restructured around them.

**Why the state snapshot matters.** A rejected draft has to roll back to the state after the pass's first token, so
`kc_step2` snapshots every GDN layer between the two tokens (19 MB, copied by one thread under `omp single`). The
first version copied on every thread with no barrier; the group after the copy then ran `gdn_mid` for the second
token, which rewrites `conv` almost immediately, so the copy raced it and all 18 conv layers rolled back wrong. The
output was still token-identical (the argmax did not move) and only the state comparison caught it. Rolling back by
replaying the first token instead would cost 0.76 of a pass against the snapshot's ~3% of one.

## How to reproduce

`kindi verify` checks the output against llama.cpp. `kindi bench` gives the write speed at several positions and
the prompt reading speed. [CARDS.md](CARDS.md) gives the settings for each card.

The real-text speculative numbers above come from `kindi bench-spec`: `KINDI_DEVICE=cpu kindi bench-spec README.md`
for prose, `kindi bench-spec kindi/cache.py` for code, with `--prompt-len` and `--prompts` for the two prompt sizes. It
is a separate command from `kindi bench` because that one needs a GPU - it imports CuPy to read memory - so the CPU
speculative path had nothing in-tree to hang a benchmark on.

The serving loop's side of it is `KINDI_DEVICE=cpu kindi spec-check`, which needs no pack (it builds the synthetic
fixture) and exits 1 if the pass changes the tokens or leaves the state anywhere but where a single step would.
