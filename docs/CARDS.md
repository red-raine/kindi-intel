# How to run Kindi on each card

The installer ([INSTALL.md](INSTALL.md)) uses these settings. It selects them from the cards that it finds.

Blocking sync (`KINDI_BLOCKING_SYNC`, on by default) lets the host thread sleep while it waits for the card. It
costs 2-3% of the decode speed on the GTX 1080. With one server on its own card, set `KINDI_BLOCKING_SYNC=0`
(the installer does this). The RTX 3080 profile keeps it on: there, many servers share few CPU cores.

This page gives the settings for each tested card. It also gives the speeds that we measured with these settings.
All the cards use the same Docker image (CUDA 10.2, CuPy 9.6). The `kindi` folder of this repository is the code.

To make a pack, use `kindi convert` (see the README). Put the pack in `$HOME/.kindi/<pack>`. In the container, it
is at `/data/<pack>`. The examples serve on `127.0.0.1:8093`.

```
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
  -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data -e HOME=/tmp \
  -e KINDI_PACK=/data/pack -e KINDI_STATE_DIR=/data/state ... kindi:0.1.0
```

Replace `...` with the variables in the section for your card. To use one card of two, set `--gpus device=N`.

## Quadro K420, 1 GB (Kepler, sm_30)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-2B |
| variables | `KINDI_CTX=24576` (the default) |
| generation | 22.9 tokens/s (fast head, 2,000 tokens of context); 20.8 tokens/s (exact head, 512 tokens) |
| prompt reading | 31.6 tokens/s at 3K tokens, 28.1 at 12K (one token at a time: 25.3 and 23.4; dev build of 10-21) |

- The prompt path uses chunks of 128 tokens on this card. Then the context stays at 24,576 tokens.
- To read prompts one token at a time, set `KINDI_CHUNK=0`.

## Quadro K420, 2 GB (Kepler, sm_30)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-4B (the installer's choice), or the 2B |
| variables | `KINDI_CTX=23808` for the 4B (the context that fits), `KINDI_CTX=65536` for the 2B |
| generation | 2B: 21.2 tokens/s (exact head), 24.8 (fast head); 4B: 9.8 tokens/s |

## Quadro K1200, 4 GB (Maxwell, sm_50)

| item | value |
|---|---|
| models | 2B (`/data/pack`), 4B (`/data/pack4b`) or 9B (`/data/pack9b`) |
| variables | `KINDI_CTX=24576` |
| generation (exact / fast head) | 2B 71.6 / 88.3 tokens/s, 4B 35.0 / 42.2, 9B 20.7 / 24.9 |
| prompt reading at 3K tokens | 2B 186.6 tokens/s, 4B 74.5, 9B 40.0 |

- The card in an x16 slot is faster than the card in an x4 slot.

## Two Quadro K1200 cards (tensor parallel)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B (`/data/pack27b`) |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CTX=24576`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=4096`, `KINDI_REPEAT_STOP=1` |
| docker | `--gpus all` (the engine uses both cards), `--memory 12g` |
| generation | 12.6 tokens/s |
| prompt reading | 20.9 tokens/s at 512 tokens, 18.4 at 22K (dev code of 10-21, the chunk-path copy) |

- The K/V cache is q8_0. The output matches llama.cpp with `-ctk q8_0 -ctv q8_0`.
- Set the host to text mode. A desktop session turns on the CUDA watchdog, and the watchdog stops the exchange
  kernels.

## Quadro K6000, 12 GB (Kepler, sm_35)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` |
| generation | 20.4 tokens/s at 512 tokens, 16.4 at 8K, 13.1 at 24K |
| prompt reading | 36.2 tokens/s at 512 tokens, 30.2 at 8K, 26.2 at 24K |

- The card needs the NVIDIA 470 driver. The 470 driver does not report the compute capability, so the installer
  reads it from the card name.
- With all of the 12 GB, the engine fits a context of 156,672 tokens. The installer sets 65,536.
- Nemotron also runs on this card (experimental): 20.1 tokens/s of generation at 512 tokens. The installer does not
  offer it on Kepler cards.

## GeForce GTX 1080, 8 GB (Pascal, sm_61)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` |
| generation | 28.4 tokens/s at a short context, 16.0 at 60K |
| prompt reading | 249 tokens/s at 512 tokens, 237 at 3K, 213 at 12K, 188 at 22K |

- The first start writes the 5-trit weights into the pack (`g5n_*.bin`, 5.6 GB). This takes about 12 minutes.
  Later starts take about 30 seconds.
- The prompt products use int8 activations on this card. For fp32, set `KINDI_CHUNK_Q8=0` (110 tokens/s at 512
  tokens).
- On a host with more than one GPU, set `CUDA_VISIBLE_DEVICES` to the GTX 1080.

## GeForce RTX 3080, 10 GB (Ampere, sm_86)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=131072`, `KINDI_MEM_RESERVE_MB=128`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=1024`, `KINDI_PARK_MAX=1`, `KINDI_REPEAT_STOP=1` |
| docker | `--memory 14g` (the first load needs it) |
| context | 110,592 tokens (the fit to the free memory) |
| generation | 47-49 tokens/s at a short context, 29 at 64K |
| prompt reading | 491 tokens/s at 512 tokens, 381 at 22K |

These figures are from kindi-rel, 10-21, not the 0.2.0 build.

Tensor cores and speculative decoding (sm_80 and later). This is the profile of the RTX 3080 cluster servers, and
the installer uses it:

| variable | value | |
|---|---|---|
| `KINDI_TC` | `1` | int8 tensor-core products and CUDA graphs for the decode |
| `KINDI_FD` | `1` | split flash-decode attention: the K/V is read once per query, in chunks of 512 positions |
| `KINDI_ATT_TC` | `1` | the attention of the prompt chunks on the tensor cores |
| `KINDI_SPEC` | `draft2b` | speculative decoding with the 2B as the draft model (also `ngram` or `both`). Off by default |
| `KINDI_SPEC_K` | `3` | the draft tokens per verify pass (default 4) |
| `KINDI_SPEC_PACK` | the 2B pack | for `draft2b` (for example `/data/pack2b`) |
| `KINDI_SPEC_WINDOW` | `8192` | the context of the draft model |
| `KINDI_TC_TMAX` | `4` | the most tokens in one verify pass |
| `KINDI_CKPT_VRAM` | `1` | a checkpoint is a copy in card memory; its copy to the host waits until the server is idle |
| `KINDI_MEM_RESERVE_MB` | `160` | the memory that the CUDA graphs need after the context fit |
| `KINDI_DENSE_SPACING` | `2048` | the spacing of the checkpoints in the opening of a conversation |
| `KINDI_CHECKPOINT_BUDGET_MB` | `1280` | the host memory for checkpoints and parked conversations (`1024` on the agent servers) |
| `KINDI_PARK_MAX` | `1` | parked conversations (`0` on the agent servers: one conversation each) |
| `KINDI_PREFIX_MAX` | `2` | saved openings on disk |

With this profile: context 84,992 tokens; at 190 W on agent prompts 99.9 tokens/s greedy and 91.3 sampled; prompt
reading 1,093 tokens/s at 512 to 855 at 22K.

- With `KINDI_TC=1`, the server reads prompts in passes of up to `KINDI_TC_TMAX` tokens.
- Speculative greedy output is the same as the output without speculation.
- The cluster runs seven servers on one 4-core host with `--memory 6g`. One server on a computer of its own can use
  the default budgets.

Q4_K and Q4_0 are the two formats with no tensor-core prompt GEMM on this card: their products run on dp4a. Whether
Marlin's INT4 kernel can fill that gap, and what blocks it, is [INT4-MARLIN.md](INT4-MARLIN.md).

## CPU (x86-64 with AVX2)

| item | value |
|---|---|
| models | 2B, 4B or 9B |
| variables | `KINDI_DEVICE=cpu`, `OMP_NUM_THREADS=<cores of one socket>`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |

- The CPU engine needs gcc with OpenMP. It builds `libkcpu.so` at the first start (`KINDI_CPU_CACHE`).
- `kindi serve` (the OpenAI-compatible server) is the only command that needs `fastapi` and `uvicorn`; chat, verify,
  bench and convert run without them. `bash setup-server.sh` makes that venv: `~/.kindi/server-venv`, built with
  `--system-site-packages` so it uses the numpy the system already has, with fastapi and uvicorn in it. It needs no
  root and no `--break-system-packages`, it is safe to run again, and it prints the command to start the server.
- `kindi bench-spec FILE` measures the speculative decode path on a real text file: draft acceptance and the
  decode rate against plain greedy, with prefill cancelled out. Acceptance belongs to the window, not the file -
  seven files read 0.36-0.82 at one prompt length - so quote the flags and the cuts with the number. `kindi bench`
  is the wrong command here: it needs a GPU.
- `kindi spec-check` tests the same pass where the server uses it: an accepted draft stopped on by token id and by stop
  string, stopped one token later, every draft rejected, and one request through the server's own loop with the pass
  off and on. Each case must end with the recurrent state and the token history where a single step would leave them.
  It needs no pack (it builds the synthetic fixture); `kindi --pack DIR spec-check` runs it on a real one. The serving
  pass is off by default: `KINDI_CPU_SPEC=1` turns it on for greedy requests, and it is skipped for a request whose
  thinking budget can close the thinking mid-stream. Measured through the server it is 0.93-1.02x at 269-273-token
  prompts and 1.26-1.37x at 65-token ones, so it stays off for agent-length traffic.
- Finer int8 activation scaling does not rescue the int8 path. On the same 48 prompts and 7,680 decisions, per-64 and
  per-32 groups move the logit drift by 12% and the divergence count by nothing (13, 12 and 16 of 48 prompts), because the
  step is a group maximum and barely shrinks with the group; a calibrated per-channel scale is 6.5x worse, because one grid
  per column cannot follow the token. `docs/CPU-PERF-2026.md` has the measurements, and `KINDI_CPU_Q8G` and
  `KINDI_CPU_Q8C` select the schemes.
- `kindi verify-tier` checks that the kernel tiers (scalar C, AVX2, int8 activations) compute the same thing, and
  reports what each costs. The engine runs the same gate at load and picks the fastest tier that is bit-exact, so
  `KINDI_CPU_TIER` is only needed to force one. On Windows the same file is built as `kcpu.dll` with MSVC (see
  `kindi/cbuild.py`); tiers agree within a build, and greedy output agrees across gcc and MSVC builds.
- `kindi moe-check` compares the Mixture-of-experts FFN against a NumPy implementation on a synthetic pack, and with
  `--paged` measures resident against paged experts on both staging paths: rebuilding the pair codes from the trits,
  and the memcpy a `kindi convert --paircodes` pack allows. All of them must produce the resident tokens. A float
  checkpoint becomes a pack with `kindi qat-pack` and is measured with `kindi drift`; see `docs/QAT-CONVERSION.md`.
- The output is token-exact with llama.cpp.
- Size `OMP_NUM_THREADS` to the **physical** cores, not the logical ones. A token is roughly 425 barriers, and a
  barrier costs under a microsecond at 4 and at 8 threads on this host, but milliseconds at 16, which is the point
  where a virtualised CPU (WSL2, a VM) can no longer co-schedule every vCPU on a physical core. More threads than
  that is slower, not faster: 16 threads measured 0.24 tokens/s against 25 at 8.
- The CPU profiler: `kc_get_times(double*)` and `kc_reset_times()` in `kcpu.c` report the seconds per stage
  (0 embed, 1 norm+fold, 2 gemv, 3 gdn-other, 4 attn-other, 5 ffn-other, 6 head) since the last reset. The gemv is
  about 90% of a token, so it is the only stage worth optimising.
- **Throw away the first run after a rebuild.** Run-to-run spread is about 12% and the first run is roughly 30%
  slower than the third (17.0 and 19.6 cold against 23.6, 26.2 and 29.2 warm for the same binaries). Take three or
  more medians, and never compare a cold run to a warm figure.
- The engine is **issue-bound, not memory-bound**. The GEMV runs at the same 18-19 GB/s whether its weights fit in
  32 MB of cache or stream from DRAM (26 GB/s of streaming read is achievable), and per core it converts 3.2 MACs per
  cycle. Cache and bandwidth work therefore buys nothing: the pair table's 32 KB of L1 thrash, pinned away entirely,
  costs 0%. To go faster, remove instructions per weight word. Thread binding measured a wash here, so leave
  `OMP_PROC_BIND=close` alone rather than tuning it. Roadmap and the twenty triaged techniques:
  `docs/CPU-PERF-2026.md`.
- The two-token kernel confirms the diagnosis from the other side, and says what is *not* the lever. 8 to 16
  independent accumulator chains changes nothing at any working set, opening a second group window is neutral, and 4
  chains instead of 8 is 5% faster at every size - so the chain-count axis is flat to negative and the kernel is not
  ILP-bound. Its cache-resident rate is 13.8 GB/s of weight stream against 14.3 GB/s at 432 MB, so residency buys it
  nothing either. Fewer permutes per weight word is the only lever left
  (`docs/CPU-PERF-2026.md#the-cache-resident-shortfall-is-not-ilp-measured`).
- `perf` is not installed on this WSL image and needs root. `~/kt/perf/gb.c` reads the PMU through
  `perf_event_open` instead, one event set per thread, and is how to re-measure the GEMV (working-set sweep, table
  pinning, prefetch on/off, binding).
- A kernel change needs an interleaved A/B, not two runs: build both variants with distinct symbol suffixes, load
  both into one process, and alternate them, reversing the order every rep. Five extra instructions per weight word
  measured 0.55-0.63x, while pinning the whole pair table into 2 KB of L1 measured 1.09x and inside the noise. Also
  check that `unattended-upgrades` is not running - it cost one run in this project its numbers.
