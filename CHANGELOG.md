# Changelog

## Unreleased

### New

- **`kindi refit`.** Keeps every ternary code and re-derives every 128-weight block scale against a target pack: the
  closed form `s = <w, c> / <c, c>` a block, clipped at 0 (a block scale is a magnitude), and a block with no codes
  keeping the scale it came in with. No training and no code moves, so the pack's size does not change. It writes a new
  pack - `dpm.bin`/`dscale.bin` re-derived, the code and float files copied or hard-linked - records the target and
  what it compared in `kindi_pack.json`, and re-reads its own output through `packio` before returning. Priced with
  `tools.nll --against` on the 27B pair whose codes are byte-identical: [docs/PACK-ZOO.md](docs/PACK-ZOO.md).
  `--retable` is its storage control - the same codes and the same scale values, only the 8-bit tables rebuilt - which
  on that pair moves 0 of 209,920,000 values and prices at +0.00000 nats, so the refit's delta is the values it writes
  and not the encoding it writes them in.
- **`kindi recipes` and `--scan`.** Reads a GGUF header (family, depth, width, experts, context, rope scaling, tensor
  types) without touching tensor data, and prints the conversion plan for a job - `coder`, `agent`, `rag`, `search`,
  `science`, `chat` - with the reasons a file cannot support it. `--scan DIR` does it for every model under a
  directory. [docs/RECIPES.md](docs/RECIPES.md).
- **`--scheme gsq`.** Fits the ternary codes and the group scale together (Gumbel-Softmax relaxation, then
  least-squares alternation). 11.4% below `lsq` in weight error on heavy-tailed matrices, level on Gaussian ones.
- **`--budget BYTES`.** Chooses which tensors stay f32 under an exact byte budget, by dynamic programming with the
  fractional bound reported when the table is too large to solve exactly.
- **`kindi fetch kaggle:OWNER/DATASET[/MEMBER]`.** Kaggle has no per-file endpoint and no hash, so the dataset archive
  is downloaded, the named member is extracted, and the limits are stated.
- **`tools.bitnet_gemm.copytest`.** Answers whether a read rate is the medium, the Windows path or a cache by putting
  one file through `ReadFile` unbuffered, `ReadFile` buffered, `CopyFileExW` and 4 KB random in one process, plus a
  read-during-commit curve. It retires this tree's 1.67 GB/s and "~9 ms expert read": a write-back cache takes writes
  at 1.84 GB/s and commits them at 68 MB/s, so reads of a 2 GB file that is still being committed run at 0.14 GB/s for
  the 30 s that takes, then return to 3.68. With the drive idle an expert read is **3.3-4.0 ms**, not 9.5, so the
  double-buffer ceiling is 33% rather than 17% and one expert read is made free by 2 tokens rather than 5-6. The copy
  itself settles nothing upward - it is read+write on one medium, so its one-way share is a floor on the device, never
  a ceiling on reads. [docs/TERNARY-LUT-AND-PAGING-2026.md](docs/TERNARY-LUT-AND-PAGING-2026.md).
- **`--read MS` on `prefetch_budget` and `cluster_budget`.** The read rate those two models were hard-wired to is the
  commit-bound one; both now take the rate they should be run with. At the idle 3.7 ms the prefetch crossover moves
  from B=8 to B=4 and its best batch from B=4-6 to B=2, and the clustering crossover from B=8 to B=2 (chosen groups)
  or B=4 (random) - the ordering and the conclusion survive, the batch numbers do not.
- **`tools.bitnet_gemm.batch_stream`.** Runs the batched expert-paging pipeline instead of modelling it: a reader thread
  fills the next 15.8 MB expert while the production kernel multiplies the current one into the batch. At batch 8
  streaming is 2.3-7x worse than residency on this host - a read costs ~20 ms with the batch computing rather than the
  9.5 ms it costs alone, and the captured router fetches 4.4 experts per token at that size. [docs/TERNARY-LUT-AND-PAGING-2026.md](docs/TERNARY-LUT-AND-PAGING-2026.md).
- **`tools.bitnet_gemm.libcompare`, and `kc_rows8_gemv` in the CPU library.** Puts the pair-table kernel and the T-MAC
  LUT on one matrix in one process, which is the comparison the paging note had been making across sources. The two
  agree to 1.9e-07 of their largest output, so parity holds; at one thread the LUT is 1.8x faster (0.219 against
  0.396 ms on 2048 x 1920), while the pair table at 16 threads is 2.4x faster than the LUT. [docs/TERNARY-LUT-AND-PAGING-2026.md](docs/TERNARY-LUT-AND-PAGING-2026.md).
- **`tools.expert_merge.prune`.** Which experts a layer can simply not have, under an exact per-layer byte budget,
  decided from the captured router log and solved by the RCO knapsack - `drop` for 0 bytes and the expert's routing
  mass as its error, `keep` for its bytes and no error. On Qwen3-Coder-30B-A3B (48x128 experts, 208.5 MB a layer)
  keeping 96, 80 and 64 of 128 costs 7.31, 6.11 and 4.89 GB of the 9.77 GB of expert weights and gives up 0.6%, 2.1%
  and 5.6% of the routing mass; 40 of 128 experts in some layers were never routed to at all. The plan is applied as a
  router mask, so the two arms of the NLL gate share one loaded pack; on 94 held-out tokens the 96-expert plan scores
  -0.0171 nats/token (-0.0625..+0.0469) and the 64-expert plan -0.0712 (-0.2194..+0.0595), so neither costs anything
  measurable and neither interval settles the sign. [docs/EXPERT-PRUNE.md](docs/EXPERT-PRUNE.md).
- **`tools.expert_merge.mergepack`.** Writes the pack the merge question needs: each co-activated group becomes one
  expert, every non-expert tensor copied byte for byte, the router one pooled row a group, `expert_count` patched to
  match - so the before/after is two models read by the same reader, not two settings. Co-activation is partitioned by
  average linkage under a size cap (64 groups of 128 experts is 64 pairs; 17.7% of a token's routing weight lands in a
  group another slot also picked) and fused by mass-weighted mean, because `merge.py`'s alignment gate refuses every
  routed pair. K-quants decode here but do not encode, so both arms are written at one type through one path and
  `--groups 128 --expert-type Q4_0` is the unmerged control. On 94 held-out tokens: control 4.9053, merged 64 pairs
  **6.2641**, i.e. **+1.3588 nats/token** (+0.8984..+1.8029) and ~3.9x the perplexity for half the expert bytes
  (8.16 GB against the control's 16.32 GB). [docs/EXPERT-MERGE.md](docs/EXPERT-MERGE.md).
- **`tools.bitnet_gemm.cluster_budget`.** Measures whether clustering experts cuts paged bytes enough to stream at
  batch 1, from a captured router log rather than from an assumption about overlap: 12 clusters take the streaming
  crossover from B=64 to B=8 and the table from 58.8 GB to 5.5 GB, but a random partition of the same size reaches
  B=8 too, and batch 1 still reads 5.24 matrices against the 1.52 it would need. It also reports what the merge
  replaces: 59.5% of a token's routing weight at 12 clusters would be computed by an averaged matrix.
  [docs/TERNARY-LUT-AND-PAGING-2026.md](docs/TERNARY-LUT-AND-PAGING-2026.md).

### Fixed

- **A format that costs zero bytes is representable in an RCO plan.** `kindi.rco.plan` read a per-format byte count with
  `e["bytes"].get(f) or _bytes(...)`, so a format priced at 0 was treated as unpriced and fell through to a lookup that
  does not know it. Pruning is the caller that needs the zero: a dropped expert costs nothing. The price is now read
  with `is not None`.
- **A CPU pack is checked against its own bytes before a pointer is taken.** A truncated `bcodes.bin`, `f32.bin` or
  `pcodes.bin` used to hand the kernel a short buffer; a model width that is not a multiple of 32, a row length that
  loses a 128-column scale group, a row count that is not a multiple of 8 or a staged offset that is not word-aligned
  used to be read past the row's end or dropped silently. All six are now refused at load, naming the file, the tensor
  ("the MoE router") and the bytes.
- **A GGUF header length the file cannot back is refused by name.** A 79 KB file whose header claims a 2.6-exabyte
  tensor name used to end in a bare `MemoryError`; every stated length is now checked against the bytes that remain.

## 0.3.0 (2026-10-28)

### New

- **AMD Radeon 780M graphics.** A ROCm image (`docker/Dockerfile.rocm`) runs the 2B, the 27B and Nemotron on the
  780M. The installer finds the card and builds the image. [docs/AMD.md](docs/AMD.md) gives the settings.
- **Quadro K6000.** The installer runs the 27B on this 12 GB Kepler card. [docs/CARDS.md](docs/CARDS.md) gives the
  settings.
- **NVIDIA Nemotron 3.5 Bolt E4B (experimental).** Kindi has its own decoder for this model, with tool calls and
  thinking. It runs on one card, or on two cards of the same kind in tensor parallel. Use `--model nemotron` with
  the installer.
- **More prompt cache.** Nemotron has its own prompt cache (`KINDI_G4_PROMPT_CACHE`). With the Bonsai models, the
  server can also resume a request inside a parked conversation (`KINDI_PARK_BRANCH`, on by default on ROCm).
  `KINDI_PROMPT_CACHE=0` turns the Bonsai cache off. [docs/PROMPT-CACHE.md](docs/PROMPT-CACHE.md) gives the
  results, and `examples/prompt_cache_bench.py` is the test.
- **A setup menu.** The installer shows each setup that the machine supports, the recommended setup first.
  `--auto` takes the recommended setup. `--config N,M` runs several setups side by side, and `--config NxK` runs
  one setup on K cards of the same kind. Each server gets its own container and port.
- **CPU servers next to GPU servers.** The installer offers a 2B server on the CPU on each machine.
- **gpu-unlock.** The installer finds the GeForce RTX 30 cards that NVIDIA limits to half-rate 16-bit arithmetic.
  It offers to unlock them with the [gpu-unlock](https://github.com/Zagitta/gpu-unlock) module by Zagitta. It never
  does this without a question.
- **`kindi models` and `kindi use`.** A model catalog, the models that fit this machine, and a resumable download
  with a sha256 check.

### Changed

- Sampling uses the order of llama.cpp by default: top-k, top-p, then temperature.
- The server returns OpenAI `logprobs` and `top_logprobs` for Nemotron.
- With the 2B draft model, a guard stops speculation when it is slower than plain steps (`KINDI_SPEC_GUARD=0` turns
  the guard off).
- The installer detects a server that restarts in a loop, and it checks that the Kindi modules load after the build.
- The installer makes one pi provider for each server.
- On a 2 GB Kepler card, the installer asks for a context of 23,808 tokens for the 4B. This is the context that fits.

### Fixed

- `model_name = auto` also reads the model size from the GGUF file name. Before, a 27B pack could report
  `bonsai2-2b`.
- `/health` shows a git commit only for a Kindi checkout. Before, a copy of Kindi in another git folder showed the
  commit of that folder.
- The installer works on machines with one GPU and with Kepler workstation cards.
- The cache brings a parked conversation back in place, and the server continues after an engine error.
- The tensor-core path starts on cards after the RTX 30 series (RTX 40, RTX 50 and Hopper). Before, the 27B
  with the installer's tensor-core profile and the Nemotron model stopped at start-up with `CUDA_ERROR_INVALID_PTX`.
- Kindi starts on RTX 50 cards. Before, no kernel compiled on these cards, because CuPy compared the card versions
  as text.

## 0.2.0 (2026-10-22)

### New

- **More models and cards.** Kindi runs Ternary-Bonsai-2-2B, -4B, -9B and -27B (the PTQ1_0 GGUFs). Tested on the
  Quadro K420 (1 GB and 2 GB), the Quadro K1200 (one card, or two in tensor parallel), the GeForce GTX 1080, the
  GeForce RTX 3080 and the CPU. [docs/CARDS.md](docs/CARDS.md) gives the settings for each card.
- **A one-command installer** (`install.sh`, [docs/INSTALL.md](docs/INSTALL.md)). It finds the cards, selects the
  model and the settings, downloads the model and checks its sha256, converts it, runs `kindi verify` and starts the
  server as a Docker service. It works with the NVIDIA Container Toolkit or without it. `--dry-run` shows each
  command. A re-run upgrades or repairs, and `--uninstall [--purge]` removes Kindi.
- **The 27B.** A plan engine (`kindi/tp.py`) runs it on one card (GTX 1080 8 GB, RTX 3080) or on two cards (2x
  Quadro K1200, tensor parallel). The K/V cache is q8_0, as llama.cpp `-ctk q8_0 -ctv q8_0`.
- **Prompts in chunks.** The engine reads up to 256 prompt tokens in one pass. The tokens are the same as with one
  token at a time. `KINDI_CHUNK=0` turns it off.
- **5-trit weights** on Pascal and later (1.75 bits per weight). The GTX 1080 holds the 27B with a context of 65,536
  tokens.
- **RTX 3080 (sm_80 and later):** int8 tensor-core products and CUDA graphs (`KINDI_TC=1`), split flash-decode
  attention (`KINDI_FD=1`) and tensor-core attention for the prompt chunks (`KINDI_ATT_TC=1`). `KINDI_SPEC=draft2b`
  uses the 2B as a draft model; the greedy output is the same as without it.
- **A CPU engine** (`KINDI_DEVICE=cpu`) for the 2B, 4B and 9B. It is token-exact with llama.cpp and needs no CUDA.
- **Checkpoints in card memory** (`KINDI_CKPT_VRAM=1`, one card): the copy to the host waits until the server is
  idle.
- **`/health` shows the version and the source commit.** `kindi verify` uses the reference of the pack's model.
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** compares Kindi and llama.cpp on the same card, with the same model
  file and the same context.

### Changed

- The context length fits the free card memory (`KINDI_MEM_RESERVE_MB`).
- The prompt cache: the memory budget counts every checkpoint and every parked conversation, and there is a
  checkpoint at the end of the opening. Saved openings belong to one engine configuration, and the server does not
  resume an opening of another configuration. A damaged opening file does not stop a request.
- Blocking sync: while it waits for the card, the host thread sleeps and does not keep a CPU core busy. It is on by
  default on CUDA (`KINDI_BLOCKING_SYNC=0` turns it off). It costs 1-3% of the decode speed, so the installer turns
  it off for one server on its own card.
- Before the 27B loads on two cards, the engine checks the free host RAM and stops with a clear message when it is
  too small (`KINDI_HOST_RAM_CHECK=0` skips the check).
- `model_name = auto` (the default) takes the model id from the pack.
- The K420 comparison in the documentation now uses the same model file for llama.cpp (PTQ1_0). The 0.1.0 page used
  llama.cpp's Q2_0 file. With the PTQ1_0 file, llama.cpp generates 1.65 tokens/s on the K420; Kindi generates 20.8.

### Fixed

- The fast head always scores the special tokens (`</think>` and the end tokens). Before, it could fail to close
  the thinking.
- Host memory: the server frees the CuPy pinned-memory pool after the load, and freed checkpoints go back to the
  system. Before, a server grew with each request.

## 0.1.0 - first public release

- License: GNU AGPL-3.0. Commercial licenses are available (see COMMERCIAL-LICENSE.md).
- An engine for Ternary-Bonsai-2-2B on the NVIDIA Quadro K420. It generates 20.8 tokens/s with the exact head and
  23 tokens/s with the fast head. It reads prompts at 26.5 tokens/s. The context is 24,576 tokens in 1 GB.
- With the exact head, the output is token-for-token identical to llama.cpp (`kindi verify`).
- `kindi convert` makes an exact conversion from the PTQ1_0 GGUF.
- An OpenAI-compatible server with chat completions, streams, tool calls, an optional API key and live progress in
  `/health`.
- Thinking mode: the server sends the thinking as `reasoning_content` (in a stream, as `delta.reasoning_content`).
  `thinking_budget_tokens` sets a limit. At the limit, the server closes the thinking and the model gives its answer.
- When `max_tokens` stops a reply in the middle of a tool call, the server returns that text. The client can then
  see that the reply is not complete.
- A prompt cache: checkpoints of the recurrent state, parked conversations in RAM and saved openings on disk.
- An optional fast (shortlist) output head. You can change the head at run time (`kindi head`).
- A Docker image with the CUDA 10.2 / CuPy 9.6 stack that Kepler needs.
- When a client disconnects (timeout, Ctrl-C), the server stops the request at the next token. This is also true
  during a long prompt read.
