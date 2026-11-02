# Configuration

Kindi reads its settings in this order - later sources win:

1. built-in defaults;
2. a config file: `--config FILE`, else `$KINDI_CONFIG`, else `./kindi.conf`, else `~/.kindi/kindi.conf` (INI, one
   `[kindi]` section). Unknown keys are an error, so typos do not pass silently;
3. environment variables `KINDI_<KEY>` (for example `KINDI_PORT=9000`, `KINDI_API_KEY=...`);
4. command-line flags (`--pack`, `--state-dir`, `--ctx`, and for `serve`: `--host`, `--port`, `--api-key`, `--head`).

`kindi info` prints the final values and the file that each value came from. The Docker image sets
`KINDI_PACK=/data/pack`, `KINDI_STATE_DIR=/data/state` and `KINDI_HOST=0.0.0.0`.

## Keys

### Paths

| key | default | |
|---|---|---|
| `pack` | `~/.kindi/pack` | the converted model (`kindi convert` writes it) and, optionally, `head_clusters.npz` |
| `state_dir` | `~/.kindi/state` | the kernel cache (`kernels/`), saved prompt openings (`openings/`) and the head mode set by `kindi head` |

### Server

| key | default | |
|---|---|---|
| `host` | `127.0.0.1` | address to listen on; `0.0.0.0` serves your network - then also set `api_key` |
| `port` | `8093` | |
| `api_key` | (empty) | if set, every `/v1` request needs `Authorization: Bearer <api_key>`; `/health` stays open |
| `model_name` | `auto` | the model id `/v1/models` reports; `auto` takes it from the pack (bonsai2-2b, bonsai2-4b or bonsai2-9b); clients may send any model name |

### Engine

| key | default | |
|---|---|---|
| `ctx` | `24576` | context length in tokens (prompt + answer). For GPU memory, see the notes below the table |
| `head` | `exact` | `exact` = full output head (token-exact with llama.cpp); `fast` = shortlist head (needs `kindi fast-head`). `kindi head exact\|fast` overrides it at run time |
| `fast_clusters` | `256` | clusters the fast head scores exactly (of 1,024). Fewer is faster and less exact (see the notes below the table) |

Notes:

- `ctx`: GPU memory is ~12 KiB per token for the fp16 K/V cache, on top of ~570 MB for weights and buffers.
- `ctx`: 24,576 leaves ~115 MB of the K420's 980 MB free. Lower it if something else needs the card.
- `fast_clusters`: 128 clusters found the full head's top token 98.1 % of the time, 256 clusters 99.4 %.

### Prompt cache

| key | default | |
|---|---|---|
| `checkpoint_every` | `256` | interval between recurrent-state checkpoints (host RAM, ~19 MB each). Smaller = less to read again after a change, more RAM |
| `prefix_max` | `8` | saved openings kept on disk (each ~12 KiB per token + 19 MB; ~80 MB for a 5,000-token opening) |
| `prefix_min` | `512` | the shortest opening that Kindi saves |
| `park_max` | `3` | other conversations kept in host RAM (each: its checkpoints + its K/V cache) |

### Generation defaults

Used when a request does not set them: `max_tokens` (`1024`), `temperature` (`0.7`; `0` = greedy), `top_p`
(`0.8`), `top_k` (`20`). These are the model publisher's recommended non-thinking settings.

## Environment variables for the engine

These variables have no config key. Use them to test or to compare.

| variable | default | |
|---|---|---|
| `KINDI_DEVICE` | `gpu` | `cpu` = the CPU engine (2B, 4B, 9B). Kindi also uses the CPU engine when CuPy is not installed |
| `KINDI_CPU_CACHE` | `~/.cache/kindi` | the folder for the built CPU library (`libkcpu.so`) |
| `KINDI_TP` | (auto) | `1` = the plan engine, `0` = the single-card engine. The 27B always uses the plan engine |
| `KINDI_CARDS` | (all, at most 2) | the number of cards for the plan engine |
| `KINDI_CHUNK` | `256` | prompt tokens per chunk; `0` = one token at a time. The default is `128` on Kepler (sm_30) |
| `KINDI_CHUNK_MIN` | `32` | the smallest part of a prompt that goes through a chunk |
| `KINDI_CHUNK_Q8` | `1` | sm_61 and later: int8 activations in the chunk products (dp4a). `0` = fp32 |
| `KINDI_CHUNK_TILE` | (auto) | `128` = the 128 x 128 tile for the 2-bit chunk products |
| `KINDI_CHUNK_ATT` | `flash` | `exact` = chunk attention with the same arithmetic as one token at a time (slower) |
| `KINDI_G5L` | `1` | sm_61 and later: the 5-trit lane-per-byte GEMV; `0` = one row per thread |
| `KINDI_DENSE_SPACING` | `256` | the spacing of the dense checkpoints in an agent opening; `0` = none |
| `KINDI_STREAM_LOAD` | `1` | the plan engine loads the cached 5-trit weights in 256 MB parts |
| `KINDI_MEM_RESERVE_MB` | `96` (`32` for the plan engine) | GPU memory that the context fit keeps free |
| `KINDI_FD` | `0` | `1` = the split flash-decode attention of the 27B on one card (with `KINDI_TC=1`) |
| `KINDI_D1` | `0` | `1` = the one-pass RMS fold (Hadamard block 512 or 1024, with `KINDI_TC=1`) |
| `KINDI_ATT_TC` | `0` | `1` = the attention of the prompt chunks on the tensor cores (sm_80 and later, with `KINDI_TC=1`) |
| `KINDI_CPU_TIER` | (auto) | force one CPU kernel tier: `scalar`, `avx2` or `q8`. Without it the engine probes at load and keeps the fastest tier that is bit-exact |
| `KINDI_CPU_Q8` | `0` | the int8 activation prototype: every CPU GEMV through the int8 form. Not a production path; [CPU-PERF-2026.md](CPU-PERF-2026.md) has the measurements |
| `KINDI_CPU_Q8G` | `128` | the activation group width of that prototype: `128`, `64`, `32`, `16`, `8`, `4` or `2` columns per scale |
| `KINDI_CPU_Q8C` | `0` | `1` = a static per-channel input scale instead of the group form, applied to the ordinary fp32 GEMV. Needs a calibration pass (`Engine.q8_cal_begin/end`), so it does nothing until scales are loaded |
| `KINDI_CPU_Q8GUARD` | `0` | a top-1/top-2 gap under this value sends the decision to an fp32 head recomputed from the same folded input; `Engine.guard_stats()` counts both. Measured on the 48-prompt sweep it does not reduce the divergence count - [CPU-PERF-2026.md](CPU-PERF-2026.md) says why |
| `KINDI_CPU_Q8TAIL` | `0` | compute the last `k` blocks and the head in fp32 and the rest in int8. `0` is the plain int8 path and `24` reproduces fp32 token for token; the divergence count plateaus part-way, see [CPU-PERF-2026.md](CPU-PERF-2026.md) |
| `KINDI_CPU_KVQ` | `0` | the CPU K/V cache format, a storage-prototype: `0` = fp16, `1` = ternary at 2 bits with one scale per (position, head), `2` = q8_0 blocks of 32, `3` = 4-bit with a fitted per-head scale, `4` = 2 bits on four levels, `5` = 4 bits as two passes of that 4-level code. Not a production path; [CPU-PERF-2026.md](CPU-PERF-2026.md) has the measurements |
| `KINDI_CPU_KVQ_K`, `KINDI_CPU_KVQ_V` | (the format) | the same choice a side at a time, so the keys can stay q8_0 while the values are 4-bit |
| `KINDI_CKPT_VRAM` | `0` | `1` = checkpoints in card memory; the copy to the host waits until the server is idle (one card) |
| `KINDI_HOST_RAM_CHECK` | `1` | before the weights of a card load, the plan engine stops when the free host RAM is too small. `0` = no check |
| `KINDI_BLOCKING_SYNC` | `1` | the host thread sleeps while it waits for the card, and does not use a CPU core. `0` = it waits in a loop (the driver default) |
