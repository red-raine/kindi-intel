# What the 27B ternary packs differ in: codes, scales, and the rest

Four public checkpoints of the same shape showed up in one week: Prism ML's
[Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), UkisAI's
[Swift-Bonsai-2](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF), and
[Ingot-XXIV-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-XXIV-27B-Q2) - all three are `qwen35` at
(64 layers, 5120 wide, 17408 ffn, 24 heads, 4 kv heads, 16 ssm groups), which is the profile `kindi convert` calls
`Ternary-Bonsai-2-27B`. This page records what actually differs between them, measured from the files rather than
from the model cards, because the answer decides which of them are worth converting and which are the same file
under two names.

The short version: **the ternary codes are identical in all of them; every 128-weight block scale is different;
and the float tensors are where the tuning lives.** That is the same lever the K/V work measured - see
[CPU-PERF-2026.md](CPU-PERF-2026.md#kv-cache-quantisation-q8_0-costs-nothing-the-small-alphabets-do-not-save-the-model).

## The two containers

A GGUF carries a tensor table, and the gap between two consecutive tensor offsets is that tensor's size on disk.
Dividing by the number of 128-weight blocks gives the block cost directly, without decoding anything:

| type | ggml id | bytes per 128 weights | bits per weight | contents |
|---|---|---|---|---|
| `PTQ1_0` | 143 | 28 | 1.75 | 128 trits at 5 a byte in 26 bytes (`qs[24] + qh[2]`) + fp16 block scale |
| `PQ2_0` | 142 | 34 | 2.125 | 32 bytes of 2-bit codes + fp16 block scale |

Both are measured: all 257 measurable tensors of the Ingot `PQ2_0` file come out at exactly 34 bytes a block and
every ternary tensor of the `PTQ1_0` files at 28, with no outliers. The `PQ2_0` file is 7.21 GB against 5.95 GB for
exactly this reason - the same weights in a container with 8 bits more a block, which decodes faster on a GPU and
is the reason both are published. `kindi convert` reads `PTQ1_0` only.

Each file holds 851 tensors: 402 ternary, 353 `F32` and 96 `BF16` (the norms, the gated-delta-rule decay and
convolution tensors, `ssm_a`, `ssm_dt.bias`). The Prism hadamard metadata is identical in every file: version 1,
1024-value blocks, `normalized-sylvester-walsh-hadamard`, 401 rotated tensors, 28,672 sign values, and the same
`sign_values` array to the last element. So all three files apply the same rotation to the activations, and the
engine's folded-weight path is unchanged.

## How to compare them without downloading six gigabytes

The tensor table sits at the start of the file (`data_start` is 11,120,992 bytes into these), so the header plus a
few 2 MB windows at chosen offsets is enough to compare the weights. Two traps:

- **Offsets are relative to `data_start`, and `data_start` is not the same in every file.** Swift's is 11,121,024 -
  32 bytes later - because its metadata is serialised differently even though the parsed key/values are equal. Read
  the same absolute offset from each file and every block decodes against the wrong bit boundary: the trits then
  differ on **66.6%** of weights, which is exactly the 2/3 you get from two independent ternary streams, and it is
  how this comparison first reported Swift as an unrelated model. Always add each file's own `data_start`.
- **Uncorrelated is not the same as different.** Two genuinely different ternary re-quantisations of the same fp16
  teacher agree more than 33% of the time, because they agree wherever the weight is far from a rounding boundary.
  A flat 2/3 disagreement is a shift or a permutation, not a different model.

The same method reaches any tensor in the file for 2 MB: read `data_start + offset`, decode, compare.

## What the files actually differ in

Each row below is a 2 MB window (about 9.6M weights) or the full tensor for the 5120-value norms.

**Bonsai 2 and Swift are the same weights.** Over `output.weight`, `blk.0.attn_gate`, `blk.32.attn_qkv` and
`blk.63.ffn_up` the codes differ on 0.00% of trits and the fp16 scales on 0.00% of blocks; five fp32 norms across
depth (`output_norm`, `blk.0.attn_norm`, `blk.31.post_attention_norm`, `blk.63.attn_norm`,
`blk.63.post_attention_norm`) come out bit-identical, max `|a-b|` = 0. The one exception is `blk.63.ffn_up`, where
0.01% of trits and 60.7% of scale blocks differ but 79.9% of the scale values agree to 1e-3 - a repack rounding
difference in one tensor, not a different model. On this evidence Swift's published `PTQ1_0` is Prism's weights in
a new container, and the "39.8% fewer thinking tokens" the card claims cannot be coming from these bytes.

**Ingot keeps every code and moves everything else.** Against Bonsai, over the same four windows:

| tensor | trits that differ | 128-block scales that differ | scales within 1e-3 |
|---|---|---|---|
| `output.weight` | 0.00% | 99.97% | 0.1% |
| `blk.0.attn_gate` | 0.00% | 99.58% | 1.2% |
| `blk.32.attn_qkv` | 0.00% | 100.00% | 0.0% |
| `blk.63.ffn_up` | 0.00% | 99.76% | 0.8% |

and the float tensors move by up to a third relative:

| tensor | max abs difference | max relative | L2 norm, Bonsai vs Ingot |
|---|---|---|---|
| `output_norm` | 0.329 | 21.1% | 139.35 vs 143.50 |
| `blk.0.attn_norm` | 0.303 | 30.4% | 69.50 vs 69.87 |
| `blk.31.post_attention_norm` | 0.140 | 34.2% | 71.75 vs 73.23 |
| `blk.63.attn_norm` | 0.140 | 14.2% | 81.52 vs 81.00 |
| `blk.63.post_attention_norm` | 0.226 | 23.5% | 88.34 vs 89.21 |

That is what the Ingot card says in words - "the same ternary codes, with the rest of its weights re-tuned by the
miner to bring it closer to Qwen3.8-27B" - and it is the reason a subnet can rank copies of one model at all: the
codes are frozen, so the only free variables are the magnitudes. Every block scale in the sampled windows was
re-fitted (99.6% to 100% of them, and the ones that survive are within 1e-3), and the norms move with them.

## Why this belongs next to the K/V measurements

The 2-bit V experiment found one thing that helps a 2-bit container: a scale per 32 values instead of per head -
19% lower KL for 21% more cache, on trained cache values. The weight side of this family is the same argument one
level up. `PQ2_0` is `PTQ1_0` with 8 more bits a block spent on a denser code, and the winner of the Ingot round
spent none of its budget on the code: it kept the trits and re-derived the scales and norms against a BF16 teacher.
Whatever the 27B can still gain at 2 bits, this evidence says it is in the scale, not the alphabet.

## Reproducing

```bash
# the header plus a window, per file, is enough to compare weights
python - <<'PY'
import numpy as np
from kindi.gguf import GGUF
from kindi.convert import trits_from_raw
a, b = GGUF("bonsai.gguf"), GGUF("ingot.gguf")          # the local files, or the heads
off = a.data_start + dict((t[0], t[3]) for t in a.tensors)["output.weight"]
n = 2 << 20
wa = np.fromfile(a.path, np.uint8, n, off).reshape(-1, 28)
wb = np.fromfile(b.path, np.uint8, n // 28 * 28, b.data_start + off).reshape(-1, 28)
print("trits differ: %.2f%%" % (100 * (trits_from_raw(wa[:, :26]) != trits_from_raw(wb[:, :26])).mean()))
print("scales differ: %.2f%%" % (100 * (wa[:, 26:28] != wb[:, 26:28]).any(axis=1).mean()))
PY
```

To convert one (the 27B profile is supported, PTQ1_0 only):

```bash
python -m kindi convert Ternary-Bonsai-2-27B-PTQ1_0.gguf --out /b/models/packs/bonsai-27b
```

## Maple-Preview is a different engine's problem

`deepgrove/maple-preview` is in the same round of releases and is not this family: a 20B-A1B Mixture-of-Experts in
fp16 safetensors (nine shards, 40 GB) for the Transformers runtime, 24 layers with 256 experts and 8 active, a
131,072-token context, and a custom architecture whose reference implementation needs Triton and FlashAttention.
There is no GGUF, no `PTQ1_0` file, and `kindi convert` reads GGUF only, so nothing in this repository can load it -
the ternary tags on its card describe the training, not the file.

## The DeepSeek expert pack: nothing to load, one thing to take

[`meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest)
is the same idea at a different scale, and it is not convertible here: 40 shards of fp16 safetensors (about 140 GB),
no GGUF, `qwen35` nowhere in sight, and a custom runtime (`goethe`'s `ternary_runtime` on SGLang) that unpacks it.
Its `ternary_config.json` names the format - `ternary-g128-hadamard-gptq`, group 128, `pack_layout: w123`, hidden
5120, intermediate 2304 - which is the family this engine already lives in: a Hadamard rotation folded into the
weights, ternary codes in 128-weight groups, one scale a group. The difference is scope. It ships **routed experts
only**; attention, router, shared experts, Engram and the draft model stay in the base checkpoint, and unused
experts fall back to FP4 from it. That is why it is 140 GB of shards but 40 files - one layer a file, with a
`manifest-layer-NN.json` beside it.

What is worth taking is not the format but the accounting. The repository carries three JSON files that answer a
question this repository could not answer before:

| file | what it holds |
|---|---|
| `nll-eval.json` | teacher-forced NLL over a held-out corpus, per domain, with token counts |
| `nll-reference.json` | the same walk for the reference (FP4) model, so the ternary cost is a subtraction |
| `nll-walk.json` | calibration and held-out NLL, and the two digests |

Both carry `code_sha256` and `corpus_sha256`. That is the part that matters: an NLL means nothing without saying
which text produced it and which code computed it, and their card's numbers (overall 0.7646 against the FP4
reference's 0.5628, so +0.2018; by domain code +0.105, math +0.086, general +0.415) are only comparable because
those digests pin the corpus.

`tools/nll.py` is that walk for a Kindi pack, writing the same key names so the artefacts sit in the same shape:

```bash
python -m tools.nll --pack PACK --corpus DIR --json out.json          # a domain per subdirectory
python -m tools.nll --pack PACK --corpus DIR --reference ref.json     # and the subtraction
```

On the 2B pack it reports, over 133 tokens of two domains, code 1.1164 and prose 4.6979 nats for 2.9475 overall -
and repeats to the digit. The number to check first is the one it prints beside that: chance is `ln(248320)` = 12.42
nats, which is also what a scoring loop aligned one token off will produce. Two limits are worth knowing before
reading a delta off it: the corpus in `nll-reference.json` upstream is not distributed, so their numbers and ours
are not comparable to each other, only within their own corpus; and teacher-forced NLL needs one head evaluation a
token, which is generation speed - minutes per hundred tokens on a 27B.
