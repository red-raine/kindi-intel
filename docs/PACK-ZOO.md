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

**What the re-fit did to the distribution.** The two files' `blocks` are the same ternary codes, so every behavioural
difference between Bonsai and Ingot is the re-fitted scales and norms. 79 tokens of prose through both converted packs
on the same positions (`tools.kl logits`, one pass each):

| | Bonsai | Ingot |
|---|---|---|
| NLL on the next token | 2.506888 | 2.513293 (**+0.006405** nats/token) |

`KL(Bonsai || Ingot)` is **0.026373** nats (median 0.021214 per position, max 0.100221 at position 64) and
`KL(Ingot || Bonsai)` **0.025880**, so neither distribution sits inside the other - the re-fit is not a rescaling of one
into the other, it is a different model that agrees on the top token 88.5% of the time. Two things follow, and neither
is a verdict on Ingot: re-deriving every scale is a change of this size, not a no-op; and on this window it is 0.0064
nats *worse* at predicting prose than the build it came from. The card's own claim is `KL(teacher || model) 0.08276`
against the king's `0.08409`, which needs Qwen3.8-27B to check and is not checkable here - a smaller distance from the
file it was derived from is the miner's lever, not evidence that the lever pointed the right way.

## Re-deriving the scales without touching the codes: `kindi refit`

Ingot is Bonsai with every code kept and every scale moved. That leaves an experiment this page could not run with two
GGUFs: take Bonsai's codes, solve each block scale against Ingot's weights, and see how much of the difference the
scales carry. `kindi refit` is that, and the solve is closed form rather than a search - one pass over the tensor, no
training, no code rewritten:

```
for each 128-weight block, with the target weights w and the block's fixed codes c in {-1, 0, +1}:
    s = <w, c> / <c, c>          # least squares: the projection of w onto the direction c picks out
    s = max(s, 0)                # a block scale is a magnitude, so a negative is clipped and counted
    all-zero codes -> keep s     # no direction to fit, and a division by zero otherwise
```

```bash
python -m kindi.refit --pack B:/models/packs/bonsai-27b --target B:/models/packs/ingot-27b \
                      --out B:/models/packs/bonsai-27b-refit --link
```

Nothing is trained, so the refit cannot make the pack bigger or slower: the code and float files are hard links with
`--link` (byte copies without it), and only `dpm.bin` (210 MB) and `dscale.bin` (170 KB) are new. On the 27B pair the
run took 29 minutes and compared all **6,717,440,000** code bytes: **0 differ** from the target, so the fit is asked to
reproduce a scale the target already holds. It does, exactly - with the codes fixed and `w = c*s`, `<w,c>/<c,c>` is `s` -
and the output pack's **209,920,000** scales decode **bit-identically** to Ingot's. 0 blocks had no codes and 0 clipped at
zero; relative RMSE to the target went **0.21593 -> 0.0**. The refit's own level tables then cost nothing (250 levels at
most, worst block 0.0%), where the 2B pack's 8-bit grid costs 0.75% of a scale in its worst block - and the control
below prices that storage at +0.00000 nats, because a refit's values are the target's table entries to begin with.

So the written refit is *Ingot's ternary weights with Bonsai's float tensors*: the scales of one file, the codes and the
norms of the other. That is a controlled decomposition rather than a better build, and four paired walks over the same
192 tokens of the same prose file - the same windows, the same tokenisation - price each lever apart:

| arm | ternary weights | float tensors | NLL | delta vs Bonsai | 95% interval | closer on |
|---|---|---|---:|---:|---|---:|
| Bonsai 27B | Prism's PTQ scales | Bonsai's | 1.2907 | - | - | - |
| `bonsai-27b-refit` | Ingot's, bit-identical | **Bonsai's** | 1.3121 | **+0.02143** | [-0.02261, +0.07851] | 91 of 192 |
| Ingot 27B | Ingot's | Ingot's | 1.3310 | **+0.04024** | [+0.01418, +0.10469] | 99 of 192 |

The two deltas add exactly, 0.02143 + 0.01881 = 0.04024, because every arm walked the same tokens: the middle row
against the last is the float tensors on their own, **+0.01881, interval [+0.00362, +0.06186]**, and it is the one that is
resolved. It is also closer on 123 of the 192 tokens while worse on average - the norm cost is carried by the rows it
hurts rather than spread over all of them.

**The middle row changes two things, and the storage half of it is zero.** A refit rewrites each scale's *value* and
its 8-bit *table* entry together, so the walk of Bonsai against the refit prices both at once. `kindi refit --retable`
separates them: it keeps every code and every scale value and only rewrites the values through the refit's own
encoder. Built from Bonsai it changes **0 of 209,920,000** scale values - 402 of 402 tensors come back exact, worst
relative error 0.0 - so the pack decodes bit-identically to the one it came from. Priced over the same 192 tokens it
costs **+0.00000 nats, interval [+0.00000, +0.00000]**, with 0 tokens closer on either side and both NLLs at 1.2907:
a null control, and the harness returns one. The whole +0.02143 is the scale values themselves:

| lever | packs walked | nats | 95% interval |
|---|---|---:|---|
| 8-bit level tables (storage) | Bonsai -> `bonsai-27b-retable` | +0.00000 | [+0.00000, +0.00000] |
| scale values | `bonsai-27b-retable` -> `bonsai-27b-refit` | +0.02143 | [-0.02261, +0.07851] |
| float tensors (norms) | `bonsai-27b-refit` -> Ingot 27B | +0.01881 | [+0.00362, +0.06186] |
| the file difference | Bonsai -> Ingot 27B | +0.04024 | [+0.01418, +0.10469] |

The middle row is the refit's own walk, recorded once: the retable pack decodes identically to Bonsai, so walking one
against the refit *is* walking the other. That the storage costs nothing is not luck about this encoder. Every code
byte matches the target, so the solve returns the target's own scale in every block - `<w, c> / <c, c>` with
`w = c*s` is `s` - which makes all 209.9 million fitted values members of Ingot's 256-entry table before a table is
built. The encoder then reproduces any 256-level set exactly, which is also what it does to Bonsai's scales. Two
files, the same 8 bits an entry, each carrying its own numbers without loss: the difference between them is which
numbers those are.

That reframes what the refit measures, and the middle row is what makes it visible. It is not fitting: with the codes
frozen it is a transplant of Ingot's scales onto Bonsai's codes, so +0.02143 is the price of Ingot's magnitude set on
this text with no alphabet, no table and no code difference in the way. What it cannot answer is whether a better fit
exists, because a fit is not what happened - the values it wrote were already the target's.

Three things follow, and none of them flatters the Ingot recipe. The whole file difference on this text is **+0.040
nats**, roughly half scales and half norms. The scales' half, +0.021 with an interval from -0.023 to +0.079, **does not
resolve a direction**: re-deriving all 209.9 million block scales against the other file's weights is, on this evidence,
neither a gain nor a measured loss. And the level table inside that arm is not a lever at all: the control prices the
refit's storage at +0.00000, because a fitted scale *is* the target's table entry. A build that has to invent levels
pays for its grid, which is what the 2B pack's +0.00182 nats pooled over 4,096 tokens measures, a tenth of the +0.021
and inside its interval ([QAT-CONVERSION.md](QAT-CONVERSION.md)) - but it is not what the refit arm is paying.

The 79-token `tools.kl` walk further up put Ingot at +0.006405 nats on this file, six times smaller than the +0.040 above.
Different protocol, shorter window; the two agree on the sign and the paired walk is the one with an interval. What both
leave is the code: the trits are identical in the two files, so no re-derivation of the scales can move a weight the
codes have already rounded wrong, and [QAT-CONVERSION.md](QAT-CONVERSION.md) measures that floor directly - refitting a
scale on drifted weights recovers at most 1.2 points of retention, never 1.0.

Two limits on the tool itself. It cannot repair a code, as above. And it needs a target pack of the same model in the
same rotation basis, and refuses rather than guesses: another Hadamard rotation, another width, a tensor the target does
not have, or a source whose `pcodes.bin` stages its own copy of the scales are errors, not warnings.

## Why this belongs next to the K/V measurements

The 2-bit V experiment found one thing that helps a 2-bit container: a scale per 32 values instead of per head -
19% lower KL for 21% more cache, on trained cache values. The weight side of this family is the same argument one
level up. `PQ2_0` is `PTQ1_0` with 8 more bits a block spent on a denser code, and the winner of the Ingot round
spent none of its budget on the code: it kept the trits and re-derived the scales and norms against a BF16 teacher.
Whatever the 27B can still gain at 2 bits, this evidence says it is in the scale rather than the alphabet - though the
refit above puts a size on that lever, and on 192 tokens the scales' half is +0.021 with an interval that covers zero.

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

The distribution comparison is one command on the two converted packs (about 15 minutes here, most of it loading a
27B pack twice; the walk itself is 47 s a pack):

```bash
python -m tools.kl logits --a packs/bonsai-27b --b packs/ingot-27b --tokens 96 --ctx 256 \
  --json packs/kl-logits-bonsai-ingot.json
```

The refit, and what it is worth:

```bash
python -m kindi.refit --pack packs/bonsai-27b --target packs/ingot-27b --out packs/bonsai-27b-refit --link
# the storage control: same codes, same scale values, only the tables rebuilt
python -m kindi.refit --pack packs/bonsai-27b --out packs/bonsai-27b-retable --retable --link
# the four walks: storage (Bonsai -> retable), scales (retable/refit), norms (refit -> Ingot), both (Bonsai -> Ingot)
python -m tools.nll --pack packs/bonsai-27b --against packs/bonsai-27b-retable \
  --corpus corpora/scale-gate/prose/prose.txt --tokens 192 --block 24 \
  --json packs/nll-retable-vs-bonsai-27b-prose.json
for pair in "bonsai-27b bonsai-27b-refit" "bonsai-27b-refit ingot-27b" "bonsai-27b ingot-27b"; do
  set -- $pair
  python -m tools.nll --pack packs/$1 --against packs/$2 --corpus corpora/scale-gate/prose/prose.txt \
    --tokens 192 --block 24 --json packs/nll-$1-vs-$2.json
done
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
