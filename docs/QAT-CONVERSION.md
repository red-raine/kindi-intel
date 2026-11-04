# Converting a float checkpoint to a ternary pack

`kindi convert` reads the GGUFs Kindi ships with: a `qwen35` model whose weights are already ternary (`PTQ1_0`, one
fp16 scale per 128-weight group). A quantisation-aware training run produces the opposite: **float** weights, trained
with that quantiser in the loop. `kindi qat-pack` runs the quantiser forwards so one of those checkpoints becomes a
pack, and `kindi drift` measures what it costs.

```
kindi qat-pack qat_final.pt --out ~/.kindi/qat255m        # least-squares scale; --scheme absmean for the GGUF one
kindi qat-pack qat_final.pt --scheme gsq                 # fit codes and scale together: 11% below lsq on heavy tails
kindi qat-pack qat_final.pt --budget 900000000           # exact choice of which tensors stay f32
kindi drift qat_step1000.pt qat_final.pt --groups 128,row,tensor
```

## What the pack holds

The same files `kindi convert` writes, in the same layout, so a pack from a checkpoint loads through the same code:

| file | content |
|---|---|
| `bcodes.bin` | 4 trits per byte (`t0 + 3 t1 + 9 t2 + 27 t3`), slot-major `[slot][row]` |
| `dpm.bin` | one fp16 scale per 128-column group per row, `[group][row]` |
| `f32.bin` | every non-ternary tensor verbatim (norms, biases, the GDN parameters, the conv kernel) |
| `index.json`, `bcodes_index.json` | offsets, rows, widths |
| `kindi_pack.json` | the geometry the checkpoint states, the quantiser, and `"hadamard": {"enabled": false}` |

Two things a float state dict cannot supply: a tokenizer, and a Hadamard rotation folded into the weights. The pack
records both as absent rather than inventing them. A checkpoint whose rows are not a multiple of 8 is padded with
zero rows (the pair-code layout walks rows in blocks of 8); `index.json` keeps the real shape in `dims`.

`kindi.qat-pack` verifies itself before it returns: it reads the pack back through its own index, dequantises every
ternary matrix, and compares it to the quantiser's output in memory. All 50 ternary matrices of the 255M checkpoint
below come back **byte for byte**, and `kindi.quant.selftest` additionally feeds one back through `cpu.paircodes()`,
the encoder the engine's loader uses, and checks the decoded matrix is the quantised one.

## Two scales, and why the second one exists

- **absmean** is the PTQ1_0 convention and the right scale for weights a ternary-aware run has already shaped into
  three levels: `s` = the group's mean `|w|`.
- **lsq** searches a fixed grid of multiples of absmean for the group and keeps the one with the smallest squared
  error. This is what an arbitrary float checkpoint needs, because a Gaussian's mean `|w|` sits well below the values
  that then clip.

The scale is rounded to fp16 *before* the trits are chosen, so what the engine runs is what the drift tool measured.

A third scheme, **gsq** (`--scheme gsq`), fits the codes and the scale together: a Gumbel-Softmax relaxation of each
weight's discrete choice, then the least-squares scale alternating to a fixed point, keeping the better end of the
trajectory. At 2 bits a weight on heavy-tailed (Student-t) matrices it lands 11.4% below `lsq` in weight error and is
level on Gaussian ones; the objective is the weights, not the layer output, and on that objective the alternation is
what wins. [`docs/RECIPES.md`](RECIPES.md) has the table and the limit.

The fp16 scale is one number per 128 weights; the other 16 bits a weight can buy are elsewhere in the file (norms,
embeddings, the head). `--budget BYTES` makes that choice exactly, by dynamic programming over which tensors stay f32
(`kindi.rco`), and reports the fractional bound and any gap.

## The drift curve

For every 2-D weight matrix `kindi drift` reports the relative Frobenius error of the ternarised weights, the cosine
similarity, and the relative error of the matrix's **output** on probe activations (unit-RMS Gaussian, one seed for
every checkpoint and scheme, 32 probes). Then it prints the curve by depth. The 255M checkpoint in
`bitqwen-qat-255m` (24 pure-GDN layers, width 1024, 255.5 M parameters, 50 ternary matrices):

| checkpoint | absmean 128 weight rel | lsq 128 weight rel | lsq output rel | lsq mean cos |
|---|---:|---:|---:|---:|
| `qat_stage1_final` | 0.6293 | **0.4876** | 0.4850 | 0.8650 |
| `qat_step1000` | 0.6285 | **0.4880** | 0.4853 | 0.8649 |
| `qat_final` | 0.6284 | **0.4881** | 0.4854 | 0.8649 |

Three things this says:

1. **The least-squares scale is worth 22%** of the error (0.6293 → 0.4876) and 8 points of cosine. That is the whole
   reason `qat-pack` defaults to `lsq`; absmean is kept because it is what the GGUF packs use.
2. **The scale granularity barely matters on this checkpoint**: per-128 groups 0.4880, per row 0.4921, per tensor
   0.5116. When the trits dominate the error, a finer scale cannot help.
3. **QAT progress does not move the drift**: the three checkpoints span the end of training and differ in the fourth
   decimal. They also differ by only 2.9% relative in weight space, and their weights are not three-peaked: 13.7% of
   `|w|` lies beyond twice the group mean `|w|`, and only 5.2% inside ±10% of it. A converged ternary-aware run looks
   very different, so this checkpoint is a float model that has been *trained with* the quantiser without collapsing
   onto three levels. Its pack is therefore a 0.49-relative-error approximation: it is a valid pack and a faithful
   measurement, not a model that should be served.

The depth curve (lsq, per-128) rises gently with depth - layer 0 at 0.4701, the middle of the stack at 0.5193, layer
23 at 0.5030 - and `out_proj` is the worst matrix in every layer. Embeddings are the easiest (0.4347) and the output
head one of the hardest (0.4884), which is the usual shape for post-hoc ternary quantisation.

## Re-ternarising an already-ternary pack

A different question from the one above: if a pack's weights move by a fraction of a scale step - a fine-tune, a
rounding pass, a re-pack - how much of the ternary structure survives re-quantisation? Measured on the served Bonsai-2B
pack (151 ternary matrices, 1880.1 M parameters, per-128 absmean scales), offsetting every dequantised weight by
`t*a` inside its own group and re-rounding on the **unchanged** grid:

| offset | retention, ties-to-even | retention, ties-away | what it is |
|---|---:|---:|---|
| `t < 0.5` | 1.0000 | 1.0000 | every code absorbed |
| `t = 0.5` | 0.6629 | 0.6736 | decided by the tie rule, not by the weights |
| `t > 0.5` | 0.3366 | 0.3366 | only the `+1` codes survive |

All 151 tensors follow the closed form exactly: `1 - P(-1)` at even ties, `1 - P(0)` at away ties, `P(+1)` beyond,
with a largest deviation of **1.1e-16** over ten offsets from 0.1 to 1.0 and both rules. Refitting the scale on the
drifted weights with absmean or least squares buys at most 1.2 points (0.6749 and 0.6640 at half a step against
0.6629 for the untouched grid) and cannot approach 1.0, so no choice of scale recovers a code sitting on a boundary.

Two consequences worth keeping:

1. **Half a step is the one offset that measures nothing about the model.** Every weight lands exactly on a rounding
   boundary, so the answer is the tie-breaker's, and the two plausible rules differ by a full point of retention
   because the code histogram is slightly asymmetric. Absorption is complete - and trivially complete - strictly below
   half a step, which is where a drift study has to sample to say anything about stability.
2. **The floor above half a step is a third, and that is arithmetic, not a quality bar.** The lowest retention
   anywhere in the sweep is 0.3348, and it equals the `+1` share of the codes.

The histogram behind those numbers: **0.337 / 0.326 / 0.337** for `-1` / `0` / `+1`, and flat over depth - the zero
fraction moves only between 0.3226 and 0.3284 across all 151 matrices (sd 0.0010), with layer 0 (0.3229-0.3284) inside
the range of layers 1+ (0.3226-0.3276). So a single layer's ternary statistics do represent the model's, which is what
makes layer-0-only measurements safe to generalise *for this quantity*; the drift curve above is a different matter,
where depth does move the error.

## Block scales at 8 bits (pack_format 2)

A ternary block is 128 weights and one fp16 scale - the absmean that the codes multiply. That scale was the widest
remaining thing in the pack: 29.4 MB of the 2B pack's 506 MB, 422 MB of a 27B pack's ~8 GB. Since the blocks of one
tensor are quantised from one distribution, their scales cluster, and 8 bits address 256 levels: the pack now stores an
index per block in `dpm.bin` (1 byte, same `[phase][row]` order) and each tensor's fp16 level table once, in
`dscale.bin`. `kindi/packio.py` owns both directions; `load_scales` hands every reader one value per block, so the
kernels, the pair-code path and the paging path were untouched.

Measured on the served **Bonsai-2-2B** pack, same GGUF either side:

| | pack_format 1 | pack_format 2 |
|---|---:|---:|
| weights (bcodes + scales + f32) | 506.31 MB | 491.70 MB |
| `dpm.bin` | 29.38 MB | 14.69 MB |
| level tables | - | 77 KB (151 tensors) |
| worst block's scale error | 0 | 0.75% (recorded in the manifest) |

**The gate: the NLL walk, 512 held-out tokens of *Automate the Boring Stuff*, `tools.nll`.**

| encoding | held-out NLL | delta | worst block | mean block |
|---|---:|---:|---:|---:|
| 2B pack, fp16 scales (v1) | 2.4837 | - | 0 | 0 |
| 8-bit, minimax grid (**shipped**) | 2.4871 | **+0.0034** | 0.75% | 0.19% |
| 8-bit, value-space k-means | 2.5416 | +0.0579 | 99% | 0.11% |
| 8-bit, log-space k-means | 2.5941 | +0.1103 | 57% | 0.11% |
| 8-bit, log-space k-medians | 2.7002 | +0.2165 | 66% | 0.12% |

That table is the finding, not just the result: the fit with the *worse average error* wins by 30x on NLL, because a
block's contribution is `d * sum(codes * x)` and a block whose scale is 57% off corrupts its whole 128-wide group,
while 256 blocks that are 0.2% off almost cancel. Density-adaptive fits (k-means, k-medians) minimise the *mean* error
in whatever space they are fitted and leave the sparse tail unserved; an even grid in log2 over each tensor's own
range minimises the *worst* block, and `packio.worst_error` reads that cap off the levels actually written - fp16
rounding moves an end level by a hair, so the promise is computed from the file, not from the request.

### The delta with an interval, over several corpora and longer sequences

One corpus and one sequence length give one number, and a reader cannot tell whether +0.0034 nats is the price of the
format or the noise of that afternoon. The same comparison is now paired token by token - both packs walk the same
tokenisation and the same windows - and the interval comes from a **moving-block bootstrap over 32-token blocks**
(`python -m tools.nll --pack A --against B --corpus DIR --ctx 1024 --block 32`):

| corpus | tokens | ctx | v1 (fp16 scales) | v2 (8-bit) | delta | 95% interval | tokens where v2 is closer |
|---|---:|---:|---:|---:|---:|---|---:|
| *Automate the Boring Stuff* | 512 | 512 | 2.4837 | 2.4871 | **+0.00339** | [+0.00206, +0.00485] | 181 / 512 |
| *Automate the Boring Stuff* | 1024 | 1024 | 2.2545 | 2.2605 | **+0.00598** | [+0.00446, +0.00766] | 316 / 1024 |
| this repository's `docs/*.md` | 1024 | 1024 | 2.9197 | 2.9197 | +0.00004 | [-0.00057, +0.00068] | 496 / 1024 |
| `kindi/*.py`, `tools/*.py`, `kcpu.c` | 1024 | 1024 | 2.6060 | 2.6063 | +0.00031 | [-0.00032, +0.00095] | 502 / 1024 |
| Go code-explanation dialogs | 1024 | 1024 | 2.6539 | 2.6548 | +0.00098 | [+0.00006, +0.00185] | 527 / 1024 |
| pooled, equal sizes | 4096 | 1024 | | | **+0.00182** | [-0.00262, +0.00627] | |

Corpora: the book from the start of chapter 1; every page under `docs/` plus the README; the engine and tool sources in
file order; and the first 64 KB of `ajibawa-2023__Go-Code-Large`'s dialogs, all under `B:/models/corpora/scale-gate/`.
Both packs convert the same GGUF, and `bcodes.bin` and `f32.bin` are byte-identical between them, so the delta is the
scale encoding and nothing else.

What the interval buys:

1. **The published +0.0034 reproduces exactly** on the same 512 tokens, so the protocol is measuring the thing the table
   above measured, not a new quantity. Its interval is [+0.0021, +0.0049]: the sign is not in doubt.
2. **The cost does not grow with the sequence, and it looked like it did.** Doubling the window on the same text moved
   the average from +0.0034 to +0.0060, and the 512-token interval did not cover the 1024 one, which reads as a cost
   that accumulates with depth. It is not: the same text walked to 2,048 positions reproduces both rows (+0.00344 over
   1-512, +0.00598 over 1-1,024) and then *falls* - +0.00575 by 1,536 and +0.00482 by 2,047 - because the second 512
   tokens happen to carry more of the cost than the third and fourth. The section below profiles it position by
   position; the doubling was where the window stopped, not a drift.
3. **It is not a uniform cost.** On code and on this repository's own measurement pages the interval covers zero; on Go
   dialogs it barely excludes it (+0.0001 to +0.0019); on prose it is six times the Go dialogs'. The spread across corpora
   (sd 0.0028 nats) is wider than any one corpus's own sampling error, which is why the pooled row is quoted with a
   *domain-clustered* interval (t over the four domain means) and it covers zero: "8-bit scales cost 0.003 nats" was a
   corpus-specific statement all along.
4. **The tail is where it shows.** The worst single token in the prose window loses 0.098 nats and the worst in the docs
   window 0.037 - a handful of positions carry most of the mean, as the block argument predicts.

### Where in the window the cost sits

A cost that accumulates with position and a cost that does not are different findings: the first is an error written
into a recurrent state and carried forward, which gets worse the longer the context, and the second is per-token
quantisation noise, which does not. The 2B is the pack to ask - 18 of its 24 layers are gated-DeltaNet, whose state is
read back at every position - and it is the clean pair, because `bcodes.bin` and `f32.bin` are byte-identical between
the two packs.

```
python -u scripts/rope-longctx.py walk --pack ~/.kindi/packs/bonsai-2b-v1 --out a.jsonl --ctx 2048 --prefill 0
python -u scripts/rope-longctx.py walk --pack ~/.kindi/packs/bonsai-2b-v2 --out b.jsonl --ctx 2048 --prefill 0
python scripts/scale-profile.py join --a a.jsonl --b b.jsonl --bins 128 --block 32
```

One 2,048-position walk per pack (under 4 minutes each), 2,047 paired tokens, delta `b - a` in nats a token:

| positions | tokens | delta (8-bit - fp16) | 95% interval |
|---|---:|---:|---|
| 1-512 | 511 | +0.00342 | [+0.00195, +0.00493] |
| 512-1,023 | 511 | +0.00851 | [+0.00642, +0.01076] |
| 1,023-1,534 | 511 | +0.00535 | [+0.00348, +0.00775] |
| 1,534-2,048 | 514 | +0.00199 | [+0.00085, +0.00318] |
| **whole walk** | **2,047** | **+0.00482** | **[+0.00383, +0.00589]** |

**There is no accumulation.** The slope of the delta against position is **-0.0016 nats per 1,000 positions**, interval
[-0.0035, -0.0000]: marginally *falling*, and certainly not rising. What moves is block to block, not with depth - the
per-128 bins run from +0.0006 to +0.0108, an eighteenfold spread, and the money is carried by whichever 128-token
stretch has the worst-fitting block in play. Positions 1-129, the least-informed tokens in the walk, are among the
cheapest (+0.0009). So the 8-bit scales cost about 0.005 nats of per-token noise on prose at any position, and a
longer context does not amplify it - which is the property that matters if the format is used at 32k rather than at
512, and it is the opposite of what the two-point table above suggested.

Three checks that separate the format from the fit:

1. `bcodes.bin`, `f32.bin` and the Hadamard sign files are **byte-identical** between the v1 and v2 packs of the same
   GGUF (sha256), so only the scale bytes moved.
2. A control pack holding the **v2 decoded scales written into the v1 fp16 layout** reproduces the v2 NLL exactly
   (2.5941 then, with the k-means fit in place), so `load_scales` delivers the numbers that were fitted and nothing else
   is different about the two formats.
3. `kindi verify` on the v2 pack: 40/40 tokens matching llama.cpp on all four prompts, 22.0 tokens/s - the prefill path
   in tests/ run against the pack as served.

A tensor with no more distinct scales than levels keeps them bit-for-bit, so the fixture pack and small tensors are
exact; on the 2B every one of the 151 ternary matrices needed the grid. Reading costs one pass of index -> fp16 at load
(the array the reader already held for v1), and the pack is 2.9% smaller on disk.

## 1-bit source files, and the dense Qwen3 layout

Two retired checkpoints - `Bonsai-1.7B` and `Bonsai-8B` - are not `qwen35` at all. They are dense Qwen3 stacks published
as ggml type 41 (`Q1_0`), which is a **binary** block: 128 weights, an fp16 scale in the first two bytes, then 16 bytes of
sign bits, where weight `j` is bit `j % 8` of byte `j / 8` and is `+d` or `-d` (ggml-quants.c, `dequantize_row_q1_0`).
There is no third code and nothing to approximate: decoded as the outer two trits (`0` for `-d`, `2` for `+d`) a `Q1_0`
tensor lands in the pack **exactly**, at the pack's 2 bits a weight rather than `Q1_0`'s 1.125.

```
kindi convert Bonsai-1.7B-Q1_0.gguf --out ~/.kindi/packs/bonsai-1.7b     # 399 MB, 310 tensors
kindi convert Bonsai-8B-Q1_0.gguf   --out ~/.kindi/packs/bonsai-8b       # 2112 MB, 399 tensors
```

What those files need that a `qwen35` pack does not, all of it read off the file rather than configured:

| difference | how the converter and the engine handle it |
|---|---|
| no GDN layers | `qwen3` profile: every layer is full attention, so `full_attention_interval` is 1 and no `ssm_*` tensor is written |
| no Prism rotation | a file with no `prism.hadamard.*` metadata gets `"fold": false, "hadamard_block": 0`; the kernel's `fold()` and `embed()` copy instead of rotating, so the transform is the identity |
| no per-head q gate | the host reads the stride off `attn_q`'s own rows: 2·head_dim means a fused `q|gate` (qwen35), head_dim means a dense projection with nothing to gate |
| `ffn_norm` | renamed to the engine's `post_attention_norm` |
| a vocabulary of 151,669 | head rows are padded to a multiple of 8 (the pair-code layout walks rows in eights) with zero-weight rows, and the engine takes the vocabulary from the manifest, so the padding is never scored |

**The gate is the NLL walk** ([TESTING.md](TESTING.md)), on 567 words of prose of the kind this repository is written
in. Chance here is `ln(151669)` = **11.93 nats**, and an off-by-one in the layout does not read as a slightly worse model,
it reads as a number at or above chance:

| pack | held-out NLL | tokens | what a wrong layout reads |
|---|---:|---:|---|
| `bonsai-1.7b` | **4.5209** | 96 | 15.44 before the q stride was read off the matrix |
| `bonsai-8b` | **3.8246** | 64 | - |

The 15.44 is worth keeping: it was produced by a pack whose trits, scales, embeddings and norms were all correct, and
whose only fault was reading q at a stride of 2·head_dim, past the end of a 2048-row projection. Every structural check
in the suite passed. What caught it was a number larger than the vocabulary's entropy - which is the reason that check is
first in [tools/nll.py](../tools/nll.py) and not a perplexity target.

Both packs then serve captures of their own V vectors, so the K/V work no longer has to reason from a synthetic pack
whose weights are noise:

```
python -m tools.kvq.capture --pack ~/.kindi/packs/bonsai-1.7b --tokens 96 --per-layer --out bonsai-1.7b-vcap.bin
python -m tools.kvq.capture --pack ~/.kindi/packs/bonsai-8b   --tokens 96 --per-layer --out bonsai-8b-vcap.bin
```

| capture | layers | vectors | file | mean relative L2, 2-bit codec |
|---|---:|---:|---:|---:|
| `bonsai-1.7b-vcap.bin` | 28 (all attention) | 21,504 | 11.3 MB | 0.3433 (one scale a head) |
| `bonsai-8b-vcap.bin` | 36 (all attention) | 27,648 | 14.6 MB | 0.3494 (one scale a head) |

## Limits worth stating

- **The retired files ask for YaRN, and the pack records it now.** Both carry `qwen3.rope.scaling` (yarn, factor 4) -
  from 8192 positions in the 1.7B and **16384** in the 8B, which are not the same window. `kindi convert` writes the
  block into `kindi_pack.json` as `rope_scaling` and `kindi.rope` builds the table from it, so the NLL table above and
  any capture past the original window are the file's positional encoding and not a plain one this engine substituted.
  What applying it is worth is measured in
  [CPU-PERF-2026.md](CPU-PERF-2026.md#positional-encoding-what-the-retired-files-ask-for-and-what-it-does-past-8k):
  across a real 9-10k context the file's own factor 4 is indistinguishable from plain RoPE, while factors 1.25 and 2 -
  neither of which these files declare - both beat plain (by 0.020 and 0.031 nats) and both beat 4 (by 0.026 and
  0.037); 1.25 against 2 is a tie. Deep positions want more stretch: at 12.3k factor 4 is worth a quarter of a nat. A
  pack converted before the key existed (the 8B was)
  keeps its plain table until `kindi.convert.rope_scaling` re-derives the block from its own GGUF - metadata only, no
  weight is touched. A factor the file does not declare is one command rather than an edit: `kindi convert ...
  --rope-factor 2` writes the block (and `--rope-factor 1` builds the plain table over a file that asks for YaRN),
  which is how the 1.25x and 2x arms above were given to the kernel.

- **No reference forward pass.** The checkpoint is 24 pure-GDN layers with per-head scalar B/C (`in_proj` = 2·1024 +
  3·16 rows, no attention tensors at all); its trainer is not on this machine, so the pack is verified against the
  quantiser and not against that model's logits. End-to-end drift therefore means "weights and per-matrix outputs",
  not "tokens and perplexity".
- **The probes are synthetic.** The weight-space numbers are exact for the checkpoint; the `output` column uses
  Gaussian probes of unit RMS because the real activation distribution of that checkpoint is not available here.
- `kindi drift` takes about 90 s and 1 GB of RAM per 255M checkpoint: it holds the full state dict as float32.
- The re-ternarisation sweep is arithmetic on the pack, not a forward pass: it says which codes survive a drifted
  re-quantisation and says nothing about what the surviving codes cost in tokens. Its closed forms are exact by
  construction, so the identities are a check on the implementation rather than an empirical discovery; what is
  empirical is the histogram and its flatness over depth.
