# Converting a float checkpoint to a ternary pack

`kindi convert` reads the GGUFs Kindi ships with: a `qwen35` model whose weights are already ternary (`PTQ1_0`, one
fp16 scale per 128-weight group). A quantisation-aware training run produces the opposite: **float** weights, trained
with that quantiser in the loop. `kindi qat-pack` runs the quantiser forwards so one of those checkpoints becomes a
pack, and `kindi drift` measures what it costs.

```
kindi qat-pack qat_final.pt --out ~/.kindi/qat255m        # least-squares scale; --scheme absmean for the GGUF one
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

## Limits worth stating

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
