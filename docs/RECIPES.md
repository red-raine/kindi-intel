# Recipes, GSQ and RCO: what a model is, and what to build from it

This page covers the conversion side of Kindi: naming a model from its header, choosing a conversion plan for a job,
and the two things this repository now does that it did not before - a fitted ternary scale (`gsq`) and an exact
size-budget choice across tensors (`rco`). It also records what two published releases do that this engine does not,
so the gap is written down rather than implied.

```
kindi recipes                                # the plans, and what each one changes
kindi recipes --for MODEL.gguf --role rag    # detect a file, print its plan and what the file cannot support
kindi fetch hf:OWNER/REPO/FILE.gguf          # any Hugging Face file, verified against its LFS sha256
kindi fetch kaggle:OWNER/DATASET[/MEMBER]    # any Kaggle dataset, one member extracted or all of them
kindi qat-pack MODEL.pt --scheme gsq         # fit codes and scale together
kindi qat-pack MODEL.pt --budget 900000000   # choose which tensors stay f32 under a byte budget
kindi convert MODEL.gguf --recipe coder      # apply a plan's conversion steps to a GGUF pack
```

## Detection

`kindi.recipes.detect` reads a GGUF header - no tensor data - and reports the architecture, depth, width, experts and
active experts, context, the rotary block, the vocabulary, and the tensor-type histogram. It is metadata only, so it
costs a header parse on a 350 GB file, and it is explicit about what the file does not say: a missing expert count is
reported as a dense model (the GGUF default), a family with no conversion path is named and refused rather than forced
through another family's reader. A pack directory is detected from its manifest the same way.

Measured on this machine, as a check that the reader agrees with real files:

| file | detected | notes |
|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | `qwen35`, 64 layers, width 5120, no experts, context 262144, 248320 tokens | `PTQ1_0 x402`, `F32 x353`, `BF16 x96` |
| `Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf` | `qwen3moe`, 48 layers, width 2048, **128 experts, 8 active**, context 262144 | mixed `Q2_K/Q3_K/Q4_K/Q6_K`: not a ternary source, so the pack path refuses it with that reason |

A whole folder at once, headers only, which is how the table above was checked against every model on this machine:

```
kindi recipes --scan B:/models
  14 model file(s) under B:/models
  Mixtral-8x22B-v0.1.Q8_0-00001-of-00005.gguf  llama 56L w6144 8x2 ctx65536       agent   2 warning(s): llama has no conversion path ...
  Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf       qwen3moe 48L w2048 128x8 ctx262144 coder   1 warning(s): the file is not ternary ...
  Ternary-Bonsai-2-27B-PTQ1_0.gguf             qwen35 64L w5120 dense ctx262144    chat    ready
  model_gsq.gguf                               cannot read the header: truncated or corrupt: a tensor name
                                               (2670395422995916235 bytes) needs ... but the file has 79681 left
```

Only the first part of a split file is listed, partial downloads and dot directories are skipped, and a file whose
header cannot be read is reported with the reason. Running it found one: `qwen-0.5b-pilot/quantized/model_gsq.gguf`
is 79 KB and its header claims a 2.6-exabyte tensor name. The reader used to accept the length and ask for the memory,
which is why it exited with a bare `MemoryError` naming nothing; `kindi.gguf` now checks every stated length - a
string, an array's count, a rank's dimensions, the tensor and metadata counts - against the bytes the file actually
has, and says which field at which offset is impossible. `kindi.gguf.selftest` keeps that fixed with a tiny valid
file, a file cut three bytes short, and an array claiming 2^40 elements.

A header is metadata, not integrity: a file whose header parses can still carry a truncated tensor. The check that
covers that is loading the pack (`kindi serve` reads every tensor it is asked for) or the sha256 `kindi fetch`
verifies at download.

## Recipes

A recipe is the set of conversion decisions this engine can actually execute, plus the gate the result must pass. It
is not a wish list: every field is read by `kindi convert` or `kindi qat-pack`, or by the benchmark gate.

| recipe | what it is for | fitter | rope window | paircodes | gate (NLL delta / top-1) |
|---|---|---|---|---|---|
| `coder` | coding agent: long context, tool calls, code-heavy prompts | `gsq` | 32768 | yes | 0.020 / 0.90 |
| `agent` | general tool-using agent, many short turns | `gsq` | 32768 | yes | 0.020 / 0.90 |
| `rag` | RAG reader: long pasted context, short answers | `lsq` | 32768 | yes | 0.030 / 0.88 |
| `search` | short queries, latency first | `lsq` | 8192 | no | 0.030 / 0.88 |
| `science` | maths and multi-step reasoning | `gsq` | 16384 | yes | 0.015 / 0.92 |
| `chat` | the default, no job-specific trade | `lsq` | 8192 | no | 0.030 / 0.88 |

The thresholds are not invented: a 0.01-nat difference is inside the noise of a 79-token window
([PACK-ZOO](PACK-ZOO.md)), and the 2-bit V comparison resolved 0.007 nats over 48 prompts
([CPU-PERF-2026](CPU-PERF-2026.md)). A gate tighter than the measurement's own resolution fails on sampling, not on
the quantiser.

When a file cannot support a recipe, the plan says so instead of building something else: a 32k window on an 8k file,
`paircodes` on a model with no experts, `gsq` on a ternary source that has no float weights to fit, or a family with no
conversion path. `kindi convert --recipe` prints those warnings and applies only the steps that are possible.

## GSQ: fit the codes and the scale together

`kindi.gsq` treats the ternary code of each weight as a discrete choice relaxed to a Gumbel-Softmax, optimises it and
the group's scale by gradient descent with a straight-through forward pass, then alternates to a fixed point - codes at
this scale, then the least-squares scale for those codes, `s = <w,c>/<c,c>` - and keeps the better of the trajectory's
start and its end. The packed invariant is unchanged: codes in {-1,0,+1}, one fp16 scale a 128 weights, and the scale
is rounded to fp16 *before* the final code assignment, so `dequantise()` is the arithmetic the kernel runs.

Measured at 2 bits a weight, relL2 against the weights (128x2048 matrices):

| weights | `absmean` | `lsq` | `gsq` |
|---|---|---|---|
| Student-t, 3 dof (heavy tails) | 0.719483 | 0.640094 | **0.566846** |
| Gaussian | 0.511718 | 0.433083 | **0.433038** |

That is 11.4% below `lsq` where it matters - real checkpoints are heavy-tailed - and level on Gaussian weights. Two
honest limits: the objective is the weights' squared error, not the layer output error the GSQ paper minimises (which
needs calibration activations), and on this objective the **alternation** is what wins: `iters=0` and `iters=60`
return the same bytes. The Gumbel phase is kept because it is the mechanism a calibration-weighted objective needs;
with an identity metric it has nothing left to improve, and this page says so rather than quoting the training as a
gain.

## RCO: an exact byte budget across tensors

`kindi.rco` answers the other question a conversion asks: with a size the machine can hold, which tensors deserve to
stay at f32? Given one measured error per tensor per format, the choice of one format a tensor with a total within the
budget is a multiple-choice knapsack, and it is solved exactly on a 1 KiB byte grid by dynamic programming. A problem
too large for the table falls back to efficiency-ordered greedy plus a 1-swap pass, and reports the fractional
knapsack bound and the gap - a heuristic result says it is one.

```
kindi qat-pack MODEL.pt --scheme gsq --budget 900000000
  rco: 2 f32, 241 ternary in dp, 858.3 MB of 900.0 MB (error 1.4e-05)
```

`kindi quant.pack(..., budget=N)` measures every candidate tensor's ternary error first, plans, then writes the plan.
Two limits belong here: the errors are measured on the weights, not on the model (so the plan is exact for the
objective it is given, not for perplexity), and a budget below the all-ternary floor is refused rather than silently
exceeded.

## Importing a model from either place

**Hugging Face** (`hf:OWNER/REPO/PATH`) is verified: the repo's tree API carries each file's LFS oid, which *is* its
sha256, so a downloaded file is checked against what the publisher stored, and a file with no LFS entry is refused
unless `--allow-unverified`. `HF_TOKEN` is sent when set, for gated repos.

**Kaggle** (`kaggle:OWNER/DATASET[/MEMBER]`) works differently and the difference is stated rather than smoothed over:
Kaggle has no per-file endpoint and publishes no sha256, so the unit is the dataset archive. It is downloaded with the
account's credentials (`KAGGLE_USERNAME`/`KAGGLE_KEY`, or `~/.kaggle/kaggle.json`), the named member is extracted, and
the archive is removed unless `--keep-zip`. Nothing is checked against a publisher hash; the checks are the file's
length and whether `kindi recipes` can read its header. Two limits: the download is not resumable (Kaggle's zip
endpoint ignores a range), and a member name that occurs twice in the archive is refused rather than guessed.

Exercised here against a live account: `kaggle:igalbronshtein/hugging-face-gguf` downloaded, extracted one member and
removed the archive; `kaggle:me/gguf` (no such dataset) is a `kindi fetch: ... returned HTTP 403 - this account may not
download it: accept the dataset's terms on its Kaggle page`. A bad spec, a member that is not in the archive, and an
archive with two members of the same name are all covered by `kindi.catalog.selftest`, which builds its zips locally
and needs no credentials.

What the search turned up for the models this project is aiming at: `rahim3/glm53-flash-iq3xxs-1` and `-2` hold
GLM-5.3-Flash as a two-part UD-IQ3_XXS GGUF (49 GB + 71 GB) on Kaggle. That is a 3-bit source, so the conversion path
is `kindi qat-pack` on float weights, not the ternary pack reader - the reader needs the family, and `kindi.recipes`
currently names `glm4`/`glm5` as having no path rather than guessing.

## The two releases this work started from

Both cards are built on [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF),
a 176.9B-parameter, 512-expert MoE with half its experts removed and the rest at 3.5 bits.

- **GSQ** ([arXiv 2604.18556](https://arxiv.org/abs/2604.18556)) is the learnt grid assignment and group scale this
  module implements in its weight-space form.
- **RCO** ([arXiv 2605.00649](https://arxiv.org/abs/2605.00649)) is budget-constrained optimisation on a Riemannian
  manifold: it chooses *which experts to remove* under one exact budget per layer, and in the base releases assigns
  per-tensor quantization types. The format assignment is what `kindi.rco` does exactly; the expert selection needs a
  forward pass per candidate and is not built here.

[`Schiltmans/Qwen3.8-Flash-Next-Coder-cut240`](https://huggingface.co/Schiltmans/Qwen3.8-Flash-Next-Coder-cut240) is a
runner around that model: an MLX path with TensorFold fused kernels, an n-gram shard served from disk, an MTP draft
head with its own experts cut to 256, sparse prefill attention, paged KV shared by prefix, and optional SSD streaming
of the routed experts. Its `cut_experts.json` / `kept_experts.json` are the record of the pruning.

[`alesha-pro/...-Strata-GGUF`](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Coder-Strata-GGUF)
adds Strata's two features: expert paging under explicit VRAM/RAM budgets (`--resident-experts`, `--mmap-experts`,
`--vram-reserve-mib`) and a run-time control-vector projection, `h = h - (h . r) r`, applied after each steered layer.

What this engine has of that: expert paging under a memory budget (CPU engine, resident vs paged arms measured in
[CPU-PERF-2026](CPU-PERF-2026.md)), and the pack-side pieces above. What it does not have: MTP drafting, sparse
attention with an indexer, n-gram heads, KV paging, SSD expert streaming, and control-vector projection. Those are
runtime features of other engines, not conversion decisions, and they are listed here so a reader does not assume
this release covers them.

## Not built yet, in the order it would pay

| piece | what it needs | why it is not here |
|---|---|---|
| MoE expert pruning under a budget | routing statistics over a calibration set, then a per-layer exact choice | `tools.expert_merge/routes.py` already records routes; the selection is the next step |
| Distillation into a smaller model | a training loop and a teacher runtime | the specification's arithmetic is in `tools/exodus/`; the loop is not written |
| MiniMax M3 / GLM5.3-Flash conversion | a reader per architecture, and a ternary/1-bit source | `kindi.recipes` names them as having no path yet rather than mis-reading them |
| Strata-style control vectors | a projection step in the engine's attention block | a runtime feature, and measured as a capability change, not a compression |
