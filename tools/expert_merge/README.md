# expert_merge: does the permutation + Fisher merge do what it claims?

The compiler specification merges expert families by permutation-aligning each member's SwiGLU MLP to an anchor expert
(cosine similarity of `gate_proj` weight rows, via `linear_sum_assignment`) and then averaging the tensors weighted by a
diagonal Fisher. This package answers two questions about that stage: with a synthetic family whose answer is known,
does the arithmetic work; and with a real pack, are the experts it would merge actually one function relabelled.

The measurement and its numbers are in [docs/EXPERT-MERGE.md](../../docs/EXPERT-MERGE.md). This file is the tool.

```
python -m tools.expert_merge.merge        # the merge against synthetic families + the shape check
python -m tools.expert_merge.permtest     # planted permutations, noisy copies, unrelated experts
python -m tools.expert_merge.fetch --verify MODEL.gguf                    # is this pack complete, and where does it break
python -m tools.expert_merge.routes --pack MODEL.gguf --out routes.npz    # a real router log
python -m tools.expert_merge.real --routes routes.npz --out merge.json --jobs 6   # test the pairs it groups
python -m tools.expert_merge.real --summary merge.json                    # the numbers, read back
```

`merge`, `permtest` and `fetch` are all in `python -m tools.selftest`, so none of them needs a pack or a GPU.

## The pack has to be checked first

`Moepack.require_complete` refuses a pack whose payload does not decode, and both `routes` and `real` call it: a
forward over 13 of 48 layers produces a router log that reads exactly like a result, and that is how the number in
this note was first measured. `--verify` prints the same diagnosis in one line:

```
  Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf    NOT A COMPLETE PACK
    fails at blk.13.ffn_up_exps.weight (byte 3504728316 of 11318112204); 13 of 48 layers decode
```

A damaged file is repaired from a mirror rather than re-fetched whole — the readable prefix is kept and only the
tail is fetched, each chunk written at a `Content-Range`-validated offset:

```
python -m tools.expert_merge.fetch --repair MODEL.gguf --url URL [--jobs 6]
python -m tools.expert_merge.fetch --url URL --file MODEL.gguf --sha256 HEX --jobs 6
```

A chunk is written only after its 206 reply names the byte range that was asked for, and the file is installed only
after its length, its sha256 and its GGUF payload all check out. Six connections are worth about three of one, which
for an 11 GB pack is the difference between twenty minutes and two hours.

## What the real pack says

Qwen3-Coder-30B-A3B-Instruct Q2_K, layers 0-12, 384 tokens of this repository's text, top-8 of 128 experts:

| | routed pairs (39) | random pairs (39) |
|---|---:|---:|
| routing-profile cosine | 0.757 | 0.000 |
| best-bijection residual, against identity | 1.368 vs 1.408 | 1.374 vs 1.404 |
| verdict | 39 merely-similar | 39 merely-similar |
| the shipped merge's aligned cosine (gate 0.85) | **0.081**, 0 of 39 pass | 0.074 |
| merged output off the anchor | 0.928 | 0.944 |

The router groups its pairs hard — they co-fire 12.5x more often than independence — and the pairs are still not
relabellings: no hidden unit has both its gate and up row matched above |cos| 0.99, and a full orthogonal relabelling
leaves 0.716 of the difference. The merge's own gate rejects every one of them, which is the only reason the stage is
safe.

## What the synthetic suite asserts

| case | mean aligned cosine | result |
|---|---:|---|
| three permuted copies of one expert | 1.000000 | fused equals the anchor to 8.9e-16 - the permutation is undone exactly |
| three unrelated experts | 0.301645 | fused differs from the anchor by 4.2, and **nothing raises** |
| the specification's own function, permuted case | 1.000000 | same 8.9e-16: the permutation logic itself is correct |

`permtest` adds the strict form: a planted signed permutation is recovered exactly (residual 0, and the planted mapping
is the one found), an exact copy is reported as `identical` rather than `permutation`, a noisy copy is neither, and
merging a permutation is exact after alignment while merging a noisy copy is lossy with and without it.

## The Fisher defect

**Stage 0's Fisher statistic cannot weight Stage 3's tensors.** Stage 0 accumulates `diag(X^T X)`, an input-channel
second moment of shape `[d_model]`; Stage 3 computes `fisher * weight` on a `[d_ffn, d_model]` tensor. Element-wise
Fisher merging needs a per-weight statistic of the weight's shape - for `gate_proj` that is
`diag(diag(G) ⊗ X^T X)` with `G` the output-side curvature, which is a different accumulator from the one specified.
`align_and_fuse` accepts a scalar, a per-expert scalar, or a same-shape tensor, and raises on anything else with that
message; `align_and_fuse_expert_family` (the specification's version) silently multiplies by whatever it is given,
which numpy will broadcast or refuse depending on the shapes involved.

The related defect is that the specification discards `row_ind` from `linear_sum_assignment` and never checks the
assignment it applies. `align_and_fuse` requires a square cost matrix, verifies the result is a permutation, and
returns the mean aligned cosine so a caller can gate on it - which is what the pipeline's own 0.85 check does.

## What this does not establish

The merge is checkable without a donor checkpoint only in the cases above. Whether a *real* family - two checkpoints
fine-tuned from one base, a teacher and a student mid-distillation - is closer to the permuted case than to the
unrelated one is not something this package can answer from one pack, and [docs/EXPERT-MERGE.md](../../docs/EXPERT-MERGE.md)
records that limit beside the numbers.
