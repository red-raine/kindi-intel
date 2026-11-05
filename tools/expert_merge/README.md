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
python -m tools.expert_merge.prune --routes routes.npz --keep 96          # which experts to drop, per layer
python -m tools.expert_merge.prune --routes routes.npz --pack MODEL.gguf --frac 0.75 --gate --corpus A,B
python -m tools.expert_merge.prune --selftest                             # the solve, the mask and the verdict
python -m tools.expert_merge.prunegate --json wide.json keep96.json keep64.json   # one table, one verdict an arm
python -m tools.expert_merge.prunegate --selftest                         # the four verdicts and the merge's checks
python -m tools.expert_merge.docbatch --pack MODEL.gguf --layers 8 --docs 3  # serial vs batched corpus forward
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

## Merging the groups into a pack

`mergepack` is the downstream half of the permutation test: it writes a real GGUF in which each co-activated group is
one expert. Non-expert tensors are copied byte for byte, the three expert stacks become one mass-weighted mean a group,
the router becomes one pooled row a group, and `expert_count` is patched to match - so the result is a pack the same
reader and the same forward can run, and the before/after comparison is two models rather than two settings.

```
python -m tools.expert_merge.mergepack --pack SRC.gguf --routes routes.npz --groups 64 --out MERGED.gguf
python -m tools.expert_merge.mergepack --pack SRC.gguf --routes routes.npz --groups 128 --expert-type Q4_0 \
    --out CONTROL.gguf                                    # the unmerged control, same path, same quantisation
python -m tools.expert_merge.mergepack --pack SRC.gguf --routes routes.npz --groups 64 --expert-type Q4_0 \
    --out MERGED.gguf --gate --baseline CONTROL.gguf --corpus A,B --docs 2 --tokens 48
python -m tools.expert_merge.mergepack --retopk MERGED.gguf --used 4 --out MERGED-k4.gguf   # same weights, k=4
The same weights at another routing breadth: `--retopk` (or `--used N` when a pack is being built) rewrites the
`expert_used_count` header field and copies everything else, so a paired walk of the copy against its source prices
routing breadth with the merge held fixed. `--compare BASE --against A.gguf,B.gguf` gates packs that already exist,
loading and scoring the baseline once for every arm:

```
python -m tools.expert_merge.mergepack --compare CONTROL.gguf --against MERGED.gguf,MERGED-k4.gguf \
    --corpus A,B --docs 2 --tokens 48 --json arms.json
python -m tools.expert_merge.mergepack --selftest         # grouping, fusion, a written pack read back, and the k rewrite
```

Run end to end on the 48-layer pack: 128 experts, top-8, `routes.npz` from 384 tokens of this repository. `--groups
64` on 128 experts with a size cap of `ceil(128/64)` is exactly 64 pairs, and over the log 17.7% of a token's routing
weight lands in a group another of its slots also picked - the share of its expert arithmetic one averaged matrix does
instead of two. There is no permutation alignment in this path: `merge.py`'s own gate rejects every routed pair (median
aligned cosine 0.102, mean 0.105, against 0.85), so the fusion is the specification's uncertainty-weighted mean,
applied to the router rows too.

Because this build of `gguf` decodes K-quants but cannot encode them, both arms are written at the same type through
the same code path, and the unmerged control is the only fair comparison:

| pack | experts | k | type | size | NLL, 94 held-out tokens |
|---|---:|---:|---|---:|---:|
| source | 128 | 8 | Q2_K / Q3_K | 11.26 GB | 4.9563 (reference) |
| control | 128 | 8 | Q4_0 | 17.03 GB | 4.9053 |
| merged | 64 pairs | 8 | Q4_0 | 8.87 GB | 6.2641 |
| merged, matched fraction | 64 pairs | 4 | Q4_0 | 8.87 GB | 6.3071 |

**+1.3588 nats/token** against the control (95% interval +0.8984..+1.8029), i.e. perplexity about 3.9x worse at half
the expert bytes *of that control* (8.16 GB against 16.32 GB; against the source's own 11.26 GB the merged file is
8.87 GB, 21% smaller, and the source is Q2_K, so only the control is like-for-like); both documents move the same way
(+1.44 and +1.28). Read at the control's own fraction (4 of 64, the same 6.25% as 8 of 128) it is **+1.4018**
(+0.9031..+1.8329): the shipped arm's extra relative capacity was flattering the merge by about 0.043 nats, so the
merge cost is +1.40 and not +1.36 - and the matched-fraction arm is also half the expert arithmetic a token. The router
grouping is real - the same pairs co-fire 10.8x more often than independence (median
10.75, mean 12.75 over the 144 pairs) - and averaging them is still ruinous, which is what
[docs/EXPERT-MERGE.md](../../docs/EXPERT-MERGE.md) measures from the weight side: a co-activated pair is not a
relabelled function, so there is nothing for a merge to preserve. The 0.85 gate is load-bearing, not conservative.

## Pruning: which experts a layer can keep

`prune` is the other half of `cluster_budget.py`: that one asks whether a paging schedule can afford the experts a
layer routes to, this one asks which experts a layer can simply not have. The value of an expert is its **routing
mass** from the log - the sum of the renormalised top-k weight it was applied with - and the budget is exact and per
layer, because layer L's experts are its own tensors. The choice is solved by `kindi.rco.plan`, the same knapsack the
converter uses for `--budget`, with two formats a expert: `drop` at 0 bytes and its mass as the error, `keep` at its
bytes and no error.

On the 48-layer Qwen3-Coder-30B-A3B pack (1,708,032 bytes an expert, 208.5 MB a layer):

| keep, of 128 a layer | expert weights | of 9.77 GB | routing mass given up |
|---:|---:|---:|---:|
| 112 | 8.11 GB | 82.9% | 0.1% |
| 96 | 7.31 GB | 74.8% | 0.6% |
| 80 | 6.11 GB | 62.5% | 2.1% |
| 64 | 4.89 GB | 50.0% | 5.6% |

`--frac F` prices each expert from the pack instead of counting it, and agrees with `--keep` where they coincide
(0.75 x 218,628,096 bytes is exactly 96 experts). The plan is applied as a mask on the router, so the two arms of an
NLL gate can share one loaded pack. On 1,785 held-out tokens (seven documents the log never saw) NLL went 3.6742 ->
3.6787 at 112 experts a layer, interval -0.0082..+0.0174: **lossless within 0.02**, certified, for 82.9% of the expert
bytes. At 96 it went to 3.6906, interval -0.0023..+0.0356 - inside the tolerance on the estimate and not established.
The corpus is that wide because `forward_docs` batches it: the dequantiser is paid once a layer for the corpus instead
of once a layer a document, which is 3.6x on three documents and grows with the corpus. A document at a time it is
17 minutes whatever its token count, which is why the first gate was 94 tokens and read the wrong way.
[docs/EXPERT-PRUNE.md](../../docs/EXPERT-PRUNE.md) has the numbers, the cost model and what they do not establish.

## What the real pack says

Qwen3-Coder-30B-A3B-Instruct Q2_K, layers 0-12, 384 tokens of this repository's text, top-8 of 128 experts; the rows
below are medians over that run's pairs:

| | routed pairs (39) | random pairs (39) |
|---|---:|---:|
| routing-profile cosine | 0.757 | 0.000 |
| best-bijection residual, against identity | 1.368 vs 1.408 | 1.374 vs 1.404 |
| verdict | 39 merely-similar | 39 merely-similar |
| the shipped merge's aligned cosine (gate 0.85) | **0.086**, 0 of 39 pass | 0.081 |
| merged output off the anchor | 0.928 | 0.944 |

The router groups its pairs hard — they co-fire 13.9x more often than independence (median 13.87, mean 16.44 over
these 39 pairs) — and the pairs are still not relabellings: no hidden unit has both its gate and up row matched above
|cos| 0.99, and a full orthogonal relabelling leaves 0.716 of the difference. The merge's own gate rejects every one of
them, which is the only reason the stage is safe.

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
