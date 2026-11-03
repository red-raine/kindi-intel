# expert_merge: does the permutation + Fisher merge do what it claims?

The compiler specification merges expert families by permutation-aligning each member's SwiGLU MLP to an anchor expert
(cosine similarity of `gate_proj` weight rows, via `linear_sum_assignment`) and then averaging the tensors weighted by a
diagonal Fisher. This tool checks that with synthetic families it can construct exactly, so the answer does not depend
on having a donor model.

```
python -m tools.expert_merge.merge
```

| case | mean aligned cosine | result |
|---|---:|---|
| three permuted copies of one expert | 1.000000 | fused equals the anchor to 8.9e-16 - the permutation is undone exactly |
| three unrelated experts | 0.301645 | fused differs from the anchor by 4.2, and **nothing raises** |
| the specification's own function, permuted case | 1.000000 | same 8.9e-16: the permutation logic itself is correct |

## What this establishes

**The alignment arithmetic is right.** `linear_sum_assignment` on `-(unit_rows(anchor) @ unit_rows(target).T)` returns
the column permutation that maps target rows onto anchor rows, and applying it to `gate`/`up` rows and `down` columns
undoes an exact permutation to float rounding. Both the specification's transcription and the corrected function agree.

**The alignment is not a validation.** On a family that cannot be aligned, the merge returns a confident average - the
same number (4.204) from both versions - with no signal. The only thing that separates the two cases is the mean aligned
cosine, 1.00 against 0.30, which is why the specification's own "assert cosine > 0.85 before quantising" gate is the
load-bearing part of the pipeline rather than a formality. On this evidence the gate has a wide margin.

**Stage 0's Fisher statistic cannot weight Stage 3's tensors.** Stage 0 accumulates `diag(X^T X)`, an input-channel
second moment of shape `[d_model]`; Stage 3 computes `fisher * weight` on a `[d_ffn, d_model]` tensor. Element-wise
Fisher merging needs a per-weight statistic of the weight's shape - for `gate_proj` that is
`diag(diag(G) ⊗ X^T X)` with `G` the output-side curvature, which is a different accumulator from the one specified.
`align_and_fuse` accepts a scalar, a per-expert scalar, or a same-shape tensor, and raises on anything else with that
message; `align_and_fuse_expert_family` (the specification's version) silently multiplies by whatever it is given,
which numpy will broadcast or refuse depending on the shapes involved.

## What this does not establish

The merge is the only stage of the seven that is checkable without a donor checkpoint. Stages 0-2 need real routing
statistics, Stages 4-5 need training, and Stage 6's "assert cosine > 0.85" needs a model that has been through the
first five. Nothing here says the merged family preserves the teacher's behaviour: case A is a family that is *by
construction* the same function, and case C shows the merge has no opinion about whether it is one. Whether a real
cluster of experts is closer to A or to C is the question the whole pipeline rests on, and it is unmeasured.
