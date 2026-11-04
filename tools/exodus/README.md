# exodus: the executable part of the distillation specification

The specification (Project Exodus-Ternera) describes a five-phase compiler that turns frontier MoEs into looped ternary
students: donor checkpoints, behavioural traces over 100k tokens, head slicing, Fisher-merged expert families, looped
distillation, and a 4 KB-aligned binary for the AVX2 runtime. None of that runs here - it needs Maple/Bonsai,
Qwen-2.5-Coder-32B and Ornith-35B-A3B, GPUs and hours of training, none of which are in this repository.

What it also contains is three Python modules small enough to execute on a 64x256 tensor and four attention heads. Those
are checked here, and checking them changed what the document says.

```bash
python -m tools.exodus          # 36 checks in three groups
python -m tools.exodus.rotate   # rotation and ternary snapping
python -m tools.exodus.router   # top-k routing, halting, distillation loss
python -m tools.exodus.heads    # attention-head slicing
```

## What was claimed, and what the arithmetic says

| the document's claim | measured |
|---|---|
| rotating by an orthogonal `A` preserves the function | true: `(W A)(A^T x) = W x` to **8e-15**. But the transform the code returns is `A`, and the identity needs `A^T`: with `A` where `A^T` belongs the output differs by **1.8 relative** |
| the Stiefel optimiser reduces quantisation error | false as written. `round()` has no derivative, so the only gradient path is through the per-row `scale`. Stepping along that truncated gradient **raises** the loss it prints (3.40 -> 5.63) and drops SQNR **9.45 -> 7.26 dB** |
| Hadamard incoherence removes outlier damage | true, and it is the largest single effect here: naive ternary snapping on a matrix with two 60-sigma channels gives **0.25 dB**, after rotation **9.45 dB** |
| Riemannian refinement improves on plain rotation | true but modest: minimising magnitude concentration on the manifold gives **10.48 dB** (+1.03), leaving `R` orthogonal to 1.3e-15 |
| the acceptance gate is SQNR > 22 dB | not reachable by this method on this matrix. Best arm **10.48 dB**. Either the matrix, the format (2-bit ternary, per-channel scale) or the threshold has to change before that gate can mean anything |
| the student block routes as the engine does | no. `kindi.moe.softmax_topk` renormalises the top-k weights to sum to 1; the supplied block keeps them at **0.699** on average. With the specification's own damping that is a **0.285** residual gain per loop, so a 3-iteration loop delivers about a quarter of one pass |
| a learned halt gate saves compute | not with a batch-mean threshold, yes with a per-sequence mask. The mean is **0.762** on three certain rows of four and halts nothing; at batch 64 it fires at **0.911** and stops the six rows still working. Replaced by a sticky per-row mask (`halt_mask`), which on the looped block removes **46.9%** of row-loops on a mixed batch with the output unchanged, where the batch mean's **12.5%** cut 7 rows and wasted 183 row-loops on 49 finished ones |
| the distillation loss is comparable across runs | not as written: `batchmean` on `[B, S, C]` divides by the batch, so one token repeated 64 times gives **64x** the loss. The `0.4 * router + 0.6 * token` weighting is fine only while both terms see the same length |
| head importance `E\|dL/dA_h * A_h\|` | the supplied hook cannot run at all: `o_proj`'s gradient is `[B, S, d_model]` and `dim=(0, 1, 3)` raises. It is also `register_backward_hook`, deprecated. And the code accumulates `\|\|dL/dA_h\|\|` - dropping the activation, which on a head with 30x the activation reverses the ranking |
| slicing heads is a row operation | only for `q/k/v`. `o_proj` must be sliced on **columns** for the same heads or the width does not match. A correct slice reproduces the kept heads' contribution to **0.0**; a fused packed QKV block needs offset `3 * h * d_head`, not `h * d_head` |

The two things this proves are worth keeping: head slicing is exact arithmetic and can be tested without any model, and
the rotation stage's quality is measurable in a few milliseconds. Everything about whether a distilled student is *good*
still needs the checkpoints.
