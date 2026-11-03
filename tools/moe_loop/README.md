# moe_loop: the note's block, run and measured

A ternary MoE block with recurrent depth, as sketched in the design note: `BitLinear158` (ternary weights, int8
activations, straight-through estimators), a two-head router over halves of the hidden state, and a damped residual
loop. Each piece exists in two versions - the note's and a corrected one - so the difference is a number instead of an
argument.

    python -m tools.moe_loop.test_moe_loop     # 6 checks, CPU only
    python -m tools.moe_loop.probe             # the same quantities printed, plus the drift curve

Nothing here trains anything, and nothing imports it. The engine's serving-side MoE is `kindi/moe.py` (one router,
renormalised top-k, Hadamard fold, verified against NumPy at 1e-7 by `kindi moe-check`); this is the training-side
counterpart the note describes.

## What the claims measure at

| claim in the note | measured here |
|---|---|
| ternary weights hold `{-1, 0, +1}` and still train | 3 distinct values, 24.9% rounded to zero, gradient reaches the weight (norm 92 through the STE) |
| looping needs damping or it "blows up" | over 8 loops: **1.016x** growth at `alpha=0.25`, **1.05x** at 0.5, **1.16x** at 1.0 |
| 1.58-bit inside a recurrent loop accumulates error | 3.5e-2 relative at 1 loop, 2.6e-1 at 8, i.e. **~linear in the loop count** (1.87-1.99x per doubling) |
| multi-head routing sends sub-vectors to different experts | **not in the note's router**: every expert receives the full state (width 32). The split router gives width 16 |
| sparse routing is worth doing | mask dispatch: **24576 expert rows**, `index_add`: **3072** - exactly 8x for 8 experts |

Two of these are worth reading closely.

**RMSNorm does not bound the loop.** The note's recipe is "strict pre-layer/post-layer RMSNorm and explicit residual
damping". Measured, the normalisation only rescales the block's *input* each pass; the residual keeps accumulating, so
the state grows geometrically with a rate set by the block's gain. Damping lowers that rate (1.16x -> 1.05x -> 1.016x
over eight loops) but does not remove it, and `alpha` is a hyper-parameter, not a guarantee. At init the growth is mild
because the gain is small; a trained block with a gain above one per loop is the case that diverges.

**The error accumulation is linear, not explosive** - which is both better and worse than the note says: better
because it does not compound multiplicatively within eight loops, worse because it never decays either, and 26%
relative error at eight loops is already beyond what a float teacher would tolerate.

## What the note's sketch gets wrong

1. **Averaging logits is not sub-vector routing.** `(head1(x1) + head2(x2)) * 0.5` produces *one* distribution over
   *one* expert set; the chosen experts then run on the whole hidden state. Measured: expert input width 32 for the
   note's router against 16 for a router that actually gives each head its own expert group. The "multi-headed routing"
   in the note is a two-view ensemble, which is a legitimate thing to build, just not the thing it says.
2. **The same averaging flattens routing.** Halving the summed logits halves the logit *scale*, so
   `softmax((l1+l2)/2)` is flatter than `softmax(l1+l2)`: mean top-2 mass **0.373 against 0.500**. Interesting nuance:
   it is within 0.010 of the true mixture `(p1+p2)/2` on this sample, so the flattening is the real effect and the
   "wrong mixture" worry is not - measured rather than assumed.
3. **The dispatch multiplies work by the expert count.** `for e in experts: if mask.any(): out += mask * (w *
   expert(normed))` runs each selected expert on the *whole batch* and then throws most of it away: 8x the arithmetic
   for 8 experts. It also evaluates the Python-level loop over every expert regardless.
4. **`if mask.any()` synchronises.** On a CUDA device that is a host round-trip per expert per slot per loop, which is
   the same class of cost that made `kindi`'s expert paging callback-bound (69 `moe_fetch` calls a token, ~0.3 ms each
   - see [docs/CPU-PERF-2026.md](../../docs/CPU-PERF-2026.md#L405)). Not measured here: this machine has no GPU.
5. **No renormalisation after top-k.** The engine's reference router divides by the selected sum
   (`softmax_topk` in [kindi/moe.py](../../kindi/moe.py)), the note's does not, so the block's output shrinks as `k` grows.
6. **Odd `d_model` breaks the router.** `x.chunk(2, dim=-1)` returns `d//2` and `d//2 + 1` for odd `d`, while both
   heads were built for `d//2`.

## What could not be tested here

- **Slicing a donor model.** There is no HF/BF16 checkpoint on this machine, and the GGUFs beside it are llama.cpp
  quants of architectures `kindi convert` refuses ([docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)), so the note's
  head/layer extraction and its "attention projections must stay dimensionally matched" rule are untested by me.
- **That ternarising a donor "needs significant retraining".** That needs a teacher/student pair and a training run.
  What is measured here is the mechanism behind it - the ternary residual is ~3.5% per block application and accumulates
  with depth - not the retraining budget.
- **Router distillation.** Nothing in this directory trains; `SplitRouter` is a structural demonstration that sub-vector
  routing is expressible and measurably different, not evidence that it distils well.
- **The framework comparison** (PyTorch vs Unsloth) is a claim about tooling I have not exercised.

## Calling it

```python
from tools.moe_loop.layers import LoopedMoEBlock

block = LoopedMoEBlock(d_model=32, num_experts=8, top_k=2, loops=3, alpha=0.5,
                       dispatch="index_add", router="split", experts_per_group=2)
out = block(torch.randn(512, 32))
```

`block.rows` counts the token-rows the experts actually multiplied, which is the portable cost number; on this CPU the
wall-clock difference between the two dispatch paths is only 39 ms against 53 ms because the Python loop over distinct
experts dominates at these sizes.
