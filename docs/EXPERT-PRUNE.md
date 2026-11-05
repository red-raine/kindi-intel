# Pruning MoE experts under an exact per-layer budget

The paging notes in this tree ask how to *afford* the experts a layer routes to. This asks the smaller question the
same router log can answer: which experts a layer can simply not have.

Two constraints make it a different problem from "drop the least-used experts". The budget is **per layer and
exact** - layer L's experts are its own 208 MB of tensors, and a layer that would use more than its allowance cannot
borrow a layer that would use less - and it is a **budget**, not a ranking, so the choice is which bytes to keep
rather than which experts to keep. The solve is therefore the one `--budget` already uses for quantisation formats:
`kindi.rco.plan`, the multiple-choice knapsack that decides which tensors stay f32 under an exact byte budget.

The tool is [tools/expert_merge/prune.py](../tools/expert_merge/prune.py); the log it reads is
[tools/expert_merge/routes.py](../tools/expert_merge/routes.py)'s capture of a real Qwen3-Coder-30B-A3B (48 layers,
128 experts, top-8, 384 tokens of four documents).

## What an expert is worth, as far as the log can say

The quantity is **routing mass**: the sum, over every token in the log, of the renormalised top-k weight the router
multiplied that expert's output by. Dropping an expert hands its mass to the experts that remain, at that layer, so
mass is exactly what the model loses there.

It is not the loss. The rest of the loss is how much the survivors disagree with the expert that went, and that
number is not in the log - it lives in the expert weights, which is a different measurement (and
[the merge note](EXPERT-MERGE.md) shows the weights are not interchangeable even when the router groups two experts
together). So the plan is fitted to mass and then **gated on NLL**: the tool does not present mass as a loss
estimate, and the gate is what decides whether a plan is usable.

## The solve is RCO's, with one format that costs nothing

`kindi.rco.plan` chooses one format per item, total bytes within a budget, summed error minimised. Pruning is that
with two formats:

| format | bytes | error |
|---|---:|---|
| `drop` | 0 | the expert's routing mass |
| `keep` | its bytes in the pack | 0 |

`weights=1` in the entries because RCO multiplies a per-weight error by a weight count to get a total, and the
quantity here is already a per-expert total. The fifteen-line function that builds those entries (`plan_layer`) is
the whole adaptation; everything else - the exact DP, the 1 KiB grid, the greedy fallback with its fractional bound -
is the solver the converter already runs.

Two things this exposed:

**A zero-byte format was unrepresentable.** `kindi.rco.plan` read a per-format price with
`e["bytes"].get(f) or _bytes(e["weights"], f)`, so a format priced at 0 fell through to a lookup that does not know
the name and raised. It is now read with `is not None`, and `kindi.rco`'s own selftest checks that a free format is
priced, not missing.

**Where the DP earns its keep, and where it does not.** With experts of equal size, the optimum *is* the top-m by
mass, and the DP reproduces the sort exactly - the selftest asserts the set equality rather than claiming the DP
adds value there. The DP matters when prices differ: a pack whose layers were quantised to different types, or a
byte budget that lands between multiples of an expert. The selftest's constructed case is three experts of masses
10/8/7 and sizes 4/3/3 KiB against a 6 KiB budget: "heaviest first that fits" takes the 4 KiB expert and gives up 15
of mass, the DP takes the two 3 KiB experts and gives up 10. On this pack the sizes happen to be uniform (Q2_K gate
and up, Q3_K down, identical for every expert), so the count path and the byte path are the same plan at
96/128 = 0.75 - which is the cross-check that the two pricings agree.

## The plan

Each expert is 1,708,032 bytes (gate 516,096 + up 516,096 + down 675,840), 208.5 MB a layer, 9.77 GB of expert
weights across 48 layers.

| keep, of 128 a layer | expert weights | share | routing mass given up |
|---:|---:|---:|---:|
| 112 | 8.11 GB | 82.9% | 0.1% |
| 96 | 7.31 GB | 74.8% | 0.6% |
| 80 | 6.11 GB | 62.5% | 2.1% |
| 64 | 4.89 GB | 50.0% | 5.6% |

Two patterns in that table are worth naming.

**The tail is nearly empty.** At 112 of 128 the mass given up is 0.1%, and one layer keeps 88 of the 112 its budget
allows because 24 of its experts were never routed to by any of the 384 tokens. The solver does not spend a byte
that buys nothing - a zero-mass expert ties with dropping it, and the DP's strict comparison leaves it dropped.

**Half the bytes cost 5.6% of the mass.** Mass is heavy-tailed: the experts a budget removes are the ones few tokens
reach. That is the case for pruning, and it is also the reason it needs a gate rather than a threshold - a 5.6%
mass deficit sounds small and says nothing about what those tokens lose.

## The gate

Pruning is applied as a mask on the router: kept ids keep their logits, the rest go to `-inf`, the softmax runs over
the kept set and the top-k renormalisation applies as it always does. The two arms therefore differ in exactly one
thing - which experts exist - and share one loaded pack, so no re-quantisation and no second routing rule enters
the comparison. The gate corpus is held out of the fit (`--corpus`); the default is the log's own four documents,
which the tool labels in-sample so nobody has to remember which one they ran.

The first gate scored 94 held-out tokens - two documents the log never saw, 48 each. It was widened to **765 tokens
over three documents, 256 each** (`docs/BENCHMARKS.md`, `kindi/tp.py`, `docs/PACK-ZOO.md`, all outside the fit), four
arms, one process an arm, and the four baselines agreed bit for bit (3.4499) - which is the check that the arms are
one experiment. `tools.expert_merge.prunegate` merges the shards, refuses them if their baselines or corpora differ,
and names what each interval establishes:

| keep, of 128 a layer | expert bytes | mass given up | NLL | change | 95% interval | width | verdict |
|---:|---:|---:|---:|---:|---|---:|---|
| 128 (as shipped) | 9.77 GB | - | 3.4499 | - | - | - | - |
| 112 (82.9%) | 8.11 GB | 0.07% | 3.4640 | **+0.0142** | -0.0061 .. +0.0377 | 0.044 | unresolved |
| 96 (74.8%) | 7.31 GB | 0.59% | 3.4719 | **+0.0220** | -0.0057 .. +0.0534 | 0.059 | unresolved |
| 64 (50.0%) | 4.89 GB | 5.61% | 3.5140 | **+0.0641** | -0.0017 .. +0.1354 | 0.137 | unresolved |
| 32 (25.0%) | 2.62 GB | 25.30% | 3.9612 | **+0.5114** | +0.3822 .. +0.6544 | 0.272 | **loss (certified)** |

Three things changed with the corpus, and the first is the one worth keeping.

**The sign flipped.** On 94 tokens of two documents pruning looked free at 96 experts (-0.017) and at 64 (-0.071). On
765 tokens of three it costs +0.014, +0.022 and +0.064, and the ordering is now monotone in what is dropped. The
negative numbers were a property of the two documents, not of the mechanism: the earlier reading should never have
been quoted without that caveat, and the caveat is now measured rather than assumed.

**At 765 tokens pruning is a spectrum and only the far end is certified.** At 112 a layer the point estimate
(+0.0142) is inside the tolerance and its interval does not exclude zero; at 96 it is just outside the tolerance; at
64 it is six times the tolerance; at 32 - a quarter of the expert bytes, 25.3% of the routing mass - the interval
sits entirely above the tolerance (+0.38 to +0.65), which is the one claim this corpus supports without
qualification. Routing mass is not the loss, and the gap is measurable: 25.3% of the mass going costs 25x what 5.6%
does in NLL.

**The tolerance is finer than this measurement resolves, and here is the arithmetic.** The paired per-token delta has
a standard deviation of 0.31 nats at 112 (0.42, 0.97, 1.92 as more is dropped), so the tokens an interval of half
-width 0.02 needs are `(1.96 * sd / 0.02)^2`: **918 at 112, 1,668 at 96, 8,983 at 64 and 35,425 at 32**. Certifying
the *verdict* costs more again, because the point estimate already sits inside the tolerance at 112 (which needs a
half-width of 0.0058, so ~10,900 tokens) and just outside it at 96 (~167,000 to push the lower bound past 0.02). The
765 tokens here are 2.2 h of 48-layer forward across four processes; 10,900 is a day and a half. That is the honest
answer to "is pruning lossless within 0.02" **at 765 tokens and this forward rate**: the gate cannot certify it, and the
decision has to be made on the point estimate with its interval printed beside it. What that rate is a limit of is the
next two sections.

The tool says this instead of printing a bare "pass": a plan whose interval straddles the tolerance is reported
`marginal` by `prune` and `unresolved` by `prunegate`, and only an interval entirely inside the tolerance (or entirely
beyond it, as at 32) is reported as established.

## 1,785 tokens: the plan that certifies

So the gate was widened to **1,785 held-out tokens over seven documents** - `docs/BENCHMARKS.md`, `kindi/tp.py`,
`docs/PACK-ZOO.md`, `docs/TESTING.md`, `docs/RECIPES.md`, `tools/nll.py`, `docs/EXPERT-MERGE.md`, 256 each, none of
them in the fit - and the three arms the note has an opinion about were re-run (112 and 96 in one process, 64 in a
second), each against a baseline it measured itself:

| keep, of 128 a layer | expert bytes | mass given up | NLL | change | 95% interval | width | verdict |
|---:|---:|---:|---:|---:|---|---:|---|
| 128 (as shipped) | 9.77 GB | - | 3.6742 | - | - | - | - |
| 112 (82.9%) | 8.11 GB | 0.07% | 3.6787 | **+0.0045** | -0.0082 .. +0.0174 | 0.0255 | **lossless** |
| 96 (74.8%) | 7.31 GB | 0.59% | 3.6906 | **+0.0164** | -0.0023 .. +0.0356 | 0.0378 | unresolved |
| 64 (50.0%) | 4.89 GB | 5.61% | 3.7707 | **+0.0965** | +0.0504 .. +0.1450 | 0.0947 | **loss** |

**At 112 experts a layer, pruning is lossless within 0.02 nats a token, and that is now a claim rather than a point
estimate.** The interval's whole width is 0.026, so it sits inside the tolerance on both sides, on text the plan was
not fitted to. 8.11 GB of expert weights for the 9.77 GB file - a sixth of the expert bytes gone, and the cost is
distinguishable from nothing at this resolution.

**The point estimates fell, which is what noise looks like when the corpus grows.** 765 tokens read +0.0142 at 112
and +0.0220 at 96; 1,785 read +0.0045 and +0.0164. Both halved, and both moved toward zero. That is the direction a
corpus made of two documents was always able to produce, and it is why the earlier numbers were labelled unresolved
rather than accepted.

**96 experts is still unresolved, and the arithmetic for resolving it is now the only thing standing in the way.**
The estimate (+0.0164) is inside the tolerance and the interval is 0.038 wide, so the honest statement is "costs at
most +0.036, point estimate under the tolerance, not certified". Pushing the upper bound under 0.02 needs a half-width
of 0.0036 against the 0.0189 measured - 27x the corpus, about 49,000 tokens, and at ~13 minutes a forward that is 13
hours of this machine rather than the day and a half it used to be. The lever exists now; whether 96 is worth 13 hours
is a different question from whether it can be answered.

**64 experts flipped from unresolved to certified loss, and that is the other half of the widening.** At 765 tokens
it read +0.0641 with an interval that reached from -0.002 to +0.135 - not established. At 1,785 it reads +0.0965,
+0.0504..+0.1450: half the expert bytes costs about a tenth of a nat a token, which is five times the tolerance and
not a number two documents could have produced by accident. So the corpus settled both arms it had an opinion about,
in opposite directions, which is the reason to trust either: a wider corpus that only ever helps is a corpus being
fitted, and this one moved a cost up while it moved another one down.

**Where that leaves the budget.** 112 of 128 a layer is 8.11 GB of 9.77 GB, certified lossless within 0.02 on held-out
text: a sixth of the expert bytes for nothing measurable, and the only plan in this note that earns the word. 96 is the
one worth another corpus - 7.31 GB, a quarter of the experts gone, point estimate under the tolerance - and 64 is out
by a fifth of a nat.

## The forward was the limit, and it is not one any more

Everything above is priced in documents because the forward paid the dequantiser for a layer's experts once **a
document**. That is not a property of the model - it is a property of walking the corpus one document at a time. The
experts a 256-token document's tokens route to already run close to all 128, so the second document unpacks the same
weights again and gets nothing for it. `forward_docs` batches the corpus into one pass: attention stays per document
(a document must not attend to its neighbour), the sparse block is one batch, and the expert union is taken over the
whole corpus and paid once.

`tools.expert_merge.docbatch` measures both paths on the real pack rather than assuming the batching is free of
consequences - 3 documents of 96 tokens, a 6-layer prefix:

| | serial | batched |
|---|---:|---:|
| wall clock, 285 tokens | 210.9 s | **58.2 s** |
| corpus NLL | 11.564841 | 11.564841 |
| max abs, per-token NLL | - | **1.0e-05** |
| mean abs, per-token NLL | - | 1.6e-06 |

The difference is fp32 rounding, not a different computation: a GEMV over 96 rows and one over 288 block differently
inside BLAS, and the corpus NLL agrees to 1.4e-08. What it buys is the whole point - 3.63x on three documents, and the
ratio grows with the corpus, because the batched cost is one dequant a layer plus a GEMV a token while the serial cost
is one dequant a layer **a document**. At full depth the batched pass is ~8 minutes for the corpus; the 96-expert
requirement of 1,668 tokens is ~45 minutes of a gate that would have been a day and a half at the serial rate.

It costs memory, and this is the honest column: one layer's experts as f32 is 2.42 GB, and the batch's logits are
152k floats a token (1.1 GB at 1,785 tokens). The serial path never holds more than one document's. That is the trade,
and it is the reason the gate batches at all: 13 GB of peak RSS to turn a day and a half into half an hour.

The wider gate above is what that looks like in practice: three forwards of 1,785 tokens in 42 minutes end to end,
against 2.2 hours for four forwards of 765 tokens before it. Same machine, and the tokens per hour went up 5.5x.

## What this does not establish

**The log is 384 tokens of four documents.** Per-expert counts are ~24 selections, so the mass of a rarely-routed
expert is estimated from a handful of tokens; a wider corpus would move the plans in the 112-96 range, where the
decision is between experts with almost no mass and almost no mass. The plans at 80 and 64 are decided on tens of
selections each and are less sensitive to that.

**The gate is small, and narrower than the corpus.** 1,785 tokens over seven documents is 2.3x the first widened run
and 19x the original 94, and all seven are repository prose and Python: documents are the unit a moving-block interval
with a block of 8 does not resample, so the interval describes token noise inside this kind of text, not the spread
over corpora. The 112 result is a statement about prose-and-Python, not about language.

**The forward is faster than this note used to say, and slower than it could be.** The dequantiser runs at ~14-65
Mvalues/s here, which was the whole cost of a document; batching the corpus pays it once for the corpus instead of
once a document, which is the difference between a day and a half and half an hour. The GEMV half of the forward is
now the part that scales with tokens, and it is numpy, not a kernel.

**No pack is written.** The artifact is a JSON plan (layer -> kept ids) plus the routing mask that applies it. What a
pruned shard *would* weigh is the table above; what this tree can serve is still the 11.26 GB file, because the
engine's CPU path does not read MoE packs. Turning the plan into bytes is a converter job and is not done here.

**Budgets below 8 experts a layer change two things at once.** The mask clamps the slot count to the experts that
exist, so a layer keeping 4 routes to 4 instead of 8 - fewer experts *and* fewer GEMV slots. The gated plans here
are all at or above 64 to keep that out of the comparison.

**Nothing here says the kept experts are the right ones for text the log never saw.** The mass was measured on
repository prose and Python; a corpus with a different distribution of subject matter routes differently, and a plan
fitted on this one is fitted on this one.
