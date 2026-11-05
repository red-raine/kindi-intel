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
and +0.0220 at 96; 1,785 read +0.0045 and +0.0164 - two thirds and a quarter of the way to zero, both in the same
direction. That is what a corpus of three documents was always able to produce, and it is why the earlier numbers were
labelled unresolved rather than accepted.

**96 experts is still unresolved, and the arithmetic for resolving it is now the only thing standing in the way.**
The estimate (+0.0164) is inside the tolerance and the interval is 0.038 wide, so the honest statement is "costs at
most +0.036, point estimate under the tolerance, not certified". Pushing the upper bound under 0.02 needs a half-width
of 0.0036 against the 0.0189 measured - 27x the corpus, about 49,000 tokens, which is ~6 hours a forward for this pack
and so ~12 for the baseline and the arm (in batches of documents: 49,000 tokens of logits would be 30 GB at once). The
lever exists now - the day and a half became a working day - and whether 96 is worth that day is a different question
from whether it can be answered.

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

## What a mass budget cannot see

Routing mass is the right currency for a byte budget and the wrong one for a hazard, because it treats an expert as a
small linear map whose removal costs its share of the output. Some experts are not that. `tools/expert_merge/superex`
measures what each expert actually writes into the residual stream - `max |down_proj(silu(gate x) * up x)|` over the
tokens that chose it, on the same `ffn_norm(x)` the router handed it - for all 48 layers of the Q2_K pack, one
dequantisation of three matrices an expert (~7 min for the pack):

| expert | output | x its layer's median | routing-mass rank | share of its layer's mass |
|---|---:|---:|---:|---:|
| L02E92 | 946.94 | **2,152x** | 38 / 128 | 1.03% |
| L01E114 | 283.32 | **927x** | 27 / 128 | 1.32% |
| L03E82 | 179.20 | **525x** | 27 / 128 | 1.40% |
| L01E68 | 64.23 | **210x** | **92 / 128** | **0.28%** |

The four are a separate population and the gap is not close: the next largest expert in the pack is 42.7x its own
layer's median (L46), then 38x, then 33x. Two of these ids are the ones Su et al. (ICLR 2026) name as super experts in
Qwen3-30B-A3B - **L2E92 and L3E82** - and their third, **L1E68**, is our fourth entry; we find one they do not,
L1E114, at four times E68's output. Different model (the Coder variant), different corpus, same structure.

**Mass does not rank them, and that is the whole finding.** L1E68 writes 210x its layer's median while carrying
**0.28%** of that layer's routing mass, at rank 92 of 128. A byte budget that keeps the top 64 by mass drops it, and the
plan's own summary reports the loss as a fraction of a percent of mass - which is exactly what it is, and exactly why
the summary cannot see it.

| plan | flagged experts dropped | its measured NLL cost |
|---|---|---|
| 112 of 128 | none | +0.0045 (lossless) |
| 96 of 128 | none | +0.0164 (unresolved) |
| 96 of 128, guarded | none | +0.0173 (unresolved; same plan, re-measured) |
| 64 of 128 | **L1E68** | +0.0965 (loss, certified above); +0.0963 re-measured beside the guarded arm |
| 64 of 128, guarded | none | +0.0914 (loss) |
| 32 of 128 | **L1E68, L2E92** | not gated |

The 64 arm's fifth of a nat has a mechanism, not just a magnitude **- but the mechanism is not the flagged expert.**
Removing L1E68 costs **0.0049 nats of the 0.0963** (the guarded and unguarded 64 arms differ by that on identical
tokens), so the outlier is 5% of the loss and the rest is ordinary expert capacity. The earlier version of this
paragraph read the overlap as causation; the guard experiment is what separates them. And the 96 plan survives the
outlier test only because 96 > 92. That is a budget comparison, not a guard: a different corpus re-ranks mass, and a slightly tighter plan drops
the expert at rank 92 while reporting the same sub-percent mass loss. The fix is to make it a guard - `prune
--protect` forcing the flagged set into the keep set before the DP runs (3 or 4 experts of 6,144, 0.05% of expert
bytes) - which is item **P0** in [the master plan](../masterplan.md).

### The guard, and what it costs

`prune --protect L:E,...` and `--protect-json superex.json` take the flagged experts out of the layer's budget first and
keep them unconditionally; the same knapsack is then solved over the experts that are left, at the smaller budget. Two
properties follow, and `--selftest` asserts both: the guard **spends** budget rather than raising it (a 96 plan with
four protected experts holds 96, not 100), and a guard that does not fit the budget **stops the run** rather than being
dropped quietly - a protection that can be silently ignored is the failure the flag exists to prevent.

| plan | unguarded | guarded | what the guard costs |
|---|---|---|---|
| 96 of 128 | 4,597 slots, 7.31 GB, mass dropped 0.5874% | 4,597 slots, 7.31 GB, 0.5874% | **nothing** - 96 > 92 keeps E68 anyway |
| 64 of 128 | 3,072 slots, 4.89 GB, 5.6100%, **drops L1E68** | 3,072 slots, 4.89 GB, 5.6160%, keeps it | E68's slot: 0.006 points of mass, **0.0049 nats** |
| the guard itself | | | 6,832,128 bytes, **0.07% of the expert weights** |

Those are plans, not losses: the mass column is the currency that cannot see the hazard, so it is reported to show what
the guard costs and not to argue that it pays. The NLL walk is what decides.

**The walk, on 1,785 held-out tokens (seven documents, `--tokens 256`).** At 96 the guarded plan is byte-identical to
the plan certified above, so that arm is a reproducibility check as much as a P0 one: NLL 3.6670 → 3.6843,
**+0.0173, 95% CI −0.0012..+0.0361** - the same verdict as the +0.0164 recorded above, on text that has since moved.
One of the seven documents was edited after that certification, so this baseline reads 3.6670 where the earlier run read
3.6742; the two are not the same corpus, and only this run's pairing is quoted. At 64 the guard does **not** rescue the
plan: 3.6670 → 3.7584, **+0.0914 [+0.0445, +0.1401]**, rejected across the whole interval.

**The unguarded 64 arm on the same 1,785 tokens is +0.0963 [+0.0505, +0.1443]** (a second process, baseline
3.6670 bit for bit - the forward is deterministic, so the two arms are quoted against the same tokens, and this also
**reproduces the certified +0.0965 to 0.0002** on the same seven documents a day later). So keeping the
expert that layer 1 would otherwise drop is worth **0.0049 nats**, 5% of that plan's cost, and both arms fail the
tolerance by a wide margin. The difference is well inside either interval: this is a point estimate in the direction
the hazard argument predicts, not a resolved effect, and no A/B interval is quoted because the per-token deltas of the
two processes were not stored against each other.

That is the result worth saying plainly: protecting the four outliers is **necessary and not sufficient**. It costs
0.07% of the expert bytes and it keeps the guard honest for the plans we ship, but at 50% expert reduction the loss is
not in those four experts, and a plan that keeps them still costs a tenth of a nat.

**`--protect-json` re-derives the set instead of trusting a stored mask.** `superex` has written two masks, and an
artifact older than the relative threshold carries the *global* 10%-of-peak cut under `flagged` - on this pack that is
31 experts, 23 of them in layer 47 at 1.7x their own layer's median, with L1E68 absent. Reading that key would have
guarded the wrong 31 experts and spent 31 of a 96-plan's slots on them. A file with an `output` profile is therefore
re-thresholded by `superex.flagged` at 100x the expert's own layer's median, and the tool prints which of the two it
took.

**The threshold has to be relative to the layer, which is why the tool's first version was wrong.** The literature's
stated cut is global (P99.5 and 10% of the largest output anywhere). On this pack the layer median rises 150x with
depth - 0.31 at layer 1, 56.6 at layer 47 - so a global 10%-of-peak cut flags **31** experts, 23 of them in layer 47
that are only 1.7x their own layer's median, while the layer-1 expert at 210x sits under it. Relative to the expert's
own layer at 100x, the population separates cleanly at four, and the ratio is a margin inside the gap (42.7x to 210x),
not a tuned number. `superex` reports both and uses the relative one.

**What this does not establish.** The profile is on the 384-token fitting corpus, not held-out text, so a different
corpus re-ranks the set - which is an argument for the guard, not against it. Output magnitude is not the paper's
test: they removed the experts and measured, and no removal arm has been run here. And a large output is a *mechanism*
for the 64 arm's cost, not a proof that E68 is the whole of it.

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

**The outlier profile is on the fitting corpus too.** `superex` measures what each expert writes over the same 384
tokens the plans were fitted on, so the flagged set is as corpus-specific as the plans are - which is the reason the
answer is a guard that protects a *superset* (every layer's top expert by output, and the 100x set) rather than a
number computed once and hard-coded.

## Reproduce

```bash
# the mass plans, priced from the pack's own bytes
python -m tools.expert_merge.prune --pack MODEL.gguf --routes routes.npz --keep 112,96,64,32 --json plans.json

# what each expert writes into the residual stream, and whether a plan drops an outlier (~7 min for 48 layers)
python -m tools.expert_merge.superex --pack MODEL.gguf --routes routes.npz \
    --plan plans.json --json superex.json

# re-check a stored profile against another plan without reading the weights again (seconds)
python -m tools.expert_merge.superex --from-json superex.json --plan plans.json

# the gate: 1,785 held-out tokens, batched, each arm against a baseline it measures itself
python -m tools.expert_merge.prune --pack MODEL.gguf --routes routes.npz --keep 112,96 --gate \
    --corpus docs/BENCHMARKS.md,kindi/tp.py,docs/PACK-ZOO.md,docs/TESTING.md,docs/RECIPES.md,tools/nll.py \
    --docs 7 --tokens 255 --json gate-112-96.json

# the same gate with the outlier guard: 96 and 64, the flagged experts forced into every layer's keep set
python -m tools.expert_merge.prune --pack MODEL.gguf --routes routes.npz --keep 96,64 \
    --protect-json superex.json --gate \
    --corpus docs/BENCHMARKS.md,kindi/tp.py,docs/PACK-ZOO.md,docs/TESTING.md,docs/RECIPES.md,tools/nll.py,docs/EXPERT-MERGE.md \
    --docs 7 --tokens 255 --json gate-guard-96-64.json
# then the verdicts, from the shards (refuses shards whose baseline disagrees)
python -m tools.expert_merge.prunegate gate-112-96.json
```
