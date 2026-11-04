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

Ninety-four held-out tokens - two documents the log never saw, 48 each - were scored through both routers in one
process, and the arms differ token by token:

| keep, of 128 a layer | NLL | change | 95% interval | tolerance +0.02 |
|---:|---:|---:|---|---|
| 128 (as shipped) | 4.9563 | - | - | - |
| 96 (74.8% of the bytes) | 4.9391 | **-0.0171** | -0.0625 .. +0.0469 | inside it |
| 64 (50.0% of the bytes) | 4.8850 | **-0.0712** | -0.2194 .. +0.0595 | inside it |

Both plans pass the tolerance on the point estimate and **neither interval excludes zero**. The honest reading is
that dropping a quarter or half of the expert bytes costs nothing measurable on these 94 tokens - not that pruning
improves the model. The negative signs are small against intervals of ±0.05 and ±0.14 and come from two documents;
what the mechanism does is visible in the routing (top-k renormalisation over a smaller set hands the survivors more
weight), and whether that helps or hurts on this or any other corpus is exactly what 94 tokens cannot settle.

The gate's own limit is arithmetic and stated by the tool: a moving-block interval of ±0.05 is wider than a 0.02
tolerance, so a plan can be **accepted without the tolerance being established** - the run prints that case instead
of a bare "pass". Narrowing it means more documents, which is about eight minutes of 48-layer forward a document on
this host (less for a pruned arm, which unpacks fewer experts), not a better argument.

## What this does not establish

**The log is 384 tokens of four documents.** Per-expert counts are ~24 selections, so the mass of a rarely-routed
expert is estimated from a handful of tokens; a wider corpus would move the plans in the 112-96 range, where the
decision is between experts with almost no mass and almost no mass. The plans at 80 and 64 are decided on tens of
selections each and are less sensitive to that.

**The gate is small.** Both documents are held out of the fit, but 48 tokens each is not a held-out set in any
statistical sense, and the moving-block interval says so out loud.

**No pack is written.** The artifact is a JSON plan (layer -> kept ids) plus the routing mask that applies it. What a
pruned shard *would* weigh is the table above; what this tree can serve is still the 11.26 GB file, because the
engine's CPU path does not read MoE packs. Turning the plan into bytes is a converter job and is not done here.

**Budgets below 8 experts a layer change two things at once.** The mask clamps the slot count to the experts that
exist, so a layer keeping 4 routes to 4 instead of 8 - fewer experts *and* fewer GEMV slots. The gated plans here
are all at or above 64 to keep that out of the comparison.

**Nothing here says the kept experts are the right ones for text the log never saw.** The mass was measured on
repository prose and Python; a corpus with a different distribution of subject matter routes differently, and a plan
fitted on this one is fitted on this one.
