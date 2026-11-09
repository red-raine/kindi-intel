# Are routing clusters permutations of one function?

The compiler specification merges an expert *family* by permutation-aligning each member's SwiGLU MLP to an anchor
(cosine similarity of `gate_proj` weight rows, `linear_sum_assignment`) and then averaging the tensors with a diagonal
Fisher. The alignment is only meaningful if the members really are one function wearing a relabelled hidden layer: two
permuted copies average back to the function exactly, two merely similar experts average to a third thing that is
neither of them.

So the question is not whether experts cluster — the router decides that — but what the cluster *is*. This note measures
it on a real pack: the experts the router treats as interchangeable, the bijection that best maps one onto the other,
and what the merge costs on the activations the router actually sent them.

**The answer is that a routing cluster is not a relabelled function.** All 144 expert pairs the router groups
together are *merely similar*: in no routed pair does a hidden unit have both its gate and up row matched in the other
expert at |cos| > 0.99, and the best relabelling — a full orthogonal map, which is more than the merge is allowed —
still leaves 0.711 of the difference behind. The shipped merge measures the same thing from the other side: its own
aligned cosine on those pairs is **0.102** median, **0.105** mean, against a gate of 0.85, and it recovers **0 of 144**. What a cluster offers
the alignment is what noise offers it: the best bijection buys 3.0% of the residual on a routed pair and 2.2% on a pair
drawn at random.

## What was measured

`Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf`: 128 experts per layer, top-8 routing, **48 layers**, every one of them
readable. That last part needed checking, and the first attempt did not: the copy on this machine was 11,318,112,204
bytes where its own index declares 11,258,612,896, its first 3,504,728,316 bytes decoded and everything past that was
uniform random. Not a truncation — 59 MB *longer* than the model — which is what a resume that writes a body at the
wrong offset leaves behind. A forward pass over it produced thirteen layers of router logits that read exactly like a
result, which is the failure this note's tooling now refuses: `Moepack.require_complete` stops the sweep, and `python -m tools.expert_merge.fetch --repair` keeps the readable prefix and re-fetches from the first byte that does
not decode, each chunk written at a `Content-Range`-validated offset. The repaired file is byte-identical to the
damaged one up to 3,490,000,000 bytes and differs from 3,504,728,316 - the exact byte `--verify` names - so the
measurement below is the same GGUF, not a substitute model.

Corpus: four documents from this repository (`docs/CPU-PERF-2026.md`, `tools/kl.py`, `README.md`, `kindi/cpu.py`),
the first 96 tokens of each, one prefill per document — **384 tokens per layer** of the text a pack like this is
quantised for. The router log stores, per layer and token, the 128 logits, the top-8 ids and renormalised weights, and
`x_n = ffn_norm(x)`: the vector the experts were actually given, which is what a merge has to be judged on.

Three pair sets per layer, three routed pairs and three random ones, seven pairs per layer across all 48 layers, so
that "no pair is a permutation" cannot be an artefact of how the pairs were picked:

| set | how the pair was chosen | pairs |
|---|---|---:|
| routed | the highest cosine between routing profiles — the experts this model treats as interchangeable on this corpus | 144 |
| weights | the pair whose flattened gate+up weights are most alike, among experts that fire at all | 48 |
| random | two firing experts drawn without replacement (control) | 144 |

A routing profile is `[128 experts, 384 tokens]`, the weight each expert was given per token; the routed pairs have a
profile cosine of **0.779** median (max 0.994, 17 of the 144 above 0.9), which is what "the router cannot tell them
apart" looks like as a number. They are not a coincidence of the corpus: they co-fire on the same tokens **10.8x** more
often than independence would give (Jaccard 0.548 over the tokens each one fired on; median pair: both on 20 of 384
tokens, the two firing on 27 and 26).

## Is one a permutation of the other?

If expert B is expert A with hidden unit *i* renamed to π(*i*) and optionally negated, then
`gate_B[π(i)] = s_i·gate_A[i]`, `up_B[π(i)] = s_i·up_A[i]` and `down_B[:, π(i)] = s_i·down_A[:, i]` — the same
relabelling shows up in all three matrices. The test uses all three as evidence for one bijection over the 768 hidden
units, then asks what the residual is after applying it, measured as `‖B − A‖ / ‖A‖` over the three matrices:

| set | pairs | identity | best bijection | gain | random permutation | units with a partner at \|cos\|>0.99 | same partner in gate and up |
|---|---:|---:|---:|---:|---:|---:|---:|
| routed | 144 | 1.414 | 1.369 | 0.030 | 1.415 | 0.0000 | 0.051 |
| weights | 48 | 1.416 | 1.369 | 0.031 | 1.421 | 0.0000 | 0.064 |
| random | 144 | 1.410 | 1.378 | 0.022 | 1.410 | 0.0000 | 0.009 |

Verdicts over all 336 pairs: **0 permutations, 0 aligned-similar, 336 merely-similar.**

Read `identity ≈ 1.41` before reading anything else: a permutation of ternary rows would leave the residual at 0 (see
the control below), while two unrelated experts land at √2 because their rows are near-orthogonal at this width. So the
number to look at is the **gain** — how much the best bijection buys over doing nothing — and it is 3.0% on the routed
pairs, against 2.2% on pairs drawn at random. The assignment is finding structure, but the same amount of it in noise:
`best bijection 1.369` against `random permutation 1.415` is the whole margin, and both are 1.41 to two digits. Median
best-matching row cosine is 0.102, and no routed pair has a hidden unit whose best partner at both gate and up reaches
0.99, where a relabelling would put all 768. A relabelling would be exact, not approximate: Q2_K is row-equivariant, so
a permuted copy stores the same 84-byte blocks in a different order, and the test would return zero.

The same conclusion survives the most generous relabelling there is. An arbitrary orthogonal map `M → MQ` fitted per
matrix (SVD of `AᵀB`) leaves a residual of **0.711** median across all 48 layers (0.681-1.051): more than half of each
pair's difference is outside anything a change of basis, permutation included, can remove.

The most weight-alike pairs are worth a line of their own, because a merge proposed on weights alone would start there:
the most alike pair in a layer reaches a flattened weight cosine of only **0.007** median over the 48 layers
(0.004-0.018), and gains **0.031** from its best bijection — slightly *more* than the routed pairs, which is to say the
weight-similarity ranking is noise at this quantisation and the relabelling it suggests is also noise.

## What the merge costs

For each routed pair, both parents and the merged expert are run on `x_n`, the activation rows the router sent, and the
merged output is compared to each parent's as a relative error. The merge under test is the module's own
(`merge.align_and_fuse`, the specification's version, scalar Fisher), not a second implementation of it:

| | relative error of the merged expert vs the anchor |
|---|---:|
| how far the two parents already are from each other | 1.401 |
| average, no alignment | 0.939 |
| average after the shipped alignment | 0.929 |
| an expert averaged with an identical copy (the floor) | 0 |

The alignment buys **1.1%** of the error. On 14 of the 144 routed pairs it makes the merge worse, and the worst case is
the same pair the 13-layer run flagged, layer 1's experts 68 and 114, where it reassigns rows that were already in a
better order: 0.535 → 1.359. Merging a routed pair lands 0.929 away from the expert it was supposed to preserve, which
is to say the merge does not reproduce its parent; the comparison that separates a cluster from a coin flip is the
relabelling gain in the table above (0.030 against 0.022), because the random pairs of this sweep were ranked, not
merged.

That is not the shipped alignment being the weak part. `permtest` can also fit the *sign-aware* bijection — the one that
applies a real permutation rather than only a row match on positive cosines — and on the 13-layer subset of an earlier
run it reached the same median, 0.928, narrowing the worst case from 0.793 to 0.702 while winning on 27 of 39 pairs.
The full sweep re-runs only the shipped alignment, which is the one the pipeline uses. Two different alignments, the
same answer to three digits, because there is no relabelling to find: an alignment is only worth what the pair it is
fitted to allows.

The pipeline's own gate sees exactly this on the 48-layer run — all 144 routed pairs of that sweep: aligned cosine
**0.102** median, **0.105** mean, 0.000-0.200 across the pairs, and **0 of 144** reach the 0.85 the specification
requires before quantising. The gate is load-bearing — it is the only thing standing between the pipeline and an
average of unrelated functions — and on this evidence it rejects everything the router calls a cluster.

## Controls: a real permutation is found exactly

The test has to be able to say yes. Four experts of layer 0 were replaced with a permutation of themselves carrying
random sign flips; the bijection search recovered the planted mapping exactly for all four (`residual 0.0`, verdict
`permutation`), and `permtest` asserts that on synthetic experts on every run of the gate.

Two things follow from the control. First, signed permutations are exactly what the shipped alignment cannot represent:
its cost is `-(unit_rows(anchor) @ unit_rows(target).T)`, an *unsigned* cosine, so on a sign-flipped permutation the
merge lands 0.58-0.67 away from the anchor instead of 0. Second, its cosine gate catches that case too (0.59 on the
synthetic check, below 0.85) — the merge is unsafe and the gate is what makes it safe, which is a thin but real margin.

## What this means for the specification

The merge's premise fails on this model. Not "the clusters are weak" — they are strong, 10.8x co-firing and a profile
cosine of 0.78 — but "a cluster is not a relabelled function", so there is nothing for the alignment to undo and
nothing for the average to preserve. A pipeline that runs this stage is safe only because its own 0.85 gate refuses
every pair; relaxed to admit them, it would replace two working experts with a third that is 93% away from both.

The premise is still plausible where the stage is aimed: two checkpoints fine-tuned from one base, or a teacher and a
student mid-distillation, can genuinely differ by a hidden-unit relabelling. That case needs a second checkpoint and is
not measured here. What is measured is that routing similarity, and weight similarity, are not evidence of it.

## What merging them anyway costs: a pack and an NLL

Everything above says the groups are not relabelled functions and that the shipped gate refuses them. That leaves the
quantity the gate cannot answer: if a pipeline merged the groups regardless, what would it cost? A mask that mapped
two slots onto one matrix would leave the router and the renormalisation untouched, which is not a merged pack, so
[tools/expert_merge/mergepack.py](../tools/expert_merge/mergepack.py) writes one: every non-expert tensor copied byte
for byte, the expert stacks replaced by one matrix a group, the router reduced to one row a group, and `expert_count`
saying so. The result is read by the same reader and run by the same forward, so the two arms are two models.

**One partition, and exactly pairs.** The grouping is average linkage on the layer-averaged co-activation matrix with
a size cap of `ceil(experts / groups)`, so `--groups 64` on 128 experts is 64 pairs, not 64 clusters of whatever size
k-means felt like. Over the log, 17.7% of a token's routing weight lands in a group that another of its slots also
picked: that is the share of its expert arithmetic one averaged matrix computes instead of two.

**No permutation alignment, because its gate rejects every pair.** `merge.py` implements the specification's
permutation-then-Fisher fusion, and the sections above record what it finds on these exact pairs: aligned cosine
**0.102** median, **0.105** mean, against a 0.85 gate, 0 of 144. Permuting members onto an anchor would align noise, so the arithmetic here is the
plain weighted mean the specification falls back to, weighted by the routing mass each member carried. The router row
of a group is the same weighted mean - a merged pack has one row a group, so the routing changes as well as the
matrices, and the gate below sees both together.

**Precision had to be matched, and that is why there is a control pack.** This build of `gguf` decodes K-quants but
cannot encode them (`quants.quantize` raises `NotImplementedError` for Q2_K through Q6_K), so a merged stack cannot be
written in the type its source used. Writing the merged experts at Q4_0 while measuring against the source's Q2_K would
compare the merge with a *coarser* model and flatter it. Both arms are therefore written by the same code path at
Q4_0, and `--groups 128` is the control: no merge, the same re-quantisation of the same dequantised weights.

| pack | experts | k | type | size | NLL, 94 held-out tokens |
|---|---:|---:|---|---:|---:|
| source | 128 | 8 | Q2_K / Q3_K | 11.26 GB | 4.9563 (reference, not paired) |
| control | 128 | 8 | Q4_0 | 17.03 GB | 4.9053 |
| merged | 64 pairs | 8 | Q4_0 | 8.87 GB | 6.2641 |
| merged, matched fraction | 64 pairs | 4 | Q4_0 | 8.87 GB | 6.3071 |

| | change | 95% interval | verdict |
|---|---:|---|---|
| 128 experts -> 64 pairs, both at Q4_0 | **+1.3588** | +0.8984 .. +1.8029 | rejected (tolerance +0.02) |

Per document the move is the same shape: 4.2356 -> 5.6721 on `docs/BENCHMARKS.md` and 5.5751 -> 6.8561 on
`kindi/tp.py`, so this is not one pathological text. `exp(1.3588) = 3.9` - the merge makes the model's perplexity about
four times worse while halving the expert bytes: 8.16 GB against the control's 16.32 GB, both Q4_0, the rest of the
file copied unchanged in each (8.87 GB and 17.03 GB on disk). That half is measured against the control, the only
like-for-like baseline there is; against the source's own 11.26 GB the merged file is 8.87 GB, **21% smaller**, and no
merged pack at Q2_K exists to compare against because this build cannot encode that type.

That is the earlier section written in nats. The pairs this averages are the ones the router groups hardest - 10.8x
co-firing against independence, routing-profile cosine 0.78 - and replacing each with its mean still costs 1.36 nats a
token. The 0.85 gate is not over-cautious: it is the difference between this regression and a size win, and it is the
only thing that stood between the pipeline and it.

What the number still mixes together: the group's pooled router row changes which experts are chosen, and the averaged
matrix changes what the chosen one computes. This measurement cannot attribute the 1.36 between the two. The arm below
prices one half of that mixture - how many of the merged experts a token reads - and leaves the pooled row in both
arms; the section after it builds the pack this one cannot, a router with the source's 128 rows over 64 merged stacks,
and finishes the split.

## At the control's own routing fraction

The shipped merged pack reads **8 of 64** experts where the control reads **8 of 128**, so a token spends twice the
share of its pool on the merged model - the capacity asymmetry the limits section used to call the merge cost a lower
bound. `mergepack --retopk` writes the same weights at another k: `expert_used_count` is a header uint32, and the copy
differs from its source in those four bytes and nothing else (8,871,398,560 bytes, every tensor still decoding), so
the two merged arms are one model at two routing breadths.

| arm | experts | k | share of the pool | NLL, 94 tokens | delta vs control | 95% interval |
|---|---:|---:|---:|---:|---:|---|
| control | 128 | 8 | 6.25% | 4.9053 | - | - |
| merged, as shipped | 64 | 8 | 12.5% | 6.2641 | +1.3588 | +0.8984..+1.8029 |
| merged, matched fraction | 64 | 4 | 6.25% | 6.3071 | **+1.4018** | +0.9031..+1.8329 |

Per document the matched-fraction arm moves the same way as its sibling: 5.7448 and 6.8694 against 5.6721 and 6.8561.
It is also **half the expert arithmetic a token** - four merged matrices instead of eight - so at equal pool share it
is the cheaper end of the same pack, and it is the comparison the shipped arm was not making.

**The breadth effect has the predicted sign and does not resolve.** Walked against each other token by token, the two
merged arms differ by **+0.0430 nats, interval -0.1414..+0.2073** over the same 94 tokens: matching the control's
fraction costs slightly more than the shipped arm, as "more relative capacity flatters the merge" says it should, but
this sample cannot separate 0.043 from zero. What the arm does establish is the quantity the shipped comparison left
open - **the merge costs +1.4018 at the control's own routing fraction**, not +1.3588 at twice it - and that the
breadth difference is at most a sixth of the cost, not the explanation of it.

## Which half of the bill is which

A merged pack can keep its source's router. `mergepack --keep-router` writes one in the merge pass, and
`mergepack --router-from SRC` splices one onto a pack already merged - its 64 stacks are copied byte for byte, while
`ffn_gate_inp` keeps the source's 128 rows and three keys record what changed (`kindi.router_rows` = 128,
`kindi.router_used` = 8, `kindi.expert_map` = one stack id a slot). The source's top-8 over 128 experts therefore
still decides *which* experts fire; only the arithmetic behind each id changed. The map is a bijection onto the 64
stacks, and the pack's payload report re-decodes every tensor, so the two files differ in the router, the map and
`expert_used_count` and nothing else.

A token whose 8 slots land in the same group twice would read one stack where it read two, so this router's 8 slots
touch about 6.6 stacks. At a fixed budget of 4 stacks (`--used 4`), the two arms cost the same arithmetic and differ
only in *which* four a token spends them on: the pooled router's top-4 of 64, or the four heaviest stacks under the
source's 128-way ordering. That is the comparison below. Corpus is this run's wider one, 1,275 tokens over five
documents, so its absolute numbers are not the 94-token table's; every delta is a paired walk inside the run.

| arm | router rows | expert stacks | stacks read | NLL, 1,275 tokens | delta vs control | 95% interval |
|---|---:|---:|---:|---:|---:|---|
| control | 128 (source) | 128 | 8 | 3.4939 | - | - |
| merged, matched fraction | 64 (pooled) | 64 | 4 | 4.5707 | +1.0768 | +0.9394..+1.2105 |
| merged, source router | 128 (source) | 64 | 4 of the mapped stacks | 4.6984 | **+1.2045** | +1.0611..+1.3562 |

| | change | 95% interval | shares of the merge cost |
|---|---:|---|---|
| pooled router -> source router, arithmetic fixed | +0.1277 | -0.0139..+0.1695 | routing **at most +0.17**, point +0.13 |
| merge, pooled router (the shipped pack) | +1.0768 | +0.9394..+1.2105 | arithmetic, **at least +1.08** |

**The router is not where the money goes, and keeping it may cost a little.** Replacing 128 rows with 64 pooled ones
moves NLL by +0.1277 in the *wrong* direction for the usual story: the source's own router is the worse of the two,
consistently, on all five documents (+0.157, +0.166, +0.058, +0.050, +0.207). It is still not resolved - the interval
crosses zero, and 0.128 is inside what 1,275 tokens can call noise - so the honest reading is that the routing half of
the merge costs **at most about 0.17 nats and plausibly nothing**, and the arithmetic half carries **all of the rest**:
averaging two experts' three matrices into one leaves at least 1.08 of the ~1.2 total. The 10.8x co-firing and 0.78
profile cosine were a real reason to expect the pairs to be interchangeable; the measurement says the router already
found near-duplicates, and the *matrix* average is what breaks them.

The sign is explicable and is not a bug in the map. After the merge there is one expert where the router sees two, so a
128-row router scores two ids for one stack and a token can spend half its 8 choices on the same matrix - 17.7% of its
routing weight lands in a group another slot already picked - while the pooled row averages those two scores into the
one decision the stack can actually serve. The pooled router is the *consistent* one: its slots are the stacks. What
this does not settle is how much of the 0.128 is that duplication penalty and how much is the pooled row's own
information; separating those needs a 128-row router re-fitted against the merged stacks rather than copied, which
would be a fine-tune, not a rewrite.

## Can a re-fitted router beat the pooled one? (P4, routing fidelity, 384 tokens)

Pooling a group's rows is forced by the merge, not estimated by it: one stack, one row. `P4` asked whether a 64-row
router *fitted* to the teacher's routing does better than that pool. That was priced before any pack was written,
because the answer decides whether a 1,275-token walk is worth its 1.5 h: `tools/expert_merge/routerfit.py` fits, per
layer, the anchored ridge map

    argmin_W  || (x W^T) − (log p_64 − L_pool) ||² + λ ||W||² ,   prediction = L_pool + x W^T

where `p_64` is the teacher's 128-way distribution summed inside each group and `L_pool` is what the shipped pooled
router scores (a routing-mass-weighted mean of member logits). The anchor is the point: at λ→∞ the arm *is* the pooled
router, so the measurement can only say how far a re-fit improves on it, never how much a bad regulariser hurts.
Fitted on three of the four logged documents and scored on the fourth, all four folds, λ chosen per fold on that
fold's held-out tokens (`λ = 0.1 × trace(XᵀX)/d` won every fold):

| arm | KL(teacher grouped routing ‖ arm), nats |
|---|---:|
| pooled (the shipped merged router) | 0.0723 |
| **anchored ridge re-fit, 64 rows a layer** | **0.0311** |
| paired difference, ridge − pooled | **−0.0412** |
| 95% interval, moving-block bootstrap | −0.0428..−0.0396 |

Per fold the direction is identical and the sizes are close: 0.0784→0.0289, 0.0695→0.0283, 0.0733→0.0352,
0.0679→0.0320. The pairing is 18,432 token-layer observations (384 tokens × 48 layers), and the interval excludes zero,
so this is a resolved result on the routing distribution: **a closed-form re-fit of 64 rows per layer on 384 tokens
recovers a little over half of the pooled router's divergence from the teacher**, at 64×2,048 parameters a layer
fitted from 288 held-in tokens.

The same fit *without* the anchor is **worse than pooling**: +0.0520 [+0.0488, +0.0552], same folds, same tokens. That
is worth stating because it is the difference between a measurement and a misreading: an unanchored ridge is free to
depart from the shipped router in directions the log cannot constrain, and on this sample it does. A re-fit is only
interesting as a correction *to* the pool, not as a replacement for it.

What this is not: an NLL result. The 1,275-token paired walk has **not** been run. That gate needs the fitted rows
written into the merged pack (`routerfit --emit` writes them: `rows (48, 64, 2048)`, `xm`, `rm`, `gids` at λ = 0.1,
fitted on all 384 logged tokens at the λ the folds picked), and then a three-arm walk — control, pooled-router merged,
fitted-router merged — on five documents. Until that runs, the finding is that routing has **headroom**, not that it
buys nats of NLL: the routing half of the merge is at most +0.17, and 55% of a KL that small is not obviously 0.1 nats
of likelihood.

## What this does not establish

- **The attribution is a budget, not two exact fractions.** The router half is +0.1277 with an interval crossing zero;
  the arithmetic half is whatever remains of +1.0768..+1.2045. Read together: routing <= +0.17, arithmetic >= +1.08.
  Stating them as "0.13 routing / 1.08 arithmetic" would over-claim a split this sample cannot resolve.
- **The source-router arm spends its 4 stacks by truncation.** Its ordering is the source's 128-way one; the mapping
  collapses duplicate groups and then the heaviest 4 stacks are kept, the tail renormalised away. A merged model with a
  4-stack budget must truncate whichever router it has, so the arm is a fair choice of *which* four - but it is not the
  same object as a router trained for 64 experts, and the arm that lets all 8 slots fire (about 6.6 stacks) would trade
  routing breadth for arithmetic and confound the two.
- **The paired walk resolves the merge, not the breadth behind it.** The shipped arm reads 8 of 64 experts against the
  control's 8 of 128; the matched-fraction arm reads 4 of 64 (the same 6.25%) and costs **+1.4018**. That is +0.0430
  more than the shipped arm, with an interval of -0.1414..+0.2073 that covers zero: the capacity argument has the right
  sign, the size of the breadth effect is not measured by these 94 tokens, and +1.36 and +1.40 are two readings of one
  merge rather than a bound and a better bound.
- **The two halves ran at different precisions.** The permutation test is on the Q2_K pack's own ternary rows; the NLL
  arm re-quantises both sides to Q4_0, so it measures the merge at a finer residual than the pack it read (which is the
  only way to keep the comparison fair here, and not the same question as a merged pack written in K-quants).
- **One corpus, one model.** 384 tokens of this repository's own prose and code. The routed pairs are grouped by
  routing on *that* text, so a different corpus proposes different pairs — the permutation test would be re-run, not
  reused, and the answer could in principle differ per layer family.
- **The NLL is 94 tokens, not a benchmark.** The merged pack's +1.36 nats is teacher-forced NLL over two held-out
  documents, and it is one partition at one group size (64 pairs) of one 384-token log. What it establishes is the cost
  of this merge at matched precision, not what a re-trained or differently grouped merged model would score.
- **Every non-expert tensor is copied, not checked.** The pack is built by copying attention, embeddings and norms as
  bytes; nothing re-validates them, so the comparison is only as trustworthy as the copy path the selftest exercises
  on a small pack.
- **`r_orth` is one pair per layer.** A 2048x2048 SVD per pair is expensive; the 13 values are the top routed pair of
  each layer, and they are tight (0.681-1.051), but they are not independent samples of the layer.

## Reproduce

```bash
# is the pack whole? prints the first tensor that does not decode, and the byte it breaks at
python -m tools.expert_merge.fetch --verify MODEL.gguf

# repair a damaged pack: keep the readable prefix, fetch from the first bad byte, re-check the hash
python -m tools.expert_merge.fetch --repair MODEL.gguf --url URL --sha256 HEX --jobs 6

# one prefill per document; writes routes.npz + routes.json (384 tokens x every layer)
python -m tools.expert_merge.routes --pack MODEL.gguf --out routes.npz

# 3 routed + 1 weight-alike + 3 random pairs per layer: verdict, residuals, merge cost on x_n
# (--jobs > 1 goes through a process pool, which loses its workers on this Windows host: the shards below are the
#  reliable way to use every core. Each shard writes its own json and resumes by layer.)
python -m tools.expert_merge.real --routes routes.npz --out merge.json --jobs 6
for i in 0 1 2 3 4 5 6 7; do
  python -m tools.expert_merge.real --routes routes.npz --out shard$i.json --shard $i/8 &
done
python -m tools.expert_merge.real --combine shard0.json shard1.json shard2.json shard3.json \
    shard4.json shard5.json shard6.json shard7.json --out merge.json

# the sign-aware alignment, as a second control on the same pairs
python -m tools.expert_merge.real --merge-only merge.json routes.npz --jobs 6

# the numbers above, read back off the json
python -m tools.expert_merge.real --summary merge.json

# the synthetic gate: planted permutations, noisy copies, unrelated experts (~1 s)
python -m tools.expert_merge.permtest

# the merged pack and its control, at the same quantisation (~35 min each on this machine: 48 layers x 128 experts)
# `--expert-type` writes the experts at one type for both arms; without it the merged stack is written in the source's
# own type, which for a K-quant pack this build cannot encode
python -m tools.expert_merge.mergepack --pack MODEL.gguf --routes routes.npz --groups 128 \
    --expert-type Q4_0 --out CONTROL.gguf
python -m tools.expert_merge.mergepack --pack MODEL.gguf --routes routes.npz --groups 64 \
    --expert-type Q4_0 --out MERGED.gguf \
    --gate --baseline CONTROL.gguf --corpus docs/BENCHMARKS.md,kindi/tp.py --docs 2 --tokens 48 \
    --json merge64.json

# the same merged weights at the control's own routing fraction: 4 of 64 is the same 6.25% as 8 of 128
python -m tools.expert_merge.mergepack --retopk MERGED.gguf --used 4 --out MERGED-k4.gguf

# all three arms against the control, the baseline walked once (~2 h on this machine), then the two merged arms
# against each other for the breadth effect's own interval (~1 h)
python -m tools.expert_merge.mergepack --compare CONTROL.gguf --against MERGED.gguf,MERGED-k4.gguf \
    --corpus docs/BENCHMARKS.md,kindi/tp.py --docs 2 --tokens 48 --json merge64-arms.json
python -m tools.expert_merge.mergepack --compare MERGED.gguf --against MERGED-k4.gguf \
    --corpus docs/BENCHMARKS.md,kindi/tp.py --docs 2 --tokens 48 --json merge64-k4-vs-k8.json

# the router half: the same 64 merged stacks under the source's own 128 router rows (~1 min, nothing is re-merged)
python -m tools.expert_merge.mergepack --pack MERGED-k4.gguf --router-from MODEL.gguf --routes routes.npz \
    --groups 64 --out MERGED-k4-srcrouter.gguf

# routing against arithmetic at a fixed budget of 4 stacks, over the wider corpus (~1.5 h)
python -m tools.expert_merge.mergepack --compare CONTROL.gguf \
    --against MERGED-k4.gguf,MERGED-k4-srcrouter.gguf \
    --corpus docs/BENCHMARKS.md,kindi/tp.py,docs/PACK-ZOO.md,docs/TESTING.md,docs/RECIPES.md \
    --docs 5 --tokens 255 --json merge64-route-attribution.json
```

`python -m tools.selftest` runs these suites (`suite:expert_merge`, `suite:expert_perm`, `suite:mergepack`) as part of
the repository gate.
