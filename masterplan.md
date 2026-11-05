# Master plan: a lossless import pipeline and a fast-enough CPU engine

This is the working plan for branch `freebuff/…-support-intel-cpu`. It exists so that a later session does not
re-derive a measurement that is already in `docs/`, and does not re-run an experiment whose answer is already in a
JSON file under `B:/models/scratch`. Every entry has a **gate** (how it is judged), a **cost** (what it takes on this
machine) and a **status**. Nothing here is a wish: an item is either measured, running, or explicitly waiting.

Two objectives, and they pull in opposite directions:

1. **Import stays lossless.** A converted pack must do what its source did. "Lossless" means a paired NLL walk inside
   0.02 nats with a 95% interval that excludes the tolerance, plus a functional screen — not a perplexity table.
2. **Inference on a 9th-gen Intel CPU gets faster.** AVX2, no AVX-512, DDR4. That machine is bandwidth-bound during
   decode, so bytes read a token is the currency, and the only free wins are ones that remove bytes or remove reads.

Anything that trades 1 for 2 has to say which nats it spent and quote the interval.

---

## 1. What we have measured (do not re-measure)

| finding | number | evidence |
|---|---|---|
| Ternary refit of a float checkpoint | **+0.021 nats** over the QAT-original, split by refit table / scale / norms arms | `docs/TERNARY-OPTIMIZATION-MATRIX.md`, `kindi/refit.py` |
| Prune to 112 of 128 experts a layer | **+0.0045 nats**, 95% CI −0.0082..+0.0174 → **lossless within 0.02**, for 82.9% of expert bytes | `docs/EXPERT-PRUNE.md`, 1,785 held-out tokens |
| Prune to 96 a layer | +0.0164, CI −0.0023..+0.0356 → **not established** | same |
| Merge 128 → 64 pairs, both arms Q4_0 | **+1.077 nats** at the control's own routing fraction (4 of 64), +1.3588 at 8 of 64 | `docs/EXPERT-MERGE.md`, 1,275 tokens |
| …of which **routing** costs | **≤ +0.17**, point +0.128 *in the pooled router's favour*, CI −0.0139..+0.1695 | kept-router arm, `mergepack --keep-router` |
| …of which **arithmetic** costs | **≥ +1.08** | same |
| Super experts in this pack | 4 experts at **210–2,152× their own layer's median output**: L2E92 (947, 2,152×), L1E114 (283, 927×), L3E82 (179, 525×), L1E68 (64, 210×). Next largest in the pack is 42.7×. **Mass ranks them 27, 27, 38 and 92 of 128** | `tools/expert_merge/superex.py`, `docs/EXPERT-PRUNE.md` |
| Their fate in our plans | 112 and 96 keep all four; **64 drops L1E68** (mass rank 92, 0.28% of its layer's mass) and 32 drops L1E68 and L2E92. So the 64 arm's +0.0965 has a mechanism, and 96 survives only because 96 > 92 | same |

The merge row is the most important: at 50% expert reduction the merge costs 1.08–1.20 nats, and the router is not
where the money went. That result is now corroborated externally (§3, REAP), which changes the plan: **merging is not
this project's compression lever. Pruning is.**

---

## 2. What the research changed

Read `appendix.md` for links and numbers. Five conclusions that moved the plan:

1. **Pruning beats merging at equal bytes on exactly this model class.** REAP (ICLR 2026) measured Qwen3-30B-A3B at
   25% and 50% reduction, no fine-tuning: pruning −1.9% coding at 25% / −6.9% at 50%; merging worse than −5% and −20%.
   Our own +1.08..+1.20 nats at 50% is the same verdict from the other side. **Consequence:** `P1` promotes pruning to
   the primary compressor; the merge becomes `P3` with explicit mitigations and a kill criterion.
2. **Three experts can carry 15 accuracy points.** Su et al. (ICLR 2026): 3 of 6,144 experts, layers 1–3, and removing
   them took Qwen3-30B-A3B from 70.22 to 55.00 with random removal costing 0.20. Our audit found the same structure —
   **L2E92 and L3E82 are the same ids** — and mass does not rank them in the top 26. This is `P0`, and it is a guard on
   the pipeline we already ship, not a new experiment.
3. **Router-only re-fitting recovers a lot after compression, and it is cheap** (~0.04% of params; helps fine-grained
   MoE most). Our own measurement says the router replacement is *small* (≤0.17), so a re-fit is unlikely to be
   worth a training run here — but a **closed-form ridge re-fit on the route log** is a CPU-hour, and PRIME does
   exactly that. That is `P4`, cheap enough to settle the question.
4. **Perplexity is not an acceptance test.** The 2026 "Broken on Arrival" audit found 2.3% of published 2–3 bpw tags
   emit fluent nonsense on code, and that a code-calibrated imatrix solved 6–13 more HumanEval+ tasks per level while
   PPL moved <4%. We already own the tool for the missing check: `tools/astvalid`. **`P2` is a functional screen**, and
   until it exists every lossless claim in this repo is a likelihood claim.
5. **The CPU wins that are measured are memory-side, not kernel-side.** On AVX2, a good ternary LUT kernel is ~2× a
   naive one but *nothing* over a good FMA kernel once decode is bandwidth-bound; the 6× figures are vs llama.cpp FP16
   on the same bytes. What is measured: pinning the small always-hot tensors (+26%), explicit merged reads instead of
   page faults (+13–14%, up to +55% with a coalesced layout), thread count at physical cores. That is `P6`.

---

## 3. The plan

Each item: hypothesis → method → gate → cost → kill criterion. Ordered by (expected gain)/(cost).

### P0 — Protect the super experts by design (not by budget) — **next, ~2 h**

*Hypothesis:* the prune's losslessness at 112 is fragile: it holds because 112 > 27. A tighter budget, or a different
corpus that re-ranks mass, would drop an expert whose removal is catastrophic rather than proportional. **Evidence
that it is not hypothetical: the 64 plan already drops L1E68** — 210× its layer's median output, mass rank 92 of 128,
0.28% of that layer's mass — and the plan's summary reports the loss as a fraction of a percent, because in mass it is.

*Method:* `prune` gains `--protect`, a set of `(layer, expert)` forced into the keep set before the DP runs, and
`--protect-json` reading the `flagged` list from `tools/expert_merge/superex.py`. Default: protect the flagged set, and
always protect layers 0–3's top-1 by output (their per-expert output is 200–2,000× the layer median, so a single
unlucky token can decide the set).

*Gate:* the plan still reports its dropped routing mass; a re-run of the 96 gate with `--protect` must not be worse
than 96 without it, and the protect set costs ≤ 0.1% of expert bytes (3 of 6,144). Report both arms paired.

*Cost:* the code is small; the gate is one 1,785-token walk (~40 min with `forward_docs` batching).

*Kill:* if protecting changes nothing at 96, the guard is still correct — it stays, and the finding is recorded as "the
mass plan already had this right on this corpus, by luck of the ranking".

### P1 — Prune on saliency, not mass, and not uniformly across layers — **~1 day**

*Hypothesis:* two literature ablations say the money is in (a) *saliency* (gate × output norm) instead of raw routing
frequency (−3.7 to −8.7 points in PRIME/REAM when removed) and (b) *non-uniform per-layer ratios* (SPARE: 57.2 → 63.8
at 50%). Our current plan is uniform and mass-only.

*Method:* score each expert with `mass × max|output|` (we now compute the output half in `superex.profile`), and give
each layer a budget from its routing entropy: high-entropy layers keep fewer experts. Compare {uniform-mass (current),
uniform-saliency, entropy-mass, entropy-saliency} at an equal total byte budget.

*Gate:* the same 1,785-token paired NLL walk, plus `P2`'s functional screen, at 96 and 80 a layer. Report which arm is
lossless and at what bytes — not which arm has the lowest mean.

*Cost:* profiling is ~41 s for 4 layers, ~8 min for 48; four plans × two budgets = eight gates, batched.

*Kill:* if saliency does not beat mass at equal bytes on the paired walk, keep mass (it is cheaper and already explained).

### P2 — A functional acceptance screen beside the NLL gate — **~3 h**

*Hypothesis:* the gate can certify a model that cannot write a valid program. NLL over 1,785 tokens of this repo's
prose is a likelihood measurement, and the literature's failure mode (fluent nonsense at 2–3 bpw) is invisible to it.

*Method:* a small fixed suite — 20 prompts with a deterministic seed, greedy, `max_tokens` fixed — scored by
`tools/astvalid` (does the output parse; does it satisfy the declared facts) and by exact-match on 20 arithmetic items.
Store the outputs so a later arm is compared to the *same* generations, not a fresh sample.

*Gate:* every arm reports (NLL delta, AST-valid rate, exact-match). The tolerance is on the pair: an arm may be
lossless on NLL and fail the screen; that is a rejection, and it is the finding.

*Cost:* 40 generations × 200 tokens at ~25 tok/s ≈ 6 min an arm on the 30B, plus the screen.

*Kill:* if the screen cannot separate a known-bad arm (the 64-way merge, +1.08 nats) from the control, it is too weak
and needs harder items before it can be used as a gate.

### P3 — Merge, but only with the mitigations the literature says are mandatory — **~1 day, or kill**

*Hypothesis:* our 1.08-nat merge is a lower bound on what merging can do, because it (i) groups by routing co-firing
rather than saliency, (ii) allows a group to average two experts with no mode connectivity, (iii) is uniform across
layers including the late ones where subspace contraction is worst, and (iv) keeps super experts un-singletoned.

*Method:* re-run the 64-pair merge with: saliency-weighted assignment; **super experts forced to singletons** (they are
3 experts — the cost is 0.05% of bytes); non-uniform group counts per layer (fewer merges in late layers); and the
router re-fit of `P4` applied to the merged pack.

*Gate:* compare against the **prune at the same byte budget**, not only against the control. This is the real question:
at 8.9 GB, is a 64-way merge better than a 96-expert prune?

*Cost:* one merge build (35 min) + one 1,275-token three-arm walk (~1.5 h).

*Kill:* if prune-at-96 wins on both NLL and the functional screen, stop merging this model class and say so in
`docs/EXPERT-MERGE.md`. (On the external evidence, this is the likely outcome.)

### P4 — Closed-form router re-fit, to finish the routing attribution — **~2 h**

*Hypothesis:* the pooled 64-row router is better than the copied source router by 0.128 nats (unresolved), because it
is *consistent* with 64 stacks rather than because it is *optimal*. A router fitted to reproduce the teacher's routing
distribution on the merged stacks should beat both, and would tell us whether routing is a lever at all.

*Method:* no gradients, no GPU — ridge regression on the route log. For each layer, fit a 64-row matrix `W'` that
minimises `KL(teacher_top8 ‖ softmax(W' x_n))` over the 384 logged tokens, regularised towards the pooled rows. This is
the same estimator PRIME uses for router reconstruction. Write it into the pack as `kindi.router_map`-free 64 rows.

*Gate:* the fused router must beat both existing arms on the 1,275-token paired walk, with an interval excluding zero.
Anything less and the finding is "routing is not the lever; the average is".

*Cost:* a few minutes of linear algebra plus one 1,275-token walk (~1.5 h for three arms).

*Kill:* if the fitted router is within noise of the pooled one, record that the router replacement is *saturated* — the
arithmetic half cannot be recovered by routing at all, which is the strongest form of the current conclusion.

### P5 — Bytes per weight: 2.0 → 1.667 — **~2 days, needs a measurement first**

*Hypothesis:* our ternary pack stores 2 bits a weight (4 trits a byte). A TL2-style layout stores **3 trits in 5 bits =
1.667 bpw** (sign plane + 4-bit index plane, `vpshufb`-friendly), which is 17% fewer bytes — and in the bandwidth-bound
regime that is ~1.17× tokens/s with no accuracy change at all, because the weights are the same trits.

*Method:* first measure whether our GEMV is already at the bandwidth roofline (`tools/bitnet_gemm/stream_probe.py`
exists for this). If it is, the win is the 17% and nothing else; if it is not, fix the kernel first.

*Gate:* tokens/s on a fixed prompt, bytes/token, and a bit-exactness check that the decoded weights equal the current
pack's. This is the one place where "lossless" is literal: same trits, same scales, fewer bytes.

*Cost:* packing change in `kindi/quant.py` + `kcpu.c`; the benchmark is minutes. The selftest must show decoded
equality, not approximate equality.

*Kill:* if the engine is not bandwidth-bound at the thread counts we ship (measure with `stream_probe`), defer — a
smaller file that is compute-bound buys nothing.

### P6 — The memory side, where CPU wins actually are — **~1 day**

*Hypothesis:* three measured wins from the research transfer to us: (a) pinning the small always-hot tensors
(embeddings, norms, head, attention — everything that is not an expert slab) gains up to **+26%** by keeping them out of
eviction's way; (b) reading expert slabs with explicit merged reads at queue depth ~4 beats demand-faulting mapped pages
by **13–14%**, and up to 55% with a coalesced layout; (c) threads = physical cores, never SMT.

*Method:* measure what we do now first — `tools/bitnet_gemm/stream_probe.py` and the existing MoE reader's read sizes.
Then: `VirtualLock`/`mlock` the hot set only (never the experts), and a small read-ahead that issues one `pread` a
whole expert (1.708 MB) instead of page faults.

*Gate:* tokens/s on a fixed prompt and major-fault count, plus peak RSS (the literature's failure mode is double-caching:
a file mapping *plus* an anonymous buffer for the same bytes, which has produced OOM at 60 GB).

*Cost:* measurement is minutes; the change is engine-side and needs a correctness re-run of the token-identity test.

*Kill:* do not touch hugepages (measured ~0% twice) and do not add `MADV_WILLNEED` on the mapping (measured 0% or
negative). If our expert slabs are ≥ 4 MB after P5 packing, coalescing will not pay — check the size before building it.

### P7 — Exact speculative decoding from an n-gram cache — **~2 days**

*Hypothesis:* the only technique in the research with a large, *quality-free* speedup: draft from an n-gram cache over
the prompt and the conversation, verify in one batched forward, accept only exact matches — so output is
token-for-token identical to the current engine's. Reported 1.0–1.5× on copy-heavy work (code edits, doc QA, summaries,
restated reasoning) and ~1.0× or slightly negative on open chat.

*Method:* the `ngram-mod` shape (a rolling-hash pool, ~16 MB, constant memory, shared across server slots) is the one
worth copying. Draft length from the longest suffix match; verify with the existing batched forward.

*Gate:* acceptance rate and tokens/s per request class, with **token-identity asserted** against the non-speculative
path. Ship it behind a flag whose default is measured, not assumed.

*Cost:* engine work; the gate is a replay of recorded prompts, which we already have in the prompt-cache tests.

*Kill:* if acceptance is < 30% on our own traffic, keep the flag off by default and record the number.

### P8 — What we will not do (and the evidence)

- **Not** hugepages for weight buffers: ~0.1–0.6% measured on x86, "no noticeable difference" in llama.cpp's own issue.
- **Not** `madvise(WILLNEED)` as a page-cache warmer: measured 0% in two independent harnesses, slightly negative in one.
- **Not** KV eviction as a latency play: measured 1.34× decode at TP=1 but prefill **0.58×**, and 0.69× at TP=2; at our
  context sizes the weight stream dominates anyway. Use KV quantisation (Q8_0 is free in quality) for *capacity* only.
- **Not** retrieval or graph front-ends as a reliability upgrade for a small model: with **oracle** retrieval, models
  ≤7B failed 85–100% of questions they could not answer alone, and retrieval destroyed 42–100% of previously correct
  answers. It is a prompt-budget cost, not a capability.
- **Not** a LUT kernel rewrite for speed: on AVX2 the honest measured gain is ~2× over a *naive* kernel and ~1.19–1.33×
  over a good one; the 6× figures are against FP16 baselines on the same bytes.

---

## 4. Order, and why

1. **P0** — it is a correctness guard on something already shipped, and it is cheap.
2. **P2** — every later gate is weaker without it; it changes what "lossless" means in this repo.
3. **P1** — the compressor that the evidence says works; needs P0 and P2 to be judged.
4. **P5** — the only accuracy-free speedup available on the compute side; measure the roofline first.
5. **P6** — the biggest measured CPU win in the literature, once we know our read sizes.
6. **P4 / P3** — settle the merge question rather than leaving "we could re-fit the router" open forever.
7. **P7** — real but workload-dependent, and the largest engineering effort for the smallest guaranteed gain.

## 5. Rules that keep this honest

- A gate is a **paired walk** over the same tokens, with a 95% interval from a block bootstrap. A sign is not a result.
- Never compare deltas across corpora. Every table states its token count; 94-token and 1,275-token deltas are not
  comparable, and the same merge reads +1.40 on one and +1.08 on the other.
- Never accept on perplexity alone once P2 exists.
- Every claim in `docs/` names the arm that produced it and the interval. If an interval covers zero, the text says so.
- Machine-local artifacts (packs, route logs, JSON) live under `B:/models/scratch` and are **not** in git; the JSON
  needed to reproduce a table is quoted in the doc, and the command is in its Reproduce block.

## 6. Status board

| item | status | next action |
|---|---|---|
| P0 protect super experts | **open** | add `--protect`/`--protect-json` to `prune`, re-gate 96 |
| P1 saliency + non-uniform budgets | open | needs P0 and P2 |
| P2 functional screen | open | wrap `astvalid` around 40 fixed prompts |
| P3 mitigated merge | open, likely kill | compare against prune-at-96, not just the control |
| P4 ridge router re-fit | open | fit on the 384-token log, gate at 1,275 tokens |
| P5 1.667 bpw packing | open | measure the roofline with `stream_probe` first |
| P6 pin hot tensors / merged reads | open | measure current read size and fault count |
| P7 n-gram speculative decode | open | flag-off prototype, acceptance measured |
| super-expert audit (done) | **measured** | 4 outliers at 210–2,152× their layer; mass ranks 27–92; the 64 plan drops L1E68 |
| merge cost split (done) | **measured** | router ≤0.17, arithmetic ≥1.08 |
| prune 112 (done) | **measured** | +0.0045, certified lossless at 1,785 tokens |
| prune 96 (done) | measured, **not established** | +0.0164, CI covers… is inside tolerance on the estimate only |

Update this file when an item moves. An item that is measured gets its numbers into `docs/`, not only here.
