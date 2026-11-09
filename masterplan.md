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
| Their fate in our plans | 112 and 96 keep all four; **64 drops L1E68** (mass rank 92, 0.28% of its layer's mass) and 32 drops L1E68 and L2E92. 96 survives only because 96 > 92; the 64 arm's +0.0965 is **not** caused by dropping E68 — keeping it is worth 0.0049 (P0, below) | same |

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
   page faults (+13–14%, up to +55% with a coalesced layout), thread count at physical cores. That is `P6`. The one
   exception, added 2026-11-05: **1.80× [1.71, 1.91]** from tiling the LUT GEMV at expert width, which is a layout and
   occupancy win inside the compute path rather than arithmetic (`docs/TERNARY-LUT-AND-PAGING-2026.md`) — so "memory-side,
   not kernel-side" is right about *what the win is*, and wrong to imply the kernel has none. That figure is a
   re-measurement: 1.58×, 1.37× and the first 1.55× were all taken with SMT siblings counted as cores (the Windows
   topology parse read two WORDs where the API has two DWORDs and fell back to per-logical masks). Under `0x5555` the
   win grows to 1.80× and the narrow-width tiling win disappears.
   **The LUT is still the wrong decoder on this host**, measured in one process: the shipped pair table (`rows8`) is
   1.7-2.2× the LUT at 2048×1920 and **2.8-2.9×** at 14336×4096, and the best tiled LUT arm (2.21 ms) is 1.22× slower
   than the untiled pair table (1.81 ms). `tools/bitnet_gemm/libcompare.py`; `P5` is unaffected (it is a byte
   cut), and no repack is justified by speed. Thread scaling is **unmeasured** here — the sweep came back non-monotone
   at 47% background load, which pairing cannot rescue.

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

*Result (2026-11-05, 1,785 held-out tokens):* the guard costs **6,832,128 bytes, 0.07% of the expert weights**, and at
96 the guarded plan is **byte-identical** to the unguarded one — all four outliers survive at 96 because 96 > 92. The
re-gate is therefore a reproducibility check: **+0.0173 [−0.0012, +0.0361]**, the same `marginal` verdict as the
+0.0164 already recorded. At 64 the guard swaps **one expert in one layer** (L1E68 in, L1E73 out) and the plan still
fails: **+0.0914 [+0.0445, +0.1401]** against the unguarded arm's **+0.0963 [+0.0505, +0.1443]** on the same 1,785
tokens (both baselines 3.6670, bit for bit). So the protection is necessary, cheap and insufficient: E68 is **0.0049
of a 0.0963** loss, and the other 95% is ordinary expert capacity. That also retracts the earlier reading that the 64
arm's cost was *caused* by the flagged expert. The unguarded arm also reproduces the certified +0.0965 to 0.0002, so
the difference is the guard and not drift. `--protect-json` also had to be built to *re-derive* the set from a `superex` profile: the
stored artifact carries the global 10%-of-peak cut (31 experts) under `flagged`, and reading it would have guarded 31
slots instead of 4.

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

*Result (2026-11-08):* **keep uniform mass.** On **1,778 held-out tokens** (baseline 3.6704; this corpus is not the
earlier 1,785-token corpus), saliency was nominally lower than mass by 0.0013 nats at 96 (**+0.0160 [−0.0090,
+0.0418]** vs mass **+0.0173 [−0.0012, +0.0361]**) and by 0.0111 nats at 80 (**+0.0413 [+0.0102, +0.0721]** vs mass
**+0.0524 [+0.0178, +0.0870]**), but no direct saliency-minus-mass per-token interval was stored, so neither is an
established advantage; saliency also drops more routing mass and has wider intervals. Entropy was rejected at both
budgets: **+0.0622 [+0.0238, +0.1054]** at 96 and **+0.1054 [+0.0555, +0.1596]** at 80, and the combined arm was
similarly rejected. At 80 all four arms fail 0.02. P2's screen was not run because no pruned kindi packs were
converted; see [`docs/EXPERT-PRUNE.md`](docs/EXPERT-PRUNE.md).

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

*Result (2026-11-08):* **deferred.** Fresh Bonsai-2B decode profiles gave 20.0–21.1 tokens/s, with GEMV at 80–82% of
the token and about 12–13 GB/s over the 499,400,704-byte streamed pair-code/scale footprint. That remains below the
documented 26.0 GB/s DRAM streaming ceiling, and the established working-set sweep is flat from 8 MB to 384 MB, so
the shipped kernel is issue-bound rather than bandwidth-bound. `stream_probe` measures disk reads (C: 1.63 GB/s in the
fresh run), not the resident DRAM stream. A 1.667 bpw repack would reduce bytes but add decode instructions; on this
evidence it is a storage change, not a decode speedup. See `docs/CPU-PERF-2026.md`.

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

*Measured first (2026-11-05), and it changes the order of this item:* the packs are on **B:**, which is a 4 TB SATA HDD —
**0.23 GB/s** cold, **100** random 4 KB IOPS — so a 14.7 MB expert read is **64 ms** there, not the 3.3-4.0 ms the
research's C: number gave. The NVMe at **C:** does 1.65-3.33 GB/s. Until the packs live on C: or in RAM, no pipeline,
merged read or pinning hides 64 ms, and the double-buffer ceiling is 2.7% rather than 33%
(`docs/TERNARY-LUT-AND-PAGING-2026.md`). Those two percentages carry a **1.8 ms** compute floor that the pinned harness
has since measured at a **3.3 ms** median, which would move them to 4.9% and 47% and the break-even to ~1 token; the
verdict does not turn on it (the read is 64 ms either way), but re-run the budget before committing anything to paging.
Placement is the first experiment; the rest of this item is after it.

*Result (2026-11-08):* **measured; engine unchanged.** Resident Bonsai-2B decode (64-token prompt, 40 tokens, 3 runs,
pack on C: NVMe): **0 disk read operations and 0–1 page faults per 40-token run**, 528.8 MB streamed from RAM per
token, peak working set 1.17 GB at load and 667 MB steady. Total faults at zero bound major faults at zero, so there
is nothing for `VirtualLock` or read-ahead to improve — the literature's +26% assumes file-backed weights under
eviction pressure, and this engine's kept sets are anonymous RAM. Paged MoE (synthetic fixture, 12 experts/2 active,
cache 2): 208,896 bytes an expert raw, 221,184 staged, 6 slices per miss, **expert matrices not contiguous** in
`bcodes.bin`; pcodes staging costs 302 faults/token against 15,711 for conversion staging, tokens identical across
resident/paged/convert arms. One pread per expert needs a convert-side contiguous layout — deferred until a real
MoE pack exists to validate it (Qwen arms are GGUF-only on this host). Threads 8 vs 16 on the i9-9980HK:
10.44 vs 10.36 tok/s, bit-identical tokens, single run each — no measurable difference, default unchanged.

*Follow-up (2026-11-08, P6b):* **contiguous layout implemented.** `packio.tensor_write_order` writes experts as
 gate → up → down in `bcodes.bin`, `dpm.bin` and `pcodes.bin`; a positional whole-expert read is used when the
 three staged matrices are adjacent (Windows `ReadFile` overlapped handle / POSIX `pread`). Repository gate:
 **106 ok, 1 skipped, 0 failed**. A real-weight one-layer fixture was built from layer 0 of the local
 Qwen3-Coder 30B GGUF with `tools/qwen_layer0_pack.py` (396 tensors, 384 pair-coded matrices, 317 MB, scratch-only).
 Cache 8, 40 tokens: resident **419.96 tok/s, 0 faults/token**; paged **93.52 tok/s, 1,774.7 faults/token,
 2.5 misses/token, ~3.32 MB/token**, tokens bit-identical. The fixture was re-ternarised from dequantised Q4 weights
 with `absmean`, so it is **not lossless vs the source GGUF** and is **not a 48-layer model**; the identity claim is
 only resident-vs-paged on the same pack. `tools/paged_cost.py` splits the per-miss cost further: read 1.537 ms,
 slot memcpy 0.123 ms, callback 0.0007 ms — **reads dominate**, not the callback (which also had a threading bug:
 it fired from every OpenMP thread, 1,280 calls per 10 tokens; fixed to `#pragma omp single`, tokens identical,
 paged 114.56 tok/s). `io_uring`/`O_DIRECT` and `MADV_WILLNEED` were not added (wrong OS / already measured
 worse); a Windows overlapped-I/O thread would only hide a cost the reads themselves are, so it stays unbuilt. `kindi moe-check` was failing at HEAD with the same numbers before and after the change
 (worst rel 6.03e-04, tolerance 1e-4); root-caused to the reference comparing against the pre-table floats rather
 than the pack's effective weights, and fixed by re-reading the reference from bcodes/dpm (worst rel now 5.82e-07,
 tolerance unchanged — see the status board). The Schackay3 Qwen3-Next artifact
 is a Q2_K (2.94 bpw) PT2-LLM asymmetric-group-128 conversion with PPL 5.75 → 8.69; its lessons (domain-matched
 calibration, protect router/trunk) transfer, but the artifact is `qwen3next` + K-quant and cannot be imported as
 a kindi ternary source today. See `docs/TERNARY-LUT-AND-PAGING-2026.md` ("P6 follow-up").

*The ternary-export research verdict (2026-11-08, PTERA):* the pasted proposal's four pillars are built on
misreadings, and one was measured and killed here. **(1) AWP:** the actual paper (arXiv 2506.10205, MERL) is
projected gradient descent with the activation covariance C as a *preconditioner* — it exists precisely to avoid
Hessian inverses; the `(X^T X + λI)^{-1}` the proposal insists on is GPTQ/OBS, a different method, and AWP
targets sparsity/INT4, not ternary. **(2) Ridge reconstruction measured:** `tools/awp_ridge/probe.py` hooks the
true per-linear inputs straight out of the C engine's scratch (not layer outputs), forms C = X^T X + λI with
relative damping, does the OBS-style projection (pruned coordinates to zero, survivors move by H^{-1}), then
ternarises. On three real Qwen3-Coder gate matrices (256 true inputs each): naive absmean output error
0.306–0.313; Wanda-flat-then-ternarise identical (the ternariser already zeroes those weights); **ridge-project
then ternarise 0.557–0.583 — worse on every expert**. Compensation moves survivors off the ternary grid, and
re-ternarising costs more than the projection buys. **(3) QAD/BitDistill:** the real paper (arXiv 2510.13998,
Microsoft) is a three-stage *training* campaign (SubLN insertion, continual pre-training, logits+attention
KD) validated on Qwen/Gemma ≤4B for *task* parity after distillation — it is not a lossless converter, and
it needs GPU training this branch does not have; deferred with that kill criterion. **(4) SLQ:** IST-DASLab's
own numbers put the task-lossless floor at **3.3 bits** and distribution-lossless at 5–6 bits — no published
method claims lossless 1.58-bit PTQ, and DeltaPQ is a similarity-search index codec, not a weight packing.
What transfers to this branch instead is what it already ships: calibration-aware scale search (lsq/gsq), and
mixed-precision budgeting (`kindi.rco`).

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

### P9 — Verifier-gated selection: the only symbolic layer with end-task numbers — **~3 days**

*Hypothesis:* the request behind this branch's §G research was a symbolic/LUT/action-graph backbone that gets *better
answers than the model*. The literature does not support a backbone, but it does support one narrow, CPU-cheap thing:
**sample k, run a cheap deterministic verifier, keep a verified answer**. That is what the measured gains are —
self-consistency (PaLM-540B GSM8K 56.5 → 74.4%), program-of-thought with an executor, CodeT-style test filtering —
and every one of them is a *selection* over a fixed model's samples, not a new representation of knowledge.

*Method:* `tools/astvalid` already parses and checks; P2's 40 fixed prompts are the harness. Add a `--select k` mode
that draws k samples at temperature, scores each with the verifier (compiles, passes the AST/type check, matches the
unit test), and returns the best verified one; ties broken by likelihood. No graph, no retrieval index, no extra model.

*Gate:* pass@1 on the 40 prompts, paired by prompt against a single greedy sample; report the verifier's own
false-accept rate and the k× cost per accepted answer. A gain that needs k=16 to appear is a latency decision, not a
capability claim, and the text must say which.

*Cost:* k× prefill, which on a bandwidth-bound CPU is k× wall clock — this is why the flag is off by default and the
gate reports cost per accepted answer rather than per sample.

*Kill:* if the verifier accepts wrong answers at a rate that erases the selection gain (measure it on the prompts where
the model is wrong), or if a single greedy sample already passes, the item is closed as "no headroom on this model".

### P10 — Group-aligned rotation before rounding — **measured, not worth building yet**

*Hypothesis:* an orthogonal transform before ternary rounding reduces what the pack loses, as QuaRot/SpinQuant/DuQuant
and the group-rotation line (GSR: 20.29 → 11.59 WikiText-2 PPL at W2) report. The catch: every published win is inside a
different quantiser (GPTQ error feedback, per-channel scales, learned rotations), and the block size in those papers is
not our block size.

*Measured* (`tools/rotq/probe.py`, six real matrices from `qwen3moe-control-q4.gguf`, both scale schemes, 26 selftest
checks):

| arm | mean vs `none` | range over matrices | adds/token |
|---|---|---|---|
| `group-128` (rotate inside the scale group) | **−2.6%** error | −16.8% .. +2.1% | 0.09–0.72% of the GEMV |
| `hadamard` (global) | −3.4% | −16.4% .. +1.5% | 0.13–0.72% |
| `orthogonal` (dense QR) | −2.5% | −16.5% .. +2.1% | 25–133% |
| `zigzag-128` (DuQuant's channel interleave) | −2.3% | −14.8% .. +1.5% | 0.10–0.52% |
| `signed-perm` | **+1.6% worse** | −0.0% .. +9.5% | 0 |
| `group-128-sequency` | identical to `group-128` | — | 0.09–0.72% |

The mean is carried by one matrix: `blk.0.attn_q` is at 0.6117 relative error without a rotation and 0.5089 with one
(−16.8%), because layer 0 is where an outlier exceeds its group's scale and gets clipped to ±1. Every expert matrix
(`ffn_gate`/`up`/`down`) comes out **1.5–2.1% worse**, and `blk.47.attn_q` is level. The one thing that improves
everywhere is the worst single group: 0.9018 → 0.6260, 0.8573 → 0.6492, 0.8215 → 0.6315.

Two structural results the probe pinned, which matter more than the table:

- **Ordering inside a group is exactly inert.** Sequency-sorted rows are a permutation *within* each scale group, so the
  group keeps its multiset and its scale: the two arms agree to the last digit. GSR's sequency gain therefore cannot
  come from the ordering at a group-local quantiser.
- **Whole-group permutation is inert too, for the same reason** (it relabels groups, it does not move a channel between
  them). The probe caught this as a dead arm before it became a dead end; the arm that *can* move an outlier is the
  zigzag, which interleaves channels across groups — and it is slightly worse than plain rotation here.

*Verdict:* **do not build this into `convert` yet.** A 2% weight-error change on experts is inside the noise of
anything we could measure, and the arithmetic is only free if the rotation is folded into the weights — which means
rotating *before* ternarising, the order BitNet v2 measured as worse at 1.58 bits (20.83 vs 11.33 PPL).

*Next, if it is ever picked up:* the cheap version is a **per-matrix flag** for layers whose outlier is provably
clipped — measured by this probe, cheaply, at conversion time — followed by a paired NLL walk over early attention only.
The gate is that walk's interval, not this table: a weight error is a proxy, and this file does not get to promote a
proxy to a result.

### P8 — What we will not do (and the evidence)

- **Not** hugepages for weight buffers: ~0.1–0.6% measured on x86, "no noticeable difference" in llama.cpp's own issue.
- **Not** `madvise(WILLNEED)` as a page-cache warmer: measured 0% in two independent harnesses, slightly negative in one.
- **Not** KV eviction as a latency play: measured 1.34× decode at TP=1 but prefill **0.58×**, and 0.69× at TP=2; at our
  context sizes the weight stream dominates anyway. Use KV quantisation (Q8_0 is free in quality) for *capacity* only.
- **Not** CAT-Q's mean-shifted ternary grid, and **not** SCALE-uG residuals: both need a per-group *shift* term
  or an extra scale the kindi pack format does not have (one fp16 scale per 128-group, no shift). Measured
  (`tools/awp_ridge/catq_probe.py`, true gate inputs from the C engine, FP32 GGUF weights, 256 inputs × 3
  experts): absmean 0.51–0.60, lsq 0.43–0.49, gsq 0.43–0.49, threshold scan 0.51–0.60, numpy-scale CAT-Q
  LM+ST replication **1.08–1.15 — worse than every arm it was measured against**. lsq and gsq already ship and
  tie each other ~15–18% below absmean; threshold is not the lever (scale is, and we already search it).
  What *does* transfer from CAT-Q / ScaleQ-1.58 (ICML 2026 oral, arXiv 2606.26650 / 2608.01078) is the
  calibration-domain finding — quantising against web text collapses reasoning tasks, quantising against the
  model's own CoT traces does not — which is why P2's functional screen, not PPL, is the gate.
- **Not** a native BitNet b1.58 from-scratch "Mini-Minion": measured locally, this CPU does ~195 tok/s on a
  0.5B fp32 forward+backward (no attention/optimizer), so the 4T-token recipe behind Microsoft's 2B native
  ternary model is ~650 years here and even a 9.9B-token hobbyist budget is ~1.6 years; the published
  evidence (BitNet b1.58, JMLR 2025; "Reloaded", arXiv 2407.09527) also shows parity with fp16 only from
  3B at equal tokens, and HumanEval is an eval set (training on it is contamination). Native ternary is a
  data/compute campaign, not a converter feature; its output already exists as a downloadable GGUF.
  A minimal research harness exists at `tools/bitnet_qat/` (`MiniMinion`, BitLinear + STE, SwiGLU,
  RMSNorm-before-BitLinear, per-group weight-decay AdamW) with a smoke test: 58.5M params, 483 tok/s on
  CPU, loss 10.49 → 8.84 over 20 steps. Token-budget extrapolation at that rate: 1.35B tokens ≈ 32 days,
  9.9B ≈ 0.65 years, 4T ≈ 263 years — a framework for small GPU experiments (50M class), not a plan for
  a competitive ternary 0.5B on CPU.  A GPU-enabled node (the Arc B-series box `sol`, verified reachable
  key-only 2026-11-08) changes the constants but not the scaling conclusion below 3B. HumanEval remains
  eval-only in that harness.
  **Cluster inventory (verified 2026-11-08, key-only SSH):** `sol` — 8 cores, 125 GB RAM, **Arc Pro B50**
  (Battlemage workstation class, 20 GB) + Arrow Lake-U iGPU, NPU accel node, OpenVINO 2026.4.1 enumerating
  `['CPU', 'GPU.0', 'GPU.1', 'NPU']`, Level Zero + `xe` driver present — so native torch XPU (no IPEX)
  can target the dGPU. `nyx` — 16 cores, 58 GB, Arrow Lake-P iGPU, torch 2.14 CPU-only. Torch-XPU install
  on sol was in progress at the close of this session (pytorch.org CDN throttled to ~40 KB/s from this
  network; no mirror carries the XPU wheel) — the 58M smoke test on the B50 is the next command and it
  replaces every projection in this file with a measured tok/s.
- **Not** retrieval or graph front-ends as a reliability upgrade for a small model: with **oracle** retrieval, models
  ≤7B failed 85–100% of questions they could not answer alone, and retrieval destroyed 42–100% of previously correct
  answers. It is a prompt-budget cost, not a capability.
- **Not** a LUT kernel rewrite for speed: on AVX2 the honest measured gain is ~2× over a *naive* kernel and ~1.19–1.33×
  over a good one; the 6× figures are against FP16 baselines on the same bytes.
- **Not** a symbolic/LUT "backbone" under the transformer, and **not** a compressed symbolic action graph as the
  answer engine. The proposal is old and respectable — ternary LUT kernels, Tsetlin machines, weightless nets, tries
  and n-gram caches — and every one of them is *cheaper*, none of them is *more accurate* at language-model scale. The
  measured end-task gains in the neuro-symbolic literature come from a **verifier selecting among the model's own
  samples** (P9), not from a symbolic representation that reasons; and the retrieval half is already rejected above by
  oracle experiments. What we keep from the proposal is the part with numbers: P5's packing, P6's memory work, P7's
  exact n-gram draft, and P9's verifier.
- **Not** a knowledge-graph or ontology layer for answer quality: with the graph *given*, small models still failed the
  questions they could not answer alone, and the same evidence shows retrieval destroying answers they had right.
  "Better answers, not just cheaper" is a claim that needs a paired pass@1 walk, and nothing in §G reports one for a
  graph without a verifier.

---

## 4. Order, and why

1. **P0** — it is a correctness guard on something already shipped, and it is cheap.
2. **P2** — every later gate is weaker without it; it changes what "lossless" means in this repo.
3. **P1** — the compressor that the evidence says works; needs P0 and P2 to be judged.
4. **P5** — the only accuracy-free speedup available on the compute side; measure the roofline first.
5. **P6** — the biggest measured CPU win in the literature, once we know our read sizes.
6. **P4 / P3** — settle the merge question rather than leaving "we could re-fit the router" open forever.
7. **P7** — real but workload-dependent, and the largest engineering effort for the smallest guaranteed gain.
8. **P9** — after P2, because its verifier *is* P2's checker; it is the only "better answers" route with numbers.

**P10 is measured and parked.** It is not in the order because the measurement says the effect is a proxy-sized 2%,
not the literature's 8-point perplexity: it moves between P5 and P6 only if an early-attention NLL walk shows a real
interval, and the probe that prices it is one command (`tools/rotq/probe.py`).

## 5. Rules that keep this honest

- A gate is a **paired walk** over the same tokens, with a 95% interval from a block bootstrap. A sign is not a result.
- Never compare deltas across corpora. Every table states its token count; 94-token and 1,275-token deltas are not
  comparable, and the same merge reads +1.40 on one and +1.08 on the other.
- Never accept on perplexity alone once P2 exists. (P2 built 2026-11-07 as `tools/funcscreen`, kill property measured on bonsai-2b generations; the control-vs-merge64 separation run is still owed, so P2 can reject a *corrupted* arm today and a *merged* arm once its packs are converted.)
- Every claim in `docs/` names the arm that produced it and the interval. If an interval covers zero, the text says so.
- Machine-local artifacts (packs, route logs, JSON) live under `B:/models/scratch` and are **not** in git; the JSON
  needed to reproduce a table is quoted in the doc, and the command is in its Reproduce block.

## 6. Status board

| item | status | next action |
|---|---|---|
| P0 protect super experts | **measured** | `--protect`/`--protect-json` in `prune`; guard 0.07% of bytes; 96 identical to unguarded (**+0.0173 [−0.0012, +0.0361]**, marginal); 64 guarded **+0.0914** vs unguarded **+0.0963** on the same tokens — the outlier is 5% of that loss, not its cause |
| P1 saliency + non-uniform budgets | **measured; keep uniform mass** | saliency only nominally ahead with overlapping baseline intervals; entropy rejected at 96 and 80 on 1,778 held-out tokens. P2 not run: no pruned kindi packs converted |
| P2 functional screen | **built, kill property measured** | `tools/funcscreen`: 40 fixed items (20 code + 20 arithmetic), greedy, outputs stored; bonsai-2b 32/40 vs token-shuffled 6/40, paired delta −0.65 [−0.80, −0.50]. **Control-vs-merge64 kill test not run** — arms are GGUF-only on this host; convert both packs and `--compare` |
| P3 mitigated merge | open, likely kill | compare against prune-at-96, not just the control |
| P4 ridge router re-fit | open | fit on the 384-token log, gate at 1,275 tokens |
| P5 1.667 bpw packing | **measured; deferred** | GEMV is issue-bound at ~12–13 GB/s vs a 26.0 GB/s DRAM ceiling; `stream_probe` measures disk, not DRAM. Repacking adds instructions and is not a decode speedup on this evidence |
| P6 pin hot tensors / merged reads | **measured** | resident decode is 0 disk reads and 0–1 faults per 40 tokens; threads 8 vs 16 tie at 10.44/10.36; pinning has nothing to fix |
| P6b contiguous experts + one-read paged staging | **implemented; callback-bound** | gate→up→down adjacency in bcodes/dpm/pcodes, whole-expert positional read when contiguous; real layer-0 Qwen3-Coder fixture: resident 419.96 tok/s, paged 93.52 tok/s @ 2.5 misses/token, tokens bit-identical; no io_uring/MADV_WILLNEED |
| P6c cost split (callback/read/memcpy) | **measured: reads dominate** | `tools/paged_cost.py` on the real fixture, 40 tokens: 2.5 misses/token × 1.66 ms read+memcpy per miss; read 1.537 ms, memcpy 0.123 ms, callback 0.0007 ms; callback path also had a threading bug (fetch fired per-thread, 1,280 calls for 10 tokens) — fixed to `#pragma omp single`; tokens bit-identical, paged 114.56 tok/s after. **No overlapped-I/O thread is built**: the split says the read *is* the cost, so hiding it needs a kernel-side iteration register/prefetch, which is not justified while paging is off by default |
| moe-check failure root-caused | **check fixed, not relaxed** | worst rel 6.03e-04 was **scale-table error in the reference, not the kernel**: `synthpack`'s returned `quantised` dict predates the 8-bit level-table re-encoding of scales, so 4 of 36 tensors differed from the pack's effective weights by ~6e-4 and exactly the trials routing those experts failed (scalar and AVX2 identical). `run()` now re-reads dequantised weights from bcodes/dpm; worst rel **5.82e-07** (the fp32 associativity floor) on the same probes, tolerance 1e-4 unchanged; `--paged` bench passes (byte-identical pcodes, all rows resident tokens); repo gate 107 ok, 0 failed |
| Ternary-export research (AWP/QAD/SLQ/DeltaPQ) | **measured; PTQ ridge rejected, QAD infeasible here** | `tools/awp_ridge/probe.py` on true gate inputs of the real fixture (256 tokens, 3 experts): naive absmean output error 0.306–0.313; Wanda-flat then ternarise identical 0.306–0.313 (the ternariser already zeroes those); **ridge-project then ternarise 0.557–0.583 — worse on every expert**. True AWP (arXiv 2506.10205) has no covariance inverse at all: PGD with C as preconditioner, and it targets sparsity/INT4, not ternary. QAD/BitDistill (2510.13998) is a GPU training campaign on ≤4B models with task-metric parity, not lossless conversion; not feasible for a 30B MoE on this CPU-only host. SLQ's own floor is 3.3 bits task-lossless. See appendix §F |
| P7 n-gram speculative decode | open | flag-off prototype, acceptance measured |
| P9 verifier-gated selection | open | after P2; `--select k` over the 40 prompts, verifier false-accept measured |
| P10 group rotation before rounding | **measured, parked** | −2.6% weight error mean, +2.1% on experts, sequency exactly inert; no NLL walk run |
| volume identity / read floor (done) | **measured** | packs sit on B: = 4 TB SATA **HDD** (0.23 GB/s, 100 IOPS, 64 ms an expert), not C: = NVMe (1.65-3.33 GB/s); the 1.66 GB/s under test is C:'s own rate |
| super-expert audit (done) | **measured** | 4 outliers at 210–2,152× their layer; mass ranks 27–92; the 64 plan drops L1E68 |
| merge cost split (done) | **measured** | router ≤0.17, arithmetic ≥1.08 |
| prune 112 (done) | **measured** | +0.0045, certified lossless at 1,785 tokens |
| prune 96 (done) | measured, **not established** | +0.0164, CI covers… is inside tolerance on the estimate only |

Update this file when an item moves. An item that is measured gets its numbers into `docs/`, not only here.

---

## Recovery note (2026-11-05, from Freebuff session cache)
Session `5b5aa0d9-f359-4bae-960d-4e41966d24c5` (plan, `deepseek/deepseek-v4-flash`, completed 2026-11-01 UTC) was recovered after a Freebuff desktop session-load failure. **Pending/unsent prompts:** none found in parseable local profile storage.

**Recovered objective:** branch to support Intel **CPU/GPU/NPU** (9th-gen i9; 2× i9 Ultra mini PCs with iGPU/NPU/CPU; 1× with Intel Arc B50). Find hardware docs/features; adapt kindi for accelerated model running (fastest per hardware profile); consider disk I/O/p2p/memory, improved inference; evaluate libraries including OpenVINO, llama.cpp (SYCL/Vulkan), oneAPI/Level Zero and bleeding-edge options. This extends beyond the current CPU/AVX2 focus in §1–§3 and should be planned as additional measured profiles once P6/P5/P7 CPU-path priorities are settled.

**Intel hardware support roadmap for kindi (user-requested, 2026-11-08 — plan item, not engine work).** Target profiles in order of expected value: (1) **CPU AVX2** — current engine, measured (P6/P6b/P6c). (2) **iGPU (Arc iGPU, both nodes)** — the free accelerator: same Level Zero stack as the dGPU, usable from SYCL/oneAPI or via the OpenVINO GPU plugin; worth a measured tok/s profile once torch-xpu lands on `sol` (GPU.0 vs GPU.1 vs NPU through the same harness). (3) **Arc Pro B50 (sol, GPU.1)** — the heavy GEMM target: torch native XPU on the 58M harness gives the first real GPU tok/s; that number sets every token budget below. (4) **NPU (both nodes, /dev/accel/accel0)** — OpenVINO NPU plugin for small static-shape workloads only; LLM decode on a ~13-TOPS NPU is not a plan pillar (verify, don't assume). (5) **OpenVINO GenAI stack** (`openvino-genai` + optimum-intel + NNCF) is a **deployment comparison arm**, not engine work: INT4 export → LLMPipeline on GPU.1 vs kindi ternary on CPU, measured. **Install unblocked 2026-11-09** (wheels pinned by URL; `pypi.org/simple` times out from sol, `files.pythonhosted.org` does not) and the arm's baseline is now measured on Qwen3-8B-int4-ov, greedy, 64 new tokens, decode excluding prefill, 3 runs: **CPU 9.78 tok/s · iGPU GPU.0 12.77 · dGPU GPU.1 (Arc Pro B50) 39.29**. That is the deployment reference rate; the kindi side needs a pack of the same model before the arm is a comparison. Detail and the exact reproduce command in `appendix.md`. IPEX is excluded per user instruction and because native torch XPU is the maintained path. **Cluster roles:** `sol` = GPU trainer + GenAI arm; `nyx` = CPU worker (prune gates, NLL walks, torch-CPU) — split, don't duplicate: route the CPU-bound walks to nyx, the GEMM-bound work to sol.

**Training/distillation queue on the cluster (user-requested).** The honest budgets at the *measured* CPU rate were: 58M harness ≈ 483 tok/s → 1.35B tokens ≈ 32 days, 4T ≈ 263 years. The measured B50 number replaces those projections: **3968.4 tok/s** on the 58M harness in `torch.xpu` (loss 10.486 → 9.263 in 20 steps) puts the 1.35B-token small-SLM budget at ~95 h of pure GPU time (4T tokens is still ~32 years, so a 4T pre-train remains out of reach); a 0.5B run at parity remains **not feasible** (published parity starts at 3B, and QAD/BitDistill remains a 4B-scale GPU campaign that the appendix already rejects: near, not equal, and rejected for ≤4B). HumanEval stays eval-only. **Prune experiments:** the pipeline is built and P0/P1 are measured (prune-112 certified lossless, 1,785 tokens). Queued, ready to run on nyx/sol CPU when given the model/pack: **P3** mitigated merge judged against prune-at-96; **P4** ridge router re-fit on the 384-token route log, gated at 1,275 tokens; **P2** control-vs-merge64 kill test (needs both packs converted). The user can supply further prune configs; the format is `tools/expert_merge/prune.py --routes <log> --keep N --protect-json <super-experts.json> --gate` with the inputs documented in `docs/EXPERT-PRUNE.md`.

**Model conversion scope (updated 2026-11-09):** the engine can build a runnable kindi weight pack today for the Qwen3-family files it already ships/accepts — the shipped Ternary-Bonsai-2 `qwen35` GGUFs (2B/4B/9B/27B via the accepted PROFILES tuple), and the retired dense **Qwen3** Bonsai-1.7B/Bonsai-8B via `QWEN3_PROFILES`. Anything else in the Qwen3 family (0.5B/1.5B/3B/4B/7B/14B, the 30B-A3B MoE, …) is convertible in principle through `kindi qat-pack` (float torch checkpoint → PTQ1_0 codes) once each size is added to the accepted-profile list and the tokenizer matches; the `qwen3moe` path needs the expert-pair-code/packer finished before an MoE GGUF becomes a runnable kindi pack, not a catalog. For non-Qwen3 Qwen models (Qwen2.5/Qwen2/Qwen1/QwQ/…) the header is detected as an unknown architecture today and refused — no FAMILIES entry, no tensor-name map, no tokenizer. Full detail and the exact missing pieces are in the new `.../MODEL-CONVERSION-SCOPE.md` section of this plan.

**Resume:** see `HANDOFF.md`. Code moved to `E:\viber_coding\freebuff\kindi-fork` (clean clone, HEAD `fee7bb0`, pushed to `origin`/`gitea` freebuff branch at recovery time).

### Cluster-free GPU training tiers, as of 2026-11-09

Use these when sol/nyx is not the chosen target, with checkpointing at or under the tier's wall-clock ceiling so a run can be resumed across sessions.

- **Google Colab CLI (google-colab-cli, PyPI `google-colab-cli`, repo `googlecolab/google-colab-cli`, latest release 0.7.5 / 2026-10-20; created 2026-04-17):** yes, there is an official Colab CLI and a Colab API; the "no API / hallucination" reading of older training data is wrong. Install: `uv tool install google-colab-cli` (or `pip install google-colab-cli`; binary lands at `~/.local/bin/colab`). Auth is the real gate: the CLI supports **two providers** via global `--auth={oauth2,adc}` (default is `oauth2`), and the Colab backends need a specific scope set, so the headless/agent-safe path is ADC via `gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory` — user credentials minted by `gcloud auth application-default login` without that `--scopes` list **ignore** the scopes kwarg and raise `NotImplementedError` on `with_scopes`, so they must be re-minted. The `oauth2` path works too but opens a browser once and is less suited to fully headless use. Runtime flow (from the CLI docs/demos and AUTH_AGENTS): `colab new -s <name> [--gpu ...]`, `colab install -s <name> ...`, run the script, `colab log -s <name>`, `colab stop -s <name>`; Drive mount is `colab drivemount -s <name> [PATH]` (default `/content/drive`), and `colab usage` shows compute-unit rate/balance. Idle timeouts still apply on the free tier (free-of-charge notebooks run at most ~12 h depending on availability/usage per the official FAQ), and the paid ladder (Pro/Pro+/Pay As You Go / Google AI plan) buys more compute units and longer runs (Pro+ can run up to ~24 h when units suffice). So: Colab is usable headlessly for checkpointed runs, but free-tier continuous-time and accelerator availability are not guaranteed and can fluctuate; treat 4-hour checkpointing as the unit of work regardless of the source.

- **Paperspace:** viable as another checkpointed GPU source (Gradient jobs / persistent notebooks / cloud GPUs), and a natural fit for the same 4-hour-or-less checkpoint discipline. No Paperspace CLI deep-dive was pulled for this turn; if we rely on it, treat it like Colab — provision, run, checkpoint, retrieve artifacts, stop — and prefer the same harness + artifact layout so a run can move between sol, nyx, Colab and Paperspace without rewriting the script.

- **Modal (modal.com):** the intended first checkpointing target per the user: prototype on Modal, then run the full thing on Colab, with Paperspace as an overflow tier. Treat Modal as the cheap/fast validation layer before the longer Colab/Paperspace runs; keep every run checkpointed so the 4-hour ceiling is a resume point, not a restart.

- **OPAL (github.com/xingjian-bai/opal, `xingjian-bai/opal`, MIT, created 2026-10-21, v0.3.10, ~1.5k lines, 415 stars / 25 forks, 0 open issues, Python, topics `deep-learning`, `llm-agents`, `optimizer`, `pytorch`):** a **PyTorch optimizer wrapper** that lets a coding agent (default Claude Code) **rewrite the optimizer program while training runs**, plus a second analysis program the agent reads to decide. Mechanics from the README: `opal.wrap(opt, model, scheduler=sched, total_steps=..., goal="...", backend=..., max_cost_usd=...)` once around a normal `torch.optim` optimizer; the agent edits `updater.py` (the gradient→update rule) and `analysis.py` (whatever it wants to measure) in `opal_runs/<folder>/workspace/`; decisions happen every `every` steps (default 50); `blocking=True` (default) makes training wait for the agent, `blocking=False` keeps training going; a new optimizer program is trial-run for 20 steps and reverted on error if a snapshot fits in `snapshot_limit_gb` (default 2); the agent sees the goal sentence, param names/shapes, training losses, its own analysis, and optionally logged values / validation via `opt.log(...)` / `opt.report_validation(...)`; HF Trainer works through `opal.hf.OpalTrainer`, DDP within one node works (NCCL, eager and `torch.compile`), Accelerate works if you wrap after `accelerator.prepare` (without fp16 GradScaler or Accelerate grad accumulation); **multi-node, FSDP, DeepSpeed are not supported**; the whole test suite runs on CPU with no agent and no cost (`pytest -q`); the CLI side is `opal doctor`, `opal estimate`, `opal show/watch/report/compare`. Measured results in the README are on vision/translation/small-LM tasks (SiT-B/2, RAEv2-B, Goldfish 125M, LightningDiT-B/1, modded-nanogpt) — **none on a 1.5B-scale native ternary language model**. Verdict for this branch: **OPAL is a training-tuner, not a training harness and not a model converter.** It does not give us a BitNet-style MiniMinion trainer, a conversion path for Qwen GGUF→kindi, or MoE/pruning logic. It slots in **only if and when we already have a PyTorch training loop for MiniMinion** (the `tools/bitnet_qat` harness is the nearest thing, but it is a smoke-scale harness, not a full trainer yet) and we want an agent to tune the optimizer/schedule/loss-spike guards during a long run. On our hardware story it is also gated by the Claude Code dependency (agent billed to a Claude subscription or API key, `max_cost_usd` caps it, and the README explicitly says Codex is an experimental backend) and by the multi-node limitation — the cluster split (sol GPU + nyx CPU) is exactly the setting OPAL says it does not support yet. So: useful as a future optimizer-tuning layer on top of a real MiniMinion trainer, **not** something to install now as part of the conversion/pipeline work, and not a substitute for the_sol/nyx cluster plan. If and when we stand up a full MiniMinion PyTorch trainer on sol, OPAL is a candidate for the optimizer-tuning layer; not before.

### Model conversion scope (updated 2026-11-09)

**Model families the engine can convert today** (a conversion path exists, so `kindi convert` or `kindi qat-pack` actually builds a runnable kindi weight pack):

- **`qwen35`** — Ternary-Bonsai-2 Qwen3.5 hybrid (gated DeltaNet + full attention). Currently the mainline conversion path: `kindi convert GGUF` reads the file's own dimensions and only imports a file whose `(layers, width, ffn, heads, kv_heads, ssm_groups)` tuple matches one of the four shipped Ternary-Bonsai-2 profiles (`2B/4B/9B/27B`), or a file within an accepted variant range (`--allow-variant`). Conversion in = PTQ1_0/Q1_0 ternary GGUF, and the pack repacks those exact codes into the kindi layout with no weight change.
- **`qwen3`** — dense Qwen3 stack (no MoE, no GDN, full attention each layer, per-head QK norm, `qwen3moe`-less metadata keys). The retired **Bonsai-1.7B** and **Bonsai-8B** are the two dense Qwen3 profiles the converter currently accepts. Same rule as `qwen35`: accepted on the profile tuple, converted only from a ternary/Q1_0 GGUF (currently `q1_0` for those files).
- **`qwen3moe`** — MoE variant of the Qwen3 stack (128 active experts per layer, stacked 3-D expert slabs, `ffn_gate_inp`/`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` naming). The converter declares it convertible, but as of this commit the **MoE expert path is not exercised in the pack writer** — `kindi convert` on a real MoE GGUF would need the expert-pair-code writer to land (or the dense-Qwen3 path extended to the MoE tensor names) before the MoE variant becomes a runnable pack on this branch.

**Qwen models the engine can already detect (head parse only, no conversion path yet):**

- `qwen35`, `qwen3`, `qwen3moe` all appear in `kindi.recipes.FAMILIES`, so the header scan names them and reports what it would do — or refuse — for each.
- **Qwen2.5 / Qwen2 / Qwen1 / QwQ / Qwen3-0.5B etc.:** not represented in `FAMILIES` as a conversion path. A file with `general.architecture` set to `qwen2_5`, `qwen2`, `qwen1`, or any non-`qwen3*` string is detected as **unknown family** and reported as no conversion path today; the tokenizer (a Qwen3.5 byte-level BPE clone) would also not match their tokenizers.

**What a Qwen conversion actually needs to land for the full Qwen3 family to be coverable from this engine:**

1. **A float path for Qwen3-series checkpoints.** `kindi qat-pack` takes a `torch` state dict and writes the PTQ1_0 codes itself, so a full Qwen3 fp16/safetensors checkpoint becomes a kindi pack for any variant the writer covers. The model-specific part is the stat/name/role mapping: Qwen3 uses the `attn_q/attn_k/attn_v/attn_output` + `ffn_gate/ffn_up/ffn_down` naming plus per-head QK norms; Qwen3.5 (`qwen35`) folds the per-head gate into `attn_q` and carries the Prism/Prism-v2 Hadamard metadata; Qwen3-MoE adds the expert slabs + router. That mapping is the real work per sub-family.
2. **MoE expert writer + router handling for `qwen3moe`.** The MoE tensor names and the stacked 3-D slab layout exist in the codebase (the merge/PR tools already read them from GGUFs), so the missing piece for a full `qwen3moe` conversion is finishing the pack writer for those tensors + the pair-code/paging path, not discovering what the tensors are.
3. **Tokenizer parity for non-Qwen3.5 Qwen models.** The Qwen2.5/Qwen2 families use a different pre-tokenizer than Qwen3.5; a conversion of those would need its own tokenizer clone (or a llama.cpp-backed tokenizer) before the pack could be served correctly.
4. **Profile list / variant allowance.** Each Qwen3 sub-size (0.5B, 1.5B, 3B, 4B, 7B, 14B, 30B-A3B MoE, etc.) is a new accepted `(layers, width, ffn, heads, kv_heads, [ssm_groups])` tuple in `kindi.convert.QWEN3_PROFILES` / `PROFILES` / the MoE variant logic. Today only the shipped Bonsai profiles + the one tested `qwen35` variant are in.

Concretely: **can we convert Qwen models with the existing pipeline?**

- **Yes, for the exact shipped Bonsai ternary GGUFs** — those are already in `convert.py`'s accepted profiles and the packs build from them.
- **Yes, for any other Qwen3-family checkpoint in principle** — the `qat-pack` float-to-ternary path is the mechanism, and the Qwen3/Qwen3.5 tensor naming is the kind of thing the converter already knows. What is not done is (a) adding each new Qwen3 size to the accepted-profile list, (b) finishing the `qwen3moe` expert pack writer so MoE models become runnable packs, and (c) adding a Qwen2.5-family tokenizer so non-Qwen3.5 Qwen models can be served. Those are the conversion-roadmap items, not a blockage on the pipeline existing.
- **No, for the non-Qwen3 Qwen families today** — no FAMILIES entry, no tensor-name map, no tokenizer; the header would be recognised as an unknown architecture and refused, as designed.

_From `kindi/convert.py`, `kindi/recipes.py`, `kindi/chat.py`, `kindi/tokenizer.py`, `kindi/__main__.py`, `kindi/catalog.py`, `kindi/ptload.py`, `kindi/drift.py`, and `docs/RECIPES.md`._

### Cluster-free GPU training tiers, as of 2026-11-09

Use these when sol/nyx is not the chosen target, with checkpointing at or under the tier's wall-clock ceiling so a run can be resumed across sessions.

- **Google Colab CLI (google-colab-cli, PyPI `google-colab-cli`, repo `googlecolab/google-colab-cli`, latest release 0.7.5 / 2026-10-20; created 2026-04-17):** yes, there is an official Colab CLI and a Colab API; the "no API / hallucination" reading of older training data is wrong. Install: `uv tool install google-colab-cli` (or `pip install google-colab-cli`; binary lands at `~/.local/bin/colab`). Auth is the real gate: the CLI supports **two providers** via global `--auth={oauth2,adc}` (default is `oauth2`), and the Colab backends need a specific scope set, so the headless/agent-safe path is ADC via `gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory` — user credentials minted by `gcloud auth application-default login` without that `--scopes` list **ignore** the scopes kwarg and raise `NotImplementedError` on `with_scopes`, so they must be re-minted. The `oauth2` path works too but opens a browser once and is less suited to fully headless use. Runtime flow (from the CLI docs/demos and AUTH_AGENTS): `colab new -s <name> [--gpu ...]`, `colab install -s <name> ...`, run the script, `colab log -s <name>`, `colab stop -s <name>`; Drive mount is `colab drivemount -s <name> [PATH]` (default `/content/drive`), and `colab usage` shows compute-unit rate/balance. Idle timeouts still apply on the free tier (free-of-charge notebooks run at most ~12 h depending on availability/usage per the official FAQ), and the paid ladder (Pro/Pro+/Pay As You Go / Google AI plan) buys more compute units and longer runs (Pro+ can run up to ~24 h when units suffice). So: Colab is usable headlessly for checkpointed runs, but free-tier continuous-time and accelerator availability are not guaranteed and can fluctuate; treat 4-hour checkpointing as the unit of work regardless of the source.

- **Paperspace:** viable as another checkpointed GPU source (Gradient jobs / persistent notebooks / cloud GPUs), and a natural fit for the same 4-hour-or-less checkpoint discipline. No Paperspace CLI deep-dive was pulled for this turn; if we rely on it, treat it like Colab — provision, run, checkpoint, retrieve artifacts, stop — and prefer the same harness + artifact layout so a run can move between sol, nyx, Colab and Paperspace without rewriting the script.

- **Modal (modal.com):** the intended first checkpointing target per the user: prototype on Modal, then run the full thing on Colab, with Paperspace as an overflow tier. Treat Modal as the cheap/fast validation layer before the longer Colab/Paperspace runs; keep every run checkpointed so the 4-hour ceiling is a resume point, not a restart.

- **OPAL (github.com/xingjian-bai/opal, `xingjian-bai/opal`, MIT, created 2026-10-21, v0.3.10, ~1.5k lines, 415 stars / 25 forks, 0 open issues, Python, topics `deep-learning`, `llm-agents`, `optimizer`, `pytorch`):** a **PyTorch optimizer wrapper** that lets a coding agent (default Claude Code) **rewrite the optimizer program while training runs**, plus a second analysis program the agent reads to decide. Mechanics from the README: `opal.wrap(opt, model, scheduler=sched, total_steps=..., goal="...", backend=..., max_cost_usd=...)` once around a normal `torch.optim` optimizer; the agent edits `updater.py` (the gradient→update rule) and `analysis.py` (whatever it wants to measure) in `opal_runs/<folder>/workspace/`; decisions happen every `every` steps (default 50); `blocking=True` (default) makes training wait for the agent, `blocking=False` keeps training going; a new optimizer program is trial-run for 20 steps and reverted on error if a snapshot fits in `snapshot_limit_gb` (default 2); the agent sees the goal sentence, param names/shapes, training losses, its own analysis, and optionally logged values / validation via `opt.log(...)` / `opt.report_validation(...)`; HF Trainer works through `opal.hf.OpalTrainer`, DDP within one node works (NCCL, eager and `torch.compile`), Accelerate works if you wrap after `accelerator.prepare` (without fp16 GradScaler or Accelerate grad accumulation); **multi-node, FSDP, DeepSpeed are not supported**; the whole test suite runs on CPU with no agent and no cost (`pytest -q`); the CLI side is `opal doctor`, `opal estimate`, `opal show/watch/report/compare`. Measured results in the README are on vision/translation/small-LM tasks (SiT-B/2, RAEv2-B, Goldfish 125M, LightningDiT-B/1, modded-nanogpt) — **none on a 1.5B-scale native ternary language model**. Verdict for this branch: **OPAL is a training-tuner, not a training harness and not a model converter.** It does not give us a BitNet-style MiniMinion trainer, a conversion path for Qwen GGUF→kindi, or MoE/pruning logic. It slots in **only if and when we already have a PyTorch training loop for MiniMinion** (the `tools/bitnet_qat` harness is the nearest thing, but it is a smoke-scale harness, not a full trainer yet) and we want an agent to tune the optimizer/schedule/loss-spike guards during a long run. On our hardware story it is also gated by the Claude Code dependency (agent billed to a Claude subscription or API key, `max_cost_usd` caps it, and the README explicitly says Codex is an experimental backend) and by the multi-node limitation — the cluster split (sol GPU + nyx CPU) is exactly the setting OPAL says it does not support yet. So: useful as a future optimizer-tuning layer on top of a real MiniMinion trainer, **not** something to install now as part of the conversion/pipeline work, and not a substitute for the_sol/nyx cluster plan. If and when we stand up a full MiniMinion PyTorch trainer on sol, OPAL is a candidate for the optimizer-tuning layer; not before.
