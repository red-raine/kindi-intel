# Working on this branch: Intel CPU support, lossless import, faster decode

You are on `freebuff/…-support-intel-cpu`. This file is the handoff. Read it, then `masterplan.md`, then the doc for the
experiment you are picking up. **`appendix.md` is the research behind the plan** — read the section you need before
proposing a technique, so you do not re-invent or re-litigate something already measured.

(This is not `docs/AGENTS.md`, which tells a user how to point their coding agent at a running Kindi server.)

## What this branch is for

1. **Intel CPU inference** — the engine must run well on a 9th-gen Intel desktop: AVX2, **no AVX-512**, DDR4, 8–64 GB,
   Windows and Linux. Decode on that machine is memory-bandwidth-bound, so bytes read a token is the currency.
2. **A lossless import pipeline** — a model that goes through `kindi convert` / `qat-pack` / GGUF conversion must do
   what its source did, and we must be able to *prove* it.
3. **MoE import at smaller sizes without wrecking the model** — pruning and merging experts, with gates that say when a
   claim is not established.

## The two rules that decide what counts as done

**A gate is a paired walk, not a mean.** Same tokens, same order, both arms; delta per token; a 95% interval from a
block bootstrap (`_bootstrap` in `tools/nll.py`, `gate()` in `tools/expert_merge/prune.py`). A sign is not a result. If
the interval covers zero, the text says so and the item stays open.

**Perplexity is not an acceptance test.** The `P2` functional screen is the second half of every lossless claim. Until
it exists, say "lossless on likelihood over N tokens", never "lossless".

Other standing rules:

- Never compare deltas across corpora. Every table states its token count. The same merge reads **+1.402** on 94 tokens
  and **+1.077** on 1,275; both are correct and they are not comparable.
- Match quantisation when comparing a merged or edited pack to a source: this build of `gguf` cannot *encode* K-quants,
  which is why both arms of the merge experiment are written at Q4_0 and the control is `--groups 128`.
- Report what was **not** run. A check that could not run is a limitation, not a pass.

## What is already established (do not re-derive)

| | |
|---|---|
| Ternary refit of a float checkpoint costs **+0.021 nats** over the QAT original | `docs/TERNARY-OPTIMIZATION-MATRIX.md` |
| Prune to **112/128** experts a layer is **lossless within 0.02** (+0.0045, CI −0.0082..+0.0174) for 82.9% of expert bytes, on 1,785 held-out tokens | `docs/EXPERT-PRUNE.md` |
| Prune to **96** is **not established** (+0.0164, CI −0.0023..+0.0356) | same |
| Prune to **64** beats merging to **64** at matched bytes: **+0.0902** [+0.0432, +0.1389] vs **+1.0770** [+0.9362, +1.2156], one 1,270-token walk, 7.59 GB vs 7.60 GB of expert weights | `docs/EXPERT-MERGE.md` §P3 |
| A 64-row router **fitted** to the teacher's routing (KL −0.0412) makes the merged pack **worse**: **+0.0836** [+0.0275, +0.1330] against the pooled rows — routing fidelity is not a likelihood lever; P4 killed | same |
| Merging 128 → 64 pairs costs **+1.077 nats** at the control's routing fraction; **routing ≤ +0.17**, **arithmetic ≥ +1.08** | `docs/EXPERT-MERGE.md` |
| 4 **super experts** (L2E92 2,152×, L1E114 927×, L3E82 525×, L1E68 210× their own layer's median output; next largest 42.7×). Routing mass ranks them **27, 27, 38, 92 of 128** — the 112 and 96 plans keep all four, the **64 plan drops L1E68** and 32 drops L1E68 + L2E92 | `tools/expert_merge/superex.py`, `docs/EXPERT-PRUNE.md` |
| Pruning beats a **uniform pairwise** merge at equal bytes - externally on this model class, and here on 1,270 tokens of this repo's prose (+0.0902 vs +1.0770 at 7.59 vs 7.60 GB). The global "merge is killed" is **withdrawn**: `mergepack --groups 64` caps a group at two experts, so it is M-SMoE shape, not REAM/RAVE | `appendix.md` §C.2, `docs/EXPERT-MERGE.md` |
| A `qwen3moe` GGUF **converts to a runnable engine pack** (`kindi/moeconv.py`, selftest loads it through `Engine`), but the pack is a **ternary re-quantisation of a Q4_0 source** — lossy by construction, `requant_mse` in the manifest. It exists so the P2 screen has arms to run; it is not an import path and carries no lossless claim | `kindi/moeconv.py`, `tools/funcscreen/README.md` |
| Absorbing the cold tail is **worse than deleting it**: absorb 32 of 128 into 96 groups (group sizes 1–4) costs **+0.2269 [+0.1549, +0.2937]** more than prune-96 on the same 96 Q4_0 slabs, walked in one process against one baseline; only RAVE's rank-4-8 residual (P13) is untested, and it must beat +0.2269 | `docs/EXPERT-MERGE.md`, 1,270 tokens |
| The 2026 state of the art is a **per-expert assignment** (keep / prune / absorb / absorb+rank-4-8 residual), not a choice between pruning and merging: REAM is ~90% pruning structurally with saliency centroids and singletons; RAVE reaches 52% memory at parity with 86% of decode speed and 0/40 degenerate vs REAM's 13/40 | `appendix.md` §C.2 |
| The prune/merge ranking is **corpus- and task-dependent in the literature** (REAP: merge "reasonably well" on MC, worse on generative; REAM: a Pareto curve in the calibration mix). Our kill is a generative-likelihood claim on one in-domain corpus | `appendix.md` §C.2, `docs/EXPERT-MERGE.md` |
| A rotation before ternary rounding is **not** the literature's win here: −2.6% mean weight error, **+1.5–2.1% worse** on expert matrices, and sequency ordering is *exactly* inert at a group-local quantiser — measured on 6 real matrices, no NLL walk | `tools/rotq/probe.py`, `appendix.md` §H |
| A symbolic/LUT/graph backbone is not supported for *answer quality*; the measured route to better answers is verifier-gated selection over the model's own samples | `appendix.md` §G |

## The plan, in short

`masterplan.md` §3 has hypothesis → method → gate → cost → kill for each. Order:

- **P0** protect the super experts in `prune` (`--protect`, `--protect-json`) and re-gate 96. *Do this first — it guards
  a pipeline we already ship.*
- **P2** a functional acceptance screen around `tools/astvalid` + 40 fixed prompts. Without it every later gate is weak.
- **P1** prune on **saliency** (mass × output) with **non-uniform per-layer budgets**; compare at equal bytes.
- **P5** 2.0 → 1.667 bits per weight (TL2-style packing) — the only accuracy-free speedup on the compute side. Measure
  the roofline with `tools/bitnet_gemm/stream_probe.py` before building it.
- **P6** pin the small always-hot tensors, read whole expert slabs instead of faulting pages, threads = physical cores.
- **P4** closed-form (ridge) router re-fit; then **P3** the mitigated merge, judged **against prune-at-96** — and
  expected to lose.
- **P11** the heterogeneous per-expert assignment (absorb, byte-matched against prune-96), then **P12** the
  out-of-domain and multiple-choice arms that scope the merge verdict; **P13** rank-4-8 residuals only if P11 leaves
  headroom.
- **P9** verifier-gated selection (`astvalid` + `--select k`): the only "better answers" route with end-task numbers, and it
  is P2's checker reused. After P2.
- **P10** rotation before rounding — **measured and parked**: the probe is one command, the verdict is above, and only an
  early-attention paired NLL walk can reopen it.
- **P7** exact n-gram speculative decoding, flag off by default, acceptance measured on our traffic.

`P8` lists what not to build and why (hugepages, `MADV_WILLNEED`, KV eviction as a speed play, retrieval as a
small-model reliability fix, a LUT-kernel rewrite for speed).

## Map

| what | where |
|---|---|
| engine, CPU kernels | `kindi/` — `cpu.py`, `kcpu.c`, `quant.py`, `packio.py`, `moe.py`, `spec.py` |
| conversion, ternary packing, refit | `kindi/convert.py`, `kindi/refit.py`, `kindi/gsq.py`, `kindi/rco.py` |
| detection, catalog packs (GLM-5.3-Flash, MiniMax M3) | `kindi/recipes.py`, `kindi/archs.py` |
| MoE analysis and editing | `tools/expert_merge/` — `routes.py` (route log), `prune.py`, `superex.py`, `merge.py`, `mergepack.py`, `real.py` |
| gates | `tools/nll.py` (NLL, bootstrap), `tools/kl.py`, `tools/expert_merge/prunegate.py` |
| kernel microbenchmarks | `tools/bitnet_gemm/` — `stream_probe.py`, `prefetch_budget.py`, `cluster_budget.py`, `tmac.py` |
| rotation vs the ternary quantiser | `tools/rotq/probe.py` — 8 orthogonal arms, our 128-group RTN, `--selftest`, `--json` |
| functional screen (unused so far) | `tools/astvalid/` |
| plan and research | `masterplan.md`, `appendix.md` |
| the repo gate | `python -m tools.selftest` (runs every `suite:*`, including `suite:superex`) |

Docs worth knowing: `docs/IMPORT-CATALOG.md`, `docs/EXPERT-MERGE.md`, `docs/EXPERT-PRUNE.md`, `docs/CPU-PERF-2026.md`,
`docs/TERNARY-LUT-AND-PAGING-2026.md`, `docs/RECIPES.md`, `docs/TESTING.md`. `masterplan.md` and `appendix.md` are the
newest and carry the current direction.

## Traps on this machine (all hit, all costly)

- **Windows host.** `multiprocessing` pools lose their workers here; the shard-and-combine pattern in
  `docs/EXPERT-MERGE.md`'s Reproduce block is the reliable way to use every core.
- **A full merge is ~35 min; a three-arm 1,275-token walk is ~1.5 h.** Start it in the background and check the log; do
  not block a session on it.
- **Redirected Python output is block-buffered** — a background run looks silent until it exits. Use `-u`, or read the
  JSON it writes at the end.
- **Big artifacts live outside the repo** under `B:/models/scratch` (packs, route logs, JSON). They are not in git and
  must not be. The JSON a doc quotes is small enough to describe, and the command that made it is in the doc.
- **`tools/selftest.py` runs the whole suite.** For a change in `tools/expert_merge/`, run that module's `--selftest`
  first; run the full gate before a commit that touches shared code.

## How to continue

1. Read `masterplan.md` §6 (status board). Pick the highest item that is `open`.
2. Read the doc it points at (`docs/EXPERT-PRUNE.md`, `docs/EXPERT-MERGE.md`, …) — the reasoning and the failed attempts
   are there, so you do not spend a day re-finding a dead end.
3. Do the measurement, then **write the numbers into the doc with their interval and their token count**, and mark the
   item on the status board.
4. Commit with the finding, not the file list: the last four commits are the house style. Push only when asked.

If a research result in `appendix.md` looks wrong for us, the way to say so is an experiment with an interval, not an
argument — and it goes in the appendix's §F with the numbers that contradict it.
