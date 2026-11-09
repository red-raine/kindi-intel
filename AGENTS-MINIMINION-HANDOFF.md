# Handoff for Antigravity / MiniMinion — sidecar + research-lab loop

This file is the written handoff for Antigravity to pick up MiniMinion and/or a MiniMinion-derived sidecar script. It is intentionally concrete: tasks, artifacts, gate, and the contract the sidecar must satisfy so the main works stays the main works.

## Where MiniMinion lives (measured from this box)

MiniMinion is **not** a theoretical target. I can see it directly on the same local Gitea instance this repo pushes to:

- Gitea instance: `http://192.168.8.217:3005` (reachable from here via HTTP; `localhost:3005` is **not** reachable from this process)
- MiniMinion repo: `antigravity-crystal-eyes/mini-minion-0` — API `GET /api/v1/repos/antigravity-crystal-eyes/mini-minion-0` returns `200`; repo id `31`, owner `antigravity-crystal-eyes`, language Python, default branch `master`, updated `2026-11-08T21:40:51Z`.
- MiniMinion UI: `http://192.168.8.217:3005/antigravity-crystal-eyes/mini-minion-0` returns `200`, title `antigravity-crystal-eyes/mini-minion-0: Zero-Token Autonomous Agent Swarm OS with PseudoDAG IR, Hermes Multi-Agent Frames & Swimba Polymorphic MoE Engine`.
- NanoResearch repo: `opencode/NanoResearch` — API `GET /api/v1/repos/opencode/NanoResearch` returns `200`; repo id `13`, owner `opencode`, MIT, Python, default branch `main`, description `Mirror of OpenRaiser/NanoResearch (MIT) - autonomous research generation engine; integrated with jitral-scientest verifier`.

I also pulled the mini-minion-0 archive from `http://192.168.8.217:3005/antigravity-crystal-eyes/mini-minion-0/archive/refs/heads/master.zip` (6,728,585 bytes, zip ok) and read enough of it to know the repo has skills under `.agents/skills/` including `mini-minion-sidecar`, `ai-science-lab`, `auto-research-in-sleep`, `autoresearch-mini-minion`, plus an `AGENTS.md`, `MANIFEST.md`, `README.md`, `minion/` tree, a `minion_harvest_config.yaml`, and docker/cloud run support. So MiniMinion and NanoResearch are local and reachable; the handoff below is written for them to pick up, not for a repo I cannot see. Today we have:
- a real measured token rate on sol: **B50 on `torch.xpu` gives 3968.4 tok/s on a 58.5M MiniMinion harness**, loss 10.486 → 9.263 in 20 steps;
- a working local bitnet smoke harness under `tools/bitnet_qat/` (a small native ternary training smoke, not a full trainer yet);
- the existing MoE prune/merge tools under `tools/expert_merge/`, with P0/P1 already gated (prune-112 certified lossless on 1,785 held-out tokens);
- cluster split: sol = Arc B50 GPU + GenAI arm; nyx = CPU worker.

The next phase is: keep the pipeline moving toward **near-lossless or lossless** conversion/compression even if we accept post-training, distillation, or better PTQ along the way. This handoff is the exact work Antigravity/MiniMinion should own.

## What MiniMinion already brings (use it)

MiniMinion is **not merely a sidecar wrapper**. From the repo I pulled, it already has:

- a `mini-minion-sidecar` skill that is explicitly framed as a universal sidecar/debugger/Wayward DAG generator/autonomous science lab for Antigravity, Claude Code, OpenCode, Freebuff, Poolside, ZCode, and GSD;
- an `ai-science-lab` skill with a 4-stage experiment protocol (pre-registration → 4-rung ablation → statistical check → OpsML packaging), Bootstrap/Welch statistical checks, and AI architecture history from GOFAI through BitNet b1.58 / Mamba / BitMamba;
- an `auto-research-in-sleep` skill plus the NanoResearch mirror (`opencode/NanoResearch`) as the autonomous research generation engine paired with a `jitral-scientest` verifier;
- a zero-token doctrine via `CostGuard.assert_zero_tokens()`, LiteLLM cluster routing across `sol`/`nyx`/`desktop`, and a `minion_harvest_config.yaml` with provider + cluster routing.

So when I say "use MiniMinion", I mean: anchor on the repo’s existing skills and contracts instead of inventing a new research-lab shape from scratch. The sidecar I ask for should plug into the existing `mini-minion-sidecar` / `ai-science-lab` shape where possible.

Concrete abilities to use from what I saw:

- **Distillation logic.** The repo’s own phase/benchmark scripts already include distillation-shaped entries (`scripts/run_poolside_20epoch_distill.py` and the longer discovery/benchmark phases). Use that as the local hint that MiniMinion can own distillation loops; keep the clean skeleton you pasted (freeze teacher, STE through the student’s BitLinear backward, `temp**2` scaling, `clip_grad_norm`, blended KL+CE) as the starting point, but gate it on the same paired-NLL + P2 language, run it on the actual target tier before committing to scale, and treat it as an experimental recovery layer, not a replacement for the lossless import pipeline.
- **MoE expert compression logic.** The repo already speaks MoE (Swimba Polymorphic MoE Engine, Zipfian tiering, Arc hardware). Use that as the home for the Fisher-weighted expert merge, routing-aware sparsification, and token-bypass/early-exit ideas — each as its own gated experiment, not one big bundle.
- **Hypothesis-experiment loops.** MiniMinion already has an `ai-science-lab` skill and an `auto-research-in-sleep` skill; the handoff should route through them. The same paired-NLL gate discipline this branch already uses for MoE pruning should extend to the other ideas below instead of inventing a new verdict language.

- **Distillation logic:** temperature-scaled KL + hard CE with a freeze-the-teacher + STE-through-the-student pattern (the `execute_ternary_distillation` shape in the pasted QAD loop is the right skeleton; keep the freeze, the `temp**2` scale, `clip_grad_norm`, and the STE hook on the student's BitLinear backward).
- **MoE expert compression logic:** activation profiling + pairwise similarity + Fisher-weighted fusion + router logit summation + physical shrink path (the pasted Experiment 2 skeleton is the right one; keep the FIM-diag accumulation only on the expert under test to control VRAM).
- **Hypothesis-experiment loops:** MiniMinion should be allowed to run background research loops: propose a hypothesis, run the smallest possible experiment that can falsify it, log the interval, and decide to keep/revise/kill. That is the same paired-NLL gate logic this branch already uses for MoE pruning — extend it to the other ideas below instead of inventing a new verdict language.

## Where we already said “no / not now” — keep these intact

Do not have MiniMinion re-open these as if they were still live options:

- **No OpenVINO IR inside the kindi engine.** OpenVINO GenAI is a *deployment comparison arm only* (INT4 export → LLMPipeline on GPU.1 vs kindi ternary on CPU, measured tok/s on the same prompts). IPEX excluded; native torch XPU is the maintained path.
- **HumanEval is eval-only in the harness.** AEGIS is a safety/reliability dataset; it is not verified code and is not a pass criterion for code correctness.
- **Large artifacts stay out of git** under `B:/models/scratch` (or the LAN-equivalent path on each node).
- **No “unlimited free Colab GPUs in 4 h chunks.”** The honest Colab model is: free tier is dynamic, T4 when available, ≤12 h free with no guarantees; Pro 100 CU/mo; faster GPUs/TPUs quota-gated and CU-expensive. Checkpoint at or under 4 h regardless of tier.
- **OPAL is a PyTorch optimizer wrapper, not MiniMinion.** MiniMinion *uses* what OPAL demonstrates (agent-tunable optimizer policy, trial-then-swap with snapshot rollback, blocking/nonblocking run loop, logged metrics + optional validation reporting). If MiniMinion runs in the cloud, it can carry the same idea into the Colab/Modal/Paperspace script we upload with it — but OPAL itself is not the trainer and not the converter, and it explicitly does not support multi-node/FSDP/DeepSpeed.

## Tasks for MiniMinion / the sidecar

### 1. Sidecar contract (the uploadable thing)

MiniMinion should produce or refine **one small sidecar** that travels with the cloud script and can be dropped into Colab/Modal/Paperspace with minimal glue. Prefer building on top of the repo’s existing `mini-minion-sidecar` skill rather than inventing a fresh one. The sidecar must:

- run headless, accept a small config (device, checkpoint_dir, run_id, artifact_prefix, max_wallclock_seconds, resume_from);
- checkpoint often and resume from the latest checkpoint deterministically;
- write a single JSON artifact per run with: run_id, start/stop, device, tok/s if measurable, loss trajectory (sampled), final loss, any gate decision + interval + token count, and a pointer to the checkpoint;
- never write large artifacts into the repo; write them to the configured scratch dir and only log the pointer;
- be importable by a tiny runner script so the same sidecar can be used on sol-local XPU, nyx CPU, Colab, Modal, and Paperspace without rewriting experiment logic.

That sidecar is the unit of parallelism: once it exists, multiple experiments can run at once on different nodes/tiers while you orchestrate.

### 2. Distillation / post-training direction MiniMinion should explore

Keep the goal narrow: **what abilities do we most want to preserve at frontier level?** Likely candidates for this branch: code generation / AST fidelity, reasoning over code, and routing/expert behavior for MoE variants. Have MiniMinion:

- treat distillation as an *experimental layer*, not a replacement for the lossless conversion pipeline;
- test whether a lightweight post-training step can recover capacity lost at a given compression point, **measured against the same paired-NLL + P2 gate** we use for prune/merge, so the verdict language stays consistent;
- if it pursues QAD-style loops, keep the steamlined skeleton above and keep teacher frozen + student STE; measure VRAM/tok-s on the actual target tier before committing to scale.

### 3. Better 1.58-bit PTQ direction MiniMinion should explore

MiniMinion should research-and-prototype the CAT-Q / ScaleQ-1.58-style idea **as a local prototype first**, and only as a **gateable experiment**, not as a claim:

- the live idea is: replace a static AbsMean-style threshold with learned group-wise scale/threshold parameters, using a soft/continuous ternarization proxy during a short calibration so gradients can tune $\alpha$ and $\Delta$ per group;
- the calibration domain matters: if we do this, calibrate on data that matches the target use (the branch already treats Python AST / HumanEval-style logic density as the material we care about preserving);
- the integration consequence is real: if we ever ship learned per-group $\alpha$ and $\Delta$, the AVX2 T-MAC path must apply them after unpacking the 2-bit codes, so this is an engine+pack change, not just a packing-script change;
- MiniMinion should **gate any such prototype the same way**: same tokens, same order, both arms, delta per token, 95% interval from block bootstrap, and say “not established” whenever the interval covers zero.

### 4. MoE compression direction MiniMinion should explore next

This branch already has prune-112 certified. The next MoE work MiniMinion can own:

- Fisher-weighted expert merging as a candidate step before or alongside pruning, gated on the same paired-NLL rule, with the router update and physical shrink step included in the experiment artifact;
- routing-aware sparsification where the per-expert covariance/reconstruction only uses tokens actually routed to that expert;
- token-bypass / early-exit ideas as a separate experiment with its own gate — do not bundle “prune + merge + token bypass + PTQ” into one experiment and then call the result “lossless.”

### 5. Background research lab loop

MiniMinion should be allowed to run a background loop that:

- keeps a running list of hypotheses with status: proposed / running / gated / killed;
- for each hypothesis, records the smallest falsifiable experiment, the metric, the interval, and the token count;
- periodically replays the status board back into a doc so the human can read it without entering the loop;
- prioritizes experiments that unblock other experiments (for example: get the sidecar working on Colab before trying a long distillation run on Colab).

## What MiniMinion should stop doing

- Do not start new big codegen/conversion features unless they come with a gate or a clear measurement plan.
- Do not declare “lossless” on likelihood alone; “lossless on likelihood over N tokens” until the P2 functional screen exists.
- Do not silently change the verdict language; if an interval covers zero, say so and leave the item open.
- Do not put model/artifact checkout or heavy output into git.

## What MiniMinion should hand back

At the end of any research burst, MiniMinion should leave:
- a short status summary (what ran, what gated, what is still open, what failed to run and why);
- the artifacts with the smallest useful JSON summaries checked into the repo and the large ones left in scratch with pointers;
- a written note of any decision that should change the cluster plan (for example: “sidecar now works on Colab; first long run should go there”).

For NanoResearch specifically: treat it as the long-running autonomous research loop, with jitral-scientest as the verifier gate, and feed it the same hypothesis/experiment/interval/tok-count format this branch uses so its output is comparable to the prune/merge verdicts.

When Antigravity picks this up, the expected first move is: build or refine the sidecar contract above, verify it runs headless on at least one tier, and then farm parallel gated experiments out to sol / nyx / Colab / Modal / Paperspace while the human keeps orchestrating. MiniMinion is the worker lab; this handoff is the lab notebook + the contract for the uploadable sidecar.

## Handoff end state

When Antigravity picks this up, the expected first move is: **build or refine the sidecar contract above, verify it runs headless on at least one tier, and then farm parallel gated experiments out to sol / nyx / Colab / Modal / Paperspace** while the human keeps orchestrating. MiniMinion is the worker lab; this handoff is the lab notebook + the contract for the uploadable sidecar.
