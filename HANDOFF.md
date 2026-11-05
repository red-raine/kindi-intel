# Handoff: intel-kindi fork (resume / finish)

**Recovered 2026-11-05** after a Freebuff desktop session-load failure. The code work was found locally in a Freebuff worktree and moved here as a normal git clone.

## Repo state (verified)
- **Path:** `E:\viber_coding\freebuff\kindi-fork`
- **Branch:** `freebuff/i-am-making-a-branch-of-this-to-support-intel-cpu--5b5aa0d9-f359-4bae-960d-4e41966d24c5`
- **HEAD:** `fee7bb0` — *Guard the super experts, and find the 64-expert loss is 95% capacity*
- **Working tree:** clean (no uncommitted changes). Committed work is pushed to **Gitea** and **GitHub** on this branch (the last 6 commits were pushed to GitHub during recovery).
- **Remotes:**
  - `origin` → `https://github.com/red-raine/kindi-intel.git`
  - `gitea`  → `http://zcode@192.168.8.217:3005/lair/kindi-intel.git` (LAN; may require network access to 192.168.8.217:3005)
- **Divergence:** 42 commits unique to this branch vs `origin/intel-cpu-engine`, 16 commits only on `origin/intel-cpu-engine` (overlapping work; compare before merging). Area diff vs `origin/intel-cpu-engine`: ~99 files, +18,660 / -576 (tools: expert_merge/superex, nll/kl, kvq, moe_loop, rotq/probe, etc.).

## Original objective (recovered from Freebuff session cache)
> *“I am making a branch of this to support intel cpu gpu and npu, i have a 9th gen intel i9 and 2 i9 ultra mini pcs…”* (Freebuff chat `5b5aa0d9-f359-4bae-960d-4e41966d24c5`, `deepseek/deepseek-v4-flash`, plan mode, completed 2026-11-01 UTC; no recoverable unsent/pending prompt text found in local profile storage.)

Intent captured: Intel **CPU/GPU/NPU** acceleration (9th-gen i9; 2× i9 Ultra mini PCs with iGPU/NPU/CPU; 1× with **Intel Arc B50**). Find hardware docs/features, adapt code for **kindi** accelerated model running — fastest path per hardware profile; consider disk I/O/p2p/memory, improved inference; research libraries (OpenVINO, llama.cpp/SYCL/oneAPI/Level Zero etc.), including bleeding-edge/experimental/academic options.

## Current technical focus (from `masterplan.md`)
Read `masterplan.md` first. Two objectives: (1) **lossless import** (paired NLL walk, 95% interval, functional screen — not PPL), (2) **faster inference on a 9th-gen Intel CPU** (AVX2, no AVX-512, DDR4; decode bandwidth-bound; CPU wins are mostly memory-side). Measured results and gates are recorded in §1–§2; plan items P0–P10 in §3 (P0 done/measured; see status board/rules). Also read `appendix.md`, `docs/EXPERT-PRUNE.md`, `docs/EXPERT-MERGE.md`, `docs/TERNARY-OPTIMIZATION-MATRIX.md` as needed.

## Where it left off
- Latest work guarded super experts by design (P0) and measured that 64-expert loss is ~95% ordinary capacity (see commit `fee7bb0` / `b254450` and masterplan P0 result).
- The branch has substantial measurement/import/tooling work (42 commits vs `intel-cpu-engine`) not fully reconciled with `origin/intel-cpu-engine`’s 16 unique commits — reconcile carefully, preserving measured evidence and gates.

## Agent prompt (paste this to continue)
You are resuming the `kindi-intel` fork at `E:\viber_coding\freebuff\kindi-fork` on branch `freebuff/i-am-making-a-branch-of-this-to-support-intel-cpu--5b5aa0d9-f359-4bae-960d-4e41966d24c5` (HEAD `fee7bb0`, clean). Read `HANDOFF.md` and `masterplan.md` (especially objectives, §1 measured results, §3 plan/status, §5 rules). 

**Goal:** finish the Intel acceleration fork: keep import lossless by the plan’s gates, and make kindi inference fast on the target hardware — starting with the 9th-gen Intel i9/AVX2/DDR4 CPU path (P6/P5/P7 priorities as masterplan indicates), then extend to the stated Intel GPU/NPU targets (i9 Ultra mini PCs: iGPU/NPU/CPU; one system with Intel Arc B50) using the best-supported path (evaluate OpenVINO / llama.cpp SYCL & Vulkan / oneAPI Level Zero / IPEX-LLM-style options as appropriate; prefer measured, reproducible wins). 

**First actions:** (1) confirm current state vs `origin/intel-cpu-engine` (16 unique commits there) and produce a safe reconciliation plan; (2) identify the next highest-ROI unfinished masterplan item consistent with measured evidence; (3) implement with the repo’s NLL/interval/functional acceptance gates — do not claim speedups or losslessness without measurement; (4) commit in logical steps and push to `origin` (and `gitea` when reachable). Report what you measured, what changed, and what remains.

## Recovery artifacts (kept outside repo)
- `E:\vibe_coding\freebuff\Freebuff-Roaming-backup-20261105-111243\` — Freebuff profile backup (session/UI state).
- `E:\vibe_coding\freebuff\kindi-intel-worktree-backup-20261105-111345\` — file copy of the original worktree.
- Original worktree still exists: `C:\Users\myste\.freebuff\worktrees\5b5aa0d9\cfd4443d-708c-45e0-965b-28754197a1f8\` (linked to `C:\Users\myste\.config\freebuff-desktop\cloud-projects\cfd4443d-708c-45e0-965b-28754197a1f8`).

## Pending prompts
No recoverable unsent/pending prompt text was found in the local Freebuff Session Storage / Local Storage / IndexedDB backup for chat `5b5aa0d9`. If work was interrupted mid-turn, re-check https://freebuff.com for the server-side session, or restate the next instruction in a new chat against this repo.
