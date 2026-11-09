# Appendix: research behind the plan

Every link here was read by a researcher agent and its number is quoted from the source, not from a summary of a
summary. Confidence tags: **[measured]** primary measurement (paper/PR/benchmark with numbers), **[replicated]** two or
more independent measurements agree, **[claim]** vendor or author assertion, **[null]** a measurement that found no
effect — those matter as much as the wins.

Grouped by the question it answers. `masterplan.md` §2 says which of these changed the plan.

---

## A. Ternary kernels on CPU: where the speed actually comes from

| source | what it says | what it implies here |
|---|---|---|
| [BitNet.cpp paper, arXiv 2502.11880](https://arxiv.org/html/2502.11880v1) | The TL1 (2 trits, 9-entry int8 table) / TL2 (3 trits in 5 bits = **1.667 bpw**, sign plane + 4-bit index, `vpshufb`) / I2_S (non-LUT integer MAD) taxonomy. Kernel-vs-kernel: **TL2 beats TQ1_0 by 1.33× on x86**, TL2 beats T-MAC 2.32× on x86 but only **1.19× on M2 Ultra**. [measured] | The honest AVX2 headroom is ~1.2–1.3×, and 1.667 bpw is a real 17% byte cut. This is `P5`. |
| [1-bit AI Infra 1.1, arXiv 2410.16144](https://arxiv.org/html/2410.16144v2) | 6.25×/6.46× are **vs llama.cpp FP16** on the same model; the i7-13700H (no AVX-512 in practice) shows them, so they are a bandwidth story, not an AVX-512 story. Energy −71.9..82.2%. Kernel exactness table: llama.cpp TQ1_0/TQ2_0 match fp32 **1.4%** of outputs, bitnet.cpp TL1/TL2/I2_S **100%**. [measured] | "Lossless" in the strict bit sense is a kernel property; ours should be checked the same way. The multipliers are a baseline artifact. |
| [BitNet codegen requirements](https://github.com/microsoft/BitNet/blob/main/docs/codegen.md) | TL1 needs `K%2==0`, block multiples of 32/64; **TL2 needs `BK % 6 == 0`** and splits K into a `%6` part plus a TL1 remainder; `vpshufb`'s 16-entry table limit is *why* TL2 exists. [measured] | Concrete packing constraints for `P5`; our ternary groups are 128 weights, so a 1.667 bpw layout must handle the remainder. |
| [T-MAC, arXiv 2407.00088](https://arxiv.org/html/2407.00088v2) | Bit-plane LUT per-bit shift-and-add, 4-bit groups, mirror consolidation; up to 6.6× kernel, 3.6× avg vs llama.cpp; its own README concedes old x86 has less bandwidth than a Pi 5 and "cannot guarantee significant speedup". [measured] | Don't chase T-MAC on DDR4. |
| [Intel, arXiv 2508.06753](https://arxiv.org/html/2508.06753v1) | Roofline for decode GEMV: `T = max(B/β, γ)` — speedup comes from **smaller B, not smaller γ**. Their non-LUT int2 AVX2 kernels beat bitnet.cpp by **up to 2.2×**, and int2 GEMV lands within 10–20% of the int8 bandwidth roofline. [measured] | Confirms `P5`'s logic: bytes are the lever; kernel cleverness stops mattering at the roofline. |
| [Vec-LUT, arXiv 2512.06443](https://arxiv.org/html/2512.06443v2) | T-MAC time breakdown: **lookup 47.6%** of single-thread mpGEMM; scalar-LUT kernels use **<40% of bandwidth** under parallel inference. [measured] | The LUT tax is real and is worst exactly where we are (multi-thread, bandwidth-bound). |
| [tommyyliu/lut_mm](https://github.com/tommyyliu/lut_mm) | Independent AVX2 run of BitNet's generated TL2 kernel: naive AVX2 85.1 Gop/s → **TL2 182.1 (2.14×)**; a 512-bit 5-trit LUT reaches 8.35× but needs AVX-512. [measured] | 2× over naive, not 6×. Same conclusion. |
| [catid/bitnet_cpu](https://github.com/catid/bitnet_cpu), [pantalipe/cpu-llm-bench](https://github.com/pantalipe/cpu-llm-bench), [andrew.ooo review](https://andrew.ooo/posts/bitnet-cpp-review-microsoft-1-bit-llm-cpu-inference) | Same-machine AVX-512 vs AVX2: 50 vs 15 tok/s (3.3× ISA gap); pre-AVX2 reproduction still 6.46 tok/s vs 1.2 but **output was broken**; a reviewer "did not reproduce any of them" and documents a live SiLU/ReLU² regression. [measured]/[claim] | Treat any ternary speedup table without a same-machine baseline as marketing. |

## B. Memory-bound CPU inference: what is measured to help

| source | what it says | what it implies here |
|---|---|---|
| [llama.cpp performance notes](https://johannesgaessler.github.io/llamacpp_performance) | **5 threads saturate dual-channel DDR4**; more threads hurt (measured drop 8→9 threads). [measured] | `P6`: thread count is a bandwidth knob with an optimum — measure ours. |
| [llama.cpp #28187](https://github.com/ggml-org/llama.cpp/discussions/28187) | mlock of only the small non-expert tensors: **+26% tok/s, 2.9× fewer major faults**. Adjacent tokens share only **35.7%** of top-8 experts; 80% coverage needs 20–33 of 64 experts; a userspace LRU lost to the page cache; RAM tier ≥75% of the box collapsed to 0.12 tok/s. [measured] | `P6`'s first item, and the reason "cache the last token's experts" is not worth building. |
| [mbolt / llama.cpp #18758](https://github.com/ggml-org/llama.cpp/discussions/18758) | Explicit merged reads instead of 16 KB faults: **+13–14%** decode; with a prefetcher **+31–52%**; layout reorder + interleave **+55%** (5.80 → 9.00 tok/s). Reads/token 1,418 → 370. Gain law `g ≈ 1 + saved·t_op/(slice/BW)` with **t_op ≈ 100 µs**: pays at ~1 MB slices, not at 4 MB. [measured] | Our expert slab is 1.708 MB — inside the paying range. `P6`. |
| [llama.cpp #23324](https://github.com/ggml-org/llama.cpp/discussions/23324) | Windows + i7-8700K + DDR4-2666 + NVMe: overlapped expert reads sustain **2.80 GB/s in generation vs 377 MB/s via mmap paging (7.4×)**; decode 0.43 → 0.75 tok/s. [measured] | Our host is Windows; this is the most relevant single number for the MoE path. |
| [colibri expert-IO study](https://raw.githubusercontent.com/JustVugg/colibri/main/docs/experiments/dsv41-expert-io-2026-09-14.md) | O_DIRECT vs buffered: QD=1 **2.6×**, QD=4 the knee (6.34 GB/s), −48% disk phase and **+63% decode** once CPU threading was fixed; 208-thread run had concluded I/O "cannot matter" and that conclusion was wrong. [measured] | Method lesson: fix threading before judging I/O. |
| [file_io_benchmark](https://github.com/m-schuetz/file_io_benchmark) | Windows, 14.8 GB/s drive: unbuffered overlapped **11.3 GB/s**, `fread` 4.2–5.0, **mmap 6.8 multi-thread but 1.2 single-thread**. [measured] | Mapped views are a serial-fault disaster at one thread. |
| [llama.cpp #26110](https://github.com/ggml-org/llama.cpp/issues/26110) | Double-caching: `mlock` + offloaded experts = file *plus* anon copy → **OOM at 60 GB**; unpinned anon buffer swapped and throughput fell **25–27 → 7.55 tok/s**. [measured] | `P6`'s failure mode to test for: exactly one copy of the bytes. |
| [llama.cpp #2251](https://github.com/ggml-org/llama.cpp/issues/2251), [strata #1958](https://www.stratat.com/de/issues/1958) | Hugepages / `MADV_HUGEPAGE` / hugetlb: "no noticeable performance difference"; direct A/B **−0.1% prefill, −0.6% gen**, i.e. noise. [null] | Don't build it. `P8`. |
| [Qdrant madvise test](https://gist.github.com/timvisee/a775f9a0016f5410b46ce7fdbd1b253b), [camunda #11377](https://github.com/camunda/camunda/issues/11377) | `MADV_WILLNEED` on a 20.9 GB mapping: 44.35 s → **47.11 s** (worse); Kafka-style log: 190 → 175–188 PI/s (worse). [null] | Don't use it as a warmer. `P8`. |
| [HotStorage'17](https://www.usenix.org/system/files/conference/hotstorage17/hotstorage17-paper-choi.pdf), [OptMLX](https://atomgradient.github.io/OptMLX/paper.pdf) | `MAP_POPULATE` beats demand faulting (+46% at 4 KB records); cold mmap load speedups ranged 0.69×–20.65× with 3/8 models *slower*. [measured] | Mapping is not automatically better; measure on our file. |
| [yzma #375](https://github.com/hybridgroup/yzma/pull/375), [llama.cpp #11765](https://github.com/ggml-org/llama.cpp/discussions/11765) | Thread pinning to P-cores removes ~30% run-to-run spread; EPYC best at 24 threads (~3 cores/memory channel), more threads slower. [measured] | `P6` item (c). |
| [Arm cross-NUMA](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/introduce-the-cross-numa-problem-and-optimization-in-llama-cpp-with-llama3-model-running-in-neoverse-n2), [llama.cpp #1437](https://github.com/ggml-org/llama.cpp/issues/1437) | Cross-NUMA imbalance costs **55%** (26.5 → 41.2 tok/s); dual-socket can be **slower** than single-socket (146 → 254 ms/token). [measured] | Irrelevant on a single-socket 9th-gen desktop — noted so nobody spends a day on it. |

## C. MoE merging vs pruning: the literature turned against merging

| source | what it says | what it implies here |
|---|---|---|
| [REAP, arXiv 2510.13999](https://arxiv.org/html/2510.13999) (ICLR 2026) | Qwen3-30B-A3B, one-shot, no FT: pruning **−1.9% coding @25%, −6.9% @50%**; M-SMoE/HC-SMoE worse than −5% and −20%. Derives the merge error floor **E ≈ E[(g_i+g_j)²]·Var[r(x)]·G_ij** — strictly positive whenever routing varies, so no α escapes it. Merged models show lower n-gram diversity and growing logit-JSD with position. [measured] | Our +1.08..+1.20 nats at 50% is the same result measured in nats. **Merge demoted; prune promoted.** `P3`/`P1`. |
| [Super Experts, arXiv 2507.23279](https://arxiv.org/html/2507.23279v3) (ICLR 2026) | **3 of 6,144** experts (L1E68, L2E92, L3E82) in Qwen3-30B-A3B; removing them: 70.22 → **55.00**, GSM8K 89.6 → 42.4, HumanEval 43.9 → **0.0**, PPL 8.7 → 59.9; random removal of the same count −0.20. DeepSeek-R1 75.64 → 1.81. [measured] | `P0`. Our audit reproduces **L2E92 and L3E82 exactly**, and finds our own L1 outlier (E114) 4× bigger than E68. |
| [Router KD, arXiv 2603.02217](https://arxiv.org/html/2603.02217) | "In almost all retraining-free approaches the router is left unchanged"; router-only distillation (~**0.04%** of params, ~2 h) recovers consistently, **more for fine-grained MoE** (Qwen3 128 experts) than coarse (Mixtral 8); fails under catastrophic collapse and can regress when compression accidentally helped. [measured] | `P4`, in its cheapest closed-form form. |
| [Beyond Retraining-Free, arXiv 2609.06076](https://arxiv.org/html/2609.06076v2) | Router-only is **insufficient**: it cannot recover expert-function distortion; Router+Top-8 expert FT beats Router+Top-16 KD; full FT recovers ~37% of the gap. [measured] | Caps what `P4` can claim: routing is necessary, not sufficient. |
| [PRIME, arXiv 2610.13982](https://arxiv.org/html/2610.13982) | Frequency-weighted prune+merge with **ridge-regression router reconstruction**; removing frequency weighting costs **−3.7 to −6.5** points; K=96 MC 70.46 vs REAP 68.95. Preprint, small calibration set. [measured, unreviewed] | The estimator for `P4`, and the ablation behind `P1`'s saliency weighting. |
| [REAM, arXiv 2604.04356](https://arxiv.org/html/2604.04356) | Best "merging wins" result is ≤1.2 points at 25% and is structurally ~90% pruning; swapping saliency for raw frequency costs **−8.7** points. [measured] | Confirms saliency over frequency (`P1`). |
| [MC-SMoE, arXiv 2310.01334](https://arxiv.org/html/2310.01334) | 60%/80% claims are merge + low-rank + sparse + KD fine-tune on a **2B T5 with 32 experts**, and the pipeline **skips the first MoE layer**. [measured, weak baseline] | The one thing to copy: never compress the first MoE layer. Reinforces `P0`. |
| [MoE-XBench, arXiv 2608.21693](https://arxiv.org/html/2608.21693) | On Qwen3-Coder-30B specifically: prune-only at 28% keeps Score_perf 96.96 but Score_robust 88.87 (MMLU 0.847 → 0.742, HumanEval 0.939 → 0.811), while **quantisation alone at 69.5% compression holds 0.8127**. [measured] | `P2`: robustness moves before perplexity does. |
| [HC-SMoE, arXiv 2410.08589](https://arxiv.org/abs/2410.08589) | Best merge method on multiple-choice, **worst on generative** (creative writing 0.811 → 0.497 at 25%, 0.008 at 50%). [measured] | Why `P2` is a functional screen and not another MC benchmark. |
| [EEP, arXiv 2407.00945](https://arxiv.org/html/2407.00945) | Router re-mapping + expert merging at a fixed expert budget beats frequency pruning by 5–7 points — on an 8-expert T5 toy. [measured, small scale] | The pro-router-refit counterweight, and why `P4` is worth an hour. |

## D. Low-bit GGUF: bits, formats, and what "lossless" is worth

| source | what it says | what it implies here |
|---|---|---|
| [llama.cpp PR #8151 (TQ1_0/TQ2_0)](https://github.com/ggml-org/llama.cpp/pull/8151) | TQ1_0 = **1.6875 bpw** (5 trits/byte), TQ2_0 = 2.0625; **identical values**, round-trip lossless; embeddings Q4_K and output Q6_K kept high; TQ2_0 ≈ **2× Q4_K on AVX2**. Raising just embeddings+output to f16 dropped KLD 5.5× (**0.00187 → 0.00034**) and raised same-top-p 97.3 → 99.0%. [measured, 390M model] | Where the error lives: the two non-matmul tensors. Our packs keep them f32 already; `P2` should confirm on our model. |
| [llama.cpp perf README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md), [quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) | The canonical table: iq2_XXS +7.86 PPL, iq1_S **+54.5** (1.88 GiB), q2_K +2.42; effective bpw for a 2.4–3.0 "2-bit" quant is 2.4–3.0 because of mixed precision. Stale (2024). [measured] | 1.x bpw is not deployment-grade; 2–3 bpw needs an imatrix and a functional check. |
| [imatrix PR #4861](https://github.com/ggml-org/llama.cpp/pull/4861), [PR #4773](https://github.com/ggml-org/llama.cpp/pull/4773) | The importance matrix is the *diagonal* of ⟨aaᵀ⟩ per input channel — one number per channel, no Hessian, no error feedback; at 2 bpw the fallback is "total garbage". `--process-output` defaults off. [measured] | If we ever calibrate a converted pack, the output tensor must stay out of the matrix. |
| ["Broken on Arrival", arXiv 2609.05881](https://arxiv.org/html/2609.05881v2) | 305 published tags: **7 non-functional on code (2.3%)**, all q2_K–q3_K_L of models ≤4B (**14.6%** of that subgroup), streaming fluent nonsense without crashing; tensor-for-tensor rebuilds reproduce the failure, so the recipe — not corruption — is the cause. A code imatrix fixed 0 → 106–113 of 164 HumanEval+ and solved **6–13 more tasks** than a WikiText imatrix at <4% PPL difference. PPL was not predictive: a Q3_K_M with 6× the fp16 PPL still solved 76/164. [measured, one author, preprint] | `P2` exists because of this. And: the acceptance test must be functional. |
| [ik_llama recipes discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/359) | Qwen3-30B-A3B at ~2.2 bpw: `attn=iq5_k`, first 6 layers' `ffn_down` at iq4, rest iq2 → +3.6% PPL at 10.2 GiB; +7–10% at 8.6 GiB. Two recipes beat bf16 PPL — calibration overfit. [measured, maintainer] | The mixed-precision shape that works: attention high, early `ffn_down` high. Relevant to how we allocate bits if we ever mixed types. |
| [ik_llama #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8) | LLaMA-3.1-70B needs **~1 bpw more** than LLaMA-2-70B for the same quantisation error; Q6_K error 0.1–0.15% pre-LLaMA-3 vs 0.65% on 3.1. [measured] | "Lossless bpw" is model-specific; never quote a universal floor. |
| [on-device sweep, arXiv 2505.15030](https://arxiv.org/html/2505.15030v5) | 11 models 0.5–14B: heavily quantised larger models beat smaller high-precision ones, with a threshold near **3.5 effective bpw**; 4-bit called the near-lossless operating point. [measured, preprint] | Sizing guidance, not a gate. |
| [Llama-3.1-8B ladder, arXiv 2601.14277](https://arxiv.org/html/2601.14277v1) | Q4_K_M ≈ 4-bit is within 0.3 avg points of F16; **GSM8K is the sensitive one** (77.6 → 68.3 at Q3_K_S) while HellaSwag barely moves. [measured, preprint] | Multi-step items belong in `P2`'s suite. |
| [BitNet 2B4T, arXiv 2504.12285](https://arxiv.org/html/2504.12285v2) | W1.58A8 trained at 4T tokens matches bf16 peers of its size on zero-shot averages and beats post-hoc 1.58-bit larger models — **no perplexity numbers at all**, self-reported. [claim] | Native ternary ≈ its own f16. Says nothing about PTQ-to-ternary. |
| [CAT-Q, arXiv 2606.26650](https://arxiv.org/html/2606.26650), [ScaleQ/AYOT, arXiv 2608.01078](https://arxiv.org/html/2608.01078v1) | 2026 PTQ-to-ternary: 512 calibration samples (~1M tokens) claimed to beat b1.58's 100B; and a **calibration collapse** — C4/WikiText-calibrated ternary Qwen3-4B scored zero on math/code, fixed by domain + CoT self-calibration (+8.97 over BitNet 2B4T). [measured, unreplicated] | Directly relevant to `kindi convert`: **calibrate on the domain, and on reasoning traces, or the ternary pack silently fails on code.** |

## E. Serving: speculation, KV cache, retrieval, embeddings

| source | what it says | what it implies here |
|---|---|---|
| [llama.cpp PR #5479](https://github.com/ggml-org/llama.cpp/pull/5479), [speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) | Static n-gram cache, `ngram-simple`/`ngram-map-k`/`ngram-mod` (rolling hash, **~16 MB constant**, shared across slots); real numbers 8.14 → 9.48 t/s (~1.16×) on a 70B, acceptance 28 → 49%; author: "top speedup is ~1.4×". [measured] | `P7`. Exact verification ⇒ quality-free. |
| [prompt-lookup 42× claim](https://jadidbourbaki.github.io/blog/prompt-lookup-llama-cpp/), [independent rebuild](https://modelfit.io/blog/llama-cpp-prompt-lookup-drafting-apple-silicon) | The 42× is drafting **microseconds per token**, from a tool that replays a text file and never runs a model; an independent rebuild confirmed load/RAM wins (**2658 → 159 ms**, 5× → 1× RAM) but **no throughput change**. [measured] | Do not repeat the 42×. Expect 1.0–1.5× on copy-heavy work. |
| [Chordiya, arXiv 2607.17283](https://arxiv.org/abs/2607.17283) | 1.61× best case; acceptance 69.7 → 37.8% with draft length; **3 of 5 configurations decelerate**, including via "parallel" verification that runs serially. [measured] | The CPU failure mode to watch in `P7`: verification must actually batch. |
| [Gao et al., MLSys'25](https://arxiv.org/abs/2503.24000) | KV eviction decode 1.34× at TP=1 but **prefill 0.58×**, and 0.69× decode at TP=2; compression lengthens outputs. [measured] | `P8`: eviction is a capacity tool, not a latency tool. |
| [PyramidKV, arXiv 2406.02069](https://arxiv.org/html/2406.02069v3) | Full-cache parity at 12% retention, +20.5 on TREC at 0.7%; but per-task losses of 15–20 points hide inside the average. [measured] | Layer-wise KV budgets only at 32K+ context. |
| [KV quant guide](https://sotaaz.com/post/llamacpp-kv-cache-quantization-guide-en), [KVTuner](https://arxiv.org/abs/2502.04420) | Q8_0 KV: **±0.0–0.1% PPL** at half the bytes; Q4_0 +0.5–1.1%; decode at 64K drops to 56%/53% of f16; bits belong on V, not K. [measured] | Q8_0 KV is free in quality — use for capacity, never claim a speed win. |
| [Pandey, arXiv 2603.11513](https://arxiv.org/html/2603.11513) | With **oracle** retrieval, models ≤7B fail 85–100% of questions they could not answer alone; retrieval **destroys 42–100%** of previously correct answers (p<0.001). [measured] | `P8`: do not ship retrieval as a small-model reliability upgrade. Measure known-answer retention if we do. |
| [graph-RAG comparison (n=420)](https://arxiv.org/abs/2601.07873) | Vanilla / vector / graph RAG all **96.25%**, at 1× / 1.5× / 3× cost (p ≥ 0.49). [measured] | Same. |
| [HF embedding quantization](https://github.com/huggingface/blog/blob/main/embedding-quantization.md), [Matryoshka×quant study](https://www.sciencedirect.com/science/article/pii/S0020028125005092) | int8 ≈ float32 (97%, sometimes 100%); binary 96.45% only **with rescoring**, and 74.77% on a model that does not tolerate it; 8× dimension + int8 = 32× smaller at 94.4%; **"no evidence that quantization speeds up evaluation"**. [measured] | Embedding compression buys RAM/disk, not latency. Not on the critical path here. |

---

## G. Symbolic, LUT and graph layers: cheaper, or better?

The question this section exists to answer: can a symbolic/LUT/action-graph layer on top of a small quantised model give
**better answers**, not just fewer flops? The honest split is: *selection with a verifier* is measured to help;
*representation without one* is not.

| source | what it says | what it implies here |
|---|---|---|
| Wang et al., *Grouped Sequency-arranged Rotation* (2025, arXiv 2505.03810) **[measured]** | A **grouped** rotation (GSR) before 2-bit quantisation: Llama-2-7B WikiText-2 PPL 20.29 → **11.59** at W2; rotating in sequency order is what they credit for part of it | The rotation half is real; the *sequency* half is not transferable to us — at a group-local quantiser a within-group permutation is exactly inert (§H, measured) |
| ISPASS'24 NSAI slides (Zishen Wan) **[measured]** | Accelerator architecture for neuro-symbolic AI: efficiency from putting symbolic work on dedicated hardware, benchmarked on control/games-style tasks | Nothing about CPU or language-model quality. A hardware-architecture claim, not an accuracy result |
| Wang et al. 2024, arXiv 2412.10390 **[measured]** | Graph/structured reasoning with LLMs: gains where the graph or the checker is supplied | The gain is the *checker*, not the graph: on tasks with no external verifier the reported advantage shrinks or vanishes |
| OpenReview s1I5R4b3KX **[measured]** | LUT-style / compressed table inference: throughput and memory wins on small task benchmarks | Throughput only; no LM-quality result at 1B+ scale, so it cannot price a pack change here |
| Self-consistency (Wang et al. 2023) **[replicated]** | PaLM-540B GSM8K 56.5 → **74.4** by sampling k and majority-voting; requires no new model, no graph, no index | The cheapest "better answers" mechanism that exists, and the one this plan keeps (`P9`) |
| Program-of-thought / PAL, CodeT-style test filtering **[measured]** | Executing candidate programs and filtering on tests beats free-form CoT on arithmetic and code benchmarks | A verifier is a *checker*, so it is cheap on a CPU and adds no bytes a token. It also cannot fire where nothing is checkable |
| Tsetlin machines, weightless networks (WiSARD/Randen), tries, n-gram caches **[measured, small scale]** | Real speedups on classification and pattern tasks; ternary/LUT kernels are fast on AVX2 | No published LM-quality result at our scale. Fast, not better; and "LUT backbone" is not a mechanism anyone has shown to reason |
| Oracle-retrieval experiments (see §E) **[replicated]** | With retrieval made **oracle**, models ≤7B still failed 85–100% of the questions they could not answer alone, and retrieval destroyed 42–100% of answers they had right | A knowledge graph in front of a small model is a prompt-budget cost, not a capability upgrade. This is what closes the "symbolic backbone" proposal |

What this changes: `P9` exists because of the first column's one positive row — selection over samples with a
**deterministic checker**, which is P2's `astvalid` pointed at k samples. The rest of the proposal is answered in `P8`.

One thing worth stating plainly, because it is the request this section came from: "surfs the web and matches the right
answer" is not a symbolic-layer question at all. It is tool use plus a verifier, and its failure mode is measured — the
retrieval row above. A system that answers *better* than its model does so by checking, not by representing.

---

## H. Rotation before low-bit rounding: what transfers to a 128-group ternary pack

The low-bit literature's biggest reported wins are rotation-based (QuaRot, SpinQuant, DuQuant, QuIP#, and GSR above).
Every one of them is measured inside a different quantiser: 4-bit with per-channel scales, GPTQ with error feedback, a
learned Cayley rotation. **None reports ternary weights, round-to-nearest, one fp16 scale a 128 weights, no error
feedback** — which is exactly our pack. That cell is empty in the literature, so this measures it.

`tools/rotq/probe.py`, six real matrices of `qwen3moe-control-q4.gguf` (layer 0 and 47 attention, layer 0/24/47 experts),
2 bits a weight, group 128, `absmean` and `lsq` scales, 26 selftest checks. Metric: relative Frobenius error of what the
pack decodes to.

| arm | mean vs `none` | range | worst-group error | add/token |
|---|---|---|---|---|
| `none` | 0 (0.50–0.61 abs) | — | 0.90 / 0.86 / 0.82 | 0 |
| `group-128` | −2.6% | −16.8% .. +2.1% | 0.63 / 0.65 / 0.63 | 0.09–0.72% of GEMV |
| `hadamard` (global) | −3.4% (5 matrices) | −16.4% .. +1.5% | 0.65 | 0.13–0.72% |
| `orthogonal` (dense QR) | −2.5% | −16.5% .. +2.1% | 0.61 | 25–133% |
| `zigzag-128` (DuQuant interleave) | −2.3% | −14.8% .. +1.5% | 0.73 | 0.10–0.52% |
| `group-128-sequency` | **identical** | — | identical | 0.09–0.72% |
| `signed-perm` (no rotation) | **+1.6% worse** | −0.0% .. +9.5% | 0.89 | 0 |

The shape of the result, which is more useful than the mean:

- **Where it helps is layer 0.** `blk.0.attn_q` sits at 0.6117 relative error unrotated — an outlier above the group
  scale is clipped to ±1 — and falls to 0.5089 (−16.8%). Layer 47's attention is level (−0.2%). Every expert matrix is
  **worse** (+1.5%, +1.6%, +2.1%); experts are the 82–100% of bytes that matter, and they are the part the rotation
  does not help.
- **Structure of the transform barely matters.** Dense random QR performs like a block Hadamard at up to 133% of the
  GEMV's arithmetic. What does the work is spreading a clipped element, not which orthogonal basis is used.
- **Ordering is inert, and it is provable.** `sequency_order` sorts the transform's rows — a permutation *inside* each
  scale group, so the group keeps its multiset and its scale, and the error is identical to the last digit. GSR's
  sequency credit cannot come from that at a group-local quantiser; it must come from the rotation, or from a global
  transform where groups *are* crossed.
- **A whole-group permutation is inert for the same reason** (it relabels groups, it never moves a channel between
  them). This probe had that arm, measured it as exactly equal to `group-128`, and replaced it with DuQuant's zigzag —
  which does interleave channels, and which is *slightly worse* than plain rotation here.

What this does not establish: an NLL delta. Weight error is a proxy, the probe says so in its own docstring, and the
only thing that promotes it is a paired walk. `P10` is therefore parked rather than built: a 2% weight-error change
concentrated in early attention, at an all-in cost of 0.1–0.7% of the GEMV, is not worth a conversion-path change until
a walk shows an interval that excludes zero.

---

## F. Where our own measurements agree, and where they do not

These are the cross-checks that make the appendix useful rather than decorative.

- **Super experts: reproduced, all three ids.** Relative to each expert's own layer median, we flag L2E92 (947, 2,152×),
  L1E114 (283, 927×), L3E82 (179, 525×) and **L1E68 (64, 210×)** — which is the paper's full set (L1E68, L2E92, L3E82)
  plus one more, on a different model (the Coder variant) and a different corpus. The next largest expert in the pack is
  42.7×, so the four are a separate population. Two caveats we can state precisely: our profile is output magnitude on
  384 fitting tokens, not their removal experiment; and their global threshold does not transfer here, because the layer
  median rises 150× with depth and a global 10%-of-peak cut flags 23 layer-47 experts at 1.7× their own layer.
- **Pruning's hazard is the one they name.** Their super experts carry routing mass; ours are ranked **27, 27, 38 and
  92 of 128** by mass. The 64-expert plan drops the rank-92 one, which is how a "0.07% of mass" summary hides a 210×
  outlier — the mechanism behind our own measured +0.0965 at 64.
- **Pruning over merging: reproduced in nats.** REAP's "merging > −20% at 50%" and our +1.08..+1.20 nats at 50%
  (perplexity ×~3–3.3) are the same finding measured differently.
- **Router re-fit: partially contradicted.** The literature expects the router to be a recoverable lever; our kept-router
  arm found the *pooled* router already better than the source's by 0.128 nats (CI −0.014..+0.170). Both can be true:
  their router is re-fitted, ours is copied, and a copy is not a re-fit. `P4` settles it on CPU in minutes.
- **The `+0.021`-nats ternary refit split** has no external counterpart, which is why `docs/TERNARY-OPTIMIZATION-MATRIX.md`
  reports it with its own control arms rather than comparing to a paper.
- **IMATRIX / calibration-domain effect: not yet tested here.** "Domain calibration beats generic by more than PPL
  shows" is the single most transferable warning for `kindi convert`, and our gate has no functional half yet (`P2`).
- **Rotation: the literature's magnitude does not transfer, and the reason is our quantiser.** GSR reports 20.29 →
  11.59 PPL at W2 with a grouped rotation; we measure **−2.6% weight error**, and **+1.5–2.1% worse** on the expert
  matrices that hold most of the bytes. Both are correct: their arm has GPTQ error feedback and per-channel scales, ours
  is plain RTN with one scale a 128 weights, and their win comes from the outlier-clipping mechanism we can only exploit
  where an outlier actually exceeds its group (§H). The disagreement is the interesting part — it is why `P10` is
  parked with a command attached instead of adopted.
- **Sequency ordering: contradicted, structurally.** GSR credits the sequency arrangement; at a group-local quantiser a
  within-group permutation cannot change the error at all, and the probe measures exact equality between the natural and
  sequency arms. If the effect is real there, it lives in the parts of their pipeline we do not have.
- **Symbolic/LUT backbone: no counterpart to reproduce.** Nothing in §G reports an end-task accuracy win on an assistant
  task from a symbolic representation *without* a verifier, and the retrieval experiments we do have point the other
  way. That is why the plan takes the verifier (`P9`) and refuses the backbone (`P8`) rather than the reverse.
- **AWP:** the paper (arXiv 2506.10205) is Algorithm 1 `Z = Θ + η(W − Θ)C; Θ⁺ = Proj_C(Z)` — the covariance is a preconditioner, and the method exists to *avoid* computing `C^{-1}` ("without requiring computationally-intensive operations such as second-order Hessian inverses"). The proposal's \"true AWP requires solving (X^T X + λI)^{-1}\" is false; that object belongs to OBS/GPTQ. AWP's targets are unstructured sparsity and INT4 grouped quantisation, never ternary. Our own probe (true gate inputs read from the C engine scratch, relative damping 0.01·mean diag, 256 tokens, 3 experts): ridge-project-then-ternarise is 0.557–0.583 output error against naive's 0.306–0.313. The proposal's core move is measured and loses.
- **BitDistill (arXiv 2510.13998):** a training campaign, not a converter. SubLN insertion + continual pre-training + logits/MHA distillation; validated on Qwen/Gemma **≤4B** on *classification/summarisation* tasks where the ternary student lands near (not equal to) the FP16 teacher. It does not produce lossless 1.58-bit weights from an arbitrary FP16 model, and it needs datacenter GPUs. For this CPU-only branch it is deferred, not adopted — the gate it would have to pass (paired NLL within 0.02) is the one no published ternary PTQ method passes.
- **SLQ (arXiv 2605.02404):** IST-DASLab's strongest lossless-quantisation result claims task-lossless at **3.3 bpw** and distribution-lossless at 5–6 bpw, built on GPTQ + asymmetric non-uniform quantisation. It is evidence *against* a 1.58-bit PTQ floor, not for it. What transfers: asymmetric scale+shift per group (already queued in the Schackay3 analysis as the only accuracy-free kernel project), and per-layer bit-width budgeting, which `kindi.rco` already solves exactly from measured errors.
- **DeltaPQ:** a PVLDB 2020 codec for product-quantisation *codes in similarity-search indexes*. It does not pack kernel weights; the proposal's use is a category error.
- **BitNet v2 (2504.18415):** native 4-bit activations + online Hadamard are *trained-from-scratch* machinery; PTQ-transfer at a4 is measured and worse. This engine already folds with a Hadamard (hence the P6 permutation result), and its int8-activation prototype exists (`KINDI_CPU_Q8`); a 4-bit activation kernel is a research item, not a conversion pipeline change.
- **Gemini P6b plan: partially confirmed, mostly misattributed.** The layout part is confirmed: an expert-first
  gate → up → down ordering is written by `packio.tensor_write_order` and the reader can do one positional read
  when the matrices are adjacent. The rest does not survive contact with this branch: `io_uring`/`O_DIRECT` are
  Linux-only, `MADV_WILLNEED` was already measured null/negative, and the real Qwen layer-0 fixture remains
  callback/memcpy-bound (93.52 tok/s paged vs 419.96 resident, tokens identical). The pasted AWP script is not AWP
  (wrong hook target, no activation covariance, unstructured Wanda threshold, no `Proj_C` constraint), and the
  proposed neuron permutation does not commute with this engine's Hadamard fold/sign vector. Schackay3's recipe
  does transfer two lessons — domain-matched calibration and protecting the router/trunk/starved experts — but the
  artifact is Q2_K/`qwen3next`, not a native kindi ternary source. Keep as reference, not as a conversion path.
