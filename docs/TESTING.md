# Testing

One command runs everything this repository checks:

```bash
python -m tools.selftest            # 80 ok, 1 skipped, 0 failed, ~62 s
python -m tools.selftest --verbose  # per-check timing, and every skip with its reason
```

Exit status is the answer: `0` means nothing regressed, `1` means something did and each failure is printed with the
command's own last lines of output. There is no pytest dependency, no network access and no GPU: the whole gate runs on a
CPU-only host, which is the case it most needs to cover.

## What runs

| check | asserts | size |
|---|---|---|
| `imports` | every module under `kindi/` and `tools/` imports on a host with no `cupy`: the CUDA modules guard the import like the rest of the package, so an unguarded one is a failure, not a skip | 52 modules |
| `kindi.quant.selftest` | ternary quantisation and the pack's code/decode round trip | engine |
| `kindi.convert.selftest` | the converter's tensor walk and layout, and the rope-factor override: composing with the file's block instead of replacing it, yarn from the declared context when the file declares none, factor 1 as the plain table, linear kept linear, a window alone, and the factors and windows that must be refused | engine |
| `suite:rope` | the rotary table against its closed form: the plain table, position 0, the correction dims of the retired 1.7B file's own parameters (17 and 34), `mscale = 1 + 0.1 ln(factor)`, extrapolation below the low dim and interpolation above the high one, `linear` dividing every angle, and a factor of 1 as the identity | 8 checks |
| `suite:packio` | the pack's block scales at 8 bits: an exact table when a tensor has no more distinct scales than levels, a minimax grid that keeps every block inside the worst error the written levels promise, zeros (padded rows) preserved and negatives refused, a v1 fp16 pack and its v2 re-encoding loading to one value per block, and a v2 pack shorter than its index refused by name | 9 checks |
| `suite:astvalid` | three-valued validity over the labelled sample, confidence accounting | 31 checks |
| `suite:moe_loop` | the C MoE FFN against a NumPy reference of the same block: routing, top-k renormalisation, expert arithmetic, and a per-sequence halt mask ending each row at its own loop depth | 8 checks |
| `suite:loop_halt` | halting the looped block per sequence against the batch mean: the mask's work is exactly the rows' own depths, its output is the shorter loop's, and the batch rule spends loops on rows that were done | 6 checks |
| `suite:bitnet_gemm` | 2-bit packing, the reference GEMV, and the CPU kernels against it | 9 checks |
| `suite:expert_merge` | permutation alignment recovers a permuted expert to float rounding, and refuses an unalignable family instead of averaging it silently | 4 cases |
| `suite:packfetch` | the verified fetch, against a local server that misbehaves: a `Content-Range` starting at a different byte than the range asked for is refused instead of written, a body one byte shorter than its range is refused, a resume continues at the partial file's own length, a partial file longer than the target is discarded, and a wrong sha256 removes the partial file so a bad prefix cannot poison the next attempt | 8 checks |
| `suite:expert_perm` | the permutation test on experts: a planted signed permutation is recovered exactly, with the planted mapping; an exact copy is `identical` rather than `permutation`; a noisy copy and unrelated experts are neither; merging a permutation is exact after alignment and merging similar experts is lossy with and without it; and the shipped merge's unsigned alignment collapses on a signed permutation while its 0.85 gate catches it | 9 checks |
| `suite:prefetch_budget` | the paging byte/time model, and that over-fetching beats exact fetch at the proposal's own recall | table |
| `suite:cluster_budget` | expert clustering against a log whose overlap is known: a batch of like-routed tokens touching more matrices than one of them does, the batch-1 read being the distinct matrices its slots land in (k when they are all distinct, one when they all merge), one group meaning one fetch per batch so per-token bytes fall with the batch, a read that grows with the batch never breaking even while a fixed one does at B=8, a random partition covering every expert exactly once, and co-activation counting exactly the co-routed pairs | 12 checks |
| `suite:exodus` | the distillation specification's arithmetic: rotation identity and direction, ternary SQNR, top-k renormalisation against the engine, per-sequence halting masks, KL loss scaling, head slicing | 37 checks |
| `suite:datahub` | the data gate and shard writer: every rejection reason fires, dedup across files, contamination through the CLI path, shard round-trip and tamper detection, and reference coverage - a complete set, the 100 missing MBPP ids reported as ranges, duplicates that do not pad coverage, `--expect` parsing, and the CLI refusing a short reference while recording an allowed one | 47 checks |
| `suite:fingerprint` | the 20-NN routing index: distance against brute force, folds that leak nothing, a planted layer recovered by tuning, interior k, baselines that must lose, and a `--corpus` path with no corpus exiting 2 | 18 checks |
| `suite:fingerprint_cell` | cell-plus-NN: the single-cell identity against full 20-NN, a split cell respecting its budget, monotone cost, both probe regimes, and a selection rule that returns None rather than the best available | 12 checks |
| `suite:fingerprint_probe` | multi-probe cells: the Hamming ball sizes and distinct keys, radius 3 actually reaching triples, p=0 as the full router, radius 1 beating an exact cell at the same budget, the budget bound, and the 95%-under-half-a-scan result with a strict bar that returns None | 15 checks |
| `suite:fingerprint_index` | the built index: p=0 reproducing the full router exactly, whole-cell budgets, a one-bit neighbour invisible to radius 0 and found by radius 1, a batched gather that visits exactly the loop's cells, candidate ids confined to training rows, the budget shrinking as a share of a growing corpus, and the measured crossover - full 20-NN wins at 192 bits, the index wins at 1,024 and 2,048 | 10 checks |
| `suite:kvq` | the K/V cache formats: fp16 reproducing itself through the selector, a quantised cache changing the logits, the block-scale format reaching a different cache than the per-head one, the bit accounting (2.5 against 2.25), and a block scale reconstructing closer with a larger gain when the range drifts | 5 checks |
| `suite:kvq_capture` | the V-vector capture: the header is the engine's shape with every position, head and attention layer present, a record is the vector the fp16 cache holds (to fp16 precision), the cache format does not change the captured value, the record cap loads as a prefix, and a capture written by another host loads and scores | 5 checks |
| `suite:kvq_sweep` | the 48-prompt paired sweep: the stream is the prompt plus the requested tokens, two fp16 caches replaying to KL 0, both 2-bit formats diverging from fp16 and from each other, the bootstrap given answers without an engine (zero width for no difference, above zero for a constant one, covering zero when the prompt signs alternate), the same interval on a re-run, a `keep_from` that scores exactly the continuation decisions and matches the tail of the full stream, and the prompt set's 48 prompts, four a category and unique ids | 7 checks |
| `suite:bench` | the timing helper: every round kept with `min <= median <= max`, a 10x workload detected as slower, a difference smaller than the measured noise reported as noise, and zero rounds refused | 5 checks |
| `suite:nll` | the held-out NLL walk: one head evaluation per scored token, the trace in token order, every token's contribution finite and non-negative, the reported mean equal to the trace's mean, a two-token walk matching a hand computation on the engine's own buffer, an unfolded dense pack (the 1-bit Qwen3 shape) loading with no fold block, a per-head q stride read off the q matrix, `1/sqrt(head_dim)` as the softmax scale, a vocabulary that is not a multiple of 8, and a token through that path, a corpus digest blind to the path but sensitive to the text, the code digest, reference comparison including the different-corpus case, the paired mode: a pack against itself differing by exactly zero, the position bands partitioning the scored tokens with a self-pair zero inside every band, the interval containing its own mean, and a correlated series getting a wider interval when the block rather than the token is the resampled unit, a prefilled walk scoring exactly the tokens after its prefix, and a pack that asks for YaRN giving the kernel the YaRN table, embeddings-in: an injected row reproducing the token id's logits exactly and a different row giving different ones, the same step at a later position, a prefix fed as rows leaving the state a prefix of ids would, rows of the wrong width, two dimensions and a position past the context all refused, and an unfolded pack taking a row through no fold | 32 checks |
| `suite:kl` | two ternary builds priced against each other: identical files differ in nothing, a flipped code byte is one group and a few trits, a halved scale is a down-ratio with no code change, self-comparison at KL 0 and agreement 1, two packs at a positive finite KL pair, one row per scored token, a sequence longer than its context refused, and a tokeniser that gives the same ids for a dict, list or pair vocabulary while reading no added-token region as special tokens | 8 checks |
| `suite:vlut` | the vector LUT against `rows8_b2`: the kernel exact against an int64 evaluation of its own quantised input, the int8 price under 1% against the fp32 kernel, N=4 equal to two N=2 passes, the fp32 kernel's own N=4/N=2 identity, 32 weights a lookup against 16, and the int16 group sum exact at the worst-case magnitude | 7 checks |
| `cli:version`, `cli:info` | the CLI starts, prints a banner, reads config, exits 0 on a host with no pack | smoke |
| `docs:links` | every relative Markdown link in `README.md` and `docs/` resolves | 51 links, 17 files |

The two data checks are fixture-based and do not touch `B:/datasets`: the gate's arithmetic and its CLI wiring are
verified offline, while the corpus measurements live in `B:/datasets/exodus/reports/` and are not part of this suite.

The suites are the ones each tool already carried; `tools/selftest.py` runs them as subprocesses and additionally requires
their summary line to appear, so a suite that exits 0 without doing anything fails here. `imports`, `docs:links` and the
CLI smoke checks were added with the gate because nothing covered them: a module that cannot be imported, or a link in a
measurement page that points at a file that moved, is a real regression that no arithmetic test would notice.

## Skips are declared, not silent

`--verbose` prints them, and there is one: `tools.bitnet_gemm.stream_probe` is skipped because it writes about a gigabyte
to disk per run; run it directly when you are measuring paging. Nothing is skipped for want of a GPU - every module in
the tree, including the CUDA execution path (`kindi.spec`, `kindi.tp`, `kindi.chunk`, `kindi.chunk1`, `kindi.tc`),
guards its import the same way (`try: import cupy / except ImportError: cp = None`), so it loads on a CPU-only host and
is counted as imported. A module that imports `cupy` unguarded fails this check here, on the host the gate actually runs
on, rather than passing on a GPU machine and breaking for everyone else.

## What this does not cover

* **The GPU path.** Nothing here executes CUDA kernels. The five `cupy` modules are imported and no further, so the gate
  says their import-time surface loads without a card, which is not the same as their arithmetic being right.
* **Model quality.** Every check is structural or arithmetic - packing, routing, layout, link targets. None of them says
  whether output text got better or worse. The agent benchmark in the README is the tool for that, and it needs a model.
* **Long-context and paging behaviour.** Covered by `tools/bitnet_gemm/prefetch_budget.py` as a model, not as a
  measurement of a real pack on a real disk. The overlap those models take as input is measured
  (`tools/bitnet_gemm/cluster_budget.py` reads a captured router log), but the log is of one model on four documents,
  and the read and GEMV rates behind both tools are one machine's.
* **Windows-specific code paths** beyond what the CPU suite happens to exercise, and no check covers the Docker images.

## Adding a check

For a new tool, add its suite to `SUITES` in `tools/selftest.py` as `(name, argv, summary-line-to-expect)`. For a new
invariant inside an existing module, add it to that module's `selftest()` so it is covered whether or not anyone
remembers this runner. Use `tools.selftest` as the pre-commit habit rather than a CI job, since this repository has no CI.

## Why it is worth running

The first execution found two defects in code that had been reported as working: `--version` printed `kindi 0.3.0` while
every other banner printed `Kindi 0.3.0`, and one link in [CPU-PERF-2026.md](CPU-PERF-2026.md) pointed at
`kindi/kcpu.c` relative to `docs/` instead of `../kindi/kcpu.c`. Both are fixed. Neither would have been caught by any
of the four arithmetic suites, which is the argument for the two structural checks existing at all.
