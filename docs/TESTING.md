# Testing

One command runs everything this repository checks:

```bash
python -m tools.selftest            # 18 ok, 6 skipped, 0 failed, ~85 s
python -m tools.selftest --verbose  # per-check timing, and every skip with its reason
```

Exit status is the answer: `0` means nothing regressed, `1` means something did and each failure is printed with the
command's own last lines of output. There is no pytest dependency, no network access and no GPU: the whole gate runs on a
CPU-only host, which is the case it most needs to cover.

## What runs

| check | asserts | size |
|---|---|---|
| `imports` | every module under `kindi/` and `tools/` imports | 38 modules |
| `kindi.quant.selftest` | ternary quantisation and the pack's code/decode round trip | engine |
| `kindi.convert.selftest` | the converter's tensor walk and layout | engine |
| `suite:astvalid` | three-valued validity over the labelled sample, confidence accounting | 31 checks |
| `suite:moe_loop` | the C MoE FFN against a NumPy reference of the same block: routing, top-k renormalisation, expert arithmetic | 6 checks |
| `suite:bitnet_gemm` | 2-bit packing, the reference GEMV, and the CPU kernels against it | 9 checks |
| `suite:expert_merge` | permutation alignment recovers a permuted expert to float rounding, and refuses an unalignable family instead of averaging it silently | 4 cases |
| `suite:prefetch_budget` | the paging byte/time model, and that over-fetching beats exact fetch at the proposal's own recall | table |
| `suite:exodus` | the distillation specification's arithmetic: rotation identity and direction, ternary SQNR, top-k renormalisation against the engine, halting, KL loss scaling, head slicing | 36 checks |
| `suite:datahub` | the data gate and shard writer: every rejection reason fires, dedup across files, contamination through the CLI path, shard round-trip and tamper detection | 32 checks |
| `suite:fingerprint` | the 20-NN routing index: distance against brute force, folds that leak nothing, a planted layer recovered by tuning, interior k, baselines that must lose, and a `--corpus` path with no corpus exiting 2 | 18 checks |
| `suite:fingerprint_cell` | cell-plus-NN: the single-cell identity against full 20-NN, a split cell respecting its budget, monotone cost, both probe regimes, and a selection rule that returns None rather than the best available | 12 checks |
| `suite:fingerprint_probe` | multi-probe cells: the Hamming ball sizes and distinct keys, radius 3 actually reaching triples, p=0 as the full router, radius 1 beating an exact cell at the same budget, the budget bound, and the 95%-under-half-a-scan result with a strict bar that returns None | 15 checks |
| `suite:fingerprint_index` | the built index: p=0 reproducing the full router exactly, whole-cell budgets, a one-bit neighbour invisible to radius 0 and found by radius 1, a batched gather that visits exactly the loop's cells, candidate ids confined to training rows, the budget shrinking as a share of a growing corpus, and the measured crossover - full 20-NN wins at 192 bits, the index wins at 1,024 and 2,048 | 10 checks |
| `suite:kvq` | the K/V cache formats: fp16 reproducing itself through the selector, a quantised cache changing the logits, the block-scale format reaching a different cache than the per-head one, the bit accounting (2.5 against 2.25), and a block scale reconstructing closer with a larger gain when the range drifts | 5 checks |
| `suite:kvq_capture` | the V-vector capture: the header is the engine's shape with every position, head and attention layer present, a record is the vector the fp16 cache holds (to fp16 precision), the cache format does not change the captured value, the record cap loads as a prefix, and a capture written by another host loads and scores | 5 checks |
| `suite:bench` | the timing helper: every round kept with `min <= median <= max`, a 10x workload detected as slower, a difference smaller than the measured noise reported as noise, and zero rounds refused | 5 checks |
| `suite:nll` | the held-out NLL walk: one head evaluation per scored token, the trace in token order, every log-probability non-positive and finite, the reported mean equal to the trace's mean, a two-token walk matching a hand computation on the engine's own buffer, a corpus digest blind to the path but sensitive to the text, the code digest, and reference comparison including the different-corpus case | 11 checks |
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

`--verbose` prints them. `kindi.spec`, `kindi.tp`, `kindi.chunk`, `kindi.chunk1` and `kindi.tc` import `cupy`
unconditionally; they are the CUDA execution path and are reported as skipped on a CPU-only host rather than as passes.
The rest of the package guards the import (`try: import cupy / except ImportError: the CPU engine`), which is why the CLI
works without a GPU. `tools.bitnet_gemm.stream_probe` is skipped because it writes about a gigabyte to disk per run; run
it directly when you are measuring paging.

## What this does not cover

* **The GPU path.** Nothing here executes CUDA kernels. The five `cupy` modules are imported nowhere on a CPU host, so
  their correctness is exactly as unverified after the gate as before it.
* **Model quality.** Every check is structural or arithmetic - packing, routing, layout, link targets. None of them says
  whether output text got better or worse. The agent benchmark in the README is the tool for that, and it needs a model.
* **Long-context and paging behaviour.** Covered by `tools/bitnet_gemm/prefetch_budget.py` as a model, not as a
  measurement of a real pack on a real disk.
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
