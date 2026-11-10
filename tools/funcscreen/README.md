# funcscreen: a functional acceptance screen beside the NLL gate (masterplan P2)

A converted pack can be lossless on likelihood and still write code that does not run. This screen scores what the
model *produces*, not what it predicts: 40 fixed items (`prompts.py`, sha256 recorded in every run), greedy
generation at a fixed token budget, outputs stored so a later arm is compared against the same generations.

```
python -m tools.funcscreen --pack PACK --tokens 200 --out run.json
python -m tools.funcscreen --compare a.json b.json     # paired pass-rate delta with a 95% interval over items
python -m tools.funcscreen --selftest
```

## What each item is scored on

- **Code (20 items):** a prompt prefix, a function name, and declared input/output facts. Three layers, kept apart:
  1. **parse** — the longest line-prefix of prompt + continuation that parses and defines the function. A free
     continuation does not stop at the end of the function: it repeats definitions and ends mid-expression, so
     whole-text parse punishes length, not wrongness (measured: 15 of 20 bonsai-2b continuations failed whole-text
     parse with a correct function in the first lines).
  2. **astvalid** — `tools.astvalid` over the scored prefix: TRUE / FALSE / UNKNOWN. UNKNOWN is an abstention,
     reported as such, never counted as a pass.
  3. **exec** — the one function node is executed against the declared facts. A parseable function that does the
     wrong thing fails here, which is the failure mode a parser cannot see.
- **Arithmetic (20 items):** exact-match on the first number in the continuation.

## Measured (2026-11-07)

- Selftest: 11/11 checks — the three layers on known-good / fluent-nonsense / wrong-but-parseable / self-contradictory
  outputs, the kill property, a zero-width interval for a pack against itself, and determinism on the synthetic
  fixture (random weights: plumbing only).
- Kill property on **real generations** (`prism-ml/Ternary-Bonsai-2-2B`, 40 items × 200 greedy tokens, ctx 263):
  the pack's own outputs pass 32/40 (code 16/20, arithmetic 16/20); the same outputs with tokens shuffled pass
  6/40. Paired pass-rate delta **−0.65, 95% interval [−0.80, −0.50]** over the 40 items — resolvable. The four code
  failures are genuinely wrong functions (a Celsius conversion that returns −491.7 at 0, `word_count` returning a
  list, `swap` returning its inputs, `flatten` with the wrong signature), and the four arithmetic failures are
  wrong numbers, not format misses. Run JSON: `B:/models/scratch/funcscreen-bonsai2b.json` (pre-prefix-scoring
  outputs; rescoring the stored generations with the prefix rule gives the 32/40 above).

## What is not yet measured

- **The plan's kill test — control vs the 64-way merge (+1.08 nats) — has not been run.** The merge and control
  arms on this host are GGUFs under `B:/models/scratch` (`qwen3moe-control-q4.gguf`,
  `qwen3moe-merged64-k4-q4.gguf`); no converted kindi packs for them exist here, and the engine runs kindi packs.
  The converter is the blocker, measured 2026-11-10: `kindi convert` refuses the arch by name — `expected a qwen35,
  qwen3, glm5-next, minimax-m3 or Gemma 4 GGUF (found 'qwen3moe')` — so the kill test waits on the `qwen3moe` writer
  (expert slabs + router) rather than on a choice of command. Command once a pack exists: the two `--pack` runs, then
  `--compare`. Until that run exists, every
  lossless claim in this repo is still a likelihood claim, and this screen is a *capable* gate, not yet a
  *proven* one on the arms it was built to reject.
- One pack (bonsai-2b, 2B) has been screened. No interval is quoted for its pass rates across packs; the paired
  bootstrap is over items within one comparison.
