# fingerprint: the 20-NN routing index, and what it is tuned on

`docs/ROUTING-FINGERPRINT.md` reports that the sign code of the folded residual stream predicts the model's own next-token
class: 20-NN over all 2,048 bits, 0.5361 hit1 at layer 23 against 0.1772 for a global majority table. That result came
from `~/kt_route`, which is not in this tree; its corpus (`fp_bits_centred.npy`, `argmax.npy`, `top5.npy`) no longer
exists either - only four scripts survive in a temp directory. **The published numbers cannot be re-derived here.**

What can be built and checked here is the machinery that produced them, which is this module:

```bash
python -m tools.fingerprint.router --selftest           # 17 checks, known answers
python -m tools.fingerprint.router                      # tuning + baselines on a synthetic corpus
python -m tools.fingerprint.router --corpus DIR         # a ~/kt_route/corpus-style directory, if you have one
```

## The protocol, taken from the original harness

* the last position of each document is dropped - its class has no observable future
* the class is the model's argmax token; `top5` is the model's own five best tokens
* **hit1** = the argmax class is among the predicted top-5 classes; **hit5** = one of the model's top-5 tokens is
* neighbours vote into a class histogram; the prediction is the five largest bins, ties broken by class index
* distance is over the **whole** code, one matmul: bits are cast to ±1 and dotted, which is `D - 2 * Hamming`
* folds are by **document**, so a held-out document contributes no prototypes to its own neighbours
* intervals bootstrap over **documents**, because positions inside one document are not independent

## Tuning is nested, on purpose

`tune()` picks `(layer, k)` for each held-out document by running leave-one-document-out **on the other documents only**.
Choosing the layer and k by scoring every document, then reporting the winner's score, is how a routing table gets a
number nobody can reproduce: the score was fitted to the test set. Here the settings scored for a document never saw it.

The default score is **hit1**, not hit5: with 40 classes and five predictions, chance-level hit1 is 0.125 while hit5 is
above 0.5 for a table that learned nothing, so tuning on hit5 optimises mostly the base rate.

## What the synthetic corpus does and does not prove

`synth_corpus` plants the class in the bits with per-layer agreement rates, document-level drift and its own topic
distribution, so the right answers are known: the tuned layer must be the planted one, k must be interior, and the
majority and previous-token baselines must lose. All 17 checks assert those.

It does **not** say anything about the best k for real fingerprints. On the planted corpus with a clean signal
(agreement 0.92) the tuner picks k = 1 and 1-NN scores 0.9875 against 0.8750 for 20-NN - one neighbour is enough when
the prototypes are clean. On the real corpus the note measured the opposite (1-NN 0.2946, 20-NN 0.4960), because real
fingerprints are noisy and a vote suppresses that noise. The tuned k tracks signal quality, which is exactly why it has to
be tuned per corpus rather than fixed at 20.

## Measured on the synthetic corpus (10 docs, 160 positions, 128 bits, 40 classes)

| router | hit1 | 95% interval |
|---|---:|---|
| tuned (layer, k chosen per held-out doc) | 0.9875 | [0.9688, 1.0000] |
| layer 2, 1-NN (the planted signal) | 0.9875 | [0.9688, 1.0000] |
| layer 2, 20-NN | 0.8750 | [0.8313, 0.9187] |
| layer 1, 20-NN | 0.5750 | [0.5062, 0.6625] |
| layer 0, 20-NN (the 0.5 noise point) | 0.4562 | [0.3812, 0.5375] |
| previous-token table | 0.5813 | [0.5125, 0.6562] |
| global majority | 0.4875 | [0.4062, 0.5687] |

## cell-plus-NN: what the sweep says (`cell.py`)

```bash
python -m tools.fingerprint.cell --selftest      # 12 checks
python -m tools.fingerprint.cell                 # the sweep and the selected setting
```

Cell = exact match on `p` probe bits under one fixed coordinate permutation; candidates = that cell, **split on 4 more bits
whenever it exceeds the budget** rather than truncated at random, because a random drop would make the miss rate an
artifact of insertion order. Cost = mean candidates scanned x D + p.

On the named corpus the answer is a **no**, and it is the note's conclusion reached from the other side:

| setting | hit1 | retention | scanned |
|---|---:|---:|---:|
| full 20-NN | 0.8000 | 100% | 144 of 144 |
| **cheapest >=95%**: probe 12, budget 256 | 0.7875 | **98.4%** | **135.1 (93.8%)** |
| probe 16, budget 128 | 0.7562 | 94.5% | 128.0 (88.9%) |
| probe 8, budget 128 | 0.6375 | 79.7% | 86.9 (60.3%) |

Two more corpora at weaker signal land in the same place: cheapest >=95% scans **94%** (reference hit1 0.7865) and
**95%** (0.8438). So a cell index saves about 5% of a scan to keep 95% of the hit rate - not a trade worth a second data
structure.

One structural fact falls out of the table and is easy to get wrong: **the budget only binds when the probe key is wide
enough that cells exceed it.** At probe 4 a cell holds about N/16 candidates, so every budget from 16 to unbounded yields
the same 0.34 - the budget is never consulted. The frontier is therefore not "few probe bits, large budget" but "probe
wide enough that the budget is the binding constraint".

## multi-probe: the answer changes when the ball is wider (`probe.py`)

```bash
python -m tools.fingerprint.probe --selftest    # 15 checks
python -m tools.fingerprint.probe               # sweep, verdict, and the frontier across three corpora
```

A single cell can only see what it is allowed to reach: a true neighbour one key bit away sits in another cell and is
invisible at any budget, which is the structural reason the single-probe sweep needed 94% of a scan. **Multi-probe**
visits cells in increasing Hamming distance from the query key - the query's own cell, then every key one bit away, then
two, then three - nearest first, stopping when the budget is reached. Completions are sorted by code distance, so a
budget that cuts the search keeps the closest codes.

With pairs only (radius 2) the best setting under half a scan reached **92.8%** retention, and the answer read as a no.
The triples were the missing ring: at `p=8` the radius-3 ball is 93 of 256 cells, so it can be searched inside half a
scan, and that is where the bar is met.

| corpus | reference hit1 | cheapest setting keeping 95% | of a scan |
|---|---:|---|---:|
| seed 61, layer 1 | 0.870 | probe 8, radius 3, budget 96 | **37%** |
| seed 62, layer 1 | 0.932 | probe 10, radius 3, budget 48 | **17%** |
| seed 63, layer 1 | 0.849 | probe 6, radius 2, budget 96 | **38%** |
| seed 51, layer 2 (the detailed sweep) | 0.865 | probe 6, radius 3, budget 96 | 55% |

**Yes, on three of four:** 95% of the hit rate for 17-38% of a scan. The fourth is not a contradiction, it is the same
measurement on a corpus whose codes carry signal differently, and it is reported rather than dropped - 95% there costs
55%, a wedge either side of the half-scan line. What changed from `cell.py` is not the arithmetic but the search: a
single cell cannot reach a near neighbour, and three bits of probing can.

Radius 3 was silently a no-op in the first version - `probe_keys` stopped at pairs, so `radius=3` returned exactly
radius 2's cells and the table read as if a wider probe bought nothing. The `radius 3 really visits triples` check exists
so that cap cannot come back.

## the cost in wall-clock, and where the index starts to pay (`index.py`)

```bash
python -m tools.fingerprint.index --selftest    # 7 checks
python -m tools.fingerprint.index               # the crossover table: same setting, four code widths
```

A candidate count is not a clock. Gathering cells costs a `searchsorted` per probed key (93 at radius 3, `p=8`), a
concatenate and a sort; the full arm costs one matmul of `N x D`. `index.py` builds the keys once and times both arms on
the same queries, and the outcome is a **crossover**, not a win:

| corpus | N | full 20-NN | probe index | ratio | scanned |
|---|---:|---:|---:|---:|---:|
| 192 bits | 176 | 122 us | 875 us | 7.2x slower | 37% |
| 1,024 bits | 176 | 1,001 us | 1,415 us | 1.4x slower | 38% |
| 2,048 bits | 176 | 1,788 us | 1,341 us | **1.3x faster** | 37% |
| 2,048 bits | 352 | 3,798 us | 1,420 us | **2.7x faster** | 28% |

The fingerprint is 2,048 bits, so the index is on the winning side of the line the note cares about - and the win grows
with corpus size, because the fixed probe cost does not. It was still worth measuring at 192 bits, where the same idea is
eight times slower: had the check only asserted the favourable side, the regime in which the idea loses would be invisible.

The `p=0, radius 0` row is the full router reached through the index (one cell, every training row) and reproduces its
predictions exactly - which is what makes the timed comparison a comparison of two correct arms rather than of one.

## To re-derive the real numbers

The index needs fingerprints, which need a pack and a forward pass. `route_dump.py` in the temp directory is the script
that produced them; without the corpus, that is the missing piece, not this module.
