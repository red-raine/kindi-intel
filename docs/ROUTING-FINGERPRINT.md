# The activation code as a routing fingerprint

Every GEMV input in the engine is a Hadamard-folded activation, so the sign of each coordinate is already a
quantised quantity the forward pass produces for free. This note records whether that code can double as a
*fingerprint* — a cheap key for choosing an expert, a route or a cache entry without another model pass — and it keeps
one earlier conclusion separate from the measurements, because that conclusion has no artifacts in this tree.

The harness is not in this repository: it lives in `~/kt_route` and reads the pack directly (`route_dump.py`,
`route_router.py`, `route_bucket.py`, `picked_baseline.py`).

## What was measured

Corpus: 48 documents (prose and code from this repository, windows spread over the files), teacher-forced, 72 positions
each, with the model's **own argmax** as the class rather than the true next token — 3,408 scored positions and 751
classes. The fingerprint is the **sign of the residual stream in the folded basis**, one bit per coordinate, 2,048 bits
(256 bytes) per position per layer, stored with and without the corpus mean removed (`fp_bits.npy`,
`fp_bits_centred.npy`).

20-NN over the whole code, leave-one-document-out, classes scored as top-5 sets (`layer.log`):

| layer | hit1 | hit5 |
|---|---:|---:|
| 0 | 0.3480 [0.3154, 0.3812] | 0.6928 [0.6555, 0.7283] |
| 3 | 0.3894 [0.3542, 0.4269] | 0.7130 [0.6784, 0.7491] |
| 9 | 0.3680 [0.3324, 0.4046] | 0.7104 [0.6758, 0.7450] |
| 15 | 0.4560 [0.4184, 0.4950] | 0.7735 [0.7394, 0.8060] |
| 18 | 0.4595 [0.4231, 0.4956] | 0.7843 [0.7529, 0.8160] |
| 21 | 0.5041 [0.4668, 0.5429] | 0.8146 [0.7864, 0.8427] |
| 23 | **0.5361** [0.4982, 0.5757] | **0.8404** [0.8151, 0.8656] |

Against the baselines on the same folds, with the same document-level intervals (`router.log`, layers 15 and 23):

| router | hit1 | hit5 |
|---|---:|---:|
| global majority (top-5 classes) | 0.1772 [0.1529, 0.2039] | 0.6367 [0.5983, 0.6743] |
| context bigram table | 0.3407 [0.2972, 0.3891] | 0.6499 [0.6050, 0.6945] |
| 1-NN over the whole code | 0.2946 [0.2548, 0.3405] | 0.5289 [0.4839, 0.5791] |
| **20-NN over the whole code** | **0.4960** [0.4589, 0.5346] | **0.8069** [0.7785, 0.8347] |
| buckets over 8 supervised bits | 0.1638 [0.1476, 0.1802] | 0.4341 [0.4030, 0.4631] |
| buckets over 8 stability bits | 0.1088 [0.0965, 0.1211] | 0.3444 [0.3178, 0.3694] |

Paired 20-NN minus the bigram baseline: hit1 **+0.1554** [+0.1279, +0.1837], hit5 **+0.1570** [+0.1320, +0.1847] — the
fingerprint carries real information a token-level table does not, and it is worth more at layer 23 than at 15.

Two negative results are worth as much as that one:

- **Compressing the fingerprint costs more than it saves.** Indexing a table by 8 chosen coordinates, or by a cell
  codebook at k = 64/256/1024, lands at 0.29-0.35 hit1 against 0.46-0.54 for the full 20-NN, and the cheapest variant
  that keeps 20-NN inside a cell is still 0.02-0.04 behind it (`sweep.log`, with intervals in `ci.log`, both in
  `~/kt_route`). The distance over all 2,048 bits is what carries the class information.

  `tools/fingerprint/cell.py` now sweeps probe bits x candidate budget on a corpus with known answers and reaches the same
  verdict from the cost side: the cheapest setting that keeps **95%** of the full router's hit1 scans **94-95%** of the
  corpus on three separate signal strengths (probe 12, budget 256: 98.4% retention for 93.8% of the scan). Below 95% the
  curve is no kinder - 88.9% of the scan buys 94.5% of the hit rate. A cell index is not worth a second data structure
  here, which is what the sentence above says in one line and this table says in seven.

  **One correction to that verdict.** The single-probe result is a statement about a cell that cannot reach its own
  neighbours: a true neighbour one key bit away is in another cell and invisible at any budget. `tools/fingerprint/probe.py`
  searches the query's cell and its Hamming neighbours - radius 1, 2, then 3, nearest first - and at `p=8` keeps **97%** of
  the hit rate for **37%** of a scan (radius 3 is 93 of 256 cells). On three corpora with comparable reference strength the
  bar lands at 17%, 37% and 38% of a scan; on a fourth it lands at 55%, just outside half. So the compression verdict holds
  for a *hash bucket* and not for a *probe ball*: what costs the 94% is the exact-match key, not the hashing.

  In wall-clock the probe ball pays only above a width: `tools/fingerprint/index.py` times the built index against the full
  20-NN on the same queries and measures **1.8x faster at 1,024 bits, 2.5x at 2,048 and 3.4x at 2,048 with twice the
  corpus** (38%, 37%, 28% of the code scanned), while at 192 bits it is still 1.7x slower. That table is a property of the
  implementation and it moved once already: searching the probe keys one at a time cost **7.2x slower at 192 bits and 1.3x
  faster at 2,048**, and batching the 93 lookups into one `searchsorted` moved the crossover from 2,048 bits down past
  1,024. Since the fingerprint is 2,048 bits and the probes' fixed cost does not grow with N, the note's regime is the
  winning side of both versions - but the line is worth measuring rather than assuming.
- **Which bits you pick barely matters.** Random bit selection scores 0.2911 against 0.2928 for the learned selection
  at layer 15 (`ablate.log`), so the "learned" table was measuring the table, not the bit choice.

## Provenance: the rotation conclusion, unverified

An earlier session concluded the following, and it is recorded here as provenance rather than as a result:

> Fit the rotation and the zero state doesn't budge: the affordable family (block-Hadamard × diagonal) has one useful
> degree of freedom, the sign vector, which the pack already ships; a real diagonal is worth 1.6% under any sane bound;
> and everything else is a relabelling. If you want the code as a fingerprint, use the sign-only form — 1 bit per
> coordinate, ρ 0.955–0.981, strictly better than the trit at any threshold. Keep the trit where it earns its place:
> storage and reconstruction, not routing.

**None of its numbers are reproducible from this tree.** There is no script and no log for a rotation fit; all that
survives is `ham_15.npy` and `ham_23.npy`, `(3408, 3408)` position-distance matrices that nothing produced or explains,
and no trit code exists in the corpus to have been compared against. So **ρ 0.955–0.981 and "a real diagonal is worth
1.6%" are unverified**, and the "zero state doesn't budge" observation has no artifact at all.

Two parts of it do check out here:

- **The sign vector ships in the pack**, which is what makes the sign-only code free: `convert.py` writes
  `signs_2048.f32` and `signs_6144.f32` from the GGUF's `prism.hadamard.*` metadata, and `cpu.py`/`engine.py` load
  them for the 512-wide fold.
- **The sign-only form is already the form that was measured.** Every number in the tables above is a 1-bit-per-
  coordinate code, so "use the sign-only code as the fingerprint" describes the existing experiment rather than a
  change to it.

One argument in its favour, which is reasoning and not a measurement: for a Hamming or nearest-neighbour fingerprint,
thresholding signs into zeros can only erase coordinates' sign information. A trit would have to earn its extra level
by the *zero pattern* itself being class-informative, and no measurement here shows that.

## The router is now in this tree

`tools/fingerprint` implements the machinery above as a component: the full-code index, folds by document, the
`hit1`/`hit5` definitions, document-level bootstrap, the three baselines, and a `tune()` that picks `(layer, k)` for each
held-out document by running leave-one-document-out on the other documents only. 18 checks assert the parts whose answers
are known - the distance recovers brute-force agreement, a held-out document contributes no neighbours, the tuner finds a
planted layer, k is interior on a noisy corpus, and a router that ignores the bits loses to one that uses them.

It does **not** reproduce the tables above, and it cannot: the corpus those numbers came from is gone. `load_corpus()`
reads a `~/kt_route/corpus`-style directory if one reappears; until then the numbers in this note remain unverifiable from
this tree for the same reason the rotation conclusion does, and for a different reason the router itself is now
testable - the arithmetic and the protocol are checked, the measurement is not.

The sweep also exposes an ordering mistake worth not making twice: **the budget binds only when the probe key is wide
enough that cells exceed it.** At 4 probe bits a cell holds about N/16 candidates, so every budget from 16 upward gives
the same hit rate - the budget is never consulted, and the table reads as if the budget did nothing.

One thing the component already shows about method: on a planted corpus with clean codes (0.92 agreement) the tuner picks
**k = 1** and 1-NN scores 0.9875 against 0.8750 for 20-NN, while on the real corpus above 1-NN scored 0.2946 against
0.4960. The best k follows signal quality, so "20-NN" is a result about that corpus rather than a constant of the method.

## What would settle it

Re-dump the activations (the corpus pipeline above regenerates them from the pack), build trit codes at several
thresholds beside the sign code, and report ρ between their pairwise distances together with per-threshold hit1/hit5 —
that answers the ρ claim and the "strictly better at any threshold" claim on the same 48 documents. The diagonal
question needs the same corpus with the fold's diagonal varied inside the block-Hadamard family, and the zero state a
definition of what is being held fixed.
