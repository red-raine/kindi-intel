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

## What would settle it

Re-dump the activations (the corpus pipeline above regenerates them from the pack), build trit codes at several
thresholds beside the sign code, and report ρ between their pairwise distances together with per-threshold hit1/hit5 —
that answers the ρ claim and the "strictly better at any threshold" claim on the same 48 documents. The diagonal
question needs the same corpus with the fold's diagonal varied inside the block-Hadamard family, and the zero state a
definition of what is being held fixed.
