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

## To re-derive the real numbers

The index needs fingerprints, which need a pack and a forward pass. `route_dump.py` in the temp directory is the script
that produced them; without the corpus, that is the missing piece, not this module.
