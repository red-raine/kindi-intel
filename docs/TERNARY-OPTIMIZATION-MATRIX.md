# The ternary optimisation matrix: where the remaining bits are

Three sources, read together, describe one object from three sides.

- [ternary-geometry](https://github.com/SuperInstance/ternary-geometry) supplies a metric on `{0,1,2}`: Lee distance,
  the cyclic distance on `Z/3Z`, together with Voronoi/hull tools on the 9- and 27-point grids.
- [the Julia thread on ternary diagrams](https://discourse.julialang.org/t/ternary-diagrams-and-surfaces-on-non-rectangular-grids/43423/13)
  names the space itself: a ternary diagram is a barycentric plot, so a composition `(n-, n0, n+)` is a point on the
  2-simplex and a "non-rectangular grid" is the simplex lattice.
- [ButterflyMoE (arXiv 2601.13563)](https://arxiv.org/abs/2601.13563) turns both into a compression scheme: experts are
  not independent matrices but *rotations of one shared ternary prototype*, `O(d log d)` per expert instead of
  `O(d^2)`, claimed at 80x for 8 experts and 150x at 256, with rotations trained jointly with quantisation.

This page is the matrix of levers that follow from that, with what this repository has measured on real ternary
weights sitting in the evidence column - so a row can be rejected by a number rather than by taste. All of it is on
`Ternary-Bonsai-2-2B`'s PTQ1_0 file (1.75 bits a weight: 26 bytes of base-3 codes and one fp16 scale per 128).

## The two coordinates of a block

A 128-weight block has exactly two things to say: **which** trits (a sign pattern) and **how big** (one scale). On the
simplex picture the pattern's composition `(n-, n0, n+)/128` is the barycentric point and the scale is the ray. Both
are measurable, and they behave very differently:

| quantity | measured | reading |
|---|---|---|
| zero fraction, block to block | `0.3267` mean, **std 0.0066** | the composition is essentially constant |
| zero fraction, tensor to tensor | `0.3234` - `0.3281` over the sampled tensors | constant everywhere, not just locally |
| blocks that are >90% one value | `0 of 98,304` | no block is degenerate; all sit mid-simplex |
| adjacent trits, Lee distance | `0.669` mean against `0.667` for independent trits | no local structure: the rotation has whitened it |
| two different matrices, aligned trits | `66.53%` / `66.68%` / `66.75%` differ | fully independent patterns |
| block scales | cv `0.10`-`0.23`, min/max `0.20`-`0.32` | a 3-5x range, the only thing that varies |

Two consequences, and both are negative for the obvious ideas:

**Lee distance is degenerate at three values.** On `Z/3Z`, `L(x,y) = min(|x-y|, 3-|x-y|) = 1` for every distinct pair,
so Lee distance *is* Hamming distance and the observed distances are only `{0, 1}` - `4.1M` zeros and `8.4M` ones, no
twos. The cyclic metric is real and it is the right way to think about ternary codes of length > 1, but it cannot rank
distinct ternary symbols, so there is no codeword geometry for this alphabet to exploit. A ternary codebook is
automatically a maximally separated one.

**There is no composition to quantise.** The tempting reading of the simplex picture is that a block's barycentric
point could be packed into a few bits - a per-block "zero rate" like a sparse format's metadata. It would be
measuring a constant: the zero fraction moves by less than a percentage point between tensors (`0.3234`-`0.3281`) and
has a standard deviation of `0.0066` within one. The bits are not there.

## The matrix

Columns: what the lever changes, what this repository measured, what it costs, and the verdict. Rows marked
*training* cannot be reached from a finished pack at all - they are decisions about how the model is made.

| lever | mechanism | measured here | cost | verdict |
|---|---|---|---|---|
| **alphabet width** at 2 bits | 3 levels against 4 | four levels 16% better on K and 8% on V, but the 2-bit rung stays 4000x q8_0's KL | none | **closed** - the deficit is the bit budget, not the codebook |
| **codeword geometry** (Lee/simplex codes) | choose code points by Lee distance | Lee = Hamming on `Z/3`; adjacent trits already independent (0.669) | none | **rejected** - nothing to rank |
| **composition coding** (per-block zero rate) | pack the simplex coordinate | zero fraction `0.3234`-`0.3281` across every tensor sampled | ~1-2 bits a block | **rejected** - it is a constant |
| **per-block scale** | one fp16 a 128 weights | cv 0.10-0.23, range 3-5x; Ingot re-fitted every scale, kept every trit, and won its subnet | 0.125 bit | **live** - the only degree of freedom that varies |
| **scale at fewer bits** | 8-bit scale from a per-tensor range | relL2 `3.3e-3` against the pack's own weights | saves `0.0625` bit = **3.6% of the file** | **do this first** |
| **scale sharing** (one a 256/512) | fewer, wider groups | G=2 costs `4.7e-2`, G=8 `6.3e-2` | same 3.6% at G=2 | **rejected** - 14x the error of quantising the scale for the same bytes |
| **shared prototype + rotations** (ButterflyMoE) | one ternary matrix, `O(d log d)` rotations an expert | patterns independent across matrices (66.6%) while their statistics are identical everywhere | training | **training only** - not extractable post-hoc, see below |
| **pair-code / vector LUT forms** | wider lookups per token | 1.7x at N=4 in the vector LUT, TLUT rejected on gather cost | kernel work | **measured** elsewhere |

The scale row is the whole page's payoff, so it is worth the numbers in full. On `blk.12.attn_qkv` (98,304 blocks):

| scale budget | relL2 against the current pack | bits a weight |
|---|---|---|
| fp16, one a block (today) | `0` | 1.750 |
| 12-bit | `2.0e-4` | 1.719 |
| **8-bit** | **`3.3e-3`** | **1.688** |
| 6-bit | `1.3e-2` | 1.672 |
| shared over 2 blocks | `4.7e-2` | 1.688 |
| shared over 8 blocks | `6.3e-2` | 1.641 |

8 bits for the scale is 3.6% of the file for 0.33% weight error; sharing over two blocks is the same 3.6% for 4.7%
error. If the scale is to be cut at all, cut its precision, not its density. (Both are measured against the pack's own
decoded weights, which is the right reference for a *repack* - the original fp16 teacher is not published for these
files.)

## Why the butterfly row says "training only"

ButterflyMoE's premise is exactly what the measurements show: after the Hadamard rotation the ternary pattern is
statistically the same object in every matrix (zero fraction 0.323-0.328, composition almost constant), while the
patterns themselves are independent (66.5-66.8% disagreement, the independent-ternary value). So there *is* one
prototype-shaped thing here, and everything that distinguishes one matrix from another lives in the rotation and the
scales.

But a finished pack has already spent that information: the rotation is folded into the weights, and no pair of
matrices in it shares an extractable pattern. Recovering `N` experts as rotations of one stored prototype would
require the rotations to have been *learned* against the quantiser - the paper's own point, that training the rotations
with quantisation is what stops the collapse. For a PTQ1_0 pack the honest statement is: the compression is real, and
it is a decision at training time, not a repack of a file that already exists.

That leaves the per-block scale as the lever this engine can still pull on existing artefacts, which is also the one
the two measured 27B derivatives disagree on and the one the K/V work found worth 19% KL at 2 bits.

## What to try, in order

1. **8-bit scales in `kindi convert`** - 3.6% off every pack for `3.3e-3` relL2, and the same trick applies to the
   K/V cache where two fp16 scales a head are a larger share. The measurement to run afterwards is `tools/nll.py`
   against the 2B pack: it is the first gate here that can see a whole-model quality change from a 3% file saving.
2. **Re-fit scales as a repack** - Ingot's winning move was keeping every trit and re-deriving every block scale
   against a BF16 teacher. The same optimisation can be run here without a teacher if the target is the pack's own
   decoded weights; with the F16 Bonsai file (53.8 GB, `Ternary-Bonsai-2-27B-F16.gguf`) it can be run against the
   unquantised ternary, which is the next best thing.
3. **Butterfly-style sharing** - only with a training loop, and only on an MoE whose experts are the memory problem.
   Not a repack; write it down as such.
