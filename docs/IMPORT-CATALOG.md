# GLM-5.3-Flash and MiniMax M3: detected, priced, and packed as a catalog

`kindi convert` used to refuse both files with one sentence ("expected a qwen35, qwen3 or Gemma 4 GGUF"). It now
recognises them and writes a pack — a **catalog pack**, which holds no weights and says so in its manifest, because the
honest answer to "can Kindi take this model" is a price list and a blocker list rather than a yes or a no.

The pack is written by `kindi/archs.py`, which also owns the two architectures' metadata keys, their tensor-role map and
their byte table. `kindi recipes --for FILE` reads the same table, so detection and the catalog cannot describe one file
two ways.

## What the pack is

Two files, a few hundred KB together, whatever the source weighs:

| file | holds |
|---|---|
| `kindi_pack.json` | the manifest: `backend: "catalog"`, the arch, the geometry the header declares, the attention the tensors prove, the expert geometry, `bytes_by_group`, `total_bytes`, `import_plan`, `blockers`, `header_sha256` |
| `tensors.json` | one row a tensor, in file order: name, shape, type, bytes, role, block, `expert` |

Nothing is copied and nothing is dequantised: the source is read for its header and its tensor table only. `header_sha256`
covers the first megabyte — the header and the first tensor's opening bytes — which identifies the tensor table without
reading a 200 GB file to say the same thing.

Two fields are deliberately separate. `geometry` is what the file **declares** (`null` where it says nothing, never a
default), and `attention.observed` is what its **tensors** prove: a block carrying `attn_kv_a_mqa` is MLA whatever
`general.architecture` claims, and `linear_layers`/`softmax_layers` count the split the file actually has. A header can
be right about the width and wrong about the experts; the catalog reports both readings instead of choosing.

## The two files as they declare themselves

| | GLM-5.3-Flash (`glm5-next`) | MiniMax M3 (`minimax-m3`) |
|---|---|---|
| blocks | 45 (+ 1 MTP) | 60 |
| width / dense ffn | 4096 / 12288 | 6144 / 12288 |
| experts × expert ffn | 288 × 2048 | 128 × 3072 |
| routed / shared | 8 / 1 | 4 / 1 |
| dense lead layers | 3 | 3 |
| attention | nope-only MLA: qk_nope 256, qk_rope 0, q_lora 1536, kv_lora 512, 64 heads | GQA 64 query / 4 key-value heads, 128 dims, partial rotary 64, per-head QK norm |
| beyond plain attention | KDA linear attention on the hybrid blocks, mHC residual, DSA indexer | MSA indexer in a block of its own |
| context / vocabulary | 1,048,576 / 154,880 | 1,048,576 / 200,064 |
| MoE tensor layout | stacked 3-D slabs (`ffn_*_exps.weight`), a router row per expert, `exp_probs_b` bias | same |

Every row is a metadata key the catalog reads (`glm5-next.expert_count`, `minimax-m3.attention.kv_lora_rank`, …), so a
file that disagrees with its model card is reported as the file says. The published configuration is in the `SPEC`
entries' own descriptions, not baked into the reader as an expectation.

## Why there is no CPU pack

`kindi.cpu` implements gated-DeltaNet and full attention. Neither model's attention fits that, and the manifest's
`blockers` says which part is missing rather than "unsupported":

- **GLM-5.3-Flash** — MLA with no rotary width on the key path and a 512-wide latent; KDA linear attention on the hybrid
  blocks; an indexer that shortlists 2048 keys inside the block; an mHC residual stream (`hc_*` tensors).
- **MiniMax M3** — the MSA indexer decides which keys a query may read; rotary covers half the head with a QK-norm pair
  per head; 4 key-value heads against 64 query heads with an indexer on top.

Two blockers are derived from the file rather than curated. A K-quant source (both publish Q2_K/Q4_K) names the build's
own limit: it decodes K-quants and cannot encode them, so a ternary pack would need the float checkpoint or a Q4_0
re-encode first — the same reason both arms of the expert-merge experiment are written at Q4_0. And the size blocker is
the source's own byte count with the expert share measured from the tensor table.

## The arithmetic: the two are the same expert problem

Both models' per-layer expert weights are `3 × width × expert_ffn × experts`:

```
GLM-5.3-Flash   3 x 4096 x 2048 x 288 = 7,247,757,312
MiniMax M3      3 x 6144 x 3072 x 128 = 7,247,757,312
```

Identical to the weight. At Q2_K (84 bytes a 256) one layer's slabs are **2.38 GB**; GLM's 42 MoE layers are **99.9 GB**
and MiniMax's 57 are **135.6 GB**. At the pack's 2 bits a weight those become 76.1 GB and 103.3 GB; at the 1.667 bpw
packing in the plan, 63.4 GB and 86.1 GB.

Which is why the last blocker is paging and not quantisation: even a lossless ternary pack leaves 60-90 GB of expert
slabs to be read through a cache the engine does not have. Anything that claims to make these models runnable on a
9th-gen desktop has to answer for that number first.

## What a ternary pass would cover

`import_plan` uses the pack's own rule: a weight whose fastest dimension is a whole number of 128-weight blocks can be
re-coded. Both vocabularies are exact multiples (154,880 = 1,210 × 128, 200,064 = 1,563 × 128), as are the expert widths
(2048, 3072) and both model widths (4096, 6144), so the expert slabs, the router rows, the projections and the head all
qualify. What stays float is the tail: every norm, the router bias, and any tensor whose last dimension is not a multiple
of 128 (a 4-wide convolution state, for instance).

On the demo header below that was **202 of 637 tensors and 4.0 MB of 104.9 GB**: the float tail is not where the bytes
are, which is the first thing a conversion budget should know.

## What this does not establish

- **No real GLM-5.3-Flash or MiniMax M3 file is on this machine.** The selftest builds headers shaped like the published
  configurations (metadata keys and tensor names from llama.cpp's naming, weights absent), and the CLI demo below runs on
  such a header. A real file can differ in tensor names, in quantisation, or in which of these keys it declares.
- **The key names are the contract that could be wrong.** They are llama.cpp's naming for these groups. A file that names
  them differently is still catalogued — unmatched names land in `other` with their bytes — so the reader must check the
  `other` group rather than assume an empty one. That is why `other` is reported, not refused.
- **Nothing here ran a forward pass.** No likelihood, no gate, no claim of lossless: a catalog has no weights to measure.
  The functional screen (`tools/astvalid` + fixed prompts) is not involved and cannot be, until a kernel exists.
- The `native_status` and the curated blockers are statements about this engine at this commit; the two derived blockers
  are recomputed from each file.

## Reproduce

```
python -m kindi.archs                                  # both fixtures, header-only, 64 checks
python -m kindi recipes --for GLM-5.3-Flash-Q2_K.gguf  # detection and the plan, no conversion
python -m kindi convert GLM-5.3-Flash-Q2_K.gguf --out B:/models/packs/glm53-catalog
ls B:/models/packs/glm53-catalog                       # kindi_pack.json, tensors.json - and no weights
python -m tools.selftest                               # the repo gate, suite:archs included
```

What the demo header prints, on a file laid out like GLM-5.3-Flash's published configuration (the tensor counts are that
header's, not the model's):

```
glm5-next: GLM-5.3-Flash
  45 blocks, 288 experts x 2048 ffn (8 used, 1 shared), 3 dense lead, 154880 vocab, ctx 1048576
  637 tensors (BF16 1, F32 268, Q2_K 127, Q4_K 241) - 104.9 GB
  attention seen: 33 blocks with a linear (KDA) layer, 12 with softmax - declared nope-only MLA + KDA hybrid, ...
  a ternary pass would cover 435 tensors, 104.9 GB of 100%; it keeps 202 at f32 (4.0 MB)
  blocker: ... the source is K-quant (Q2_K/Q4_K): this build decodes K-quants but cannot encode them ...
catalog pack written to ... (no weights; 6 blocker(s))
catalog only: no weights copied - kindi_pack.json names what a kernel would have to implement
```

## Extending it to a third architecture

One entry in `kindi.archs.SPEC` (its metadata keys, its required ones, its blockers, its description) and, if the file
brings tensor groups no rule in `ROLES` knows, one rule per group. The geometry field names are the ones `kindi recipes`
already reports (`layers`, `width`, `ffn`, `heads`, `kv_heads`, `experts`, `used`, `ctx`, …), so a new family shows up in
`kindi recipes --for` and `--scan` without a second table. `kindi convert` routes any `SPEC` architecture to the catalog
by itself; a family that should become a *CPU pack* needs the kernel first, not another detection branch.

Related: [RECIPES.md](RECIPES.md) for detection and conversion plans, [QAT-CONVERSION.md](QAT-CONVERSION.md) for the
float-to-ternary path a kernel-less architecture would eventually take, [TESTING.md](TESTING.md) for the suite list.
