# datahub: audit, gate and shard JSONL text on CPU

Three commands, no dependencies beyond the standard library, streaming so a 2 GB file costs memory proportional to its
duplicate set rather than its size.

```bash
python -m tools.datahub audit  <files...> --max-records 2000 [--report out.json]
python -m tools.datahub build  <files...> --out DIR [--field prompt,response] [--no-decontam]
python -m tools.datahub verify DIR --prefix shard
```

| command | writes data | exits nonzero when |
|---|---|---|
| `audit` | no, only a report | never - it is a measurement, and a low accept rate is information |
| `build` | shards + manifest + index | nothing survives the gate, or `--decontam` was asked for and no reference file exists |
| `verify` | nothing | any shard, its size, or the index disagrees with the manifest |

## What the gate rejects, and in what order

Cheapest first, and each rejection is attributed to exactly one reason so the report sums to the input:

| reason | meaning |
|---|---|
| `encoding` | the line is not valid UTF-8 (counted, not guessed) |
| `json` | not a JSON object - includes blank lines and arrays |
| `schema` | the text field is missing, not a string, or empty after cleaning |
| `short` / `long` | outside `--min-chars` / `--max-chars` |
| `duplicate` | the folded text was seen before, **including in an earlier file** |
| `near` | LSH-confirmed paraphrase, Jaccard ≥ 0.7 over 9-word shingles (`--near-dup`) |
| `contaminated` | a 13-gram of the record appears in the HumanEval/MBPP reference sets |

Order matters for interpretation: schema failures are refused before an n-gram pass, so the contamination rate is
measured against records that were otherwise usable, which is what makes it comparable between sources.

`clean` (what is written) and `fold` (what is compared) are deliberately different: the shard keeps the record's
capitalisation and punctuation, and only loses CRLF, NULs and control bytes. Duplicate and contamination decisions use
the case-folded, whitespace-collapsed form.

## Shards

`<prefix>-00000.bin` are raw concatenated UTF-8 bytes of accepted records - no header, no compression, no tokenizer. A
byte-level model needs bytes and boundaries; a tokenizer needs its own vocabulary and is not this tool's business.
`<prefix>.index.jsonl` holds `{"s":shard,"o":offset,"n":length}` per record, so sequences can be cut at document
boundaries. `<prefix>.manifest.json` holds the sha256 and byte count of every shard and of the index, plus the gate
configuration and per-source reports, so a corpus carries its own provenance.

## Limits, measured rather than implied

* **A reference shorter than `--ngram` words is invisible.** A 7-word snippet produces zero 13-grams, so nothing can match
  it. The shared MBPP/HumanEval records are long enough; a fixture that was not is how this was found.
* **The deployed reference cache is incomplete.** As of 2026-11-02 `reference_sets.json` holds HumanEval 164 (ids 0-163),
  MBPP 500 (ids 11-510) and MBPP-sanitized 374 (ids 601-974). MBPP ids 1-10 and 511-600 are absent, so ~100 of 974
  problems cannot be detected. Reported in `B:/datasets/research/msg/exodus-2026-11-02.md`.
* **`--near-dup` is O(candidates) and keeps a shingle set per accepted record in memory.** It is off by default for that
  reason; on a 2 GB corpus it is not affordable without an external LSH index.
* **No semantic checks.** Nothing here decides whether a record is *good*, only whether it is well-formed, unique,
  uncontaminated and the right language. A judge model is a different tool.
* **Byte counts, not tokens.** The report's `mean_chars` is characters; a tokenizer would give a different number.

## What it was run on

`audit` over three real sources and `build` over two produced these numbers on 2026-11-02 (see
`B:/datasets/exodus/reports/`):

| source | records | kept | note |
|---|---|---|---|
| Apsara reasoning SFT | 20,000 sampled | 20,000 | 79.9 MB, 2 shards, no contamination hits |
| `clean/eq_cleaned.jsonl` | 2,031 sampled | 2,000 | 31 exact duplicates (1.5%) inside one file |
| `local/model-zoo-d5-teach-full.jsonl` | 124 | 102 | needs `--field prompt,response`; 22 short records |
| decontamination probe | 4 | 1 | 3 planted HumanEval solutions rejected, unrelated prose kept |
