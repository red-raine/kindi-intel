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
| `audit` | no, only a report | the reference set holds fewer benchmark ids than it claims - a low accept rate is information, a short reference is not |
| `build` | shards + manifest + index | nothing survives the gate; `--decontam` was asked for and no reference file exists; the reference set is short of its claim |
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

## Reference coverage

Decontamination can only reject what the reference contains, so a set that is short of the benchmark fails silently:
records built on a missing task are reported clean. Before a reference set's 13-grams reach the gate, its task ids are
checked against the ids it claims. `task_id` is grouped into families (`HumanEval/12` -> `humaneval`, `mbpp_s/974` ->
`mbpp`, so the MBPP layers are one id space) and each family is compared with its published size.

```
reference coverage: 1038 record(s), 2 id families
  humaneval     0-163   164 present   complete
             from humaneval 164
  mbpp          1-974   874 present   MISSING 100 of 974: 1-10, 511-600
             from mbpp 500, mbpp_sanitized 374
FAIL: mbpp: 100 of 974 ids missing (1-10, 511-600)
```

That is the shared reference file as it stands on 2026-11-03, and the gate refuses it: those 100 ids are MBPP's
`prompt` (1-10) and `validation` (511-600) splits, so a corpus can quote either verbatim and pass as uncontaminated.
`proposals/mbpp-reference-gap/` carries the fill and the behavioural proof.

* `--expect FAMILY=N` or `FAMILY=LO:HI` restates the claim, repeatably. Naming a family enforces it even when the set
  holds none of it; a default claim (HumanEval 164 from 0, MBPP 974 from 1) applies only to a family the set mentions,
  so an MBPP-only reference is not failed for holding no HumanEval.
* `--allow-short-reference` proceeds with the gap, prints a warning, and records the coverage in the manifest
  (`gate.reference_coverage`) and in the `--report` JSON, so an incomplete reference is a decision on the record.
* Duplicate ids are counted and reported but do not pad coverage, and a record whose `task_id` cannot be parsed fails
  the check: a set holding one cannot be said to hold what it claims.

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
* **The deployed reference cache is incomplete, and the coverage check above refuses it.** As of 2026-11-02
  `reference_sets.json` holds HumanEval 164 (ids 0-163), MBPP 500 (ids 11-510) and MBPP-sanitized 374 (ids 601-974);
  MBPP ids 1-10 and 511-600 are absent, so 100 of 974 problems cannot be detected. The check exists to make that a
  refusal rather than a footnote - it reports the gap, it does not close it. Reported in
  `B:/datasets/research/msg/exodus-2026-11-02.md`, filled by `proposals/mbpp-reference-gap/`.
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
