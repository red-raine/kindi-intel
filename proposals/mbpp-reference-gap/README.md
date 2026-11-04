# Proposal: fill the 100 missing MBPP ids in the shared decontamination reference set

**Status:** proposal, not applied. `B:/datasets/data-miner/decontam/reference_sets.json` is unchanged
(532,081 bytes, sha256 `44c8ae9c…`), verified after every command below. Nothing here writes into `data-miner/`.

**What it is:** 100 MBPP records - ids **1-10** and **511-600** - that the deployed reference cache never had, ready to
insert as a payload, a reviewable diff, or an applied copy. After the fill the set covers **974 of 974** MBPP ids plus
164 HumanEval, up from 874.

## Why the gap exists: two of MBPP's four `full` splits were never read

The reference file's MBPP layers are verbatim rows of the `google-research-datasets/mbpp` `full` config, split by
split - and only two splits were used:

| `full` split | task ids | problems | in the reference set |
|---|---|---|---|
| `prompt` | 1-10 | 10 | **no - the gap** |
| `test` | 11-510 | 500 | yes, as source `mbpp` |
| `validation` | 511-600 | 90 | **no - the gap** |
| `train` | 601-974 | 374 | yes, as source `mbpp_sanitized` |

So the 874 present ids are exactly `test` + `train`, and the 100 absent ones are exactly `prompt` + `validation`. The
same 100 are what the exodus report listed as undetectable (`B:/datasets/research/msg/exodus-2026-11-02.md`).

Two side findings, neither of which changes the coverage:

* The layer named `mbpp_sanitized` is actually `full/train`, not the `sanitized` config. Verbatim equality to
  `full/train` holds for all 374 rows, and `sanitized` is a different 427-record subset with a `prompt` field - none of
  its rows appear here. The label is wrong; the ids are right.
* The copy rule is exact, which is why this can be a mechanical patch: every one of the 874 existing MBPP records is
  byte-identical to its parquet row (`prompt := text`, `code := code`), including the CRLF the parquet itself carries.
  `build.py --check` re-verifies that on every run rather than trusting it once.

## What the payload adds

| insert | after | records | source |
|---|---|---|---|
| 1 | `HumanEval/163` | 10 (`mbpp/1` … `mbpp/10`) | `full/prompt` |
| 2 | `mbpp/510` | 90 (`mbpp/511` … `mbpp/600`) | `full/validation` |

Both groups land where they belong in the file's existing order, so the result reads `humaneval` →
`mbpp 1-600` → `mbpp_sanitized 601-974` with no re-sorting. Records are `{source, task_id, prompt, code}` like every
other row; nothing new is invented.

## Files

| file | what it is |
|---|---|
| [gap-fill.json](gap-fill.json) | the payload: 100 records, the three pinned shas, and the expected result |
| [reference_sets-fill.diff](reference_sets-fill.diff) | unified diff for review (+600 / -0 lines, CRLF preserved) |
| [apply.py](apply.py) | applies the payload: sha-gated, anchor-checked, verifies the result sha before writing |
| [build.py](build.py) | regenerates the payload and the diff from the parquet; `--check` re-validates them |
| [probe.py](probe.py) | the behavioural proof: planted quotations, audited against the old and filled set |

## Applying it

```bash
python build.py --check                      # 100 records, ids 1-10 + 511-600, copy rule re-verified
python apply.py --dry-run                    # coverage before/after, writes nothing
python apply.py --out /tmp/reference_sets.filled.json     # a copy to review or test with
python apply.py --in-place                   # the owner's call, on the shared file
```

`apply.py` refuses to write if the target's sha256 is not the one the payload was built against, if an anchor record is
missing, or if the result is not byte-identical to the reviewed payload. `--in-place` writes a temporary file and
renames, so an interrupted run cannot leave a truncated reference set.

The diff is an alternative for whoever prefers a patch to a script:

```bash
cd /path/to/checkout && git apply path/to/reference_sets-fill.diff
```

Tested in a scratch repo against the current file: `git apply --check` and `git apply` both exit 0 with
`core.autocrlf=false`, and the applied file is 569,183 bytes / sha256 `e6b0bb61…` - exactly the payload's `expected`.

## Evidence

Coverage, from `apply.py --dry-run`:

```
target  B:\datasets\data-miner\decontam\reference_sets.json   1038 records, 532081 bytes
result  1138 records, 569183 bytes, sha256 e6b0bb618431675a0683fa8a4bfbeced070e6fd6423878d4625f437a7c9d8ce5
        mbpp            1..600, n=600, none missing
        mbpp_sanitized  601..974, n=374, none missing
        humaneval       0..163, n=164, none missing
        mbpp distinct ids across layers: 974 of 974, missing none
```

The gap as behaviour, from `probe.py` - each row is a training-corpus record quoting a benchmark problem verbatim, run
through `tools.datahub.gate` with only the reference file changed:

| planted quotation | before | after |
|---|---|---|
| `mbpp/5` prompt (gap 1-10) | **kept** | contaminated |
| `mbpp/555` code (gap 511-600) | **kept** | contaminated |
| `mbpp/100` code (control) | contaminated | contaminated |
| unrelated prose | kept | kept |

The first two lines are the defect: on today's file a corpus can carry MBPP 511-600 verbatim and pass. The reference
index grows 26,809 → 28,469 13-grams, so the gate gets 1,660 more grams to reject on.

## Caveats worth deciding on deliberately

* **Ids 1-10 are MBPP's `prompt` split** - the few-shot examples. Adding them means a corpus that *legitimately*
  contains MBPP's own prompt examples (a few-shot-formatted training set, say) will now be rejected as contaminated.
  That is usually what decontamination wants, but it is a behaviour change and it is the only one here.
* **Three identical copies exist.** `data-miner/decontam/`, `bitqwen/data/decontam/` and `distilling/n8n/` are
  byte-identical today (same sha256), so the same patch applies to all three; a fix applied to one leaves two stale.
* **Corpora already gated are not retroactively clean.** Anything that went through the gate before this fill - e.g.
  `bitqwen/cleaned/bitqwen_v6_decontaminated.jsonl` - was checked against 874 MBPP ids, so records quoting 1-10 or
  511-600 would have survived. Re-gating is the only way to know.
* **This widens verbatim detection only.** The gate matches ≥13-word 13-grams, so a paraphrase of MBPP 555 is still
  invisible - the hub README's own stated limit, unchanged by this patch.
* **The `sanitized` config is deliberately not merged.** It is a 427-record subset of the same underlying problems with
  a different field name; merging it would duplicate ids already present rather than add coverage.

## Reproducing

`build.py --write` needs the HF cache at
`B:/datasets/forge/downloads/hf/hub/datasets--google-research-datasets--mbpp` (snapshot `4bb6404f…`, `full` config,
`prompt` and `validation` splits, parquet shas recorded in the payload) and pyarrow. `apply.py` and `probe.py` need only
the Python standard library plus this repo's `tools/datahub`.
