"""Build (or re-check) `gap-fill.json`: the 100 MBPP records the shared reference cache never had.

Why the gap exists is a one-line fact about splits. The deployed file's MBPP layers were built from two of MBPP's four
`full` splits:

    full/prompt        ids   1-10    10 problems    -> in no layer           (the gap)
    full/test          ids  11-510  500 problems    -> the "mbpp" layer
    full/validation    ids 511-600   90 problems    -> in no layer           (the gap)
    full/train         ids 601-974  374 problems    -> the "mbpp_sanitized" layer

So 100 of 974 MBPP problems are undiscoverable by a 13-gram gate built on that file, and the gap is exactly
`prompt + validation`, the two splits nobody read. The records are copied verbatim - `prompt := text`, `code := code`,
including the CRLF the parquet itself carries - because that is what the existing 874 entries are: byte-identical to
their parquet rows (verified, not assumed, by `--check`).

    python build.py --check            # validate the committed payload offline; needs only pyarrow
    python build.py --write            # regenerate it from the HF cache and the current reference file

`--write` records three shas in the payload: the reference file it was built against, the parquet files it read, and
the result it expects. `apply.py` will not write anything unless the first and third match, which is what makes this a
proposal rather than an edit.
"""
import argparse
import difflib
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import apply as ap                                       # noqa: E402  (the serialisation and insert rules live in one place)

PAYLOAD = os.path.join(HERE, "gap-fill.json")
DIFF = os.path.join(HERE, "reference_sets-fill.diff")
REFERENCE = r"B:\datasets\data-miner\decontam\reference_sets.json"
HF = r"B:\datasets\forge\downloads\hf\hub\datasets--google-research-datasets--mbpp\snapshots"
SPLITS = (("prompt", 1, 10), ("validation", 511, 600))


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def snapshot(root=HF):
    found = sorted(glob.glob(os.path.join(root, "*")))
    if not found:
        raise SystemExit("build: no MBPP snapshot under %s (download google-research-datasets/mbpp first)" % root)
    return found[-1]


def split_rows(snap, name):
    """Rows of one `full` split, keyed by task_id. pyarrow reads the parquet; nothing is downloaded."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("build: reading the MBPP parquet needs pyarrow")
    path = os.path.join(snap, "full", "%s-00000-of-00001.parquet" % name)
    if not os.path.exists(path):
        raise SystemExit("build: missing %s" % path)
    table = pq.read_table(path)
    return path, {int(r["task_id"]): r for r in table.to_pylist()}


def records_for(snap):
    """The 100 records, in the order they belong in the file, with the parquet they came from."""
    out, files = [], []
    for name, lo, hi in SPLITS:
        path, rows = split_rows(snap, name)
        ids = sorted(rows)
        if ids != list(range(lo, hi + 1)):
            raise SystemExit("build: %s does not hold ids %d-%d (found %d..%d)" % (name, lo, hi, ids[0], ids[-1]))
        files.append({"split": name, "file": os.path.basename(path), "sha256": sha256_file(path),
                      "ids": [lo, hi], "records": len(ids)})
        for i in ids:
            r = rows[i]
            out.append({"source": "mbpp", "task_id": "mbpp/%d" % i, "prompt": r["text"], "code": r["code"]})
    return out, files


def existing(snap):
    """The 874 MBPP entries already in the reference file, to prove the copy rule against them."""
    if not os.path.exists(REFERENCE):
        return []
    with open(REFERENCE, "rb") as f:
        ref = json.loads(f.read().decode("utf-8"))
    full_test = split_rows(snap, "test")[1]
    full_train = split_rows(snap, "train")[1]
    out = []
    for rec in ref:
        tail = str(rec.get("task_id")).rsplit("/", 1)[-1]
        if not tail.isdigit():
            continue
        i, src = int(tail), rec.get("source")
        rows = full_test if src == "mbpp" else full_train if src == "mbpp_sanitized" else None
        if rows and i in rows:
            out.append((rec, rows[i]))
    return out


def build(snap):
    recs, files = records_for(snap)
    raw = open(REFERENCE, "rb").read()
    ref = json.loads(raw.decode("utf-8"))
    filled = ap.apply_groups(ref, [{"after_task_id": "HumanEval/163", "records": recs[:10]},
                                   {"after_task_id": "mbpp/510", "records": recs[10:]}])
    payload = {
        "about": "Fill the 100 MBPP ids absent from the shared decontamination reference set: 1-10 (full/prompt) "
                 "and 511-600 (full/validation). Records are parquet rows verbatim; apply with apply.py.",
        "target": {"path": REFERENCE, "sha256": ap.sha256(raw), "bytes": len(raw), "records": len(ref)},
        "source": {"dataset": "google-research-datasets/mbpp", "config": "full",
                   "snapshot": os.path.basename(snap), "files": files,
                   "mapping": "prompt := text, code := code (verbatim, including native CRLF)"},
        "insert": [{"after_task_id": "HumanEval/163", "gap": "MBPP ids 1-10 (full/prompt split)", "ids": [1, 10],
                    "records": recs[:10]},
                   {"after_task_id": "mbpp/510", "gap": "MBPP ids 511-600 (full/validation split)", "ids": [511, 600],
                    "records": recs[10:]}],
        "expected": {"records": len(filled), "bytes": len(ap.dump(filled)), "sha256": ap.sha256(ap.dump(filled)),
                     "mbpp_ids": [1, 974], "gap_filled": len(recs)},
    }
    return payload, ref, filled, raw


def unified(before, after):
    """A reviewable diff of the two files, line endings preserved, so a human can read the 100 records."""
    a = before.decode("utf-8").splitlines(keepends=True)
    b = after.decode("utf-8").splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile="a/reference_sets.json", tofile="b/reference_sets.json",
                                        n=3)).encode("utf-8")


def check(payload, snap=None, verbose=True):
    """Assertions that do not need the reference file: shape, ids, and (when the parquet is present) verbatim copy."""
    recs = [r for g in payload["insert"] for r in g["records"]]
    ap.check_shape(recs, "payload")
    ids = sorted(int(r["task_id"].split("/")[1]) for r in recs)
    want = list(range(1, 11)) + list(range(511, 601))
    if ids != want:
        raise SystemExit("check: payload ids are not 1-10 + 511-600 (%d records)" % len(ids))
    if len({r["task_id"] for r in recs}) != len(recs):
        raise SystemExit("check: duplicate task ids in the payload")
    if payload["expected"]["gap_filled"] != len(recs):
        raise SystemExit("check: expected.gap_filled disagrees with the payload")
    if snap:
        fresh, _files = records_for(snap)
        if fresh != recs:
            raise SystemExit("check: the payload is not the parquet's current rows (rebuild with --write)")
        # the rule the new records follow, tested against the records that are already deployed
        pairs = existing(snap)
        off = [r["task_id"] for r, row in pairs if r["prompt"] != row["text"] or r["code"] != row["code"]]
        if off:
            raise SystemExit("check: %d existing records already disagree with the parquet rule (e.g. %s)"
                             % (len(off), off[0]))
        if verbose:
            print("  copy rule: %d existing MBPP records agree with the parquet verbatim" % len(pairs))
    if verbose:
        print("  payload: %d records, ids 1-10 + 511-600, unique, 4 string fields each" % len(recs))
    return True


def main(argv):
    a = argparse.ArgumentParser(description="build or check the MBPP gap-fill payload")
    a.add_argument("--write", action="store_true", help="regenerate gap-fill.json")
    a.add_argument("--check", action="store_true", help="validate gap-fill.json (default)")
    a.add_argument("--snapshot", default=None)
    args = a.parse_args(argv)

    snap = None
    try:
        snap = args.snapshot or snapshot()
    except SystemExit as e:
        if args.write:
            raise
        print("note: %s - checking the payload without the parquet" % str(e).splitlines()[0])

    if args.write:
        payload, ref, filled, raw = build(snap)
        check(payload, snap)
        with open(PAYLOAD, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        with open(DIFF, "wb") as f:
            f.write(unified(raw, ap.dump(filled)))
        print("\nwrote %s" % PAYLOAD)
        print("wrote %s" % DIFF)
        print("  target  %d records -> %d records" % (payload["target"]["records"], payload["expected"]["records"]))
        print("  result  %d bytes, sha256 %s" % (payload["expected"]["bytes"], payload["expected"]["sha256"]))
        cov = ap.coverage(filled)
        print("  mbpp    %d distinct ids, missing %s" % (cov["mbpp_distinct"], cov["mbpp_missing_1_974"] or "none"))
        return 0

    with open(PAYLOAD, encoding="utf-8") as f:
        payload = json.load(f)
    check(payload, snap)
    print("ok   gap-fill.json (%d records, sha256 %s)"
          % (payload["expected"]["gap_filled"], payload["expected"]["sha256"][:16] + "..."))
    if os.path.exists(REFERENCE):
        raw = open(REFERENCE, "rb").read()
        if ap.sha256(raw) != payload["target"]["sha256"]:
            print("note: the reference set is no longer the file this payload targets; the diff cannot be re-checked")
        else:
            filled = ap.apply_groups(json.loads(raw.decode("utf-8")), payload["insert"])
            want = unified(raw, ap.dump(filled))
            have = open(DIFF, "rb").read() if os.path.exists(DIFF) else b""
            if want != have:
                raise SystemExit("check: reference_sets-fill.diff does not match the payload and the reference set")
            lines = want.split(b"\n")
            print("ok   reference_sets-fill.diff (+%d / -%d lines)"
                  % (sum(1 for l in lines if l.startswith(b"+") and not l.startswith(b"+++")),
                     sum(1 for l in lines if l.startswith(b"-") and not l.startswith(b"---"))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
