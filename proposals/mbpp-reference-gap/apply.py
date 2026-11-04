"""Apply the MBPP gap-fill payload to a shared decontamination reference set.

The payload (`gap-fill.json`) is 100 MBPP records that the deployed reference cache never had: ids 1-10 (MBPP's
`full/prompt` split) and 511-600 (`full/validation`). It is *applied*, not merged by hand, because the file it targets
is a 532 KB JSON list with a specific byte style - `json.dumps(indent=2, ensure_ascii=False)` with CRLF line endings
and no trailing newline - and a hand edit that changes one of those is a 532 KB diff that hides the 100 records.

    python apply.py --dry-run                      # verify, print coverage, write nothing
    python apply.py --out /tmp/reference_sets.filled.json
    python apply.py --in-place                     # the owner's call, on the shared file

Three refusals, and each is deliberate:

  - the target's sha256 must equal the one the payload was built against, so a reference set that has moved since is
    not silently half-filled;
  - every anchor (`after_task_id`) must exist, so an insert cannot land in the wrong place;
  - the result's sha256 must equal the payload's `expected.sha256`, so the applied file is the reviewed file and not a
    reconstruction of it. If that check fails, nothing is written.

`--in-place` is the only mode that writes to the target path, and it writes a temporary file then renames, so an
interrupted run cannot leave a truncated reference set behind. Nothing here touches `data-miner/` unless asked.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAYLOAD = os.path.join(HERE, "gap-fill.json")
KEYS = ("source", "task_id", "prompt", "code")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def dump(records):
    """The reference file's exact serialisation: 2-space indent, UTF-8 literals, CRLF, no trailing newline."""
    return json.dumps(records, indent=2, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8")


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return raw, json.loads(raw.decode("utf-8"))


def apply_groups(records, groups):
    """Insert each group after its anchor record. Anchors are resolved as the list is built, in payload order."""
    out = [rec for rec in records]
    for g in groups:
        anchor = g["after_task_id"]
        idx = None
        for i, rec in enumerate(out):
            if rec.get("task_id") == anchor:
                idx = i
                break
        if idx is None:
            raise SystemExit("apply: anchor %r is not in the reference set" % anchor)
        out[idx + 1:idx + 1] = g["records"]
    return out


def coverage(records):
    """Per-source id ranges and duplicates: what a reader of the file can see, computed rather than assumed."""
    by = {}
    dupes = []
    seen = set()
    for rec in records:
        src = rec.get("source")
        tid = str(rec.get("task_id"))
        k = (src, tid)
        if k in seen:
            dupes.append(tid)
        seen.add(k)
        by.setdefault(src, []).append(tid)

    def ids(src):
        out = []
        for tid in by.get(src, ()):
            tail = tid.rsplit("/", 1)[-1]
            if tail.isdigit():
                out.append(int(tail))
        return sorted(out)

    def describe(v):
        if not v:
            return "-"
        missing = sorted(set(range(v[0], v[-1] + 1)) - set(v))
        gaps = "none" if not missing else "%d missing (%d..%d)" % (len(missing), missing[0], missing[-1])
        return "%d..%d, n=%d, %s" % (v[0], v[-1], len(v), gaps)

    out = {"duplicates": dupes}
    for src in sorted(by):
        out[src] = describe(ids(src))
    mb = set(ids("mbpp")) | set(ids("mbpp_sanitized"))
    out["mbpp_distinct"] = len(mb)
    out["mbpp_covered_1_974"] = len(mb & set(range(1, 975)))
    out["mbpp_missing_1_974"] = sorted(set(range(1, 975)) - mb)
    return out


def check_shape(records, where):
    bad = [r for r in records if sorted(r) != sorted(KEYS)]
    if bad:
        raise SystemExit("%s: %d records do not have exactly the keys %s (e.g. %r)"
                         % (where, len(bad), KEYS, sorted(bad[0])))
    empty = [r for r in records if not all(isinstance(r[k], str) and r[k].strip() for k in KEYS)]
    if empty:
        raise SystemExit("%s: %d records have an empty or non-string field (e.g. %s)"
                         % (where, len(empty), empty[0].get("task_id")))


def main(argv):
    ap = argparse.ArgumentParser(description="apply the MBPP gap-fill payload to a reference set")
    ap.add_argument("--target", default=None, help="reference set to fill (default: the payload's path)")
    ap.add_argument("--payload", default=PAYLOAD)
    ap.add_argument("--out", default=None, help="where to write the filled file")
    ap.add_argument("--in-place", action="store_true", help="write to --target (atomic rename)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    with open(a.payload, encoding="utf-8") as f:
        pay = json.load(f)
    target = a.target or pay["target"]["path"]
    want = pay["target"]["sha256"]

    raw, records = load(target)
    got = sha256(raw)
    print("target  %s" % target)
    print("        %d bytes, %d records, sha256 %s" % (len(raw), len(records), got))
    if got != want:
        raise SystemExit("apply: REFUSED - the target is not the file this payload was built against\n"
                         "        expected %s\n        found    %s\n"
                         "        (a reference set that changed since 2026-11-03 needs the payload rebuilt: build.py)" % (want, got))

    new = apply_groups(records, pay["insert"])
    check_shape(new, "result")
    out_bytes = dump(new)
    res = sha256(out_bytes)

    if res != pay["expected"]["sha256"]:
        raise SystemExit("apply: REFUSED - the result is not byte-identical to the reviewed payload\n"
                         "        expected %s\n        computed %s" % (pay["expected"]["sha256"], res))

    cov = coverage(new)
    print("\nresult  %d records, %d bytes, sha256 %s" % (len(new), len(out_bytes), res))
    for src in ("mbpp", "mbpp_sanitized", "humaneval"):
        if src in cov:
            print("        %-15s %s" % (src, cov[src]))
    print("        mbpp distinct ids across layers: %d of 974, missing %s"
          % (cov["mbpp_distinct"], cov["mbpp_missing_1_974"] or "none"))
    if cov["duplicates"]:
        raise SystemExit("apply: REFUSED - duplicate task ids in the result: %r" % cov["duplicates"][:5])

    added = len(new) - len(records)
    print("        added %d records" % added)
    if a.dry_run:
        print("\ndry run: nothing written")
        return 0
    if a.in_place:
        dest = target
    elif a.out:
        dest = a.out
    else:
        raise SystemExit("apply: give --out PATH, or --in-place, or --dry-run")
    if os.path.dirname(os.path.abspath(dest)) == os.path.dirname(os.path.abspath(target)) and not a.in_place:
        print("note: writing beside the shared file; the owner's copy is still %s" % target)
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out_bytes)
    os.replace(tmp, dest)
    print("wrote   %s" % dest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
