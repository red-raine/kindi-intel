"""Audit data, gate it, and write byte shards. Nothing here trains anything.

    python -m tools.datahub audit  B:/datasets/stage/chunk_x.jsonl --max-records 2000
    python -m tools.datahub build  <files...> --out B:/datasets/exodus/cleaned --decontam
    python -m tools.datahub verify B:/datasets/exodus/cleaned

`audit` reads and reports without writing data. `build` applies the same gate and writes shards plus a manifest, and
exits nonzero if the gate's own hard rules fail - not if it rejects records, which is its job. `verify` re-hashes an
existing output and is the check to run before training on it.

The reference sets for decontamination are the shared ones at `B:/datasets/data-miner/decontam/reference_sets.json`
(HumanEval, MBPP); `--reference` can point elsewhere. With `--no-decontam` the gate runs without them, and the report
says so, because an unflagged corpus is not the same thing as a clean one.
"""
import argparse
import json
import os
import sys

from . import gate, shard, text

REFERENCE = r"B:\datasets\data-miner\decontam\reference_sets.json"


def _reference_grams(path, n):
    """13-grams of the HumanEval/MBPP reference records, built once."""
    grams = set()
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    recs = blob if isinstance(blob, list) else blob.get("records", [])
    for r in recs:
        if isinstance(r, dict):
            for key in ("code", "prompt", "canonical_solution"):
                v = r.get(key)
                if isinstance(v, str):
                    grams |= text.ngrams(v, n)
    return grams, len(recs)


def cmd_audit(a):
    cfg = gate.Config(field=a.field, min_chars=a.min_chars, max_chars=a.max_chars, ngram=a.ngram,
                      near_dup=a.near_dup, decontam=a.decontam, max_per_source=a.max_records)
    g = gate.Gate(cfg)
    if a.decontam:
        if not os.path.exists(a.reference):
            print("no reference sets at %s - auditing without decontamination" % a.reference)
        else:
            grams, n = _reference_grams(a.reference, a.ngram)
            g.set_reference(grams, "humaneval+mbpp")
            print("reference: %d records, %d %d-grams" % (n, len(grams), a.ngram))
    total = dict(lines=0, accepted=0, bytes=0)
    reports = []
    for path in a.files:
        rep = g.run(path)
        reports.append(rep.as_dict())
        total["lines"] += rep.lines
        total["accepted"] += rep.accepted
        total["bytes"] += rep.accepted_bytes
        r = reports[-1]
        print("%-58s %7d lines  %7d kept (%5.1f%%)  %6.1f s"
              % (os.path.basename(path), rep.lines, rep.accepted,
                 100.0 * rep.accepted / max(rep.lines, 1), rep.seconds))
        for reason, count in sorted(r["reasons"].items()):
            if count:
                print("      %-13s %7d  %s" % (reason, count, r["samples"].get(reason, "")[:90].replace("\n", " ")))
        if r["scripts"]:
            print("      scripts       %s" % r["scripts"])
    print("\ntotal: %d lines, %d kept, %.1f MB, accept rate %.1f%%"
          % (total["lines"], total["accepted"], total["bytes"] / 1e6,
             100.0 * total["accepted"] / max(total["lines"], 1)))
    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(dict(total=total, files=reports), fh, indent=1)
        print("report: %s" % a.report)
    return 0


def cmd_build(a):
    cfg = gate.Config(field=a.field, min_chars=a.min_chars, max_chars=a.max_chars, ngram=a.ngram,
                      near_dup=a.near_dup, decontam=a.decontam, max_per_source=a.max_records)
    g = gate.Gate(cfg)
    if a.decontam:
        if not os.path.exists(a.reference):
            print("FAIL: --decontam asked for but no reference sets at %s" % a.reference)
            return 2
        grams, n = _reference_grams(a.reference, a.ngram)
        g.set_reference(grams, "humaneval+mbpp")
    with shard.ShardWriter(a.out, shard_bytes=a.shard_bytes, prefix=a.prefix) as w:
        reports = []
        for path in a.files:
            rep = g.run(path, emit=lambda _src, b: w.write(b))
            reports.append(rep.as_dict())
            print("%-58s %7d lines  %7d kept" % (os.path.basename(path), rep.lines, rep.accepted))
        meta = dict(gate=dict(field=a.field, min_chars=a.min_chars, max_chars=a.max_chars, ngram=a.ngram,
                              near_dup=a.near_dup, decontam=a.decontam, reference=a.reference if a.decontam else None),
                    sources=reports)
        mpath, man = w.close(meta)
    if man["records"] == 0:
        for shard_file in man["shards"]:
            os.remove(os.path.join(a.out, shard_file["name"]))
        for extra in (mpath, os.path.join(a.out, man["index"]["name"])):
            if os.path.exists(extra):
                os.remove(extra)
        print("FAIL: nothing survived the gate; the refused shards were removed rather than left empty")
        return 2
    print("\n%d records, %.2f MB in %d shards -> %s"
          % (man["records"], man["bytes"] / 1e6, len(man["shards"]), mpath))
    return 0


def cmd_verify(a):
    ok, problems = shard.verify(a.dir, prefix=a.prefix)
    for p in problems:
        print("FAIL %s" % p)
    print("verify: %s" % ("all shards match the manifest" if ok else "%d problems" % len(problems)))
    return 0 if ok else 1


def main(argv):
    p = argparse.ArgumentParser(prog="datahub", description="Audit, gate and shard JSONL text data on CPU")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("files", nargs="+")
        sp.add_argument("--field", default="text", help="field holding the text, or a comma list to join")
        sp.add_argument("--min-chars", type=int, default=32)
        sp.add_argument("--max-chars", type=int, default=200000)
        sp.add_argument("--ngram", type=int, default=13)
        sp.add_argument("--near-dup", action="store_true", help="LSH near-duplicate detection (slower)")
        sp.add_argument("--max-records", type=int, default=None, help="per file, for a bounded run")
        sp.add_argument("--reference", default=REFERENCE)

    ap = sub.add_parser("audit")
    common(ap)
    ap.add_argument("--decontam", action="store_true", default=True)
    ap.add_argument("--no-decontam", dest="decontam", action="store_false")
    ap.add_argument("--report", default=None)
    ap.set_defaults(func=cmd_audit)

    bp = sub.add_parser("build")
    common(bp)
    bp.add_argument("--out", required=True)
    bp.add_argument("--shard-bytes", type=int, default=64 << 20)
    bp.add_argument("--prefix", default="shard")
    bp.add_argument("--decontam", action="store_true", default=True)
    bp.add_argument("--no-decontam", dest="decontam", action="store_false")
    bp.set_defaults(func=cmd_build)

    vp = sub.add_parser("verify")
    vp.add_argument("dir")
    vp.add_argument("--prefix", default="shard")
    vp.set_defaults(func=cmd_verify)

    a = p.parse_args(argv)
    field = getattr(a, "field", None)
    if isinstance(field, str) and "," in field:
        a.field = [f.strip() for f in field.split(",")]
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
