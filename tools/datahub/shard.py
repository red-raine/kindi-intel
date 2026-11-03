"""Byte shards and the manifest that makes them auditable.

A shard is the concatenation of accepted records' UTF-8 bytes, plus an index of `(offset, length)` per record so a
training loop can cut sequences at real document boundaries instead of through the middle of one. The manifest records
the sha256 of every shard and of the index, so a corrupted copy is detected rather than trained on.

The format is deliberately the dumbest thing that works: no compression, no header, no alignment tricks. A byte-level
model needs the bytes and the boundaries; anything else is a loader's problem, not the data's.
"""
import hashlib
import json
import os


def _sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ShardWriter(object):
    def __init__(self, out_dir, shard_bytes=64 << 20, prefix="shard"):
        self.dir = out_dir
        self.limit = shard_bytes
        self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self.fh = None
        self.shards = []
        self.index = []             # (shard, offset, length)
        self.records = 0
        self._open()

    def _open(self):
        self._close_current()
        name = "%s-%05d.bin" % (self.prefix, len(self.shards))
        self.path = os.path.join(self.dir, name)
        self.fh = open(self.path, "wb")
        self.written = 0
        self.part_records = 0

    def _close_current(self):
        if self.fh is None:
            return
        self.fh.close()
        self.shards.append(dict(name=os.path.basename(self.path), bytes=self.written, records=self.part_records))
        self.fh = None

    def write(self, payload):
        if self.written and self.written + len(payload) > self.limit:
            self._open()
        self.fh.write(payload)
        self.index.append((len(self.shards), self.written, len(payload)))
        self.written += len(payload)
        self.records += 1
        self.part_records += 1

    def close(self, meta=None):
        self._close_current()
        idx_path = os.path.join(self.dir, "%s.index.jsonl" % self.prefix)
        with open(idx_path, "w", encoding="utf-8") as fh:
            for shard, off, ln in self.index:
                fh.write('{"s":%d,"o":%d,"n":%d}\n' % (shard, off, ln))
        manifest = dict(format="datahub-byte-shards/1", prefix=self.prefix, shards=self.shards,
                        records=self.records, bytes=sum(s["bytes"] for s in self.shards),
                        index=dict(name=os.path.basename(idx_path), records=len(self.index),
                                   sha256=_sha(idx_path)),
                        sha256=dict((s["name"], _sha(os.path.join(self.dir, s["name"]))) for s in self.shards))
        manifest.update(meta or {})
        mpath = os.path.join(self.dir, "%s.manifest.json" % self.prefix)
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        return mpath, manifest

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.fh is not None:
            self._close_current()


def verify(out_dir, prefix="shard"):
    """Recompute every hash in the manifest. Returns (ok, list of problems)."""
    path = os.path.join(out_dir, "%s.manifest.json" % prefix)
    if not os.path.exists(path):
        return False, ["no manifest at %s" % path]
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    problems = []
    for shard in man["shards"]:
        p = os.path.join(out_dir, shard["name"])
        if not os.path.exists(p):
            problems.append("missing shard %s" % shard["name"])
            continue
        if _sha(p) != man["sha256"][shard["name"]]:
            problems.append("sha256 mismatch: %s" % shard["name"])
        if os.path.getsize(p) != shard["bytes"]:
            problems.append("size mismatch: %s is %d, manifest says %d"
                            % (shard["name"], os.path.getsize(p), shard["bytes"]))
    idx = os.path.join(out_dir, man["index"]["name"])
    if not os.path.exists(idx):
        problems.append("missing index %s" % man["index"]["name"])
    elif _sha(idx) != man["index"]["sha256"]:
        problems.append("sha256 mismatch: %s" % man["index"]["name"])
    return (not problems), problems


def read(shard_path, offset, length):
    with open(shard_path, "rb") as fh:
        fh.seek(offset)
        return fh.read(length)
