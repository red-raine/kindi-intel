"""The stream gate: read JSONL records, reject what must not reach a shard, count every rejection by reason.

The output is a `Report` and, optionally, a stream of accepted `(source, text)` pairs. Nothing is buffered: a 2 GB file
costs memory proportional to its duplicate set, not its size, and `--no-dedup-set` trades that for rate-only detection.

Gate order is deliberate, cheapest first, and each rejection is attributed to exactly one reason so the report sums:

  json          the line is not an object
  schema        a required field is missing, not a string, or empty after cleaning
  short/long    outside the configured character bounds
  duplicate     folded text seen before (exact); `near` counts LSH-confirmed paraphrases when enabled
  contaminated  a 13-gram of the record appears in the HumanEval/MBPP reference sets

A record rejected for schema is refused before it can cost an n-gram pass, so the contamination rate is measured against
records that were otherwise usable - which is what makes the number comparable between sources.
"""
import json
import os
import time

from . import text


class Config(object):
    def __init__(self, field="text", min_chars=32, max_chars=200000, ngram=13, near_dup=False,
                 decontam=True, require_fields=(), max_per_source=None):
        self.field = field                        # str, or a list whose values are joined
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.ngram = ngram
        self.near_dup = near_dup
        self.decontam = decontam
        self.require_fields = tuple(require_fields)
        self.max_per_source = max_per_source


class Report(object):
    REASONS = ("json", "schema", "short", "long", "duplicate", "near", "contaminated", "encoding")

    def __init__(self, source):
        self.source = source
        self.lines = 0
        self.accepted = 0
        self.accepted_bytes = 0
        self.input_bytes = 0
        self.reasons = dict((r, 0) for r in Report.REASONS)
        self.scripts = {}
        self.samples = {}
        self.seconds = 0.0
        self.char_total = 0
        self.char_count = 0
        self.char_max = 0

    def reject(self, reason, detail=None):
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        if detail is not None and reason not in self.samples:
            self.samples[reason] = detail[:300]

    @property
    def rejected(self):
        return self.lines - self.accepted

    def mean_chars(self):
        return int(self.char_total / self.char_count) if self.char_count else 0

    def as_dict(self):
        return dict(source=self.source, lines=self.lines, accepted=self.accepted, rejected=self.rejected,
                    accepted_bytes=self.accepted_bytes, input_bytes=self.input_bytes, reasons=self.reasons,
                    scripts=self.scripts, mean_chars=self.mean_chars(), max_chars=self.char_max,
                    seconds=round(self.seconds, 2),
                    samples=self.samples,
                    accept_rate=round(self.accepted / self.lines, 4) if self.lines else 0.0)


class Gate(object):
    """Streaming gate. One instance may be reused across sources; its duplicate set is shared, so duplicates across
    files are caught - which is the only way a merge of per-source files can be deduplicated."""

    def __init__(self, cfg, reference=None, log=print):
        self.cfg = cfg
        self.seen = set()
        self.band_map = {}          # LSH band -> indices into self.near
        self.near = []              # kept records' 9-word shingle sets, only when near_dup is on
        self.reference = reference
        self.log = log
        self.refs = {}              # name -> reference 13-grams, contamination only

    # -- reference sets ---------------------------------------------------------------------------------------------
    def ngram_index(self, source_name, path, field="auto"):
        """13-grams of a reference set (HumanEval/MBPP records), built once per file."""
        if source_name in self.refs:
            return self.refs[source_name]
        grams = set()
        n = self.cfg.ngram
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                for key in ("code", "prompt", "text", "canonical_solution") if field == "auto" else (field,):
                    v = rec.get(key)
                    if isinstance(v, str):
                        grams |= text.ngrams(v, n)
        self.refs[source_name] = grams
        return grams

    def set_reference(self, grams, name="reference"):
        self.refs[name] = grams

    # -- one record -------------------------------------------------------------------------------------------------
    def text_of(self, rec):
        if isinstance(self.cfg.field, (list, tuple)):
            parts = [rec.get(f) for f in self.cfg.field]
            if not all(isinstance(p, str) for p in parts):
                return None
            return "\n\n".join(parts)
        v = rec.get(self.cfg.field)
        return v if isinstance(v, str) else None

    def offer(self, rec, rep):
        """Returns the clean bytes to keep, or None after recording one rejection reason."""
        for f in self.cfg.require_fields:
            if f not in rec:
                rep.reject("schema", "missing field %r" % f)
                return None
        raw = self.text_of(rec)
        if raw is None:
            rep.reject("schema", "field %r missing or not a string" % (self.cfg.field,))
            return None
        body = text.clean(raw)
        n = len(body)
        if n < self.cfg.min_chars:
            rep.reject("short", body)
            return None
        if n > self.cfg.max_chars:
            rep.reject("long", body)
            return None
        fp = text.fingerprint(body)
        if fp in self.seen:
            rep.reject("duplicate", body)
            return None
        if self.cfg.decontam and self.refs:
            grams = text.ngrams(body, self.cfg.ngram)
            for name, ref in self.refs.items():
                if grams & ref:
                    rep.reject("contaminated", "%s: %s" % (name, body))
                    return None
        if self.cfg.near_dup:
            sig = text.minhash(body)
            shingles = text.ngrams(body, 9)
            for key in text.bands(sig):
                for idx in self.band_map.get(key, ()):
                    if text.confusion(shingles, self.near[idx]) > 0.7:
                        rep.reject("near", body)
                        return None
            self.near.append(shingles)
            for key in text.bands(sig):
                self.band_map.setdefault(key, []).append(len(self.near) - 1)
        self.seen.add(fp)
        rep.accepted += 1
        sc = text.script(body)
        rep.scripts[sc] = rep.scripts.get(sc, 0) + 1
        rep.char_total += n
        rep.char_count += 1
        rep.char_max = max(rep.char_max, n)
        payload = body.encode("utf-8", "surrogatepass")
        rep.accepted_bytes += len(payload)
        return payload

    # -- one file ---------------------------------------------------------------------------------------------------
    def run(self, path, emit=None):
        rep = Report(os.path.basename(path))
        t0 = time.perf_counter()
        rep.input_bytes = os.path.getsize(path)
        kept = 0
        with open(path, "rb") as fh:
            for lineno, raw in enumerate(fh, 1):
                rep.lines += 1
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError as e:
                    rep.reject("encoding", "line %d: %s" % (lineno, e))
                    continue
                line = line.strip()
                if not line:
                    rep.reject("json", "line %d: empty" % lineno)
                    continue
                try:
                    rec = json.loads(line)
                except ValueError as e:
                    rep.reject("json", "line %d: %s" % (lineno, e))
                    continue
                if not isinstance(rec, dict):
                    rep.reject("json", "line %d: %s, not an object" % (lineno, type(rec).__name__))
                    continue
                data = self.offer(rec, rep)
                if data is None:
                    continue
                kept += 1
                if emit is not None:
                    emit(rep.source, data)
                if self.cfg.max_per_source and kept >= self.cfg.max_per_source:
                    break
        rep.seconds = time.perf_counter() - t0
        return rep
