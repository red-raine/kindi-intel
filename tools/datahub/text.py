"""Text handling for the data gate: normalisation, fingerprints, n-grams, MinHash, script detection.

Two notions of "the same text" are needed and they must not be confused. `clean` is what gets written to a shard - it
removes what a model cannot use (NUL and control bytes, CRLF) and touches nothing else. `fold` is what gets compared -
lowercased, whitespace collapsed - and is only ever used to decide duplicate or contamination, never to train.
"""
import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(s):
    """What a byte-level model should see: NFC, LF line endings, no control bytes, no NULs."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL.sub("", s)
    return s


def fold(s):
    """The comparison form: case-folded, whitespace collapsed. Never written to a shard."""
    return _WS.sub(" ", unicodedata.normalize("NFC", s).casefold()).strip()


def fingerprint(s):
    """sha256 of the folded text: exact duplicates collide here and nowhere else."""
    return hashlib.sha256(fold(s).encode("utf-8", "surrogatepass")).hexdigest()


def words(s):
    return fold(s).split()


def ngrams(s, n=13):
    """Word n-grams of the folded text; the unit decontamination is measured in."""
    w = words(s)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def confusion(a, b):
    """Jaccard of two shingle sets, used to confirm a suspicious pair rather than to find it."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _h(token, seed):
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8", "surrogatepass"), digest_size=8,
                                          person=b"%d" % seed).digest(), "little")


def minhash(s, k=9, perms=64):
    """MinHash signature over k-word shingles. Deterministic: the same text always yields the same signature."""
    w = words(s)
    shingles = {" ".join(w[i:i + k]) for i in range(max(len(w) - k + 1, 0))} or ({" ".join(w)} if w else set())
    return tuple(min(_h(t, p) for t in shingles) for p in range(perms))


def bands(sig, rows=4):
    """LSH bands: two texts agreeing on any whole band are compared exactly. Fewer rows, more recall."""
    return [(i, sig[i * rows:(i + 1) * rows]) for i in range(len(sig) // rows)]


_SCRIPTS = (
    ("cjk", re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")),
    ("cyrillic", re.compile(r"[\u0400-\u04ff]")),
    ("arabic", re.compile(r"[\u0600-\u06ff]")),
    ("devanagari", re.compile(r"[\u0900-\u097f]")),
    ("latin", re.compile(r"[A-Za-z\u00c0-\u024f]")),
)


def script(s, sample=4000):
    """The dominant writing system of a sample, so a mixed-language pull can be bucketed without a model."""
    counts = {}
    for name, rx in _SCRIPTS:
        counts[name] = len(rx.findall(s[:sample]))
    best = max(counts, key=counts.get)
    return best if counts[best] else "none"


def pack_bytes(s):
    """UTF-8 bytes of the cleaned text: the unit a byte-level ternary model trains on."""
    return clean(s).encode("utf-8", "surrogatepass")
