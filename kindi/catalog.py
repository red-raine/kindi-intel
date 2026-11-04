"""The models Kindi runs: `kindi models` lists them and what fits this machine, `kindi use ID` downloads one (with
resume and a sha256 check), builds its pack and selects it.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import configparser
import hashlib
import json
import os
import sys
import urllib.request

HF = "https://huggingface.co"

# id: file, download URL (None: give the file with --gguf), sha256, size in bytes, card memory in MiB for one card
# (and for each of two cards in tensor parallel, 0 when the model does not split), and a short description
MODELS = {
    "bonsai-2b": dict(file="Ternary-Bonsai-2-2B-PTQ1_0.gguf", repo="prism-ml/Ternary-Bonsai-2-2B-gguf",
                      sha="36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa", size=429183456,
                      one=900, two=0, what="Ternary-Bonsai-2 2B (ternary weights)"),
    "bonsai-4b": dict(file="Ternary-Bonsai-2-4B-PTQ1_0.gguf", repo="prism-ml/Ternary-Bonsai-2-4B-gguf",
                      sha="56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60", size=949549920,
                      one=1900, two=0, what="Ternary-Bonsai-2 4B (ternary weights)"),
    "bonsai-9b": dict(file="Ternary-Bonsai-2-9B-PTQ1_0.gguf", repo="prism-ml/Ternary-Bonsai-2-9B-gguf",
                      sha="3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b", size=1997486976,
                      one=3800, two=0, what="Ternary-Bonsai-2 9B (ternary weights)"),
    "bonsai-27b": dict(file="Ternary-Bonsai-2-27B-PTQ1_0.gguf", repo="prism-ml/Ternary-Bonsai-2-27B-gguf",
                       sha="53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3", size=5946648928,
                       one=7600, two=3800, what="Ternary-Bonsai-2 27B (ternary weights)"),
    "nemotron-e4b": dict(file="NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q3_K_XL.gguf", repo="unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF",
                         sha="3f8032b9d2c18536c4ac51b97959a563cc27434de99823640c98147edc62ce9b", size=4219133120,
                         one=3300, two=2400, what="NVIDIA Nemotron 3.5 Bolt E4B (Gemma 4), UD-Q3_K_XL"),
}


def _cards():
    """(name, MiB) of each visible GPU; [] without a GPU"""
    try:
        import cupy as cp
        out = []
        for i in range(cp.cuda.runtime.getDeviceCount()):
            p = cp.cuda.runtime.getDeviceProperties(i)
            n = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
            out.append((n, int(p["totalGlobalMem"]) >> 20))
        return out
    except Exception:
        return []


def fits(m, cards):
    """'one card', 'two cards' or '' for this machine"""
    if any(mb >= m["one"] for _, mb in cards):
        return "one card"
    if m["two"] and len(cards) >= 2 and sorted(mb for _, mb in cards)[-2] >= m["two"]:
        return "two cards"
    return ""


def model_dir(cfg):
    return os.path.join(os.path.dirname(os.path.expanduser(cfg["pack"]).rstrip("/")) or ".", "models")


def list_models(cfg, log=print):
    cards = _cards()
    log("GPUs: %s" % (", ".join("%s (%d MiB)" % c for c in cards) or "none (the CPU engine runs the Bonsai models)"))
    d = model_dir(cfg)
    for mid, m in MODELS.items():
        have = os.path.isfile(os.path.join(d, m["file"]))
        log("  %-13s %-52s %5.1f GB  %-10s %s" % (mid, m["what"], m["size"] / 1e9, fits(m, cards) or "too large",
                                                  "downloaded" if have else ""))
    log("kindi use ID  downloads a model, builds its pack and selects it")


def content_range_start(value):
    """the first byte of a `Content-Range: bytes A-B/T` reply, or None if that is not its form

    A resume is only correct if the server's body starts where the file ends. Checking this is the difference between
    a resumed download and a file with a body written at the wrong offset - which is not a short file, it is a file
    of the wrong size whose middle does not decode, and every later measurement believes it.
    """
    if not value:
        return None
    v = value.strip()
    if not v.lower().startswith("bytes "):
        return None
    a, _, b = v[6:].split("/")[0].partition("-")
    try:
        start, end = int(a), int(b)
    except ValueError:
        return None
    return start if end >= start else None


def _download(url, path, size, sha, log, verify=True):
    """resume a partial download, then check its length and (unless `verify` is off) its sha256"""
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        log("  %s is longer than the %d bytes expected; starting it again" % (part, size))
        os.remove(part)
        have = 0
    if have < size:
        req = urllib.request.Request(url, headers={"Range": "bytes=%d-" % have} if have else {})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
            if have:
                if r.status != 206:
                    raise SystemExit("kindi: the server does not resume downloads; delete %s and retry" % part)
                at = content_range_start(r.headers.get("Content-Range"))
                if at != have:
                    raise SystemExit("kindi: asked to resume at %d but the server sent a range starting at %r; the "
                                     "body would be written at the wrong offset" % (have, at))
            got, step = have, 1 << 30
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b); got += len(b)
                if got // step != (got - len(b)) // step:
                    log("  %.1f of %.1f GB" % (got / 1e9, size / 1e9))
    wrote = os.path.getsize(part)
    if wrote != size:
        raise SystemExit("kindi: %s is %d bytes after the download, expected %d" % (part, wrote, size))
    if not verify:
        log("no sha256 in the repo's metadata: the length is checked, the bytes are not")
        os.replace(part, path)
        return
    log("checking sha256")
    h = hashlib.sha256()
    with open(part, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != sha:
        os.remove(part)
        raise SystemExit("kindi: %s has the wrong sha256; the partial file was removed, run the download again" % part)
    os.replace(part, path)


def hf_tree(repo, revision="main"):
    """the files of a Hugging Face repo, with the size and sha256 its LFS entry states.

    `?expand=true` is what makes the tree carry `lfs`, and the LFS oid *is* the sha256 of the file - so an arbitrary
    repo can be fetched and verified by the same path the catalog's own entries use, without a hand-written entry.
    Gated or private repos need a token: HF_TOKEN in the environment is sent when it is set.
    """
    url = "https://huggingface.co/api/models/%s/tree/%s?recursive=true&expand=true" % (repo, revision)
    req = urllib.request.Request(url, headers={"User-Agent": "kindi"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = json.loads(r.read().decode("utf-8"))
    out = []
    for row in rows:
        if row.get("type") != "file":
            continue
        lfs = row.get("lfs") or {}
        out.append(dict(path=row["path"], size=int(row.get("size") or lfs.get("size") or 0),
                        sha=(lfs.get("oid") or lfs.get("sha256") or None), lfs=bool(lfs)))
    return out


def fetch(spec, out_dir, log=print, revision="main", allow_unverified=False):
    """`hr:repo/path/in/repo` -> a verified local file.

    The size and sha256 come from the repo's own LFS metadata, so the download is checked against what the publisher
    stored rather than what the connection said. A file with no LFS entry (a small config, a tokenizer) is fetched
    without a sha and says so. `--allow-unverified` is for a file whose LFS entry is missing on a very large download
    where the risk is accepted; without it such a file is refused rather than written unverified.
    """
    if not spec.startswith("hf:"):
        raise SystemExit("kindi fetch: give hf:REPO/PATH (a Kaggle or HTTP URL is not supported yet)")
    # the repo id is the first two segments, the file is the rest: a repo id without an owner is not a repo id
    parts = spec[3:].split("/")
    if len(parts) < 3:
        raise SystemExit("kindi fetch: hf: needs OWNER/REPO/PATH (got %r), e.g. hf:prism-ml/"
                         "Ternary-Bonsai-2-2B-gguf/Ternary-Bonsai-2-2B-PTQ1_0.gguf" % spec)
    repo, path = "/".join(parts[:2]), "/".join(parts[2:])
    log("hf: %s, file %s (%s)" % (repo, path, revision))
    rows = hf_tree(repo, revision)
    row = next((r for r in rows if r["path"] == path), None)
    if row is None:
        near = [r["path"] for r in rows if path.split("/")[-1][:16] in r["path"]][:5]
        raise SystemExit("kindi fetch: %s is not in %s%s" % (path, repo, ("; close: " + ", ".join(near)) if near else ""))
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, os.path.basename(path))
    if os.path.isfile(dest) and (not row["sha"] or size_ok(dest, row["size"])):
        log("already present: %s" % dest)
        return dest
    if not row["sha"] and not allow_unverified:
        raise SystemExit("kindi fetch: %s has no LFS sha256 in the repo, so it cannot be verified; pass "
                         "--allow-unverified to fetch it anyway" % path)
    log("downloading %.2f GB" % (row["size"] / 1e9))
    _download("https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, path), dest, row["size"],
              row["sha"] or "", log, verify=bool(row["sha"]))
    log("ok: %s" % dest)
    return dest


def size_ok(path, size):
    return size and os.path.getsize(path) == size


def _select(cfg, pack, log):
    """write pack = PACK into the config file in use (or ~/.kindi/kindi.conf)"""
    src = cfg.get("source", "")
    path = src if src and os.path.isfile(src) else os.path.expanduser("~/.kindi/kindi.conf")
    c = configparser.ConfigParser()
    if os.path.isfile(path):
        c.read(path)
    if not c.has_section("kindi"):
        c.add_section("kindi")
    c.set("kindi", "pack", pack)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        c.write(f)
    log("selected: pack = %s in %s" % (pack, path))
    env = os.environ.get("KINDI_PACK")
    if env and os.path.abspath(os.path.expanduser(env)) != os.path.abspath(pack):
        link = os.path.expanduser(env)
        if os.path.islink(link) or not os.path.exists(link):    # the installer's layout: KINDI_PACK is a link
            tmp = link + ".new"
            if os.path.lexists(tmp):
                os.remove(tmp)
            os.symlink(pack, tmp); os.replace(tmp, link)
            log("KINDI_PACK (%s) now points to %s" % (link, pack))
        else:
            log("note: KINDI_PACK=%s is set in the environment and wins over the config file" % env)


def use(cfg, mid, gguf=None, log=print):
    if mid not in MODELS:
        raise SystemExit("kindi: unknown model %r (kindi models lists them)" % mid)
    m = MODELS[mid]
    cards = _cards()
    if cards and not fits(m, cards):
        log("note: %s needs a card with %d MiB%s; this machine may not hold it" %
            (mid, m["one"], " (or two with %d MiB each)" % m["two"] if m["two"] else ""))
    if not gguf:
        d = model_dir(cfg); os.makedirs(d, exist_ok=True)
        gguf = os.path.join(d, m["file"])
        if not os.path.isfile(gguf):
            if not m["repo"]:
                raise SystemExit("kindi: %s has no download address in this version; download %s and run "
                                 "kindi use %s --gguf FILE" % (mid, m["file"], mid))
            log("downloading %s (%.1f GB)" % (m["file"], m["size"] / 1e9))
            _download("%s/%s/resolve/main/%s" % (HF, m["repo"], m["file"]), gguf, m["size"], m["sha"], log)
    pack = os.path.join(os.path.dirname(model_dir(cfg)), "packs", mid)
    from .convert import convert
    if not os.path.isfile(os.path.join(pack, "kindi_pack.json")):
        convert(gguf, pack, log=log)
    _select(cfg, pack, log)
    log("restart kindi serve to load %s" % mid)
    return 0
