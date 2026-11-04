"""Fetch a large file with the resume checks that a damaged pack proved were missing.

The pack this repository measured was 11,318,112,204 bytes where its own index declares 11,258,612,896: 59,499,308
bytes too many, and the payload past byte 3,635,731,872 is uniform random. That is not a truncated download, which
would simply be short. It is a resume that appended a body to a file it had not positioned correctly - the failure
mode a range request has when nobody checks the answer:

    * a `Range: bytes=N-` request that the server honours with 206, but starting somewhere other than N;
    * a body shorter than the range promised, with the shortfall written as if it were a complete chunk;
    * a `.part` resumed from its own length after the server had already written past it.

So every chunk here is a fixed-length ranged request, and the reply is checked before a byte of it is written: the
status, the `Content-Range` start, and the exact length. A chunk that fails any of the three aborts the transfer with
the offset in the message instead of a file that measures as a model and is not one.

Two things this does that a plain downloader does not:

  * **`--repair`** keeps the readable prefix and fetches only from the first byte that does not decode, so a file with
    a bad tail costs its tail, not 11 GB.
  * **the result is verified as a model**, not only as a file: `Moepack.payload_report` must report every layer
    decoding and the size the GGUF index declares, before the download replaces anything.

    python -m tools.expert_merge.fetch --url URL --file MODEL.gguf [--size N] [--sha256 HEX]
    python -m tools.expert_merge.fetch --repair MODEL.gguf --url URL
    python -m tools.expert_merge.fetch --verify MODEL.gguf
"""
import hashlib
import os
import sys
import urllib.error
import urllib.request

from kindi.catalog import content_range_start

CHUNK = 8 << 20


def _open(url, start, end, timeout=120):
    """one ranged request; returns (stream, total size or None) after checking the reply is the range asked for"""
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end - 1)})
    r = urllib.request.urlopen(req, timeout=timeout)
    if r.status != 206:
        r.close()
        raise SystemExit("fetch: the server answered %d to a range request (not 206); refusing to guess where the "
                         "body starts" % r.status)
    got = content_range_start(r.headers.get("Content-Range"))
    if got != start:
        r.close()
        raise SystemExit("fetch: asked for a range starting at %d, the server sent one starting at %r - this is the "
                         "condition that writes a body at the wrong offset" % (start, got))
    total = None
    cr = r.headers.get("Content-Range") or ""
    if "/" in cr:
        try:
            total = int(cr.rsplit("/", 1)[1])
        except ValueError:
            total = None
    return r, total


def _read_chunk(r, expect):
    """read exactly `expect` bytes, or fail naming the shortfall"""
    buf = bytearray()
    while len(buf) < expect:
        try:
            b = r.read(min(1 << 20, expect - len(buf)))
        except Exception as e:                       # http.client raises on a body shorter than Content-Length
            b = b""
            why = " (%s)" % type(e).__name__
        else:
            why = ""
        if not b:
            raise SystemExit("fetch: the body ended after %d of %d bytes of a chunk%s; not writing a short one"
                             % (len(buf), expect, why))
        buf += b
    return bytes(buf)


def _parallel(url, path, part, size, jobs, have, sha256, log):
    """the same fetch over `jobs` connections, tracked by a one-byte-per-chunk map

    One connection to a CDN gets about a third of the bandwidth six do, and 11 GB at one connection is two hours.
    The map is what makes that safe: a chunk is marked only after its exact-length body is written at its own
    validated offset, so an interrupted run resumes the chunks it does not have and re-fetches any it was in the
    middle of writing.
    """
    import threading

    n = (size + CHUNK - 1) // CHUNK
    mpath = part + ".map"
    if os.path.exists(mpath):
        m = bytearray(open(mpath, "rb").read(n).ljust(n, b"\0"))
    else:
        m = bytearray(n)
        done = min(have // CHUNK, n)          # a single-connection run appends, so its prefix is whole chunks
        m[:done] = b"\x01" * done
        with open(mpath, "wb") as f:
            f.write(bytes(m))
    if not os.path.exists(part):
        with open(part, "wb") as f:
            f.truncate(size)
    elif os.path.getsize(part) < size:
        with open(part, "r+b") as f:
            f.truncate(size)
    todo = [i for i in range(n) if not m[i]]
    log("  %d of %d chunks to fetch over %d connection(s)" % (len(todo), n, jobs))
    cursor = {"k": 0}
    lock = threading.Lock()
    errors = []

    def worker():
        fh = open(part, "r+b")
        mp = open(mpath, "r+b")
        try:
            while True:
                with lock:
                    k = cursor["k"]
                    if k >= len(todo):
                        return
                    cursor["k"] += 1
                    i = todo[k]
                start, end = i * CHUNK, min((i + 1) * CHUNK, size)
                try:
                    r, total = _open(url, start, end)
                    if total is not None and total != size:
                        r.close()
                        raise SystemExit("fetch: the file is %d bytes at this URL, not the %d expected" % (total, size))
                    body = _read_chunk(r, end - start)
                    r.close()
                    fh.seek(start)
                    fh.write(body)
                    fh.flush()
                    mp.seek(i)
                    mp.write(b"\x01")
                    mp.flush()
                except BaseException as e:
                    with lock:
                        errors.append((start, e))
                    return
        finally:
            fh.close()
            mp.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(jobs, max(1, len(todo))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        start, e = errors[0]
        raise SystemExit("fetch: chunk at byte %d failed: %s (run it again to resume)" % (start, e))
    with open(mpath, "wb") as f:
        f.write(b"\x01" * n)
    return _finish(path, part, size, sha256, log)


def _finish(path, part, size, sha256, log):
    """length, hash, and that it decodes as a model, before anything replaces the target"""
    got = os.path.getsize(part)
    if got != size:
        raise SystemExit("fetch: wrote %d bytes, expected %d" % (got, size))
    if sha256:
        log("  checking sha256")
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        if h.hexdigest() != sha256.lower():
            for p in (part, part + ".map"):
                if os.path.exists(p):
                    os.remove(p)              # a bad prefix must not survive to poison the next resume
            raise SystemExit("fetch: sha256 is %s, expected %s (the partial file was removed)"
                             % (h.hexdigest(), sha256.lower()))
    os.replace(part, path)
    if os.path.exists(part + ".map"):
        os.remove(part + ".map")
    return verify(path, log=log)


def download(url, path, size=None, sha256=None, jobs=1, log=print):
    """write `path` from `url`, resuming a `.part`, then verify size, hash and that it decodes as a model"""
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if size is not None and have > size:
        log("  %s is longer than the %d bytes expected; discarding it" % (part, size))
        os.remove(part)
        have = 0
    if size is None:
        # a one-byte range is enough to learn the length without trusting a HEAD
        r, total = _open(url, 0, 1)
        r.close()
        if total is None:
            raise SystemExit("fetch: the server did not report a total size; pass --size")
        size = total
    log("  %s: %d of %d bytes present (%.1f%%)" % (path, have, size, 100.0 * have / max(size, 1)))
    if jobs > 1:
        return _parallel(url, path, part, size, jobs, have, sha256, log)
    pos = have
    with open(part, "ab" if have else "wb") as f:
        while pos < size:
            end = min(pos + CHUNK, size)
            r, total = _open(url, pos, end)
            if total is not None and total != size:
                r.close()
                raise SystemExit("fetch: the file is %d bytes at this URL, not the %d expected" % (total, size))
            body = _read_chunk(r, end - pos)
            r.close()
            f.write(body)
            pos = end
            if pos // (1 << 30) != (pos - len(body)) // (1 << 30):
                log("  %.1f of %.1f GB" % (pos / 1e9, size / 1e9))
    return _finish(path, part, size, sha256, log)


def verify(path, log=print):
    """is this a complete pack: the size its index declares, and every layer decoding"""
    from .ggufmoe import Moepack
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        size = os.path.getsize(path)
        log("  %-58s %d bytes, not a GGUF (size checked only)" % (os.path.basename(path), size))
        return dict(ok=True, gguf=False, size=size, path=path, first_bad=None, break_offset=None,
                    layers_ok=0, extra=0)
    pk = Moepack(path)
    rep = pk.payload_report()
    rep["path"] = path
    if not rep["ok"]:
        log("  %-58s NOT A COMPLETE PACK" % os.path.basename(path))
        log("    fails at %s (byte %d of %d); %d of %d layers decode"
            % (rep["first_bad"], rep["break_offset"], rep["size"], rep["layers_ok"], pk.n_layer))
        return rep
    log("  %-58s %d bytes, %d of %d layers decode%s"
        % (os.path.basename(path), rep["size"], rep["layers_ok"], pk.n_layer,
           "" if not rep["extra"] else ", %d bytes past the index" % rep["extra"]))
    return rep


def repair(url, path, size=None, sha256=None, jobs=1, log=print):
    """keep the decodable prefix of `path` and re-fetch from the first byte that does not decode"""
    from .ggufmoe import Moepack
    rep = verify(path, log=log)
    if rep["ok"]:
        return rep
    have = os.path.getsize(path)
    if size is None:
        size = rep["declared"]
    if have > size:
        log("  the file is %d bytes where the index declares %d; truncating before the repair"
            % (have, size))
        with open(path, "r+b") as f:
            f.truncate(min(have, size))
    cut = rep["break_offset"] or 0
    # back up one chunk so the resumed download re-fetches the boundary; a chunk boundary is what a bad resume
    # wrote at, and the byte that tests bad is not necessarily the byte it started at.
    cut = max(0, (cut // CHUNK) * CHUNK)
    log("  keeping the first %.3f GB and fetching the rest from byte %d" % (cut / 1e9, cut))
    part = path + ".part"
    if os.path.exists(part):
        os.remove(part)
    with open(part, "wb") as f:
        with open(path, "rb") as src:
            left = cut
            while left:
                b = src.read(min(1 << 24, left))
                if not b:
                    break
                f.write(b)
                left -= len(b)
    return download(url, path, size=size, sha256=sha256, jobs=jobs, log=log)


# --- self-test: a server that misbehaves, in the three ways that matter ---------------------------------------

def _serve(payload, mode):
    """serve `payload` over loopback; mode 'ok' answers ranges, 'shift' lies about where a range starts, 'short'
    stops one byte early - the two replies that turn a resumed download into a file nobody can trust"""
    import http.server
    import socketserver
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            total = len(payload)
            try:
                a, b = self.headers.get("Range", "").split("=")[1].split("-")
                start, end = int(a), int(b)
            except Exception:
                start, end = 0, total - 1
            start = max(0, min(start, total - 1))
            end = max(start, min(end, total - 1))
            body = payload[start:end + 1]
            if mode == "shift":
                start = start - 1 if start else start + 1
            elif mode == "short":
                body = body[:-1]
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, total))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/pack.bin" % srv.server_address[1]


def selftest(log=print):
    import hashlib
    import shutil
    import tempfile

    import numpy as np

    nchk = 0

    def check(name, cond, detail):
        nonlocal nchk
        nchk += 1
        if not cond:
            raise AssertionError("%s: %s" % (name, detail))
        log("  ok   %-58s %s" % (name, detail))

    check("content-range start is read from a 206 reply", content_range_start("bytes 1024-2047/4096") == 1024,
          "bytes 1024-2047/4096 -> 1024")
    check("a range header that is not a range is refused",
          content_range_start(None) is None and content_range_start("bytes */4096") is None
          and content_range_start("items 1-2/3") is None and content_range_start("bytes 9-4/9") is None,
          "None, */total, other units and an inverted range all give None")

    tmp = tempfile.mkdtemp(prefix="packfetch")
    try:
        payload = np.random.default_rng(0).integers(0, 256, 2 << 20, dtype=np.uint8).tobytes()
        sha = hashlib.sha256(payload).hexdigest()
        for mode in ("ok", "shift", "short"):
            srv, url = _serve(payload, mode)
            path = os.path.join(tmp, mode + ".bin")
            try:
                rep = download(url, path, size=len(payload), sha256=sha, log=lambda *_a: None)
                if mode == "ok":
                    check("a correct range download is byte-exact", open(path, "rb").read() == payload,
                          "%d bytes, sha256 checked" % len(payload))
                else:
                    check("a %r reply is refused, not written" % mode, False, "download succeeded anyway")
            except SystemExit as e:
                msg = str(e)
                want = "wrong offset" if mode == "shift" else "body ended"
                check("a %r reply is refused, not written" % mode, want in msg and not os.path.exists(path),
                      msg.splitlines()[0][:96])
            finally:
                srv.shutdown()
                srv.server_close()

        # resuming: a .part holding the true prefix must continue at its own length and verify
        srv, url = _serve(payload, "ok")
        path = os.path.join(tmp, "resume.bin")
        with open(path + ".part", "wb") as f:
            f.write(payload[: (1 << 20)])
        download(url, path, size=len(payload), sha256=sha, log=lambda *_a: None)
        srv.shutdown(); srv.server_close()
        check("a resume continues at the partial file's own length", open(path, "rb").read() == payload,
              "1 MiB kept, %d bytes fetched" % (len(payload) - (1 << 20)))

        # parallel: several chunks written at their own offsets, then the map removed
        big = np.random.default_rng(1).integers(0, 256, 20 << 20, dtype=np.uint8).tobytes()
        big_sha = hashlib.sha256(big).hexdigest()
        srv, url = _serve(big, "ok")
        path = os.path.join(tmp, "par.bin")
        download(url, path, size=len(big), sha256=big_sha, jobs=4, log=lambda *_a: None)
        srv.shutdown(); srv.server_close()
        check("a parallel fetch writes every chunk at its own offset",
              open(path, "rb").read() == big and not os.path.exists(path + ".part.map"),
              "%d MiB over 4 connections, map cleaned up" % (len(big) >> 20))

        # ... and a .part that is longer than the file is discarded rather than appended to
        srv, url = _serve(payload, "ok")
        path = os.path.join(tmp, "long.bin")
        with open(path + ".part", "wb") as f:
            f.write(b"\0" * (len(payload) + 1024))
        download(url, path, size=len(payload), sha256=sha, log=lambda *_a: None)
        srv.shutdown(); srv.server_close()
        check("a partial file longer than the target is discarded", open(path, "rb").read() == payload,
              "%d bytes of padding thrown away, then re-fetched" % (len(payload) + 1024))

        # a sha mismatch removes the partial file, so a bad prefix cannot poison the next resume
        srv, url = _serve(payload, "ok")
        path = os.path.join(tmp, "bad.bin")
        try:
            download(url, path, size=len(payload), sha256="0" * 64, log=lambda *_a: None)
            check("a wrong sha256 is refused and removed", False, "download succeeded anyway")
        except SystemExit as e:
            check("a wrong sha256 is refused and removed",
                  "sha256" in str(e) and not os.path.exists(path) and not os.path.exists(path + ".part"),
                  str(e).splitlines()[0][:96])
        finally:
            srv.shutdown(); srv.server_close()
        log("\n  %d checks passed" % nchk)
        return nchk
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    if "--selftest" in argv:
        return 0 if selftest() else 1
    if "--verify" in argv:
        verify(arg("--verify"), log=print)
        return 0
    path, url = arg("--file"), arg("--url")
    size = int(arg("--size")) if arg("--size") else None
    sha = arg("--sha256")
    jobs = int(arg("--jobs", 1))
    if "--repair" in argv:
        path = arg("--repair")
        if not url:
            print("usage: python -m tools.expert_merge.fetch --repair FILE --url URL [--size N] [--sha256 HEX]")
            return 2
        rep = repair(url, path, size=size, sha256=sha, jobs=jobs, log=print)
    elif path and url:
        rep = download(url, path, size=size, sha256=sha, jobs=jobs, log=print)
    elif path:
        rep = verify(path, log=print)
    else:
        print(__doc__.strip().splitlines()[-4] + "\n" + __doc__.strip().splitlines()[-3])
        return 2
    print("  OK" if rep["ok"] else "  INCOMPLETE")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
