"""Is a read rate the medium, the Windows I/O path, or a cache? One file, three paths, one process.

`stream_probe.py` reports 1.5-1.6 GB/s for a 14.7 MB unbuffered read and 3.3 GB/s for a 200 MB one, and both are
numbers from a *virtualised* host: the same probe reads at 2.5-3.6 GB/s from a 4 TB spinning disk, which no such disk
can do. So the 1.67 GB/s figure cannot settle whether a KIOXIA NVMe is that fast, or whether the guest's I/O path is
throttling a faster drive. A file copy is the natural control, because it moves the *same* bytes through the ordinary
buffered path (Cache Manager, readahead, CopyFile) instead of the probe's `FILE_FLAG_NO_BUFFERING` reads:

    python -m tools.bitnet_gemm.copytest --path C:/ --size 4096 --evict B:/models/anybig.gguf
    python -m tools.bitnet_gemm.copytest --path B:/ --src B:/models/big.gguf --offset 8192 --size 4096

The second form skips the write and reads a pre-existing file instead, which is the stronger evidence: bytes this run
wrote may still be in a cache on either side, while an old file at an offset nothing has touched is not.

Every arm covers the same bytes of one freshly written file of incompressible data:

* **write, buffered** - the write path a copy uses. The unbuffered write is timed beside it because the earlier
  "both disks write at ~60 MB/s" came from that flag: `FILE_FLAG_NO_BUFFERING` defeats write combining, so the same
  drive that writes at GB/s through the Cache Manager measures at tens of MB/s under it.
* **read, unbuffered** - `FILE_FLAG_NO_BUFFERING`, over the whole file (an average, which compares with the buffered
  pass) and as a best single read (which is what `stream_probe` quotes, and is not the same number: a best-of sample
  lands wherever the host cache still holds data).
* **read, buffered** - `ReadFile` with no flags, which is the path a copy uses. Pass 1 is cold, pass 2 warm.
* **copy** - a buffered copy of half the file, timed. This is the arm the question asks for.

Two things make the answer honest rather than merely plausible:

* **Pass 2 must be much faster than pass 1.** If it is not, nothing was cached and the numbers describe the medium. If
  it is, pass 1's rate is a device number only for the region the cache did not hold - which is why `--size` should
  exceed the host's cache, and why `--evict` reads that much cold data between the write and the first cold read.
* **`--evict` needs data the caches cannot already hold.** A file that has been read in this session is useless as an
  evictor; point it at a large file and pass `--evict-offset` past anything an earlier run has touched.

The tool prints the 1.67 GB/s claim beside every arm, because the question is whether any path exceeds it, and then
`--flush` writes a file without fsync and reads it back on a timer. That last arm is what the answer turned out to be: a
drive with a write-back cache takes writes at ~1.8 GB/s and commits them at ~68 MB/s, and while a commit is running,
reads of the very file being committed drop to 0.14 GB/s - then come back to 3.2 GB/s when it ends. A probe that writes
its own test file and reads it immediately measures that collision, which is where 1.67, and the "9 ms expert read"
derived from it, came from.
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys
import tempfile
import time

MB = 1048576
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x1
CREATE_ALWAYS = 2
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
INVALID_HANDLE = ctypes.c_void_p(-1).value
CLAIM = 1.67                                 # GB/s, the figure under test


class _Win(object):
    """CreateFile/ReadFile/WriteFile with the two flags that matter here, over one aligned buffer."""

    def __init__(self):
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.CreateFileW.restype = ctypes.c_void_p
        self.k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD,
                                         ctypes.c_void_p]
        self.k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD),
                                      ctypes.c_void_p]
        self.k32.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD),
                                       ctypes.c_void_p]
        self.k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, wt.DWORD]
        self.k32.CloseHandle.argtypes = [ctypes.c_void_p]

    def open(self, path, access, disposition, flags, size):
        h = self.k32.CreateFileW(path, access, FILE_SHARE_READ, None, disposition, flags, None)
        if h == INVALID_HANDLE:
            raise OSError("CreateFile(%s) failed: %d" % (path, ctypes.get_last_error()))
        raw = ctypes.create_string_buffer(size + 4096)
        return h, raw, (ctypes.addressof(raw) + 4095) & ~4095

    def write(self, h, addr, n):
        done = wt.DWORD(0)
        if not self.k32.WriteFile(h, ctypes.c_void_p(addr), n, ctypes.byref(done), None) or done.value != n:
            raise OSError("WriteFile failed: %d, %d of %d" % (ctypes.get_last_error(), done.value, n))

    def read(self, h, addr, n):
        done = wt.DWORD(0)
        if not self.k32.ReadFile(h, ctypes.c_void_p(addr), n, ctypes.byref(done), None) or done.value != n:
            raise OSError("ReadFile failed: %d, %d of %d" % (ctypes.get_last_error(), done.value, n))

    def close(self, h):
        self.k32.CloseHandle(ctypes.c_void_p(h))


def _buffered_write(path, size, block):
    """write `size` bytes of fresh random data through the cache, which is the path a copy uses

    The unbuffered write is timed too, but only to show what it costs: `FILE_FLAG_NO_BUFFERING` defeats write
    combining, so a drive that writes at GB/s through the Cache Manager can measure at tens of MB/s here. An earlier
    probe reported "both disks write at ~60 MB/s" from exactly that artifact.
    """
    t0 = time.perf_counter()
    with open(path, "wb", buffering=0) as f:
        left = size
        while left:
            n = min(block, left)
            f.write(os.urandom(n))                           # incompressible: NTFS compression must see noise
            left -= n
        f.flush()
        os.fsync(f.fileno())
    return time.perf_counter() - t0


def _unbuffered_write(win, path, size, block):
    """write `size` bytes straight to the device, no cache in the way - timed only to price the flag"""
    h, raw, addr = win.open(path, GENERIC_WRITE, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING, block)
    try:
        t0 = time.perf_counter()
        left = size
        while left:
            n = min(block, left)
            ctypes.memmove(addr, os.urandom(n), n)
            win.write(h, addr, n)
            left -= n
        return time.perf_counter() - t0
    finally:
        win.close(h)


def _buffered_read(win, path, size, block, offset=0):
    """one sequential pass through the Cache Manager, which is the path CopyFile uses

    `ReadFile` into a preallocated buffer, not `f.read()`: a Python loop that allocates a fresh bytes object per block
    tops out near 2 GB/s on this host, which would put its own ceiling under the number being measured.
    """
    h, raw, addr = win.open(path, GENERIC_READ, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, block)
    try:
        if not win.k32.SetFilePointerEx(ctypes.c_void_p(h), offset, None, 0):
            raise OSError("seek to %d failed" % offset)
        t0 = time.perf_counter()
        left = size
        while left >= block:
            win.read(h, addr, block)
            left -= block
        return time.perf_counter() - t0, size - left
    finally:
        win.close(h)


def _copy(win, src, dst):
    """`CopyFileExW` - a real file copy, done in the kernel: no Python buffer, no user-mode copy to blame"""
    if os.path.exists(dst):
        os.remove(dst)
    fn = win.k32.CopyFileExW
    fn.restype = wt.BOOL
    fn.argtypes = [wt.LPCWSTR, wt.LPCWSTR, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wt.DWORD]
    t0 = time.perf_counter()
    if not fn(src, dst, None, None, None, 0):
        raise OSError("CopyFileEx failed: %d" % ctypes.get_last_error())
    return time.perf_counter() - t0, os.path.getsize(src)


def _slice(src, dst, size, offset=0):
    """materialise the first `size` bytes of `src` at `offset` - untimed setup for a kernel copy of a big file"""
    with open(src, "rb", buffering=0) as fi, open(dst, "wb", buffering=0) as fo:
        fi.seek(offset)
        left = size
        while left:
            b = fi.read(min(16 * MB, left))
            if not b:
                break
            fo.write(b)
            left -= len(b)
    return os.path.getsize(dst)


def _unbuffered_reads(win, path, size, block, offset=0):
    """two arms over the whole file: the sequential average, and the best single read's latency

    The average is what compares with the buffered pass; the best single read is what `stream_probe` quotes, and they
    are not the same number - a best-of sample lands wherever the host cache still holds data.
    """
    h, raw, addr = win.open(path, GENERIC_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
                            block)
    try:
        if not win.k32.SetFilePointerEx(ctypes.c_void_p(h), offset, None, 0):
            raise OSError("seek to %d failed" % offset)
        best = None
        t0 = time.perf_counter()
        left = size
        while left >= block:
            t1 = time.perf_counter()
            win.read(h, addr, block)
            dt = time.perf_counter() - t1
            best = dt if best is None else min(best, dt)
            left -= block
        total = time.perf_counter() - t0
        return total, size - left, best
    finally:
        win.close(h)


def _flush_curve(win, path, size, block, chunk, log):
    """write `size` bytes without fsync, then read them back while the drive commits them

    This is the arm that names the number this tool was built to explain. A drive with a write-back cache accepts writes
    at ~1.8 GB/s and commits them at ~68 MB/s, so for the length of the commit the medium is busy writing and reads of
    the file that is being written run at a fraction of their own rate. A probe that writes its own test file and reads
    it immediately therefore measures the commit, not the device.
    """
    with open(path, "wb", buffering=0) as f:
        left = size
        while left:
            n = min(block, left)
            f.write(os.urandom(n))
            left -= n
        f.flush()                                   # deliberately not fsync'd: the commit is left running
    t0 = time.perf_counter()
    rows = []
    while True:
        dt, got, _ = _unbuffered_reads(win, path, 10 * chunk, chunk)
        rate = got / dt / 1e9
        rows.append((time.perf_counter() - t0, rate))
        log("    %5.1f s after the write: %5.2f GB/s" % rows[-1])
        if (len(rows) >= 5 and rate > 2.0) or rows[-1][0] > 48.0:
            break                                   # the first reads hit cache; the commit shows up a few seconds in
        time.sleep(3.0)
    return rows


def _random_reads(win, path, span, n, block=4096):
    """`n` random 4 KB reads over `span` bytes: the one pattern a cache cannot make fast at scale

    This is the arm that identifies the medium. Sequential reads can be served from a host cache several GB deep and
    report numbers no drive of that class can produce; random 4 KB reads over a span larger than any plausible cache
    fall back to the device, and a spinning disk cannot fake them.
    """
    import random
    rng = random.Random(7)
    h, raw, addr = win.open(path, GENERIC_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, block)
    try:
        offsets = [rng.randrange(0, max(span - block, block)) // 4096 * 4096 for _ in range(n)]
        t0 = time.perf_counter()
        for off in offsets:
            if not win.k32.SetFilePointerEx(ctypes.c_void_p(h), off, None, 0):
                raise OSError("seek to %d failed" % off)
            win.read(h, addr, block)
        dt = time.perf_counter() - t0
        return dt, n, block
    finally:
        win.close(h)


def _evict(win, path, size, block, offset=0):
    """read `size` bytes of cold data to push the test file out of whatever cache wrote it

    The offset matters: the first region of a file is the one a previous run has already read, and cached data cannot
    evict anything.
    """
    h, raw, addr = win.open(path, GENERIC_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
                            block)
    try:
        if not win.k32.SetFilePointerEx(ctypes.c_void_p(h), offset, None, 0):
            raise OSError("seek to %d failed" % offset)
        t0 = time.perf_counter()
        left = min(size, max(os.path.getsize(path) - offset, 0))
        while left >= block:
            win.read(h, addr, block)
            left -= block
        return time.perf_counter() - t0
    finally:
        win.close(h)


def main(argv):
    where, evict_path, evict_off, src_override = None, None, 0, None
    size_mb, chunk_mb, copy_mb, passes, offset, n_rand, flush_mb = 4096, 14.7, None, 3, 0, 0, None
    for i, a in enumerate(argv):
        if a == "--path":
            where = argv[i + 1]
        elif a == "--src":
            src_override = argv[i + 1]
        elif a == "--offset":
            offset = int(argv[i + 1]) * MB
        elif a == "--size":
            size_mb = int(argv[i + 1])
        elif a == "--chunk":
            chunk_mb = float(argv[i + 1])
        elif a == "--copy":
            copy_mb = int(argv[i + 1])
        elif a == "--evict":
            evict_path = argv[i + 1]
        elif a == "--evict-offset":
            evict_off = int(argv[i + 1]) * MB
        elif a == "--rand":
            n_rand = int(argv[i + 1])
        elif a == "--flush":
            flush_mb = int(argv[i + 1])
    if os.name != "nt":
        print("this probe is Windows-only: the question is about the Windows I/O path")
        return 2
    size = size_mb * MB
    copy_mb = copy_mb if copy_mb else max(size_mb // 2, 512)
    chunk = int(chunk_mb * MB) // 4096 * 4096
    win = _Win()

    tmpdir = tempfile.mkdtemp(prefix="kindi-copy-", dir=where)
    created = os.path.join(tmpdir, "src.bin")
    src = src_override or created
    dst = os.path.join(tmpdir, "dst.bin")
    if src_override:
        avail = os.path.getsize(src) - offset
        size = min(size, max(avail, 0))
    copy_mb = min(copy_mb, size // MB)
    vol = os.path.splitdrive(dst)[0] or "/"
    print("%s  %d MB of %s%s, %.1f MB unbuffered reads, %d MB copy"
          % (vol, size // MB, os.path.basename(src), " at %d MB" % (offset // MB) if offset else "", chunk / MB,
             copy_mb))
    try:
        if src_override:
            w_rate = None
            print("  %-30s %7s       pre-existing source, not written by this run" % ("write", "-"))
        else:
            w_dt = _buffered_write(created, size, 4 * MB)
            w_rate = size / w_dt / 1e9
            print("  %-30s %7.2f s   %6.2f GB/s" % ("write, buffered", w_dt, w_rate))
            uw_n = min(size, 512 * MB)
            u_dt = _unbuffered_write(win, dst, uw_n, 4 * MB)
            print("  %-30s %7.2f s   %6.2f GB/s   (the flag's cost, not the drive's)"
                  % ("write, unbuffered", u_dt, uw_n / u_dt / 1e9))

        if evict_path:
            e_dt = _evict(win, evict_path, size, 4 * MB, evict_off)
            print("  %-30s %7.2f s   %6.2f GB/s   (%s)" % ("evict read", e_dt, size / e_dt / 1e9,
                                                           os.path.basename(evict_path)))

        u_dt, u_moved, u_best = _unbuffered_reads(win, src, size, chunk, offset)
        u_rate = u_moved / u_dt / 1e9
        best_rate = chunk / u_best / 1e9
        print("  %-30s %7.2f s   %6.2f GB/s   (whole file, %.1f MB a read)" % ("read, unbuffered", u_dt, u_rate,
                                                                                chunk / MB))
        print("  %-30s %7.2f ms  %6.2f GB/s   (one read, best of %d)" % ("read, unbuffered, best", u_best * 1e3,
                                                                          best_rate, size // chunk))

        cold = warm = 0.0
        for p in range(passes):
            dt, got = _buffered_read(win, src, size, 8 * MB, offset)
            rate = got / dt / 1e9
            tag = "read, buffered pass %d" % (p + 1) + (" (cold)" if p == 0 else "")
            print("  %-30s %7.2f s   %6.2f GB/s" % (tag, dt, rate))
            if p == 0:
                cold = rate
            warm = rate

        slice_path = os.path.join(tmpdir, "slice.bin")
        if src_override:
            c_mb = _slice(src, slice_path, copy_mb * MB, offset)   # untimed: this is just the payload
        else:
            slice_path = src
        c_dt, c_moved = _copy(win, slice_path, dst)
        c_rate = c_moved / c_dt / 1e9
        print("  %-30s %7.2f s   %6.2f GB/s   (%.2f GB/s read+write, %d MB via CopyFileEx)"
              % ("copy, kernel", c_dt, c_rate, 2 * c_rate, c_moved // MB))

        if flush_mb:
            print("\n  reads of a file whose own write is still committing (%d MB, no fsync):" % flush_mb)
            _flush_curve(win, os.path.join(tmpdir, "flush.bin"), flush_mb * MB, 4 * MB, chunk, print)

        if n_rand:
            r_dt, r_n, r_blk = _random_reads(win, src, os.path.getsize(src), n_rand)
            print("  %-30s %7.2f s   %6.0f IOPS   %.2f MB/s   (4 KB random over %.1f GB)"
                  % ("read, random", r_dt, r_n / r_dt, r_n * r_blk / r_dt / 1e6, os.path.getsize(src) / 1e9))

        arms = [("buffered read, whole file (cold)", cold), ("buffered read, warm", warm),
                ("copy (share of one direction)", c_rate),
                ("unbuffered read, whole file", u_rate),
                ("unbuffered read, best single", best_rate)]
        if w_rate is not None:
            arms.append(("buffered write", w_rate))
        print("\n  the claim under test is %.2f GB/s, and the arms compare against it as follows:" % CLAIM)
        for name, rate in arms:
            verdict = "above" if rate > CLAIM else "below"
            print("    %-30s %6.2f GB/s  %s the claim" % (name, rate, verdict))
        print("\n  a copy is read+write on the same medium, so it cannot exceed the read rate: %.2f GB/s of device"
              % c_rate)
        print("  traffic in each direction means the medium sustains at least twice the copy's one-way share.")
        if warm > 1.5 * cold:
            print("  pass 2 is %.1fx pass 1, so the caches did answer part of pass 1: treat its rate as an upper bound"
                  % (warm / max(cold, 1e-9)))
        else:
            print("  pass 2 is %.2fx pass 1, i.e. the cold pass was not served from cache." % (warm / max(cold, 1e-9)))
    finally:
        for p in (created, dst, os.path.join(tmpdir, "slice.bin"), os.path.join(tmpdir, "flush.bin")):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
