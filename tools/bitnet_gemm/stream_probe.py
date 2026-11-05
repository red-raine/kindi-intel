"""How long does one expert read actually take? The number that decides whether streaming pays.

`tmac_gemv_avx2_tiled` needs ~1.8 ms for a 14336 x 4096 expert (14.7 MB packed). A paging pipeline can hide the
*shorter* of the two sides, so the ceiling on any double-buffering scheme is that compute time divided by the sum -
and the sum is set by the device. This measures the device at steady state: unbuffered reads (POSIX `O_DIRECT`, or
`FILE_FLAG_NO_BUFFERING` on Windows) so the OS page cache cannot answer them, over a window larger than the drive's own
cache, with the first pass discarded.

    python -m tools.bitnet_gemm.stream_probe                      # 1 GB file, 256 MB window, 14.7 MB reads
    python -m tools.bitnet_gemm.stream_probe --threads 4          # four reads in flight (queue depth)
    python -m tools.bitnet_gemm.stream_probe --path B:/           # another volume, as a control

Two effects make a naive version of this measure the wrong thing, and both were observed here before the fix:

* **A fresh write is not a device read.** A file smaller than the drive's DRAM/SLC cache is served from it, so this box
  reported 4 GB/s from a *spinning* disk and 3.2 GB/s from an NVMe that does 1.7 GB/s at steady state. Hence a 1 GB
  file and a discarded first pass.
* **A repeated block compresses to nothing** under NTFS compression, which measures the decompressor. The file is
  filled with fresh random data per megabyte.
* **A read of the file you just wrote measures the commit.** This probe writes its test file and reads it
  immediately. A write-back cache takes the write at ~1.8 GB/s and commits at ~68 MB/s, so a 2 GB file is still being
  written for ~30 s and reads of it run at 0.14 GB/s until that ends (`copytest.py --flush` measures the curve). The
  numbers here came out low even after an `os.fsync`, and the same code on a file written hours earlier reaches
  3.1-3.9 GB/s - so read a settled file, or use `copytest.py --src`, before quoting a rate. That is where 1.67 GB/s and
  the "~9 ms expert read" came from.

`--path` is a control, not a feature: a probe that cannot separate an NVMe from a USB disk is measuring nothing.
The file is written under the target directory and removed.
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys
import tempfile
import threading
import time

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
INVALID_HANDLE = ctypes.c_void_p(-1).value

EXPERT_BYTES = 14.7 * 1048576.0        # 14336 x 4096, packed at 2 bits a weight
EXPERT_MS = 1.8                        # tmac_gemv_avx2_tiled as a best-of-N floor (tmac.py); its pinned median is
                                       # 3.3 ms, so every ceiling printed here is conservative - see the tiling
                                       # section of docs/TERNARY-LUT-AND-PAGING-2026.md before building on it
MB = 1048576


# --- readers --------------------------------------------------------------------------------------------------------
# Every reader takes (path, offsets) and reads `chunk` bytes at each offset, returning the seconds each took.

class _WinReader(object):
    def __init__(self, path, chunk):
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.CreateFileW.restype = ctypes.c_void_p
        self.k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD,
                                         ctypes.c_void_p]
        self.k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD),
                                      ctypes.c_void_p]
        self.k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, wt.DWORD]
        self.k32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.path, self.chunk = path, chunk

    def open(self):
        h = self.k32.CreateFileW(self.path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                                 FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
        if h == INVALID_HANDLE:
            raise OSError("CreateFile(FILE_FLAG_NO_BUFFERING) failed: %d" % ctypes.get_last_error())
        raw = ctypes.create_string_buffer(self.chunk + 4096)      # unbuffered needs an aligned destination
        return h, raw, (ctypes.addressof(raw) + 4095) & ~4095

    def read(self, handle, addr, off):
        if not self.k32.SetFilePointerEx(handle, off, None, 0):
            raise OSError("seek to %d failed" % off)
        got = wt.DWORD(0)
        if not self.k32.ReadFile(handle, ctypes.c_void_p(addr), self.chunk, ctypes.byref(got), None):
            raise OSError("ReadFile failed: %d" % ctypes.get_last_error())
        if got.value != self.chunk:
            raise OSError("short read: %d of %d bytes" % (got.value, self.chunk))

    def close(self, handle):
        self.k32.CloseHandle(ctypes.c_void_p(handle))


class _PosixReader(object):
    def __init__(self, path, chunk):
        import mmap
        self.path, self.chunk = path, chunk
        self.mmap = mmap

    def open(self):
        fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        buf = self.mmap.mmap(-1, self.chunk)                      # page-aligned, so O_DIRECT accepts it
        return fd, buf, memoryview(buf)

    def read(self, handle, addr, off):
        fd, view = handle, addr
        got = 0
        while got < self.chunk:
            n = os.preadv(fd, [view[got:]], off + got)            # a short pread is legal
            if n <= 0:
                raise OSError("short O_DIRECT read")
            got += n

    def close(self, fd):
        os.close(fd)


def _seq(reader, offsets):
    """read chunk bytes at each offset, one at a time"""
    h, raw, addr = reader.open()
    times = []
    try:
        for off in offsets:
            t0 = time.perf_counter()
            reader.read(h, addr, off)
            times.append(time.perf_counter() - t0)
    finally:
        reader.close(h)
    return times


def _threaded(reader, offsets):
    """chunk-sized reads in flight at once; one wall-clock time covers the batch, so it is a throughput sample"""
    handles = []

    def worker(off):
        h, raw, addr = reader.open()
        handles.append(h)
        try:
            reader.read(h, addr, off)
        finally:
            reader.close(h)

    errors = []
    threads = []

    def guarded(off):
        try:
            worker(off)
        except OSError as e:
            errors.append(e)

    t0 = time.perf_counter()
    for off in offsets:
        t = threading.Thread(target=guarded, args=(off,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    if errors:
        raise errors[0]
    return [dt]                                                    # the batch, as one sample


def _buffered(path, offsets, chunk):
    times = []
    with open(path, "rb", buffering=0) as f:
        for off in offsets:
            f.seek(off)
            t0 = time.perf_counter()
            if len(f.read(chunk)) != chunk:
                raise OSError("short buffered read")
            times.append(time.perf_counter() - t0)
    return times


# --- benchmark ------------------------------------------------------------------------------------------------------

def bench(reader, path, chunk, window, passes, n_parallel, buffered):
    """sequential passes over `window` at a spread of start offsets; returns the best steady-state pass in seconds"""
    blocks = max(window // chunk, 1)
    best = None
    for p in range(passes):
        start = (window * p) % max(reader_size(path) - window, chunk)
        start -= start % 4096
        offsets = [start + i * chunk for i in range(blocks)]
        groups = [offsets[i:i + n_parallel] for i in range(0, len(offsets), n_parallel)]
        t0 = time.perf_counter()
        if buffered:
            for g in groups:
                _buffered(path, g, chunk)
        elif n_parallel > 1:
            for g in groups:
                _threaded(reader, g)
        else:
            _seq(reader, offsets)
        dt = time.perf_counter() - t0
        if p > 0:                                                  # pass 0 is the discarded warm-up
            best = dt if best is None else min(best, dt)
    return best if best is not None else dt, blocks * chunk


def reader_size(path):
    return os.path.getsize(path)


def main(argv):
    size_mb, read_mb, threads, where, passes, window_mb = 1024, 14.7, 0, None, 3, 256
    for i, a in enumerate(argv):
        if a == "--size":
            size_mb = int(argv[i + 1])
        elif a == "--read":
            read_mb = float(argv[i + 1])
        elif a == "--threads":
            threads = int(argv[i + 1])
        elif a == "--path":
            where = argv[i + 1]
        elif a == "--passes":
            passes = int(argv[i + 1])
        elif a == "--window":
            window_mb = int(argv[i + 1])
    chunk = int(read_mb * MB) // 4096 * 4096
    window = max(window_mb * MB, chunk)
    n_parallel = max(threads, 1)
    size = max(size_mb * MB, window + chunk)

    tmpdir = tempfile.mkdtemp(prefix="kindi-stream-", dir=where)
    path = os.path.join(tmpdir, "experts.bin")
    try:
        t_write = time.perf_counter()
        with open(path, "wb") as f:                                # incompressible: see the module docstring
            for _ in range(size // MB):
                f.write(os.urandom(MB))
            f.flush()
            os.fsync(f.fileno())
        write_s = time.perf_counter() - t_write
        print("%s  %.0f MB written in %.1fs  probe %.1f MB, %d in flight, %d passes over %.0f MB"
              % (os.path.splitdrive(path)[0] or "/", size / MB, write_s, chunk / MB, n_parallel, passes,
                 window / MB))

        reader = _WinReader(path, chunk) if os.name == "nt" else _PosixReader(path, chunk)
        label = "unbuffered" + (" x%d" % n_parallel if n_parallel > 1 else "")
        arms = [(label, False), ("buffered", True)]
        rates = {}
        for name, buffered in arms:
            dt, moved = bench(reader, path, chunk, window, passes, n_parallel, buffered)
            gbps = moved / dt / 1e9
            rates[name] = gbps
            print("  %-14s %8.1f MB in %8.2f s   %6.2f GB/s" % (name, moved / MB, dt, gbps))
    finally:
        try:
            os.remove(path)
            os.rmdir(tmpdir)
        except OSError:
            pass

    gbps = rates[label]
    per_expert = EXPERT_BYTES / (gbps * 1e9) * 1e3
    print("\nsteady state %.2f GB/s: one 14.7 MB expert is %.2f ms to read against %.1f ms to multiply."
          % (gbps, per_expert, EXPERT_MS))
    print("  ceiling for double buffering: %.1f / (%.1f + %.2f) = %.0f%% saved - and only that."
          % (EXPERT_MS, EXPERT_MS, per_expert, 100.0 * EXPERT_MS / (EXPERT_MS + per_expert)))
    print("  tokens per expert read needed to make the read free: %.1f" % (per_expert / EXPERT_MS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
