"""How long does one expert read actually take? The number that decides whether streaming pays.

`tmac_gemv_avx2_tiled` needs ~1.8 ms for a 14336 x 4096 expert (14.7 MB packed). A paging pipeline can hide the
*shorter* of the two sides, so the ceiling on any double-buffering scheme is that compute time divided by the sum -
and the sum is set by the device. This measures the device: unbuffered reads (POSIX `O_DIRECT`, or
`FILE_FLAG_NO_BUFFERING` on Windows) so the page cache cannot answer them, against buffered reads of the same file for
comparison.

    python -m tools.bitnet_gemm.stream_probe                 # 64 MB file, 14.7 MB reads
    python -m tools.bitnet_gemm.stream_probe --size 256 --read 14.7

The file is written under the system temp directory and removed.
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys
import tempfile
import time

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
INVALID_HANDLE = ctypes.c_void_p(-1).value


def _win_unbuffered_reads(path, chunk, reps):
    """Time `reps` reads of `chunk` bytes at spread offsets, bypassing the page cache."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, ctypes.c_void_p]
    k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
    k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, wt.DWORD]

    handle = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                             FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
    if handle == INVALID_HANDLE:
        raise OSError("CreateFile(FILE_FLAG_NO_BUFFERING) failed: %d" % ctypes.get_last_error())
    # the buffer itself has to be sector-aligned for unbuffered I/O, so allocate aligned and slice from it
    raw = ctypes.create_string_buffer(chunk + 4096)
    addr = ctypes.addressof(raw)
    aligned = (addr + 4095) & ~4095
    total = os.path.getsize(path)
    times = []
    try:
        for i in range(reps):
            off = (total - chunk) * i // max(reps - 1, 1)
            off -= off % 4096
            if not k32.SetFilePointerEx(handle, off, None, 0):
                raise OSError("seek failed")
            got = wt.DWORD(0)
            t0 = time.perf_counter()
            ok = k32.ReadFile(handle, ctypes.c_void_p(aligned), chunk, ctypes.byref(got), None)
            dt = time.perf_counter() - t0
            if not ok or got.value != chunk:
                raise OSError("unbuffered read got %d of %d bytes (err %d)"
                              % (got.value, chunk, ctypes.get_last_error()))
            times.append(dt)
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))
    return times


def _posix_unbuffered_reads(path, chunk, reps):
    """O_DIRECT needs an aligned destination; an mmap slice at a page-aligned offset is one."""
    import mmap
    total = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    buf = mmap.mmap(-1, chunk)
    view = memoryview(buf)
    times = []
    try:
        for i in range(reps):
            off = (total - chunk) * i // max(reps - 1, 1)
            off -= off % 4096
            t0 = time.perf_counter()
            got, want = 0, chunk
            while want:                            # a short pread is legal
                n = os.preadv(fd, [view[got:]], off + got)
                if n <= 0:
                    raise OSError("short O_DIRECT read")
                got += n
                want -= n
            times.append(time.perf_counter() - t0)
    finally:
        view.release()
        buf.close()
        os.close(fd)
    return times


def _buffered_reads(path, chunk, reps):
    total = os.path.getsize(path)
    times = []
    with open(path, "rb", buffering=0) as f:
        for i in range(reps):
            off = (total - chunk) * i // max(reps - 1, 1)
            f.seek(off)
            t0 = time.perf_counter()
            buf = f.read(chunk)
            dt = time.perf_counter() - t0
            if len(buf) != chunk:
                raise OSError("short buffered read")
            times.append(dt)
    return times


def main(argv):
    size_mb, read_mb, reps = 64, 14.7, 6
    for i, a in enumerate(argv):
        if a == "--size":
            size_mb = int(argv[i + 1])
        elif a == "--read":
            read_mb = float(argv[i + 1])
    size = size_mb * 1024 * 1024
    chunk = int(read_mb * 1024 * 1024)
    chunk -= chunk % 4096

    path = os.path.join(tempfile.mkdtemp(prefix="kindi-stream-"), "experts.bin")
    block = os.urandom(1024 * 1024)
    with open(path, "wb") as f:
        for _ in range(size_mb):
            f.write(block)
        f.flush()
        os.fsync(f.fileno())
    print("file %.0f MB, probe %s, %d reads at spread offsets"
          % (size_mb, ("%.1f MB" % (chunk / 1048576.0)), reps))

    arms = [("buffered (page cache answers if it can)", _buffered_reads)]
    if os.name == "nt":
        arms.insert(0, ("unbuffered, FILE_FLAG_NO_BUFFERING", _win_unbuffered_reads))
    elif hasattr(os, "O_DIRECT"):
        arms.insert(0, ("unbuffered, O_DIRECT", _posix_unbuffered_reads))

    best = {}
    try:
        for label, fn in arms:
            ts = fn(path, chunk, reps)
            ms = min(ts) * 1e3
            med = sorted(ts)[len(ts) // 2] * 1e3
            best[label] = ms
            print("  %-42s min %8.2f ms   median %8.2f ms   %6.2f GB/s"
                  % (label, ms, med, chunk / (ms / 1e3) / 1e9))
    finally:
        try:
            os.remove(path)
            os.rmdir(os.path.dirname(path))
        except OSError:
            pass

    cold = min(best.values())
    # tmac_gemv_avx2_tiled takes 1.8 ms for 14336 x 4096 (14.7 MB packed) at 8 threads; scale with the weights read
    expert_bytes = 14.7 * 1048576.0
    compute_ms = 1.8 * chunk / expert_bytes
    print("\n%.1f MB read: %.2f ms to read (best arm) against ~%.2f ms to multiply."
          % (chunk / 1048576.0, cold, compute_ms))
    print("  ceiling for double buffering: %.2f / (%.2f + %.2f) = %.0f%% saved - and only that."
          % (compute_ms, compute_ms, cold, 100.0 * compute_ms / (compute_ms + cold)))
    print("  Per token, per layer, per expert: a top-2 layer pays two reads to hide two multiplies.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
