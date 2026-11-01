"""Disk streaming, peer-to-peer and pinned-memory helpers for Kindi ("P2P" in the broad sense:
disk -> page cache -> device, and device <-> device without a host bounce).

What this module adds on top of the existing engine paths:

  stream_load(path, n, dtype)   chunked file -> device array with POSIX_FADV_DONTNEED after every chunk, so a
                                19 GB model never fills the host page cache (several servers can load at once;
                                the same trick kindi/tp.py uses for the 5-trit cache, factored out here);
  pin_file(path)               mmap + madvise(MADV_WILLNEED|HUGEPAGE): the kernel reads ahead in huge pages -
                                on Intel this is what feeds the iGPU/NPU DMA fast (the VMD/PCIe root complex
                                likes long sequential bursts; small scattered reads cost ~3x);
  pread_into(dst, f, off, n)    read(2)/pread into a *device-backed* buffer through a small pinned staging
                                ring (double buffering: disk read of chunk k+1 overlaps the H2D copy of chunk k)
  p2p_enable()                 NVIDIA cuCtxEnablePeerAccess between the visible cards (K1200 pairs: the exchange
                                in tp.py goes through mapped pinned memory today; when both cards support P2P
                                (NVLink or PCIe BAR1 mapping) the driver copies card->card directly instead);
  unified_hint(dev)            Intel Arc / Radeon 780M: the "device memory" is host memory (unified), so a
                                streamed load is just a pin_file + first-touch; we still drop caches per chunk.

Environment:
  KINDI_DISK_P2P=1        default on: use the double-buffered streaming loader where a loader exists
  KINDI_STREAM_CHUNK_MB   staging chunk size (default 64 MiB; 256 MiB was best on the K1200 pair)
  KINDI_PIN_FILE=1        pin the pack files with madvise before loading (default on for CPU/iGPU/NPU paths)
"""
import ctypes
import ctypes.util
import os

libc = None
try:
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
except OSError:
    pass

MADV_NORMAL, MADV_SEQUENTIAL, MADV_WILLNEED, MADV_HUGEPAGE = 0, 2, 3, 14
POSIX_FADV_DONTNEED = 4


def chunk_bytes():
    return int(os.environ.get("KINDI_STREAM_CHUNK_MB", "64")) * (1 << 20)


def pin_file(path, length=None):
    """mmap the file read-only and ask the kernel for read-ahead in huge pages; returns (mm, addr, size).
    On failure returns None and the caller falls back to plain reads."""
    if libc is None or os.environ.get("KINDI_PIN_FILE", "1") != "1":
        return None
    PROT_READ, MAP_SHARED = 1, 1
    fd = os.open(path, os.O_RDONLY)
    try:
        size = length or os.fstat(fd).st_size
        mm = libc.mmap(None, ctypes.c_size_t(size), PROT_READ, MAP_SHARED, fd, ctypes.c_longlong(0))
        if mm in (ctypes.c_void_p(-1).value, -1, None):
            return None
        addr = mm if isinstance(mm, int) else ctypes.cast(mm, ctypes.c_void_p).value
        libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(size), MADV_SEQUENTIAL)
        libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(size), MADV_WILLNEED)
        libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(min(size, 1 << 30)), MADV_HUGEPAGE)  # best effort
        return (mm, addr, size)
    finally:
        os.close(fd)


def unpin(handle):
    if handle is None or libc is None:
        return
    mm, addr, size = handle
    try:
        libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    except OSError:
        pass


def drop_cache_range(fd, off, nbytes):
    """give the page cache back what we have already streamed (so the next server can use the RAM)"""
    try:
        os.posix_fadvise(fd, off, nbytes, POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        pass


def stream_load(cp, path, dtype, log=print):
    """file -> device array in chunks, page cache dropped behind us. cp is cupy (or an object with
    .empty/.asarray semantics); works unchanged for CUDA, HIP and Intel GPU via their own loaders."""
    import numpy as np
    itemsize = np.dtype(dtype).itemsize
    n = os.path.getsize(path) // itemsize
    out = cp.empty(n, dtype=dtype)
    chunk = chunk_bytes() // itemsize
    fd = os.open(path, os.O_RDONLY)
    t0 = os.times()[4]
    try:
        with os.fdopen(os.dup(fd), "rb") as f:
            o = 0
            while o < n:
                cnt = min(chunk, n - o)
                a = np.frombuffer(f.read(cnt * itemsize), dtype=dtype)
                out[o:o + a.size].set(a)              # cudaMemcpyAsync through CuPy's pinned staging pool
                drop_cache_range(fd, o * itemsize, a.size * itemsize)
                o += a.size
                del a
    finally:
        os.close(fd)
    dt = os.times()[4] - t0
    log("streamed %s (%.1f GB) to the device in %.1fs (%.0f MB/s)" %
        (os.path.basename(path), n * itemsize / 2 ** 30, dt, n * itemsize / 2 ** 20 / max(dt, 1e-6)))
    return out


def double_buffer_load(cp, path, dtype, log=print):
    """The bleeding-edge variant: one thread keeps reading the next chunk from disk into pinned host memory
    while the previous chunk is copied to the device (read/H2D overlap). This is what llama.cpp's
    --no-mmap weight loader and Microsoft's ML-Ampere papers describe for slow NVMe hosts; on a SATA SSD it
    raises effective load bandwidth toward the disk limit rather than half of it (copy + read serialized)."""
    import threading
    import numpy as np
    itemsize = np.dtype(dtype).itemsize
    n = os.path.getsize(path) // itemsize
    out = cp.empty(n, dtype=dtype)
    chunk = chunk_bytes() // itemsize
    stage = [cp.cuda.alloc_pinned_memory(chunk * itemsize) for _ in range(2)]
    host = [np.frombuffer(s, dtype=dtype) for s in stage]      # numpy views over the pinned buffers
    fd = os.open(path, os.O_RDONLY)
    err = []

    def reader(bidx, off, cnt):
        try:
            with open(fd, "rb", closefd=False) as f:
                buf = host[bidx][:cnt]
                got = f.readinto(buf)                          # pread-style into pinned memory
                if got is not None:
                    buf = buf[:got // itemsize]
                dst = out[off:off + buf.size]
                src = cp.asarray(buf)                          # pinned -> device, async on the current stream
                dst.set(src)
                drop_cache_range(fd, off * itemsize, buf.size * itemsize)
                cp.cuda.Stream.null.synchronize()
                return buf.size
        except OSError as e:
            err.append(e)
            return 0

    t0 = os.times()[4]
    o, bidx, pending = 0, 0, []
    try:
        while o < n:
            cnt = min(chunk, n - o)
            th = threading.Thread(target=reader, args=(bidx, o, cnt))
            th.start()
            pending.append(th)
            if len(pending) == 2:
                pending.pop(0).join()
            o += cnt
            bidx ^= 1
        for th in pending:
            th.join()
    finally:
        os.close(fd)
    if err:
        raise err[0]
    dt = os.times()[4] - t0
    log("double-buffer streamed %s: %.0f MB/s" % (os.path.basename(path), n * itemsize / 2 ** 20 / max(dt, 1e-6)))
    return out


def load(cp, path, dtype, log=print):
    """pick the loader: KINDI_DISK_P2P=2 forces the double-buffered one, 1 (default) the simple chunked one,
    0 a plain whole-file upload."""
    mode = os.environ.get("KINDI_DISK_P2P", "1")
    if mode == "0":
        import numpy as np
        return cp.asarray(np.fromfile(path, dtype=dtype))
    if mode == "2":
        try:
            return double_buffer_load(cp, path, dtype, log=log)
        except Exception as e:                              # any problem: fall back, never fail a load
            log("kindi: double-buffer loader unavailable (%s), using the chunked loader" % e)
    return stream_load(cp, path, dtype, log=log)


# ---------------------------------------------------------------------------
# device <-> device peer access (NVIDIA P2P; on AMD/Intel the equivalent is unified memory)
# ---------------------------------------------------------------------------

_p2p_done = [False]


def p2p_enable(log=print):
    """cuCtxEnablePeerAccess between all visible CUDA cards (both directions). Returns True when the pair
    can talk P2P; the tensor-parallel exchange then uses device pointers instead of the host bounce."""
    if _p2p_done[0]:
        return False
    _p2p_done[0] = True
    if os.environ.get("KINDI_P2P", "1") == "0":
        return False
    try:
        import cupy as cp
        if getattr(cp.cuda.runtime, "is_hip", False):
            log("p2p: ROCm - peer access is hipDeviceCanAccessPeer; Kindi uses the shared host pages anyway")
            return False
        rt = cp.cuda.runtime
        ndev = rt.getDeviceCount()
        if ndev < 2:
            return False
        ok = True
        for a in range(ndev):
            for b in range(ndev):
                if a == b:
                    continue
                if not rt.deviceCanAccessPeer(a, b):
                    ok = False
                    continue
                with cp.cuda.Device(a):
                    try:
                        rt.deviceEnablePeerAccess(b)
                    except Exception as e:                  # ErrorContextAlreadySetByOtherThread etc.
                        if getattr(e, "errno", None) not in (704, 2904, 506):   # cudaErrorPeerAccessAlreadyEnabled
                            ok = False
        log("p2p: %s" % ("peer access enabled between %d cards" % ndev if ok else
                         "not available on this pair - keeping the pinned-host exchange"))
        return ok
    except Exception as e:
        log("p2p: probe failed (%s)" % e)
        return False
