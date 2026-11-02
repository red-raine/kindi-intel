"""Read a torch zip-format `.pt` state dict with NumPy alone (no torch install).

A `.pt` saved by `torch.save` is a zip holding `data.pkl` plus one `data/<key>` blob per storage. The pickle refers
to tensors as `(_rebuild_tensor_v2, storage, offset, size, stride, ...)` and to storages as
`('storage', Type, key, location, numel)`, so stubbing those two classes and reading the raw blob bytes rebuilds
every tensor exactly - the same values torch would hand back, without importing it.

  from .ptload import load
  sd = load("qat_final.pt")            # {name: np.ndarray}, float32/bfloat16/int64 as stored

`load` also reports what it skipped: a checkpoint saved with `torch.save(model)` (not a state dict) holds no tensor
list this can rebuild, and that is an error rather than an empty dict.
"""
import collections
import pickle
import zipfile

import numpy as np

DTYPES = {
    "FloatStorage": "<f4", "DoubleStorage": "<f8", "HalfStorage": "<f2", "BFloat16Storage": "<u2",
    "LongStorage": "<i8", "IntStorage": "<i4", "ShortStorage": "<i2", "ByteStorage": "u1",
    "CharStorage": "i1", "BoolStorage": "u1",
}


def _storage_type(name):
    """a stand-in class whose __name__ is the storage type, so persistent_load can look the dtype up"""
    return type(name, (object,), {})


class _Tensor(object):
    def __init__(self, storage, offset, size, stride, *rest):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fp, blobs):
        pickle.Unpickler.__init__(self, fp)
        self.blobs = blobs

    def find_class(self, mod, name):
        if mod.startswith("torch"):
            if name.startswith("_rebuild"):
                return lambda *a, **k: _Tensor(*a)
            if name.endswith("Storage"):
                return _storage_type(name)
            return lambda *a, **k: None
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        return pickle.Unpickler.find_class(self, mod, name)

    def persistent_load(self, pid):
        _, stype, key, _loc, numel = pid
        dt = DTYPES.get(getattr(stype, "__name__", ""))
        if dt is None:
            raise SystemExit("kindi ptload: unsupported storage type %r" % (getattr(stype, "__name__", stype),))
        return np.frombuffer(self.blobs[key], dtype=dt, count=int(numel))


def load(path):
    """{tensor name: ndarray}. bfloat16 storages come back as their raw uint16 bits (callers widen as they wish)."""
    z = zipfile.ZipFile(path)
    blobs = {n.split("/")[-1]: z.read(n) for n in z.namelist() if n.split("/")[-1].isdigit()}
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")]
    if not pkl:
        raise SystemExit("kindi ptload: %s has no data.pkl - not a torch zip checkpoint" % path)
    obj = _Unpickler(z.open(pkl[0]), blobs).load()
    for key in ("model", "state_dict", "module"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise SystemExit("kindi ptload: %s holds %s, not a state dict" % (path, type(obj).__name__))
    out = {}
    for k, v in obj.items():
        if not isinstance(v, _Tensor):
            continue
        if not v.size:
            continue
        base = v.storage[v.offset:]
        out[k] = np.lib.stride_tricks.as_strided(
            base, shape=v.size, strides=tuple(s * base.itemsize for s in v.stride)).copy()
    if not out:
        raise SystemExit("kindi ptload: %s has no tensors in its state dict" % path)
    return out


def widen(a):
    """bfloat16 bits -> float32 (exact), anything else through astype(float32)"""
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)
