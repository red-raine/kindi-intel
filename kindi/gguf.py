"""Minimal GGUF (v2/v3) reader: metadata key/values and tensor infos. Tensor data is read by offset.

A header is trusted only as far as the file can back it. Every length a header states - a string, an array's element
count, a rank's dimensions, the number of tensors or key/values - is checked against the bytes that remain before it is
allocated or read, so a truncated or corrupted file raises a `ValueError` naming the field. Without that check a file
whose tokenizer array claims four billion entries does not fail, it asks for 32 GB and dies with a bare `MemoryError`
that says nothing about which model, which key, or which byte.
"""
import os
import struct

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9

MAX_DIMS = 8                      # ggml's own limit; a larger rank is a corrupt header, not a model

# the smallest a header entry can be, used to reject an impossible count before looping over it
_MIN_KV_BYTES = 12                # key length (8) + type (4)
_MIN_TENSOR_BYTES = 26            # name length (8) + rank (4) + type (4) + offset (8) + at least 2 bytes of name


class GGUF(object):
    def __init__(self, path):
        self.path = path
        self.kv = {}
        self.tensors = []                  # [(name, ne (fastest first), type, offset)] in file order
        self.size = os.path.getsize(path)
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("%s is not a GGUF file" % path)
            self.version = self._u(f, "I")
            if self.version not in (2, 3):
                raise ValueError("unsupported GGUF version %d" % self.version)
            n_tensors, n_kv = self._u(f, "Q"), self._u(f, "Q")
            self._fits(f, n_tensors * _MIN_TENSOR_BYTES, "%d tensors" % n_tensors)
            self._fits(f, n_kv * _MIN_KV_BYTES, "%d metadata entries" % n_kv)
            for _ in range(n_kv):
                key = self._str(f, "a metadata key")
                self.kv[key] = self._value(f, self._u(f, "I"), key)
            for _ in range(n_tensors):
                name = self._str(f, "a tensor name")
                nd = self._u(f, "I")
                if nd > MAX_DIMS:
                    raise ValueError("%s: tensor %r claims rank %d, more than the %d a GGUF can carry"
                                     % (path, name, nd, MAX_DIMS))
                self._fits(f, 8 * nd, "the %d dimensions of tensor %r" % (nd, name))
                ne = [self._u(f, "Q") for _ in range(nd)]
                ty = self._u(f, "I")
                off = self._u(f, "Q")
                self.tensors.append((name, ne, ty, off))
            align = int(self.kv.get("general.alignment", 32))
            pos = f.tell()
            self.data_start = (pos + align - 1) // align * align

    def _u(self, f, fmt):
        n = struct.calcsize("<" + fmt)
        if self.size - f.tell() < n:
            raise ValueError("%s is truncated: a %d-byte field at offset %d, but the file has %d left"
                             % (self.path, n, f.tell(), self.size - f.tell()))
        return struct.unpack("<" + fmt, f.read(n))[0]

    def _fits(self, f, n, what):
        """refuse a length the file cannot back, naming it: the alternative is a silent short read or a MemoryError"""
        if n < 0:
            raise ValueError("%s: %s is negative (%d)" % (self.path, what, n))
        left = self.size - f.tell()
        if n > left:
            raise ValueError("%s is truncated or corrupt: %s needs %d bytes at offset %d, but the file has %d left"
                             % (self.path, what, n, f.tell(), left))

    def _str(self, f, what="a string"):
        n = self._u(f, "Q")
        self._fits(f, n, "%s (%d bytes)" % (what, n))
        return f.read(n).decode("utf-8", errors="replace")

    def _value(self, f, ty, key=""):
        if ty == STRING:
            return self._str(f, "the value of %r" % key)
        if ty == ARRAY:
            et, n = self._u(f, "I"), self._u(f, "Q")
            if et == STRING:
                # every element carries at least its own 8-byte length, so an impossible count is refused here
                self._fits(f, 8 * n, "the %d-string array %r" % (n, key))
                return [self._str(f, "element %d of the array %r" % (i, key)) for i in range(n)]
            if et not in _SCALAR:
                raise ValueError("%s: array %r has element type %d, which is not a scalar this reader knows"
                                 % (self.path, key, et))
            fmt = _SCALAR[et]
            size = struct.calcsize("<" + fmt)
            self._fits(f, size * n, "the %d-element array %r" % (n, key))
            return list(struct.unpack("<%d%s" % (n, fmt), f.read(size * n)))
        if ty not in _SCALAR:
            raise ValueError("%s: metadata %r has type %d, which is not a type this reader knows"
                             % (self.path, key, ty))
        return self._u(f, _SCALAR[ty])


def selftest():
    """A tiny valid file reads back; a truncated one and an impossible length each fail by name, not by MemoryError."""
    import tempfile

    def write(path, kv, cut=0, bump=None):
        b = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
        for k, v in kv:
            kb = k.encode()
            b += struct.pack("<Q", len(kb)) + kb
            if isinstance(v, tuple):                        # (element type, count, payload)
                b += struct.pack("<I", ARRAY) + struct.pack("<I", v[0]) + struct.pack("<Q", bump or v[1])
                b += v[2]
            elif isinstance(v, str):
                vb = v.encode()
                b += struct.pack("<I", STRING) + struct.pack("<Q", len(vb)) + vb
            else:
                b += struct.pack("<I", 4) + struct.pack("<I", v)
        b = b[:len(b) - cut] if cut else b
        with open(path, "wb") as f:
            f.write(b)

    td = tempfile.mkdtemp(prefix="kindi-gguf-")
    try:
        good = os.path.join(td, "good.gguf")
        write(good, [("general.name", "kindi"), ("kindi.count", 7),
                     ("kindi.list", (STRING, 2, struct.pack("<Q", 1) + b"a" + struct.pack("<Q", 1) + b"b"))])
        g = GGUF(good)
        assert g.kv["general.name"] == "kindi" and g.kv["kindi.count"] == 7
        assert g.kv["kindi.list"] == ["a", "b"], g.kv["kindi.list"]
        assert g.tensors == [] and g.size == os.path.getsize(good)

        # a file cut three bytes short: the message must say so, and name the file
        short = os.path.join(td, "short.gguf")
        write(short, [("general.name", "kindi"), ("kindi.count", 7)], cut=3)
        try:
            GGUF(short)
        except ValueError as e:
            assert "truncated" in str(e) and os.path.basename(short) in str(e), str(e)
        else:
            raise AssertionError("a truncated header must not read")

        # the case that used to be a MemoryError: an array claiming 2^40 elements it does not have
        absurd = os.path.join(td, "absurd.gguf")
        write(absurd, [("tokenizer.ggml.tokens", (STRING, 2, struct.pack("<Q", 1) + b"a" + struct.pack("<Q", 1) + b"b"))],
              bump=1 << 40)
        try:
            GGUF(absurd)
        except ValueError as e:
            assert "tokenizer.ggml.tokens" in str(e), str(e)
        except MemoryError:
            raise AssertionError("an impossible array length must be refused before it is allocated")
        else:
            raise AssertionError("an array longer than the file must not read")
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)
    return 0


if __name__ == "__main__":
    selftest()
    print("checks passed")
    raise SystemExit(0)
