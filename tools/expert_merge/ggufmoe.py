"""A real GGUF mixture-of-experts file, read tensor by tensor.

Nothing here converts anything: the pack stays in the quantisation it shipped in and a tensor is dequantised
only when a measurement asks for it, with `gguf.quants` - llama.cpp's own companion code - doing the unpack. That
is deliberate. A hand-written Q2_K/Q3_K/Q4_K/Q6_K unpack is exactly the kind of code whose bugs look like a
result ("this model has no structure"), so the arithmetic comes from the reference implementation and the thing
that gets checked here is the *layout*: dims are declared fastest-first in GGUF and every consumer wants
[out, in], and a transposed weight still predicts something plausible-looking.

The expert tensors are `[in, out, n_expert]`, so each expert's matrices are one contiguous byte range - which is
what lets `expert_weight` read one of 128 experts without touching the other 127.
"""
import numpy as np

try:
    from gguf import GGMLQuantizationType
    from gguf import quants as GQ
except ImportError:                                          # checked at use, not at import: the gate imports every
    GGMLQuantizationType = None                              # module, and a tool that needs a package should not take
    GQ = None                                                # the whole tree down on a host that lacks it

from kindi.gguf import GGUF

# GGUF tensor type ids as they appear in the file
_TYPES = {int(t): t for t in GGMLQuantizationType} if GGMLQuantizationType is not None else {}


class Moepack(object):
    def __init__(self, path):
        if GQ is None:
            raise RuntimeError("this reader needs the `gguf` package (llama.cpp's Python companion): pip install gguf")
        self.path = path
        self.g = GGUF(path)
        self.info = {name: (tuple(ne), int(ty), int(off)) for name, ne, ty, off in self.g.tensors}
        self._raw = np.memmap(path, dtype=np.uint8, mode="r")
        self.arch = self.g.kv.get("general.architecture", "")

    # --- metadata ---------------------------------------------------------------------------------------------
    def meta(self, key, default=None):
        return self.g.kv.get("%s.%s" % (self.arch, key), default)

    @property
    def n_layer(self):
        return int(self.meta("block_count"))

    @property
    def n_embd(self):
        return int(self.meta("embedding_length"))

    @property
    def n_head(self):
        return int(self.meta("attention.head_count"))

    @property
    def n_head_kv(self):
        return int(self.meta("attention.head_count_kv"))

    @property
    def head_dim(self):
        return int(self.meta("attention.key_length"))

    @property
    def n_expert(self):
        return int(self.meta("expert_count"))

    @property
    def n_expert_used(self):
        return int(self.meta("expert_used_count"))

    # --- a kept-router pack -------------------------------------------------------------------------------------
    # `mergepack --keep-router` writes the source's 128 router rows over 64 merged stacks, so the file has two
    # expert counts: `expert_count` is what the stacks hold, and `kindi.router_rows` is how many routing slots the
    # router scores. A normal pack has neither key and routes over `expert_count` slots, exactly as before.
    @property
    def router_rows(self):
        """Routing slots the router scores: the source's when the router was kept, else `expert_count`."""
        return int(self.g.kv.get("kindi.router_rows") or self.n_expert)

    @property
    def router_used(self):
        """Slots a token selects at the router: `expert_used_count`, unless a kept-router pack says otherwise."""
        return int(self.g.kv.get("kindi.router_used") or self.n_expert_used)

    @property
    def router_map(self):
        """slot -> stack, or None for a normal pack. A pack with a map routes over `router_rows` logits.

        This is the key that makes a kept-router pack not-a-normal-MoE: without it the router's 128 rows would be
        read against 64 stacks by index, which would be a model with 64 experts and a doubled router, not this one.
        """
        m = self.g.kv.get("kindi.expert_map")
        return None if m is None else np.asarray(m, dtype=np.int64)

    @property
    def n_ff_exp(self):
        return int(self.meta("expert_feed_forward_length"))

    @property
    def rope_base(self):
        return float(self.meta("rope.freq_base", 10000.0))

    @property
    def eps(self):
        return float(self.meta("attention.layer_norm_rms_epsilon", 1e-6))

    def describe(self):
        return dict(path=self.path, arch=self.arch, layers=self.n_layer, embd=self.n_embd,
                    head=self.n_head, head_kv=self.n_head_kv, head_dim=self.head_dim,
                    experts=self.n_expert, active=self.n_expert_used, ff_exp=self.n_ff_exp,
                    rope_base=self.rope_base, eps=self.eps)

    # --- bytes ------------------------------------------------------------------------------------------------
    @staticmethod
    def _span(ty, n_values):
        """bytes a run of `n_values` values takes in type `ty` (block-quantised types pad to a whole block)"""
        t = _TYPES[ty]
        if t in (GGMLQuantizationType.F32, GGMLQuantizationType.I32):
            return n_values * 4
        if t in (GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
            return n_values * 2
        bs, ts = GQ.GGML_QUANT_SIZES[t]
        if n_values % bs:
            raise ValueError("%d values is not a whole number of %s blocks of %d" % (n_values, t.name, bs))
        return n_values // bs * ts

    def _dequant(self, ty, start, n_values):
        t = _TYPES[ty]
        raw = self._raw[start:start + self._span(ty, n_values)]
        return GQ.dequantize(raw, t)

    def tensor(self, name):
        """[out, in] for a matrix, [n] for a vector, [n_expert, out, in] for an expert stack."""
        ne, ty, off = self.info[name]
        n = int(np.prod(ne))
        flat = self._dequant(ty, self.g.data_start + off, n)
        return flat.reshape(tuple(reversed(ne)))

    def expert_weight(self, name, e):
        """One expert's matrix, out of the `[in, out, n_expert]` stack: [out, in].

        Contiguous by construction: the expert index is the slowest dimension, so expert `e` starts at
        `e * in * out` values into the tensor. 128 experts therefore never get dequantised to read one.
        """
        ne, ty, off = self.info[name]
        n_in, n_out, n_exp = ne
        if not 0 <= e < n_exp:
            raise IndexError("expert %d outside %d" % (e, n_exp))
        per = n_in * n_out
        start = self.g.data_start + off + e * self._span(ty, per)
        return self._dequant(ty, start, per).reshape(n_out, n_in)

    def rows(self, name, ids):
        """Selected rows of a 2-D tensor, dequantising only those rows (the embedding table is 151,936 rows)."""
        ne, ty, off = self.info[name]
        n_in, n_out = ne
        per = n_in
        out = np.empty((len(ids), n_in), dtype=np.float32)
        for j, i in enumerate(ids):
            start = self.g.data_start + off + int(i) * self._span(ty, per)
            out[j] = self._dequant(ty, start, per)
        return out

    def _block(self, ty):
        """(values, bytes) of this type's quantisation block"""
        return GQ.GGML_QUANT_SIZES[_TYPES[ty]]

    def decodes(self, start, ty, n_blocks=512):
        """does the run of `n_blocks` blocks at this byte offset dequantise to finite, sane weights?

        One test for the whole class of damage this reader has to survive: a block scale that is not finite (what
        random bytes look like to Q2_K) or a magnitude no trained weight has. Layout-independent, because it asks
        the same dequantiser the measurement uses. `start` must sit on a block boundary - every caller derives it
        from a tensor offset and a whole number of blocks, which is what keeps a probe from reading mid-block and
        calling good weights broken.
        """
        t = _TYPES[ty]
        bs, ts = self._block(ty)
        nb = max(1, int(n_blocks))
        raw = self._raw[start:start + nb * ts]
        if raw.size < nb * ts:
            return False
        with np.errstate(invalid="ignore", over="ignore"):
            v = GQ.dequantize(raw, t)
        return bool(np.isfinite(v).all() and np.abs(v).max() < 1e3)

    def _break_in(self, start, span, ty, probe_bytes=1 << 16):
        """first block boundary in [start, start+span) after which the payload stops decoding"""
        bs, ts = self._block(ty)
        nblk = max(1, span // ts)
        probe = min(nblk, max(1, probe_bytes // ts))
        if not self.decodes(start, ty, probe):
            return start
        lo, hi = 0, nblk - probe
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.decodes(start + mid * ts, ty, probe):
                lo = mid
            else:
                hi = mid - 1
        return start + (lo + probe) * ts

    def declared_size(self):
        """the file size the index promises: data_start plus every tensor's span"""
        return self.g.data_start + max(self.info[n][2] + self._span(self.info[n][1], int(np.prod(self.info[n][0])))
                                       for n in self.info)

    def payload_report(self, probe=512):
        """where the payload stops being this model, if it does

        The failure this exists for, found on this machine: an 11.26 GB MoE GGUF whose tensor table is intact and
        whose first 3.6 GB decode, then 7.6 GB of random bytes, with 59 MB more than the model needs - a resume
        that wrote at the wrong offset. Every consumer of the file needs that answer *before* it measures, because
        NaN experts dequantise silently and a model with ten readable layers looks like a result.
        """
        size = int(self._raw.size)
        declared = self.declared_size()
        rep = dict(size=size, declared=declared, extra=max(0, size - declared), short=max(0, declared - size),
                   first_bad=None, break_offset=None, bad_layer=None)
        for name, (ne, ty, off) in sorted(self.info.items(), key=lambda kv: kv[1][2]):
            start = self.g.data_start + off
            span = self._span(ty, int(np.prod(ne)))
            bs, ts = self._block(ty)
            nblk = max(1, span // ts)
            pb = min(max(1, probe), nblk)
            ok = (start + span <= size and self.decodes(start, ty, pb)
                  and self.decodes(start + (nblk - pb) * ts, ty, pb))
            if not ok:
                rep["first_bad"] = name
                rep["break_offset"] = start if start + span > size else self._break_in(start, span, ty)
                m = name.split(".")
                if len(m) > 2 and m[0] == "blk" and m[1].isdigit():
                    rep["bad_layer"] = int(m[1])
                break
        n = 0
        for il in range(self.n_layer):
            if not self.layer_valid(il):
                break
            n += 1
        rep["layers_ok"] = n
        rep["ok"] = rep["first_bad"] is None
        return rep

    def require_complete(self, log=None, allow_partial=False):
        """refuse to model a damaged pack unless the caller asked for a partial one, loudly

        `valid_layers` used to shorten the model and say nothing a caller could not ignore; the measurement then
        covered 13 of 48 layers and read like a property of the model. Silence is the bug.
        """
        rep = self.payload_report()
        if rep["ok"]:
            return rep
        msg = ("%s is not a complete pack: %s at byte %d of %d (%.1f%% of the file decodes), %d of %d layers"
               % (self.path, rep["first_bad"], rep["break_offset"], rep["size"],
                  100.0 * rep["break_offset"] / max(rep["size"], 1), rep["layers_ok"], self.n_layer))
        if not allow_partial:
            raise RuntimeError(msg + "\n  repair it with: python -m tools.expert_merge.fetch --repair FILE --url URL")
        if log:
            log("  WARNING: " + msg)
        return rep

    def layer_valid(self, il, probe=512):
        """does this layer's data decode to plausible weights?

        Called because a MoE GGUF on this machine turned out to be a partial download: the tensor table is intact
        and every byte range is where the table says, but the payload past the last fully written layer is noise.
        A quantised tensor block always dequantises to values bounded by its own block scale, so "finite and not
        enormous" is a cheap, layout-independent test - and the alternative, NaN through 34 layers, is a
        measurement that looks like a model with no structure rather than a broken file.
        """
        s = "blk.%d." % il
        names = [s + n for n in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                 "attn_q_norm.weight", "attn_k_norm.weight", "attn_norm.weight", "ffn_norm.weight",
                                 "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                 "ffn_down_exps.weight")]
        for name in names:
            if name not in self.info:
                return False
            ne, ty, off = self.info[name]
            n = int(np.prod(ne))
            span = self._span(ty, n)
            # both ends of the tensor: a part-written file cuts a byte range, not a tensor, so a layer can decode
            # at its start and be noise at its end - and it is the last two layers before the break that sit on it.
            start = self.g.data_start + off
            bs, ts = self._block(ty)
            nblk = max(1, span // ts)
            pb = min(max(1, probe), nblk)
            try:
                if not self.decodes(start, ty, pb):
                    return False
                if not self.decodes(start + (nblk - pb) * ts, ty, pb):
                    return False
            except Exception:
                return False
        return True

    def valid_layers(self, log=None):
        """every layer that decodes, and the prefix length: a truncated file is usable up to the break."""
        good = [self.layer_valid(il) for il in range(self.n_layer)]
        n = 0
        for ok in good:
            if not ok:
                break
            n += 1
        if log and n < self.n_layer:
            log("  WARNING: %d of %d layers decode; layer %d is the first that does not (truncated or damaged\n"
                "           payload). Layers 0-%d are used and nothing is claimed beyond them."
                % (n, self.n_layer, n, n - 1))
        return n, good

    def type_names(self):
        counts = {}
        for _n, (ne, ty, _off) in self.info.items():
            counts[_TYPES[ty].name] = counts.get(_TYPES[ty].name, 0) + 1
        return counts
