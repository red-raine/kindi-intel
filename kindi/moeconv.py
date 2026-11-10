"""kindi moeconv: a `qwen3moe` GGUF -> a Kindi pack the CPU engine can run, streamed layer by layer.

This is the missing writer behind `kindi convert`'s refusal. The engine already knows how to run a
MoE block - `kindi/cpu.py` takes `blk.L.moe_router.weight` plus `moe_gate.e%d` / `moe_up.e%d` /
`moe_down.e%d` per expert, and `kindi/moe.py` checks its arithmetic - but nothing could *write* those
names from a GGUF. `tools/qwen_layer0_pack.py` proved the mapping on one layer; this walks all of them.

The source is not ternary. `qwen3moe` checkpoints in the wild (and the merge/prune arms this branch
builds from them) are Q4_0 or K-quant, so this is a **lossy re-quantisation to the engine's 2-bit
ternary form**, not an import: a Q4_0 weight that was never trained against a ternary quantiser loses
information here, and `kindi.drift` is the measurement of how much. Nothing in this module may be
called lossless. The two things it is for:

  * a control-vs-arm comparison in the engine's own arithmetic (both arms re-quantised the same way,
    so the difference is the edit under test and not the re-quantisation), which is what the P2
    functional screen needs;
  * a servable pack of a MoE that had none, at 2 bits a weight.

Streaming is not an optimisation. A 30B MoE is 120 GB of float once dequantised, and `kindi.quant.pack`
takes a whole state dict in memory. This writer holds one tensor at a time and appends to the same
files in the same layout `quant.pack` writes (pack_format 1: fp16 group scales in dpm.bin, slot-major
codes in bcodes.bin, everything else verbatim in f32.bin), so `kindi.cpu` loads it through the path it
already loads the synthetic fixtures through.

  python -m kindi.moeconv SOURCE.gguf PACKDIR [--scheme lsq] [--no-paircodes]
"""
import json
import os
import time

import numpy as np

from . import __version__, quant

PACK_FORMAT = 1

# ggml name -> engine name, for the tensors that stay float (norms are read verbatim by the kernel)
NORM = (("attn_norm.weight", "attn_norm.weight"),
        ("ffn_norm.weight", "post_attention_norm.weight"))     # pre-FFN norm is the engine's post_attn_norm
ATTN = (("attn_q.weight", "attn_q.weight"), ("attn_k.weight", "attn_k.weight"),
        ("attn_v.weight", "attn_v.weight"), ("attn_output.weight", "attn_output.weight"))


def _fmt(n):
    return "%.2f GB" % (n / 1e9) if n >= 1e9 else "%.1f MB" % (n / 1e6)


class Writer(object):
    """the three weight files plus their two index files, written in the layout `kindi.cpu` loads"""

    def __init__(self, out_dir, paircodes=True):
        self.bc = open(os.path.join(out_dir, "bcodes.bin"), "wb")
        self.dp = open(os.path.join(out_dir, "dpm.bin"), "wb")
        self.f3 = open(os.path.join(out_dir, "f32.bin"), "wb")
        self.pc = open(os.path.join(out_dir, "pcodes.bin"), "wb") if paircodes else None
        self.index = {"tensors": {}, "base": 0, "hadamard": {}}
        self.bindex = {"tensors": {}}
        self.bc_off = self.dpm_off = self.f32_off = self.pc_off = 0
        self.n_tern = self.n_pc = 0

    def add_ternary(self, name, w, scheme="lsq", paircodes=True):
        from .cpu import pair_bytes
        R0, C0 = w.shape
        ap, rows = quant._pad8(np.ascontiguousarray(w, dtype=np.float32))
        codes, sc, deq = quant.quantise(ap, quant.GROUP, scheme)
        self.bc.write(codes.tobytes())
        self.dp.write(np.ascontiguousarray(sc.view(np.uint16)).tobytes())
        rec = {"dims": [R0, C0], "type": quant.PTQ1_0, "folded": False, "rows": rows, "ne0": C0,
               "bc_off": self.bc_off, "dpm_off": self.dpm_off}
        self.index["tensors"][name] = rec
        brec = {"bc_off": self.bc_off, "dpm_off": self.dpm_off, "rows": rows, "ne0": C0}
        self.bindex["tensors"][name] = brec
        if self.pc is not None and paircodes and quant.is_expert_name(name) and rows % 8 == 0 and C0 % 128 == 0:
            Wb, Sb = pair_bytes(codes, sc, rows, C0)
            self.pc.write(Wb)
            self.pc.write(Sb)
            brec["pc_off"] = self.pc_off
            self.pc_off += len(Wb) + len(Sb)
            self.n_pc += 1
        self.bc_off += rows * C0 // 4
        self.dpm_off += (C0 // quant.GROUP) * rows * 2
        self.n_tern += 1
        return float(((ap - deq) ** 2).mean())

    def add_float(self, name, a):
        f = np.ascontiguousarray(a, dtype=np.float32)
        self.f3.write(f.tobytes())
        rows = int(f.shape[0]) if f.ndim else 1
        self.index["tensors"][name] = {"dims": list(f.shape), "type": quant.F32, "folded": False,
                                       "rows": rows, "ne0": int(f.size // max(1, rows)),
                                       "f32_off": self.f32_off, "nfloats": int(f.size)}
        self.f32_off += 4 * f.size

    def close(self):
        for h in (self.bc, self.dp, self.f3, self.pc):
            if h is not None:
                h.close()


def convert(src, out_dir, log=print, scheme="lsq", paircodes=True, progress=8):
    """write a runnable Kindi pack from a `qwen3moe` GGUF, and return its manifest

    `scheme` is `kindi.quant`'s scale fitter. The default is `lsq` and not `absmean`: a QAT checkpoint
    is already shaped to three levels, so its mean |w| is the right scale, but a Q4_0 or K-quant source
    was shaped to sixteen, and on those the absmean scale undershoots - it is the same finding that made
    `lsq` the default in `quant.pack`. The per-tensor ternary error is summed into the manifest so a
    reader can see what the re-quantisation cost before any walk is run.
    """
    from .gguf import GGUF
    try:
        from tools.expert_merge.ggufmoe import Moepack
    except ImportError:
        from . import _paths                     # noqa: F401  (the repo root is on sys.path in-tree)
        from tools.expert_merge.ggufmoe import Moepack
    t0 = time.time()
    mp = Moepack(src)
    d = mp.describe()
    g = mp.g
    n_layer, nexp, nact = int(d["layers"]), int(d["experts"]), int(d["active"])
    if not nexp:
        raise SystemExit("kindi moeconv: %s declares no experts, so it is not a qwen3moe file" % src)
    hd = int(d["head_dim"])
    dims = {"layers": n_layer, "width": int(d["embd"]), "ffn": int(d["ff_exp"]),
            "heads": int(d["head"]), "kv_heads": int(d["head_kv"]), "head_dim": hd, "rope_dim": hd,
            "rope_base": float(d["rope_base"]), "experts": nexp, "experts_active": nact,
            # a Qwen3-MoE stack is full attention on every block; the kernel still sizes a GDN state,
            # so the numbers are kept sane rather than absent (the same choice `kindi.convert` makes)
            "groups": 1, "state_size": 128, "gdn_inner": int(d["embd"]), "full_attention_interval": 1}
    os.makedirs(out_dir, exist_ok=True)

    vocab = g.kv.get("tokenizer.ggml.tokens") or []
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for m in g.kv.get("tokenizer.ggml.merges", []):
            f.write(m + "\n")
    # no rotation was folded into these weights, so the sign files are +1 and the manifest turns the
    # engine's activation fold off. They exist because the layout names them.
    # The loader checks that a sign file exists for every width the layout rotates. With `fold` off nothing reads
    # them, but a missing one is a refused pack, so the set is built from every width the manifest states and their
    # halves: an extra unused file costs 4 bytes a column and a missing one costs the whole pack.
    widths = {dims["width"], dims["gdn_inner"], dims["ffn"], dims["heads"] * hd, int(d["ff_exp"]),
              dims["state_size"]}
    widths |= {w // 2 for w in list(widths) if w % 2 == 0 and w > 2}
    for width in sorted(widths):
        np.ones(int(width), dtype=np.float32).tofile(os.path.join(out_dir, "signs_%d.f32" % width))

    wr = Writer(out_dir, paircodes=paircodes)
    err_sum, n_err = 0.0, 0
    facts = {}

    def tern(name, src_name=None, w=None):
        nonlocal err_sum, n_err
        w = mp.tensor(src_name or name) if w is None else w
        e = wr.add_ternary(name, w, scheme, paircodes)
        err_sum += e * w.size
        n_err += w.size

    try:
        tern("token_embd.weight")
        for il in range(n_layer):
            p = "blk.%d." % il
            q = "blk.%d." % il
            for a, b in NORM:
                if q + a in mp.info:
                    wr.add_float(p + b, mp.tensor(q + a))
            router = q + "ffn_gate_inp.weight"
            if router not in mp.info:
                raise SystemExit("kindi moeconv: %s has no %s, so layer %d is not an MoE block"
                                 % (src, router, il))
            tern(p + "moe_router.weight", router)
            for e in range(nexp):
                for ggml_suf, eng_suf in (("ffn_gate_exps.weight", "moe_gate"), ("ffn_up_exps.weight", "moe_up"),
                                          ("ffn_down_exps.weight", "moe_down")):
                    nm = q + ggml_suf
                    if nm not in mp.info:
                        raise SystemExit("kindi moeconv: %s has no %s (expert slabs are what this writes)"
                                         % (src, nm))
                    tern(p + "%s.e%d.weight" % (eng_suf, e), w=mp.expert_weight(nm, e))
            for a, b in ATTN:
                if q + a in mp.info:
                    tern(p + b, q + a)
            for a in ("attn_q_norm.weight", "attn_k_norm.weight"):
                if q + a in mp.info:
                    wr.add_float(p + a, mp.tensor(q + a))
            if progress and (il + 1) % progress == 0:
                log("  %d / %d layers, %s of codes" % (il + 1, n_layer, _fmt(wr.bc_off)))
        if "output_norm.weight" in mp.info:
            wr.add_float("output_norm.weight", mp.tensor("output_norm.weight"))
        head = "output.weight" if "output.weight" in mp.info else "token_embd.weight"
        if head == "output.weight":
            tern("output.weight")
    finally:
        wr.close()
    for name, obj in (("index.json", wr.index), ("bcodes_index.json", wr.bindex)):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f)

    man = {"kindi_version": __version__, "pack_format": PACK_FORMAT,
           "model": g.kv.get("general.name") or os.path.basename(src),
           "source_file": os.path.basename(src), "model_path": os.path.abspath(src),
           "arch": "qwen3moe", "backend": "kindi_cpu",
           "tensors": len(wr.index["tensors"]), "ternary_tensors": wr.n_tern,
           "bcodes_bytes": wr.bc_off, "dpm_bytes": wr.dpm_off, "f32_bytes": wr.f32_off,
           "quant": {"scheme": scheme, "group": quant.GROUP, "scale": "fp16", "values": [-1, 0, 1],
                     "lossy": True,
                     "note": "the source was not ternary; this is a 2-bit re-quantisation of it, not an import"},
           "requant_mse": round(err_sum / max(1, n_err), 10), "requant_weights": n_err,
           "fold": False, "hadamard_block": 0, "head": head, "vocab": len(vocab),
           "context_length": g.kv.get("qwen3moe.context_length") or g.kv.get("context_length"),
           "dimensions": dims, "expert_layout": "gate_up_down_contiguous",
           "chat_template": g.kv.get("tokenizer.chat_template", "")}
    if wr.n_pc:
        man["paircodes"] = {"tensors": wr.n_pc, "bytes": wr.pc_off}
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    log("pack written to %s (%s + %s float) in %.0fs: %d layers, %d experts x %d active, %d pair-coded matrices"
        % (out_dir, _fmt(wr.bc_off + wr.dpm_off), _fmt(wr.f32_off), time.time() - t0, n_layer, nexp, nact, wr.n_pc))
    log("requant_mse %.3g over %d weights - this pack is a ternary re-quantisation of a non-ternary source, "
        "so no lossless claim attaches to it" % (man["requant_mse"], n_err))
    return man


def selftest():
    """a one-layer `qwen3moe` GGUF written here, converted, and loaded by the engine

    The fixture is deliberately tiny (width 128, 128-wide experts, 4 experts) so the check is the *mapping*, not a
    model: an expert slab is the axis order the reader slices along, a router row per expert, the pre-FFN norm
    landing on the engine's post_attention_norm, and a pack that the loader accepts. The invariant that matters is
    exact rather than statistical - every packed expert must decode to the ternary quantisation of **that** expert,
    and to no other - because a shape that is square hides a swapped expert index from any error statistic.
    """
    import tempfile
    try:
        from gguf import GGUFWriter
    except ImportError:
        return "moeconv selftest skipped: the `gguf` package is not installed"
    d, ff, E, H, HD = 128, 128, 4, 4, 32
    td = tempfile.mkdtemp(prefix="kindi-moeconv-")
    src = os.path.join(td, "tiny.gguf")
    kv = [("general.architecture", "qwen3moe"), ("general.name", "moeconv-selftest"),
          ("qwen3moe.block_count", 1), ("qwen3moe.embedding_length", d),
          ("qwen3moe.feed_forward_length", 2 * ff), ("qwen3moe.attention.head_count", H),
          ("qwen3moe.attention.head_count_kv", H), ("qwen3moe.attention.key_length", HD),
          ("qwen3moe.attention.value_length", HD), ("qwen3moe.expert_count", E),
          ("qwen3moe.expert_used_count", 2), ("qwen3moe.expert_feed_forward_length", ff),
          ("qwen3moe.rope.freq_base", 10000.0), ("qwen3moe.context_length", 256)]
    w = GGUFWriter(src, "qwen3moe", use_temp_file=True)
    for k, v in kv:
        (w.add_string if isinstance(v, str) else w.add_float32 if isinstance(v, float) else w.add_uint32)(k, v)
    w.add_token_list(["tok%d" % i for i in range(64)])
    rng = np.random.RandomState(3)

    def W(r, c, s=0.2):
        return (rng.randn(r, c) * s).astype(np.float32)

    for name, a in (("token_embd.weight", W(64, d)), ("output_norm.weight", np.ones(d, np.float32)),
                    ("output.weight", W(64, d)), ("blk.0.attn_norm.weight", np.ones(d, np.float32)),
                    ("blk.0.ffn_norm.weight", np.ones(d, np.float32)), ("blk.0.ffn_gate_inp.weight", W(E, d)),
                    ("blk.0.attn_q.weight", W(H * HD, d)), ("blk.0.attn_k.weight", W(H * HD, d)),
                    ("blk.0.attn_v.weight", W(H * HD, d)), ("blk.0.attn_output.weight", W(d, H * HD)),
                    ("blk.0.attn_q_norm.weight", np.ones(HD, np.float32)),
                    ("blk.0.attn_k_norm.weight", np.ones(HD, np.float32))):
        w.add_tensor(name, a)
    # gguf writes dims reversed: an [in, out, expert] slab is a numpy (expert, out, in) array
    gate = W(E * ff * d, 1).reshape(E, ff, d)
    w.add_tensor("blk.0.ffn_gate_exps.weight", gate)
    w.add_tensor("blk.0.ffn_up_exps.weight", W(E * ff * d, 1).reshape(E, ff, d))
    w.add_tensor("blk.0.ffn_down_exps.weight", W(E * d * ff, 1).reshape(E, d, ff))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    out = os.path.join(td, "pack")
    man = convert(src, out, log=lambda *a: None)
    assert man["dimensions"]["experts"] == E and man["dimensions"]["experts_active"] == 2, "expert dims"
    bi = json.load(open(os.path.join(out, "bcodes_index.json")))["tensors"]
    idx = json.load(open(os.path.join(out, "index.json")))["tensors"]
    assert "blk.0.moe_gate.e3.weight" in bi and "blk.0.moe_router.weight" in bi, "engine names"
    assert "blk.0.post_attention_norm.weight" in idx, "pre-FFN norm not renamed"
    bc = np.fromfile(os.path.join(out, "bcodes.bin"), np.uint8)
    dp = np.fromfile(os.path.join(out, "dpm.bin"), np.uint16).view(np.float16)
    for e in range(E):
        r = bi["blk.0.moe_gate.e%d.weight" % e]
        R, C = r["rows"], r["ne0"]
        codes = bc[r["bc_off"]:r["bc_off"] + R * (C // 4)].reshape(C // 4, R)
        sc = dp[r["dpm_off"] // 2:r["dpm_off"] // 2 + (C // 128) * R].reshape(C // 128, R)
        got = quant.dequantise(codes, sc, R, C)[:ff]
        assert np.array_equal(got, quant.quantise(gate[e], quant.GROUP, "lsq")[2]), \
            "expert %d does not decode to its own ternary weights" % e
        other = (e + 1) % E
        assert not np.array_equal(got, quant.quantise(gate[other], quant.GROUP, "lsq")[2]), \
            "expert %d decodes to expert %d: the slab's expert index is wrong" % (e, other)
    from .cpu import Engine
    eng = Engine(out, ctx=64, log=lambda *a: None)          # the loader is the check that the pack is complete
    assert int(eng.dims["experts"]) == E and int(eng.dims["layers"]) == 1, "engine read different dims"
    return "moeconv: %d tensors, %d pair-coded, engine loads; every expert decodes to its own weights" % (
        man["tensors"], man["paircodes"]["tensors"])
