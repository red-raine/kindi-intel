"""kindi convert: supported Ternary-Bonsai-2 PTQ1_0 GGUF -> Kindi weight pack. Exact (same information, new layout).

Ternary tensors (GGUF type 143 = PTQ1_0: per 128 weights, qs[24] | qh[2] | fp16 scale d; 5 trits per byte) are
re-coded 4 trits per byte (byte = t0 + 3 t1 + 9 t2 + 27 t3, t in {0,1,2}, weight = (t-1) * d) and laid out
slot-major ([slot][row]); the scales are transposed to [phase][row]. The engine repacks this at load time into its
own layout (layout.py). The Hadamard rotation Prism ML folded into the weights stays folded: the engine applies the
matching transform to the activations, using the sign vectors stored in the GGUF (prism.hadamard.*).
Float32 tensors are copied verbatim. Vocabulary and BPE merges come from the GGUF tokenizer metadata.

Block scales are stored at 8 bits (pack_format 2): dpm.bin holds one byte a block, an index into the tensor's own fp16
level table in dscale.bin, so a scale costs 1 byte per 128 weights instead of 2 and the pack is 3.5% smaller (211 MB
of a 27B pack's scale bytes). A tensor with no more than 256 distinct scales keeps them exactly; the rest are fitted by
k-means in log2 space. `kindi/packio.py` owns the encoding and decodes it back to one value a block for every reader.

Pack files: bcodes.bin, dpm.bin, dscale.bin, f32.bin, signs_2048.f32, signs_6144.f32, vocab.json, merges.txt,
index.json, bcodes_index.json, kindi_pack.json.

`--paircodes` adds pcodes.bin: every MoE expert matrix in the GEMV pair-code form the CPU engine stages into its
paging slots, so a paged expert is loaded by memcpy rather than rebuilt from the trits at every miss. It is about 6%
of the pack for the experts it covers, and only packs with experts get one.

Two recovered architectures - GLM-5.3-Flash (`glm5-next`) and MiniMax M3 (`minimax-m3`) - have no CPU kernel here, and
this function does not pretend they do: they are routed to `kindi.archs.catalog`, which writes a manifest and a tensor
inventory with no weights in it. Everything below is the pack path a file with ternary weights takes.

`--rope-factor F --rope-window N` builds a pack whose rotary table uses factor F (2 stretches an 8k window to 16k; 1
is the plain table, and drops a block the file declares). What the GGUF asks for is recorded verbatim by
default; `rope_override` is the only place the manifest's `rope_scaling` block is decided, and `kindi.rope` reads it,
so an override is the same key every reader already goes through rather than a second path."""
import hashlib
import json
import os
import time

import numpy as np

from . import __version__, archs, packio
from .cpu import is_expert, pair_bytes
from .gguf import GGUF

PTQ1_0, F32, BF16, Q1_0 = 143, 0, 30, 41
POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint32)
PROFILES = {
    (24, 2048, 6144, 8, 2, 16): "Ternary-Bonsai-2-2B",
    (32, 2560, 9216, 16, 4, 16): "Ternary-Bonsai-2-4B",
    (32, 4096, 12288, 16, 4, 16): "Ternary-Bonsai-2-9B",
    (64, 5120, 17408, 24, 4, 16): "Ternary-Bonsai-2-27B",
}
# dense Qwen3 stacks (`qwen3` arch): no GDN, no Hadamard fold, every layer full attention with QK-norm. The retired
# Bonsai 1.7B and 8B are these, published as 1-bit Q1_0 (a binary block: each weight is +d or -d).
QWEN3_PROFILES = {
    (28, 2048, 6144, 16, 8): "Bonsai-1.7B",
    (36, 4096, 12288, 32, 8): "Bonsai-8B",
}
PACK_FORMAT = 2


def trits_from_raw(raw):
    """raw (nb, 28) uint8 PTQ1_0 blocks -> (nb, 128) trits in {0,1,2} (ggml's staged element order)."""
    nb = raw.shape[0]
    out = np.empty((nb, 128), dtype=np.uint8)
    for n in range(5):
        q = (raw[:, 0:16].astype(np.uint32) * POW3[n]) & 0xFF
        out[:, n * 16:(n + 1) * 16] = (q * 3) >> 8
    for n in range(5):
        q = (raw[:, 16:24].astype(np.uint32) * POW3[n]) & 0xFF
        out[:, 80 + n * 8:88 + n * 8] = (q * 3) >> 8
    for n in range(4):
        q = (raw[:, 24:26].astype(np.uint32) * POW3[n]) & 0xFF
        out[:, 120 + n * 2:122 + n * 2] = (q * 3) >> 8
    return out


def trits_from_q1(bits):
    """(nb, 16) uint8 sign bits of a Q1_0 block -> (nb, 128) trits in {0, 2}.

    ggml's Q1_0 is a binary block: element j is bit (j % 8) of byte (j / 8), and it is +d when the bit is set and -d
    when it is clear (ggml-quants.c, quantize_row_q1_0_ref/dequantize_row_q1_0). The ternary pack holds it exactly -
    only the outer two codes are used, so a Q1_0 tensor converts with no error at all, at the pack's 2 bits a weight
    rather than Q1_0's own 1.125.
    """
    t = np.unpackbits(np.ascontiguousarray(bits), axis=1, bitorder="little")
    return (t * 2).astype(np.uint8)


def phase_bytes(tr):
    """(nb, 128) trits -> (nb, 32) bytes, byte s = t0 + 3 t1 + 9 t2 + 27 t3 of elements 4s .. 4s+3."""
    g = tr.reshape(-1, 32, 4).astype(np.int32)
    return (g[:, :, 0] + 3 * g[:, :, 1] + 9 * g[:, :, 2] + 27 * g[:, :, 3]).astype(np.uint8)


def selftest():
    rng = np.random.RandomState(7)
    tr = rng.randint(0, 3, size=(4096, 128)).astype(np.uint8)
    b = phase_bytes(tr).astype(np.int32)
    back = np.stack([b % 3, (b // 3) % 3, (b // 9) % 3, (b // 27) % 3], axis=2).reshape(-1, 128)
    assert np.array_equal(back, tr), "trit/byte round trip failed"

    # Q1_0: a binary block decodes to the outer two trits, bit j%8 of byte j/8, and never to the middle one. Checked
    # against ggml's own rule for one byte at a time, so a byte-order or bit-order mistake cannot pass.
    bits = rng.randint(0, 256, size=(64, 16)).astype(np.uint8)
    q1 = trits_from_q1(bits)
    want = np.zeros((64, 128), dtype=np.uint8)
    for nb in range(64):
        for j in range(128):
            want[nb, j] = 2 if (int(bits[nb, j // 8]) >> (j % 8)) & 1 else 0
    assert np.array_equal(q1, want), "Q1_0 bit order does not match ggml's block"
    assert set(np.unique(q1)) == {0, 2}, "Q1_0 produced a ternary code it cannot mean"
    # and the two codes mean -d and +d through the same path a PTQ1_0 code takes: (t - 1) * d
    assert np.array_equal(np.unique(q1.astype(np.int16) - 1), [-1, 1]), "Q1_0 codes are not the outer pair"

    # the rope factor a pack is built with. These are checks on the block `kindi.rope` reads, not on a copy of its
    # arithmetic: what matters is that the override composes with a file's block instead of replacing it wholesale, and
    # that the cases which cannot mean anything are refused rather than defaulted.
    from . import rope
    y4 = {"type": "yarn", "factor": 4.0, "original_context_length": 8192,
          "beta_fast": 32.0, "beta_slow": 1.0}
    assert rope_override(y4, None, None) is y4, "no override must leave the file's block untouched"
    assert rope_override(None, None, None) is None, "a file with no block and no override stays plain"
    b2 = rope_override(y4, 2, None, default_window=4096)
    assert b2["factor"] == 2.0, "the override is not applied"
    assert b2["original_context_length"] == 8192, "the file's own window must win over the default"
    assert b2["type"] == "yarn" and b2["beta_fast"] == 32.0, "an override must keep the file's scheme and ramp"
    assert rope_override(None, 2, None, default_window=8192) == {
        "type": "yarn", "factor": 2.0, "original_context_length": 8192,
        "beta_fast": 32.0, "beta_slow": 1.0}, "a file with no block must get yarn from the declared context"
    assert rope_override(None, 1, None, default_window=8192) is None, "factor 1 is the plain table"
    assert rope_override(y4, 1) is None, "factor 1 must drop a block the file declares"
    assert rope_override({"type": "linear", "factor": 4.0}, 2)["type"] == "linear", "linear stays linear"
    w = rope_override(y4, None, 16384)
    assert w["factor"] == 4.0 and w["original_context_length"] == 16384, "a window alone re-scales from it"
    # and the block is the table's input, so the factor reaches cos/sin rather than only the JSON
    assert abs(float(rope.tables(64, 128, 1e6, b2)[0][0, 0]) - (1.0 + 0.1 * np.log(2.0))) < 1e-6, \
        "the override's block is not the table the kernel is given"
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        try:
            rope_override(None, bad, None, default_window=8192)
        except SystemExit:
            pass
        else:
            raise AssertionError("a factor of %r must be refused" % (bad,))
    try:                                    # a window with no block is not a scaling to apply it to
        rope_override(None, None, 8192)
    except SystemExit:
        pass
    else:
        raise AssertionError("a window alone over no block must be refused")


def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(bufsize)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def rope_scaling(g, arch):
    """the `<arch>.rope.scaling.*` block of a GGUF as the manifest records it, or None when the file asks for none

    Split out of `convert` because a pack converted before this key existed can be brought up to date from its own
    GGUF without re-quantising its weights: the block is metadata, and the table it describes is built at load.
    """
    pre = "%s.rope.scaling." % arch
    rs = dict((k[len(pre):], v) for k, v in g.kv.items() if k.startswith(pre))
    if not rs:
        return None
    kind = str(rs.get("type") or "none").lower()
    factor = float(rs.get("factor") or 1.0)
    if kind in ("none", "") or factor == 1.0:
        return None
    out = {"type": kind, "factor": factor,
           "original_context_length": int(rs.get("original_context_length") or 0),
           "beta_fast": float(rs.get("beta_fast") or 32.0),
           "beta_slow": float(rs.get("beta_slow") or 1.0)}
    if rs.get("yarn_attn_factor"):
        out["attn_factor"] = float(rs["yarn_attn_factor"])
    return out


def rope_override(base, factor=None, window=None, default_window=None):
    """the manifest's `rope_scaling` block for a factor the caller names, over what the file declares

    `base` is `rope_scaling(g, arch)`, or None when the file asks for none. The type, the ramp parameters and (unless
    `window` names another) the window stay the file's own; a file with no block gets `yarn`, the scheme whose
    interpolation is what a factor beyond the trained window is for. A factor of 1 is the plain table - exactly what
    `kindi.rope` reads from a missing block - so the block is dropped rather than written, which is how a pack is
    built plain over a file that asks for YaRN. YaRN's crossover is computed *from* the window, so a factor with no
    window anywhere (the file names none and `--rope-window` was not given) is refused: a table built against a
    guessed window is silently the wrong table, and the runtime context is not a substitute for it.
    """
    if factor is None and window is None:
        return base
    if factor is None:                      # a window alone still has to re-scale something
        if not base:
            raise SystemExit("kindi convert: --rope-window needs a scaling to apply it to; the file declares none, "
                             "so give --rope-factor too")
        out = dict(base)
        out["original_context_length"] = int(window)
        return out
    f = float(factor)
    if not np.isfinite(f) or f <= 0.0:
        raise SystemExit("kindi convert: --rope-factor must be positive and finite (got %r)" % (factor,))
    if f == 1.0:
        return None
    out = dict(base or {})
    kind = str(out.get("type") or "none").lower()
    out["type"] = kind if kind in ("yarn", "linear") else "yarn"
    out["factor"] = f
    win = window or out.get("original_context_length") or default_window
    if out["type"] == "yarn":
        if not win:                         # yarn's crossover is a function of the window; linear's is not
            raise SystemExit("kindi convert: --rope-factor %g needs a window to scale from - the file names none, "
                             "so give --rope-window N" % f)
        out["original_context_length"] = int(win)
    elif win:
        out["original_context_length"] = int(win)
    out.setdefault("beta_fast", 32.0)
    out.setdefault("beta_slow", 1.0)
    return out


def convert(gguf_path, out_dir, log=print, paircodes=False, rope_factor=None, rope_window=None):
    t0 = time.time()
    selftest()
    g = GGUF(gguf_path)
    if g.kv.get("general.architecture") == "gemma4":
        from .gemma4 import inspect_model
        info = inspect_model(gguf_path)
        os.makedirs(out_dir, exist_ok=True)
        manifest = {
            "kindi_version": __version__, "pack_format": PACK_FORMAT, "backend": "kindi_gemma4",
            "model": info["name"], "model_id": "nemotron-bolt-e4b" if "Nemotron" in info["name"] else "gemma4",
            "source_file": os.path.basename(gguf_path), "model_path": os.path.abspath(gguf_path),
            "source_sha256": sha256(gguf_path), "context_length": info["context_length"],
            "dimensions": info["dimensions"], "tensor_count": info["tensor_count"],
            "tensor_types": info["tensor_types"], "native_status": "Kindi Gemma 4 CUDA decoder",
        }
        with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        log("recognized Gemma 4 GGUF: %d tensors, %s quantization; Kindi native CUDA decoder" %
            (info["tensor_count"], info["tensor_types"]))
        return manifest
    arch = g.kv.get("general.architecture")
    if arch in archs.SPEC:
        # A checkpoint this engine can recognise and price but not run (MLA/KDA, or MSA attention). The catalog is the
        # artifact an import needs whether or not the kernel exists, and writing a weight pack here would be the one
        # thing worse than writing nothing: a manifest that says "pack" for a file the engine cannot decode a block of.
        for flag, val in (("--paircodes", paircodes), ("--rope-factor", rope_factor),
                          ("--rope-window", rope_window)):
            if val:
                log("kindi convert: %s is a catalog pack (no weights copied), so %s does nothing here" % (arch, flag))
        return archs.catalog(gguf_path, out_dir, log=log, g=g)
    if arch not in ("qwen35", "qwen3"):
        raise SystemExit("kindi convert: expected a qwen35, qwen3, glm5-next, minimax-m3 or Gemma 4 GGUF (found %r)"
                         % (arch,))
    types = {t[2] for t in g.tensors}
    if not types <= {PTQ1_0, Q1_0, F32, BF16}:
        raise SystemExit("kindi convert: needs a ternary (PTQ1_0) or 1-bit (Q1_0) GGUF - found tensor types %s"
                         % sorted(types))
    if arch == "qwen35":
        profile = (g.kv.get("qwen35.block_count"), g.kv.get("qwen35.embedding_length"),
                   g.kv.get("qwen35.feed_forward_length"), g.kv.get("qwen35.attention.head_count"),
                   g.kv.get("qwen35.attention.head_count_kv"), g.kv.get("qwen35.ssm.group_count"))
        if profile not in PROFILES:
            raise SystemExit("kindi convert: unsupported Ternary-Bonsai-2 dimensions %r" % (profile,))
        dims = {"layers": profile[0], "width": profile[1], "ffn": profile[2], "heads": profile[3],
                "kv_heads": profile[4], "groups": profile[5],
                "head_dim": g.kv["qwen35.attention.key_length"],
                "rope_dim": g.kv["qwen35.rope.dimension_count"],
                "state_size": g.kv["qwen35.ssm.state_size"],
                "gdn_inner": g.kv["qwen35.ssm.inner_size"],
                "rope_base": g.kv["qwen35.rope.freq_base"],
                "full_attention_interval": g.kv["qwen35.full_attention_interval"]}
        model_name = PROFILES[profile]
    else:
        profile = (g.kv.get("qwen3.block_count"), g.kv.get("qwen3.embedding_length"),
                   g.kv.get("qwen3.feed_forward_length"), g.kv.get("qwen3.attention.head_count"),
                   g.kv.get("qwen3.attention.head_count_kv"))
        if profile not in QWEN3_PROFILES:
            raise SystemExit("kindi convert: unsupported Qwen3 dimensions %r" % (profile,))
        hdim = int(g.kv["qwen3.attention.key_length"])
        dims = {"layers": profile[0], "width": profile[1], "ffn": profile[2], "heads": profile[3],
                "kv_heads": profile[4], "head_dim": hdim, "rope_dim": hdim,
                # a Qwen3 stack has no gated-DeltaNet layer; the kernel's struct still sizes that state, so the
                # numbers are kept sane rather than absent. The rotary frequencies are the file's own base; whatever
                # scaling it asks for on top of that base is below, and `kindi.rope` builds the table.
                "groups": 1, "state_size": 128, "gdn_inner": profile[1],
                "rope_base": g.kv.get("qwen3.rope.freq_base", 1e6),
                "full_attention_interval": 1}
        model_name = QWEN3_PROFILES[profile]
    # The rotary scaling a file asks for (the retired 1-bit Qwen3 checkpoints ask for YaRN, factor 4, from 8192
    # positions in the 1.7B and 16384 in the 8B). It is recorded verbatim so the engine can build the same table
    # llama.cpp would; `kindi.rope` owns the arithmetic, and a file that names none keeps the plain table.
    # `--rope-factor` / `--rope-window` are the way to build a pack that asks for a different stretch: the block they
    # write is the same key every reader already goes through, so serving 2 is a one-command pack rather than an edit.
    n_ctx_file = g.kv.get("%s.context_length" % arch)
    file_scaling = rope_scaling(g, arch)
    scaling = rope_override(file_scaling, rope_factor, rope_window, default_window=n_ctx_file)
    if rope_factor is not None or rope_window is not None:
        asked = " ".join(x for x in ("--rope-factor %g" % rope_factor if rope_factor is not None else "",
                                     "--rope-window %d" % rope_window if rope_window is not None else "") if x)
        mine = ("asks for %s, factor %.4g, from %d positions"
                % (file_scaling["type"], file_scaling["factor"], file_scaling["original_context_length"])
                if file_scaling else "asks for none")
        log("rope scaling: %s (%s; the file %s)"
            % ("%s, factor %.4g, from %d positions" % (scaling["type"], scaling["factor"],
                                                       scaling["original_context_length"])
               if scaling else "none, plain table", asked, mine))
    elif scaling:
        log("rope scaling: %s, factor %.4g, from %d positions"
            % (scaling["type"], scaling["factor"], scaling["original_context_length"]))
    hd = {k.split("prism.hadamard.")[-1]: v for k, v in g.kv.items() if k.startswith("prism.hadamard.")}
    fold = bool(hd)
    if fold:
        sv = np.array(hd["sign_values"], dtype=np.float32)
        signs, off = {}, 0
        for w in hd["sign_widths"]:
            signs[int(w)] = sv[off:off + int(w)]
            off += int(w)
        folded = set(hd.get("weight_names", []))
        if hd.get("tied_output"):
            folded |= set(hd.get("inverse_weight_names", []))
    else:
        # nothing was rotated into these weights, so the engine must not fold activations either: the manifest turns
        # the transform off, and the sign files exist only because the layout names them (+1 is then unused).
        widths = {dims["width"], dims["gdn_inner"], dims["heads"] * dims["head_dim"], dims["ffn"]}
        signs = dict((int(w), np.ones(int(w), np.float32)) for w in widths)
        folded = set()
    os.makedirs(out_dir, exist_ok=True)
    for width, values in signs.items():
        values.astype("<f4").tofile(os.path.join(out_dir, "signs_%d.f32" % width))
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(g.kv["tokenizer.ggml.tokens"], f)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for m in g.kv["tokenizer.ggml.merges"]:
            f.write(m + "\n")

    index = {"tensors": {}, "base": g.data_start,
             "hadamard": {"tied_output": bool(hd.get("tied_output")), "fold": fold,
                          "sign_widths": [int(x) for x in hd.get("sign_widths", sorted(signs))]}}
    bindex = {"tensors": {}}
    fb = open(os.path.join(out_dir, "bcodes.bin"), "wb")
    fd = open(os.path.join(out_dir, "dpm.bin"), "wb")
    ft = open(os.path.join(out_dir, "dscale.bin"), "wb")
    ff = open(os.path.join(out_dir, "f32.bin"), "wb")
    fp = open(os.path.join(out_dir, "pcodes.bin"), "wb") if paircodes else None
    src = open(gguf_path, "rb")
    bc_off = dpm_off = dblk = dtab_off = f32_off = pc_off = 0
    n_t = n_pc = n_exact = n_tern = max_levels = 0
    max_rel = 0.0
    rename = {".ffn_norm.weight": ".post_attention_norm.weight"} if arch == "qwen3" else {}
    for name, ne, ty, toff in g.tensors:
        for suf, rep in rename.items():        # Qwen3's pre-FFN norm is the engine's post_attention_norm
            if name.endswith(suf):
                name = name[:-len(suf)] + rep
        ne0 = ne[0]
        rows = 1
        for x in ne[1:]:
            rows *= x
        rows_src = rows
        # the pair-code layout walks rows in eights, and a Qwen3 vocabulary (151,669) is not a multiple: pad with
        # zero-weight rows (four zero trits, scale 0), which nothing reads - the head scores `vocab` rows
        if ty in (PTQ1_0, Q1_0):
            rows = (rows + 7) // 8 * 8
        rec = {"dims": list(reversed(ne)), "type": ty, "folded": name in folded, "rows": rows, "ne0": ne0}
        src.seek(g.data_start + toff)
        if ty in (PTQ1_0, Q1_0):
            bpr = ne0 // 128
            blk = 28 if ty == PTQ1_0 else 18    # PTQ1_0: 26 bytes of trits + fp16 scale; Q1_0: fp16 scale + 16 sign bytes
            out = np.zeros((rows, bpr * 32), dtype=np.uint8)
            dm = np.zeros((rows, bpr), dtype=np.uint16)
            RC = 16384
            for c0 in range(0, rows_src, RC):
                c1 = min(rows_src, c0 + RC)
                raw = np.frombuffer(src.read(blk * (c1 - c0) * bpr), dtype=np.uint8).reshape((c1 - c0) * bpr, blk)
                if ty == PTQ1_0:
                    out[c0:c1] = phase_bytes(trits_from_raw(raw)).reshape(c1 - c0, bpr * 32)
                    dm[c0:c1] = np.frombuffer(raw[:, 26:28].tobytes(), dtype=np.uint16).reshape(c1 - c0, bpr)
                else:
                    out[c0:c1] = phase_bytes(trits_from_q1(raw[:, 2:])).reshape(c1 - c0, bpr * 32)
                    dm[c0:c1] = np.frombuffer(raw[:, 0:2].tobytes(), dtype=np.uint16).reshape(c1 - c0, bpr)
            fb.write(np.ascontiguousarray(out.T).tobytes())          # [slot][row]
            sc = np.ascontiguousarray(dm.view(np.float16))           # the fp16 scales the GGUF itself stored
            table, sidx, exact = packio.encode(sc)                   # 8 bits a block, this tensor's own levels
            sc_q = packio.decode(table, sidx)                         # what every reader will decode it back to
            fd.write(np.ascontiguousarray(sidx.T).tobytes())          # [phase][row], one index a block
            ft.write(table.tobytes())
            rec.update({"bc_off": bc_off, "dblk": dblk, "dtab_off": dtab_off, "dtab_n": int(table.size)})
            bindex["tensors"][name] = {"bc_off": bc_off, "dblk": dblk, "dtab_off": dtab_off,
                                       "dtab_n": int(table.size), "rows": rows, "ne0": ne0}
            # pair codes need whole 8-row blocks and whole 128-column groups; an expert that has neither stages
            # through the conversion path instead, so its absence from pcodes.bin is not an error
            if fp is not None and is_expert(name) and rows % 8 == 0 and ne0 % 128 == 0:
                Wb, Sb = pair_bytes(np.ascontiguousarray(out.T), np.ascontiguousarray(sc_q.T), rows, ne0)
                fp.write(Wb)
                fp.write(Sb)
                bindex["tensors"][name]["pc_off"] = pc_off
                pc_off += len(Wb) + len(Sb)
                n_pc += 1
            bc_off += rows * bpr * 32
            dpm_off += rows * bpr
            dblk += rows * bpr
            dtab_off += 2 * int(table.size)
            n_exact += int(exact)
            n_tern += 1
            max_levels = max(max_levels, int(table.size))
            pos = sc[sc > 0]
            if pos.size:                                       # the worst block's relative error, over every tensor
                max_rel = max(max_rel, packio.worst_error(table, float(pos.min()), float(sc.max())))
        else:
            n = rows * ne0
            if ty == BF16:                      # the 27B stores ssm_alpha / ssm_beta in bf16: widen exactly to f32
                v = np.frombuffer(src.read(2 * n), dtype="<u2").astype(np.uint32) << 16
                ff.write(v.astype("<u4").tobytes())
            else:
                ff.write(np.frombuffer(src.read(4 * n), dtype="<f4").tobytes())
            rec.update({"f32_off": f32_off, "nfloats": n})
            f32_off += 4 * n
        index["tensors"][name] = rec
        n_t += 1
        if n_t % 40 == 0:
            log("  %d / %d tensors" % (n_t, len(g.tensors)))
    for h in (fb, fd, ft, ff, fp, src):
        if h is not None:
            h.close()
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(out_dir, "bcodes_index.json"), "w") as f:
        json.dump(bindex, f)
    log("hashing the source file ...")
    manifest = {"kindi_version": __version__, "pack_format": PACK_FORMAT, "model": g.kv.get("general.name"),
                "source_file": os.path.basename(gguf_path), "source_sha256": sha256(gguf_path),
                "tensors": n_t, "bcodes_bytes": bc_off, "dpm_bytes": dpm_off, "f32_bytes": f32_off,
                "block_scales": {"bits": packio.BITS, "file": "dscale.bin", "bytes": dtab_off,
                                 "ternary_tensors": n_tern, "exact_tensors": n_exact, "max_levels": max_levels,
                                 "max_relative_error": round(max_rel, 6)},
                "vocab": len(g.kv["tokenizer.ggml.tokens"]),
                "context_length": n_ctx_file,
                "arch": arch, "dimensions": dict(dims),
                # Hadamard block of the folded inputs (prism v2: 512, v1: e.g. 4096) and the output head tensor.
                # `fold` false means the weights carry no rotation, so the engine must not rotate activations.
                "fold": fold, "hadamard_block": int(hd.get("block_size", 512)) if fold else 0,
                "head": "output.weight" if any(t[0] == "output.weight" for t in g.tensors) else "token_embd.weight",
                # the chat template: the 27B's differs from the 2B/4B/9B one (chat.template_variant)
                "chat_template": g.kv.get("tokenizer.chat_template", "")}
    # the rotary interpolation the table is built with (`kindi.rope`). Absent means the plain table, and that is how
    # every pack that asks for none has always been written, so a pack built plain over a YaRN file carries no key.
    if scaling:
        manifest["rope_scaling"] = scaling
    if rope_factor is not None or rope_window is not None:
        # the file's own block is kept beside the one built, so a pack that disagrees with its checkpoint says so
        # rather than looking like a file that asked for the factor
        manifest["rope_scaling_source"] = {"rope_factor": rope_factor, "rope_window": rope_window,
                                           "file": file_scaling}
    if n_pc:
        manifest["paircodes"] = {"tensors": n_pc, "bytes": pc_off}
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log("pack written to %s (%.0f MB) in %.0fs: %s, %s%s"
        % (out_dir, (bc_off + dpm_off + f32_off + pc_off) / 1e6, time.time() - t0, model_name,
           "Hadamard-folded weights" if fold else "unfolded weights, no activation fold",
           ", %d expert matrices as pair codes (%.1f MB)" % (n_pc, pc_off / 1e6) if n_pc else ""))
    return manifest
