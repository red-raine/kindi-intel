"""A tiny random pack in the engine's own file format, so the CPU kernels can be tested without a real model.

The published packs are 0.4-14 GB and come from GGUFs that are not part of this repository, which leaves the kernel
work - tiers, the LUT paths, MoE expert routing - with nothing to run against. This builds a pack of the same shape
with three layers instead: same file set, same index format, same tensor names, same pair-code encoding, randomly
initialised with a fixed seed. It is a fixture for equivalence tests (two kernels must agree bit for bit), not a
model: the weights are noise, so the logits mean nothing.

  python -m kindi.synthpack /tmp/tinypack            # 3 layers, width 256, ffn 512
  python -m kindi.synthpack /tmp/tinypack --layers 5 --width 512
"""
import contextlib
import json
import os

import numpy as np

from . import __version__, packio, quant
from .cpu import is_expert, pair_bytes

DIMS = {"layers": 3, "width": 256, "ffn": 512, "heads": 4, "kv_heads": 2, "head_dim": 64, "groups": 2,
        "state_size": 64, "gdn_inner": 128, "rope_dim": 32, "rope_base": 1e4, "full_attention_interval": 2}


def _dims(**kw):
    d = dict(DIMS, **kw)
    for k, n in (("width", d["width"]), ("ffn", d["ffn"]), ("gdn_inner", d["gdn_inner"]),
                 ("heads*head_dim", d["heads"] * d["head_dim"])):
        if n % 128:
            raise SystemExit("kindi synthpack: %s (%d) must be a multiple of 128 (the pair-code group)" % (k, n))
    d["hadamard_block"] = kw.get("hadamard_block") or 128
    return d


def build(out_dir, seed=7, log=print, **kw):
    """write the pack and return (directory, quantised ternary tensors) - the tests compare against the latter

    `paircodes=True` also writes pcodes.bin, the pair-code variant: one expert matrix's GEMV form as bytes, so a paged
    expert stages by memcpy instead of rebuilding it (see `cpu.pair_bytes`)."""
    d = _dims(**kw)
    fold = bool(kw.get("fold", True))
    if not fold:
        d["hadamard_block"] = 0          # an unfolded pack records no block size, as `kindi convert` writes it
    nexp, nact = int(kw.get("experts", 0)), int(kw.get("experts_active", 2))
    rng = np.random.RandomState(seed)
    D, NL, NFF = d["width"], d["layers"], d["ffn"]
    NH, NHKV, HD = d["heads"], d["kv_heads"], d["head_dim"]
    NK, HK = d["groups"], d["state_size"]
    NV = d["gdn_inner"] // HK
    CC = 2 * NK * HK + NV * HK
    vocab = kw.get("vocab", 128)
    # `dense=True`: every layer full attention, which is the layout `kindi convert` writes for a dense Qwen3 file.
    # (`attn_q.weight` is NH*HD rows either way, so the host reads the per-head stride off the matrix: HD here, and
    # 2*HD only for a real qwen35 pack, whose q carries a fused per-head gate.)
    interval = 1 if kw.get("dense") else d["full_attention_interval"]
    d["full_attention_interval"] = interval      # the manifest and the tensor walk must agree on which layers attend
    tern, flt = {}, {}

    def w(rows, cols, scale=0.35):
        return (rng.randn(rows, cols) * scale).astype(np.float32)

    def v(n, scale=0.35):
        return (rng.randn(n) * scale + 1.0).astype(np.float32)

    tern["token_embd.weight"] = w(vocab, D)
    tern["output.weight"] = w(vocab, D)
    flt["output_norm.weight"] = v(D)
    for il in range(NL):
        p = "blk.%d." % il
        flt[p + "attn_norm.weight"] = v(D)
        flt[p + "post_attention_norm.weight"] = v(D)
        if nexp:                                    # a Mixture-of-experts layer: router plus nexp experts
            tern[p + "moe_router.weight"] = w(nexp, D, 0.2)
            for e in range(nexp):
                tern[p + "moe_gate.e%d.weight" % e] = w(NFF // 2, D)
                tern[p + "moe_up.e%d.weight" % e] = w(NFF // 2, D)
                tern[p + "moe_down.e%d.weight" % e] = w(D, NFF // 2)
        else:
            tern[p + "ffn_gate.weight"] = w(NFF, D)
            tern[p + "ffn_up.weight"] = w(NFF, D)
            tern[p + "ffn_down.weight"] = w(D, NFF)
        if (il + 1) % interval == 0:
            tern[p + "attn_q.weight"] = w(NH * HD, D)
            tern[p + "attn_k.weight"] = w(NHKV * HD, D)
            tern[p + "attn_v.weight"] = w(NHKV * HD, D)
            tern[p + "attn_output.weight"] = w(D, NH * HD)
            flt[p + "attn_q_norm.weight"] = v(HD)
            flt[p + "attn_k_norm.weight"] = v(HD)
        else:
            tern[p + "attn_qkv.weight"] = w(CC, D)
            tern[p + "attn_gate.weight"] = w(NV * HK, D)
            tern[p + "ssm_out.weight"] = w(D, NV * HK)
            flt[p + "ssm_beta.weight"] = w(NV, D, 0.1)
            flt[p + "ssm_alpha.weight"] = w(NV, D, 0.1)
            flt[p + "ssm_dt.bias"] = (rng.randn(NV) * 0.1).astype(np.float32)
            flt[p + "ssm_a"] = (-np.abs(rng.randn(NV)) - 0.5).astype(np.float32)
            flt[p + "ssm_conv1d.weight"] = w(4 * CC, 1, 0.2)
            flt[p + "ssm_norm.weight"] = v(HK)

    os.makedirs(out_dir, exist_ok=True)
    quantised = {}
    index = {"tensors": {}, "base": 0, "hadamard": {}}
    bindex = {"tensors": {}}
    bc_off = dpm_off = dblk = dtab_off = f32_off = 0
    want_pc = bool(kw.get("paircodes"))
    pc_off = n_pc = 0
    with open(os.path.join(out_dir, "bcodes.bin"), "wb") as fb, \
            open(os.path.join(out_dir, "dpm.bin"), "wb") as fd, \
            open(os.path.join(out_dir, "dscale.bin"), "wb") as ft, open(os.path.join(out_dir, "f32.bin"), "wb") as ff, \
            (open(os.path.join(out_dir, "pcodes.bin"), "wb") if want_pc else contextlib.nullcontext()) as fp:
        for name in packio.tensor_write_order(tern):
            a, rows = quant._pad8(np.ascontiguousarray(tern[name], dtype=np.float32))
            codes, sc, deq = quant.quantise(a, 128)
            quantised[name] = deq[:tern[name].shape[0], :tern[name].shape[1]]
            fb.write(codes.tobytes())
            table, sidx, _exact = packio.encode(sc)          # the 8-bit form `kindi convert` writes
            sc_q = packio.decode(table, sidx)
            fd.write(sidx.tobytes())
            ft.write(table.tobytes())
            index["tensors"][name] = {"dims": list(tern[name].shape), "type": quant.PTQ1_0, "folded": False,
                                      "rows": rows, "ne0": a.shape[1], "bc_off": bc_off, "dblk": dblk,
                                      "dtab_off": dtab_off, "dtab_n": int(table.size)}
            bindex["tensors"][name] = {"bc_off": bc_off, "dblk": dblk, "dtab_off": dtab_off,
                                       "dtab_n": int(table.size), "rows": rows, "ne0": a.shape[1]}
            if want_pc and is_expert(name):
                Wb, Sb = pair_bytes(codes, sc_q, rows, a.shape[1])
                fp.write(Wb)
                fp.write(Sb)
                bindex["tensors"][name]["pc_off"] = pc_off
                pc_off += len(Wb) + len(Sb)
                n_pc += 1
            bc_off += codes.shape[0] * rows
            dpm_off += sc.shape[0] * rows
            dblk += sc.shape[0] * rows
            dtab_off += 2 * int(table.size)
        for name in sorted(flt):
            a = np.ascontiguousarray(flt[name].astype(np.float32)).ravel()
            ff.write(a.tobytes())
            index["tensors"][name] = {"dims": list(flt[name].shape), "type": quant.F32, "folded": False,
                                      "rows": 1, "ne0": int(a.size), "f32_off": f32_off, "nfloats": int(a.size)}
            f32_off += 4 * a.size
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(out_dir, "bcodes_index.json"), "w") as f:
        json.dump(bindex, f)
    widths = {D, NV * HK, NH * HD, NFF}
    if nexp:
        widths.add(NFF // 2)                      # the experts' own fold width
    for width in sorted(widths):
        s = np.where(rng.rand(D if width == D else width) > 0.5, 1.0, -1.0)[:width].astype(np.float32)
        s.tofile(os.path.join(out_dir, "signs_%d.f32" % width))
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(["t%d" % i for i in range(vocab)], f)
    with open(os.path.join(out_dir, "merges.txt"), "w") as f:
        f.write("#version: 0.2\n")
    man = {"kindi_version": __version__, "pack_format": 2, "model": "kindi-tiny-fixture", "source_file": "synth",
           "source_sha256": None, "tensors": len(tern) + len(flt), "vocab": vocab, "context_length": 256,
           "head": "output.weight",           "hadamard_block": d["hadamard_block"], "fold": fold, "synthetic": True,
           "expert_layout": "gate_up_down_contiguous",
           "rope_scaling": kw.get("rope_scaling"),
           "dimensions": dict({k: d[k] for k in ("layers", "width", "ffn", "heads", "kv_heads", "groups",
                                                 "head_dim", "rope_dim", "state_size", "gdn_inner", "rope_base",
                                                 "full_attention_interval")},
                               **{"experts": nexp, "experts_active": nact} if nexp else {})}
    if n_pc:
        man["paircodes"] = {"tensors": n_pc, "bytes": pc_off}
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    (log or (lambda *a: None))("kindi synthpack: %d ternary + %d f32 tensors -> %s (layers %d, width %d%s)"
                               % (len(tern), len(flt), out_dir, NL, D, ", %d experts" % nexp if nexp else ""))
    return out_dir, quantised, dict(flt)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="kindi.synthpack", description="write a tiny random pack for kernel tests")
    p.add_argument("out")
    p.add_argument("--layers", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--ffn", type=int)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--experts", type=int, default=0, help="make the FFN a Mixture of experts with this many experts")
    p.add_argument("--active", type=int, default=2, help="experts per token (default 2)")
    p.add_argument("--vocab", type=int, default=0, help="tokens in the vocabulary (default 128; need not be a multiple of 8)")
    p.add_argument("--dense", action="store_true", help="every layer full attention (a dense Qwen3 file's shape)")
    p.add_argument("--no-fold", dest="fold", action="store_false", help="unrotated weights: the engine must not fold")
    a = p.parse_args()
    kw = {k: getattr(a, k) for k in ("layers", "width", "ffn", "vocab") if getattr(a, k)}
    if a.dense:
        kw["dense"] = True
    if not a.fold:
        kw["fold"] = False
    if a.experts:
        kw.update(experts=a.experts, experts_active=a.active)
    build(a.out, seed=a.seed, log=print, **kw)
