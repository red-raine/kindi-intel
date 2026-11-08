"""Build a one-layer, real-weight Qwen3-MoE Kindi fixture from a GGUF MoE.

The fixture exists only to answer a layout question: is there a complete 48-layer Qwen3-MoE kindi pack on this
host? There is not. This takes layer 0 (full attention plus all 128 real experts), the first `vocab` embedding and
output rows, and the output norm from the source GGUF, and packs them with kindi.quant.pack. It is a first-layer
function, not a Qwen3 model; comparisons are made between layouts of these same bytes, never against a 48-layer
source. The output pack is written expert-first: gate -> up -> down, with pcodes.bin.

  python tools/qwen_layer0_pack.py B:/models/scratch/qwen3moe-control-q4.gguf B:/models/scratch/qwen-layer0-pack
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.expert_merge.ggufmoe import Moepack
from kindi import quant


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="Qwen3-MoE GGUF")
    p.add_argument("out", help="output kindi pack directory")
    p.add_argument("--vocab", type=int, default=1024, help="rows of embedding/output to keep (default 1024)")
    p.add_argument("--scheme", default="absmean", choices=("absmean", "lsq", "gsq"))
    a = p.parse_args()

    pack = Moepack(a.source)
    d = pack.describe()
    sd = {}

    # Truncated vocabulary: only the first N rows of the source embedding/output tables.
    emb = pack.tensor("token_embd.weight")[: a.vocab]
    out = pack.tensor("output.weight")[: a.vocab]
    sd["token_embd.weight"] = np.ascontiguousarray(emb, dtype=np.float32)
    sd["output.weight"] = np.ascontiguousarray(out, dtype=np.float32)
    sd["output_norm.weight"] = pack.tensor("output_norm.weight").astype(np.float32)

    # Layer 0: the same names the CPU layout reads.
    sd["blk.0.attn_norm.weight"] = pack.tensor("blk.0.attn_norm.weight").astype(np.float32)
    sd["blk.0.post_attention_norm.weight"] = pack.tensor("blk.0.ffn_norm.weight").astype(np.float32)
    sd["blk.0.moe_router.weight"] = pack.tensor("blk.0.ffn_gate_inp.weight").astype(np.float32)
    sd["blk.0.attn_q.weight"] = pack.tensor("blk.0.attn_q.weight").astype(np.float32)
    sd["blk.0.attn_k.weight"] = pack.tensor("blk.0.attn_k.weight").astype(np.float32)
    sd["blk.0.attn_v.weight"] = pack.tensor("blk.0.attn_v.weight").astype(np.float32)
    sd["blk.0.attn_output.weight"] = pack.tensor("blk.0.attn_output.weight").astype(np.float32)
    sd["blk.0.attn_q_norm.weight"] = pack.tensor("blk.0.attn_q_norm.weight").astype(np.float32)
    sd["blk.0.attn_k_norm.weight"] = pack.tensor("blk.0.attn_k_norm.weight").astype(np.float32)
    for e in range(d["experts"]):
        sd["blk.0.moe_gate.e%d.weight" % e] = pack.expert_weight("blk.0.ffn_gate_exps.weight", e).astype(np.float32)
        sd["blk.0.moe_up.e%d.weight" % e] = pack.expert_weight("blk.0.ffn_up_exps.weight", e).astype(np.float32)
        sd["blk.0.moe_down.e%d.weight" % e] = pack.expert_weight("blk.0.ffn_down_exps.weight", e).astype(np.float32)

    dims = {
        "layers": 1,
        "width": d["embd"],
        "ffn": 2 * d["ff_exp"],
        "heads": d["head"],
        "kv_heads": d["head_kv"],
        "head_dim": d["head_dim"],
        "rope_dim": d["head_dim"],
        "rope_base": d["rope_base"],
        "groups": 16,
        "state_size": 128,
        "gdn_inner": d["embd"],
        "full_attention_interval": 1,
        "experts": d["experts"],
        "experts_active": d["active"],
    }
    extra = {
        "arch": d["arch"],
        "head": "output.weight",
        "fold": False,
        "hadamard_block": 0,
        "vocab": a.vocab,
        "context_length": 4096,
        "dimensions": dims,
        "source_model": os.path.basename(a.source),
        "fixture": "qwen_layer0_real_weights",
        "note": "One-layer function built from layer 0 of the source; not a 48-layer model.",
    }
    man = quant.pack(sd, a.out, log=print, name="Qwen3-MoE layer 0 real-weight fixture",
                     source=a.source, scheme=a.scheme, paircodes=True, extra=extra)

    # Files the CPU loader expects beside the weight bytes.
    with open(os.path.join(a.out, "vocab.json"), "w") as f:
        json.dump(pack.g.kv["tokenizer.ggml.tokens"][: a.vocab], f)
    with open(os.path.join(a.out, "merges.txt"), "w", encoding="utf-8") as f:
        for m in pack.g.kv.get("tokenizer.ggml.merges", []):
            f.write(m + "\n")
    for width in (dims["width"], dims["gdn_inner"], dims["heads"] * dims["head_dim"], dims["ffn"], d["ff_exp"]):
        np.ones(int(width), dtype=np.float32).tofile(os.path.join(a.out, "signs_%d.f32" % width))
    print("wrote %s (%d tensors, %d pair-coded matrices)" % (a.out, man["tensors"], man.get("paircodes", {}).get("tensors", 0)))


if __name__ == "__main__":
    main()
