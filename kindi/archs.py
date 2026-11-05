"""GLM-5.3-Flash (`glm5-next`) and MiniMax M3 (`minimax-m3`): recognise them, price them, and write their pack.

Neither is a CPU pack, and the reason is attention rather than the experts. GLM-5.3-Flash runs MLA with no rotary
width on the key path over 45 blocks, and 34 of those blocks are KDA linear attention with an mHC residual stream and a
DSA indexer that shortlists 2048 keys. MiniMax M3 is GQA (64 query heads, 4 key/value) with partial rotary, per-head QK
norm, and an MSA indexer in a block of its own. `kindi.cpu` implements gated-DeltaNet plus full attention; a manifest
that claimed a CPU pack for either file would be wrong on the first forward pass, so this module does not write one.

What it writes instead is a **catalog pack**: the `kindi_pack.json` every pack has, filled from the header and the tensor
table, plus `tensors.json` naming every tensor with its shape, type, byte count and role in its block. No weight is
copied, so the pack is a few hundred KB whatever the source is, and it answers the questions an import has before a
kernel exists:

  * what the file says it is (geometry, experts, context, rotary base) and what it only appears to be - the two are
    reported separately, because a header can be right about the width and wrong about the experts;
  * where the bytes are. An expert slab per layer is what a paging plan is for, and the catalog prices every role, so
    "how much of this file would a ternary pass even touch" is a sum rather than an estimate;
  * which tensor groups have no slot in this engine's layout, so the work of supporting the architecture is a list,
    not a guess.

The byte table is ggml's block geometry (`TYPES`), and `selftest` cross-checks it against the `gguf` package when that
is installed - a wrong block size is a wrong price, and a price is the whole point of the artifact.

Extending this to a third architecture is one entry in `SPEC` and, if it brings tensor names no rule in `ROLES` knows,
one rule. A name nothing matches is reported as `other` with its bytes, never dropped: an architecture's unusual
tensors are exactly what the next reader needs to see.
"""
import hashlib
import json
import os
import re

from .gguf import GGUF

# ggml tensor type id -> (name, weights per block, bytes per block). PTQ1_0 is this repository's ternary container (26
# bytes of trits and an fp16 scale per 128 weights); TQ1_0/TQ2_0 are BitNet's. A type absent here is reported by number
# with no byte count rather than priced at a guess, so a file of a type this table does not know still catalogues.
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    41: ("Q1_0", 128, 18), 143: ("PTQ1_0", 128, 28),
}

# The two architectures, as the metadata keys they state themselves in. `geometry` uses the field names
# `kindi.recipes` already reports (layers, width, ffn, heads, kv_heads, experts, used, ctx, ...) and adds the ones these
# two need; `required` is what a catalog cannot be written without. The published configuration is in the docstring of
# each entry's `what`, not baked in: what the pack says is what the file declared, and a file that disagrees with its
# model card is the case this exists to catch.
SPEC = {
    "glm5-next": dict(
        what="GLM-5.3-Flash: nope-only MLA (qk_nope 256, qk_rope 0, q_lora 1536, kv_lora 512) over 45 blocks, KDA "
             "linear attention on 34 of them, mHC residual, DSA indexer; 288 experts x 2048 ffn, 8 routed + 1 shared, "
             "3 dense lead layers; 64 heads, 1M context, 154,880-token vocabulary",
        path="catalog pack",
        attention="nope-only MLA + KDA hybrid, DSA indexer",
        geometry=dict(layers="block_count", width="embedding_length", ffn="feed_forward_length",
                      expert_ffn="expert_feed_forward_length", experts="expert_count", used="expert_used_count",
                      shared="expert_shared_count", dense_lead="leading_dense_block_count",
                      heads="attention.head_count", kv_heads="attention.head_count_kv",
                      head_dim="attention.key_length", value_dim="attention.value_length",
                      q_lora="attention.q_lora_rank", kv_lora="attention.kv_lora_rank",
                      ctx="context_length", rope_base="rope.freq_base", rope_dim="rope.dimension_count",
                      eps="attention.layer_norm_rms_epsilon"),
        required=("block_count", "embedding_length", "expert_count", "attention.head_count"),
        blockers=(
            "attention is MLA: the key path is all nope (qk_rope 0) and the keys/values live in a 512-wide latent, "
            "while kindi.cpu implements gated-DeltaNet and full attention",
            "the hybrid blocks run KDA (linear attention with a short convolution state), which has no kernel here",
            "the DSA indexer picks a 2048-key shortlist inside the block; the layout has no tensor for it",
            "the residual stream is mHC (hc_* tensors), not a plain residual add",
        ),
    ),
    "minimax-m3": dict(
        what="MiniMax M3: GQA (64 query, 4 key/value heads, 128 dims) with partial rotary (64 of 128) and per-head QK "
             "norm, MSA indexer in a block of its own; 60 blocks, 128 experts x 3072 ffn with 4 routed + 1 shared and "
             "3 dense lead layers; 6144 wide, 1M context, 200,064-token vocabulary",
        path="catalog pack",
        attention="GQA + per-head QK norm, partial rotary, MSA indexer",
        geometry=dict(layers="block_count", width="embedding_length", ffn="feed_forward_length",
                      expert_ffn="expert_feed_forward_length", experts="expert_count", used="expert_used_count",
                      shared="expert_shared_count", dense_lead="leading_dense_block_count",
                      heads="attention.head_count", kv_heads="attention.head_count_kv",
                      head_dim="attention.key_length", value_dim="attention.value_length",
                      ctx="context_length", rope_base="rope.freq_base", rope_dim="rope.dimension_count",
                      eps="attention.layer_norm_rms_epsilon"),
        required=("block_count", "embedding_length", "expert_count", "attention.head_count"),
        blockers=(
            "attention is MSA: the indexer block selects which keys a query may read, which no kernel here does",
            "the rotary width is half the head (64 of 128 dims) and each head carries its own QK norm tensors; the "
            "CPU kernel builds one table over the whole head",
            "the KV cache is 4 heads against 64 query heads with an indexer on top: the paging plan assumes the "
            "gated-DeltaNet state, not this",
        ),
    ),
}

# tensor name -> its role in the block. Order matters where two rules could match the same name, so the specific ones
# come first. These are llama.cpp's names for the groups the two models published; the set a file actually carries is
# what `survey` reports, and a name no rule knows lands in `other` with its bytes rather than being dropped.
ROLES = (
    ("embedding", r"^token_embd\.weight$"),
    ("head", r"^output\.weight$"),
    ("final_norm", r"^output_norm\.weight$"),
    ("mtp", r"^nextn\."),
    ("router", r"\.ffn_gate_inp\.weight$"),
    ("router.bias", r"\.exp_probs_b\.bias$"),
    ("expert.gate", r"\.ffn_gate_exps\.weight$"),
    ("expert.up", r"\.ffn_up_exps\.weight$"),
    ("expert.down", r"\.ffn_down_exps\.weight$"),
    ("shared_expert.gate", r"\.ffn_gate_shexp\.weight$"),
    ("shared_expert.up", r"\.ffn_up_shexp\.weight$"),
    ("shared_expert.down", r"\.ffn_down_shexp\.weight$"),
    ("mla.q_a", r"\.attn_q_a\.weight$"),
    ("mla.q_a_norm", r"\.attn_q_a_norm\.weight$"),
    ("mla.q_b", r"\.attn_q_b\.weight$"),
    ("mla.kv_a", r"\.attn_kv_a_mqa\.weight$"),
    ("mla.kv_a_norm", r"\.attn_kv_a_norm\.weight$"),
    ("mla.k_b", r"\.attn_k_b\.weight$"),
    ("mla.v_b", r"\.attn_v_b\.weight$"),
    ("attn.q", r"\.attn_q\.weight$"),
    ("attn.k", r"\.attn_k\.weight$"),
    ("attn.v", r"\.attn_v\.weight$"),
    ("attn.output", r"\.attn_output\.weight$"),
    ("attn.q_norm", r"\.attn_q_norm\.weight$"),
    ("attn.k_norm", r"\.attn_k_norm\.weight$"),
    ("indexer.q", r"\.indexer\.q_proj\.weight$"),
    ("indexer.k", r"\.indexer\.k_proj\.weight$"),
    ("indexer.norm", r"\.indexer\.[qk]_norm\.weight$"),
    ("linear_attn", r"\.(ssm_|conv1d|time_mix|attn_qkv|attn_gate)"),
    ("residual.hc", r"\.hc_"),
    ("mlp.gate", r"\.ffn_gate\.weight$"),
    ("mlp.up", r"\.ffn_up\.weight$"),
    ("mlp.down", r"\.ffn_down\.weight$"),
    ("attn_norm", r"\.attn_norm\.weight$"),
    ("post_attn_norm", r"\.post_attention_norm\.weight$"),
    ("ffn_norm", r"\.ffn_norm\.weight$"),
)
_RULES = tuple((role, re.compile(rx)) for role, rx in ROLES)
_BLK = re.compile(r"^blk\.(\d+)\.")

_CHECKS = [0]                                  # how many claims `selftest` made, for docs/TESTING.md to quote


def classify(name):
    """the role a tensor plays in its block, from its llama.cpp name; `other` when no rule matches"""
    for role, rx in _RULES:
        if rx.search(name):
            return role
    return "other"


def group_of(role):
    """the summary line a role is counted under

    Three expert roles and one shared-expert group a model reads as a unit, so they collapse; the router's bias does
    not, because it is a routing correction (`exp_probs_b`) and not a row of the router matrix, and a price list that
    added them together would hide the tensor an aux-loss-free router keeps.
    """
    return role if role == "router.bias" else role.split(".")[0]


def blk_of(name):
    """the block index a tensor belongs to, or None for a file-level tensor (embeddings, head, final norm)"""
    m = _BLK.match(name)
    return int(m.group(1)) if m else None


def tensor_bytes(ne, ty):
    """the bytes a tensor occupies on disk, from its shape and type, or None when either is not priceable

    None is not zero: a type this table does not know, or a row length that is not a whole number of blocks, is a file
    whose price this catalog cannot state, and the summary says how many rows that was.
    """
    t = TYPES.get(int(ty))
    if t is None:
        return None
    n = 1
    for d in ne:
        n *= int(d)
    _, elems, size = t
    if n % elems:
        return None                    # ggml pads a row to a block; a shape that is not whole blocks is not ours to price
    return n // elems * size


def declared(g, arch):
    """the geometry the header states, field by field, with None where the file says nothing

    None is the answer, not a default: an expert count read as zero because the key was missing would turn a 300B MoE
    into a dense model in the manifest.
    """
    spec = SPEC[arch]
    out = {}
    for field, key in spec["geometry"].items():
        v = g.kv.get("%s.%s" % (arch, key))
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        out[field] = (int(v) if float(v).is_integer() else float(v)) if ok else None
    return out


def missing(g, arch):
    """the metadata keys this catalog cannot be written without, by their full name"""
    spec = SPEC[arch]
    return ["%s.%s" % (arch, k) for k in spec["required"] if g.kv.get("%s.%s" % (arch, k)) is None]


def survey(g):
    """what the tensor table says, which is not always what the header says

    Every row keeps its own price so a sum can be checked against the file; the aggregates are what a plan reads. The
    attention flavour of a block is decided by the tensors it carries, not by the metadata: a layer with `attn_kv_a_mqa`
    is MLA whatever the header calls the model, and that is the fact a kernel has to match.
    """
    rows, groups, by_type, flavour = [], {}, {}, {}
    layers = 0
    for name, ne, ty, _off in g.tensors:
        role = classify(name)
        nbytes = tensor_bytes(ne, ty)
        block = blk_of(name)
        rows.append(dict(name=name, shape=list(reversed(ne)), type=TYPES.get(int(ty), (str(int(ty)),))[0],
                         bytes=nbytes, role=role, block=block,
                         expert=role.startswith("expert.")))
        tname = TYPES.get(int(ty), ("type%d" % int(ty),))[0]
        by_type[tname] = by_type.get(tname, 0) + 1
        g_key = group_of(role)
        rec = groups.setdefault(g_key, dict(tensors=0, bytes=0, unpriced=0))
        rec["tensors"] += 1
        if nbytes is None:
            rec["unpriced"] += 1
        else:
            rec["bytes"] += nbytes
        if block is not None:
            layers = max(layers, block + 1)
            if role.startswith("linear_attn"):
                flavour.setdefault(block, set()).add("linear")
            elif role.startswith(("mla.", "attn.q", "attn.k", "attn.v", "attn.output")):
                flavour.setdefault(block, set()).add("softmax")
    linear = sum(1 for f in flavour.values() if "linear" in f)
    softmax = sum(1 for f in flavour.values() if "softmax" in f)
    seen = []
    if any(r["role"].startswith("mla.") for r in rows):
        seen.append("mla")
    if any(r["role"].startswith("attn.q") for r in rows) and "mla" not in seen:
        seen.append("gqa")
    if any(r["role"].startswith("indexer.") for r in rows):
        seen.append("sparse-indexer")
    if linear:
        seen.append("linear-attention")
    return dict(rows=rows, groups=groups, by_type=dict(sorted(by_type.items())), layers=layers,
                blocks=len(flavour), linear_layers=linear, softmax_layers=softmax, attention=seen,
                unpriced=sum(1 for r in rows if r["bytes"] is None),
                total_bytes=sum(r["bytes"] for r in rows if r["bytes"] is not None))


def import_plan(sv, arch):
    """what a ternary pass over this file would cover, and what it would not

    The rule is the pack's own: a weight whose fastest dimension is a whole number of 128-weight blocks can be
    re-coded, which is why the expert slabs of both models qualify - their widths are 4096, 2048 and 3072, and a stacked
    slab is a rank-3 tensor whose last dimension is the row - and why every norm does not. Nothing is converted here:
    this is the price a conversion would be buying, which is the number a "do we support it" question actually needs.
    """
    tern, keep, float_roles = [], [], {}
    for r in sv["rows"]:
        w = r["shape"][-1] if r["shape"] else 0
        if len(r["shape"]) >= 2 and w and w % 128 == 0:
            tern.append(r)
        else:
            keep.append(r)
            g_key = group_of(r["role"])
            float_roles[g_key] = float_roles.get(g_key, 0) + (r["bytes"] or 0)
    return dict(arch=arch, ternary=dict(tensors=len(tern), bytes=sum(r["bytes"] or 0 for r in tern)),
                kept_float=dict(tensors=len(keep), bytes=sum(r["bytes"] or 0 for r in keep),
                                by_group=dict(sorted(float_roles.items()))),
                block_grid=128, ternary_roles=sorted({r["role"] for r in tern}),
                blocked_roles=sorted({r["role"] for r in keep if len(r["shape"]) >= 2}))


def blockers(sv, arch, size):
    """every reason this file is a catalog and not a pack, the file's own facts stated beside the engine's gaps

    The curated list is what the architecture needs and this engine has no kernel for. The two derived ones are read off
    the file: a K-quant source cannot be re-encoded by this build (it decodes K-quants and encodes Q4_0/Q4_K-free
    containers), and a file this size needs a paging plan to be serveable at all.
    """
    out = list(SPEC[arch]["blockers"])
    kq = [t for t in sv["by_type"] if t.endswith("_K")]
    if kq:
        out.append("the source is K-quant (%s): this build decodes K-quants but cannot encode them, so a ternary pack "
                   "would need the float checkpoint or a Q4_0 re-encode first" % "/".join(sorted(kq)))
    out.append("the source is %s, %.0f%% of it expert slabs: serving it needs a paging plan for experts this large, "
               "which the engine does not have" %
               (_gb(size), 100.0 * sv["groups"].get("expert", {}).get("bytes", 0) / max(1, sv["total_bytes"])))
    return out


def _gb(n):
    return "%.1f GB" % (n / 1e9) if n >= 1e9 else "%.1f MB" % (n / 1e6)


def catalog(gguf_path, out_dir, log=print, g=None):
    """write the catalog pack for a `glm5-next` or `minimax-m3` GGUF, and return its manifest

    The manifest is `kindi_pack.json` with `backend: "catalog"`, and it holds no weights: the source is read for its
    header and its tensor table only. `tensors.json` is the inventory, in file order, one row a tensor. `header_sha256`
    covers the first megabyte - the header and the first tensor's opening bytes - which identifies the tensor table
    without reading a 200 GB file to say the same thing.
    """
    from . import __version__
    from .convert import PACK_FORMAT                  # lazily: convert imports this module for the arch check

    g = g or GGUF(gguf_path)
    arch = g.kv.get("general.architecture")
    spec = SPEC.get(arch)
    if spec is None:
        raise SystemExit("kindi archs: %s is %r, not an architecture this module catalogues (have: %s)"
                         % (os.path.basename(gguf_path), arch, ", ".join(sorted(SPEC))))
    miss = missing(g, arch)
    if miss:
        raise SystemExit("kindi archs: %s does not state %s, so its geometry cannot be written down (it may be a "
                         "partially downloaded or repacked file)" % (os.path.basename(gguf_path), ", ".join(miss)))
    geo = declared(g, arch)
    sv = survey(g)
    os.makedirs(out_dir, exist_ok=True)
    with open(gguf_path, "rb") as f:
        head = f.read(1 << 20)
    manifest = {
        "kindi_version": __version__, "pack_format": PACK_FORMAT, "backend": "catalog",
        "arch": arch, "model": g.kv.get("general.name"), "model_path": os.path.abspath(gguf_path),
        "source_file": os.path.basename(gguf_path), "source_bytes": g.size,
        "header_sha256": hashlib.sha256(head).hexdigest(), "header_hash_bytes": len(head),
        "context_length": geo["ctx"], "vocab": len(g.kv.get("tokenizer.ggml.tokens") or []) or None,
        "geometry": geo, "attention": {"declared": spec["attention"], "observed": sv["attention"],
                                       "linear_layers": sv["linear_layers"], "softmax_layers": sv["softmax_layers"],
                                       "blocks_seen": sv["blocks"]},
        "experts": {"count": geo["experts"], "used": geo["used"], "shared": geo["shared"],
                    "feed_forward": geo["expert_ffn"], "dense_lead": geo["dense_lead"],
                    "layout": "stacked 3-D slabs (ffn_*_exps.weight), a router row per expert",
                    "router_bias": any(r["role"] == "router.bias" for r in sv["rows"])},
        "tensors": len(sv["rows"]), "layers_seen": sv["layers"], "tensor_types": sv["by_type"],
        "bytes_by_group": dict((k, v["bytes"]) for k, v in sorted(sv["groups"].items())),
        "total_bytes": sv["total_bytes"], "unpriced_tensors": sv["unpriced"],
        "import_plan": import_plan(sv, arch), "blockers": blockers(sv, arch, g.size),
        "native_status": "catalog only - %s has no %s kernel in this engine" % (arch, "MLA/KDA"
                                                                               if arch == "glm5-next" else "MSA"),
        "chat_template": g.kv.get("tokenizer.chat_template", ""),
    }
    with open(os.path.join(out_dir, "tensors.json"), "w") as f:
        json.dump({"arch": arch, "tensors": sv["rows"]}, f, indent=1)
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log("%s: %s" % (arch, g.kv.get("general.name")))
    log("  %d blocks, %s experts x %s ffn (%s used, %s shared), %s dense lead, %s vocab, ctx %s" %
        (geo["layers"], geo["experts"], geo["expert_ffn"], geo["used"], geo["shared"], geo["dense_lead"],
         manifest["vocab"], geo["ctx"]))
    log("  %d tensors (%s) - %s%s" %
        (len(sv["rows"]), ", ".join("%s %d" % kv for kv in sv["by_type"].items()), _gb(sv["total_bytes"]),
         ", %d unpriced" % sv["unpriced"] if sv["unpriced"] else ""))
    log("  attention seen: %d blocks with a linear (KDA) layer, %d with softmax - declared %s" %
        (sv["linear_layers"], sv["softmax_layers"], spec["attention"]))
    plan = manifest["import_plan"]
    log("  a ternary pass would cover %d tensors, %s of %.0f%%; it keeps %d at f32 (%s)" %
        (plan["ternary"]["tensors"], _gb(plan["ternary"]["bytes"]),
         100.0 * plan["ternary"]["bytes"] / max(1, sv["total_bytes"]), plan["kept_float"]["tensors"],
         _gb(plan["kept_float"]["bytes"])))
    for b in manifest["blockers"]:
        log("  blocker: %s" % b)
    log("catalog pack written to %s (no weights; %d blocker(s))" % (out_dir, len(manifest["blockers"])))
    return manifest


def selftest():
    """Recognition, pricing and the catalog itself, on headers shaped like the two published configurations.

    The fixtures carry the metadata keys and tensor names the models publish, not their weights, so what is checked is
    that the keys are the ones read and the byte table gives the price the geometry implies - a 288-expert layer of
    2048-wide experts is 3 * 288 * 4096 * 2048 weights whatever the file, and that number is asserted rather than
    printed. Nothing here can be checked against a real GLM or MiniMax file: neither is on this machine, and the
    limitation is recorded in docs/IMPORT-CATALOG.md rather than papered over with a fixture that would pass anyway.
    """
    import struct
    import tempfile

    def check(name, ok, detail=""):
        _CHECKS[0] += 1
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    def write_gguf(path, kv, tensors):
        """a header only: the catalog never reads the data section, and that is worth asserting too

        Parts go into a list and are joined once: a 154,880-token vocabulary appended to a bytes object one entry at a
        time is quadratic, and this fixture is written twice.
        """
        out = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", len(tensors)), struct.pack("<Q", len(kv))]
        for k, v in kv:
            kb = k.encode()
            out.append(struct.pack("<Q", len(kb)) + kb)
            if isinstance(v, str):
                vb = v.encode()
                out.append(struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb)
            elif isinstance(v, list):
                out.append(struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(v)))
                out.extend(struct.pack("<Q", len(s)) + s for s in (x.encode() for x in v))
            elif isinstance(v, float):
                out.append(struct.pack("<I", 6) + struct.pack("<f", v))
            else:
                out.append(struct.pack("<I", 4) + struct.pack("<I", v))
        for name, ne, ty, off in tensors:
            nb = name.encode()
            out.append(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(ne)))
            out.extend(struct.pack("<Q", d) for d in ne)
            out.append(struct.pack("<I", ty) + struct.pack("<Q", off))
        b = b"".join(out)
        b += b"\0" * ((-len(b)) % 32) + b"\0" * 32
        with open(path, "wb") as f:
            f.write(b)

    Q4_K, Q2_K, F32, BF16 = 12, 10, 0, 30

    # the byte table against the reference one, when the package is installed. A wrong block size is a wrong price.
    try:
        import gguf as _GQ
        for tid, (nm, elems, size) in TYPES.items():
            try:                                   # PTQ1_0 (143) is this repository's own type; upstream has no name for it
                ref = _GQ.GGML_QUANT_SIZES.get(_GQ.GGMLQuantizationType(tid))
            except ValueError:
                continue
            if ref is None:
                continue
            check("types: %s block geometry" % nm, tuple(ref) == (elems, size), "%r vs %r" % (ref, (elems, size)))
    except ImportError:
        pass

    # --- GLM-5.3-Flash ---------------------------------------------------------------------------------------------
    glm_kv = [("general.architecture", "glm5-next"), ("general.name", "GLM-5.3-Flash"),
              ("glm5-next.block_count", 45), ("glm5-next.embedding_length", 4096),
              ("glm5-next.feed_forward_length", 12288), ("glm5-next.expert_feed_forward_length", 2048),
              ("glm5-next.expert_count", 288), ("glm5-next.expert_used_count", 8),
              ("glm5-next.expert_shared_count", 1), ("glm5-next.leading_dense_block_count", 3),
              ("glm5-next.attention.head_count", 64), ("glm5-next.attention.head_count_kv", 64),
              ("glm5-next.attention.key_length", 256), ("glm5-next.attention.value_length", 256),
              ("glm5-next.attention.q_lora_rank", 1536), ("glm5-next.attention.kv_lora_rank", 512),
              ("glm5-next.context_length", 1048576), ("glm5-next.rope.freq_base", 1000000.0),
              ("glm5-next.attention.layer_norm_rms_epsilon", 1e-05),
              ("tokenizer.ggml.tokens", ["t%d" % i for i in range(154880)])]
    glm_t = [("token_embd.weight", [4096, 154880], BF16, 0), ("output_norm.weight", [4096], F32, 1),
             ("output.weight", [4096, 154880], Q4_K, 2)]
    for L in range(45):
        base = 4096
        if L < 3:
            glm_t += [("blk.%d.ffn_gate.weight" % L, [base, 12288], Q4_K, 0),
                      ("blk.%d.ffn_up.weight" % L, [base, 12288], Q4_K, 0),
                      ("blk.%d.ffn_down.weight" % L, [12288, base], Q4_K, 0)]
        else:
            # llama.cpp stacks the experts in one 3-D slab, fastest dim first: [width, ffn, experts]
            glm_t += [("blk.%d.ffn_gate_inp.weight" % L, [base, 288], F32, 0),
                      ("blk.%d.exp_probs_b.bias" % L, [288], F32, 0),
                      ("blk.%d.ffn_gate_exps.weight" % L, [base, 2048, 288], Q2_K, 0),
                      ("blk.%d.ffn_up_exps.weight" % L, [base, 2048, 288], Q2_K, 0),
                      ("blk.%d.ffn_down_exps.weight" % L, [2048, base, 288], Q2_K, 0),
                      ("blk.%d.ffn_gate_shexp.weight" % L, [base, 2048], Q4_K, 0),
                      ("blk.%d.ffn_up_shexp.weight" % L, [base, 2048], Q4_K, 0),
                      ("blk.%d.ffn_down_shexp.weight" % L, [2048, base], Q4_K, 0)]
        glm_t += [("blk.%d.attn_norm.weight" % L, [base], F32, 0), ("blk.%d.ffn_norm.weight" % L, [base], F32, 0)]
        if L % 4 == 0:                                  # the 11 or so blocks that run MLA
            glm_t += [("blk.%d.attn_q_a.weight" % L, [base, 1536], Q4_K, 0),
                      ("blk.%d.attn_q_a_norm.weight" % L, [1536], F32, 0),
                      ("blk.%d.attn_q_b.weight" % L, [1536, 16384], Q4_K, 0),
                      ("blk.%d.attn_kv_a_mqa.weight" % L, [base, 576], Q4_K, 0),
                      ("blk.%d.attn_kv_a_norm.weight" % L, [512], F32, 0),
                      ("blk.%d.attn_k_b.weight" % L, [512, 16384], Q4_K, 0),
                      ("blk.%d.attn_v_b.weight" % L, [8192, 512], Q4_K, 0),
                      ("blk.%d.attn_output.weight" % L, [16384, base], Q4_K, 0),
                      ("blk.%d.indexer.q_proj.weight" % L, [base, 2048], F32, 0),
                      ("blk.%d.indexer.k_proj.weight" % L, [base, 128], F32, 0),
                      ("blk.%d.hc_attn_post.weight" % L, [base], F32, 0)]
        else:                                           # ... and the rest KDA
            glm_t += [("blk.%d.ssm_conv1d.weight" % L, [4, base], F32, 0),
                      ("blk.%d.ssm_in.weight" % L, [base, base], Q4_K, 0)]
    glm_t.append(("nextn.ffn_gate_exps.weight", [4096, 2048, 288], Q2_K, 0))

    td = tempfile.mkdtemp(prefix="kindi-archs-")
    try:
        glm = os.path.join(td, "glm53.gguf")
        write_gguf(glm, glm_kv, glm_t)
        g = GGUF(glm)
        check("detect: glm5-next is a catalog family", g.kv["general.architecture"] in SPEC)
        geo = declared(g, "glm5-next")
        check("geometry: 45 blocks x 4096 wide, 288 experts x 2048, 8 used", (
            geo["layers"], geo["width"], geo["experts"], geo["expert_ffn"], geo["used"]) == (45, 4096, 288, 2048, 8))
        check("geometry: MLA ranks and the nope-only head", (geo["q_lora"], geo["kv_lora"], geo["head_dim"],
                                                            geo["kv_heads"]) == (1536, 512, 256, 64))
        check("geometry: the 3 dense lead layers and the 1M context", (geo["dense_lead"], geo["ctx"]) == (3, 1048576))
        check("detect: a missing required key is named", missing(g, "glm5-next") == [], str(missing(g, "glm5-next")))

        sv = survey(g)
        check("survey: 45 blocks seen", sv["layers"] == 45, str(sv["layers"]))
        check("survey: expert slabs on the 42 MoE blocks", sv["groups"]["expert"]["tensors"] == 42 * 3,
              str(sv["groups"].get("expert")))
        check("survey: one router and one router bias a MoE block",
              sv["groups"]["router"]["tensors"] == 42 and sv["groups"]["router.bias"]["tensors"] == 42)
        check("survey: the shared expert is its own group, not an `expert`",
              sv["groups"]["shared_expert"]["tensors"] == 42 * 3)
        check("survey: no tensor falls outside the role table", sv["groups"].get("other") is None,
              str(sv["groups"].get("other")))
        check("survey: MLA and KDA separated by the tensors, not the header",
              (sv["softmax_layers"], sv["linear_layers"]) == (12, 33), "%d/%d" % (sv["softmax_layers"],
                                                                                  sv["linear_layers"]))
        check("survey: the indexer is seen", "sparse-indexer" in sv["attention"] and "mla" in sv["attention"])
        check("survey: the MTP head is its own role", any(r["role"] == "mtp" for r in sv["rows"]))

        # the price: a 288-expert slab of 2048-wide experts over a 4096-wide model, three matrices a layer, 42 layers -
        # 2 * 4096 * 2048 * 288 + 2048 * 4096 * 288 weights at Q2_K's 84 bytes a 256
        want = (2 * 4096 * 2048 * 288 + 2048 * 4096 * 288) // 256 * 84 * 42
        check("price: expert bytes are the geometry, not an estimate", sv["groups"]["expert"]["bytes"] == want,
              "%d vs %d" % (sv["groups"]["expert"]["bytes"], want))
        check("price: the groups sum to the total",
              sum(v["bytes"] for v in sv["groups"].values()) == sv["total_bytes"])
        check("price: nothing in the fixture is unpriced", sv["unpriced"] == 0, str(sv["unpriced"]))
        check("price: expert slabs dominate the file", 100.0 * sv["groups"]["expert"]["bytes"] / sv["total_bytes"] > 80)

        plan = import_plan(sv, "glm5-next")
        check("plan: the expert slabs are ternary-eligible", "expert.gate" in plan["ternary_roles"])
        check("plan: norms are not", not any(r.endswith("_norm") for r in plan["ternary_roles"]),
              str(plan["ternary_roles"]))
        check("plan: the head is, a 154,880-row vocabulary being 1,210 blocks of 128",
              "head" in plan["ternary_roles"])

        # --- the pack itself ---------------------------------------------------------------------------------------
        out = os.path.join(td, "glm-pack")
        man = catalog(glm, out, log=lambda *_: None, g=g)
        check("catalog: the backend says what it is", man["backend"] == "catalog")
        check("catalog: it holds no weights",
              not any(os.path.isfile(os.path.join(out, n)) for n in ("index.json", "bcodes.bin", "f32.bin")))
        check("catalog: one inventory row a tensor",
              json.load(open(os.path.join(out, "tensors.json")))["tensors"].__len__() == len(glm_t))
        check("catalog: the manifest counts the tensors", man["tensors"] == len(glm_t))
        check("catalog: it names the missing kernels", len(man["blockers"]) >= 3, str(man["blockers"]))
        check("catalog: it says the K-quant build cannot re-encode",
              any("K-quant" in b for b in man["blockers"]), str(man["blockers"]))
        check("catalog: the geometry in the manifest is the file's",
              man["geometry"]["layers"] == 45 and man["experts"]["count"] == 288)
        check("catalog: the header hash is of the header, not the file",
              man["header_hash_bytes"] == min(1 << 20, os.path.getsize(glm)))

        # a header that states no geometry is refused by key, not written as zeros
        thin = os.path.join(td, "thin.gguf")
        write_gguf(thin, [("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 4096)], [])
        try:
            catalog(thin, os.path.join(td, "thin-pack"), log=lambda *_: None)
        except SystemExit as e:
            check("catalog: a thin header is refused by key", "glm5-next.block_count" in str(e), str(e))
        else:
            raise AssertionError("a header with no block count must not catalogue")

        # --- MiniMax M3 --------------------------------------------------------------------------------------------
        mm_kv = [("general.architecture", "minimax-m3"), ("general.name", "MiniMax-M3"),
                 ("minimax-m3.block_count", 60), ("minimax-m3.embedding_length", 6144),
                 ("minimax-m3.feed_forward_length", 12288), ("minimax-m3.expert_feed_forward_length", 3072),
                 ("minimax-m3.expert_count", 128), ("minimax-m3.expert_used_count", 4),
                 ("minimax-m3.expert_shared_count", 1), ("minimax-m3.leading_dense_block_count", 3),
                 ("minimax-m3.attention.head_count", 64), ("minimax-m3.attention.head_count_kv", 4),
                 ("minimax-m3.attention.key_length", 128), ("minimax-m3.attention.value_length", 128),
                 ("minimax-m3.rope.dimension_count", 64), ("minimax-m3.rope.freq_base", 5000000.0),
                 ("minimax-m3.context_length", 1048576),
                 ("tokenizer.ggml.tokens", ["t%d" % i for i in range(200064)])]
        mm_t = [("token_embd.weight", [6144, 200064], BF16, 0), ("output_norm.weight", [6144], F32, 1),
                ("output.weight", [6144, 200064], Q4_K, 2)]
        for L in range(60):
            base = 6144
            if L < 3:
                mm_t += [("blk.%d.ffn_gate.weight" % L, [base, 12288], Q4_K, 0),
                         ("blk.%d.ffn_down.weight" % L, [12288, base], Q4_K, 0)]
            else:
                mm_t += [("blk.%d.ffn_gate_inp.weight" % L, [base, 128], F32, 0),
                         ("blk.%d.ffn_gate_exps.weight" % L, [base, 3072, 128], Q4_K, 0),
                         ("blk.%d.ffn_up_exps.weight" % L, [base, 3072, 128], Q4_K, 0),
                         ("blk.%d.ffn_down_exps.weight" % L, [3072, base, 128], Q4_K, 0)]
            mm_t += [("blk.%d.attn_norm.weight" % L, [base], F32, 0), ("blk.%d.ffn_norm.weight" % L, [base], F32, 0),
                     ("blk.%d.attn_q.weight" % L, [base, 8192], Q4_K, 0),
                     ("blk.%d.attn_k.weight" % L, [base, 512], Q4_K, 0),
                     ("blk.%d.attn_v.weight" % L, [base, 512], Q4_K, 0),
                     ("blk.%d.attn_q_norm.weight" % L, [128], F32, 0),
                     ("blk.%d.attn_k_norm.weight" % L, [128], F32, 0),
                     ("blk.%d.attn_output.weight" % L, [8192, base], Q4_K, 0),
                     ("blk.%d.indexer.q_proj.weight" % L, [base, 2048], F32, 0),
                     ("blk.%d.indexer.k_norm.weight" % L, [128], F32, 0)]
        mm = os.path.join(td, "m3.gguf")
        write_gguf(mm, mm_kv, mm_t)
        gm = GGUF(mm)
        geom = declared(gm, "minimax-m3")
        check("geometry: MiniMax is 60 blocks x 6144 wide, 128 experts x 3072, 4 used",
              (geom["layers"], geom["width"], geom["experts"], geom["expert_ffn"], geom["used"]) ==
              (60, 6144, 128, 3072, 4))
        check("geometry: 64 query heads over 4 key/value heads, 128 dims, half of them rotary",
              (geom["heads"], geom["kv_heads"], geom["head_dim"], geom["rope_dim"]) == (64, 4, 128, 64))
        svm = survey(gm)
        check("survey: GQA recognised from the projections", "gqa" in svm["attention"], str(svm["attention"]))
        check("survey: every block is softmax attention, none linear", svm["linear_layers"] == 0)
        check("survey: the per-head QK norms are their own roles",
              svm["groups"]["attn"]["tensors"] and any(r["role"] == "attn.q_norm" for r in svm["rows"]))
        check("survey: MiniMax has no unclassified tensor too", svm["groups"].get("other") is None)
        manm = catalog(mm, os.path.join(td, "mm-pack"), log=lambda *_: None, g=gm)
        check("catalog: MiniMax's status names MSA", "MSA" in manm["native_status"], manm["native_status"])
        check("catalog: MiniMax's 200,064-token head is 1,563 blocks of 128",
              "head" in manm["import_plan"]["ternary_roles"], str(manm["import_plan"]["ternary_roles"]))
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)
    return 0


if __name__ == "__main__":
    selftest()
    print("%d checks passed" % _CHECKS[0])
    raise SystemExit(0)
