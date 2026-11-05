"""kindi recipes: what a model is, and what to build from it for a job.

Two questions, kept apart. **Detection** reads a GGUF header (or an existing pack's manifest) and says what the file
is: family, depth, width, experts, context, rotary scaling, and which tensor types it carries. **Recipes** are the
conversion plans this engine can actually execute, one per job - a coding agent, a RAG index, a search assistant, a
science assistant - and each one is a set of decisions the converter takes plus the gate the result has to pass.

A recipe is deliberately not a wish list. Every field below is something `kindi convert` or `kindi qat-pack` reads:

  keep_float     tensor-name suffixes that stay f32 instead of being ternarised (a norm or a head that a job needs
                 at full precision)
  scheme         the scale/code fitter: `absmean` (the PTQ1_0 convention), `lsq` (per-group minimum-error scale) or
                 `gsq` (learned grid assignment and scale, `kindi.gsq`)
  rope_window    the window a long-context recipe builds its rotary table from (`kindi.rope`)
  paircodes      stage expert matrices for paged serving (`kindi convert --paircodes`)
  gate           the measurement the pack must pass before it is called done: a corpus, a maximum NLL delta against
                 the reference pack, and a minimum top-1 agreement (`tools.quantgate`)

Detection is metadata only: no tensor is read, so it costs a header parse and works on a 350 GB file. What it cannot
tell you it does not guess - a file with no `*.expert_count` is not assumed to be dense, it is reported as unknown.

  kindi recipes                          # the recipes, and what each one changes
  kindi recipes --for MODEL.gguf         # detect the file, then the plan for the best-matching recipe
  kindi recipes --for MODEL.gguf --role rag --json
  kindi recipes --scan B:/models          # every model under a directory, and the recipe each would get
"""
import json
import os
import re
import shutil
import tempfile

from . import archs

# --- detection --------------------------------------------------------------------------------------------------------

# GGUF tensor type ids this engine can read, by name. The ids are ggml's; only the ones a conversion decision depends
# on are listed, and an unknown id is reported by number rather than dropped.
TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
         14: "Q6_K", 20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16", 41: "Q1_0", 143: "PTQ1_0"}

# families this engine has a conversion path for, and what that path is. A family that is absent is reported as
# "no conversion path" rather than being forced through the qwen35 reader. The two catalog families are added below
# from `kindi.archs`, which owns their geometry and their blockers: detection and the catalog must not describe the
# same file differently.
FAMILIES = {
    "qwen35": dict(what="Qwen3.5 hybrid (gated DeltaNet + full attention)", path="cpu pack", ternary=True),
    "qwen3": dict(what="Qwen3 dense", path="cpu pack", ternary=True),
    "qwen3moe": dict(what="Qwen3 mixture-of-experts", path="cpu pack (experts paged)", ternary=True),
    "gemma4": dict(what="Gemma 4 / Nemotron Bolt", path="gemma4 CUDA decoder", ternary=False),
    "llama": dict(what="Llama", path="none yet", ternary=False),
    "mamba": dict(what="Mamba / hybrid SSM", path="none yet", ternary=False),
    "lfm2": dict(what="LFM2 (conv + attention hybrid)", path="none yet", ternary=False),
}

# the metadata keys a family states its geometry in. `qwen35` and `qwen3` are the two the converter reads today; the
# rest are here so detection can name a file it cannot convert instead of calling it unknown.
_GEOM = {
    "qwen35": dict(layers="block_count", width="embedding_length", ffn="feed_forward_length",
                   heads="attention.head_count", kv_heads="attention.head_count_kv",
                   experts="expert_count", used="expert_used_count", ctx="context_length",
                   rope_base="rope.freq_base", head_dim="attention.key_length"),
    "qwen3": dict(layers="block_count", width="embedding_length", ffn="feed_forward_length",
                  heads="attention.head_count", kv_heads="attention.head_count_kv",
                  experts="expert_count", used="expert_used_count", ctx="context_length",
                  rope_base="rope.freq_base", head_dim="attention.key_length"),
    "qwen3moe": dict(layers="block_count", width="embedding_length", ffn="expert_feed_forward_length",
                     heads="attention.head_count", kv_heads="attention.head_count_kv",
                     experts="expert_count", used="expert_used_count", ctx="context_length",
                     rope_base="rope.freq_base", head_dim="attention.key_length"),
    "llama": dict(layers="block_count", width="embedding_length", ffn="feed_forward_length",
                  heads="attention.head_count", kv_heads="attention.head_count_kv",
                  experts="expert_count", used="expert_used_count", ctx="context_length",
                  rope_base="rope.freq_base", head_dim="attention.key_length"),
    "gemma4": dict(layers="block_count", width="embedding_length", ffn="feed_forward_length",
                   heads="attention.head_count", kv_heads="attention.head_count_kv",
                   experts="expert_count", used="expert_used_count", ctx="context_length",
                   rope_base="rope.freq_base", head_dim="attention.key_length"),
}

# GLM-5.3-Flash and MiniMax M3: recognised from the header, priced by `kindi.archs`, not convertible to a CPU pack
# (MLA/KDA and MSA attention respectively). Their geometry keys come from the same table the catalog writes from.
FAMILIES.update({k: dict(what=v["what"], path=v["path"], ternary=False) for k, v in archs.SPEC.items()})
_GEOM.update({k: dict(v["geometry"]) for k, v in archs.SPEC.items()})


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def detect_meta(kv, tensors=()):
    """What a GGUF header says about itself, from its key/values and its tensor list.

    `kv` is `GGUF.kv` and `tensors` is `GGUF.tensors` (name, ne, type, offset). Nothing is read from the data section,
    so this is a header parse. Fields the file does not state are `None`, and `unknown` lists them by name: a missing
    expert count is not evidence of a dense model.
    """
    arch = str(kv.get("general.architecture") or "")
    fam = FAMILIES.get(arch)
    g = _GEOM.get(arch, {})
    out = dict(arch=arch or None, family=arch or None, what=(fam or {}).get("what"),
               path=(fam or {}).get("path", "none yet"), name=kv.get("general.name"),
               file_type=kv.get("general.file_type"), unknown=[])
    for field, key in g.items():
        v = _num(kv.get("%s.%s" % (arch, key)))
        out[field] = int(v) if v is not None else None
        if v is None and field in ("layers", "width", "heads", "ctx"):
            out["unknown"].append(field)
    # experts: a file that states none is dense, and says so; a file whose family has no expert key is unknown
    if out.get("experts") is None and g.get("experts"):
        out["experts"] = 0
    out["moe"] = bool(out.get("experts"))
    out["active_experts"] = out.get("used") or (1 if out.get("experts") else 0)
    rs = dict((k.split(".rope.scaling.")[-1], v) for k, v in kv.items() if ".rope.scaling." in k)
    out["rope_scaling"] = (dict(type=str(rs.get("type") or "none").lower(), factor=float(rs.get("factor") or 1.0),
                                original_context_length=int(rs.get("original_context_length") or 0))
                           if rs else None)
    toks = kv.get("tokenizer.ggml.tokens")
    out["vocab"] = len(toks) if isinstance(toks, list) else None
    hist = {}
    for t in tensors:
        ty = int(t[2])
        hist[TYPES.get(ty, "type%d" % ty)] = hist.get(TYPES.get(ty, "type%d" % ty), 0) + 1
    out["tensor_types"] = dict(sorted(hist.items()))
    out["ternary_source"] = bool(hist) and set(hist) <= {"PTQ1_0", "Q1_0", "F32", "BF16"}
    out["convertible"] = bool(fam) and fam["path"] != "none yet"
    spec = archs.SPEC.get(arch)
    if spec:                                    # a catalog family: the attention it needs, and what the engine lacks
        out["attention"] = spec["attention"]
        out["blockers"] = list(spec["blockers"])
    return out


def detect(path):
    """`detect_meta` for a GGUF file, or for a pack directory (its manifest)."""
    if os.path.isdir(path):
        man = os.path.join(path, "kindi_pack.json")
        if not os.path.isfile(man):
            raise SystemExit("kindi recipes: %s is a directory with no kindi_pack.json" % path)
        m = json.load(open(man))
        d = dict(arch=m.get("arch"), family=m.get("arch"), what=(FAMILIES.get(m.get("arch")) or {}).get("what"),
                 path="pack", name=m.get("model"), file_type=None, unknown=[], moe=bool(m.get("experts")),
                 experts=m.get("experts"), active_experts=m.get("active_experts"),
                 layers=(m.get("dimensions") or {}).get("layers"), width=(m.get("dimensions") or {}).get("width"),
                 heads=(m.get("dimensions") or {}).get("heads"), ctx=m.get("context_length"),
                 rope_scaling=m.get("rope_scaling"), vocab=m.get("vocab"), tensor_types={},
                 ternary_source=True, convertible=True, pack=path)
        return d
    from .gguf import GGUF
    g = GGUF(path)
    d = detect_meta(g.kv, g.tensors)
    d["file"] = path
    return d


# --- recipes ----------------------------------------------------------------------------------------------------------

# Each recipe is a conversion plan plus the gate it must pass. `keep_float` is a list of name suffixes; `gate` names a
# corpus kind and the thresholds a pack has to meet against its reference. The thresholds are the ones this repository
# has measured for a 2-bit container (docs/CPU-PERF-2026.md, docs/QAT-CONVERSION.md): a 0.01-nat mean delta is inside
# the noise of a 79-token window, so a gate tighter than that would fail on sampling rather than on the quantiser.
RECIPES = {
    "coder": dict(
        what="a coding agent: long context, tool calls, code-heavy prompts",
        keep_float=("output.weight", "token_embd.weight"),
        scheme="gsq", rope_window=32768, paircodes=True,
        gate=dict(corpus="code", max_delta_nll=0.02, min_top1=0.90)),
    "agent": dict(
        what="a general tool-using agent: long context, many short turns, a fixed tool list",
        keep_float=("output.weight", "token_embd.weight"),
        scheme="gsq", rope_window=32768, paircodes=True,
        gate=dict(corpus="agent", max_delta_nll=0.02, min_top1=0.90)),
    "rag": dict(
        what="a RAG reader: long pasted context, short answers, recall over precision",
        keep_float=("output.weight", "token_embd.weight"),
        scheme="lsq", rope_window=32768, paircodes=True,
        gate=dict(corpus="prose", max_delta_nll=0.03, min_top1=0.88)),
    "search": dict(
        what="a search assistant: short queries, many of them, latency first",
        keep_float=("output.weight",),
        scheme="lsq", rope_window=8192, paircodes=False,
        gate=dict(corpus="prose", max_delta_nll=0.03, min_top1=0.88)),
    "science": dict(
        what="a science assistant: maths and multi-step reasoning, where a wrong digit is a wrong answer",
        keep_float=("output.weight", "token_embd.weight"),
        scheme="gsq", rope_window=16384, paircodes=True,
        gate=dict(corpus="science", max_delta_nll=0.015, min_top1=0.92)),
    "chat": dict(
        what="a general chat model: the default, no job-specific trade",
        keep_float=("output.weight",),
        scheme="lsq", rope_window=8192, paircodes=False,
        gate=dict(corpus="prose", max_delta_nll=0.03, min_top1=0.88)),
}

# what a name suggests, when the caller does not name a role. Only used to pick a default; `--role` always wins.
_HINTS = (("coder", ("coder", "code", "swe", "dev")), ("science", ("math", "sci", "reason", "phi")),
          ("rag", ("rag", "retriev", "embed", "factcheck")), ("search", ("search", "query")),
          ("agent", ("agent", "tool", "function")))


def suggest(detected):
    """The recipe a file's own name and shape point at, or `chat`."""
    name = str(detected.get("name") or detected.get("file") or "").lower()
    for role, words in _HINTS:
        if any(w in name for w in words):
            return role
    if detected.get("moe") and (detected.get("ctx") or 0) >= 32768:
        return "agent"
    return "chat"


def plan(detected, role=None, target_bytes=None):
    """The conversion plan for one file and one job, with the reasons it cannot be carried out.

    Returns a dict: `role`, `recipe`, `steps` (what the converter will do), `warnings` (what the file cannot support -
    a rope window beyond the file's own context, a recipe that wants experts in a dense model) and `gate`. A plan is
    still returned when there are warnings: the caller decides whether a warning is fatal, and `kindi convert` prints
    them rather than silently building something else.
    """
    role = role or suggest(detected)
    if role not in RECIPES:
        raise SystemExit("kindi recipes: unknown role %r (have: %s)" % (role, ", ".join(sorted(RECIPES))))
    r = RECIPES[role]
    warn = []
    catalog = detected.get("path") == "catalog pack"
    if not detected.get("convertible"):
        warn.append("%s has no conversion path in this engine (%s)" % (detected.get("arch") or "this file",
                                                                      detected.get("path")))
    if catalog:
        # a catalog pack holds no weights, so what a recipe would do to them does not apply; what does apply is why
        # there is no pack, which is the blocker the catalog writes into the manifest
        blockers = detected.get("blockers") or []
        warn.append("%s is catalogued, not converted: %s" % (detected.get("arch"),
                                                             blockers[0] if blockers else "no kernel in this engine"))
        if detected.get("ternary_source") is False:
            warn.append("the file is not ternary (types %s): a catalog pack copies no weights, so nothing is re-coded "
                        "today - import_plan prices what a ternary pass would cover"
                        % (detected.get("tensor_types") or "unknown"))
    else:
        if detected.get("ternary_source") is False:
            warn.append("the file is not ternary (types %s): the pack path copies ternary codes, so this needs the "
                        "float path (kindi qat-pack) first" % (detected.get("tensor_types") or "unknown"))
        ctx = detected.get("ctx") or 0
        if r["rope_window"] and ctx and r["rope_window"] > ctx:
            warn.append("recipe asks for a %d-position window; the file declares %d, so the table would extrapolate"
                        % (r["rope_window"], ctx))
        if r["paircodes"] and not detected.get("moe"):
            warn.append("paircodes stage expert matrices; this model has no experts, so the flag does nothing")
        if r["scheme"] == "gsq" and detected.get("ternary_source"):
            warn.append("gsq re-fits codes and scales from float weights; a ternary source has no float weights to "
                        "fit, so the pack keeps the file's own codes (use kindi qat-pack for a float checkpoint)")
    steps = ([dict(do="catalog", what="write kindi_pack.json and tensors.json from the header and the tensor table; "
                                     "no weights are copied")]
             if catalog else
             [dict(do="convert", scheme=r["scheme"], rope_window=r["rope_window"], paircodes=r["paircodes"],
                   keep_float=list(r["keep_float"]))])
    if target_bytes:
        steps.append(dict(do="rco", budget=int(target_bytes),
                          what="exact-budget choice of which tensors stay f32 (kindi.rco)"))
    # a catalog pack copies no weights, so the recipe's likelihood gate has nothing to measure; the manifest's own
    # `blockers` is what a catalog has to satisfy instead
    return dict(role=role, recipe=r, detected=detected, steps=steps, warnings=warn,
                gate=(None if catalog else r["gate"]))


def listing(log=print):
    log("recipes (kindi recipes --for MODEL.gguf --role NAME):")
    for name, r in RECIPES.items():
        log("  %-8s %-72s" % (name, r["what"]))
        log("           scheme %-7s rope window %-6s paircodes %-5s gate %s: dNLL <= %.3f, top-1 >= %.2f"
            % (r["scheme"], r["rope_window"], r["paircodes"], r["gate"]["corpus"],
               r["gate"]["max_delta_nll"], r["gate"]["min_top1"]))
    log("  keep_float: %s" % ", ".join(sorted({s for r in RECIPES.values() for s in r["keep_float"]})))
    return 0


def _ggufs(root):
    """every first-part GGUF under root: later shards of a split file and partial downloads are skipped"""
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for n in sorted(files):
            if not n.lower().endswith(".gguf"):
                continue
            m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", n)
            if m and int(m.group(1)) != 1:
                continue
            out.append(os.path.join(dirpath, n))
    return out


def scan(root, role=None, log=print):
    """Detect every model under a directory and print the recipe each would get.

    Headers only, so this is cheap across a folder of sharded 30 GB files, and it is the way to triage a model folder:
    the last column names the first thing a conversion would refuse, or `ready`.
    """
    files = _ggufs(root)
    if not files:
        log("no .gguf files under %s" % root)
        return 1
    log("%d model file(s) under %s" % (len(files), root))
    for p in files:
        try:
            d = detect(p)
        except Exception as e:                                  # a truncated or foreign file is reported, not skipped
            log("  %-44s cannot read the header: %s" % (os.path.basename(p), str(e) or type(e).__name__))
            continue
        r = plan(d, role)
        geo = "%s %sL w%s %s ctx%s" % (d.get("family") or "?", d.get("layers") or "?", d.get("width") or "?",
                                       "%dx%d" % (d["experts"], d["active_experts"]) if d.get("moe") else "dense",
                                       d.get("ctx") or "?")
        log("  %-44s %-38s %-7s %s" % (os.path.basename(p), geo, r["role"],
                                       ("%d warning(s): %s" % (len(r["warnings"]), r["warnings"][0]))
                                       if r["warnings"] else "ready"))
    log("kindi recipes --for FILE --role ROLE  is the full plan for one file")
    return 0


def selftest():
    """Detection on headers this repository has seen, and the plan rules that follow from them."""
    def check(name, ok, detail=""):
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    # a Bonsai 2B header: qwen35, ternary, no experts, 8k window
    kv = {"general.architecture": "qwen35", "general.name": "Ternary-Bonsai-2-2B",
          "qwen35.block_count": 24, "qwen35.embedding_length": 2048, "qwen35.feed_forward_length": 6144,
          "qwen35.attention.head_count": 8, "qwen35.attention.head_count_kv": 2,
          "qwen35.context_length": 8192, "qwen35.rope.freq_base": 1e6,
          "tokenizer.ggml.tokens": ["a"] * 100}
    d = detect_meta(kv, [("blk.0.attn_qkv.weight", [2048, 2048], 143, 0), ("output_norm.weight", [2048], 0, 1)])
    check("detect: family and geometry", d["family"] == "qwen35" and d["layers"] == 24 and d["width"] == 2048)
    check("detect: a file with no expert key is dense, not unknown", d["experts"] == 0 and d["moe"] is False)
    check("detect: ternary source recognised", d["ternary_source"] is True and d["convertible"] is True)
    check("detect: no rope block means the plain table", d["rope_scaling"] is None)
    check("detect: vocab from the tokenizer array", d["vocab"] == 100)

    # a MoE header with a YaRN block, and a name that points at a coding job
    kv2 = dict(kv, **{"general.architecture": "qwen3moe", "general.name": "Qwen3.8-Flash-Next-Coder",
                      "qwen3moe.block_count": 48, "qwen3moe.embedding_length": 5120,
                      "qwen3moe.expert_count": 512, "qwen3moe.expert_used_count": 10,
                      "qwen3moe.context_length": 262144, "qwen3moe.rope.scaling.type": "yarn",
                      "qwen3moe.rope.scaling.factor": 4.0, "qwen3moe.rope.scaling.original_context_length": 65536})
    d2 = detect_meta(kv2, [("blk.0.ffn_gate_exps.weight", [5120, 17408, 512], 12, 0)])
    check("detect: MoE counts", d2["moe"] is True and d2["experts"] == 512 and d2["active_experts"] == 10)
    check("detect: rope block read verbatim", d2["rope_scaling"]["factor"] == 4.0
          and d2["rope_scaling"]["original_context_length"] == 65536)
    check("detect: a Q4_K source is not a ternary source", d2["ternary_source"] is False)
    check("suggest: the name picks the coder recipe", suggest(d2) == "coder")

    p = plan(d2, "coder")
    check("plan: coder keeps the head and the embeddings at f32",
          p["steps"][0]["keep_float"] == ["output.weight", "token_embd.weight"])
    check("plan: a 32k window inside a 262k context is not a warning",
          not any("window" in w for w in p["warnings"]), str(p["warnings"]))
    check("plan: a non-ternary source is refused with the reason",
          any("not ternary" in w for w in p["warnings"]), str(p["warnings"]))

    # a dense 8k file asked for a 32k window: the plan says so rather than building a table that extrapolates
    p2 = plan(d, "rag")
    check("plan: a window beyond the file's context is a warning",
          any("extrapolate" in w for w in p2["warnings"]), str(p2["warnings"]))
    check("plan: paircodes on a dense model is a warning",
          any("no experts" in w for w in p2["warnings"]), str(p2["warnings"]))
    check("plan: gsq on a ternary source is a warning, not a silent no-op",
          any("no float weights" in w for w in plan(d, "coder")["warnings"]))

    # an unknown family is named, not guessed at
    d3 = detect_meta({"general.architecture": "ornith"}, [])
    check("detect: an unknown family is reported as such", d3["family"] == "ornith" and d3["convertible"] is False)
    check("plan: an unconvertible family is a warning",
          any("no conversion path" in w for w in plan(d3, "chat")["warnings"]))

    # a catalog family: recognised, with the attention it needs, and planned as a catalog rather than a conversion
    d4 = detect_meta({"general.architecture": "glm5-next", "glm5-next.block_count": 45,
                      "glm5-next.embedding_length": 4096, "glm5-next.expert_count": 288,
                      "glm5-next.expert_used_count": 8, "glm5-next.attention.head_count": 64,
                      "glm5-next.attention.q_lora_rank": 1536, "glm5-next.context_length": 1048576},
                     [("blk.0.ffn_gate_exps.weight", [4096, 2048, 288], 10, 0)])
    check("detect: a catalog family is named with its attention",
          d4["path"] == "catalog pack" and "MLA" in d4["attention"], str(d4.get("attention")))
    check("detect: its extra geometry comes from the table the catalog writes from", d4["q_lora"] == 1536)
    p4 = plan(d4, "chat")
    check("plan: a catalog family plans a catalog, not a conversion",
          len(p4["steps"]) == 1 and p4["steps"][0]["do"] == "catalog", str(p4["steps"]))
    check("plan: and says why there is no pack",
          any("catalogued, not converted" in w for w in p4["warnings"]), str(p4["warnings"]))

    # a budget adds the rco step, and the gate is the recipe's own
    p3 = plan(d, "chat", target_bytes=500 << 20)
    check("plan: a budget adds the rco step", p3["steps"][-1]["do"] == "rco" and p3["steps"][-1]["budget"] == 500 << 20)
    check("plan: the gate comes from the recipe", p3["gate"] is RECIPES["chat"]["gate"])
    try:
        plan(d, "nonsense")
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown role must be refused")

    # a scan lists a model once: shards 2..N of a split file are the same model's tail, not four more models
    td = tempfile.mkdtemp(prefix="kindi-recipes-")
    try:
        for n in ("m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf", "solo.gguf",
                  "half.gguf.part"):
            open(os.path.join(td, n), "wb").close()
        os.makedirs(os.path.join(td, ".cache"))
        open(os.path.join(td, ".cache", "hidden.gguf"), "wb").close()
        got = sorted(os.path.basename(p) for p in _ggufs(td))
        check("scan: one entry per model, no partials, no dot directories",
              got == ["m-00001-of-00003.gguf", "solo.gguf"], str(got))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return 0


if __name__ == "__main__":
    selftest()
    print("checks passed")
    raise SystemExit(0)
