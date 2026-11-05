"""kindi - command line.

  kindi convert MODEL.gguf      build the weight pack from the PTQ1_0 GGUF (once; ~1 minute)
  kindi fast-head               build the cluster file for the fast head (optional; ~2 minutes)
  kindi verify                  check greedy output against llama.cpp's reference (token for token)
  kindi bench                   measure speed and GPU memory
  kindi bench-spec TEXTFILE      measure speculative decode on real text (CPU)
  kindi serve                   run the OpenAI-compatible server
  kindi chat                    chat in the terminal (no server needed)
  kindi head exact|fast         choose the output head of a running server (next request, no restart)
  kindi info                    show the configuration, the pack and the GPU
Global options: --config FILE, --pack DIR, --state-dir DIR, --ctx N."""
import argparse
import json
import os
import sys
import time

from . import __version__
from . import config as C


def log(msg):
    sys.stdout.write("[kindi %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def _setup_gpu_env(cfg):
    """Kernel caches under state_dir, set before CUDA starts: CuPy's (CUDA C -> PTX) and the driver's JIT cache
    (PTX -> sm_30 code, ~1 minute on a K420 - without a writable cache the driver redoes it on every start)."""
    kdir = os.path.join(cfg["state_dir"], "kernels")
    jdir = os.path.join(cfg["state_dir"], "kernels", "driver")
    os.makedirs(jdir, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", kdir)
    os.environ.setdefault("CUDA_CACHE_PATH", jdir)
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(512 * 1024 * 1024))


def cmd_convert(cfg, a):
    if not a.gguf:
        raise SystemExit("kindi convert: give the model GGUF")
    from .convert import convert
    out = a.out or cfg["pack"]
    if os.path.exists(os.path.join(out, "kindi_pack.json")) and not a.force:
        raise SystemExit("kindi: %s already holds a pack (use --force to rebuild)" % out)
    paircodes = bool(getattr(a, "paircodes", False))
    factor, window = getattr(a, "rope_factor", None), getattr(a, "rope_window", None)
    role = getattr(a, "recipe", None)
    if role:
        from . import recipes
        det = recipes.detect(a.gguf)
        p = recipes.plan(det, role)
        log("recipe %s: %s" % (role, p["recipe"]["what"]))
        for w in p["warnings"]:
            log("  warning: %s" % w)
        step = p["steps"][0]
        paircodes = paircodes or bool(step["paircodes"])
        # a recipe that wants a longer window than the file declares builds the table that reaches it; the plan has
        # already said so as a warning, so the stretch is the caller's decision rather than a silent extrapolation
        if step["rope_window"] and det.get("ctx") and step["rope_window"] > det["ctx"] and factor is None:
            factor, window = float(step["rope_window"]) / float(det["ctx"]), int(det["ctx"])
            log("  rope: factor %g from the file's %d positions, to reach %d" % (factor, det["ctx"], step["rope_window"]))
    manifest = convert(a.gguf, out, log=log, paircodes=paircodes, rope_factor=factor, rope_window=window)
    if manifest.get("backend") == "kindi_gemma4":
        log("Gemma 4 model ready for Kindi native CUDA inference")
    else:
        log("next: kindi verify")
    return 0


def cmd_fast_head(cfg, a):
    _setup_gpu_env(cfg)
    manifest_path=os.path.join(cfg["pack"],"kindi_pack.json")
    manifest=json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}
    if manifest.get("backend")=="kindi_gemma4":
        from .gemma4 import build_fast_head
        build_fast_head(manifest["model_path"],cfg["pack"],K=a.clusters,iters=a.iters,log=log)
    else:
        from .cluster import build
        build(cfg["pack"], K=a.clusters, iters=a.iters, log=log)
    log("restart `kindi serve`, then `kindi head fast` to use it")
    return 0


def cmd_verify(cfg, a):
    _setup_gpu_env(cfg)
    from .verify import run
    return run(cfg, log=log)


def cmd_bench(cfg, a):
    _setup_gpu_env(cfg)
    from .bench import run
    return run(cfg, log=log)


def cmd_recipes(cfg, a):
    from . import recipes
    if a.scan:
        return recipes.scan(a.scan, role=a.role, log=log)
    if not a.src:
        return recipes.listing(log=log)
    det = recipes.detect(a.src)
    p = recipes.plan(det, a.role, target_bytes=a.budget)
    if a.json:
        print(json.dumps(dict(detected=det, plan=dict(role=p["role"], steps=p["steps"], warnings=p["warnings"],
                                                    gate=p["gate"])), indent=1, default=str))
        return 0
    log("detected: %s%s, %s layers, width %s, %s, context %s%s"
        % (det.get("family") or "unknown", " (%s)" % det["what"] if det.get("what") else "", det.get("layers"),
           det.get("width"), "%d experts, %d active" % (det["experts"], det["active_experts"]) if det.get("moe")
           else "no experts", det.get("ctx"), ", ternary source" if det.get("ternary_source") else ""))
    if det.get("tensor_types"):
        log("  tensor types: %s" % ", ".join("%s x%d" % kv for kv in det["tensor_types"].items()))
    log("recipe %s: %s" % (p["role"], p["recipe"]["what"]))
    for s in p["steps"]:
        log("  step: %s" % ", ".join("%s=%s" % (k, v) for k, v in s.items() if k != "do"))
    log("  gate: %s corpus, NLL delta <= %.3f, top-1 >= %.2f"
        % (p["gate"]["corpus"], p["gate"]["max_delta_nll"], p["gate"]["min_top1"]))
    for w in p["warnings"]:
        log("  warning: %s" % w)
    return 0


def cmd_qat_pack(cfg, a):
    from . import ptload, quant
    sd = {k: ptload.widen(v) for k, v in ptload.load(a.pt).items() if v.dtype.kind in "fiu"}
    out = a.out or os.path.join(os.path.expanduser("~"), ".kindi", os.path.splitext(os.path.basename(a.pt))[0])
    quant.selftest()
    man = quant.pack(sd, out, log=log, name=os.path.basename(a.pt), source=a.pt, scheme=a.scheme,
                     budget=getattr(a, "budget", None), within=getattr(a, "within", 0.0))
    log("kindi qat-pack: %d tensors (%d ternary, %s scale) -> %s, %.0f MB codes + %.1f MB f32"
        % (man["tensors"], man["ternary_tensors"], a.scheme, out, man["bcodes_bytes"] / 2 ** 20,
           man["f32_bytes"] / 2 ** 20))
    from . import drift
    return 0 if drift.verify_pack(sd, out, a.scheme, log=log) else 1


def cmd_verify_tier(cfg, a):
    from .tiers import run
    return run(cfg, log=log, tokens=a.tokens, prompts=a.prompts)


def cmd_moe_check(cfg, a):
    from .moe import run
    return run(a, log=log)


def cmd_drift(cfg, a):
    from . import drift
    return drift.run(a, log=log)


def cmd_bench_spec(cfg, a):
    from .benchspec import run
    return run(cfg, log=log, text=a.text, n=a.tokens, prompts=a.prompts, plen=a.prompt_len)


def cmd_spec_check(cfg, a):
    from .speccheck import run
    return run(cfg, log=log, pack=a.pack, tokens=a.tokens)


def cmd_serve(cfg, a):
    _setup_gpu_env(cfg)
    from .server import serve
    serve(cfg, log=log)
    return 0


def cmd_chat(cfg, a):
    _setup_gpu_env(cfg)
    from .server import Kindi
    K = Kindi(cfg, log=log)
    msgs = []
    if a.system:
        msgs.append({"role": "system", "content": a.system})
    print("Kindi %s - type a message, /reset to start over, /exit to quit." % __version__)
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip() in ("/exit", "/quit"):
            return 0
        if line.strip() == "/reset":
            msgs = msgs[:1] if a.system else []
            continue
        if not line.strip():
            continue
        msgs.append({"role": "user", "content": line})
        sys.stdout.write("kindi> ")
        reply = ""
        for kind, val in K.generate({"messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens}):
            if kind == "text":
                reply += val
                sys.stdout.write(val)
                sys.stdout.flush()
            else:
                sys.stdout.write("\n  (%.1f tokens/s)" % val["timing"]["gen_tok_s"])
        msgs.append({"role": "assistant", "content": reply})


def cmd_head(cfg, a):
    os.makedirs(cfg["state_dir"], exist_ok=True)
    if a.mode == "fast" and not os.path.exists(os.path.join(cfg["pack"], "head_clusters.npz")):
        raise SystemExit("kindi: the fast head needs the cluster file - run: kindi fast-head")
    with open(os.path.join(cfg["state_dir"], "head_mode"), "w") as f:
        f.write(a.mode + "\n")
    log("head mode: %s (the next request uses it)" % a.mode)
    return 0


def cmd_models(cfg, a):
    from .catalog import list_models
    list_models(cfg, log=log)
    return 0


def cmd_use(cfg, a):
    from .catalog import use
    return use(cfg, a.model, gguf=a.gguf, log=log)


def cmd_fetch(cfg, a):
    from .catalog import fetch, model_dir
    out = a.out or model_dir(cfg)
    got = fetch(a.spec, out, log=log, revision=a.revision, allow_unverified=a.allow_unverified,
                member=a.member, keep_zip=a.keep_zip)
    first = got[0] if isinstance(got, list) else got
    log("next: kindi recipes --for %s --role %s" % (first, a.role or "coder"))
    return 0


def cmd_info(cfg, a):
    print("Kindi %s" % __version__)
    print("config: %s" % cfg["source"])
    for k in sorted(C.DEFAULTS):
        v = cfg[k]
        print("  %-17s %s" % (k, "(set)" if k == "api_key" and v else v))
    man = os.path.join(cfg["pack"], "kindi_pack.json")
    if os.path.exists(man):
        m = json.load(open(man))
        print("pack: %s - %s, from %s" % (cfg["pack"], m.get("model"), m.get("source_file")))
        if m.get("backend") == "kindi_gemma4":
            print("backend: Kindi Gemma 4 CUDA")
            print("fast head: %s" % ("available" if os.path.exists(os.path.join(cfg["pack"],"head_clusters.npz"))
                                     else "not built (kindi fast-head)"))
        else:
            print("fast head: %s" % ("available" if os.path.exists(os.path.join(cfg["pack"], "head_clusters.npz"))
                                     else "not built (kindi fast-head)"))
    else:
        print("pack: none at %s (kindi convert MODEL.gguf)" % cfg["pack"])
    try:
        import cupy as cp
        d = cp.cuda.Device()
        free, total = cp.cuda.runtime.memGetInfo()
        name = cp.cuda.runtime.getDeviceProperties(d.id)["name"]
        name = name.decode() if isinstance(name, bytes) else name
        print("GPU: %s, compute %s, %.0f of %.0f MiB free" % (name, d.compute_capability, free / 2 ** 20, total / 2 ** 20))
    except Exception as e:  # noqa: BLE001
        print("GPU: not available (%s)" % e)
    return 0


def _malloc_tuning():
    """Checkpoints are dozens of 2 MB host arrays per state copy. glibc raises its mmap threshold as large blocks are
    freed, after which such arrays come from the heap of each thread and are never given back (a 9B server grew by
    ~60 MB per request). A fixed 1 MB threshold keeps every such array in its own mapping, returned when freed."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").mallopt(-3, 1 << 20)       # M_MMAP_THRESHOLD = -3
    except (OSError, AttributeError):
        pass                                               # not glibc: nothing to tune


def main(argv=None):
    _malloc_tuning()
    p = argparse.ArgumentParser(prog="kindi", description="Kindi %s - native CUDA inference for ternary models and Gemma 4" % __version__)
    p.add_argument("--config")
    p.add_argument("--pack")
    p.add_argument("--state-dir", dest="state_dir")
    p.add_argument("--ctx", type=int)
    p.add_argument("--version", action="version", version="Kindi " + __version__)
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("convert", help="build the weight pack from the GGUF")
    s.add_argument("gguf", nargs="?")
    s.add_argument("--out", help="pack directory (default: the configured pack)")
    s.add_argument("--force", action="store_true")
    s.add_argument("--paircodes", action="store_true",
                   help="also write pcodes.bin: expert matrices staged by memcpy when experts are paged")
    s.add_argument("--rope-factor", dest="rope_factor", type=float, metavar="F",
                   help="build the pack's rotary table at factor F (2 stretches an 8k window to 16k; 1 is the plain "
                        "table); default: whatever the GGUF asks for")
    s.add_argument("--rope-window", dest="rope_window", type=int, metavar="N",
                   help="the window the factor scales from (YaRN's original_context_length); default: the file's own, "
                        "else the model's declared context length")
    s.add_argument("--recipe", choices=("coder", "agent", "rag", "search", "science", "chat"),
                   help="apply a recipe's conversion plan (paircodes, and the rope window it asks for) and print what "
                        "the file cannot support; `kindi recipes --for FILE` shows the plan without building")
    s = sub.add_parser("fast-head", help="build the fast-head cluster file")
    s.add_argument("--clusters", type=int, default=1024)
    s.add_argument("--iters", type=int, default=12)
    sub.add_parser("verify", help="check against llama.cpp's reference output")
    sub.add_parser("bench", help="measure speed")
    s = sub.add_parser("verify-tier", help="check that the CPU kernel tiers compute the same thing (CPU)")
    s.add_argument("--tokens", type=int, default=24, help="tokens per prompt (default 24)")
    s.add_argument("--prompts", type=int, default=3, help="prompts (default 3)")
    s = sub.add_parser("recipes", help="what a model is, and the conversion plan for a job (coder, rag, search, ...)")
    s.add_argument("src", nargs="?", help="a GGUF file or a pack directory (default: list the recipes)")
    s.add_argument("--scan", metavar="DIR",
                   help="detect every .gguf under DIR (headers only) and print the recipe each one would get")
    s.add_argument("--for", dest="src2", help="same as the positional argument")
    s.add_argument("--role", choices=("coder", "agent", "rag", "search", "science", "chat"),
                   help="the job to plan for (default: inferred from the file's name and shape)")
    s.add_argument("--budget", type=int, metavar="BYTES", help="add the exact-budget format choice to the plan")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("qat-pack", help="ternarise a float torch checkpoint into a Kindi pack")
    s.add_argument("pt", help="a .pt state dict of float weights")
    s.add_argument("--out", help="pack directory (default: ~/.kindi/<checkpoint name>)")
    s.add_argument("--scheme", choices=("absmean", "lsq", "gsq", "auto"), default="lsq",
                   help="absmean is the PTQ1_0/GGUF convention; lsq searches the minimum-error scale; gsq fits the "
                        "codes and the scale together (kindi.gsq) - 11%% closer than lsq on heavy-tailed weights; "
                        "auto measures all three a tensor and takes the best (kindi.autotune)")
    s.add_argument("--budget", type=int, metavar="BYTES",
                   help="spend a size budget: choose which tensors stay f32 to minimise squared error (kindi.rco)")
    s.add_argument("--within", type=float, default=0.0, metavar="FRAC",
                   help="with --scheme auto: take a cheaper scheme when it is within FRAC of the best error a "
                        "tensor (absmean is one pass, lsq a 33-point grid, gsq an alternation to a fixed point)")
    s = sub.add_parser("moe-check", help="check the CPU Mixture-of-experts FFN against a NumPy reference")
    s.add_argument("--experts", type=int, default=4)
    s.add_argument("--active", type=int, default=2)
    s.add_argument("--layers", type=int, default=3)
    s.add_argument("--paged", action="store_true", help="also measure resident vs paged experts")
    s = sub.add_parser("drift", help="measure what the ternary quantiser costs on a float checkpoint")
    s.add_argument("checkpoint", nargs="+", help="one or more .pt state dicts")
    s.add_argument("--groups", help="scale granularities to compare: 128,row,tensor (default 128)")
    s.add_argument("--schemes", help="absmean,lsq (default both)")
    s.add_argument("--pack", help="also write packs here, and check they decode back exactly")
    s = sub.add_parser("bench-spec", help="measure speculative decode on a text file (CPU)")
    s.add_argument("text", help="file to cut prompts from")
    s.add_argument("--tokens", type=int, default=128, help="new tokens per prompt (default 128)")
    s.add_argument("--prompts", type=int, default=8, help="how many prompts (default 8)")
    s.add_argument("--prompt-len", dest="prompt_len", type=int, default=48,
                   help="prompt length in tokens, i.e. the drafter's history (default 48)")
    s = sub.add_parser("spec-check", help="check the CPU serving loop's two-token pass and its stop path")
    s.add_argument("--tokens", type=int, default=8, help="greedy tokens to run (default 8)")
    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--api-key", dest="api_key")
    s.add_argument("--head", choices=("exact", "fast"))
    s = sub.add_parser("chat", help="chat in the terminal")
    s.add_argument("--system", default="")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    s = sub.add_parser("head", help="exact or fast output head for a running server")
    s.add_argument("mode", choices=("exact", "fast"))
    sub.add_parser("info", help="show configuration, pack and GPU")
    sub.add_parser("models", help="list the models and what fits this machine")
    s = sub.add_parser("use", help="download a model, build its pack and select it")
    s.add_argument("model")
    s.add_argument("--gguf", help="use this GGUF file instead of downloading it")
    s = sub.add_parser("fetch", help="fetch a model file from Hugging Face (sha256-verified) or Kaggle (dataset zip)")
    s.add_argument("spec", help="hf:OWNER/REPO/PATH, e.g. hf:prism-ml/Ternary-Bonsai-2-2B-gguf/"
                                "Ternary-Bonsai-2-2B-PTQ1_0.gguf - or kaggle:OWNER/DATASET[/MEMBER]")
    s.add_argument("--out", help="directory to write into (default: the configured models directory)")
    s.add_argument("--revision", default="main")
    s.add_argument("--allow-unverified", dest="allow_unverified", action="store_true",
                   help="fetch a file whose repo entry carries no sha256 (only its length is checked)")
    s.add_argument("--member", help="for kaggle:, the file inside the dataset to extract (default: all of them)")
    s.add_argument("--keep-zip", dest="keep_zip", action="store_true",
                   help="for kaggle:, keep the downloaded dataset archive after extracting")
    s.add_argument("--role", help="the recipe to suggest for it afterwards")
    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1
    ov = {k: getattr(a, k, None) for k in ("pack", "state_dir", "ctx", "host", "port", "api_key", "head")}
    cfg = C.load(a.config, ov)
    if getattr(a, "src2", None) and not getattr(a, "src", None):
        a.src = a.src2
    return {"convert": cmd_convert, "recipes": cmd_recipes, "fast-head": cmd_fast_head,
            "verify": cmd_verify, "bench": cmd_bench,
            "bench-spec": cmd_bench_spec, "spec-check": cmd_spec_check, "qat-pack": cmd_qat_pack, "drift": cmd_drift,
            "verify-tier": cmd_verify_tier, "moe-check": cmd_moe_check, "serve": cmd_serve,
            "chat": cmd_chat, "head": cmd_head,
            "info": cmd_info, "models": cmd_models, "use": cmd_use, "fetch": cmd_fetch}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
