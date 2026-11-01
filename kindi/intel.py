"""Kindi Intel backend: CPU (oneAPI / OpenVINO), Arc iGPU and NPU through OpenVINO.

This branch of Kindi adds Intel data-center-free hardware: 6th-10th gen Core ("Skylake" to "Comet Lake",
the user's 9th-gen i9), Core Ultra ("Meteor Lake" / "Arrow Lake" / "Lunar Lake") with the built-in Arc iGPU
and NPU 3, and discrete Arc A-series GPUs with the Xe2 "Battlemage" architecture (the i9 Ultra mini PCs'
built-in graphics, and the Arc B50 / B58-class cards).

Hardware profiles (docs/INTEL.md has the full table and sources):

  CPU      AMX INT8 needs Sapphire Rapids or newer (4th Xeon / Core Ultra server parts); a 9th-gen i9 has
           AVX2 + FMA3 only; Meteor Lake and later client CPUs add AVX-VNNI (VEX-encoded VNNI). llama.cpp
           builds -O3 -march=native with AVX2 run well on all three; that is what kindi.cpu already does.
  iGPU     Arc integrated (Core Ultra) and discrete Arc A-series (DG2, Xe HPG) and B-series (BMG, Xe2):
           OpenVINO GPU plugin (Level Zero) runs fp16 and quantized models; oneDNN/Xe kernel; DP4A-style
           integer dot product on DG2/BMG. llama.cpp SYCL supports these cards directly.
  NPU      NPU 2 (M series) and NPU 3 (Core Ultra 200V / Arrow Lake): 4-13 TOPS, LPDDR-bound. OpenVINO
           NPU plugin compiles an IR/graph once per shape and runs it at very low power; KV cache stays
           in host memory (NPU has no global scratch beyond its compile-time graph). llama.cpp and
           openvino-ai-sdk "ov_npu" sample show the pattern: static shapes, prompt cache, weight streaming.

Backend selection (KINDI_DEVICE, default auto):
  KINDI_DEVICE=intel        use the best Intel accelerator available (GPU -> NPU -> OPENVINO CPU)
  KINDI_DEVICE=openvino     force the OpenVINO Python backend (any device via KINDI_OV_DEVICE)
  KINDI_INTEL_GPU=0         do not use the Intel GPU for the main model (use it for draft/spec tokens)
  KINDI_INTEL_NPU=0         do not use the NPU for the main model
  KINDI_INTEL_NPU_MIN_TPS   the NPU must beat this decode rate or the CPU takes over (default 8)
  KINDI_OV_DEVICE           override the OpenVINO device string ("CPU", "NPU", "GPU", "GPU.0", "MULTI:...")
  KINDI_OV_HINT             performance hint: latency | throughput | cumulative (default latency)
  KINDI_OV_WEIGHT_STREAMING  1 (default): ov::hint::enable_weight_streaming for models bigger than RAM
  KINDI_DISK_P2P            1 (default): zero-copy disk->device weight streaming (see diskp2p.py)

The OpenVINO path is exact-token against the same GGUF through llama.cpp where the arithmetic agrees;
where it does not (quantization kernels differ), verify reports the prefix-match length like the Gemma 4 path.
"""
import ctypes
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np


# ---------------------------------------------------------------------------
# hardware discovery
# ---------------------------------------------------------------------------

def _lscpu():
    info = {}
    try:
        out = subprocess.run(["lscpu"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True, timeout=10).stdout
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip().lower()] = v.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _cpu_model():
    vendor, model, fam, step = "", "", None, None
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        model = m.group(1).strip() if m else ""
        m = re.search(r"vendor_id\s*:\s*(\S+)", txt)
        vendor = m.group(1) if m else ""
        m = re.search(r"cpu family\s*:\s*(\d+)", txt)
        fam = int(m.group(1)) if m else None
        m = re.search(r"model\s*:\s*(\d+)", txt)
        step = int(m.group(1)) if m else None
    except OSError:
        pass
    return vendor, model, fam, step


# family 6 model numbers (Intel SDM Vol 2 / linux cpu_table.c) used to pin the generation
_SKU_MODELS = {
    (142, 158, 165, 166): "Coffee Lake / Comet Lake (8th-9th/10th gen)",
    (154,): "Alder Lake / Raptor Lake (12th-14th gen)",
    (170, 171, 183): "Sapphire Rapids / Emerald Rapids / Granite Rapids (Xeon)",
    (170,): "Sapphire Rapids (AMX INT8/BF16)",
    (170, 178, 179, 180, 181, 182, 183, 190, 191): "server",
    (170, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 190, 191): "x",
}

def _gen_of(fam, model, flags):
    """coarse microarchitecture classification for the inference knobs it unlocks"""
    if fam != 6:
        return "generic x86-64"
    if model in (143, 154, 155, 170, 171, 175, 176, 178, 179, 180, 181, 182, 183, 184, 190, 191):
        if "amx_tile" in flags and "amx_int8" in flags:
            return "server (Sapphire/Emerald/Granite Rapids): AMX INT8 + BF16"
        if model in (154, 155):
            return "client hybrid (Alder/Raptor Lake): AVX2 + VNNI"
        return "server (Xeon Scalable)"
    if model in (170,):
        return "Sapphire Rapids: AMX INT8/BF16"
    if model in (175, 190):                     # Meteor Lake / Arrow Lake client
        return "Core Ultra (Meteor/Arrow Lake): AVX-VNNI + Arc iGPU + NPU 3"
    if model in (181, 182):                     # Lunar Lake
        return "Core Ultra 200V (Lunar Lake): AVX-VNNI + Arc 140V iGPU + NPU 4"
    if model in (142, 158, 165, 166):
        return "Core 6th-10th gen (Skylake..Comet Lake): AVX2 + FMA3, no VNNI"
    if model in (125, 126, 140, 141, 142, 158, 165, 166):
        return "Core 8th-10th gen: AVX2 + FMA3"
    return "x86-64 CPU"


def _pci(name_contains):
    """devices from lspci whose line contains one of the names"""
    hits = []
    try:
        out = subprocess.run(["lspci", "-nn"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True, timeout=10).stdout
        for line in out.splitlines():
            low = line.lower()
            if any(s in low for s in name_contains):
                hits.append(line.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return hits


_ARC_D = ["arc", "dg2", "bmg", "xe hp", "a380", "a580", "a750", "b580", "b570", "b50", "b60"]
_ARC_I = ["ultra", "iris xe", "hd graphics p", "uhd graphics"]


def intel_gpu_names():
    """all Intel display-class PCI devices seen by lspci"""
    return _pci(_ARC_D + _ARC_I + ["vga compatible controller: intel", "display controller: intel"])


def npu_present():
    hits = _pci(["npu", "accelerator"])
    if any("intel" in h.lower() and "npu" in h.lower() for h in hits):
        return True, hits
    # Meteor/Lunar/Arrow Lake NPU shows as "System peripheral" JVDG-... ; check the modalias too
    try:
        for d in os.listdir("/sys/bus/pci/devices"):
            try:
                with open(os.path.join("/sys/bus/pci/devices", d, "vendor")) as f:
                    if f.read().strip() != "0x8086":
                        continue
                with open(os.path.join("/sys/bus/pci/devices", d, "device")) as f:
                    dev = f.read().strip()
                if dev in ("0x112c", "0x1163", "0xae40", "0xbfc7"):   # NPU 2 / NPU 3 device ids
                    return True, [d]
            except OSError:
                continue
    except OSError:
        pass
    return False, []


def level_zero_devices():
    """[(index, name, type)] discovered through the Level Zero loader (what OpenVINO's GPU plugin uses)"""
    out = []
    try:
        ze = ctypes.CDLL(os.environ.get("KINDI_ZELIB", "libze_loader.so.1"))
    except OSError:
        return out
    n = ctypes.c_uint32()
    if ze.zeInit(0) != 0:
        return out
    if ze.zeDriverGet(ctypes.byref(n), None) != 0 or n.value == 0:
        return out
    drv = (ctypes.c_void_p * n.value)()
    ze.zeDriverGet(ctypes.byref(n), drv)
    idx = 0
    for d in drv:
        cnt = ctypes.c_uint32()
        if ze.zeDeviceGet(d, ctypes.byref(cnt), None) != 0 or cnt.value == 0:
            continue
        devs = (ctypes.c_void_p * cnt.value)()
        ze.zeDeviceGet(d, ctypes.byref(cnt), devs)
        for dv in devs:
            props = ctypes.create_string_buffer(512)   # ze_device_properties_t: first field is struct type/size
            if ze.zeDeviceGetProperties(dv, props) == 0:
                name = props.raw[16:16 + 256].split(b"\0")[0].decode("utf-8", "replace")
                out.append((idx, name, "gpu"))
                idx += 1
    return out


def discover(log=print):
    """the Intel hardware profile of this machine, as a dict (kindi intel-info prints it)"""
    prof = {}
    vendor, model, fam, step = _cpu_model()
    fl = cpu_flags()
    prof["cpu"] = {"vendor": vendor, "name": model, "family": fam, "model": step and fam,
                   "gen": _gen_of(fam, step, fl), "cores": os.cpu_count(),
                   "avx2": "avx2" in fl, "avx_vnni": "avx_vnni" in fl, "fma": "fma" in fl,
                   "amx": ("amx_tile" in fl and "amx_int8" in fl)}
    gpus = intel_gpu_names()
    prof["igpu"] = [g for g in gpus if any(s in g.lower() for s in ("vga", "display"))]
    prof["dgpu"] = [g for g in gpus if any(s in g.lower() for s in _ARC_D)]
    npu, npu_ids = npu_present()
    prof["npu"] = {"present": npu, "ids": npu_ids}
    prof["level_zero"] = level_zero_devices()
    prof["openvino"] = _openvino_versions()
    prof["oneapi"] = bool(shutil.which("icx") or shutil.which("icpx") or
                          os.path.isdir("/opt/intel/oneapi"))
    if log:
        log("intel cpu : %s [%s]" % (prof["cpu"]["name"] or "?", prof["cpu"]["gen"]))
        for k in ("igpu", "dgpu"):
            for g in prof[k]:
                log("intel %s : %s" % (k, g))
        log("intel npu : %s" % ("found (%s)" % ", ".join(map(str, prof['npu']['ids'][:2])) if npu else "not present"))
        if prof["level_zero"]:
            log("level zero: %s" % ", ".join(n for _, n, _t in prof["level_zero"]))
    return prof


def _openvino_versions():
    try:
        import openvino as ov
        d = {"version": ov.__version__}
        try:
            core = ov.Core()
            d["devices"] = list(core.available_devices)
        except Exception as e:
            d["error"] = str(e)[:120]
        return d
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# recommendation
# ---------------------------------------------------------------------------

def pick_backend(prof=None, log=print):
    """(backend, device string, reason) for KINDI_DEVICE=intel/auto on this machine."""
    forced = os.environ.get("KINDI_OV_DEVICE")
    if forced:
        return "openvino", forced, "KINDI_OV_DEVICE=%s" % forced
    want = os.environ.get("KINDI_DEVICE", "auto").lower()
    if want in ("cpu", "gpu", "cuda", "hip"):
        return want, None, "explicit KINDI_DEVICE=%s" % want
    prof = prof or discover(log=log)
    ov = prof.get("openvino") or {}
    devs = ov.get("devices") or []
    gpu_ok = os.environ.get("KINDI_INTEL_GPU", "1") != "0"
    npu_ok = os.environ.get("KINDI_INTEL_NPU", "1") != "0"
    if "GPU" in devs and gpu_ok:
        return "openvino", "GPU", "OpenVINO GPU plugin (Level Zero) on the Arc iGPU/dGPU"
    if "NPU" in devs and npu_ok:
        return "openvino", "NPU", "OpenVINO NPU plugin (low-power decode)"
    if "CPU" in devs:
        return "openvino", "CPU", "OpenVINO CPU plugin (oneDNN: AMX/VNNI/AVX2 by generation)"
    return "cpu", None, "no OpenVINO runtime: the native Kindi CPU engine (gcc -march=native)"


# ---------------------------------------------------------------------------
# OpenVINO engine: ternary packs exported to OV IR, and GGUF via ov_genai
# ---------------------------------------------------------------------------

class Engine(object):
    """Kindi engine interface (reset/feed/step/generate, ylog/ylog_n/ylog_map) on OpenVINO.

    Two model kinds are supported:
      * a Kindi pack (Ternary-Bonsai-2): exported to OpenVINO IR once (kindi export-openvino) and run here;
      * a plain GGUF (any llama.cpp-supported model): run through openvino-genai (ov_genai), which uses the
        same weights llama.cpp uses - the fast path for iGPU/NPU on Nemotron/Gemma/Qwen files.
    """

    def __init__(self, pack, ctx=4096, clusters=None, log=print, device=None, ir_dir=None):
        t0 = time.time()
        self.log = log
        self.ctx = int(ctx)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        self.man = man
        self.dims = dict(man.get("dimensions", {}))
        self.backend = "openvino"
        self.device = device or os.environ.get("KINDI_OV_DEVICE") or "CPU"
        self.vocab = json.load(open(os.path.join(pack, "vocab.json")))
        self.ylog = np.zeros(1, dtype=np.float32)
        self.ylog_n = 0
        self.ylog_map = None
        self._pos = 0
        self._pending = None
        ir_dir = ir_dir or os.path.join(pack, "ov_ir")
        if os.path.isfile(os.path.join(ir_dir, "model.xml")):
            self._mode = "ir"
            self._load_ir(ir_dir)
        elif man.get("source_file") and os.path.isfile(os.path.join(pack, "..", man["source_file"])) or \
                os.environ.get("KINDI_GGUF"):
            self._mode = "genai"
            self._load_genai(os.environ.get("KINDI_GGUF") or man.get("gguf") or "")
        else:
            raise SystemExit("kindi(intel): no OpenVINO IR at %s - run: kindi export-openvino --pack %s"
                             % (ir_dir, pack))
        log("kindi-intel ready: %s on %s (%.1fs)" % (self._mode, self.device, time.time() - t0))

    # -- IR path (ternary packs) -----------------------------------------------------
    def _load_ir(self, ir_dir):
        import openvino as ov
        self.ov = ov
        self.core = ov.Core()
        cfg = {"PERFORMANCE_HINT": os.environ.get("KINDI_OV_HINT", "LATENCY")}
        if os.environ.get("KINDI_OV_WEIGHT_STREAMING", "1") == "1":
            cfg["ov::hint::enable_weight_streaming"] = "true"
        if os.environ.get("KINDI_NUM_THREADS"):
            cfg["INFERENCE_NUM_THREADS"] = os.environ["KINDI_NUM_THREADS"]
        try:
            self.core.set_property(self.device, cfg)
        except Exception:
            pass                                    # older OV: hints unknown on this plugin, keep defaults
        self.net = self.core.compile_model(os.path.join(ir_dir, "model.xml"), self.device)
        self.inf = self.net.create_infer_request()
        ins = self.net.inputs
        self.in_tok = ins[0] if len(ins) else None
        self.state = self.net.query_state()         # OV 2-state dynamic LSTM form: GDN recurrent state lives here
        self.inf.reset_state()

    def reset(self):
        if self._mode == "ir":
            self.inf.reset_state()
        self._pos, self._pending = 0, None

    def feed(self, tok, pos):
        self._forward(int(tok), int(pos), head=False)

    def step(self, tok, pos):
        return self._forward(int(tok), int(pos), head=True)

    def _forward(self, tok, pos, head):
        ins = list(self.net.inputs)
        req = {ins[0]: np.array([[tok]], dtype=np.int64), ins[1]: np.array([[pos]], dtype=np.int64)}
        self.inf.start_async(req)
        self.inf.wait()
        outs = list(self.net.outputs)
        logits = self.inf.get_output_tensor(0).data.reshape(-1).astype(np.float32, copy=False)
        self.ylog, self.ylog_n, self.ylog_map = logits, logits.size, None
        self._pos = pos + 1
        return int(np.argmax(logits)) if head else None

    def generate(self, ids, n):
        self.reset()
        for i, t in enumerate(ids[:-1]):
            self.feed(t, i)
        out = [self.step(ids[-1], len(ids) - 1)]
        pos = len(ids)
        while len(out) < n:
            out.append(self.step(out[-1], pos))
            pos += 1
        return out

    # -- ov_genai path (GGUF models on iGPU/NPU) --------------------------------------
    def _load_genai(self, gguf_or_model_dir):
        try:
            import openvino_genai as ov_genai
        except ImportError:
            raise SystemExit("kindi(intel): the GGUF path needs `pip install openvino-genai` "
                             "(or export the pack: kindi export-openvino)")
        self.cfg = ov_genai.GenerationConfig()
        self.cfg.max_new_tokens = int(os.environ.get("KINDI_MAX_TOKENS", "1024"))
        self.pipe = ov_genai.LLMPipeline(gguf_or_model_dir, self.device)
        self._mode = "genai"

    def raw_generate(self, prompt_text, **kw):
        """text in, text out through ov_genai (used by `kindi chat` when a GGUF is loaded)"""
        c = self.cfg
        for k, v in kw.items():
            setattr(c, k, v)
        return self.pipe.generate(prompt_text, c)


# ---------------------------------------------------------------------------
# pack -> OpenVINO IR exporter (the ternary GEMV becomes an INT8 dequant matmul)
# ---------------------------------------------------------------------------

def export_ir(pack, out_dir=None, log=print, ctx=None):
    """Build an OpenVINO IR of the pack's forward step with the Frontend-of-NumPy route:
    every ternary tensor is stored as int8 trit codes (-1/0/+1) plus an fp16 scale per 128 columns; the
    matmul is ov::MatMul over the dequantized fp16 weights, so the CPU plugin can fold them into
    AMX/VNNI int8 paths and the GPU plugin keeps them fp16 on the Arc iGPU. The GDN recurrence is the
    2-state dynamic form (state_size x width matrices carried in OV state), attention gets a KV-cache
    state per layer. Shapes: batch 1, token-by-token; positions dynamic."""
    import openvino as ov
    import openvino.runtime as r
    from .engine import DEFAULT_DIMS
    out_dir = out_dir or os.path.join(pack, "ov_ir")
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    dims = dict(DEFAULT_DIMS, **man.get("dimensions", {}))
    D, NL, NFF = dims["width"], dims["layers"], dims["ffn"]
    ctx = ctx or int(os.environ.get("KINDI_CTX", "4096"))
    bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
    bc = np.memmap(os.path.join(pack, "bcodes.bin"), dtype=np.uint8, mode="r")
    dp = np.memmap(os.path.join(pack, "dpm.bin"), dtype=np.float16, mode="r")
    log("exporting %d ternary tensors to int8 (-1/0/+1) + fp16 scales ..." % len(bi))
    wt_dir = os.path.join(out_dir, "weights")
    os.makedirs(wt_dir, exist_ok=True)
    meta = {"dims": dims, "ctx": ctx, "hadamard_block": int(man.get("hadamard_block", 512)),
            "head": man.get("head", "token_embd.weight"), "tensors": {}, "vecs": {}}
    # f32.bin holds every non-ternary tensor (norm gammas, ssm gates/conv weights, rope tables are
    # rebuilt by the graph); index.json maps names -> byte offsets. Copy each referenced slice into
    # <out>/weights/<name>.f32 so ov_build.py can bake them as OV constants exactly like cpu.py does.
    vi = json.load(open(os.path.join(pack, "index.json")))["tensors"] if \
        os.path.isfile(os.path.join(pack, "index.json")) else {}
    f32p = os.path.join(pack, "f32.bin")
    if os.path.isfile(f32p):
        fm = np.memmap(f32p, dtype=np.float32, mode="r")
        for name, rec in vi.items():
            if not name.endswith(".weight") and not name.endswith(".bias") and "_a" not in name:
                continue
            R, C = int(rec.get("rows", 0)), int(rec.get("ne0", 0))
            if R <= 0 or C <= 0:
                continue
            off = int(rec["f32_off"]) // 4
            v = np.ascontiguousarray(fm[off:off + R * C])
            v.tofile(os.path.join(wt_dir, name + ".f32"))
            meta["vecs"][name] = {"R": R, "C": C}
        del fm
    for w in sorted({dims["width"], dims["ffn"], dims["heads"] * dims["head_dim"],
                     dims["gdn_inner"]}):
        sp = os.path.join(pack, "signs_%d.f32" % w)
        if os.path.isfile(sp):
            shutil.copyfile(sp, os.path.join(out_dir, "signs_%d.f32" % w))
    for name, rec in bi.items():
        R, C = rec["rows"], rec["ne0"]
        ns = C // 4
        o = rec["bc_off"]
        b = np.asarray(bc[o:o + ns * R]).reshape(ns, R).T.astype(np.int8)   # base-3 digit pair -> trits below
        tr = np.stack([b % 3, b // 3 % 3, b // 9 % 3, b // 27 % 3], axis=2)  # [R][C/4][4] trits 0..2
        w8 = (tr.reshape(R, C) - 1).astype(np.int8)                          # -1/0/+1
        q = rec["dpm_off"] // 2
        S = np.asarray(dp[q:q + (C // 128) * R]).reshape(C // 128, R).T.copy()   # [R][C/128] fp16
        w8.tofile(os.path.join(wt_dir, name + ".w8"))
        S.tofile(os.path.join(wt_dir, name + ".ws"))
        meta["tensors"][name] = {"R": int(R), "C": int(C)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("weight arrays written; building the graph ...")
    _build_graph(out_dir, meta, log=log)
    log("OpenVINO IR at %s (run with KINDI_DEVICE=openvino)" % out_dir)
    return out_dir


def _build_graph(out_dir, meta, log=print):
    """The IR itself is produced by the NumPy frontend in a helper script, because the graph (hybrid GDN +
    attention + SwiGLU + tied head) is long; we emit the builder and run it so this module stays import-safe
    without OpenVINO installed."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ov_build.py")
    rc = subprocess.call(["python3", src, out_dir])
    if rc != 0:
        raise SystemExit("kindi(intel): the IR builder failed (rc %d); see above" % rc)


# ---------------------------------------------------------------------------
# OpenVINO side probe for the CUDA engines: Arc cards as a speculative-draft device
# ---------------------------------------------------------------------------

def spec_ready():
    """True when the OpenVINO runtime is importable (the CUDA plan engine can then put a 2B draft model
    on the iGPU/NPU and verify on the NVIDIA card - the cheap heterogeneous pipeline)."""
    try:
        import openvino  # noqa: F401
        return True
    except ImportError:
        return False
