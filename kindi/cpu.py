"""Kindi CPU engine (KINDI_DEVICE=cpu): the 2B / 4B / 9B models on the CPU, token-exact with llama.cpp.

The whole forward pass of a token runs in C (kcpu.c, one OpenMP parallel region per token). Same interface as
kindi.engine.Engine: reset(), feed(tok, pos), step(tok, pos) -> greedy token, generate(ids, n), logits in
self.ylog[:self.ylog_n]. A step can also take a float row instead of a token id (feed_emb / step_emb: D floats in
row order, before the activation fold), which is the same pass after its first instruction; embed_row(tok) is the
row the lookup would have supplied, so injecting it back is bit-identical to stepping that token. `feed_emb` / `step_emb` take a float row where the token id would go, so a caller can inject
an input vector instead of looking one up; `embed_row(tok)` returns the row that lookup would have produced. Loads a Kindi pack (kindi convert) and repacks the ternary weights into pair codes
("perm8": per 8-row block and 128-column group, 64 words of 32 bits; one vpermps gives 8 rows x 2 columns).
kcpu.c is built on first use with gcc (-O3 -march=native -fopenmp) into KINDI_CPU_CACHE (default ~/.cache/kindi).
Threads: OMP_NUM_THREADS (with OMP_PROC_BIND=close; on a two-socket host, the cores of one socket)."""
import ctypes
import json
import os
import time

import numpy as np

from . import cbuild, packio, rope

VOCAB = 248320
DEFAULT_DIMS = {"layers": 24, "width": 2048, "ffn": 6144, "heads": 8, "kv_heads": 2,
                "groups": 16, "head_dim": 256, "rope_dim": 64, "state_size": 128, "gdn_inner": 2048,
                "rope_base": 1e7, "full_attention_interval": 4}
HERE = os.path.dirname(os.path.abspath(__file__))
F32 = np.float32
PF = ctypes.POINTER(ctypes.c_float)
PU8 = ctypes.POINTER(ctypes.c_uint8)
PU16 = ctypes.POINTER(ctypes.c_uint16)


class _Index(dict):
    """A pack index whose missing key names the tensor and the file, not a bare KeyError.

    `_layout_problem` reports a whole mismatch before the build starts; this is the backstop for a name the build
    reads and that list does not know about (a tensor added to the loop and not to `_pack_names`).
    """

    def __init__(self, data, pack, where):
        dict.__init__(self, data)
        self.pack, self.where = pack, where

    def __getitem__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            raise SystemExit("kindi-cpu: %s has no %s, which the layout reads by name.\n"
                             "  The pack does not match this model; `kindi info --pack %s` shows what it is."
                             % (os.path.join(self.pack, self.where), k, self.pack))


def _pack_names(d, head):
    """The tensor names the CPU layout reads, split by the file they live in: ternary codes, and f32 vectors.

    The other half of the layout is the wiring in `Engine.__init__`. A pack from another model, another variant of
    the same family, or an older `kindi convert` differs in these names, and looked up one at a time they end in a
    KeyError from the middle of the build.
    """
    tern, f32 = ["token_embd.weight", head], ["output_norm.weight"]
    nexp = int(d.get("experts") or 0)
    interval = max(1, int(d["full_attention_interval"]))
    for il in range(int(d["layers"])):
        p = "blk.%d." % il
        f32 += [p + "attn_norm.weight", p + "post_attention_norm.weight"]
        if nexp:                                     # a Mixture-of-experts FFN: a router and nexp expert triples
            tern.append(p + "moe_router.weight")
            for e in range(nexp):
                tern += [p + "moe_gate.e%d.weight" % e, p + "moe_up.e%d.weight" % e, p + "moe_down.e%d.weight" % e]
        else:
            tern += [p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight"]
        if (il + 1) % interval == 0:                 # a full-attention layer, else a gated-DeltaNet one
            tern += [p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "attn_output.weight"]
            f32 += [p + "attn_q_norm.weight", p + "attn_k_norm.weight"]
        else:
            tern += [p + "attn_qkv.weight", p + "attn_gate.weight", p + "ssm_out.weight"]
            f32 += [p + "ssm_beta.weight", p + "ssm_alpha.weight", p + "ssm_dt.bias", p + "ssm_a",
                    p + "ssm_conv1d.weight", p + "ssm_norm.weight"]
    return tern, f32


def _few(names):
    """the first few names, then how many are left, so one line stays one line"""
    names = list(dict.fromkeys(names))
    return ", ".join(names[:4]) + (" and %d more" % (len(names) - 4) if len(names) > 4 else "")


def is_expert(name):
    """a matrix of one MoE expert - the three per expert that can be paged; the router is resident"""
    return ".moe_gate.e" in name or ".moe_up.e" in name or ".moe_down.e" in name


def _pack_files(pack):
    """The files a CPU pack holds. A missing one means the directory is not a CPU pack at all, which is a different
    answer from a tensor mismatch and has a different fix."""
    need = ["kindi_pack.json", "index.json", "bcodes_index.json", "bcodes.bin", "dpm.bin", "f32.bin", "vocab.json"]
    miss = [n for n in need if not os.path.isfile(os.path.join(pack, n))]
    if miss:
        raise SystemExit("kindi-cpu: %s is not a Kindi CPU pack - it has no %s.\n"
                         "  A pack is written by `kindi convert MODEL.gguf PACK_DIR`. A Gemma 4 or Nemotron pack holds\n"
                         "  model_path and is served by the GPU backend, not by KINDI_DEVICE=cpu."
                         % (pack, _few(miss)))


def _layout_problem(pack, man, d, bi, idx, head, widths):
    """None when the pack holds every tensor this layout reads, else the message to stop with.

    A bare KeyError says which dict was missing a key and nothing about why: not whether the pack is for another
    model, which file it is missing from, or that the rest of the pack is intact. This answers those.
    """
    tern, f32 = _pack_names(d, head)
    miss_t = [n for n in tern if n not in bi]
    miss_f = [n for n in f32 if n not in idx]
    miss_s = [w for w in sorted(set(widths))
              if not os.path.isfile(os.path.join(pack, "signs_%d.f32" % w))]
    known = set(tern)
    extra = [n for n in bi if n not in known]
    # a Mixture-of-experts layer is taken by name in the build, so a manifest and a pack that disagree about whether
    # the model has experts would fail on an empty expert table rather than on a name
    nexp = int(d.get("experts") or 0)
    routers = [n for n in bi if n.endswith(".moe_router.weight")]
    confused = bool(routers) != bool(nexp)
    if not (miss_t or miss_f or miss_s or confused):
        return None
    lines = ["kindi-cpu: the pack at %s does not match the model layout." % pack,
             "  pack: %s - %d layers, width %d, ffn %d, %d heads, attention every %d layers, %s"
             % (man.get("model") or "unnamed", d["layers"], d["width"], d["ffn"], d["heads"],
                d["full_attention_interval"],
                "%d experts" % d["experts"] if d.get("experts") else "dense ffn")]
    if miss_t:
        lines.append("  bcodes_index.json is missing %d of the %d ternary tensors: %s"
                     % (len(miss_t), len(tern), _few(miss_t)))
    if miss_f:
        lines.append("  index.json is missing %d of the %d f32 tensors: %s" % (len(miss_f), len(f32), _few(miss_f)))
    if miss_s:
        lines.append("  the fold signs are missing: %s" % _few(["signs_%d.f32" % w for w in miss_s]))
        lines.append("  (their widths are `width`, `gdn_inner`, `heads*head_dim` and `ffn` from the manifest)")
    if extra:
        lines.append("  the pack holds %s this layout never reads: %s"
                     % ("1 tensor" if len(extra) == 1 else "%d tensors" % len(extra), _few(extra)))
    if confused:
        lines.append("  the manifest says %s, the pack holds %s: its dimensions are not its own"
                     % ("%d experts" % nexp if nexp else "no experts",
                        "1 MoE router tensor" if len(routers) == 1 else "%d MoE router tensors" % len(routers)))
    if miss_t or miss_f or confused:
        lines.append("  Tensor names come from the GGUF the pack was converted from: convert the matching one again")
        lines.append("  with `kindi convert MODEL.gguf DIR` (DIR is the path above); `kindi info --pack DIR` shows a pack.")
    else:
        lines.append("  The signs are written beside the codes by `kindi convert`: the pack is incomplete, or its")
        lines.append("  dimensions are not the ones its weights were written for.")
    return "\n".join(lines)


class Mat(ctypes.Structure):
    _fields_ = [("W", PU8), ("S", PF), ("R", ctypes.c_int), ("C", ctypes.c_int)]


BLOCK = 32                  # the column block every kernel walks a row in (kcpu.c: `for b < n / 32`)
GROUP = 128                 # the columns one fp16 scale covers, and the pair-code staging group
PAGE = 8                    # the rows the pair-code layout stages together


def _check_pack_bytes(pack, d, bi, idx):
    """Does every tensor the loader will point at fit in its file, at a geometry the kernels can walk?

    The loader hands the C side raw pointers into `bcodes.bin`, `f32.bin` and `pcodes.bin`. A truncated file does not
    fail at that moment: the slice comes back short, the pointer is taken anyway, and the kernel reads past the end -
    a wrong answer, or a segfault that names neither the file nor the tensor. A geometry the layouts cannot represent
    fails the same way but earlier: a row length that is not a whole number of 32-column blocks is walked as `n / 32`
    blocks and its tail is dropped, a row that is not a whole number of 128-column groups has no scale for its last
    columns, and a row count that is not a multiple of 8 cannot be staged in pair-code blocks at all.

    So this is the check that has to run before the first pointer exists. It reports every problem it finds, names the
    tensor ("the MoE router" for the router, since that is the one a reader looks for), and says which file and which
    bytes are involved rather than a number from inside a reshape.
    """
    D = int(d["width"])
    bad = []
    if D % BLOCK:
        bad.append("the model width is %d, which is not a multiple of %d: %d columns are left over, and every kernel "
                   "walks a row in %d-column blocks, so that tail would be dropped or read past the row's end"
                   % (D, BLOCK, D % BLOCK, BLOCK))
    size = {}
    for n in ("bcodes.bin", "f32.bin", "pcodes.bin"):
        p = os.path.join(pack, n)
        if os.path.isfile(p):
            size[n] = os.path.getsize(p)
    for name, rec in bi.items():
        R, C = int(rec["rows"]), int(rec["ne0"])
        what = "the MoE router" if name.endswith(".moe_router.weight") else "tensor %s" % name
        if C % BLOCK:
            bad.append("%s: its rows are %d wide, which is not a multiple of %d - the kernel's last block would read "
                       "%d columns past the row's end" % (what, C, BLOCK, C % BLOCK))
        elif C % GROUP:
            bad.append("%s: its rows are %d wide, and one scale covers %d columns, so the last %d would have no "
                       "scale (the pack stores one fp16 a group)" % (what, C, GROUP, C % GROUP))
        if R % PAGE:
            bad.append("%s: it has %d rows, and the pair-code layout stages rows in blocks of %d" % (what, R, PAGE))
        off, need = int(rec["bc_off"]), R * (C // 4)
        if off + need > size.get("bcodes.bin", 0):
            bad.append("%s: bytes %d..%d of bcodes.bin, which holds %d - the file is truncated, or its index belongs "
                       "to another pack" % (what, off, off + need, size.get("bcodes.bin", 0)))
        if "pc_off" in rec:
            po = int(rec["pc_off"])
            if po % 4:
                bad.append("%s: pcodes.bin offset %d is not a multiple of 4, and the staged pair codes are read as "
                           "32-bit words" % (what, po))
            elif po + _pair_len(rec) > size.get("pcodes.bin", 0):
                bad.append("%s: bytes %d..%d of pcodes.bin, which holds %d - the staged form is incomplete"
                           % (what, po, po + _pair_len(rec), size.get("pcodes.bin", 0)))
    for name, rec in idx.items():
        if "f32_off" not in rec:            # index.json lists the ternary tensors too; their bytes are bcodes.bin
            continue
        R, C, off = int(rec["rows"]), int(rec["ne0"]), int(rec["f32_off"])
        if off % 4:
            bad.append("tensor %s: f32.bin offset %d is not a multiple of 4, and the file is read as float32"
                       % (name, off))
        elif off + 4 * R * C > size.get("f32.bin", 0):
            bad.append("tensor %s: bytes %d..%d of f32.bin, which holds %d - the file is truncated"
                       % (name, off, off + 4 * R * C, size.get("f32.bin", 0)))
    if bad:
        more = "" if len(bad) <= 8 else "\n  and %d more" % (len(bad) - 8)
        raise SystemExit("kindi-cpu: %s cannot be loaded as it stands.\n  %s%s\n"
                         "  The pack's bytes and its index disagree; convert the model again "
                         "(`kindi convert MODEL.gguf DIR`), and `kindi info --pack DIR` shows the geometry it records."
                         % (pack, "\n  ".join(bad[:8]), more))


class Moe(ctypes.Structure):
    """the router and the expert table of a Mixture-of-experts FFN (nexp == 0 on a dense layer)"""
    _fields_ = [("router", Mat), ("nexp", ctypes.c_int), ("nact", ctypes.c_int), ("staged", ctypes.c_int),
                ("experts", ctypes.POINTER(Mat)), ("sF", PF)]


class Layer(ctypes.Structure):
    _fields_ = [("is_attn", ctypes.c_int), ("attn_norm", PF), ("post_norm", PF),
                ("moe", Moe),
                ("qkv", Mat), ("gate", Mat), ("ssm_out", Mat),
                ("beta_w", PF), ("alpha_w", PF), ("dt_b", PF), ("ssm_a", PF), ("conv_w", PF), ("ssm_norm", PF),
                ("M", PF), ("conv", PF),
                ("q", Mat), ("k", Mat), ("v", Mat), ("o", Mat), ("q_norm", PF), ("k_norm", PF),
                ("Kc", PU16), ("Vc", PU16),
                ("Kt", PU8), ("Vt", PU8), ("Ks", PU16), ("Vs", PU16),
                ("Kq", ctypes.POINTER(ctypes.c_int8)), ("Vq", ctypes.POINTER(ctypes.c_int8)),
                ("Kqs", PU16), ("Vqs", PU16),
                ("K4", PU8), ("V4", PU8), ("K4s", PU16), ("V4s", PU16),
                ("K2", PU8), ("V2", PU8), ("K2s", PU16), ("V2s", PU16),
                ("K2b", PU8), ("V2b", PU8), ("K2bs", PU16), ("V2bs", PU16),
                ("K5", PU8), ("V5", PU8), ("K5s", PU16), ("V5s", PU16),
                ("ffg", Mat), ("ffu", Mat), ("ffd", Mat)]


PI8 = ctypes.POINTER(ctypes.c_int8)


_SCR = ["h", "xn", "xp", "y", "qkv", "zv", "cs", "qk", "gout", "outn", "m", "qn", "kr", "vv", "qr", "kro", "sc",
        "att", "ffg", "ffu", "T", "T2", "logits", "gb", "gg"]


class Model(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int) for n in ("D", "NL", "NFF", "NH", "NHKV", "HD", "NK", "HK", "NV", "CONV_CH", "HB",
                                             "rope_half", "ctx", "vocab")]
                + [("L", ctypes.POINTER(Layer)), ("emb", Mat), ("head", Mat)]
                + [(n, PF) for n in ("out_norm", "sD", "sV", "sA", "sF", "cos_t", "sin_t")]
                + [(n, PF) for n in _SCR]
                + [("Q", PU8), ("QS", PF)]   # int8 activation-quantisation prototype (KINDI_CPU_Q8=1)
                + [("red", ctypes.POINTER(ctypes.c_double))]
                + [("pr", PF), ("ex", PF), ("psel", PF), ("sel", ctypes.c_void_p),
                   ("moe_fetch", ctypes.c_void_p), ("moe_ctx", ctypes.c_void_p)]
                + [("kvq", ctypes.c_int), ("kvq_k", ctypes.c_int), ("kvq_v", ctypes.c_int),
                   ("fold_on", ctypes.c_int), ("qhd", ctypes.c_int), ("qscale", ctypes.c_float)]
                + [("emb_in", PF)])          # non-NULL only for the duration of a kc_step_emb call


def _lib():
    src = os.path.join(HERE, "kcpu.c")
    cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    os.makedirs(cache, exist_ok=True)
    so = os.environ.get("KINDI_CPU_LIB") or cbuild.build(src, cache)      # gcc here, MSVC on Windows
    lib = ctypes.CDLL(so)
    lib.kc_init()
    lib.kc_step.argtypes = [ctypes.POINTER(Model), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.kc_step.restype = ctypes.c_int
    lib.kc_step_emb.argtypes = [ctypes.POINTER(Model), PF, ctypes.c_int, ctypes.c_int]
    lib.kc_step_emb.restype = ctypes.c_int
    lib.kc_embed_row.argtypes = [ctypes.POINTER(Model), ctypes.c_int, PF]
    lib.kc_tier_mask.restype = ctypes.c_int
    lib.kc_get_tier.restype = ctypes.c_int
    lib.kc_kv_quant.argtypes = [ctypes.POINTER(Model), ctypes.c_int]
    lib.kc_kv_quant.restype = ctypes.c_int
    lib.kc_guard_stats.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong)]
    assert lib.kc_sizeof_model() == ctypes.sizeof(Model) and lib.kc_sizeof_layer() == ctypes.sizeof(Layer)
    return lib


_CODE = np.array([0, 1, 2, 3, 8, 4, 5, 6, 7], dtype=np.uint32)     # t0 + 3 t1 -> pair code (4 = both zero -> 8)


def paircodes(codes, R, C, chunk=4096):
    """Pack bytes [C/4][R] (base-3, 4 trits each) -> uint32 [R/8][C/128][64]: word p = the 8 rows' 4-bit codes of
    columns 2p, 2p+1 (row-block major: one thread's rows are one contiguous stream)."""
    G = C // 128
    out = np.empty((R // 8, G, 64), dtype=np.uint32)
    codes = codes.reshape(C // 4, R)
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        b = np.asarray(codes[:, r0:r1]).T.astype(np.uint8)                  # [rows][C/4]
        t = np.stack([b % 3, b // 3 % 3, b // 9 % 3, b // 27 % 3], axis=2).reshape(r1 - r0, G, 64, 2)
        pc = _CODE[t[..., 0].astype(np.uint32) + 3 * t[..., 1].astype(np.uint32)]      # [rows][G][64]
        pc = pc.reshape((r1 - r0) // 8, 8, G, 64)
        word = np.zeros(((r1 - r0) // 8, G, 64), dtype=np.uint32)
        for l in range(8):
            word |= pc[:, l] << np.uint32(4 * l)
        out[r0 // 8:r1 // 8] = word
    return out


def _p(a, t=PF):
    return a.ctypes.data_as(t)


def pair_scales(sc):
    """(G, R) fp16 scales -> the GEMV order as float32, [R/8][G][8]: 8 rows' scales a group, contiguous per block"""
    G, R = sc.shape
    return np.ascontiguousarray(np.asarray(sc).reshape(G, R // 8, 8).transpose(1, 0, 2).astype(F32))


def pair_mat(bc, dp, rec):
    """one ternary matrix from the pack's bytes: pair codes [R/8][C/128][64] and the row-block scales"""
    R, C = rec["rows"], rec["ne0"]
    ns = C // 4
    o = rec["bc_off"]
    W = paircodes(bc[o:o + ns * R], R, C)
    S = pair_scales(np.asarray(packio.tensor_scales(dp, rec)).reshape(C // 128, R))
    return W, S


def pair_bytes(codes, sc, R, C):
    """The staged form of one ternary matrix as bytes, built once at convert time.

    A paged expert is copied into its slot when it is selected, and `pair_mat` takes milliseconds a matrix to build
    that form from the pack's own bytes. Written down, the same bytes stage with two memcpys. `codes` is the pack's
    layout (C//4, R) and `sc` its scales (C//128, R) fp16 - the two arrays `kindi convert` and `synthpack` stream.
    """
    W = paircodes(np.ascontiguousarray(codes), R, C)
    S = pair_scales(np.ascontiguousarray(sc))
    return W.tobytes(), S.tobytes()


def _pair_len(rec):
    """bytes one tensor takes in pcodes.bin: its pair codes, then its float32 scales"""
    R, C = rec["rows"], rec["ne0"]
    return R * (C // 4) + (C // 128) * (R // 8) * 8 * 4


def _pair_views(pc, rec):
    """One tensor's staged bytes out of pcodes.bin, as views: no conversion, two slices.

    The layouts are the ones pair_mat builds - [R/8][C/128][64] words and [R/8][C/128][8] float32 - so a slot is
    filled by copying these two arrays and nothing else.
    """
    R, C = rec["rows"], rec["ne0"]
    o, nw = rec["pc_off"], R * (C // 4)
    W = np.frombuffer(pc, dtype=np.uint32, count=nw // 4, offset=o)
    S = np.frombuffer(pc, dtype=F32, count=(_pair_len(rec) - nw) // 4, offset=o + nw)
    return W.reshape(R // 8, C // 128, 64), S.reshape(R // 8, C // 128, 8)


class Engine(object):
    def __init__(self, pack, ctx=4096, clusters=None, log=print, threads=None, expert_resident=None):
        t0 = time.time()
        if threads:
            os.environ["OMP_NUM_THREADS"] = str(threads)
        self.pack = pack
        self.ctx = int(ctx)
        _pack_files(pack)                 # before _lib(): a wrong pack should not start a kernel build
        self.lib = _lib()
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        d = self.dims = dict(DEFAULT_DIMS, **man.get("dimensions", {}))
        self.HB = int(man.get("hadamard_block", 512))
        self.head_name = man.get("head", "token_embd.weight")
        D, NL, NFF = d["width"], d["layers"], d["ffn"]
        NH, NHKV, HD = d["heads"], d["kv_heads"], d["head_dim"]
        NK, HK = d["groups"], d["state_size"]
        NV = d["gdn_inner"] // HK
        CC = 2 * NK * HK + NV * HK
        interval = d["full_attention_interval"]
        idx = json.load(open(os.path.join(pack, "index.json")))["tensors"]
        bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
        # the bytes and the geometry first: a width no kernel can walk is the root cause of the missing-signs
        # message below, and a short file must stop before the names are checked against a pack that cannot be read
        _check_pack_bytes(pack, d, bi, idx)
        problem = _layout_problem(pack, man, d, bi, idx, self.head_name,
                                  [D, NV * HK, NH * HD, NFF] + ([NFF // 2] if d.get("experts") else []))
        if problem:
            raise SystemExit(problem)
        idx, bi = _Index(idx, pack, "index.json"), _Index(bi, pack, "bcodes_index.json")
        bc = np.memmap(os.path.join(pack, "bcodes.bin"), dtype=np.uint8, mode="r")
        dp = packio.load_scales(pack, man)     # one fp16 a block, whether the pack stores 8-bit indices or fp16
        self.keep = []                    # every array the C side points at

        def keep(a):
            self.keep.append(a)
            return a
        mats = _Index({}, pack, "bcodes_index.json")
        self.moe_arrays = []
        self.pager = None                 # set when experts are paged rather than resident
        self.pc = None                    # pcodes.bin, when the pack carries pair codes (enable_paging maps it)
        for name, rec in bi.items():
            W, S = pair_mat(bc, dp, rec)
            mats[name] = Mat(_p(keep(W), PU8), _p(keep(S)), rec["rows"], rec["ne0"])
        fl = keep(np.fromfile(os.path.join(pack, "f32.bin"), dtype=F32))

        def f(name):
            rec = idx[name]
            off = rec["f32_off"] // 4
            return _p(fl[off:off + rec["rows"] * rec["ne0"]])
        sg = lambda w: keep(np.fromfile(os.path.join(pack, "signs_%d.f32" % w), dtype=F32))  # noqa: E731
        z = lambda n, t=F32: keep(np.zeros(n, dtype=t))  # noqa: E731
        Ls = (Layer * NL)()
        self.M, self.conv, self.Kc, self.Vc = [], [], [], []
        # K/V cache format: 0 fp16, 1 ternary at 2 bits with one scale per (position, head), 2 q8_0, 3 4-bit, 4 a
        # 2-bit four-level code (levels {-3,-1,1,3}), 5 four bits as two 4-level passes. KINDI_CPU_KVQ sets both
        # sides; KINDI_CPU_KVQ_K and KINDI_CPU_KVQ_V override one side each, so K can be q8_0 while V is 4-bit. The
        # arrays are held here so their lifetime is the engine's, exactly like the fp16 ones.
        def kvq_of(name, dflt):
            v = int(os.environ.get(name) or dflt)
            return v if v in (1, 2, 3, 4, 5, 6) else dflt
        kvq = int(os.environ.get("KINDI_CPU_KVQ") or 0)
        kvq = kvq if kvq in (1, 2, 3, 4, 5, 6) else 0
        self.kvq_k = kvq_of("KINDI_CPU_KVQ_K", kvq)
        self.kvq_v = kvq_of("KINDI_CPU_KVQ_V", kvq)
        self.kvq = self.kvq_k                      # the K side, which is what callers and dumps quote
        self.kvq_bufs = []
        for il in range(NL):
            p = "blk.%d." % il
            L = Ls[il]
            L.is_attn = int((il + 1) % interval == 0)
            L.attn_norm, L.post_norm = f(p + "attn_norm.weight"), f(p + "post_attention_norm.weight")
            if p + "moe_router.weight" in mats:                 # a Mixture-of-experts layer
                nexp = int(d.get("experts", 0))
                L.moe.router = mats[p + "moe_router.weight"]
                L.moe.nexp, L.moe.nact, L.moe.staged = nexp, int(d.get("experts_active", 2)), 0
                arr = (Mat * (3 * nexp))()
                self.moe_arrays.append(arr)
                for e in range(nexp):
                    arr[3 * e] = mats[p + "moe_gate.e%d.weight" % e]
                    arr[3 * e + 1] = mats[p + "moe_up.e%d.weight" % e]
                    arr[3 * e + 2] = mats[p + "moe_down.e%d.weight" % e]
                L.moe.experts = ctypes.cast(arr, ctypes.POINTER(Mat))
                L.moe.sF = _p(sg(int(arr[0].R)))          # the fold signs of the expert width
                if expert_resident is not None and nexp > expert_resident:
                    self._add_pager(il, arr, bi, d, expert_resident, keep)
            else:
                L.ffg, L.ffu, L.ffd = mats[p + "ffn_gate.weight"], mats[p + "ffn_up.weight"], mats[p + "ffn_down.weight"]
            if L.is_attn:
                L.q, L.k, L.v, L.o = (mats[p + "attn_q.weight"], mats[p + "attn_k.weight"], mats[p + "attn_v.weight"],
                                      mats[p + "attn_output.weight"])
                L.q_norm, L.k_norm = f(p + "attn_q_norm.weight"), f(p + "attn_k_norm.weight")
                Kc, Vc = z(self.ctx * NHKV * HD, np.uint16), z(self.ctx * NHKV * HD, np.uint16)
                self.Kc.append(Kc)
                self.Vc.append(Vc)
                L.Kc, L.Vc = _p(Kc, PU16), _p(Vc, PU16)
                if 1 in (self.kvq_k, self.kvq_v):                  # ternary: 4 codes a byte, one scale per head
                    Kt, Vt = z(self.ctx * NHKV * (HD // 4), np.uint8), z(self.ctx * NHKV * (HD // 4), np.uint8)
                    Ks, Vs = z(self.ctx * NHKV, np.uint16), z(self.ctx * NHKV, np.uint16)
                    L.Kt, L.Vt = _p(Kt, PU8), _p(Vt, PU8)
                    L.Ks, L.Vs = _p(Ks, PU16), _p(Vs, PU16)
                    self.kvq_bufs += [Kt, Vt, Ks, Vs]
                if 3 in (self.kvq_k, self.kvq_v):                  # 4-bit: two signed nibbles a byte, one scale a head
                    K4, V4 = z(self.ctx * NHKV * (HD // 2), np.uint8), z(self.ctx * NHKV * (HD // 2), np.uint8)
                    K4s, V4s = z(self.ctx * NHKV, np.uint16), z(self.ctx * NHKV, np.uint16)
                    L.K4, L.V4 = _p(K4, PU8), _p(V4, PU8)
                    L.K4s, L.V4s = _p(K4s, PU16), _p(V4s, PU16)
                    self.kvq_bufs += [K4, V4, K4s, V4s]
                if 4 in (self.kvq_k, self.kvq_v):                  # 2-bit: four codes a byte, one scale a head
                    K2, V2 = z(self.ctx * NHKV * (HD // 4), np.uint8), z(self.ctx * NHKV * (HD // 4), np.uint8)
                    K2s, V2s = z(self.ctx * NHKV, np.uint16), z(self.ctx * NHKV, np.uint16)
                    L.K2, L.V2 = _p(K2, PU8), _p(V2, PU8)
                    L.K2s, L.V2s = _p(K2s, PU16), _p(V2s, PU16)
                    self.kvq_bufs += [K2, V2, K2s, V2s]
                if 6 in (self.kvq_k, self.kvq_v):                  # 2-bit four-level, one scale per 32 values
                    K2b, V2b = z(self.ctx * NHKV * (HD // 4), np.uint8), z(self.ctx * NHKV * (HD // 4), np.uint8)
                    K2bs, V2bs = z(self.ctx * NHKV * (HD // 32), np.uint16), z(self.ctx * NHKV * (HD // 32), np.uint16)
                    L.K2b, L.V2b = _p(K2b, PU8), _p(V2b, PU8)
                    L.K2bs, L.V2bs = _p(K2bs, PU16), _p(V2bs, PU16)
                    self.kvq_bufs += [K2b, V2b, K2bs, V2bs]
                if 5 in (self.kvq_k, self.kvq_v):                  # 4 bits as two 4-level passes, one scale a head
                    K5, V5 = z(self.ctx * NHKV * (HD // 2), np.uint8), z(self.ctx * NHKV * (HD // 2), np.uint8)
                    K5s, V5s = z(self.ctx * NHKV, np.uint16), z(self.ctx * NHKV, np.uint16)
                    L.K5, L.V5 = _p(K5, PU8), _p(V5, PU8)
                    L.K5s, L.V5s = _p(K5s, PU16), _p(V5s, PU16)
                    self.kvq_bufs += [K5, V5, K5s, V5s]
                if 2 in (self.kvq_k, self.kvq_v):                  # q8_0: 32 int8 + an fp16 scale per block
                    Kq, Vq = z(self.ctx * NHKV * HD, np.int8), z(self.ctx * NHKV * HD, np.int8)
                    Kqs, Vqs = z(self.ctx * NHKV * (HD // 32), np.uint16), z(self.ctx * NHKV * (HD // 32), np.uint16)
                    L.Kq, L.Vq = _p(Kq, PI8), _p(Vq, PI8)
                    L.Kqs, L.Vqs = _p(Kqs, PU16), _p(Vqs, PU16)
                    self.kvq_bufs += [Kq, Vq, Kqs, Vqs]
            else:
                L.qkv, L.gate, L.ssm_out = mats[p + "attn_qkv.weight"], mats[p + "attn_gate.weight"], mats[p + "ssm_out.weight"]
                L.beta_w, L.alpha_w = f(p + "ssm_beta.weight"), f(p + "ssm_alpha.weight")
                L.dt_b, L.ssm_a, L.conv_w, L.ssm_norm = (f(p + "ssm_dt.bias"), f(p + "ssm_a"), f(p + "ssm_conv1d.weight"),
                                                         f(p + "ssm_norm.weight"))
                Mst, cv = z(NV * HK * HK), z(3 * CC)
                self.M.append(Mst)
                self.conv.append(cv)
                L.M, L.conv = _p(Mst), _p(cv)
        self.Ls = Ls
        m = self.m = Model()
        # the vocabulary the head scores, and whether the weights carry Prism's Hadamard rotation: a pack converted
        # from a GGUF with no `prism.hadamard` metadata (the 1-bit Qwen3 files) is unfolded and must not rotate its
        # activations, which is the same fold() call with the identity in place of the FWHT
        V = int(man.get("vocab") or VOCAB)
        self.fold = bool(man.get("fold", True))
        # qwen35's attention fuses a per-head gate into q (2*HD rows a head, the gate in the second HD); a dense Qwen3
        # stack has a plain q (HD rows a head, no gate). The matrix's own row count says which, so no manifest key is
        # needed and a wrong pack cannot silently pick the other one.
        qrows = next((mm.R for n, mm in mats.items() if n.endswith(".attn_q.weight")), 2 * NH * HD)
        qhd = qrows // NH
        if qhd not in (HD, 2 * HD):
            raise SystemExit("kindi: attn_q has %d rows, which is neither %d nor %d for %d heads"
                             % (qrows, HD * NH, 2 * HD * NH, NH))
        for k, v in (("D", D), ("NL", NL), ("NFF", NFF), ("NH", NH), ("NHKV", NHKV), ("HD", HD), ("NK", NK),
                     ("HK", HK), ("NV", NV), ("CONV_CH", CC), ("HB", self.HB), ("rope_half", d["rope_dim"] // 2),
                     ("ctx", self.ctx), ("vocab", V), ("fold_on", int(self.fold)), ("qhd", qhd),
                     ("qscale", 1.0 / np.sqrt(HD))):
            setattr(m, k, v)
        m.L = ctypes.cast(Ls, ctypes.POINTER(Layer))
        m.emb, m.head = mats["token_embd.weight"], mats[self.head_name]
        m.out_norm = f("output_norm.weight")
        m.sD, m.sV, m.sA, m.sF = (_p(sg(D)), _p(sg(NV * HK)), _p(sg(NH * HD)), _p(sg(NFF)))
        # the rotary table, including whatever interpolation the file asks for (`kindi.rope`): plain for the floats the
        # packs here ship, YaRN for the retired 1-bit Qwen3 files, whose manifest records factor 4 from 8192 positions
        cos, sin = rope.tables(self.ctx, d["rope_dim"], d["rope_base"], man.get("rope_scaling"))
        m.cos_t, m.sin_t = _p(keep(np.ascontiguousarray(cos, dtype=F32))), _p(keep(np.ascontiguousarray(sin, dtype=F32)))
        W = max(D, NFF, NV * HK, NH * HD)
        sizes = {"h": D, "xn": D, "xp": W, "y": D, "qkv": CC, "zv": NV * HK, "cs": CC, "qk": 2 * NK * HK,
                 "gout": NV * HK, "outn": NV * HK, "m": W, "qn": NH * 2 * HD, "kr": NHKV * HD, "vv": NHKV * HD,
                 "qr": NH * HD, "kro": NHKV * HD, "sc": NH * self.ctx, "att": NH * HD, "ffg": NFF, "ffu": NFF,
                 "T": W // 2 * 8, "T2": W // 2 * 8, "logits": max(V, mats[self.head_name].R), "gb": NV, "gg": NV}
        for k in _SCR:
            a = z(sizes[k])
            setattr(m, k, _p(a))
            if k == "logits":
                self.logits = a
        m.red = _p(z(256, np.float64), ctypes.POINTER(ctypes.c_double))
        # int8 activation-quantisation prototype (KINDI_CPU_Q8=1): quantised input and per-group scales; the group
        # width is KINDI_CPU_Q8G (128 by default), so there is one scale per group, not one per 128 columns
        qg = int(os.environ.get("KINDI_CPU_Q8G") or 128)
        if qg not in (2, 4, 8, 16, 32, 64, 128):
            qg = 128
        self.q8buf, self.q8sc = keep(np.zeros(W, dtype=np.int8)), keep(np.zeros(W // qg + 1, dtype=F32))
        m.Q, m.QS = _p(self.q8buf, PU8), _p(self.q8sc)
        # a second scratch set over the same weights and the same Layer state, for the two-token pass (kc_step2)
        self._sizes = sizes
        m2 = Model()
        for k in ("D", "NL", "NFF", "NH", "NHKV", "HD", "NK", "HK", "NV", "CONV_CH", "HB", "rope_half",
                  "ctx", "vocab", "qhd", "qscale"):
            setattr(m2, k, getattr(m, k))
        for k in ("L", "emb", "head", "out_norm", "sD", "sV", "sA", "sF", "cos_t", "sin_t"):
            setattr(m2, k, getattr(m, k))
        for k in _SCR:
            setattr(m2, k, _p(keep(np.zeros(sizes[k], dtype=F32))))
        m2.red = _p(keep(np.zeros(256, np.float64)), ctypes.POINTER(ctypes.c_double))
        m2.Q, m2.QS = m.Q, m.QS     # the two-token pass stays fp32; set only so the pointers are valid
        self.m2 = m2
        # the cache format is a property of both Models: kc_step2 alternates them over the same Layer state
        self.lib.kc_kv_quant2.argtypes = [ctypes.POINTER(Model), ctypes.c_int, ctypes.c_int]
        self.lib.kc_kv_quant2.restype = ctypes.c_int
        want = self.kvq_k * 8 + self.kvq_v        # base 8: the formats run 0..6, and base 4 stopped being injective
        for mm in (m, m2):
            got = int(self.lib.kc_kv_quant2(ctypes.byref(mm), self.kvq_k, self.kvq_v))
            if got != want:
                raise SystemExit("kindi-cpu: K/V cache formats (K %d, V %d) were refused by the kernel (got %d)"
                                 % (self.kvq_k, self.kvq_v, got))
        # kc_step2's snapshot of the recurrent state after its first token (GDN layers only)
        self._snapf = max(1, NL * (NV * HK * HK + 3 * CC))
        self.snap = keep(np.zeros(self._snapf, dtype=F32))
        # expert routing scratch, shared by both Models: one token runs at a time, and kc_step2 runs a MoE layer's
        # experts per token in turn
        nexp = max([Ls[i].moe.nexp for i in range(NL)] or [0])
        nact = max([Ls[i].moe.nact for i in range(NL)] or [0])
        # every Mat is padded to a multiple of 8 rows, so its output buffer has to hold the padded count
        pad = lambda n: max(8, ((n + 7) // 8) * 8)  # noqa: E731
        self.pr, self.ex, self.psel = (keep(np.zeros(pad(nexp), dtype=F32)), keep(np.zeros(pad(D), dtype=F32)),
                                       keep(np.zeros(pad(nact), dtype=F32)))
        self.sel = keep(np.zeros(max(1, nact), dtype=np.int32))
        self._fetch_ref = None
        for mm in (m, m2):
            mm.pr, mm.ex, mm.psel = _p(self.pr), _p(self.ex), _p(self.psel)
            mm.sel = self.sel.ctypes.data
            mm.moe_fetch, mm.moe_ctx = None, None
        self.nexp, self.nact = nexp, nact
        self.ylog, self.ylog_n, self.ylog_map = self.logits, V, None
        self.hsl, self.shortlist = None, 0
        self.shortlist = 0
        self.tier = "auto"
        self._pick_tier(log=log)
        self.vocab = json.load(open(os.path.join(pack, "vocab.json")))
        (log or (lambda *a: None))("kindi-cpu ready: %s, %s threads, ctx %d (%.1fs)%s" % (
            man.get("model"), os.environ.get("OMP_NUM_THREADS", os.cpu_count()), self.ctx, time.time() - t0,
            ", experts paged" if self.pager else ""))

    # --------------------------------------------------------------- kernel tiers (bit-exact selection)
    TIERS = ((1, "scalar"), (2, "avx2"), (4, "q8"))

    def _pick_tier(self, log=print):
        """KINDI_CPU_TIER, or auto: the fastest tier whose logits match the scalar tier exactly"""
        want = (os.environ.get("KINDI_CPU_TIER") or "").lower()
        # KINDI_CPU_Q8=1 is the prototype's own switch, and tier() in C honours it too; naming the tier here keeps the
        # probe and the log line consistent with what the kernel will actually run
        if not want:
            want = "q8" if os.environ.get("KINDI_CPU_Q8") == "1" else "auto"
        bits = dict((n, b) for b, n in self.avail_tiers())
        if want in bits:
            self.set_tier(bits[want])
            self.reset()
            return self.tier
        res = self.probe_tiers(log=log)
        exact = [n for n, _same_tok, _t, same_l in res if same_l]
        for pref in ("avx2", "scalar"):
            if pref in exact:
                self.set_tier(bits[pref])
                break
        self.reset()
        (log or (lambda *a: None))("  kernel tier: %s, bit-exact against scalar (%s)"
                                   % (self.tier, ", ".join(n for n, _t, _tk, _l in res)))
        return self.tier

    def reset(self):
        for a in self.M + self.conv + self.Kc + self.Vc + self.kvq_bufs:
            a.fill(0)

    def avail_tiers(self):
        m = self.lib.kc_tier_mask()
        return [(bit, name) for bit, name in self.TIERS if m & bit]

    def set_tier(self, bit):
        self.lib.kc_set_tier(int(bit))
        self.tier = dict(self.TIERS).get(int(bit), "auto")
        return self.tier

    # --------------------------------------------- per-channel input scales (KINDI_CPU_Q8C=1)
    def q8_cal_begin(self):
        """start recording, per gemv width, the largest magnitude each input column reaches"""
        self.lib.kc_cal_begin()

    def q8_cal_end(self):
        """stop recording; returns the number of gemv widths seen"""
        return int(self.lib.kc_cal_end())

    def q8_cal_save(self, path):
        """write the recorded maxima, so a calibration pass runs once and every later run loads it. The key of each
        entry is its gemv call site and width, because two sites can share a width and not a scale"""
        n = 512
        sites, widths = (ctypes.c_int * n)(), (ctypes.c_int * n)()
        got = min(int(self.lib.kc_cal_keys(sites, widths, n)), n)
        out = {}
        for i in range(got):
            C = int(widths[i])
            a = np.zeros(C, dtype=F32)
            self.lib.kc_cal_get(int(sites[i]), C, _p(a))
            out["s%d_w%d" % (int(sites[i]), C)] = a
        np.savez(path, **out)
        return out

    def q8_cal_load(self, path):
        """load scales saved by q8_cal_save"""
        d = np.load(path)
        for k in d.files:
            site, C = [int(x) for x in k[1:].split("_w")]
            a = np.ascontiguousarray(d[k], dtype=F32)
            if a.shape[0] != C:
                raise SystemExit("kindi: %s holds %d scales for a width of %d" % (k, a.shape[0], C))
            self.lib.kc_cal_put(site, C, _p(a))
        return sorted(d.files)

    def _logits_at(self, tok, pos):
        """the logits of one token at pos, as bytes: the unit every tier is compared in"""
        self.reset()
        t = self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 1)
        return t, self.logits.tobytes()[:4 * self.m.head.R]

    def probe_tiers(self, tok=41, pos=0, log=None):
        """run one token in every tier the build has and say which reproduce the scalar tier bit for bit"""
        out = []
        self.set_tier(1)
        ref, ref_l = self._logits_at(tok, pos)
        for bit, name in self.avail_tiers():
            self.set_tier(bit)
            tkn, lb = self._logits_at(tok, pos)
            out.append((name, tkn == ref and lb == ref_l, tkn, lb == ref_l))
        self.set_tier(1)
        for name, same_tok, _t, same_l in out:
            if log and not same_l:
                log("  tier %s: logits differ from scalar - not bit-exact, so it cannot be the default" % name)
        return out

    def feed(self, tok, pos):
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 0)

    def step(self, tok, pos):
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        return self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 1)

    # --------------------------------------------------------------- embeddings in (a row instead of a token id)
    def embed_row(self, tok):
        """the D floats a token id feeds, before the activation fold (`kc_embed_row`)

        Reading a row leaves the engine untouched, so it can be called between steps. A row taken here and injected
        through `step_emb` is the same input the lookup would have supplied, and the fold is applied to it once either
        way.
        """
        out = np.zeros(self.m.D, dtype=F32)
        self.lib.kc_embed_row(ctypes.byref(self.m), int(tok), _p(out))
        return out

    def _input_row(self, row):
        a = np.ascontiguousarray(row, dtype=F32)
        if a.ndim != 1 or a.shape[0] != self.m.D:
            raise ValueError("kindi-cpu: an injected row holds %d float(s); the model's width is %d"
                             % (a.size, self.m.D))
        return a

    def feed_emb(self, row, pos):
        """one position from a float row (D floats, row order, not folded) instead of a token id; no head"""
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        self.lib.kc_step_emb(ctypes.byref(self.m), _p(self._input_row(row)), int(pos), 0)

    def step_emb(self, row, pos):
        """one token from a float row instead of a token id: the same pass, guards and head as `step`"""
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        return int(self.lib.kc_step_emb(ctypes.byref(self.m), _p(self._input_row(row)), int(pos), 1))

    def guard_stats(self):
        """(decisions examined, decisions that fell back to the fp32 head) under KINDI_CPU_Q8GUARD"""
        seen, hits = ctypes.c_longlong(0), ctypes.c_longlong(0)
        self.lib.kc_guard_stats(ctypes.byref(seen), ctypes.byref(hits))
        return int(seen.value), int(hits.value)

    def step2(self, tok0, tok1, pos):
        """two tokens at pos and pos + 1 in one weight pass (kc_step2): the greedy token after each"""
        if pos + 1 >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos + 1, self.ctx))
        out = (ctypes.c_int * 2)()
        self.lib.kc_step2(ctypes.byref(self.m), ctypes.byref(self.m2), int(tok0), int(tok1), int(pos), 1,
                          ctypes.byref(out), self.snap.ctypes.data_as(PF))
        return int(out[0]), int(out[1])

    def rollback(self):
        """put the recurrent state back to the state after step2's first token (its draft was rejected)"""
        self.lib.kc_state_load(ctypes.byref(self.m), self.snap.ctypes.data_as(PF))

    # --------------------------------------------------------------- Mixture of experts
    FETCH = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)

    def _add_pager(self, il, arr, bi, dims, resident, keep):
        """experts past `resident` are staged: one pool of slots per layer, refilled by the fetch hook.

        A slot holds the three matrices of one expert in pair-code form. The Mats of a paged expert point into a slot,
        so the kernel needs no notion of paging at all: the hook rewrites the slot's bytes and re-points the Mats
        before that expert's products run. `nact` slots is enough because a token uses each selected expert once.
        """
        p = self.pager if self.pager is not None else {"slots": {}, "in_slot": {}, "next": {}, "src": {}, "arr": {}}
        self.pager = p
        nexp = int(dims["experts"])
        E, C = bi["blk.%d.moe_gate.e0.weight" % il]["rows"], dims["width"]
        slots = []
        for _ in range(max(1, resident)):
            slots.append(tuple((keep(np.zeros((E // 8, C // 128, 64), np.uint32)),
                                keep(np.zeros((E // 8, C // 128, 8), F32))) if j < 2 else
                               (keep(np.zeros((C // 8, E // 128, 64), np.uint32)),
                                keep(np.zeros((C // 8, E // 128, 8), F32))) for j in range(3)))
        p["slots"][il] = slots
        p["in_slot"][il] = [-1] * len(slots)
        p["next"][il] = 0
        p["arr"][il] = arr
        for e in range(nexp):            # every expert goes through the cache once it is smaller than the table
            p["src"][(il, e)] = (bi["blk.%d.moe_gate.e%d.weight" % (il, e)], bi["blk.%d.moe_up.e%d.weight" % (il, e)],
                                 bi["blk.%d.moe_down.e%d.weight" % (il, e)])
            for j in range(3):                      # every paged expert starts on the same slot, filled by its fetch
                arr[3 * e + j] = Mat(_p(slots[0][j][0], PU8), _p(slots[0][j][1]), arr[3 * e + j].R, arr[3 * e + j].C)

    def _page_expert(self, big, layer, expert):
        p = self.pager
        ins = p["in_slot"].get(layer)
        if ins is None or expert in ins:
            return
        k = ins.index(-1) if -1 in ins else p["next"][layer]
        ins[k] = expert
        p["next"][layer] = (k + 1) % len(ins)
        arr = p["arr"][layer]
        for j, rec in enumerate(p["src"][(layer, expert)]):
            dw, ds = p["slots"][layer][k][j]
            if self.pc is not None and "pc_off" in rec:
                W, S = _pair_views(self.pc, rec)      # the pair-code pack: two memcpys, no conversion
            else:
                W, S = pair_mat(big["bc"], big["dp"], rec)
            np.copyto(dw, W.reshape(dw.shape))
            np.copyto(ds, S.reshape(ds.shape))
            arr[3 * expert + j] = Mat(_p(dw, PU8), _p(ds), rec["rows"], rec["ne0"])

    def enable_paging(self):
        """wire the fetch hook to the pack's own bytes: a paged expert is staged when it is selected.

        With the pair-code pack variant (`pcodes.bin`) staging is a memcpy of both halves. Without it every miss
        rebuilds the pair codes from the pack's trits, which costs milliseconds a matrix set - that is what made
        paging too slow to keep."""
        if self.pager is None:
            raise SystemExit("kindi: expert_resident was not set, so nothing is paged")
        self._paging_src = {"bc": np.memmap(os.path.join(self.pack, "bcodes.bin"), dtype=np.uint8, mode="r"),
                            "dp": packio.load_scales(self.pack)}
        pc = os.path.join(self.pack, "pcodes.bin")
        if os.path.isfile(pc):
            self.pc = np.memmap(pc, dtype=np.uint8, mode="r")
            self._paging_src["pc"] = self.pc
            size = int(self.pc.shape[0])
            for (il, e), recs in self.pager["src"].items():        # a short file would stage half an expert
                for rec in recs:
                    if "pc_off" in rec and rec["pc_off"] + _pair_len(rec) > size:
                        raise SystemExit(
                            "kindi-cpu: pcodes.bin holds %d bytes, but layer %d expert %d needs bytes %d..%d.\n"
                            "  The pair-code file is incomplete; convert the model again (`kindi convert MODEL.gguf DIR`)."
                            % (size, il, e, rec["pc_off"], rec["pc_off"] + _pair_len(rec)))
        self.set_fetcher(lambda _ctx, layer, expert: self._page_expert(self._paging_src, layer, expert),
                         self._paging_src)
        return self.pager

    def set_fetcher(self, fn, keepalive=None):
        """fn(ctx, layer, expert) runs before that expert's products: the hook a paged expert load uses"""
        self._fetch_ref = self.FETCH(fn) if fn is not None else None
        for mm in (self.m, self.m2):
            mm.moe_fetch = ctypes.cast(self._fetch_ref, ctypes.c_void_p) if self._fetch_ref else None
            mm.moe_ctx = None

    def ffn_probe(self, il, x):
        """one layer's FFN block on an explicit input (the MoE test surface): returns (out, routed experts)"""
        out = np.zeros(self.dims["width"], dtype=F32)
        self.lib.kc_ffn_probe(ctypes.byref(self.m), int(il), _p(np.ascontiguousarray(x, dtype=F32)), _p(out))
        return out, list(int(i) for i in self.sel[:self.nact]), list(float(p) for p in self.psel[:self.nact])

    def generate(self, ids, n):
        self.reset()
        for i, t in enumerate(ids[:-1]):
            self.feed(t, i)
        pos = len(ids)
        out = [self.step(ids[-1], pos - 1)]
        while len(out) < n:
            out.append(self.step(out[-1], pos))
            pos += 1
        return out


class Ngram(object):
    """prompt lookup: the tokens that followed the latest earlier occurrence of the longest matching suffix"""

    def __init__(self, nmax=4, nmin=1):
        self.nmax, self.nmin, self.idx, self.h = nmax, nmin, {}, []

    def push(self, t):
        h = self.h
        h.append(t)
        L = len(h)
        for n in range(self.nmin, self.nmax + 1):
            if L - 1 - n >= 0:
                self.idx[tuple(h[L - 1 - n:L - 1])] = L - 1

    def propose(self, k):
        h = self.h
        for n in range(min(self.nmax, len(h)), self.nmin - 1, -1):
            pat = tuple(h[-n:])
            j = self.idx.get(pat)
            if j is not None and j < len(h) and tuple(h[j - n:j]) == pat:
                return list(h[j:j + k])
        return []


def generate_spec(eng, ids, n, k=1, stats=None):
    """greedy generation with n-gram speculative verify (T = 2, one draft): the same tokens as generate(ids, n)"""
    if os.environ.get("KINDI_CPU_Q8") == "1":
        raise RuntimeError("KINDI_CPU_Q8 is a single-token numerics prototype; the two-token pass is fp32, so the two "
                           "cannot be combined in one generation")
    st = stats if stats is not None else {}
    st.update(passes=0, drafted=0, accepted=0, rejected=0)
    eng.reset()
    for i, t in enumerate(ids[:-1]):
        eng.feed(t, i)
    pos, last, out = len(ids) - 1, ids[-1], []
    dr = Ngram()
    for t in ids:
        dr.push(t)
    while len(out) < n:
        d = dr.propose(1) if n - len(out) > 1 else []
        if not d:
            a = [eng.step(last, pos)]
        else:
            A = eng.step2(last, d[0], pos)
            if A[0] == d[0]:
                a = [A[0], A[1]]
                st["accepted"] += 1
            else:
                eng.rollback()          # back to the state after `last`
                a = [A[0]]
                st["rejected"] += 1
            st["drafted"] += 1
        st["passes"] += 1
        for t in a:
            dr.push(t)
        out += a
        pos += len(a)
        last = a[-1]
    return out[:n]


# --- checks -----------------------------------------------------------------------------------------------------------

def selftest():
    """A good pack loads; a truncated file, a width no kernel can walk and a misaligned offset each stop by name.

    The three failures used to be a short slice, a reshape error and a pointer past the end of a buffer, none of which
    names the file the reader has to fix. This builds one synthetic MoE pack and breaks it one way at a time.
    """
    import json
    import shutil
    import tempfile

    from . import synthpack
    from .cpu import _check_pack_bytes

    def check(name, ok, detail=""):            # silent: tools.selftest runs this in process and prints the summary
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    def refuses(name, want, fn):
        """fn must stop with a SystemExit that contains every string in `want`"""
        try:
            fn()
        except SystemExit as e:
            msg = str(e)
            check(name, all(w in msg for w in want), msg)
        else:
            check(name, False, "it did not refuse")

    td = tempfile.mkdtemp(prefix="kindi-cpu-check-")
    try:
        pack, _, _ = synthpack.build(td, log=lambda *x: None, experts=4, experts_active=2, paircodes=True)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        d = man["dimensions"]
        bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
        idx = json.load(open(os.path.join(pack, "index.json")))["tensors"]

        eng = Engine(pack, ctx=64, log=lambda *x: None)      # the check is wired in, and a whole pack passes it
        check("a complete pack loads, so the checks are not refusing good packs", eng.dims["width"] == d["width"])
        check("the pack has a router to check", any(n.endswith(".moe_router.weight") for n in bi))

        router = next(n for n in bi if n.endswith(".moe_router.weight"))
        broken = json.loads(json.dumps(bi))
        broken[router]["ne0"] = 250                          # the router's row length, one kernel block short
        refuses("a router whose width is not a multiple of 32 is refused and named",
                ["the MoE router", "not a multiple of 32"], lambda: _check_pack_bytes(pack, d, broken, idx))

        broken = json.loads(json.dumps(bi))
        broken[router]["ne0"] = 256 + 64                     # 320: a multiple of 32, not of the 128-column group
        refuses("a row length that loses a scale group is refused",
                ["no scale", "128"], lambda: _check_pack_bytes(pack, d, broken, idx))

        broken = json.loads(json.dumps(bi))
        broken[router]["rows"] = 5                           # 4 experts + 1: cannot be staged in 8-row blocks
        refuses("a row count that is not a multiple of 8 is refused",
                ["blocks of 8"], lambda: _check_pack_bytes(pack, d, broken, idx))

        staged = [n for n in bi if "pc_off" in bi[n]]       # only the paged expert matrices are staged
        check("the pack stages some tensors in pcodes.bin", bool(staged))
        broken = json.loads(json.dumps(bi))
        broken[staged[0]]["pc_off"] += 2                     # the staged form is read as 32-bit words
        refuses("a pcodes offset that is not word-aligned is refused",
                ["multiple of 4"], lambda: _check_pack_bytes(pack, d, broken, idx))

        bad = dict(d, width=250)
        refuses("a model width that is not a multiple of 32 is refused before a pointer exists",
                ["width is 250", "not a multiple of 32"], lambda: _check_pack_bytes(pack, bad, bi, idx))
        man_bad = os.path.join(td, "width250")
        shutil.copytree(pack, man_bad)
        mp = os.path.join(man_bad, "kindi_pack.json")
        m2 = json.load(open(mp))
        m2["dimensions"]["width"] = 250
        json.dump(m2, open(mp, "w"))
        refuses("the engine itself refuses that manifest", ["not a multiple of 32"],
                lambda: Engine(man_bad, ctx=64, log=lambda *x: None))

        def cut(name, n):
            p = os.path.join(pack, name)
            with open(p, "r+b") as f:
                f.truncate(os.path.getsize(p) - n)

        cut("bcodes.bin", 4096)
        refuses("a truncated bcodes.bin is refused with the file and the tensor",
                ["bcodes.bin", "truncated"], lambda: Engine(pack, ctx=64, log=lambda *x: None))
        cut("f32.bin", 4)
        refuses("a truncated f32.bin is refused", ["f32.bin", "truncated"],
                lambda: Engine(pack, ctx=64, log=lambda *x: None))
        cut("pcodes.bin", 4096)
        refuses("a truncated pcodes.bin is refused", ["pcodes.bin"],
                lambda: Engine(pack, ctx=64, log=lambda *x: None))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return 0


if __name__ == "__main__":
    selftest()
    print("checks passed")
    raise SystemExit(0)
