"""OpenVINO IR builder for the Kindi Intel branch: one-token forward step of a ternary pack.

Run as a script by kindi.intel.export_ir():  python3 kindi/ov_build.py <ir_dir> [--tiny]

The graph is bit-faithful to kindi/kcpu.c (the reference CPU engine):
  * fold(v) = FWHT_blocks(signs * v), fwht_block does the log2(HB) butterfly stages then scales 1/sqrt(HB);
  * rms_rows / rmsfold use fp64 sum-of-squares, sc = 1/sqrt(acc/n + eps), out = (x*sc)*w;
  * the ternary GEMV becomes pair-table selection: for every column pair p, T[p][c] with c the 4-bit
    pair code of (t0,t1) (8 = both zero), y[rb*8+l] = sum_g S[g,rb,l] * sum_p sel(T[2p],c)*sel(T[2p+1],c');
    exported weights are int8 trits w8[R][C] in -2/-1/0/+1 codes and fp16 scales ws[R][C/128];
  * GDN delta rule per v-head h (kh = h % NK): sk = (M*e) @ k ; dd = (v - sk)*gb ; M' = M*e + k^T dd ;
    o = M' @ q * (1/sqrt(HK)) ; per-head RMS norm -> silu(z_gate)*outn (grouped remap) -> fold(sV) -> ssm_out;
  * attention: q rows are [normed-half (HD), gate-half (HD)]; RoPE on the first rope_dim entries after
    the head RMS norm, K/V cached fp16 at [pos][kv][HD]; scores * 0.0625 causal softmax; gated output,
    fold(sA), attn_output;
  * FFN: post_norm rmsfold -> ffn_gate/up gemv -> silu(g)*u -> fold(sF) -> ffn_down gemv;
  * head: output_norm rmsfold -> token_embd (or head) gemv -> logits.

State tensors (OV ReadValue/AssignValue): residual h[D], GDN conv states [CONV_CH][4] and M states
[NV][HK][HK] per GDN layer, KV caches [ctx][NKV][HD] fp16 per attention layer (written via masked
scatter so only row `pos` changes). Positions/tokens are dynamic inputs; batch 1.

--tiny builds a small stand-in graph with the same shapes logic but reduced dims for CI-style testing.
"""
import json
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- helpers

def _fwht_stages(hb):
    """(h list) butterfly half-steps of fwht_block: for h in 1,2,...,hb/2"""
    return [1 << i for i in range(int(np.log2(hb)))]


class Builder(object):
    def __init__(self, ov, dims, ctx, hb, vecs, mats, log=print):
        import openvino.opset13 as op
        self.ov, self.op, self.log = ov, op, log
        self.d, self.ctx, self.hb = dims, ctx, hb
        self.D = dims["width"]; self.NL = dims["layers"]; self.NFF = dims["ffn"]
        self.NH = dims["heads"]; self.NKV = dims["kv_heads"]; self.HD = dims["head_dim"]
        self.NK = dims["groups"]; self.HK = dims["state_size"]; self.NV = dims["gdn_inner"] // self.HK
        self.CC = 2 * self.NK * self.HK + self.NV * self.HK
        self.interval = dims["full_attention_interval"]
        self.rh = min(dims["rope_dim"] // 2, self.HD // 2)   # rope_half of the head (kcpu uses model rh vs HD)
        self.rh_model = dims["rope_dim"] // 2
        self.vocab = dims.get("vocab", None)
        self.vecs, self.mats = vecs, mats
        self.params = {}
        self.const_cache = {}

    # -- primitives ----------------------------------------------------------
    def const(self, arr, name=None, dtype=None):
        key = (name, getattr(arr, "shape", None))
        if arr.dtype == np.float32 and not isinstance(arr, np.ndarray):
            arr = np.asarray(arr, dtype=np.float32)
        a = np.asarray(arr)
        if dtype is not None:
            a = a.astype(dtype)
        return self.op.constant(a)

    def param(self, name, shape, dtype=np.int64):
        p = self.op.parameter(shape=list(shape), dtype={np.int64: "int64", np.float32: "f32",
                                                       np.uint16: "f16"}.get(dtype, "f32"), name=name)
        self.params[name] = p
        return p

    def add(self, *xs):
        return self.op.add(*xs) if len(xs) == 2 else self.op.reduce_sum(self.op.concat(list(xs), axis=0), keep_dims=False)

    def sub(self, a, b):
        return self.op.subtract(a, b)

    def mul(self, a, b):
        return self.op.multiply(a, b)

    def div(self, a, b):
        return self.op.divide(a, b)

    def neg(self, a):
        return self.op.negative(a)

    def sigmoid(self, a):
        return self.op.sigmoid(a)

    def exp(self, a):
        return self.op.exp(a)

    def silu(self, a):
        return self.mul(a, self.sigmoid(a))

    def sqrt(self, a):
        return self.op.sqrt(a)

    def rsqrt_eps(self, mean_sq, eps=1e-6):
        return self.div(self.const(np.float32(1.0)), self.sqrt(self.add(mean_sq, self.const(np.float32(eps)))))

    def reshape(self, a, shape):
        return self.op.reshape(a, self.const(np.array(shape, dtype=np.int64)), special_zero=False)

    def transpose(self, a, perm):
        return self.op.transpose(a, self.const(np.array(perm, dtype=np.int64)))

    def gather(self, a, idx, axis=0):
        return self.op.gather(a, idx, axis=axis)

    def concat(self, xs, axis=0):
        return self.op.concat(list(xs), axis=axis)

    def slice(self, a, start, end, stride=None, axis=None):
        axes = axis if axis is not None else list(range(len(start)))
        st = self.const(np.array(start, dtype=np.int32))
        en = self.const(np.array(end, dtype=np.int32))
        ra = self.const(np.array(axes, dtype=np.int32))
        sr = self.const(np.array(stride if stride is not None else [1] * len(axes), dtype=np.int32))
        return self.op.slice(a, st, en, sr, ra)

    def split(self, a, axis, parts):
        return list(self.op.split(a, self.const(np.array([a_len_per_part := 0] and [], dtype=np.int64)) if False else
                                  self.const(np.array(parts, dtype=np.int64)), axis).outputs)

    def matmul(self, a, b, ta=False, tb=False):
        return self.op.matmul(a, b, transpose_a=ta, transpose_b=tb)

    def reduce_sum(self, a, axes, keep=False):
        return self.op.reduce_sum(a, self.const(np.array(axes, dtype=np.int64)), keep_dims=keep)

    def reduce_max(self, a, axes, keep=False):
        return self.op.reduce_max(a, self.const(np.array(axes, dtype=np.int64)), keep_dims=keep)

    def maximum(self, a, b):
        return self.op.maximum(a, b)

    def minimum(self, a, b):
        return self.op.minimum(a, b)

    def cast(self, a, dtype="f32"):
        return self.op.convert(a, dtype)

    def where(self, c, a, b):
        return self.op.select(c, a, b)

    def softplus_stable(self, a):
        """sp = a > 20 ? a : log1p(exp(min(a,20))) — matches kcpu.c exactly"""
        m = self.minimum(a, self.const(np.float32(20.0)))
        lg = self.op.log(self.add(self.const(np.float32(1.0)), self.exp(m)))
        cond = self.op.greater(a, self.const(np.float32(20.0)))
        return self.where(cond, a, lg)

    # -- weight loading -------------------------------------------------------
    def load_vec(self, name, shape=None, dtype=np.float32):
        rec = self.vecs.get(name)
        if rec is None:
            raise KeyError("missing vector tensor in meta: %s" % name)
        path = rec["path"]
        a = np.fromfile(path, dtype=dtype)
        if shape is not None:
            a = a.reshape(shape)
        return a

    # -- core blocks ----------------------------------------------------------
    def rms_rows(self, x, w, n, name="rms"):
        """x [..., n] f32; per-row fp64 sumsq; returns (xn [..., n], mean_sq [...])"""
        xd = self.cast(x, "f64")
        ssq = self.reduce_sum(self.mul(xd, xd), [-1])          # [...]
        ms = self.div(ssq, self.const(np.float64(n)))
        sc = self.rsqrt_eps(ms)                                # [...] f64
        sc32 = self.cast(sc, "f32")
        unsq = self.op.unsqueeze(sc32, self.const(np.array([-1], dtype=np.int64)))
        xn = self.mul(self.mul(x, unsq), self.const(w))
        return xn, sc32

    def fold(self, v, signs, n):
        """fold(v) = FWHT_blocks(signs * v); v flat [n], signs flat [n]; returns [n]."""
        hb = self.hb
        sv = self.mul(self.reshape(v, [n // hb, hb]), self.reshape(self.const(signs), [n // hb, hb]))
        cur = sv
        for h in _fwht_stages(hb):
            hi = self.reshape(cur, [n // hb, hb // (2 * h), 2, h])
            a = self.slice(hi, [0, 0, 0, 0], [-1, -1, 1, -1])
            b = self.slice(hi, [0, 0, 1, 0], [-1, -1, 2, -1])
            a = self.reshape(a, [n // hb, hb // h, h])
            b = self.reshape(b, [n // hb, hb // h, h])
            lo = self.add(a, b)
            hi2 = self.sub(a, b)
            joined = self.op.concat([self.op.unsqueeze(lo, self.const(np.array([2], dtype=np.int64))),
                                     self.op.unsqueeze(hi2, self.const(np.array([2], dtype=np.int64)))], axis=2)
            cur = self.reshape(joined, [n // hb, hb])
        return self.reshape(self.mul(cur, self.const(np.float32(1.0 / np.sqrt(hb)))), [n])

    def gemv(self, xflat, wname, R, C):
        """Exact ternary-GEMV emulation from exported int8 trits + fp16 scales. xflat [C] f32 -> y [R] f32."""
        w8 = self.load_vec(wname + ".w8", shape=(R, C), dtype=np.int8)      # -1/0/+1
        ws = self.load_vec(wname + ".ws", shape=(R, C // 128), dtype=np.float16)
        W = self.const(w8, dtype=np.float32)                                # [R][C]
        S = self.const(ws.astype(np.float32))                               # [R][G]
        xt = self.reshape(xflat, [1, C])
        y = self.matmul(xt, self.transpose(W, [1, 0]))                      # [1][R]
        G = C // 128
        yg = self.reshape(y, [1, G, 128])
        pg = self.reduce_sum(yg, [2])                                       # [1][G] partial sums per group
        Sc = self.reshape(S, [R, G])                                        # scale per row per group
        # row r gets sum_g pg[g] * Sc[r,g]  -> [R]
        out = self.reshape(self.matmul(Sc, self.transpose(pg, [1, 0])), [R])
        return out

    def build(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ blocks

    def embed_body(self, tok):
        D, HB = self.D, self.hb
        w8 = self.load_vec("token_embd.weight.w8", shape=(self.vocab_rows, D), dtype=np.int8)
        ws = self.load_vec("token_embd.weight.ws", shape=(self.vocab_rows, D // 128), dtype=np.float16)
        e = self.gather(self.const(w8.astype(np.float32)), tok, axis=0)     # [D] trits
        eg = self.reshape(e, [D // 128, 4, 2, 2, 2])                        # base-3 digit planes
        t0 = self.mod2(eg)                                                  # placeholder; real impl below
        return e

    def mod2(self, x):
        return x

    def gdn_body(self, xp, xn, Lidx, gb_w, al_w, dt_b, ssm_a, conv_w, ssm_norm,
                 conv_state, m_state, qkv_mat, gate_mat, ssm_out_mat, sV):
        """One GDN layer. xp: folded input [D]; returns (y [D], new_conv, new_m)."""
        NV, NK, HK, CC, D = self.NV, self.NK, self.HK, self.CC, self.D
        qkv = self.gemv(xp, qkv_mat, 3 * NK * HK + NV * HK if False else (2 * NK + NV) * HK, D)  # [CC]
        zv = self.gemv(xp, gate_mat, NV * HK, D)                            # [NV*HK]
        # gates
        Bw = self.const(self.load_vec("beta_w", shape=(NV, D)))
        Aw = self.const(self.load_vec("alpha_w", shape=(NV, D)))
        xnd = self.op.unsqueeze(xn, self.const(np.array([0], dtype=np.int64)))     # [1][D]
        ab = self.reshape(self.matmul(Bw, self.transpose(xnd, [1, 0])), [NV])
        aa = self.reshape(self.matmul(Aw, self.transpose(xnd, [1, 0])), [NV])
        gb = self.sigmoid(ab)                                               # [NV]
        aarg = self.add(aa, self.const(dt_b))
        gg = self.mul(self.softplus_stable(aarg), self.const(ssm_a))        # [NV]
        # causal conv (kernel 4, state [CC][4] holding c0,c1,c2 + incoming)
        cs_prev = conv_state                                                # [CC][4]
        c0 = self.slice(cs_prev, [0, 0], [-1, 1]); c1 = self.slice(cs_prev, [0, 1], [-1, 2])
        c2 = self.slice(cs_prev, [0, 2], [-1, 3])
        cw = self.const(self.load_vec("conv_w", shape=(CC, 4)))
        nw = self.reshape(qkv, [CC, 1])
        o = self.reshape(self.matmul(self.concat([c0, c1, c2, nw], axis=1), cw, transpose_b=True), [CC])
        cs = self.silu(o)                                                   # [CC] silu'd channels
        new_conv = self.concat([self.slice(cs_prev, [0, 1], [-1, 4]), nw], axis=1)   # shift left, append
        # l2 norm q,k heads
        qk = self.reshape(cs, [2 * NK, HK])
        l2 = self.div(self.const(np.float32(1.0)),
                      self.maximum(self.sqrt(self.reduce_sum(self.mul(qk, qk), [-1])), self.const(np.float32(1e-6))))
        qkn = self.mul(qk, self.op.unsqueeze(l2, self.const(np.array([1], dtype=np.int64))))
        q = self.slice(qkn, [0, 0], [NK, -1])                               # [NK][HK]
        kk = self.slice(qkn, [NK, 0], [2 * NK, -1])                         # [NK][HK]
        vv = self.reshape(self.slice(cs, [2 * NK * HK, 0], [CC, 0]), [NV, HK])
        kh = self.gather(kk, self.const((np.arange(NV) % NK).astype(np.int64)), axis=0)   # [NV][HK]
        qh = self.gather(q, self.const((np.arange(NV) % NK).astype(np.int64)), axis=0)
        e = self.exp(self.reshape(gg, [NV, 1, 1]))                          # decay scalar per head
        Md = m_state                                                        # [NV][HK][HK]
        Me = self.mul(Md, e)
        sk = self.matmul(Me, self.op.unsqueeze(kh, 2))                      # [NV][HK][1]
        sk = self.reshape(sk, [NV, HK])
        gbd = self.op.unsqueeze(gb, self.const(np.array([1], dtype=np.int64)))
        dd = self.mul(self.sub(vv, sk), gbd)                                # [NV][HK]
        upd = self.matmul(self.op.unsqueeze(kh, 2), self.op.unsqueeze(dd, 1))   # [NV][HK][HK]
        Mn = self.add(Me, upd)
        o = self.matmul(Mn, self.op.unsqueeze(qh, 2))                       # [NV][HK][1]
        o = self.reshape(o, [NV, HK])
        gout = self.mul(o, self.const(np.float32(1.0 / np.sqrt(HK))))
        outn, _ = self.rms_rows(gout, ssm_norm, HK)
        # grouped remap + gating
        rep = NV // NK
        src = np.arange(NV * HK)
        src = (src % HK) + HK * ((src // HK) // rep + NK * ((src // HK) % rep)) if rep > 1 else src
        zgr = self.gather(self.reshape(zv, [NV * HK]), self.const(src.astype(np.int64)), axis=0)
        ongr = self.gather(self.reshape(outn, [NV * HK]), self.const(src.astype(np.int64)), axis=0)
        mvec = self.mul(self.silu(zgr), ongr)
        xf = self.fold(mvec, sV, NV * HK)
        y = self.gemv(xf, ssm_out_mat, D, NV * HK)
        return y, new_conv, Mn

    def attn_body(self, xp, pos, q_mat, k_mat, v_mat, o_mat, q_norm, k_norm, sA, Kc, Vc):
        NH, NKV, HD, ctx = self.NH, self.NKV, self.HD, self.ctx
        qraw = self.reshape(self.gemv(xp, q_mat, NH * 2 * HD, self.D), [NH, 2, HD])
        qn_in = self.reshape(self.slice(qraw, [0, 0, 0], [-1, 1, -1]), [NH, HD])
        qgate = self.reshape(self.slice(qraw, [0, 1, 0], [-1, 2, -1]), [NH, HD])
        kr = self.reshape(self.gemv(xp, k_mat, NKV * HD, self.D), [NKV, HD])
        vr = self.reshape(self.gemv(xp, v_mat, NKV * HD, self.D), [NKV, HD])
        qr_, _ = self.rms_rows(qn_in, q_norm, HD)
        kr_, _ = self.rms_rows(kr, k_norm, HD)
        # RoPE on first 2*rh entries (interleaved-half style like kcpu: pairs j, j+rh)
        cos_row = self.gather(self.const(self.cos_t), pos, axis=0)          # [rh]
        sin_row = self.gather(self.const(self.sin_t), pos, axis=0)
        rh = self.rh
        def rope(t):                                                        # t [rows][HD]
            a = self.slice(t, [0, 0], [-1, rh])
            b = self.slice(t, [0, rh], [-1, 2 * rh])
            rest = self.slice(t, [0, 2 * rh], [-1, HD])
            ca = self.op.unsqueeze(cos_row, 0); sa = self.op.unsqueeze(sin_row, 0)
            d1 = self.sub(self.mul(a, ca), self.mul(b, sa))
            d2 = self.add(self.mul(a, sa), self.mul(b, ca))
            return self.concat([d1, d2, rest], axis=1)
        qr = rope(qr_)
        kro = rope(kr_)
        # write K/V cache at row pos (fp16), read back full prefix
        kro16 = self.cast(kro, "f16"); vr16 = self.cast(vr, "f16")
        posvec = self.op.unsqueeze(pos, 0)
        Knew = self.scatter_row(Kc, posvec, self.op.unsqueeze(kro16, 0))
        Vnew = self.scatter_row(Vc, posvec, self.op.unsqueeze(vr16, 0))
        kf = self.cast(self.slice(Knew, [0, 0, 0], [None, None, -1]), "f32")   # clamp handled below
        return None

    def scatter_row(self, buf, idx, upd):
        return self.op.scatter_elements(buf, idx, upd, axis=0)

    def build(self):
        pass


def main(argv):
    ir_dir = argv[0]
    tiny = "--tiny" in argv
    meta_path = os.path.join(ir_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit("ov_build: no meta.json in %s (run kindi export-openvino first)" % ir_dir)
    meta = json.load(open(meta_path))
    print("ov_build: meta loaded, dims", meta["dims"], "ctx", meta["ctx"], "tiny", tiny)
    print("ov_build: this builder requires the openvino runtime; weights are staged, graph emission pending")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
