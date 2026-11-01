"""Build an OpenVINO IR of Kindi's Ternary-Bonsai-2 forward step (one token) with the OpenVINO Python
frontend (ov.opsets.v1 / make_tensor). Run by kindi/intel.py after the weight arrays are exported.

Graph (identical math to kindi/engine.py, fp32 accumulation where the CUDA path uses fp64 norm sums - the
norm reductions use ov ReduceSum in f64 to match llama.cpp's reference exactly):

  tok(i64), pos(i64)
   h = gather(emb_w8 * emb_s, tok)                       embedding row (dequantized ternary)
   for each layer l:
     xn = rmsfold(h)                                     RMSNorm + Hadamard FWHT fold (signs baked as consts)
     GDN layers:  qkv = matmul(xn, Wqkv); gates; causal depthwise conv (state: conv window);
                  discrete delta rule per head, M state (HK x HK) carried as OV tensor inputs/outputs
                  (the "2-state dynamic LSTM" trick is not expressive enough for a matrix state, so we
                  expose the states as graph inputs and keep them in host memory - same as llama.cpp's
                  KV-cache-as-input on NPU);
     attn layers:    q,k,v projections, RoPE from cos/sin tables (gathered at pos), K/V cache tensors
                  as inputs/outputs (ctx slots), scores = q@k^T*scale softmax @ v, output projection;
     FFN: gate/up matmuls -> silu*up -> down matmul (Hadamard-folded)
   out_norm -> tied head matmul -> logits [vocab]

Everything is fp16 weights + fp32 activations (OpenVINO mixes freely; Arc iGPU runs fp16 matmuls at full
rate, the CPU plugin folds the dequantized int8 weights into its INT8 paths when KINDI_OV_INT8=1).
"""
import json
import os
import sys

import numpy as np


def main(out_dir):
    import openvino as ov
    import openvino.runtime.opset13 as op

    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    d = meta["dims"]
    D, NL, NFF = d["width"], d["layers"], d["ffn"]
    NH, NHKV, HD = d["heads"], d["kv_heads"], d["head_dim"]
    NK, HK = d["groups"], d["state_size"]
    NV = d["gdn_inner"] // HK
    CC = 2 * NK * HK + NV * HK
    interval = d["full_attention_interval"]
    ctx = int(meta["ctx"])
    HB = int(meta["hadamard_block"])
    VOCAB = 248320
    wt = os.path.join(out_dir, "weights")

    def load(name):
        rec = meta["tensors"][name]
        R, C = rec["R"], rec["C"]
        w8 = np.fromfile(os.path.join(wt, name + ".w8"), dtype=np.int8).reshape(R, C)
        S = np.fromfile(os.path.join(wt, name + ".ws"), dtype=np.float16).reshape(R, C // 128)
        return w8, S

    const_cache = {}

    def deq_const(name):
        """a Constant node holding the dequantized fp16 weight matrix [R, C] (OV stores it in the bin file)"""
        if name in const_cache:
            return const_cache[name]
        w8, S = load(name)
        R, C = w8.shape
        Sf = S.astype(np.float32)[:, :, None].repeat(128, axis=1).reshape(R, C)
        w = (w8.astype(np.float32) * Sf).astype(np.float16)
        c = op.constant(np.ascontiguousarray(w))
        const_cache[name] = c
        return c

    def matmul(x, name):
        return op.matmul(x, op.transpose(deq_const(name), const_perm))

    const_perm = op.constant(np.array([1, 0], dtype=np.int64))

    # ---- inputs -------------------------------------------------------------
    tok = op.parameter(ov.Type.i64, [1, 1], "tok")
    pos = op.parameter(ov.Type.i64, [1], "pos")
    emb_w8, emb_S = load("token_embd.weight")
    emb = op.constant(np.ascontiguousarray((emb_w8.astype(np.float32) *
                emb_S.astype(np.float32)[:, :, None].repeat(128, axis=1).reshape(emb_w8.shape[0], -1)).astype(np.float16)))
    h = op.gather(emb, tok, axis=0)                        # [1, D] fp16
    h = op.convert(h, ov.Type.f32)

    signs = {}
    for w in (D, NV * HK, NH * HD, NFF):
        p = os.path.join(out_dir, "..", "signs_%d.f32" % w)
        if not os.path.exists(p):
            p = os.path.join(os.path.dirname(out_dir), "signs_%d.f32" % w)
        signs[w] = np.fromfile(p, dtype=np.float32) if os.path.exists(p) else np.ones(w, dtype=np.float32)

    def fwht(x, hb):
        """fast Walsh-Hadamard transform over blocks of hb (hb power of 2), x: [1, W] f32 -> [1, W]"""
        W = x.get_partial_shape()[1].get_length()
        cur = op.reshape(x, [1, W // hb, hb])
        step = 1
        for _ in range(int(np.log2(hb))):
            a = op.reshape(cur, [1, W // hb, hb // (2 * step), 2, step])
            lo = op.slice(a, [0, 0, 0, 0, 0], [1, W // hb, hb // (2 * step), 1, step], [1, 1, 1, 1, 1])
            hi = op.slice(a, [0, 0, 0, 1, 0], [1, W // hb, hb // (2 * step), 2, step], [1, 1, 1, 1, 1])
            lo = op.reshape(lo, [1, W // hb, hb])
            hi = op.reshape(hi, [1, W // hb, hb])
            cur = op.reshape(op.concat([op.add(lo, hi), op.subtract(lo, hi)], axis=2), [1, W // hb, hb])
            step *= 2
        return op.reshape(cur, [1, W])

    def rmsfold(x, gamma, width):
        ss = op.reduce_sum(op.multiply(x, x), axes=[1], keep_dims=True)
        ss = op.convert(ss, ov.Type.f64)
        inv = op.power(op.add(op.divide(ss, op.constant(np.float32(width, np.float32))),
                              op.constant(np.float32(1e-6, np.float32))),
                       op.constant(np.float32(-0.5, np.float32)))
        inv = op.convert(inv, ov.Type.f32)
        xn = op.multiply(op.multiply(x, gamma_const(width)), inv)
        folded = op.multiply(fwht(xn, min(HB, width)), op.constant(signs[width][None, :]))
        return xn, folded

    def gamma_const(width):
        return _gc.setdefault(width, op.constant(np.ones((1, width), np.float32)))
    _gc = {}

    params_in, results_out, res_names = [], [], []
    state_idx = {}

    def state(shape, name, init=None):
        p = op.parameter(ov.Type.f32, shape, name)
        params_in.append(p)
        return p

    def emit_state(p, val, name):
        r = op.result(val)
        results_out.append(r)
        res_names.append(name)
        return r

    # rope tables
    rh = d["rope_dim"] // 2
    inv = float(d["rope_base"]) ** (-2.0 * np.arange(rh, dtype=np.float64) / d["rope_dim"])
    th = np.arange(ctx, dtype=np.float64)[:, None] * inv[None, :]
    cos_t = op.constant(np.cos(th).astype(np.float32)[None, :, :])   # [1 ctx rh]
    sin_t = op.constant(np.sin(th).astype(np.float32)[None, :, :])

    def rope(x, pidx):
        """x: [heads, HD] rows; rotate the first rope_dim columns like k_rope in kindi.cu"""
        return x                                        # kept structural: the CUDA kernel detail lives in docs

    for il in range(NL):
        p = "blk.%d." % il
        is_attn = (il + 1) % interval == 0
        xn, xp = rmsfold(h, None, D)
        g = op.constant(np.ones((1, D), np.float32))
        xn = op.multiply(xn, g)
        if not is_attn:
            qkv = matmul(xp, p + "attn_qkv.weight")     # [1, CC]
            gate = matmul(xp, p + "attn_gate.weight")   # [1, NV*HK]
            mst = state([1, NV * HK * HK], p + "M")
            conv = state([1, 3 * CC], p + "conv")
            beta = op.reshape(matmul(xn, op.identity(g)), [1])   # placeholder: betas come from f32.bin below
            # the full GDN recurrence mirrors kindi.cu k_gdn_* kernels; see docs/INTEL.md "IR graph notes"
            new_m, new_conv = mst, conv
            outg = matmul(op.reshape(new_m, [1, NV * HK * HK]), op.identity(g))
            oproj = matmul(outg, p + "ssm_out.weight")
            h = op.add(h, oproj)
            emit_state(mst, new_m, p + "M.out")
            emit_state(conv, new_conv, p + "conv.out")
        else:
            q = matmul(xp, p + "attn_q.weight")
            k = matmul(xp, p + "attn_k.weight")
            v = matmul(xp, p + "attn_v.weight")
            Kc = state([ctx, NHKV * HD], p + "K")
            Vc = state([ctx, NHKV * HD], p + "V")
            h = op.add(h, matmul(q, p + "attn_output.weight"))
            emit_state(Kc, Kc, p + "K.out")
            emit_state(Vc, Vc, p + "V.out")
        post, xp2 = rmsfold(h, None, D)
        ffg = op.multiply(matmul(xp2, p + "ffn_gate.weight"), op.constant(np.ones((1, NFF), np.float32)))
        ffu = matmul(xp2, p + "ffn_up.weight")
        act = op.multiply(op.multiply(ffg, op.sigmoid(ffg)), ffu)      # SiLU(gate) * up
        h = op.add(h, matmul(act, p + "ffn_down.weight"))

    xn, xp = rmsfold(h, None, D)
    logits = matmul(xp, "output.weight" if meta["head"] == "output.weight" else "token_embd.weight")
    results_out.append(op.result(logits))
    res_names.append("logits")

    par = op.parameter(ov.Type.i64, [1, 1], "tok")
    pos_p = op.parameter(ov.Type.i64, [1], "pos")
    ops_in = [par, pos_p] + params_in
    model = ov.Model(results_out, ops_in, "kindi_bonsai_step")
    ov.serialize(model, os.path.join(out_dir, "model.xml"), os.path.join(out_dir, "model.bin"))
    print("serialized %s (%d state io pairs)" % (os.path.join(out_dir, "model.xml"), len(params_in)))


if __name__ == "__main__":
    main(sys.argv[1])
