"""A forward pass over a real Qwen3-MoE, in numpy, so the router log is the model's own.

`kindi` serves dense packs; this file exists because the question - are the experts a router groups together the
same function wearing a permuted hidden layer, or merely similar functions - needs the router's real decisions on
real weights, and no pack on this machine is a MoE. The model is Qwen3-Coder-30B-A3B (48 layers, 128 experts,
top-8, expert width 768) read straight out of its Q2_K GGUF, dequantised a layer at a time.

The forward is deliberately plain: RMSNorm, GQA attention with QK-norm and rope, then the sparse block with the
top-8 weights renormalised (Qwen3's `norm_topk_prob`). One prefill, no generation. What comes out is exactly what
the measurement needs: for every layer and token, the 128 router logits, the eight experts chosen and their
weights, and the vector `x_n` the experts saw (`ffn_norm(x)`), which is the input distribution any merge has to
be judged on.

Costs, measured on this host: ~80M dequantised values/s, so one 48-layer pass over 128 tokens is a few minutes,
and only the experts a token actually routes to are unpacked.
"""
import numpy as np

from .ggufmoe import Moepack


def rmsnorm(x, w, eps=1e-6):
    s = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return (x * s) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_tables(n, dim, base):
    inv = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    pos = np.arange(n, dtype=np.float64)[:, None]
    ang = pos * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, cos, sin):
    """[n, heads, dim] with the half-split rotation (llama.cpp's NORM rope)"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    n = x.shape[0]
    c = cos[:n][:, None, :]
    s = sin[:n][:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(np.float32)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Qwen3Moe(object):
    def __init__(self, path, log=print, allow_partial=False):
        self.pack = Moepack(path)
        p = self.pack
        self.d = p.describe()
        self.cos, self.sin = rope_tables(4096, p.head_dim, p.rope_base)
        log("  %s: %d layers, %d experts (%d active), width %d, expert width %d, %s"
            % (self.d["arch"], p.n_layer, p.n_expert, p.n_expert_used, p.n_embd, p.n_ff_exp,
               "/".join("%s x%d" % kv for kv in sorted(p.type_names().items()))))
        # a pack whose tail does not decode is refused here rather than modelled short: a forward over 13 of 48
        # layers produces a router log that looks like a result, and nothing downstream can tell the difference
        rep = p.require_complete(log=log, allow_partial=allow_partial)
        # a pack that decodes everywhere models every layer; one that does not, and was asked for anyway, models
        # the prefix that decodes and says so - the caller sees `truncated`
        self.n_layers = p.n_layer if rep["ok"] else rep["layers_ok"]
        self.truncated = self.n_layers < p.n_layer
        self._attn = {}

    # --- pieces -----------------------------------------------------------------------------------------------
    def attn_weights(self, il):
        w = self._attn.get(il)
        if w is None:
            p, s = self.pack, "blk.%d." % il
            w = dict(q=p.tensor(s + "attn_q.weight"), k=p.tensor(s + "attn_k.weight"),
                     v=p.tensor(s + "attn_v.weight"), o=p.tensor(s + "attn_output.weight"),
                     qn=p.tensor(s + "attn_q_norm.weight"), kn=p.tensor(s + "attn_k_norm.weight"),
                     an=p.tensor(s + "attn_norm.weight"), pn=p.tensor(s + "ffn_norm.weight"),
                     router=p.tensor(s + "ffn_gate_inp.weight"))
            self._attn[il] = w
        return w

    def attend(self, x, il):
        w = self.attn_weights(il)
        p = self.pack
        n, H, Hkv, D = x.shape[0], p.n_head, p.n_head_kv, p.head_dim
        h = rmsnorm(x, w["an"], p.eps)
        q = (h @ w["q"].T).reshape(n, H, D)
        k = (h @ w["k"].T).reshape(n, Hkv, D)
        v = (h @ w["v"].T).reshape(n, Hkv, D)
        q = rope(rmsnorm(q, w["qn"], p.eps), self.cos, self.sin)
        k = rope(rmsnorm(k, w["kn"], p.eps), self.cos, self.sin)
        rep = H // Hkv
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(D)
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        sc = np.where(mask[None], -np.inf, sc)
        ctx = np.einsum("hts,shd->thd", softmax(sc), v).reshape(n, H * D)
        return x + ctx @ w["o"].T

    def route(self, h, il):
        p = self.pack
        logits = h @ self.attn_weights(il)["router"].T
        probs = softmax(logits)
        idx = np.argsort(-probs, axis=-1)[:, :p.n_expert_used]
        w = np.take_along_axis(probs, idx, axis=1)
        return logits, idx, w / w.sum(axis=1, keepdims=True)

    def moe(self, x, il, capture=None):
        """Sparse block: only the experts some token in this batch chose are unpacked."""
        p = self.pack
        s = "blk.%d." % il
        h = rmsnorm(x, self.attn_weights(il)["pn"], p.eps)
        logits, idx, wts = self.route(h, il)
        if capture is not None:
            capture["logits"].append(logits.astype(np.float16))
            capture["ids"].append(idx.astype(np.int16))
            capture["wts"].append(wts.astype(np.float16))
            capture["xn"].append(h.astype(np.float16))
        out = np.zeros_like(x)
        for e in np.unique(idx):
            sel = idx == e                                    # [n, k] bool: which slots picked expert e
            tok, slot = np.nonzero(sel)
            xe = h[tok]
            g = xe @ p.expert_weight(s + "ffn_gate_exps.weight", int(e)).T
            u = xe @ p.expert_weight(s + "ffn_up_exps.weight", int(e)).T
            d = (silu(g) * u) @ p.expert_weight(s + "ffn_down_exps.weight", int(e)).T
            np.add.at(out, tok, d * wts[tok, slot][:, None])
        return x + out

    def forward(self, ids, capture=False, log=None):
        p = self.pack
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) > 4096:
            raise ValueError("rope tables cover 4096 positions")
        x = p.rows("token_embd.weight", ids)
        cap = None
        if capture:
            cap = dict(logits=[], ids=[], wts=[], xn=[])
        for il in range(self.n_layers):
            x = self.attend(x, il)
            x = self.moe(x, il, cap)
            if log:
                log("    layer %d/%d" % (il + 1, self.n_layers))
        if self.truncated:
            out = dict(logits=None, hidden=x)
        else:
            h = rmsnorm(x, p.tensor("output_norm.weight"), p.eps)
            out = dict(logits=h @ p.tensor("output.weight").T, hidden=x)
        if cap is not None:
            out.update(dict(route_logits=np.stack(cap["logits"]), route_ids=np.stack(cap["ids"]),
                            route_wts=np.stack(cap["wts"]), xn=np.stack(cap["xn"])))
        return out

    def next_tokens(self, ids, k=8, log=None):
        if self.truncated:
            raise ValueError("the pack only decodes through layer %d of %d: there is no next-token distribution"
                             % (self.n_layers - 1, self.pack.n_layer))
        ids = np.asarray(ids, dtype=np.int64)
        got = self.forward(ids, log=log)
        lp = got["logits"][-1]
        pr = softmax(lp.astype(np.float64))
        top = np.argsort(-pr)[:k]
        return [(int(i), float(pr[i])) for i in top], got


# --- tokenizer from the pack's own metadata ------------------------------------------------------------------

def tokenizer(pack, tmpdir=None):
    """kindi's byte-level BPE over the GGUF's own token/merge arrays, so the corpus is the model's tokenisation."""
    import os
    import tempfile
    from kindi.tokenizer import Tok
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="moe-tok-")
    path = os.path.join(tmpdir, "merges.txt")
    with open(path, "w", encoding="utf-8") as f:
        for m in pack.g.kv.get("tokenizer.ggml.merges", []):
            f.write(m + "\n")
    vocab = pack.g.kv.get("tokenizer.ggml.tokens")
    if not isinstance(vocab, list):
        raise ValueError("the pack has no tokenizer.ggml.tokens array")
    return Tok(vocab, path)
