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

Costs, measured on this host: the dequantiser runs at ~14-65M values/s and is nearly all of a document's cost, so a
48-layer pass over a document is ~17 minutes *whatever its token count* - the experts its tokens route to saturate
near all 128 within a couple of hundred tokens. `forward_docs` (and `tools.expert_merge.docbatch`, which prices it)
takes that union over a batch of documents instead, so the same ~13 minutes scores 1,785 tokens rather than 256, at
the price of holding one layer's f32 experts and the batch's logits.

A plan from `tools.expert_merge.prune` is applied with `set_prune`: the router's logits are masked and the top-k
renormalised, so the weights are untouched and the two arms of an NLL gate can share one loaded pack.
"""
import os
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


def mask_logits(logits, keep):
    """The router's scores with every expert outside `keep` at -inf. Unchanged when `keep` is None.

    Masking here rather than after the top-k is what makes a pruned model *this* model with fewer options: the
    softmax is over the kept set, so the probabilities the surviving experts are applied with are the ones the
    router would have given them, renormalised - not the old weights with some of them zeroed.
    """
    if keep is None:
        return logits
    keep = np.asarray(sorted(int(e) for e in keep), dtype=np.int64)
    n = logits.shape[-1]
    if keep.size == 0:
        raise ValueError("a layer with no experts kept has nothing to route to")
    if keep.min() < 0 or keep.max() >= n:
        raise ValueError("kept expert %d is outside the %d this router has" % (keep.max(), n))
    out = np.full(logits.shape, -np.inf, dtype=np.float32)
    out[:, keep] = logits[:, keep]
    return out


def masked_route(logits, k):
    """Top-`k` of the masked scores with Qwen3's renormalisation, `k` clamped to the experts that exist.

    A budget below the model's active count leaves fewer slots than the model was trained to fill, and taking the
    top-k of a vector holding four finite values and 124 `-inf` ones would spend slots on experts at probability
    zero. The clamp is what keeps a pruned layer well-defined; `prune` reports when it happens.
    """
    finite = int(np.isfinite(logits).sum(axis=-1).min())
    k = max(1, int(min(int(k), finite)))
    probs = softmax(logits)
    idx = np.argsort(-probs, axis=-1)[:, :k]
    w = np.take_along_axis(probs, idx, axis=1)
    return idx, w / w.sum(axis=1, keepdims=True)


def pool_to_stacks(logits, slot_map, router_used, stacks_used, n_stacks=None):
    """Route over the source's slots, then pool that decision onto the stacks the pack actually holds.

    A kept-router pack scores 128 slots but owns 64 merged experts, so a token's top-8 slots can land on fewer than
    eight stacks - two members of a group share one stack, and their weights add. Keeping the `stacks_used` heaviest
    distinct stacks is what makes this arm read the same number of matrices a token as the matched-fraction arm it is
    compared with, so the difference between them is the router and not the arithmetic.
    """
    idx, wts = masked_route(logits, int(router_used))
    stacks = slot_map[idx]
    n = int(logits.shape[0])
    width = int(n_stacks) if n_stacks is not None else int(stacks.max()) + 1
    total = np.zeros((n, width), dtype=np.float32)
    np.add.at(total, (np.arange(n)[:, None], stacks), wts.astype(np.float32))
    top = np.argsort(-total, axis=1, kind="stable")[:, :int(stacks_used)]
    w = np.take_along_axis(total, top, 1)
    return top, (w / np.maximum(w.sum(axis=1, keepdims=True), 1e-30))


def spans_of(seqs):
    """The (start, stop) of each sequence in a batch they are concatenated into, in the order given."""
    spans, off = [], 0
    for s in seqs:
        spans.append((off, off + len(s)))
        off += len(s)
    return spans


def per_doc(x, spans, fn):
    """Apply a per-sequence layer to each document of a batch in place, so nothing crosses a document boundary.

    Attention is the one part of a batched forward that must **not** be batched: token 0 of the second document
    has no business seeing the last token of the first. Passing each document's slice through the same code a
    single-document forward runs is what makes the batch exactly the serial computation rather than nearly it.
    """
    for a, b in spans:
        x[a:b] = fn(x[a:b])
    return x


class Qwen3Moe(object):
    def __init__(self, path, log=print, allow_partial=False, router_extra=None):
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
        self.prune = None                                    # None: every expert available, as the pack shipped
        # a router correction added to the pack's own rows, in memory: `tools.expert_merge.routerfit` fits a 64-row
        # map to the teacher's grouped routing and emits the *difference* from the pooled rows, so adding it here
        # measures the router the pack would carry without rewriting 8.9 GB of stacks to carry it
        self.router_extra = None
        if router_extra is not None:
            z = np.load(router_extra)
            ex = z["extra"] if "extra" in z.files else z["rows"]
            if ex.shape[1] != p.n_expert:
                raise ValueError("router extras score %d experts, the pack has %d" % (ex.shape[1], p.n_expert))
            if ex.shape[0] < self.n_layers:
                raise ValueError("router extras cover %d layers, the pack models %d" % (ex.shape[0], self.n_layers))
            self.router_extra = np.asarray(ex, dtype=np.float64)
            log("  router: +%d fitted rows a layer from %s" % (p.n_expert, os.path.basename(router_extra)))

    def set_prune(self, keep_per_layer):
        """Restrict each layer to a set of experts and leave everything else alone; `None` restores the model.

        The plan is one list of expert ids a layer, in layer order - what `tools.expert_merge.prune` writes. Applying
        it here rather than rewriting the pack is what lets the two arms of an NLL gate run in one process over the
        same weights: a pruned pack would have to be re-read, and a re-quantised one would differ in more than the
        experts that were dropped.
        """
        if keep_per_layer is None:
            self.prune = None
            return self
        if self.pack.router_map is not None:
            raise ValueError("a kept-router pack scores %d slots over %d stacks; a per-slot mask has no meaning here"
                             % (self.pack.router_rows, self.pack.n_expert))
        keep = [None if k is None else sorted(int(e) for e in k) for k in keep_per_layer]
        if len(keep) != self.n_layers:
            raise ValueError("a plan for %d layers cannot be applied to %d" % (len(keep), self.n_layers))
        self.prune = keep
        return self

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
        if self.router_extra is not None:
            logits = logits + h @ self.router_extra[il].T
        logits = mask_logits(logits, None if self.prune is None else self.prune[il])
        m = p.router_map
        if m is not None:
            # a kept-router pack: the source's routing, the merged stacks underneath it
            idx, wts = pool_to_stacks(logits, m, p.router_used, p.n_expert_used, n_stacks=p.n_expert)
            return logits, idx, wts
        idx, wts = masked_route(logits, p.n_expert_used)
        return logits, idx, wts

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

    def forward_docs(self, seqs, log=None):
        """Several documents in one forward, sharing each layer's expert work.

        Attention is per document - a document must not attend to its neighbour - but the sparse block is one batch,
        and the experts a layer unpacks are the union over it. For a single 256-token document that union already
        runs close to all 128, so adding a second document costs a GEMV rather than another dequant: the expensive
        half of the forward is paid once a layer for the corpus instead of once a layer a document. That is the whole
        cost model of a gate - `docs/EXPERT-PRUNE.md` priced a document at ~17 minutes for this reason.

        Positions restart at 0 for each document, which is what makes this the same computation as scoring them one
        at a time: a document's second token sees its own first token and nothing from the document before it. The
        arithmetic per token is the same operations in the same order, so the logits agree to fp32 rounding rather
        than bit for bit - a GEMV over 256 rows and one over 768 block differently inside BLAS.
        `tools.expert_merge.docbatch` measures that difference on the real pack and prices the batching; its
        `--selftest` covers the part that is a property rather than a measurement, that a batched forward keeps
documents apart and returns them in the order given.
        """
        p = self.pack
        seqs = [np.asarray(s, dtype=np.int64) for s in seqs]
        if not seqs:
            raise ValueError("no documents to forward")
        if any(len(s) > 4096 for s in seqs):
            raise ValueError("rope tables cover 4096 positions")
        spans = spans_of(seqs)
        x = np.concatenate([p.rows("token_embd.weight", s) for s in seqs], axis=0)
        for il in range(self.n_layers):
            per_doc(x, spans, lambda xs: self.attend(xs, il))       # per document: no cross-document attention
            x = self.moe(x, il, None)                               # one batch: one expert union a layer
            if log:
                log("    layer %d/%d" % (il + 1, self.n_layers))
        if self.truncated:
            return dict(per_doc=[dict(logits=None, hidden=x[a:b]) for a, b in spans])
        h = rmsnorm(x, p.tensor("output_norm.weight"), p.eps)
        z = h @ p.tensor("output.weight").T
        return dict(per_doc=[dict(logits=z[a:b], hidden=x[a:b]) for a, b in spans])

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
