"""tools/bitnet_qat/model.py — a minimal native BitNet b1.58 research harness, not a converter.

Implements the published recipe (arXiv 2402.10095 / JMLR 2025) honestly at research scale:
  - BitLinear: weights ternarised per-tensor by AbsMean in the forward pass, STE in the backward pass;
    activations quantised per-token by AbsMax (W1.58A8).
  - RMSNorm strictly before every BitLinear (zero-mean, unit-variance inputs for AbsMean).
  - SwiGLU MLP.
  - AdamW with the published asymmetry: weight decay on 1-D params (norms/scales), 0.0 on the 2-D
    latent matrices so the ternary scales are not collapsed.

This is a *research* harness. The token budget for a useful 0.5B from-scratch model is measured in
`smoke.py` and recorded in appendix.md: below ~3B parameters, published results show native ternary
underperforming fp16 at equal tokens (BitNet b1.58, JMLR 2025), and this box's CPU throughput makes
multi-billion-token runs infeasible locally. It is a framework for small experiments (50M class,
~1–10B tokens on a GPU node), not a production training pipeline.

  from tools.bitnet_qat.model import MiniMinion, adamw_from_model
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.serialization.add_safe_globals([int, float])


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps)
        return (x * v.rsqrt() * self.w).to(x.dtype)


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mode, scale):
        if mode == "w":
            w = x / scale.clamp_min(1e-8)
            return w.clamp(-1, 1).round() * scale
        m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        return (x / m).clamp(-128, 127).round() * m

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class BitLinear(nn.Linear):
    """Ternary weights (AbsMean, per-tensor), 8-bit activations (AbsMax, per-token), STE gradients."""

    def forward(self, x):
        # weights: fp32 latent, ternarised forward (AbsMean)
        s = self.weight.float().abs().mean().clamp_min(1e-8)
        w = _STE.apply(self.weight, "w", s)
        # activations: per-token absmax to int8 range, STE
        xq = _STE.apply(x, "a", None)
        return F.linear(xq, w.to(x.dtype), self.bias)


class Block(nn.Module):
    def __init__(self, d, hidden, n_head):
        super().__init__()
        self.n_head, self.head_dim = n_head, d // n_head
        self.attn_norm = RMSNorm(d)
        self.qkv = BitLinear(d, 3 * d, bias=False)
        self.proj = BitLinear(d, d, bias=False)
        self.mlp_norm = RMSNorm(d)
        self.w1 = BitLinear(d, hidden, bias=False)   # SwiGLU gate
        self.w3 = BitLinear(d, hidden, bias=False)   # SwiGLU up
        self.w2 = BitLinear(hidden, d, bias=False)

    def forward(self, x):
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(x.size(0), x.size(1), 3, self.n_head, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))  # (B, heads, T, head_dim)
        # simple causal attention (research scale, no flash kernel)
        T = x.size(1)
        mask = torch.ones(1, 1, T, T, device=x.device).tril()
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask == 0, float("-inf")).softmax(-1)
        h = (att @ v).transpose(1, 2).reshape(x.size(0), T, -1)
        x = x + self.proj(h)
        h = self.mlp_norm(x)
        gate = F.silu(self.w1(h)) * self.w3(h)
        return x + self.w2(gate)


class MiniMinion(nn.Module):
    def __init__(self, vocab=32000, d=512, layers=8, heads=8, hidden=1408):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, hidden, heads) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)  # LM head stays fp16
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and not isinstance(m, BitLinear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.tok(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def adamw_from_model(model, lr=3e-2, wd_1d=0.01, betas=(0.9, 0.999), eps=1e-8):
    """Published asymmetry: weight decay on 1-D params only, 0.0 on latent 2-D matrices.
    LR is 5–10× a standard fp16 AdamW (3e-3 → 3e-2) to compensate for STE attenuation."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim <= 1 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd_1d},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=eps)
