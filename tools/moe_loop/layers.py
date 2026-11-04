"""The design note's block, made runnable, with the corrections the note needs.

Three pieces, each the note's own sketch or a corrected version of it, so the differences can be measured instead of
argued:

* `BitLinear158` - ternary weights ({-1, 0, +1} times one absmean scale) and int8 activations per token, both through
  a straight-through estimator. The note's version subclasses `nn.Linear`; this one does not, because the quantised
  weight has to be inspectable (the test asserts it takes three values, and that a gradient still flows through it).
* `AvgHeadRouter` - the note's router: two heads over disjoint halves of the hidden state, their logits **averaged**
  and one top-k taken over the average. It is a two-view ensemble, not sub-vector routing: there is one expert set and
  one choice, and the chosen experts are applied to the *whole* hidden state.
* `SplitRouter` - what "routing sub-vectors to different micro-experts" has to mean: each head owns its own group of
  experts, each expert maps that half of the hidden state to itself, and the two halves never share a selection.

`LoopedMoEBlock` runs the same parameters several times with a damped residual, and can dispatch either the note's way
(mask and multiply over every expert) or the sparse way (`index_add_`), so the cost difference is measurable rather
than claimed. `forward(x, halt=...)` takes a `[B, loops]` per-sequence halt mask (`tools.exodus.router.halt_mask`): a
halted row stops being routed and stops costing work, and its state is the one a shorter loop would have produced.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary(w):
    """{-1, 0, +1} with one scale per tensor: the absmean rule, `|w| < scale/2` rounds to zero."""
    scale = w.abs().mean().clamp(min=1e-5)
    q = (w / scale).round().clamp(-1, 1)
    return q * scale, scale


def quant_int8(x, eps=1e-5):
    """Per-token int8 absmax quantisation: one scale per row of the (..., d) activation."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    return (x * scale).round().clamp(-128, 127) / scale


class BitLinear158(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)          # what nn.Linear does: an O(1) update per layer
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.quantise = True                # the float twin: same parameters, quantisation off

    def quantised(self):
        return ternary(self.weight)

    def forward(self, x):
        if not self.quantise:
            return F.linear(x, self.weight, self.bias)
        w_q, _scale = ternary(self.weight)
        x_q = quant_int8(x)
        w_eff = self.weight + (w_q - self.weight).detach()      # straight-through: value w_q, gradient 1
        x_eff = x + (x_q - x).detach()
        return F.linear(x_eff, w_eff, self.bias)


def _topk(logits, k):
    """Top-k over a probability vector, renormalised - the same rule the engine's reference router uses."""
    p = F.softmax(logits, dim=-1)
    w, idx = torch.topk(p, k, dim=-1)
    return w / w.sum(-1, keepdim=True).clamp(min=1e-9), idx


class AvgHeadRouter(nn.Module):
    """The note's router: two heads see half the state each and their logits are averaged."""

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.head1 = nn.Linear(d_model // 2, num_experts, bias=False)
        self.head2 = nn.Linear(d_model // 2, num_experts, bias=False)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return _topk((self.head1(x1) + self.head2(x2)) * 0.5, self.top_k)


class SplitRouter(nn.Module):
    """Each head owns half the experts; a head's selection never crosses into the other group."""

    def __init__(self, d_model, experts_per_group=2, top_k=1):
        super().__init__()
        self.top_k = top_k
        self.per = experts_per_group
        self.head1 = nn.Linear(d_model // 2, experts_per_group, bias=False)
        self.head2 = nn.Linear(d_model // 2, experts_per_group, bias=False)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        w1, i1 = _topk(self.head1(x1), self.top_k)
        w2, i2 = _topk(self.head2(x2), self.top_k)
        return (w1, i1), (w2, i2)


class LoopedMoEBlock(nn.Module):
    """RMSNorm -> router -> experts -> damped residual, repeated `loops` times over the same parameters."""

    def __init__(self, d_model=32, num_experts=8, top_k=2, loops=3, alpha=0.5, dispatch="index_add",
                 router="avg", experts_per_group=2):
        super().__init__()
        self.loops, self.alpha, self.dispatch = loops, alpha, dispatch
        self.norm = nn.RMSNorm(d_model)
        self.top_k = top_k
        if router == "avg":
            self.router = AvgHeadRouter(d_model, num_experts, top_k)
            self.experts = nn.ModuleList([nn.Sequential(BitLinear158(d_model, d_model * 2),
                                                        nn.SiLU(),
                                                        BitLinear158(d_model * 2, d_model))
                                          for _ in range(num_experts)])
            self.router_kind = "avg"
        else:
            self.router = SplitRouter(d_model, experts_per_group, top_k)
            half = d_model // 2
            self.experts = nn.ModuleList([nn.Sequential(BitLinear158(half, half * 2),
                                                        nn.SiLU(),
                                                        BitLinear158(half * 2, half))
                                          for _ in range(2 * experts_per_group)])
            self.router_kind = "split"
        self.calls = 0                       # expert forwards, to make the dispatch difference countable
        self.rows = 0                        # token-rows an expert actually multiplied: the work, not the calls
        self.seen_width = None               # the width an expert was actually given
        self.active_rows = 0                 # row-loop units computed: what halting is supposed to reduce
        self.loops_run = 0                   # loops with at least one live row

    def _expert(self, e, h):
        self.calls += 1
        self.rows += h.numel() // h.shape[-1]
        self.seen_width = h.shape[-1]
        return self.experts[e](h)

    def _dispatch_avg(self, normed, weights, indices):
        """`mask`: the note's loop - every expert on the whole batch, then multiply by a one-hot mask.

        `index_add`: each selected expert runs on the rows that chose it, and the weighted results are summed back in.
        That is the same arithmetic with the wasted work removed, which is what makes the two countable against each
        other.
        """
        if self.dispatch == "mask":
            out = torch.zeros_like(normed)
            for slot in range(weights.shape[-1]):
                idx, w = indices[..., slot], weights[..., slot].unsqueeze(-1)
                for e in range(len(self.experts)):
                    mask = (idx == e).unsqueeze(-1)
                    if mask.any():
                        out = out + mask * (w * self._expert(e, normed))
            return out
        flat = normed.reshape(-1, normed.shape[-1])
        acc = torch.zeros_like(flat)
        for slot in range(weights.shape[-1]):
            idx = indices[..., slot].reshape(-1)
            w = weights[..., slot].reshape(-1)
            for e in torch.unique(idx).tolist():
                sel = idx == e
                y = self._expert(e, flat[sel])
                acc.index_add_(0, sel.nonzero(as_tuple=True)[0], y * w[sel].unsqueeze(-1))
        return acc.reshape(normed.shape)

    def _dispatch_split(self, normed, routed):
        """Each head's experts see only that head's half of the hidden state, and write back into it."""
        (w1, i1), (w2, i2) = routed
        half = normed.shape[-1] // 2
        parts = []
        for x, w, i, base in ((normed[..., :half], w1, i1, 0),
                              (normed[..., half:], w2, i2, self.router.per)):
            flat_x = x.reshape(-1, x.shape[-1])
            acc = torch.zeros_like(flat_x)
            for slot in range(w.shape[-1]):
                idx = i[..., slot].reshape(-1)
                ww = w[..., slot].reshape(-1)
                for e in torch.unique(idx).tolist():
                    sel = idx == e
                    y = self._expert(base + e, flat_x[sel])
                    acc.index_add_(0, sel.nonzero(as_tuple=True)[0], y * ww[sel].unsqueeze(-1))
            parts.append(acc.reshape(x.shape))
        return torch.cat(parts, dim=-1)

    def forward(self, x, collect_norms=False, halt=None):
        """One pass over the looped block; `halt` is the per-sequence mask that ends a row's loop early.

        `halt` is `[B, loops]` of booleans, cumulative along the loop axis, exactly as
        `tools.exodus.router.halt_mask` builds it: `halt[b, l]` means row b is already finished and must not run loop l.
        A halted row is dropped from the norm, the router and the experts, so it costs nothing from then on, and its
        residual state stays where it was - which is the same value a shorter loop would have produced, since nothing
        later reads a halted row. That equality is the property the measurement checks; `halting` does not change any
        live row's arithmetic, because every operation here is row-wise.

        `halt=None` runs every row for every loop, which is what every earlier caller did.
        """
        h = x
        norms = []
        self.active_rows = 0
        self.loops_run = 0
        for loop in range(self.loops):
            live = None if halt is None else ~halt[:, loop]
            if live is not None and not bool(live.any()):
                break
            self.loops_run += 1
            rows = h if live is None else h[live]
            self.active_rows += int(rows.shape[0])
            normed = self.norm(rows)
            if self.router_kind == "avg":
                w, i = self.router(normed)
                out = self._dispatch_avg(normed, w, i)
            else:
                out = self._dispatch_split(normed, self.router(normed))
            if live is None:
                h = h + self.alpha * out
            else:
                h = h.index_add(0, live.nonzero(as_tuple=True)[0], self.alpha * out)
            if collect_norms:
                norms.append(float(h.norm()) / max(1e-9, float(x.shape[-1]) ** 0.5))
        return (h, norms) if collect_norms else h
