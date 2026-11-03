"""Attention-head slicing: what is exact about it, and what the supplied profiler gets wrong.

Slicing heads is arithmetic before it is a quality claim. For a head set S, the attention output is a sum over heads, so a
sliced block must compute exactly the kept heads' contribution and nothing else - `q/k/v` rows for the kept heads and
`o_proj` *columns* for the same heads. Test `slice` checks that against an explicit per-head reference to 1e-13, and
test `column` shows the row slice of `o_proj` that the supplied code has no counterpart for.

The profiler it supplies does not compile into that. Three defects:

1. **`register_backward_hook` on `o_proj` returns a 3-D gradient** (`[B, S, d_model]`), and the hook takes
   `torch.norm(g, dim=(0, 1, 3))`. Dimension 3 does not exist. Test `hook` runs it and records the exception, then shows
   the reshape that gives per-head energies.

2. **The metric is not the formula.** The specification defines `I_h = E |dL/dA_h * A_h|`; the code accumulates
   `||dL/dA_h||` and never multiplies by the activation. The two rank heads in opposite orders as soon as a head has a
   large activation and a small gradient. Test `metric` constructs that case.

3. **A fused QKV block is not laid out as `h * d_head`.** For packed `[q h0, k h0, v h0, q h1, ...]` the row offset of
   head `h` is `3 * h * d_head` (plus the q/k/v selector), so the supplied `idx * d_head` reads another head's rows.
   Test `fused` measures which rows each convention actually selects.

    python -m tools.exodus.heads
"""
import warnings

import numpy as np
import torch

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# --- reference attention, per head ------------------------------------------------------------------------------------

def ref_attention(x, wq, wk, wv, wo, heads, dh=None):
    """Explicit per-head attention. Returns (total output, list of per-head contributions scaled into d_model).

    `dh` must be passed for a sliced block: its `d_model` is unchanged but it has fewer heads, so `d // heads` is the
    wrong head width - the same confusion that makes a sliced o_proj hard to get right.
    """
    b, s, d = x.shape
    dh = dh or d // heads
    head_outs = []
    for h in range(heads):
        sl = slice(h * dh, (h + 1) * dh)
        q = x @ wq[sl].T
        k = x @ wk[sl].T
        v = x @ wv[sl].T
        a = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dh))
        head_outs.append((a @ v) @ wo[:, sl].T)
    return sum(head_outs), head_outs


def slice_rows(w, idx, dh):
    return np.concatenate([w[h * dh:(h + 1) * dh] for h in idx], axis=0)


def slice_cols(w, idx, dh):
    return np.concatenate([w[:, h * dh:(h + 1) * dh] for h in idx], axis=1)


def sliced_attention(x, wq, wk, wv, wo, idx, dh, heads):
    """A block with only heads `idx` kept: q/k/v rows, o_proj columns, and its own head count."""
    return ref_attention(x, slice_rows(wq, idx, dh), slice_rows(wk, idx, dh), slice_rows(wv, idx, dh),
                         slice_cols(wo, idx, dh), len(idx), dh=dh)[0]


# --- tests ------------------------------------------------------------------------------------------------------------

def _problem(heads=4, d=16, s=5, b=2, seed=0):
    rs = np.random.RandomState(seed)
    dh = d // heads
    x = rs.randn(b, s, d)
    wq, wk, wv = (rs.randn(d, d) / np.sqrt(d) for _ in range(3))
    wo = rs.randn(d, d) / np.sqrt(d)
    return x, wq, wk, wv, wo, dh


def t_slice():
    x, wq, wk, wv, wo, dh = _problem()
    total, per_head = ref_attention(x, wq, wk, wv, wo, heads=4)
    idx = [0, 2]
    kept = sliced_attention(x, wq, wk, wv, wo, idx, dh, heads=4)
    want = per_head[0] + per_head[2]
    err = float(np.abs(kept - want).max())
    check("slice: two kept heads equal their contribution to the full block", err < 1e-13,
          "max abs error %.1e" % err)
    check("slice: dropping heads changes the output, so the claim is about what is kept",
          float(np.abs(kept - total).max()) > 1e-2,
          "kept vs full differ by %.3f" % float(np.abs(kept - total).max()))


def t_column():
    """o_proj must be sliced on columns for the kept heads; rows would mix other heads' inputs."""
    x, wq, wk, wv, wo, dh = _problem(seed=1)
    idx = [1, 3]
    right = sliced_attention(x, wq, wk, wv, wo, idx, dh, heads=4)
    # the same block but with o_proj sliced on rows instead - what a row-only slicer produces
    wrong_wo = np.concatenate([wo[h * dh:(h + 1) * dh] for h in idx], axis=0)
    wrong = ref_attention(x, slice_rows(wq, idx, dh), slice_rows(wk, idx, dh), slice_rows(wv, idx, dh),
                          wrong_wo, len(idx))[0] if wrong_wo.shape[1] == 2 * dh else None
    check("column: o_proj sliced on columns has the kept heads' input width", slice_cols(wo, idx, dh).shape == (16, 2 * dh),
          "shape %s" % (slice_cols(wo, idx, dh).shape,))
    check("column: a row slice of o_proj has the wrong width for the sliced q", wrong is None,
          "row slice is %s, needs input width %d" % (wrong_wo.shape, len(idx) * dh))
    check("column: the column-sliced block still computes", right.shape == (2, 5, 16),
          "output %s" % (right.shape,))


def t_hook():
    """The supplied profiler's hook cannot run: o_proj's gradient is 3-D and dim 3 does not exist."""
    torch.manual_seed(0)
    d, heads, s, b = 16, 4, 5, 2
    dh = d // heads
    o_proj = torch.nn.Linear(d, d, bias=False)
    x = torch.randn(b, s, d, requires_grad=True)
    seen = {}

    def hook(module, grad_input, grad_output):
        g = grad_output[0].detach()
        seen["shape"] = tuple(g.shape)
        try:
            seen["spec"] = torch.norm(g, dim=(0, 1, 3)).cpu()
            seen["error"] = None
        except (IndexError, RuntimeError) as e:
            seen["error"] = "%s: %s" % (type(e).__name__, e)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = o_proj.register_backward_hook(hook)
        (o_proj(x) ** 2).sum().backward()
        h.remove()
    deprecations = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    check("hook: o_proj's grad_output is [B, S, d_model]", seen["shape"] == (b, s, d),
          "shape %s" % (seen["shape"],))
    check("hook: the specification's dim=(0, 1, 3) raises", seen["error"] is not None,
          seen["error"] or "it did not raise")
    check("hook: register_backward_hook is deprecated, not merely wrong", bool(deprecations),
          (deprecations[0][:90] + "...") if deprecations else "no warning")
    g = torch.randn(b, s, heads, dh)
    per_head = torch.linalg.vector_norm(g, dim=(0, 1, 3))
    check("hook: reshaping to [B, S, H, d_head] gives per-head energies", tuple(per_head.shape) == (heads,),
          "energies %s" % [round(float(v), 3) for v in per_head])


def t_metric():
    """|grad * activation| and ||grad|| rank heads differently, so the code is not computing the formula."""
    act = np.array([1.0, 30.0])                 # head 1 has the large activation
    grad = np.array([10.0, 0.5])                # head 0 has the large gradient
    spec = np.abs(grad * act)                   # [10, 15] - head 1 is the important one
    code = np.abs(grad)                         # [10, 0.5] - head 0 is
    check("metric: the two orderings disagree", int(np.argmax(spec)) != int(np.argmax(code)),
          "<|grad*A|> picks head %d (%.1f vs %.1f), <|grad|> picks head %d (%.1f vs %.1f)"
          % (int(np.argmax(spec)), spec[0], spec[1], int(np.argmax(code)), code[0], code[1]))
    ratio = spec / code
    check("metric: the difference is the activation, which the code drops",
          np.allclose(ratio, np.abs(act)),
          "|grad*A| / ||grad|| per head = %s, activations are %s"
          % ([round(float(v), 3) for v in ratio], [round(float(v), 3) for v in act]))


def t_fused():
    """Packed QKV rows are 3 * h * d_head apart, not h * d_head."""
    heads, dh = 4, 4
    packed = np.arange(3 * heads * dh).reshape(3 * heads * dh, 1)      # row index tells you whose row it is
    h = 2
    naive = packed[h * dh:(h + 1) * dh, 0]
    correct = packed[(3 * h) * dh:(3 * h + 1) * dh, 0]                  # the q rows of head h
    check("fused: the naive offset reads another head's rows", not np.array_equal(naive, correct),
          "naive rows %s, head %d's q rows %s" % (naive.tolist(), h, correct.tolist()))
    check("fused: the packed offset 3*h*d_head is the q rows of head h", correct[0] == 3 * h * dh,
          "first row %d = 3 * %d * %d" % (correct[0], h, dh))


def selftest():
    del PASS[:]
    for t in (t_slice, t_column, t_hook, t_metric, t_fused):
        t()
    expected = 13
    ok = len(PASS) == expected
    print("\n%d checks passed%s" % (len(PASS), "" if ok else " of %d - a check did not run" % expected))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
