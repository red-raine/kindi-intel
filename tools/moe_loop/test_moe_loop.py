"""Assertions over the measured behaviour of the note's block, plus the corrected router and dispatch.

Every threshold here is derived from a measurement in `probe.py`, with slack for a different seed or machine; none of
them asserts a number that was not first observed.

    python -m tools.moe_loop.test_moe_loop
    python -m tools.moe_loop.probe            # the same quantities, printed rather than asserted
"""
import sys
import time

import torch

from ..bench import bench, compare as bench_cmp
from . import layers

PASS = []


def check(name, fn):
    fn()
    PASS.append(name)
    print("ok   %s" % name)


def _block(d=32, loops=8, alpha=0.5, **kw):
    torch.manual_seed(2)
    b = layers.LoopedMoEBlock(d_model=d, num_experts=8, top_k=2, loops=loops, alpha=alpha, **kw)
    return b.eval()


def _set_quantise(block, on):
    for m in block.modules():
        if isinstance(m, layers.BitLinear158):
            m.quantise = on


def test_ternary():
    torch.manual_seed(0)
    lin = layers.BitLinear158(64, 32)
    w, scale = lin.quantised()
    assert torch.unique(w).numel() == 3, "the ternary weight is not three values: %r" % torch.unique(w)
    assert float((w == 0).float().mean()) > 0.05, "absmean rounding produced almost no zeros"
    assert torch.allclose(w / scale, (w / scale).round()), "the quantised weight is not a multiple of one scale"
    lin(torch.randn(4, 64)).sum().backward()
    assert lin.weight.grad is not None and float(lin.weight.grad.abs().sum()) > 0, "no gradient through the STE"


def test_loop_growth():
    x = torch.randn(8, 32)
    growth = {}
    for alpha in (0.25, 0.5, 1.0):
        b = _block(alpha=alpha)
        with torch.no_grad():
            _out, norms = b(x, collect_norms=True)
        growth[alpha] = norms[-1] / norms[0]
    assert growth[0.25] < growth[0.5] < growth[1.0], growth
    assert growth[1.0] > 1.05, "undamped looping did not grow the state at all: %.3f" % growth[1.0]


def test_drift_accumulates():
    x = torch.randn(16, 32)
    err = {}
    for n in (1, 2, 4, 8):
        b = _block(loops=n)
        with torch.no_grad():
            _set_quantise(b, False)
            ref = b(x)
            _set_quantise(b, True)
            err[n] = float((b(x) - ref).norm() / ref.norm())
    order = [err[1], err[2], err[4], err[8]]
    assert order == sorted(order) and len(set(order)) == 4, "ternary error did not grow with loops: %r" % err
    ratio = err[8] / err[1]
    assert 3.0 < ratio < 12.0, "error grew %.1fx over 8 loops, outside the ~linear band measured" % ratio


def test_split_router_sees_a_subvector():
    x = torch.randn(8, 32)
    with torch.no_grad():
        split = _block(router="split", experts_per_group=2)
        split(x)
        avg = _block(router="avg")
        avg(x)
    assert split.seen_width == 16, "a sub-vector expert got width %s, not half" % split.seen_width
    assert avg.seen_width == 32, "the averaged router gave an expert the whole state (%s)" % avg.seen_width
    assert split.calls < avg.calls, "splitting did not reduce expert forwards: %d vs %d" % (split.calls, avg.calls)


def test_dispatch_work():
    """The two dispatches must agree on the answer and differ by 8x in arithmetic - the timing is printed, not asserted.

    It used to assert `index_add` was faster, and that assertion failed at random: one call of this block is ~50-70 ms,
    and on a shared machine the two arms swap places between runs (measured 1.6x faster in a quiet process, and slower
    inside the gate, with nothing changed in between). A wall-clock ordering is not a property of this code on this host,
    so the gate does not decide one; the row count and the output are properties, and they do get asserted.
    """
    x = torch.randn(512, 32)
    rows, out, times = {}, {}, {}
    for dispatch in ("mask", "index_add"):
        b = _block(loops=3, dispatch=dispatch)      # same seed, so both arms carry the same weights
        with torch.no_grad():
            y = b(x)                                # warm-up, also allocates the cached tensors
            b.calls = b.rows = 0
            times[dispatch] = bench(lambda: b(x), rounds=7, warmup=1)
            out[dispatch] = b(x)
        rows[dispatch] = b.rows
    assert rows["mask"] == 8 * rows["index_add"], \
        "the mask dispatch should do 8x the rows for 8 experts: %r" % rows
    assert torch.allclose(out["mask"], out["index_add"], atol=1e-5, rtol=1e-4), \
        "the two dispatches disagree: max |diff| %.2e" % float((out["mask"] - out["index_add"]).abs().max())
    c = bench_cmp(times["mask"], times["index_add"], "mask", "index_add")
    print("     measured, not asserted: mask %.1f ms (%.0f%% spread), index_add %.1f ms (%.0f%% spread) -> %s"
          % (1e3 * times["mask"]["median"], 100 * times["mask"]["rel"],
             1e3 * times["index_add"]["median"], 100 * times["index_add"]["rel"], c["text"]))


def test_logit_average_flattens():
    torch.manual_seed(6)
    d = 32
    r = layers.AvgHeadRouter(d, 8, top_k=2)
    x = torch.randn(512, d)
    x1, x2 = x.chunk(2, dim=-1)
    l1, l2 = r.head1(x1), r.head2(x2)
    halved = torch.softmax((l1 + l2) * 0.5, -1).topk(2, -1).values.sum(-1).mean()
    summed = torch.softmax(l1 + l2, -1).topk(2, -1).values.sum(-1).mean()
    assert halved < summed - 0.05, \
        "averaging the logits did not flatten the routing weights: %.3f vs %.3f" % (halved, summed)


def test_halt_mask_ends_rows_at_their_own_depth():
    """A per-sequence mask must stop each row after its own loop count, and leave what it computed alone."""
    b = _block(d=16, loops=6)
    torch.manual_seed(3)
    x = torch.randn(8, 16)
    depth = [1, 1, 2, 2, 3, 3, 6, 6]                       # 6 is "needs every loop", so never halted
    halt = torch.zeros(8, 6, dtype=torch.bool)
    for i, r in enumerate(depth):
        halt[i, r:] = True
    with torch.no_grad():
        y = b(x, halt=halt)
        active = b.active_rows
        ref = torch.zeros_like(x)                           # what each row must equal: loops = its own depth
        saved = b.loops
        for r in sorted(set(depth)):
            idx = [i for i, d in enumerate(depth) if d == r]
            b.loops = r
            ref[idx] = b(x[idx])
        b.loops = saved
    assert active == sum(depth), \
        "the mask did not skip the halted rows' remaining loops: %d row-loops, expected %d" % (active, sum(depth))
    err = float((y - ref).abs().max())
    assert err < 1e-5, "a halted row is not the state a shorter loop gives: max |diff| %.2e" % err
    print("     measured, not asserted: %d row-loops against %d unhalted (%.0f%% less work)"
          % (active, 8 * b.loops, 100 * (1 - active / float(8 * b.loops))))


def test_no_mask_is_bit_identical():
    """`halt=None` and an all-False mask are the same computation, so nothing above changed the default path."""
    b1, b2 = _block(d=16, loops=4), _block(d=16, loops=4)
    torch.manual_seed(4)
    x = torch.randn(5, 16)
    with torch.no_grad():
        a = b1(x)
        c = b2(x, halt=torch.zeros(5, 4, dtype=torch.bool))
    assert torch.equal(a, c) and b1.active_rows == b2.active_rows == 20, \
        "an empty mask changed the output or the work: max |diff| %.2e, rows %d vs %d" \
        % (float((a - c).abs().max()), b1.active_rows, b2.active_rows)


def main():
    check("ternary weight is three values, with a gradient", test_ternary)
    check("damping reduces per-loop growth, RMSNorm alone does not", test_loop_growth)
    check("ternary error grows with the loop count", test_drift_accumulates)
    check("the split router gives an expert half the state", test_split_router_sees_a_subvector)
    check("the mask dispatch does 8x the expert arithmetic", test_dispatch_work)
    check("averaging logits flattens the top-k weights", test_logit_average_flattens)
    check("a per-sequence halt mask ends each row at its own depth", test_halt_mask_ends_rows_at_their_own_depth)
    check("no mask and an empty mask are the same computation", test_no_mask_is_bit_identical)
    print("\n%d checks passed" % len(PASS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
