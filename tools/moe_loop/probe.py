"""Measure the design note's claims on this machine, before any of them are asserted.

Four questions, all answerable on a CPU with no training:

1. does the ternary layer really hold three values, and does a gradient still reach the weight?
2. does looping stay bounded, and does the damping (`alpha`) matter for that, or for the output?
3. does 1.58-bit error grow with the number of loops, as the note's BitNet/BitMamba warning says?
4. what does the note's dispatch actually cost against a sparse one, and what width does an expert get in each router?

    python -m tools.moe_loop.probe
"""
import time

import torch

from . import layers


def set_quantise(block, on):
    for m in block.modules():
        if isinstance(m, layers.BitLinear158):
            m.quantise = on


def rel(a, b):
    return float((a - b).norm() / max(1e-9, b.norm()))


def ternary_check():
    torch.manual_seed(0)
    lin = layers.BitLinear158(64, 32)
    w_q, scale = lin.quantised()
    values = torch.unique(w_q).numel()
    zeros = float((w_q == 0).float().mean())
    x = torch.randn(4, 64, requires_grad=True)
    lin(x).sum().backward()
    grad_norm = float(lin.weight.grad.norm())
    print("1. ternary layer")
    print("   distinct values in the quantised weight: %d (scale %.5f)" % (values, float(scale)))
    print("   fraction of weights rounded to zero:     %.1f%%" % (100 * zeros))
    print("   weight grad norm through the STE:        %.3e" % grad_norm)


def loop_check(loops=8, d=32):
    torch.manual_seed(1)
    x = torch.randn(8, d)
    print("\n2. looping and damping (d=%d, 8 tokens, ternary experts)" % d)
    print("   %-8s %s" % ("alpha", "  ".join("L%d" % (i + 1) for i in range(loops))))
    outs = {}
    for alpha in (0.0, 0.25, 0.5, 1.0):
        torch.manual_seed(2)
        block = layers.LoopedMoEBlock(d_model=d, num_experts=8, top_k=2, loops=loops, alpha=alpha)
        block.eval()
        with torch.no_grad():
            out, norms = block(x, collect_norms=True)
        outs[alpha] = (out, norms)
        print("   %-8s %s" % (alpha, "  ".join("%5.2f" % n for n in norms)))
    base = outs[0.0][0]

    def growth(alpha):
        return outs[alpha][1][-1] / max(1e-9, outs[alpha][1][0])

    print("   per-loop growth: alpha 0 -> %.2fx, 0.5 -> %.2fx, 1.0 -> %.2fx"
          % (growth(0.0), growth(0.5), growth(1.0)))
    print("   displacement from the identity (alpha 0): 0.5 -> %.3f, 1.0 -> %.3f"
          % (rel(outs[0.5][0], base), rel(outs[1.0][0], base)))


def drift_check(loops=(1, 2, 4, 8), d=32):
    torch.manual_seed(3)
    x = torch.randn(16, d)
    print("\n3. ternary error against the same weights in float, per loop count")
    print("   %-6s %-12s %-12s" % ("loops", "rel error", "growth"))
    prev = 0.0
    rows = []
    for n in loops:
        torch.manual_seed(4)
        block = layers.LoopedMoEBlock(d_model=d, num_experts=8, top_k=2, loops=n, alpha=0.5)
        block.eval()
        with torch.no_grad():
            set_quantise(block, False)
            ref = block(x)
            set_quantise(block, True)
            got = block(x)
        err = rel(got, ref)
        rows.append((n, err))
        print("   %-6d %-12.3e %-12s" % (n, err, "-" if not prev else "%.2fx" % (err / prev)))
        prev = err
    return rows


def router_check(d=32, tokens=512):
    print("\n4a. what averaging the two heads' logits does to the routing weights")
    torch.manual_seed(6)
    router = layers.AvgHeadRouter(d, 8, top_k=2)
    x = torch.randn(tokens, d)
    x1, x2 = x.chunk(2, dim=-1)
    l1, l2 = router.head1(x1), router.head2(x2)
    p1, p2 = torch.softmax(l1, -1), torch.softmax(l2, -1)
    variants = {
        "softmax((l1+l2)/2) - the note": torch.softmax((l1 + l2) * 0.5, -1),
        "softmax(l1+l2)": torch.softmax(l1 + l2, -1),
        "(p1+p2)/2 - the mixture": (p1 + p2) * 0.5,
    }
    for name, p in variants.items():
        top = p.topk(2, -1).values.sum(-1)
        print("   %-32s mean top-2 mass %.3f   mean entropy %.3f"
              % (name, float(top.mean()), float(-(p * p.clamp(min=1e-12).log()).sum(-1).mean())))
    print("   note vs mixture: mean |difference| in top-2 mass %.3f"
          % float((variants["softmax((l1+l2)/2) - the note"].topk(2, -1).values.sum(-1)
                   - variants["(p1+p2)/2 - the mixture"].topk(2, -1).values.sum(-1)).abs().mean()))

    print("\n4b. routing and dispatch (d=%d, 8 experts, top-2, %d tokens, 3 loops)" % (d, tokens))
    x = torch.randn(tokens, d)
    for kind, dispatch in (("avg", "mask"), ("avg", "index_add"), ("split", "index_add")):
        torch.manual_seed(5)
        block = layers.LoopedMoEBlock(d_model=d, num_experts=8, top_k=2, loops=3, alpha=0.5,
                                      dispatch=dispatch, router=kind, experts_per_group=2)
        block.eval()
        with torch.no_grad():
            block(x)                                      # warm
            block.calls = block.rows = 0
            t0 = time.perf_counter()
            block(x)
            dt = time.perf_counter() - t0
        print("   router %-6s dispatch %-10s forwards %3d  rows %5d  width %2s  %6.1f ms"
              % (kind, dispatch, block.calls, block.rows, block.seen_width, dt * 1e3))


def main():
    torch.set_grad_enabled(True)
    ternary_check()
    loop_check()
    drift_check()
    router_check()


if __name__ == "__main__":
    main()
