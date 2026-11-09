"""tools/bitnet_qat/smoke.py — measure training throughput of the native ternary harness and extrapolate budgets.

Runs a 50M-class MiniMinion on this machine's CPU (no XPU/GPU here) for a few dozen steps, measures
tokens/sec, and prints how long the published token budgets would take at that rate. This is the
feasibility gate for the from-scratch plan: it is why appendix.md rejects a 0.5B run locally.

  python -u tools/bitnet_qat/smoke.py [--steps 20]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from tools.bitnet_qat.model import MiniMinion, adamw_from_model

BUDGETS = [
    ("small-SLM parity budget (BitNet b1.58 Reloaded, 6–48M models)", 1.35e9),
    ("hobbyist 0.15B RWKV run, documented", 9.9e9),
    ("Microsoft 2B native ternary pre-train", 4e12),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(0)
    m = MiniMinion(vocab=32000, d=512, layers=8, heads=8, hidden=1408)
    n = m.num_parameters()
    opt = adamw_from_model(m)
    x = torch.randint(0, 32000, (a.batch, a.seq))
    y = torch.randint(0, 32000, (a.batch, a.seq))

    losses = []
    t0 = time.time()
    for _ in range(a.steps):
        opt.zero_grad()
        out = m(x)
        loss = torch.nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), y[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss))
    dt = time.time() - t0
    tps = a.steps * a.batch * a.seq / dt
    print(f"params: {n/1e6:.1f}M  device: cpu  tok/s: {tps:.1f}")
    print(f"loss first/last: {losses[0]:.3f} -> {losses[-1]:.3f}")
    for name, tokens in BUDGETS:
        print(f"  {name}: {tokens:.1e} tokens -> {tokens/tps/3.15e7:.2f} years at this rate")


if __name__ == "__main__":
    main()

