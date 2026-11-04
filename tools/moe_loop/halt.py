"""What a per-sequence halt mask saves over a batch-mean halt, on the note's looped block.

The specification's halt rule is `sigmoid(halt_gate(h)).mean() > 0.85` - one flag for the whole batch, checked each
loop. `tools.exodus.router` now ships `halt_mask` instead: one sticky flag per row, cumulative over loops. `LoopedMoEBlock`
applies it by dropping halted rows from the norm, the router and the experts, so this measures the two things that
follow - how much of the block's work the mask removes, and what the batch rule's flag costs the rows it stops.

The halting here is not trained: nothing in this directory trains. Each row is given a *required depth* `r` (the loop
at which its own gate would say it is done) and the gates are built to say exactly that, `+12` before and at `r`, `-12`
after. That makes the saving a function of the halt distribution alone, which is the honest thing to measure - a trained
gate decides *which* rows halt, not whether halting rows individually is worth more than halting them together.

    python -m tools.moe_loop.halt              # three distributions, work and output error per rule
    python -m tools.moe_loop.halt --selftest   # six checks, CPU only

Two arms are compared, plus the no-h-alt baseline:

    none    every row runs every loop
    batch   the old rule: the mean fires, and the whole batch stops before that loop
    mask    the new rule: each row stops before its own `r`

The comparison that matters is **work at the same answer**. The mask is exact - a halted row's state is the state a
shorter loop produces, checked row by row - so its saving costs nothing. The batch rule's saving is not comparable,
because it buys work with the rows that were still running; the report gives that error rather than a ratio, and at
equal batch composition the rule most often saves nothing at all (the mean sits below 0.85 while a third of the batch is
finished).
"""
import sys

import numpy as np
import torch

from ..bench import bench, compare as bench_cmp
from ..exodus.router import halt_mask, halts_batch_mean
from . import layers

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def gates_for(required, loops, seed=0, certainty=12.0, noise=0.5):
    """Gate logits that say `done` from loop `required[b]` on, decisively (sigmoid(12) = 0.99999 against 0.85)."""
    rng = np.random.RandomState(seed)
    required = np.asarray(required)
    g = np.where(np.arange(loops)[None, :] >= required[:, None], certainty, -certainty)
    return g + rng.normal(0.0, noise, size=g.shape)


def masks_for(gates, loops):
    """The three halt masks for the same gates: none, the batch mean, and the per-sequence mask."""
    n = gates.shape[0]
    per = halt_mask(gates).numpy()
    batch = np.zeros((n, loops), dtype=bool)
    stopped = False
    for loop in range(loops):
        stopped = stopped or halts_batch_mean(gates[:, loop])
        batch[:, loop] = stopped
    return {"none": np.zeros((n, loops), dtype=bool), "batch": batch, "mask": per}


def run(block, x, halt):
    """One arm: work counters and the output, with the block's counters reset first."""
    with torch.no_grad():
        block.calls = block.rows = block.active_rows = block.loops_run = 0
        return block(x, halt=None if halt is None else torch.as_tensor(halt)), dict(
            rows=block.rows, calls=block.calls, active=block.active_rows, loops=block.loops_run)


def reference_states(block, x, halt):
    """What each row's state must be: the block run for exactly the number of loops that row performed.

    A row halted at loop `r` computed loops `0..r-1`, so its state must equal a run with `loops=r` over the rows that
    share that `r`. This is the property the mask has to have to be worth anything: halting must remove work and change
    nothing that was computed.
    """
    required = np.where(halt.any(axis=1), halt.argmax(axis=1), halt.shape[1])
    out = torch.zeros_like(x)
    saved = block.loops
    try:
        for r in sorted(set(required.tolist())):
            idx = np.flatnonzero(required == r)
            if r == 0:
                out[idx] = x[idx]                      # never ran a loop: the input is the state
                continue
            block.loops = int(r)
            with torch.no_grad():
                out[idx] = block(x[idx])
    finally:
        block.loops = saved
    return out, required


def rel_l2(got, want, idx=None):
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if idx is not None:
        g, w = g[idx], w[idx]
    return float(np.linalg.norm(g - w) / max(np.linalg.norm(w), 1e-30))


def sweep(block, x, gates, loops):
    """All three arms on one gate set: work, and the output error against the per-row reference."""
    arms = masks_for(gates, loops)
    outs, work = {}, {}
    for name, halt in arms.items():
        out, w = run(block, x, halt)
        outs[name] = out
        work[name] = w
    ref, required = reference_states(block, x, arms["mask"])
    err = {name: rel_l2(outs[name], ref) for name in arms}
    # per row, how many loops each rule actually ran it for. Both directions of error are counted, because a residual
    # loop is not idempotent: stopping a row before its own depth and running it past its own depth are different
    # mistakes, and the batch rule can make either depending on when its mean crosses.
    ran = {name: (~arms[name]).sum(axis=1) for name in arms}
    batch_ran = ran["batch"]
    early = required > batch_ran                       # stopped before it was done
    late = batch_ran > required                        # kept looping after it was done
    return dict(arms=arms, work=work, outs=outs, err=err, required=required, ran=ran,
                cut=int(early.sum()), cut_err=rel_l2(outs["batch"], ref, np.flatnonzero(early)) if early.any() else 0.0,
                waste=int((batch_ran - required)[late].sum()), late=int(late.sum()))


def main(argv):
    loops, d, batch = 8, 32, 64
    torch.manual_seed(1)
    block = layers.LoopedMoEBlock(d_model=d, num_experts=8, top_k=2, loops=loops, alpha=0.5, dispatch="index_add")
    block.eval()
    x = torch.randn(batch, d)

    rng = np.random.RandomState(3)
    cases = (("mixed  (r uniform 1..8)", rng.randint(1, loops + 1, size=batch)),
             ("early  (r = 2 for all)", np.full(batch, 2)),
             ("late   (r = 8 for all)", np.full(batch, loops)))

    print("looped block: d_model %d, 8 experts, top-2, %d loops, batch %d, alpha 0.5\n" % (d, loops, batch))
    print("%-24s %-6s %9s %8s %8s %7s   %s" % ("halt distribution", "rule", "row-loops", "expert rows", "saved",
                                               "waste", "output vs per-row reference"))
    saved = {}
    for label, required in cases:
        gates = gates_for(required, loops)
        s = sweep(block, x, gates, loops)
        base = s["work"]["none"]["active"]
        for name in ("none", "batch", "mask"):
            w = s["work"][name]
            saved[(label, name)] = 1.0 - w["active"] / float(base)
            note = "exact" if s["err"][name] < 1e-5 else "relL2 %.3f" % s["err"][name]
            waste = "%d" % s["waste"] if name == "batch" else "0"
            if name == "batch" and (s["cut"] or s["waste"]):
                note += "  (%d rows cut, %d kept looping after their gate; relL2 %.3f on the cut rows)" \
                        % (s["cut"], s["late"], s["cut_err"])
            print("%-24s %-6s %9d %8d %7.1f%% %7s   %s"
                  % (label if name == "none" else "", name, w["active"], w["rows"],
                     100 * saved[(label, name)], waste, note))
        print()

    # Timing across widths, because the saving is arithmetic and a loop here is also several Python dispatches: if the
    # clock does not move as the arithmetic grows, it is the dispatcher that is being timed, not the work.
    arms = masks_for(gates_for(rng.randint(1, loops + 1, size=batch), loops, seed=5), loops)
    print("wall clock, mixed batch, 9 rounds (two calls each, alternating inside the loop):")
    for width in (d, 128, 512):
        torch.manual_seed(1)
        blk = layers.LoopedMoEBlock(d_model=width, num_experts=8, top_k=2, loops=loops, alpha=0.5,
                                    dispatch="index_add").eval()
        xx = torch.randn(batch, width)
        tm = {}
        for name in ("none", "mask"):
            h = None if name == "none" else torch.as_tensor(arms["mask"])
            tm[name] = bench(lambda hh=h: run(blk, xx, hh), rounds=9, warmup=1)
        c = bench_cmp(tm["mask"], tm["none"], "mask", "none")
        print("  d_model %3d   none %6.1f ms (%2.0f%% spread)   mask %6.1f ms (%2.0f%% spread)   %s"
              % (width, 1e3 * tm["none"]["median"], 100 * tm["none"]["rel"],
                 1e3 * tm["mask"]["median"], 100 * tm["mask"]["rel"], c["text"]))
    print("  The row-loop count is the portable number here. At 46.9% fewer row-loops the clock moves by single digits,"
          " flips")
    print("  direction between runs, and does not improve with width - what is being timed is the dispatch loop, not the")
    print("  experts.")
    print("  The saving becomes time in a dispatcher that is not Python - the same reason kindi's expert paging is")
    print("  callback-bound at ~0.3 ms a fetch (docs/CPU-PERF-2026.md).")
    return 0


def selftest(argv=None):
    del PASS[:]
    loops, d, batch = 6, 16, 12
    torch.manual_seed(2)
    block = layers.LoopedMoEBlock(d_model=d, num_experts=4, top_k=2, loops=loops, alpha=0.5, dispatch="index_add")
    block.eval()
    x = torch.randn(batch, d)

    required = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6])
    gates = gates_for(required, loops, seed=1)
    s = sweep(block, x, gates, loops)

    check("halt: the mask's work is exactly the rows' required depths",
          s["work"]["mask"]["active"] == int(required.sum()),
          "%d row-loops for %s against %d for no halting" % (int(required.sum()), required.tolist(),
                                                             s["work"]["none"]["active"]))
    check("halt: halting changes nothing that was computed", s["err"]["mask"] < 1e-5,
          "relL2 %.1e against a run with loops = each row's depth" % s["err"]["mask"])
    check("halt: the batch rule spends loops on rows that were already done", s["waste"] > 0,
          "%d row-loops past %d rows' own depth; the mask wastes 0" % (s["waste"], s["late"]))
    check("halt: so the batch rule's answer is not the one the gates asked for", s["err"]["batch"] > 1e-5,
          "relL2 %.3f against the per-row reference, where the mask is %.1e" % (s["err"]["batch"], s["err"]["mask"]))

    late = np.full(batch, loops)
    gates_late = gates_for(late, loops, seed=7)
    s_late = sweep(block, x, gates_late, loops)
    check("halt: a batch that needs every loop saves nothing, under either rule",
          s_late["work"]["mask"]["active"] == batch * loops and s_late["work"]["batch"]["active"] == batch * loops,
          "mask %d, batch %d, none %d row-loops"
          % (s_late["work"]["mask"]["active"], s_late["work"]["batch"]["active"], s_late["work"]["none"]["active"]))

    early = np.full(batch, 2)
    gates_early = gates_for(early, loops, seed=9)
    s_early = sweep(block, x, gates_early, loops)
    check("halt: a batch that is uniformly done saves the same under both rules",
          s_early["work"]["mask"]["active"] == s_early["work"]["batch"]["active"] == batch * 2
          and s_early["err"]["mask"] < 1e-5 and s_early["err"]["batch"] < 1e-5,
          "mask %d, batch %d, none %d row-loops; both exact"
          % (s_early["work"]["mask"]["active"], s_early["work"]["batch"]["active"],
             s_early["work"]["none"]["active"]))
    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == 6 else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv[1:] else main(sys.argv[1:]))
