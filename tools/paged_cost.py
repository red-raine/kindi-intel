"""tools/paged_cost.py — split the paged-expert cost on a real-weight kindi MoE pack.

Loads the pack with expert_resident < n_expert, enables paging, then times the three components that make up
one cache miss on the *real* fixture (no synthetic weights):

  read     — one positional whole-staged-expert read via Engine._pread_pc
  memcpy   — copy the three staged matrices into the resident slot arrays
  callback — fetch-hook overhead when the expert is already in a slot (the no-op miss path)

End-to-end paged decode on the same pack is printed for scale (tokens, faults/token is measured in the
caller shell with psapi; this script does not duplicate that counter).

Usage:
  python tools/paged_cost.py B:/models/scratch/qwen-layer0-pack [--cache 8] [--reps 20]
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from kindi.cpu import Engine, _pair_len, _pair_views


def median_ms(fn, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", help="kindi MoE pack directory with pcodes.bin")
    ap.add_argument("--cache", type=int, default=8, help="expert_resident cache slots (default 8)")
    ap.add_argument("--reps", type=int, default=20, help="timing repetitions per component (default 20)")
    a = ap.parse_args()

    eng = Engine(a.pack, ctx=64, log=lambda *x: None, expert_resident=a.cache)
    eng.enable_paging()
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    eng.generate(prompt, 2)  # warm the cache, as a server would

    # --- miss/token and end-to-end scale -------------------------------------
    calls = [0]

    def counted(big, layer, expert):
        ins = eng.pager["in_slot"].get(layer)
        if ins is not None and expert not in ins:
            calls[0] += 1
        eng._page_expert(eng._paging_src, layer, expert)

    eng.set_fetcher(counted)
    t0 = time.time()
    out = eng.generate(prompt, 40)
    dt = time.time() - t0
    miss_per_token = calls[0] / 40.0

    # --- read cost: one positional read of a whole staged expert -------------
    recs4 = eng.pager["src"][(0, 4)]
    lens = [_pair_len(r) for r in recs4]
    offs = [int(r["pc_off"]) for r in recs4]
    total = sum(lens)
    read_ms = median_ms(lambda: eng._pread_pc(total, offs[0]), a.reps)

    # --- memcpy cost: pair views from the pre-read buffer into the slots ----
    data = eng._pread_pc(total, offs[0])
    slots = eng.pager["slots"][0][0]

    def do_memcpy():
        for j, rec in enumerate(recs4):
            dw, ds = slots[j]
            W, S = _pair_views(data, dict(rec, pc_off=int(rec["pc_off"]) - offs[0]))
            np.copyto(dw, W.reshape(dw.shape))
            np.copyto(ds, S.reshape(ds.shape))

    memcpy_ms = median_ms(do_memcpy, a.reps)

    # --- callback cost: the no-op path (expert already in a slot) -----------
    # The first slot of layer 0 holds some expert after the warmup; call _page_expert on it.
    resident_expert = next(e for e in eng.pager["in_slot"][0] if e >= 0)
    cb_ms = median_ms(lambda: eng._page_expert(eng._paging_src, 0, resident_expert), max(a.reps, 100))

    print(json.dumps({
        "pack": a.pack,
        "cache": a.cache,
        "staged_expert_bytes": total,
        "matrix_bytes": lens,
        "read_ms_per_expert": round(read_ms, 3),
        "memcpy_ms_per_expert": round(memcpy_ms, 3),
        "callback_ms_per_call": round(cb_ms, 6),
        "e2e_tok_s": round(40.0 / dt, 2),
        "miss_per_token": round(miss_per_token, 2),
        "miss_cost_read_plus_memcpy_ms": round(read_ms + memcpy_ms, 3),
        "miss_count": calls[0],
        "tokens_first5": out[:5],
    }, indent=2))
    del eng
    gc.collect()


if __name__ == "__main__":
    main()
