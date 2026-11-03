"""Time a callable the way TorchBench does, and report the noise that makes a ratio believable.

A single `perf_counter` sample of a 50 ms call decided a gate twice in this repository - once in each direction - because
the machine's own spread is about 6% on a fixed workload. TorchBench's answer is to warm up, repeat, and keep statistics
rather than a sample, and to say what the machine is like before trusting a comparison:

    python -m tools.bench            # the noise floor of this host, measured
    python -m tools.bench --selftest # checks with known answers

`bench` returns min/median/mean/stdev over R rounds after W warm-up calls; `compare` turns two of those into a ratio and
says whether the ratio clears the noise it measured, instead of printing a number that may be one scheduler hiccup.

The floor is a calibration, not a promise: it is measured on this host in this process, and the machine has other work on
it. A ratio within the measured noise is reported as noise.
"""
import sys
import time

import numpy as np

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def bench(fn, rounds=7, warmup=1):
    """Warm up, then time `rounds` calls. Returns seconds: min, median, mean, stdev, and their relative size."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts, dtype=np.float64)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return dict(min=float(a.min()), max=float(a.max()), median=float(np.median(a)), mean=mean, stdev=sd,
                rel=sd / mean if mean > 0 else 0.0, rounds=rounds, samples=[float(t) for t in a])


def compare(a, b, name_a="a", name_b="b"):
    """Ratio of medians, and whether it clears the noise of either arm.

    `faster` is the arm with the smaller median; `clear` is False when the two medians are closer than the larger of the
    arms' relative spreads, which is the case in which the honest answer is "not distinguishable on this host".
    """
    ra, rb = a["median"], b["median"]
    ratio = ra / rb if rb > 0 else float("inf")
    noise = max(a["rel"], b["rel"])
    clear = abs(ratio - 1.0) > noise
    faster = name_a if ra < rb else name_b
    return dict(ratio=ratio, faster=faster, clear=bool(clear), noise=float(noise),
                text="%s/%s = %.2fx (%s%s)" % (name_a, name_b, ratio, faster,
                                               " faster" if clear else " - within the %.0f%% noise" % (100 * noise)))


def noise_floor(rounds=9):
    """A fixed, dependency-free workload, so the floor is reproducible on any host."""
    def work():
        a = np.zeros(64, np.float64)
        for _ in range(2000):
            a = a * 1.0001 + 0.5
    r = bench(work, rounds=rounds, warmup=1)
    return r


def selftest():
    del PASS[:]
    r = noise_floor(rounds=5)
    # min <= median <= max holds for any samples; median <= mean does not (a fast lower tail puts the mean below it), and
    # asserting that one failed on a run of this file.
    check("bench: rounds are all kept and min <= median <= max",
          len(r["samples"]) == 5 and r["min"] <= r["median"] <= r["max"], "min %.3g, median %.3g, max %.3g"
          % (r["min"], r["median"], r["max"]))
    slow = bench(lambda: sum(range(200000)), rounds=3)
    fast = bench(lambda: sum(range(20000)), rounds=3)
    c = compare(slow, fast, "slow", "fast")
    check("bench: a ten-times workload is detected as slower and clears the noise", c["clear"] and c["faster"] == "fast",
          c["text"])
    # the decision logic is checked on constructed statistics, not on two noisy runs: timing two equal workloads and hoping
    # they land within 20% is a flaky test of the machine, and it did fail on one run of this file.
    quiet = compare(dict(median=1.00, rel=0.05), dict(median=1.03, rel=0.05), "a", "b")
    check("bench: a difference smaller than the noise is reported as noise", not quiet["clear"], quiet["text"])
    loud = compare(dict(median=1.00, rel=0.05), dict(median=2.00, rel=0.05), "a", "b")
    check("bench: a 2x difference clears the noise", loud["clear"] and loud["faster"] == "a", loud["text"])
    try:
        bench(lambda: None, rounds=0)
        ok = False
    except ValueError:
        ok = True
    check("bench: zero rounds is a refusal, not a zero", ok)
    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == 5 else 1


def main(argv):
    r = noise_floor()
    print("noise floor of this host: a fixed numpy workload, 9 rounds\n")
    print("  median %.3f ms   min %.3f   max %.3f   rel-stdev %.1f%%"
          % (1e3 * r["median"], 1e3 * r["min"], 1e3 * r["max"], 100 * r["rel"]))
    print("\n  a ratio smaller than %.0f%% is not distinguishable here; report it as noise." % (100 * r["rel"]))
    print("  (%d samples: %s)" % (r["rounds"], " ".join("%.2f" % (1e3 * t) for t in r["samples"])))
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv[1:] else main(sys.argv[1:]))
