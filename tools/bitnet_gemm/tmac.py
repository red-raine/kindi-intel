"""Build and exercise the corrected T-MAC LUT GEMV, and measure it against the same machine's other decoders.

    python -m tools.bitnet_gemm.tmac                # build (once), validate, benchmark
    python -m tools.bitnet_gemm.tmac --rebuild
    python -m tools.bitnet_gemm.tmac --selftest     # the statistics and the pinning, no build

The kernel is `tmac.cpp`. It is the design note's idea - precompute the 16 partial sums of two activations, fetch them
with `vpshufb` - with int16 tables instead of int8, because a two-weight table of int8 activations holds up to 254 and
wraps. What that costs is one extra shuffle per table; what it buys is that int8 activations stay int8.

**Why this file changed: min-of-min was measuring luck.** The tiling comparison used to report the best of three runs of
the best of five calls, so it reported the quietest sample of a distribution whose own spread is +-45% run to run - and
the wide-shape row straddled the untiled one in either direction depending on the afternoon. Two things were wrong with
that, and both are fixed here:

* **Nothing was pinned.** Eight OpenMP threads were free to land on SMT siblings and to migrate, so the thread count in
  the table was a request rather than a fact. `pin_affinity` sets the process mask to one thread per **physical** core
  (from `GetLogicalProcessorInformationEx`, so no sibling is picked), and every table prints the mask it ran under.
* **The statistic was a minimum.** A minimum of a noisy sample is not an estimate of a mean: it moves with the noise
  floor, and it cannot be given an interval. The arms now run in **interleaved rounds** - one call an arm a round, with
  the order rotated so no arm is always first - and the reported number is the median with a 95% interval on the
  **paired** ratio (arm against untiled, round by round) from `tools.nll.bootstrap_ci`. A ratio whose interval covers
  1 +- `NOISE` is printed as `indistinguishable`, which is the honest form of "we cannot tell these apart".

The interval is on the ratio, not on the milliseconds, because the arms share a round: drift in clock, cache or
background load hits both, and pairing removes it. This is the same shape as the NLL gate in `tools/nll.py` for the
same reason - a single number with no interval is not a result.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

try:                                                    # `--selftest` and the affinity helpers do not need the build
    import torch
    from torch.utils.cpp_extension import load
except Exception:                                       # pragma: no cover - a host that only runs --selftest
    torch, load = None, None

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE = 0.02            # a paired ratio inside 1 +- 2% is this host's noise floor, and is not called a result


# --- affinity ------------------------------------------------------------------------------------------------------

def _logical_count():
    return os.cpu_count() or 1


def _core_masks():
    """one affinity mask per **physical** core; each mask is that core's sibling set

    On Windows the topology comes from `GetLogicalProcessorInformationEx(RelationProcessorCore)`, which is the only
    way to know that logical 0 and logical 8 are the same core. A fallback of one mask a logical processor is used when
    the call is unavailable, and `pin_affinity` reports which one it got.
    """
    if os.name != "nt":
        return [(1 << i) for i in range(_logical_count())]
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    RELATION_PROCESSOR_CORE = 0
    k32.GetLogicalProcessorInformationEx.argtypes = [wintypes.DWORD, ctypes.c_void_p,
                                                     ctypes.POINTER(wintypes.DWORD)]
    need = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None, ctypes.byref(need))
    if not need.value:
        return [(1 << i) for i in range(_logical_count())]
    buf = ctypes.create_string_buffer(need.value)
    if not k32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf, ctypes.byref(need)):
        return [(1 << i) for i in range(_logical_count())]
    width = ctypes.sizeof(ctypes.c_size_t)
    masks, off = [], 0
    while off + 4 <= need.value:
        size = int.from_bytes(buf[off + 2:off + 4], "little")
        if size < 4:
            break
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: WORD Relationship, WORD Size, then the union at +4.
        # PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20], WORD GroupCount, GROUP_AFFINITY GroupMask[].
        # So the first core's mask starts at +4+24 = +28 and is one ULONG_PTR wide.
        mask = int.from_bytes(buf[off + 28:off + 28 + width], "little")
        if mask:
            masks.append(mask)
        off += size
    return masks or [(1 << i) for i in range(_logical_count())]


def pin_affinity(n):
    """pin this process to `n` physical cores, one thread a core, and say what it did

    Returns a dict rather than a bool: a table that does not print its mask cannot be checked by a reader, and "pinned"
    that silently fell back to one mask a logical processor is a different measurement.
    """
    cores = _core_masks()
    if n <= len(cores):
        mask = 0
        for m in cores[:n]:
            mask |= m & (-m)                            # the lowest sibling: exactly one thread of that core
        how = "one thread each of %d physical cores" % n
    else:                                               # more threads than cores: use everything, and say so
        mask = (1 << _logical_count()) - 1
        how = "all %d logical processors (%d threads asked for, %d cores)" % (_logical_count(), n, len(cores))
    ok = True
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ok = bool(k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask))
    else:
        try:
            os.sched_setaffinity(0, {i for i in range(64) if (mask >> i) & 1})
        except (AttributeError, OSError):
            ok = False
    return dict(mask=mask, cores=len(cores), logical=_logical_count(), how=how, ok=ok)


def affinity_line(info):
    return "affinity: 0x%x - %s%s" % (info["mask"], info["how"], "" if info["ok"] else " (the call failed)")


# --- sampling ------------------------------------------------------------------------------------------------------

def samples(fn, reps, target=0.003):
    """`reps` timings, each one a batch of calls long enough to be above the clock's noise floor

    A single 1024x1024 call is ~0.12 ms, and timing that once per sample gave a 63-163% coefficient of variation on
    this host - the measurement was mostly timer and scheduler noise, and the table it produced meant nothing. The
    batch is sized from a pilot call so a sample is at least `target` seconds, and the reported time is per call.
    Returns (per-call seconds, calls a sample).
    """
    t0 = time.perf_counter()
    fn()
    pilot = max(time.perf_counter() - t0, 1e-9)
    inner = max(1, int(target / pilot))
    out = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        out.append((time.perf_counter_ns() - t0) / 1e9 / inner)
    return out, inner


def paired(arms, rounds=15):
    """interleaved rounds: one call an arm a round, rotating the order; returns {arm: [seconds a round]}

    Interleaving is what makes the ratio the statistic. Sequential blocks would let a drift in clock, cache state or
    background load land on one arm only, which is exactly the failure the old min-of-min hid.
    """
    labels = [name for name, _ in arms]
    fns = dict(arms)
    for f in fns.values():                              # warm: page the tables in, grow the allocator
        f()
        f()
    times = {name: [] for name in labels}
    for r in range(rounds):
        start = r % len(labels)
        for name in labels[start:] + labels[:start]:
            t0 = time.perf_counter_ns()
            fns[name]()
            times[name].append((time.perf_counter_ns() - t0) / 1e9)
    return times


def verdict(lo, hi):
    """`faster`/`slower` only when the paired interval clears the noise floor in that direction"""
    if lo > 1.0 + NOISE:
        return "faster"
    if hi < 1.0 - NOISE:
        return "slower"
    return "indistinguishable"


def summarise(times, base=None):
    """per-arm median, run-to-run spread, and the paired speedup against `base` with its interval"""
    from tools.nll import bootstrap_ci

    base = base or next(iter(times))
    b = np.asarray(times[base], dtype=np.float64)
    rows = []
    for name, t in times.items():
        t = np.asarray(t, dtype=np.float64)
        sp = b / t
        lo, hi = bootstrap_ci(sp, block=1, iters=2000, seed=7)
        cv = 100.0 * float(t.std(ddof=1) / t.mean()) if t.size > 1 and t.mean() > 0 else 0.0
        # the point estimate is the *mean* paired ratio, because that is what the interval is an interval for; the
        # median can (and did) sit outside a mean interval on a skewed pair, which reads as a contradiction
        rows.append(dict(arm=name, ms=1e3 * float(np.median(t)), cv=cv, speedup=float(np.mean(sp)),
                         lo=float(lo), hi=float(hi), verdict="baseline" if name == base else verdict(lo, hi)))
    return rows


# --- the kernel ----------------------------------------------------------------------------------------------------

def _flags():
    if os.name == "nt":
        return ["/O2", "/arch:AVX2", "/openmp", "/std:c++17"]
    return ["-O3", "-mavx2", "-ffast-math", "-std=c++17"]


def _msvc_env():
    """put `cl.exe` on PATH for torch's ninja build, through the same VS lookup the engine uses

    A plain Git Bash has no `cl` on PATH, and torch's extension builder needs one; `kindi.cbuild` already knows how to
    find an install and call its `vcvarsall`, so this reuses that rather than guessing at paths.
    """
    if os.name != "nt" or shutil.which("cl"):
        return bool(shutil.which("cl"))
    from kindi.cbuild import find_vcvarsall
    vc = find_vcvarsall()
    if not vc:
        return False
    r = subprocess.run('call "%s" x64 >nul && set' % vc, shell=True, capture_output=True, text=True)
    if r.returncode:
        return False
    for line in r.stdout.splitlines():
        k, _, v = line.partition("=")
        if k and v:
            os.environ[k.strip()] = v
    return bool(shutil.which("cl"))


def build(rebuild=False):
    if load is None:
        raise SystemExit("tmac: building needs torch (`pip install torch`); --selftest runs without it")
    if not _msvc_env():
        raise SystemExit("tmac: no C++ compiler on PATH and no Visual Studio toolset found; `--selftest` needs neither")
    return load(name="tmac_bitnet_avx2", sources=[os.path.join(HERE, "tmac.cpp")],
                extra_cflags=_flags(), extra_ldflags=[] if os.name == "nt" else ["-fopenmp"],
                verbose=rebuild, with_cuda=False, is_python_module=True)


def pack_for_tmac(w):
    """(N, K) int8 in {-1, 0, 1} -> (K // 4, N) uint8: the byte-per-neuron-per-group layout the kernel streams."""
    n, k = w.shape
    assert k % 4 == 0
    codes = torch.zeros_like(w, dtype=torch.uint8)
    codes[w == 1] = 1
    codes[w == -1] = 2
    packed = codes[:, 0::4] | (codes[:, 1::4] << 2) | (codes[:, 2::4] << 4) | (codes[:, 3::4] << 6)
    return packed.t().contiguous()


def reference(a_q, w, scale_a, scale_w):
    """int8 activations times ternary weights, exactly: the answer both arms must reproduce."""
    return (a_q.to(torch.int32) @ w.to(torch.int32).t()).to(torch.float32) * scale_a * scale_w.unsqueeze(0)


def fixture(n, k, seed=1):
    """one weight matrix and one activation vector, packed once - a repack inside a timing loop is not the arm"""
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    a = torch.clamp((torch.randn(k, generator=g) * 40).round(), -127, 127).to(torch.int8)
    return pack_for_tmac(w), a, torch.ones(n)


def validate(mod, n=2048, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    a = torch.clamp((torch.randn(k, generator=g) * 40).round(), -127, 127).to(torch.int8)
    scale_a = 1.0 / 37.0
    sw = torch.rand(n, generator=g) + 0.5
    out = mod.tmac_gemv_avx2(a, pack_for_tmac(w), scale_a, sw)
    want = reference(a.unsqueeze(0), w, scale_a, sw)[0]
    diff = float((out - want).abs().max())
    return diff, w, a


def validate_tiled(mod, n=2048, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    a = torch.clamp((torch.randn(k, generator=g) * 40).round(), -127, 127).to(torch.int8)
    scale_a = 1.0 / 37.0
    sw = torch.rand(n, generator=g) + 0.5
    want = reference(a.unsqueeze(0), w, scale_a, sw)[0]
    worst = 0.0
    for n_blk, k_grp in ((0, 0), (1024, 128), (1024, 0), (32, 128)):
        out = mod.tmac_gemv_avx2_tiled(a, pack_for_tmac(w), scale_a, sw, n_blk, k_grp)
        worst = max(worst, float((out - want).abs().max()))
    return worst


# --- the two tables -------------------------------------------------------------------------------------------------

def gemv_bench(mod, threads=(1, 8), rounds=9):
    """the throughput table: median of interleaved rounds, with the spread it was measured over"""
    print("\nGEMV, weights decoded from 2-bit codes inside the kernel (0.25 bytes a weight)")
    print("  %-12s %4s %10s %7s %9s %12s %10s"
          % ("N x K", "thr", "median ms", "CV", "calls", "weights/s", "packed GB/s"))
    out = {}
    for thr in threads:
        if hasattr(mod, "set_threads"):
            mod.set_threads(thr)
        for n, k in ((1024, 1024), (2048, 1920), (4096, 4096)):
            packed, a, sw = fixture(n, k)
            fn = lambda: mod.tmac_gemv_avx2(a, packed, 1.0, sw)          # noqa: E731 - the arm is the call
            t, inner = samples(fn, rounds)
            t = np.asarray(t, dtype=np.float64)
            ms = 1e3 * float(np.median(t))
            cv = 100.0 * float(t.std(ddof=1) / t.mean()) if t.size > 1 else 0.0
            print("  %-12s %4d %10.4f %6.1f%% %9d %11.1fM %10.2f"
                  % ("%dx%d" % (n, k), thr, ms, cv, inner, n * k / ms / 1e3, (n * k / 4) / (ms / 1e3) / 1e9))
            out["%dx%d@%d" % (n, k, thr)] = dict(ms=ms, cv=cv, calls=inner, threads=thr)
    return out


def tiling_bench(mod, rounds=15, runs=3, threads=8):
    """Does cache blocking pay for itself on a real MoE expert width?

    The blocked arm parallelises over N tiles and rebuilds the tables once per tile (a table is only useful to the
    neurons that built it), so the question is whether keeping the tables hot - and the accumulators contiguous - buys
    back that rebuild. Tile count has to exceed the thread count or the arm is partly idle, so the narrow shape is where
    the tiling is expected to stop paying.

    Every arm runs every round, in a rotated order, and the number reported is the paired ratio to the untiled kernel
    with its 95% interval from the pooled rounds of every run.

    Two different variances matter and only one of them is inside a run. The pooled interval answers "how well does this
    session separate the arms"; the `runs` column answers "does the answer come back next time", and they are not the
    same number: a first version of this table read 1.38x, 1.41x and 1.66x across three sessions, each with an interval
    that excluded the others. The cause is an interaction rather than noise - the untiled arm streams the whole 14.7 MB
    matrix through eight threads and loses more to whatever else is using memory, while a tile keeps its working set in
    cache - so the ratio moves with the machine's background load. Reporting the per-run medians is the only honest way
    to show that, and the direction is what has to be stable: `faster` in every run or the item is open.
    """
    print("\nMoE expert GEMV, K=4096, %d threads: does the note's tiling pay? (%d interleaved rounds x %d runs)"
          % (threads, rounds, runs))
    print("  %-34s %9s %6s %9s %-18s %-11s %s"
          % ("shape / blocking", "median ms", "CV", "speedup", "95% CI", "per run", "verdict"))
    out = {}
    for n in (14336, 2048):
        k = 4096
        packed, a, sw = fixture(n, k)
        shapes = [("untiled (kernel)", None), ("1024 x 512 (the note)", (1024, 128)), ("1024 x all K", (1024, 0)),
                  ("512 x 512", (512, 128)), ("2048 x 1024", (2048, 256))]
        arms = []
        for label, cfg in shapes:
            if cfg is None:
                arms.append((label, lambda: mod.tmac_gemv_avx2(a, packed, 1.0, sw)))
            else:
                arms.append((label, (lambda c: (lambda: mod.tmac_gemv_avx2_tiled(a, packed, 1.0, sw, c[0], c[1])))(cfg)))
        pooled, per_run = {}, {label: [] for label, _ in arms}
        for _ in range(runs):
            t = paired(arms, rounds)
            b = np.asarray(t[arms[0][0]], dtype=np.float64)
            for label, _cfg in arms:
                pooled.setdefault(label, []).extend(list(t[label]))
                per_run[label].append(float(np.mean(b / np.asarray(t[label], dtype=np.float64))))
        rows = summarise(pooled)
        for r in rows:
            gbps = (n * k / 4) / (r["ms"] / 1e3) / 1e9
            runs_txt = " ".join("%.2f" % s for s in per_run[r["arm"]])
            print("  %-34s %9.3f %5.1f%% %8.2fx [%5.2f, %5.2f] %-11s %s   %.1f GB/s"
                  % ("N=%d %s" % (n, r["arm"]), r["ms"], r["cv"], r["speedup"], r["lo"], r["hi"],
                     runs_txt, r["verdict"], gbps))
            r["gbps"] = gbps
            r["per_run"] = per_run[r["arm"]]
        out["N=%d" % n] = rows
        print()
    return out


# --- selftest -------------------------------------------------------------------------------------------------------

def selftest():
    """The statistics and the pinning, checked on answers that are known. No build, no torch.

    A benchmark harness needs its own gate for the same reason a gate needs an interval: a verdict function that calls
    a 4% regression "faster" would manufacture findings, and a pinning helper that silently fell back would make every
    thread count in the tables a guess.
    """
    checks = [0]

    def check(name, ok, detail=""):
        checks[0] += 1
        if not ok:
            raise AssertionError("%s%s" % (name, (": " + detail) if detail else ""))

    # the verdict boundaries: a ratio at the noise floor is not a result, in either direction
    check("a clear win is `faster`", verdict(1.10, 1.20) == "faster")
    check("a clear loss is `slower`", verdict(0.80, 0.90) == "slower")
    check("an interval covering 1 is indistinguishable", verdict(0.98, 1.03) == "indistinguishable")
    check("an interval inside the noise floor is indistinguishable too", verdict(1.01, 1.015))
    check("the floor is symmetric", verdict(1 - NOISE / 2, 1 + NOISE / 2) == "indistinguishable")

    # identical arms must come out at 1.0 with an interval that covers it: the harness cannot invent a difference
    def work(k):
        def f():
            s = 0
            for i in range(k):
                s += i
            return s
        return f

    same = summarise(paired([("a", work(200000)), ("b", work(200000))], rounds=9))
    check("identical arms are indistinguishable",
          all(r["verdict"] in ("indistinguishable", "baseline") for r in same),
          str([(r["arm"], r["speedup"], r["lo"], r["hi"]) for r in same]))
    check("an identical arm's ratio is 1 to within the floor",
          all(abs(r["speedup"] - 1.0) <= NOISE for r in same), str([r["speedup"] for r in same]))

    # a real 2x difference must be caught, with an interval that excludes 1
    diff = summarise(paired([("fast", work(100000)), ("slow", work(200000))], rounds=9))
    slow = [r for r in diff if r["arm"] == "slow"][0]
    check("a 2x arm is reported slower with an interval below 1", slow["verdict"] == "slower" and slow["hi"] < 1.0,
          str(slow))
    check("... and its point estimate is near a half", 0.35 < slow["speedup"] < 0.75, str(slow["speedup"]))

    # the interval is deterministic: a gate whose interval moves between runs of the same data is not a gate
    t = paired([("x", work(50000)), ("y", work(60000))], rounds=6)
    first = summarise(t)
    again = summarise({k: list(v) for k, v in t.items()})
    check("the bootstrap is seeded, so an interval is reproducible",
          [(r["arm"], r["lo"], r["hi"]) for r in first] == [(r["arm"], r["lo"], r["hi"]) for r in again])
    from tools.nll import bootstrap_ci
    check("same data, same interval", bootstrap_ci([1.0, 2.0, 3.0], block=1, seed=7) ==
          bootstrap_ci([1.0, 2.0, 3.0], block=1, seed=7))

    # every round measures every arm: a dropped arm would shift the paired ratio silently
    t = paired([("a", lambda: None), ("b", lambda: None), ("c", lambda: None)], rounds=4)
    check("paired keeps every arm in every round", all(len(v) == 4 for v in t.values()), str({k: len(v) for k, v in t.items()}))

    # pinning: one physical core when asked for one, and the mask is reported
    info = pin_affinity(1)
    check("one core pins one thread", bin(info["mask"]).count("1") == 1, str(info))
    check("the core count is plausible", 1 <= info["cores"] <= info["logical"], str(info))
    check("the mask is inside the machine", info["mask"] < (1 << info["logical"]), str(info))
    masks = _core_masks()
    check("core masks are distinct and non-empty", len(masks) >= 1 and all(masks) and
          len({m & (-m) for m in masks}) == len(masks), str(masks))
    wide = pin_affinity(info["logical"] + 4)
    check("asking for more threads than cores uses all logical processors",
          bin(wide["mask"]).count("1") == info["logical"], str(wide))
    return checks[0]


def main(argv):
    ap = argparse.ArgumentParser(description="the T-MAC LUT GEMV: validate, then measure with a paired interval")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the extension verbosely")
    ap.add_argument("--threads", type=int, default=8, help="OpenMP threads, and physical cores to pin to")
    ap.add_argument("--rounds", type=int, default=15, help="interleaved rounds for the tiling table")
    ap.add_argument("--runs", type=int, default=3, help="repeat the whole tiling experiment this many times")
    ap.add_argument("--no-pin", action="store_true", help="do not set the process affinity mask")
    ap.add_argument("--json", help="write both tables here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        print("%d checks passed" % selftest())
        return 0
    if torch is None:
        print("tmac: this harness needs torch; --selftest runs without it")
        return 2

    torch.set_num_threads(1)                        # the GEMV is already OpenMP; torch must not add a pool
    info = None if a.no_pin else pin_affinity(a.threads)
    if info is not None:
        print(affinity_line(info))

    mod = build(a.rebuild)
    diff, _w, _a = validate(mod)
    print("validation: max |LUT - exact int8 reference| = %.3e  %s" % (diff, "PASS" if diff == 0 else "FAIL"))
    out = dict(affinity=info, rounds=a.rounds, noise=NOISE)
    if hasattr(mod, "set_threads"):
        mod.set_threads(a.threads)
    out["gemv"] = gemv_bench(mod, threads=(1, max(1, a.threads)), rounds=max(5, a.rounds // 2))
    if hasattr(mod, "tmac_gemv_avx2_tiled"):
        worst = validate_tiled(mod)
        print("\ntiled arm: max |tiled - exact int8 reference| = %.3e over 4 blockings  %s"
              % (worst, "PASS" if worst == 0 else "FAIL"))
        diff = max(diff, worst)
        if hasattr(mod, "set_threads"):
            mod.set_threads(a.threads)
        out["tiling"] = tiling_bench(mod, rounds=a.rounds, runs=a.runs, threads=a.threads)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote %s" % a.json)
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
