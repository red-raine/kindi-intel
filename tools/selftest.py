"""One command that runs every check this repository has, and refuses to pass quietly.

    python -m tools.selftest              # the gate: nonzero exit if anything regressed
    python -m tools.selftest --verbose    # also list what was skipped, and why

There is no pytest and no CI in this repository, so each part of it carries its own self-check: `kindi.quant.selftest`,
`kindi.convert.selftest`, the four tool suites, and the CLI. This runner is the thing that runs them in one go and turns
their exit codes into a single answer, so "it still works" is a command rather than a memory of eight terminals.

Three of the checks are new, because nothing covered them:

  * **imports** - import every module under `kindi/` and `tools/`. A module that cannot be imported is broken even if no
    test touches it. Modules that need `cupy` are reported as skipped with that reason: they are the CUDA execution path
    and this branch runs on a CPU-only host, where requiring a GPU to parse a config would be the bug.
  * **docs** - every relative Markdown link in `README.md` and `docs/` must resolve. Measurement pages are worthless if
    their links rot.
  * **cli** - `python -m kindi --version` and `info` must run end to end on a host with no pack and no GPU.

Not covered, stated rather than implied: no check exercises the GPU execution path (`spec`, `tp`, `chunk`, `chunk1`,
`tc` are imported only where `cupy` exists), the streaming probe is skipped because it writes hundreds of megabytes,
and nothing here tests model quality - these are structural and arithmetic regressions only.
"""
import importlib
import os
import pkgutil
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GPU_ONLY = "needs cupy (CUDA execution path; skipped on a CPU-only host)"


class Result(object):
    def __init__(self, name, status, detail=""):
        self.name, self.status, self.detail = name, status, detail


def _run(cmd, timeout=300):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or ""), time.perf_counter() - t0


# --- checks ---------------------------------------------------------------------------------------------------------

def check_imports():
    """import everything; a module that cannot load is a regression regardless of coverage"""
    results = []
    for pkg, path in (("kindi", os.path.join(ROOT, "kindi")), ("tools", HERE)):
        for info in sorted(pkgutil.iter_modules([path]), key=lambda m: m.name):
            name = "%s.%s" % (pkg, info.name)
            if name == "tools.selftest":
                continue
            try:
                importlib.import_module(name)
            except ImportError as e:
                missing = getattr(e, "name", "") or ""
                if "cupy" in str(e) or "cupy" in missing:
                    results.append(Result(name, "skip", GPU_ONLY))
                else:
                    results.append(Result(name, "FAIL", "%s: %s" % (type(e).__name__, e)))
            except Exception as e:
                results.append(Result(name, "FAIL", "%s: %s" % (type(e).__name__, e)))
    return results


def check_engine_selftests():
    results = []
    for mod, fn in (("kindi.quant", "selftest"), ("kindi.convert", "selftest")):
        try:
            m = importlib.import_module(mod)
            getattr(m, fn)()
            results.append(Result("%s.%s" % (mod, fn), "ok"))
        except Exception as e:
            results.append(Result("%s.%s" % (mod, fn), "FAIL", "%s: %s" % (type(e).__name__, e)))
    return results


SUITES = [
    ("suite:astvalid", [sys.executable, "-m", "tools.astvalid", "selftest"], "checks passed"),
    ("suite:moe_loop", [sys.executable, "-m", "tools.moe_loop.test_moe_loop"], "checks passed"),
    ("suite:bitnet_gemm", [sys.executable, "-m", "tools.bitnet_gemm.test_bitnet"], "checks passed"),
    ("suite:expert_merge", [sys.executable, "-m", "tools.expert_merge.merge"], "permutation logic verified"),
    ("suite:prefetch_budget", [sys.executable, "-m", "tools.bitnet_gemm.prefetch_budget"], "over-fetch"),
    ("suite:exodus", [sys.executable, "-m", "tools.exodus"], "check groups passed"),
]


def check_suites():
    results = []
    for name, cmd, expect in SUITES:
        try:
            rc, out, dt = _run(cmd)
        except subprocess.TimeoutExpired:
            results.append(Result(name, "FAIL", "timed out"))
            continue
        if rc != 0:
            tail = "\n".join(out.strip().splitlines()[-6:])
            results.append(Result(name, "FAIL", "exit %d\n%s" % (rc, tail)))
        elif expect and expect not in out:
            results.append(Result(name, "FAIL", "exit 0 but the summary line %r is missing" % expect))
        else:
            results.append(Result(name, "ok", "%.1fs" % dt))
    return results


def check_cli():
    results = []
    for name, cmd, expect in (
            ("cli:version", [sys.executable, "-m", "kindi", "--version"], "Kindi"),
            ("cli:info", [sys.executable, "-m", "kindi", "info"], "config:"),
    ):
        try:
            rc, out, dt = _run(cmd, timeout=120)
        except subprocess.TimeoutExpired:
            results.append(Result(name, "FAIL", "timed out"))
            continue
        if rc != 0:
            results.append(Result(name, "FAIL", "exit %d: %s" % (rc, out.strip().splitlines()[-1:] or [""])))
        elif expect not in out:
            results.append(Result(name, "FAIL", "output does not contain %r" % expect))
        else:
            results.append(Result(name, "ok", "%.1fs" % dt))
    return results


LINK = re.compile(r"\]\(([^)\s]+)\)")


def check_docs():
    """every relative markdown link must resolve, so the measurement pages cannot rot"""
    files = [os.path.join(ROOT, "README.md")]
    docs = os.path.join(ROOT, "docs")
    if os.path.isdir(docs):
        files += [os.path.join(docs, n) for n in sorted(os.listdir(docs)) if n.endswith(".md")]
    for extra in ("WHY.md",):
        p = os.path.join(ROOT, extra)
        if os.path.isfile(p):
            files.append(p)
    broken, total = [], 0
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for target in LINK.findall(text):
            if "://" in target or target.startswith(("mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            total += 1
            if not os.path.exists(os.path.join(os.path.dirname(f), path)):
                broken.append("%s -> %s" % (os.path.relpath(f, ROOT), target))
    if broken:
        return [Result("docs:links", "FAIL", "%d of %d broken:\n  %s" % (len(broken), total, "\n  ".join(broken)))]
    return [Result("docs:links", "ok", "%d relative links in %d files" % (total, len(files)))]


SKIPPED = [
    ("tools.bitnet_gemm.stream_probe", "writes ~1 GB to disk per run; run it directly when measuring paging"),
]


# --- runner ---------------------------------------------------------------------------------------------------------

def main(argv):
    verbose = "--verbose" in argv or "-v" in argv
    t0 = time.perf_counter()
    results = []
    for part in (check_imports, check_engine_selftests, check_suites, check_cli, check_docs):
        try:
            results += part()
        except Exception as e:                          # a broken check is a failure, not a crash
            results.append(Result(part.__name__, "FAIL", "%s: %s" % (type(e).__name__, e)))

    fails = [r for r in results if r.status == "FAIL"]
    skips = [r for r in results if r.status == "skip"]
    oks = [r for r in results if r.status == "ok"]

    for r in fails:
        print("FAIL %s\n     %s" % (r.name, r.detail.replace("\n", "\n     ")))
    if verbose:
        for r in skips:
            print("skip %-34s %s" % (r.name, r.detail))
        for name, why in SKIPPED:
            print("skip %-34s %s" % (name, why))
    for r in oks:
        if verbose and r.detail:
            print("ok   %-34s %s" % (r.name, r.detail))

    print("\n%d ok, %d skipped, %d failed in %.1fs"
          % (len(oks), len(skips) + len(SKIPPED), len(fails), time.perf_counter() - t0))
    if not verbose and skips:
        print("(the skipped modules are the CUDA path; --verbose lists them)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
