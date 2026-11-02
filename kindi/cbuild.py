"""Build kcpu.c into a native library on this machine: gcc + OpenMP elsewhere, MSVC on Windows.

`kindi.cpu` needs one shared object called `libkcpu.so` (or `kcpu.dll`) with a C ABI it can call through ctypes.
On Linux and WSL that is `gcc -O3 -march=native -fopenmp -shared -fPIC`. Windows has no gcc here, so the same file
is built with cl.exe out of a Visual Studio install: /O2 /arch:AVX2 /openmp /LD. OpenMP matters more than it looks -
the forward pass is one parallel region with ~425 barriers per token - and MSVC's /openmp does implement the
constructs kcpu.c uses (parallel, for, single, barrier, nowait sections), so the Windows build is the same engine
and not a degraded one.

The AVX2 tier needs /arch:AVX2 on MSVC because there is no per-function target attribute; kcpu.c keeps a scalar tier
for builds without it, and `kindi verify-tier` compares whatever the build actually has.
"""
import glob
import os
import subprocess
import sys

VS_PATHS = [
    r"C:\Program Files\Microsoft Visual Studio",
    r"C:\Program Files (x86)\Microsoft Visual Studio",
    r"C:\Program Files\Microsoft Visual Studio\2022",
]


def find_vcvarsall():
    """the newest Visual Studio install that has a VC toolset, or None"""
    found = []
    for root in VS_PATHS:
        for p in glob.glob(os.path.join(root, "*", "*", "VC", "Auxiliary", "Build", "vcvarsall.bat")):
            found.append(p)
        for p in glob.glob(os.path.join(root, "*", "VC", "Auxiliary", "Build", "vcvarsall.bat")):
            found.append(p)
    if not found:
        return None
    return sorted(found)[-1]


def build(src, cache):
    """compile kcpu.c into <cache>/kcpu.dll (Windows) or libkcpu.so (elsewhere); returns the library path"""
    if sys.platform == "win32":
        out = os.path.join(cache, "kcpu.dll")
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            return out
        vc = find_vcvarsall()
        if not vc:
            raise SystemExit("kindi: no Visual Studio VC toolset found; install the C++ build tools, or set "
                             "KINDI_CPU_LIB to a prebuilt kcpu.dll")
        obj = os.path.join(cache, "kcpu.obj")
        cmd = 'call "%s" x64 >nul && cl /nologo /O2 /arch:AVX2 /openmp /LD "%s" /Fo"%s" /Fe:"%s"' % (vc, src, obj, out)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)      # runs through cmd.exe /c
        if r.returncode or not os.path.exists(out):
            raise SystemExit("kindi: could not build the CPU engine with cl.exe\n%s\n%s"
                             % (r.stdout[-3000:], r.stderr[-3000:]))
        return out
    out = os.path.join(cache, "libkcpu.so")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        flags = os.environ.get("KINDI_CPU_CFLAGS", "-O3 -march=native -fopenmp -fno-math-errno -shared -fPIC")
        r = subprocess.run("gcc %s -o %s %s -lm" % (flags, out, src), shell=True, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("kindi: could not build the CPU engine (%s); it needs gcc with OpenMP\n%s"
                             % (out, r.stderr[-3000:]))
    return out
