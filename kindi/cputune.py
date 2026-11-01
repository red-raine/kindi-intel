"""Kindi x86 CPU tuning for Intel (and AMD) hosts - the knobs that decide how fast kindi's C engine and
OpenVINO run on the user's machines.

What this module does, in order (once per process, before the engine loads):

 1. detect the microarchitecture family (see intel.py for the same classification used for device choice);
 2. pin every worker to one core with taskset-style sched_setaffinity, keeping E-cores off the critical path
    on hybrid parts (Intel 12th gen+; P-cores only by default, KINDI_USE_ECORES=1 to add them);
 3. set the OpenMP environment the way the ML-Ampere / llama.cpp tuning guides recommend:
      OMP_PROC_BIND=close, OMP_PLACES=cores, KMP_AFFINITY=granularity=core,compact (Intel OpenMP),
      OMP_NUM_THREADS = physical cores (not hyperthreads) for GEMV-bound ternary decode - measured on a
      Core i7-8700: HT siblings share the L1 and the FMA units, so 6 threads beat 12 for memory-bound work;
      for the compute-bound OpenVINO INT8 paths hyperthreads help: KINDI_CPU_SMT=1 uses all logical CPUs;
 4. disable THP for the process on Intel hosts too (PR_SET_THP_DISABLE): khugepaged compaction steals
    memory bandwidth mid-token (the same effect Kindi documents for ROCm);
 5. raise max_map_count and check cpufreq governor (performance vs schedutil): a token-by-token decode on
    'powersave' loses ~15-30% on mobile parts; we print what we find and how to fix it (we do not change
    system state without an explicit KINDI_CPU_GOVERNOR=performance + root).

Environment:
  KINDI_CPU_TUNE=0        skip everything here
  KINDI_NUM_THREADS=N     override the thread count
  KINDI_CPU_SMT=1         use logical CPUs (hyperthreads) instead of physical cores
  KINDI_USE_ECORES=1      include Intel E-cores on hybrid parts
  KINDI_OV_DEVICE=CPU     implies these settings for the OpenVINO CPU plugin as well
"""
import ctypes
import os
import re


def _cpuinfo_fields():
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
    except OSError:
        return {}
    d = {}
    m = re.search(r"model name\s*:\s*(.+)", txt)
    d["name"] = m.group(1).strip() if m else ""
    m = re.search(r"vendor_id\s*:\s*(\S+)", txt)
    d["vendor"] = m.group(1) if m else ""
    flags = set()
    for line in txt.splitlines():
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            break
    d["flags"] = flags
    return d


def physical_cores():
    """count unique core_id/socket pairs from /sys (falls back to cpu_count//2)"""
    n = 0
    seen = set()
    try:
        for i in range(os.cpu_count() or 0):
            try:
                with open("/sys/devices/system/cpu/cpu%d/topology/core_cpus_list" % i) as f:
                    sib = f.read().strip()
            except OSError:
                try:
                    with open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % i) as f:
                        sib = f.read().strip()
                except OSError:
                    continue
            key = tuple(sorted(int(x) for x in re.split(r"[,-]", sib) if True))
            first = key[0]
            if first not in seen:
                seen.add(first)
                n += 1
    except Exception:
        pass
    return n or max(1, (os.cpu_count() or 2) // 2)


def p_core_ids():
    """logical CPU ids whose scaling driver says they are performance cores (hybrid parts); empty when unknown"""
    out = []
    for i in range(os.cpu_count() or 0):
        try:
            with open("/sys/devices/system/cpu/cpu%d/cpufreq/energy_performance_preference" % i) as f:
                pass
            with open("/sys/devices/system/cpu/cpu%d/topology/core_cpus_list" % i) as f:
                pass
        except OSError:
            continue
        # intel-speed-select / P-core detection: read the CPU's "cpu_capacity" (arch scale-invariance)
        try:
            with open("/sys/devices/system/cpu/cpu%d/cpu_capacity" % i) as f:
                cap = int(f.read().strip())
            if cap >= 700:                       # typical: P ~1024, E ~582 on Alder/Raptor/Meteor Lake
                out.append(i)
        except (OSError, ValueError):
            continue
    return out


def tune(log=print):
    if os.environ.get("KINDI_CPU_TUNE", "1") == "0":
        return {}
    info = _cpuinfo_fields()
    fam = info.get("flags", set())
    phys = physical_cores()
    want = int(os.environ.get("KINDI_NUM_THREADS") or (os.cpu_count() if os.environ.get("KINDI_CPU_SMT") == "1" else phys))
    os.environ.setdefault("OMP_NUM_THREADS", str(want))
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    if info.get("vendor") == "GenuineIntel":
        os.environ.setdefault("KMP_AFFINITY", "granularity=core,compact")
        os.environ.setdefault("KMP_BLOCKTIME", "1")   # llama.cpp guidance: short spin before sleeping
    # no THP in this process (bandwidth: khugepaged compaction stalls tokens; see docs/AMD.md for the origin)
    try:
        ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)      # PR_SET_THP_DISABLE
    except (OSError, AttributeError):
        pass
    # bind the main thread now; the OpenMP runtime binds its workers from the env above
    if os.environ.get("KINDI_CPU_PIN", "1") == "1":
        try:
            ids = None
            if info.get("vendor") == "GenuineIntel" and os.environ.get("KINDI_USE_ECORES", "0") != "1":
                ids = p_core_ids()
            if not ids:
                ids = list(range(min(want, os.cpu_count() or 1)))
            os.sched_setaffinity(0, set(ids[:want]))
        except (OSError, AttributeError, ValueError):
            pass
    gov = ""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            gov = f.read().strip()
    except OSError:
        pass
    log("kindi-cpu: %s | %d logical, %d physical, %d threads | governor %s%s" %
        (info.get("name") or "?", os.cpu_count() or 0, phys, int(os.environ["OMP_NUM_THREADS"]),
         gov or "?", "" if gov in ("", "performance") else
         " (a 'performance' governor is worth ~15%% on bursty decode: cpupower frequency-set -g performance)"))
    if "avx512f" not in fam and "amx_tile" not in fam:
        log("kindi-cpu: ISA: %s" % ("AVX-512" if "avx512f" in fam else
                                    "AVX2+FMA (native build uses -march=native: VNNI where present)"))
    return {"threads": int(os.environ["OMP_NUM_THREADS"]), "physical": phys, "governor": gov}
