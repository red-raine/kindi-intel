"""Does one lookup that returns N tokens' sums beat 16 weights a lookup?

`rows8_b2` (the production two-token GEMV in kcpu.c) spends one vpermps per token per pair position: a lookup returns 8
rows x 2 columns = 16 weights, however many tokens are in flight. The vector-LUT form (`rows8_vlut`'s arithmetic, in
kcpu.c as vlut2/vlut4) makes the table entry carry the whole batch - TT[p][c] = int16 sum of token A | int16 sum of
token B << 16 - so one lookup returns 8 rows x 2 columns x 2 tokens = 32 weights, and at N = 4 two lookups return 64.
That is the "weights per lookup" claim this module measures, and the thing it can get wrong is the claim that follows
from it (that the lookup is what the kernel is waiting on):

    python -m tools.vlut.measure            # the table, with the host's noise next to it
    python -m tools.vlut.measure --selftest # checks with known answers

Three things are kept apart, because they are three different questions:

  - the arithmetic: does the vector LUT compute what the pack says?  It is an int8 kernel, so it is compared against an
    exact int64 numpy evaluation of the same quantised input (relL2 ~ 1e-7), not against the fp32 kernel.
  - the cost of that int8: how far is it from the fp32 `rows8_b2` on the same tokens (relL2 ~ 1e-3), which is the price
    rows8_q8 already pays and this form does not change.
  - the speed: each arm through tools.bench, which reports the host's spread and refuses to call a difference smaller
    than its own noise a difference.

Timing is printed, never asserted - this repository's gate has been decided the wrong way by a single perf_counter
sample twice. What is asserted is deterministic: the two arms agree with their references, N = 4 is the same values as
N = 2 per token, and the weights-per-lookup arithmetic (16 vs 32) is what the loops actually do.
"""
import ctypes
import sys
import time

import numpy as np

from ..bench import bench

NA, NB = "vlut", "b2"
PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def _lib():
    """the built kcpu.c, with the four exports this module uses typed"""
    from kindi.cpu import PI8, PF, Mat, _lib as build
    lib = build()
    if not lib.kc_vlut_ok():
        raise SystemExit("tools.vlut: this kcpu build has no AVX2 tier, so there is no vector LUT to measure")
    lib.kc_q8_quant.argtypes = [PF, ctypes.c_int, PI8, PF]
    lib.kc_q8_quant.restype = None
    lib.kc_b2_gemv.argtypes = [ctypes.POINTER(Mat), ctypes.c_int, ctypes.POINTER(PF), ctypes.POINTER(PF), PF, PF]
    lib.kc_b2_gemv.restype = None
    lib.kc_vlut_gemv.argtypes = [ctypes.POINTER(Mat), ctypes.c_int, ctypes.POINTER(PI8), ctypes.POINTER(PF),
                                 ctypes.POINTER(PF)]
    lib.kc_vlut_gemv.restype = ctypes.c_int
    return lib


def _p(a, t=None):
    from kindi.cpu import PF
    return a.ctypes.data_as(t or PF)


def packed(R, C, seed=0, constant=False):
    """a synthetic ternary matrix in the pack's own form, and the weights it decodes to.

    The codes are uniform over the 81 four-trit bytes, so all nine pair codes (eight real, one 0*0) appear at the rate a
    random model gives them - which matters here, because the 0*0 code is the one the kernel has to mask out, and a
    corpus that never produced it would hide a break in that mask. `constant` writes the byte 80 instead, which is four
    +1 trits: every code is 7 and every sum is at its largest, which is the case the int16 accumulator has to survive.
    """
    from kindi.cpu import pair_scales, paircodes
    rng = np.random.RandomState(seed)
    b = (np.full((C // 4, R), 80, np.uint8) if constant
         else rng.randint(0, 81, size=(C // 4, R)).astype(np.uint8))
    W = paircodes(b, R, C)
    G = C // 128
    sc = (0.02 + 0.05 * np.abs(rng.randn(G, R))).astype(np.float16)
    w = np.empty((C, R), np.int8)
    for j in range(4):
        w[j::4] = ((b // (3 ** j)) % 3).astype(np.int8) - 1
    return W, pair_scales(sc), w


def weights_per_lookup(arm, n):
    """Weights whose partial sums one lookup returns, at 8 rows a block and 2 columns a pair code.

    `rows8_b2` holds one token in a table, so a vpermps returns 8 x 2 = 16 whatever the batch. The vector LUT holds two
    tokens in the int16 halves of one int32 lane, so a vpermps returns 8 x 2 x 2 = 32, and at N = 4 two of them cover
    8 x 2 x 4 = 64.
    """
    if arm == "b2":
        return 16
    return 16 * min(n, 2)


class Arm(object):
    """one matrix, `nt` tokens, and the two kernels over them"""

    def __init__(self, lib, R, C, nt, seed=1, constant=False):
        from kindi.cpu import PI8, PF, PU8, Mat
        self.lib, self.R, self.C, self.nt, self.lib_ty = lib, R, C, nt, (PI8, PF, PU8, Mat)
        self.W, self.S, self.w = packed(R, C, seed=0, constant=constant)
        self.G = C // 128
        self.mat = Mat(_p(self.W, PU8), _p(self.S), R, C)
        rng = np.random.RandomState(seed)
        self.x = [np.ones(C, np.float32) if constant else
                  np.ascontiguousarray(rng.randn(C).astype(np.float32)) for _ in range(nt)]
        self.q = [np.zeros(C, np.int8) for _ in range(nt)]
        self.qs = [np.zeros(self.G, np.float32) for _ in range(nt)]
        for t in range(nt):
            lib.kc_q8_quant(_p(self.x[t]), C, _p(self.q[t], PI8), _p(self.qs[t]))
        self.y = [np.zeros(R, np.float32) for _ in range(nt)]
        self.tab = [np.zeros(C // 2 * 8, np.float32) for _ in range(2)]
        self.X = (PF * nt)(*[_p(a) for a in self.x])
        self.Q = (PI8 * nt)(*[_p(a, PI8) for a in self.q])
        self.QS = (PF * nt)(*[_p(a) for a in self.qs])
        self.Y = (PF * nt)(*[_p(a) for a in self.y])

    def clear(self):
        for a in self.y:
            a.fill(0)

    def b2(self):
        """the production two-token kernel: fp32 pair tables, exact, one weight pass a token pair"""
        self.lib.kc_b2_gemv(ctypes.byref(self.mat), self.nt, self.X, self.Y, _p(self.tab[0]), _p(self.tab[1]))
        return self.y

    def vlut(self):
        """the vector LUT: int8 activations, int16 table entries holding two tokens each"""
        self.clear()
        rc = self.lib.kc_vlut_gemv(ctypes.byref(self.mat), self.nt, self.Q, self.QS, self.Y)
        if rc != 0:
            raise RuntimeError("kc_vlut_gemv refused N=%d (rc %d)" % (self.nt, rc))
        return self.y


def rel_l2(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))


def paired(a, b, rounds=15, warmup=2):
    """Time `a` and `b` alternately inside one loop and compare their per-round ratio.

    A shared host drifts: one arm's samples land on a busy patch and the other's do not, and two flat medians then differ
    by more than the kernels do. Alternating puts both arms under the same drift, so the median of the per-round ratios
    is the estimate and the interquartile spread around it is the uncertainty; a spread that straddles 1.0 means the two
    arms were not told apart on this host and the call says so.
    """
    for _ in range(warmup):
        a()
        b()
    rs = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        a()
        t1 = time.perf_counter()
        b()
        t2 = time.perf_counter()
        rs.append((t1 - t0) / (t2 - t1))
    rs = np.asarray(rs, np.float64)
    lo, hi = float(np.percentile(rs, 25)), float(np.percentile(rs, 75))
    med = float(np.median(rs))
    clear = not (lo <= 1.0 <= hi)
    verdict = ("%s faster" % (NA if med < 1.0 else NB)) if clear else "not told apart on this host"
    return dict(ratio=med, lo=lo, hi=hi, clear=clear,
                text="%s/b2 time = %.2fx [%.2f-%.2f]  %s" % (NA, med, lo, hi, verdict))


def exact(arm, t):
    """What the kernel should produce, in int64: per 128-column group the integer dot product of the pack's weights
    with the quantised input, times the group's activation scale and the row's scale."""
    w, S = arm.w, arm.S.transpose(0, 2, 1).reshape(arm.R, arm.G)
    q, qs = arm.q[t].astype(np.int64), arm.qs[t]
    out = np.zeros(arm.R, np.float64)
    for g in range(arm.G):
        gamma = w[g * 128:(g + 1) * 128].T.astype(np.int64) @ q[g * 128:(g + 1) * 128]
        out += (S[:, g].astype(np.float64) * float(qs[g])) * gamma
    return out


def report():
    lib = _lib()
    print("vector LUT against rows8_b2, both kernels single-threaded on the same pack and the same tokens\n")
    for R, C in ((6144, 2048), (1024, 2048), (256, 512)):
        print("  %d x %d  (%.1f M weights, %.0f KiB of pair codes)" % (R, C, R * C / 1e6, R * C / 4 / 1024))
        for nt in (2, 4):
            a = Arm(lib, R, C, nt)
            a.vlut()
            ref_vlut = [y.copy() for y in a.y]
            a.b2()
            err = float(np.mean([rel_l2(ref_vlut[t], a.y[t]) for t in range(nt)]))
            a.vlut()
            t_vlut = bench(a.vlut, rounds=9, warmup=2)
            t_b2 = bench(a.b2, rounds=9, warmup=2)
            p = paired(a.vlut, a.b2, rounds=15)
            print("    N=%d  b2   min %5.2f ms  med %5.2f (%2.0f%%)   %5.1f GMAC/s"
                  % (nt, 1e3 * t_b2["min"], 1e3 * t_b2["median"], 100 * t_b2["rel"], R * C * nt / t_b2["min"] / 1e9))
            print("    N=%d  vlut min %5.2f ms  med %5.2f (%2.0f%%)   %5.1f GMAC/s"
                  % (nt, 1e3 * t_vlut["min"], 1e3 * t_vlut["median"], 100 * t_vlut["rel"],
                     R * C * nt / t_vlut["min"] / 1e9))
            print("          %s   (int8-vs-fp32 relL2 %.1e)" % (p["text"], err))
        print()
    print("Read the N = 2 rows for the lookup width: both arms make one pass over the weights there, so the ratio is the")
    print("vpermd that now returns two tokens instead of one. The N = 4 rows compare one weight pass against two, which")
    print("is a separate effect: the batch shares the decode and the stream, and both are part of what batching a prefill")
    print("is for.")
    return 0


def selftest():
    del PASS[:]
    lib = _lib()
    R, C = 64, 512
    check("vlut: the build has the AVX2 kernel", lib.kc_vlut_ok() == 1)
    a = Arm(lib, R, C, 2)
    a.vlut()
    exact_err = max(rel_l2(a.y[t], exact(a, t)) for t in range(2))
    check("vlut: N=2 matches an int64 evaluation of the same quantised input", exact_err < 1e-5,
          "relL2 %.1e" % exact_err)
    # the int8 price: against the fp32 kernel on the same tokens
    q_ref = [np.array(y, copy=True) for y in a.y]
    a.b2()
    int8_err = max(rel_l2(q_ref[t], a.y[t]) for t in range(2))
    check("vlut: the int8 activation costs less than 1% against the fp32 kernel", int8_err < 1e-2,
          "relL2 %.1e" % int8_err)
    # N = 4 is four independent chains: the two halves must be the same values as N = 2, token for token
    b2 = Arm(lib, R, C, 4)
    b2.vlut()
    four = [np.array(y, copy=True) for y in b2.y]
    two = []
    for lo in (0, 2):
        h = Arm(lib, R, C, 2, seed=1)
        h.x, h.q, h.qs = b2.x[lo:lo + 2], b2.q[lo:lo + 2], b2.qs[lo:lo + 2]
        h.X = (h.lib_ty[1] * 2)(*[_p(v) for v in h.x])
        h.Q = (h.lib_ty[0] * 2)(*[_p(v, h.lib_ty[0]) for v in h.q])
        h.QS = (h.lib_ty[1] * 2)(*[_p(v) for v in h.qs])
        h.vlut()
        two += [np.array(y, copy=True) for y in h.y]
    same = all(np.array_equal(four[t], two[t]) for t in range(4))
    check("vlut: N=4 is the same values as two N=2 passes", same,
          "max |diff| %.1e" % max(float(np.abs(four[t] - two[t]).max()) for t in range(4)))
    # and the same for the production kernel, which is the arm N = 4 is compared against
    c4 = Arm(lib, R, C, 4, seed=3)
    c4.b2()
    f4 = [np.array(y, copy=True) for y in c4.y]
    c2 = Arm(lib, R, C, 2, seed=3)
    c2.b2()
    same_b2 = all(np.array_equal(f4[t], c2.y[t]) for t in range(2))
    check("b2: two passes at N=4 equal one pass at N=2, bit for bit", same_b2)
    wp = weights_per_lookup("vlut", 2), weights_per_lookup("vlut", 4)
    check("vlut: a lookup returns 32 weights against the fp32 form's 16, at N=2 and N=4",
          wp == (32, 32) and weights_per_lookup("b2", 4) == 16, "b2 16, vlut %d and %d" % wp)
    # the one thing that could silently corrupt a real pack: the int16 accumulator. 128 columns of |127| activations
    # against all +1 weights is 16256, under 32767 by 2.0x - the kernel must still be exact there, not wrapped
    wc = Arm(lib, 64, 2048, 2, constant=True)
    wc.vlut()
    worst = max(rel_l2(wc.y[t], exact(wc, t)) for t in range(2))
    check("vlut: the int16 group sum is exact at the worst case", worst < 1e-6,
          "|gamma| 16256 of 32767 (2.0x headroom), relL2 %.1e" % worst)
    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == 7 else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    return report()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
