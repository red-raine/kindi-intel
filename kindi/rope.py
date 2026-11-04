"""Rotary position tables, including the interpolation a checkpoint can ask for.

For a plain model the table is one line: `cos/sin(pos * base^(-2j/d))` for `j` in `[0, d/2)`. A file may instead carry
`<arch>.rope.scaling.*`, which asks for one of two schemes:

* **linear**: every frequency is divided by the factor, so the table stretches by `factor` positions;
* **yarn** (Peng et al., "YaRN: Efficient Context Window Extension"): the *high* frequencies are interpolated (divided
  by the factor), the *low* frequencies are extrapolated (left alone), the crossover is a linear ramp between two
  correction dimensions, and every cosine and sine is multiplied by `mscale = 1 + 0.1 ln(factor)`.

The retired Bonsai 1.7B and 8B declare `yarn, factor 4, original_context_length 8192`; the 27B family declares none, so
its table is unchanged by this module.

This is llama.cpp's arithmetic, not a re-derivation: `rope_yarn` / `rope_yarn_ramp` in
`ggml/src/ggml-cpu/ops.cpp` and `ggml_rope_yarn_corr_dims` in `ggml/src/ggml.c`, with the parameters
`llama-context.cpp` passes for a `rope.scaling.type = yarn` file (`freq_scale = 1/factor`, `ext_factor = 1`, and the
attention factor that cancels `yarn_attn_factor` against the ramp's own `1 + 0.1 ln(factor)`, leaving `mscale`).
Numerically it is exact to float32, which is the precision the kernels read the table in.
"""
import math

import numpy as np


def corr_dims(rope_dim, n_ctx_orig, base, beta_fast=32.0, beta_slow=1.0):
    """the `[low, high]` pair index range over which the ramp goes from extrapolation to interpolation

    `corr_dim(n_rot) = d * ln(n_ctx_orig / (n_rot * 2 pi)) / (2 ln(base))`, floored for `beta_fast` and ceiled for
    `beta_slow`, clipped to the table. For the 1.7B file (d 128, 8192 positions, base 1e6, 32 and 1) that is `[17, 34]`:
    the first 18 pairs keep their frequencies and the pairs from 34 on are divided by the factor.
    """
    def corr_dim(n_rot):
        return rope_dim * math.log(n_ctx_orig / (n_rot * 2.0 * math.pi)) / (2.0 * math.log(base))

    low = max(0.0, math.floor(corr_dim(beta_fast)))
    high = min(float(rope_dim - 1), math.ceil(corr_dim(beta_slow)))
    return low, high


def ramp(rope_dim, low, high):
    """1 at a pair index at or below `low` (extrapolate), 0 at or above `high` (interpolate)"""
    j = np.arange(rope_dim // 2, dtype=np.float64)
    return 1.0 - np.clip((j - low) / max(0.001, high - low), 0.0, 1.0)


def tables(ctx, rope_dim, base, scaling=None, dtype=np.float32):
    """-> (cos, sin), each `[ctx][rope_dim / 2]`, the shape every kernel's rope reads

    `scaling` is the manifest's `rope_scaling` block: `None` (or `{"type": "none"}`) for the plain table, `linear` or
    `yarn` otherwise. A factor of 1 is the same as none, because dividing by 1 and `mscale(1) = 1` are both identity.
    """
    rh = rope_dim // 2
    inv = np.power(np.float64(base), -2.0 * np.arange(rh, dtype=np.float64) / rope_dim)
    pos = np.arange(int(ctx), dtype=np.float64)[:, None]
    theta = pos * inv[None, :]

    sc = scaling or {}
    kind = str(sc.get("type") or "none").lower()
    factor = float(sc.get("factor") or 1.0)
    if kind in ("", "none") or factor == 1.0:
        cos, sin = np.cos(theta), np.sin(theta)
        return cos.astype(dtype), sin.astype(dtype)
    if kind not in ("linear", "yarn"):
        raise SystemExit("kindi.rope: rope scaling %r is not one this engine applies (linear or yarn)" % (kind,))

    freq_scale = 1.0 / factor
    if kind == "linear":
        cos, sin = np.cos(theta * freq_scale), np.sin(theta * freq_scale)
        return cos.astype(dtype), sin.astype(dtype)

    # yarn: extrapolate where the ramp says so, interpolate where it does not, and scale both by mscale
    low, high = corr_dims(rope_dim, int(sc.get("original_context_length") or ctx), base,
                          float(sc.get("beta_fast") or 32.0), float(sc.get("beta_slow") or 1.0))
    mix = ramp(rope_dim, low, high)[None, :]
    theta = (freq_scale * theta) * (1.0 - mix) + theta * mix
    mscale = 1.0 + 0.1 * math.log(factor)
    mscale *= float(sc.get("attn_factor") or 1.0)
    cos, sin = np.cos(theta) * mscale, np.sin(theta) * mscale
    return cos.astype(dtype), sin.astype(dtype)


def selftest():
    """the table against its closed form, and the two properties a scaled table must have"""
    import numpy as np
    ok = []

    def check(name, cond, detail=""):
        print(("ok   " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
        if not cond:
            raise SystemExit("kindi.rope: %s" % name)
        ok.append(name)

    d, base, ctx = 128, 1e6, 64
    c0, s0 = tables(ctx, d, base)
    inv = base ** (-2.0 * np.arange(d // 2) / d)
    pos = np.arange(ctx)[:, None]
    check("the plain table is cos/sin of pos * base^(-2j/d)",
          np.allclose(c0, np.cos(pos * inv), atol=1e-6) and np.allclose(s0, np.sin(pos * inv), atol=1e-6))
    check("position 0 has no rotation", bool(np.allclose(c0[0], 1.0) and np.allclose(s0[0], 0.0)))

    yarn = {"type": "yarn", "factor": 4.0, "original_context_length": 8192}
    low, high = corr_dims(d, 8192, base)
    check("the correction dims of this file's own parameters", (low, high) == (17.0, 34.0), "%r" % ((low, high),))
    c1, s1 = tables(ctx, d, base, yarn)
    mscale = 1.0 + 0.1 * np.log(4.0)
    check("mscale is 1 + 0.1 ln(factor) on both tables",
          np.allclose(c1[0], mscale, atol=1e-6) and np.allclose(s1[0], 0.0, atol=1e-6), "%.6f" % c1[0, 0])
    # the angles wrap, so each check is against the closed form of the angle it is supposed to use, not against a
    # difference of two wrapped angles: cos/sin carry the position of the angle unambiguously
    j = np.arange(d // 2)
    check("the lowest frequencies keep their angle (extrapolation)",
          np.allclose(c1[1:, j <= low], mscale * c0[1:, j <= low], atol=1e-5)
          and np.allclose(s1[1:, j <= low], mscale * s0[1:, j <= low], atol=1e-5))
    check("the highest frequencies are divided by the factor (interpolation)",
          np.allclose(c1[1:, j >= high], mscale * np.cos(pos[1:] * inv[None, j >= high] / 4.0), atol=1e-5)
          and np.allclose(s1[1:, j >= high], mscale * np.sin(pos[1:] * inv[None, j >= high] / 4.0), atol=1e-5))
    cf, sf = tables(ctx, d, base, {"type": "linear", "factor": 4.0})
    check("linear divides every angle by the factor",
          np.allclose(cf[1:], np.cos(pos[1:] * inv[None, :] / 4.0), atol=1e-6)
          and np.allclose(sf[1:], np.sin(pos[1:] * inv[None, :] / 4.0), atol=1e-6))
    check("a factor of 1 is the plain table",
          np.array_equal(tables(ctx, d, base, {"type": "yarn", "factor": 1.0})[0], c0))

    print("\n%d checks passed" % len(ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
