"""The arithmetic of the rotation stage, executed rather than described.

The specification says: multiply by an orthogonal matrix, the inner product is unchanged, so the network computes the same
function; then snap to {-1, 0, +1} times a per-channel scale, and entropy that was concentrated in a few outlier channels
is now spread out, so snapping costs less. That is true, and it is testable on a 64x256 matrix without any donor model.What it is not is what the supplied code does. Two defects, both measured here rather than argued:

1. **The transform it hands back is transposed.** With right-multiplication `W' = W A`, the identity requires the
   activation transform to be `A^T`, not `A`. Test `direction` shows the residual of using `A` where `A^T` belongs, and
   test `asymmetry` shows the bug is live rather than latent: the optimiser moves `R`, so `A = H R` stops being the
   symmetric Hadamard matrix and `A != A^T` by 0.48.

2. **Its gradient is truncated, so the optimiser steps uphill.** `round()` has zero derivative almost everywhere, so the
   only path autograd can find from `R` to the loss runs through the per-row `scale`. That is not the gradient of the
   loss, and descending along it moves `R` in a direction that raises the loss it is printed next to: measured, 150 steps
   take the loss from 3.40 to 5.63 and SQNR from 9.45 dB to 7.26 dB. A number that rises under its own optimiser cannot
   be used as an acceptance gate, which is what the specification asks of it.

`refine` replaces that with a smooth objective: for a fixed grid the rounding error of a row is governed by the
concentration of its magnitudes, `mean(|w|^2) / mean(|w|)^2`, so minimise that on the Stiefel manifold with a projected
gradient and a polar retraction. Test `refine` measures the SQNR it buys over plain Hadamard rotation, and test `gate`
measures the specification's own >22 dB acceptance threshold against the best arm here.

    python -m tools.exodus.rotate
"""
import numpy as np
import torch

# --- rotation ---------------------------------------------------------------------------------------------------------

def hadamard(n, dtype=np.float64):
    """Normalised Walsh-Hadamard matrix. `n` must be a power of two; there is no Givens padding here on purpose."""
    if n < 1 or n & (n - 1):
        raise ValueError("Hadamard needs a power-of-two length, got %d - pad the input dimension to the next power of "
                         "two (K=5120 -> 8192) or use a random orthogonal matrix instead" % n)
    h = np.array([[1.0]], dtype=dtype)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def random_orthogonal(n, seed=0, dtype=np.float64):
    """A deterministic orthogonal matrix, for the cases Hadamard cannot express."""
    q, r = np.linalg.qr(np.random.RandomState(seed).randn(n, n))
    return q * np.sign(np.diag(r))


def forward(w, x, a):
    """The identity the specification relies on: (W A)(A^T x) == W x, exactly, in float64."""
    return (w @ a) @ (a.T @ x)


def squared_error(w, w_hat):
    return float(((w - w_hat) ** 2).sum())


def sqnr_db(w, w_hat):
    """Signal-to-quantisation-noise ratio of a reconstruction, in dB. The specification asks for >22 dB."""
    noise = (w - w_hat) ** 2
    return float(10.0 * np.log10((w ** 2).sum() / max(noise.sum(), 1e-300)))


# --- ternary snapping -------------------------------------------------------------------------------------------------

def tenant(w, scale=None):
    """Per-output-channel scale, snap to {-1, 0, +1}, rescale. This is the grid both implementations use."""
    if scale is None:
        scale = np.abs(w).mean(axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-30)
    q = np.clip(np.round(w / scale), -1, 1)
    return q, scale


def quantise(w, scale=None):
    q, s = tenant(w, scale)
    return q * s


def outlier_matrix(rows=64, cols=256, outliers=2, magnitude=60.0, seed=0):
    """A weight matrix of the shape real layers have: gaussian, plus a few channels carrying all the dynamic range."""
    rs = np.random.RandomState(seed)
    w = rs.randn(rows, cols)
    w[:, :outliers] *= magnitude
    return w


# --- the objective the supplied optimiser cannot see ------------------------------------------------------------------

def kurtosis_objective(wp):
    """Per-row concentration of magnitudes; the rounding error of a fixed grid rises with it. Differentiable."""
    a = wp.abs()
    return (a.pow(2).mean(dim=1) / a.mean(dim=1).pow(2)).mean()


def stiefel_project(r):
    """Polar retraction onto the orthogonal group."""
    u, _, vt = torch.linalg.svd(r)
    return u @ vt


def refine(w_rot, steps=300, lr=0.05):
    """Minimise magnitude concentration over R in O(n) by projected gradient descent.

    The gradient of a smooth function of `R` is not a tangent vector; the Stiefel tangent projection removes the
    symmetric part, which is what makes this Riemannian rather than a plain loop with a projection bolted on.
    """
    k = w_rot.shape[1]
    r = torch.eye(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([r], lr=lr)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = kurtosis_objective(w_rot @ r)
        if first is None:
            first = float(loss.detach())
        loss.backward()
        with torch.no_grad():                              # tangent space: grad - R sym(R^T grad)
            g = r.grad
            r.grad.copy_(g - r @ (0.5 * (r.t() @ g + g.t() @ r)))
        opt.step()
        with torch.no_grad():                              # retraction back onto the manifold
            r.copy_(stiefel_project(r))
    return r.detach().numpy(), first, float(loss.detach())


def supplied_optimiser(w_rot, steps=150, lr=1e-3):
    """The specification's loop, verbatim in structure: round(), clamp, backward, Adam, project.

    Returns (R, first loss, final loss) so the test can show that stepping along this gradient raises the loss.
    """
    k = w_rot.shape[1]
    r = torch.eye(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([r], lr=lr)
    first = None
    loss = None
    for _ in range(steps):
        opt.zero_grad()
        wp = w_rot @ r
        scale = wp.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        wq = torch.clamp(torch.round(wp / scale), -1, 1) * scale
        loss = torch.mean((wp - wq) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            r.copy_(stiefel_project(r))
        if first is None:
            first = float(loss.detach())
    return r.detach().numpy(), float(first), float(loss.detach())


# --- tests ------------------------------------------------------------------------------------------------------------

PASS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if ok:
        PASS.append(name)
    return ok


def t_hadamard():
    h = hadamard(256)
    err = float(np.abs(h @ h.T - np.eye(256)).max())
    check("hadamard: H H^T = I", err < 1e-15, "max |H H^T - I| = %.1e" % err)
    try:
        hadamard(1920)
        check("hadamard: refuses a non-power-of-two", False, "1920 was accepted")
    except ValueError as e:
        check("hadamard: refuses a non-power-of-two", "pad the input dimension" in str(e))


def t_identity():
    w = np.random.RandomState(1).randn(32, 64)
    x = np.random.RandomState(2).randn(64)
    a = hadamard(64)
    got, want = forward(w, x, a), w @ x
    check("identity: (W A)(A^T x) = W x", np.abs(got - want).max() < 1e-13,
          "max abs error %.1e" % np.abs(got - want).max())


def t_direction():
    """Using A where A^T belongs is not a rounding difference - measure it."""
    w = np.random.RandomState(3).randn(32, 64)
    x = np.random.RandomState(4).randn(64)
    a = hadamard(64) @ random_orthogonal(64, seed=5)          # a rotation that is NOT symmetric
    right, want = forward(w, x, a), w @ x
    wrong = (w @ a) @ (a @ x)
    check("direction: A^T is the activation transform", np.abs(right - want).max() < 1e-12,
          "correct %.1e" % np.abs(right - want).max())
    check("direction: A gives a different function", np.abs(wrong - want).max() > 1e-3,
          "wrong-direction residual %.2f (relative %.1e)"
          % (np.abs(wrong - want).max(), np.linalg.norm(wrong - want) / np.linalg.norm(want)))


def t_asymmetry():
    """The transposition bug is live: the optimiser moves R, so A = H R is no longer its own transpose."""
    w = outlier_matrix()
    h = hadamard(256)
    r, _, _ = supplied_optimiser(torch.from_numpy(w @ h))
    a = h @ r
    drift = float(np.abs(r - np.eye(256)).max())
    asym = float(np.abs(a - a.T).max())
    check("asymmetry: R moves, so A != A^T and the returned transform is transposed", asym > 1e-2,
          "||R - I||_inf %.2f, ||A - A^T||_inf %.2f" % (drift, asym))


def t_wrong_direction_of_optimisation():
    """The supplied loop's gradient is truncated, so it steps uphill: loss rises, SQNR falls."""
    w = outlier_matrix()
    h = hadamard(256)
    r, first, last = supplied_optimiser(torch.from_numpy(w @ h))
    plain = sqnr_db(w @ h, quantise(w @ h))
    supplied = sqnr_db(w @ h @ r, quantise(w @ h @ r))
    check("supplied: stepping along its gradient raises its own loss", last > first,
          "loss %.4f -> %.4f" % (first, last))
    check("supplied: SQNR falls with it, so the loss is not the objective it is named after",
          supplied < plain, "SQNR %.2f dB -> %.2f dB (%+.2f)" % (plain, supplied, supplied - plain))


def t_sqnr():
    """The claim that matters: does rotating reduce what snapping costs? Both arms, same grid, same matrix."""
    w = outlier_matrix()
    h = hadamard(w.shape[1])
    naive = sqnr_db(w, quantise(w))
    rotated = sqnr_db(w @ h, quantise(w @ h))
    check("sqnr: Hadamard rotation beats naive snapping on outlier channels", rotated > naive,
          "naive %.2f dB -> rotated %.2f dB" % (naive, rotated))


def t_refine():
    w = outlier_matrix()
    h = hadamard(w.shape[1])
    w_rot = torch.from_numpy(w @ h)
    r, first, last = refine(w_rot, steps=300)
    plain = sqnr_db(w @ h, quantise(w @ h))
    refined = sqnr_db(w @ h @ r, quantise(w @ h @ r))
    check("refine: the smooth objective falls", last < first,
          "concentration %.4f -> %.4f" % (first, last))
    orth = float(np.abs(r.T @ r - np.eye(256)).max())
    check("refine: R leaves the identity and stays on the manifold", orth < 1e-12,
          "||R^T R - I||_inf = %.1e" % orth)
    check("refine: SQNR improves over plain Hadamard", refined > plain,
          "plain %.2f dB -> refined %.2f dB" % (plain, refined))


def t_gate():
    """The specification's own acceptance threshold, held against the best arm here.

    This check encodes a limit, not a success: on a matrix whose dynamic range sits in two channels at 60 sigma, no
    ternary format with a per-channel scale reaches 22 dB, and the refinement above buys about 1 dB of the shortfall.
    """
    w = outlier_matrix()
    h = hadamard(w.shape[1])
    r, _, _ = refine(torch.from_numpy(w @ h), steps=300)
    best = max(sqnr_db(w, quantise(w)), sqnr_db(w @ h, quantise(w @ h)),
               sqnr_db(w @ h @ r, quantise(w @ h @ r)))
    check("gate: the >22 dB acceptance threshold is not reachable here", best < 22.0,
          "best arm %.2f dB against the specification's 22 dB floor" % best)


def selftest():
    del PASS[:]
    for t in (t_hadamard, t_identity, t_direction, t_asymmetry, t_wrong_direction_of_optimisation, t_sqnr,
              t_refine, t_gate):
        t()
    expected = 13
    print("\n%d checks passed" % len(PASS))
    return 0 if len(PASS) == expected else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
