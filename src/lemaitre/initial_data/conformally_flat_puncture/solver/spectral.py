"""LM-initial-data — 1-D spectral primitives (§3.1 of plan.md).

Standalone: depends only on numpy (matrix construction) and jax/jax.numpy.
Re-derives every building block from scratch (Chebyshev D-matrix, algebraic
radial map + chain-rule derivatives, even-Legendre analysis/synthesis).

All public arrays are float64 jax arrays; we build with numpy then convert.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # mandatory: float64 everywhere

import numpy as np
import jax.numpy as jnp


# --------------------------------------------------------------------------
# Chebyshev–Gauss–Lobatto differentiation (Trefethen, *Spectral Methods*)
# --------------------------------------------------------------------------
def cheb(N: int):
    """Chebyshev–Gauss–Lobatto nodes and differentiation matrix on [-1, 1].

    Nodes ``x_k = cos(k*pi/N)``, ``k = 0..N`` (descending: x[0]=+1, x[N]=-1).
    Returns ``(x, D)`` with ``D`` the (N+1)x(N+1) first-derivative matrix.
    The diagonal uses the negative-sum trick for numerical accuracy.
    """
    if N == 0:
        return jnp.array([1.0]), jnp.zeros((1, 1))
    k = np.arange(N + 1)
    x = np.cos(np.pi * k / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** k
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))   # off-diagonal
    D = D - np.diag(D.sum(axis=1))                    # diagonal: negative-sum trick
    return jnp.asarray(x), jnp.asarray(D)


# --------------------------------------------------------------------------
# Algebraic radial map  [-1, 1) -> [0, inf)  with chain-rule derivatives
# --------------------------------------------------------------------------
def radial_grid(N: int, L: float):
    """Radial grid and *physical* radial derivative matrices.

    Map ``r(x) = L (1+x)/(1-x)`` : x=-1 <-> r=0, x->+1 <-> r->inf.
    With CGL ordering, index 0 is r=inf and index N is r=0; both are dropped
    from the interior (they become BC rows in operators.py).

    Chain rule (derived in plan.md §3.1):
        dx/dr   = (1-x)^2 / (2L)          (= 0 at x=+1)
        d2x/dr2 = -(1-x)^3 / (2L^2)       (= 0 at x=+1)
        Dr  = diag(dx/dr) @ D
        Dr2 = diag(dx/dr)^2 @ D2 + diag(d2x/dr2) @ D

    Returns ``(r, Dr, Dr2, x)`` as jax arrays. ``r[0] = +inf``, ``r[N] = 0``.
    """
    x_j, D_j = cheb(N)
    x = np.asarray(x_j)
    D = np.asarray(D_j)
    D2 = D @ D

    r = np.empty(N + 1)
    r[0] = np.inf                                    # x=+1  -> r=inf
    r[-1] = 0.0                                       # x=-1  -> r=0
    r[1:-1] = L * (1.0 + x[1:-1]) / (1.0 - x[1:-1])

    dxdr = (1.0 - x) ** 2 / (2.0 * L)                 # finite everywhere
    d2xdr2 = -((1.0 - x) ** 3) / (2.0 * L ** 2)        # finite everywhere

    Dr = dxdr[:, None] * D
    Dr2 = (dxdr ** 2)[:, None] * D2 + d2xdr2[:, None] * D
    return jnp.asarray(r), jnp.asarray(Dr), jnp.asarray(Dr2), jnp.asarray(x)


# --------------------------------------------------------------------------
# Even-Legendre angular transforms  (modal Legendre <-> nodal GL)
# --------------------------------------------------------------------------
def even_ells(L_theta: int) -> np.ndarray:
    """Even mode degrees ell = 0, 2, ..., 2(L_theta-1)."""
    return 2 * np.arange(L_theta)


def _legendre_P(ell: int, mu: np.ndarray) -> np.ndarray:
    """Legendre polynomial P_ell evaluated at mu (numpy, no scipy)."""
    coef = np.zeros(ell + 1)
    coef[ell] = 1.0
    return np.polynomial.legendre.legval(mu, coef)


def legendre_transforms(L_theta: int, J: int | None = None):
    """Gauss–Legendre nodes/weights and even-mode synthesis/analysis matrices.

    Modes: even ell in {0, 2, ..., 2(L_theta-1)}  (L_theta of them).
    Nodes: J GL nodes.  The Gauss–Legendre analysis matrix is an *exact*
    inverse of synthesis on the even-mode subspace iff the quadrature is exact
    for products P_a P_b up to degree 2*ell_max = 4(L_theta-1), i.e. J >= 2L_theta-1.
    We therefore default to ``J = 2*L_theta`` (the plan's J=L_theta+2 fails the
    A@S=I gate for L_theta>=4; see report).  De-aliases the nonlinear source too.

    Synthesis  S[j, a] = P_{ell_a}(mu_j)                 (J x L_theta)
    Analysis   A[a, j] = (2 ell_a + 1)/2 * w_j P_{ell_a}(mu_j)  (L_theta x J)

    Returns ``(mu, w, ells, S, A)`` (jax arrays for mu/w/S/A, numpy ells).
    """
    if J is None:
        J = 2 * L_theta
    ells = even_ells(L_theta)
    ell_max = int(ells[-1]) if L_theta > 0 else 0
    if 2 * J - 1 < 2 * ell_max:
        raise ValueError(
            f"J={J} too small for exact even-mode quadrature (need J>={ell_max+1} "
            f"for ell_max={ell_max}); A@S would not be identity."
        )
    mu, w = np.polynomial.legendre.leggauss(J)
    S = np.empty((J, L_theta))
    A = np.empty((L_theta, J))
    for a, ell in enumerate(ells):
        Pell = _legendre_P(int(ell), mu)
        S[:, a] = Pell
        A[a, :] = (2 * ell + 1) / 2.0 * w * Pell
    return jnp.asarray(mu), jnp.asarray(w), ells, jnp.asarray(S), jnp.asarray(A)


def legendre_P_eval(ell: int, mu) -> np.ndarray:
    """Public Legendre P_ell at arbitrary mu (numpy)."""
    return _legendre_P(int(ell), np.asarray(mu, dtype=float))


# --------------------------------------------------------------------------
# Barycentric Lagrange interpolation on Chebyshev–Gauss–Lobatto nodes
# (Berrut & Trefethen 2004).  Reused by the radial field evaluator (M2) and
# the parameter-space interpolant (M3) — both live on CGL nodes.
# --------------------------------------------------------------------------
def cgl_bary_weights(N: int) -> np.ndarray:
    """Barycentric weights for CGL nodes x_k=cos(k pi/N): (-1)^k, halved at ends."""
    w = (-1.0) ** np.arange(N + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def bary_interp_matrix(x_eval, x_nodes, w) -> np.ndarray:
    """Barycentric interpolation matrix B: f(x_eval) = B @ f(x_nodes).

    ``x_eval`` (M,), ``x_nodes`` (N+1,), ``w`` (N+1,) the bary weights.
    Returns B of shape (M, N+1).  Rows that hit a node exactly are set to the
    corresponding unit (delta) row.
    """
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
    x_nodes = np.asarray(x_nodes, dtype=float)
    w = np.asarray(w, dtype=float)
    M = x_eval.shape[0]
    Np1 = x_nodes.shape[0]
    B = np.zeros((M, Np1))
    for i, xe in enumerate(x_eval):
        diff = xe - x_nodes
        hit = np.isclose(diff, 0.0, atol=1e-14)
        if np.any(hit):
            B[i, np.argmax(hit)] = 1.0
        else:
            terms = w / diff
            B[i, :] = terms / terms.sum()
    return B
