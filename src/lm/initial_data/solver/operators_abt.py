"""LM-initial-data-2C — single-patch ABT / prolate-spheroidal grid + Laplacian (M2-A).

The Ansorg-Brügmann-Tichy axisymmetric coordinates, re-derived from scratch and
recognised to be **prolate-spheroidal coordinates with a compactified radial
coordinate**.  With punctures (foci) at z=±b,

    rho = b * 2A/(1-A^2) * sqrt(1-B^2),     z = b * (1+A^2)B/(1-A^2),
    A in [0,1], B in [-1,1],

equivalently the prolate-spheroidal map ``z = b xi eta``, ``rho = b
sqrt((xi^2-1)(1-eta^2))`` with ``xi = (1+A^2)/(1-A^2) >= 1`` (radial, A=tanh(psi/2)
compactified) and ``eta = B`` (angle).  Coordinate edges:

    A=1 edge        : spatial infinity (a full edge)     -> Dirichlet u=0
    A=0 edge        : inner axis |z|<=b (z=bB)           -> Neumann d/dA u=0
    B=+-1 edges     : outer axis |z|>=b (poles)          -> interior GL B nodes avoid them
    corners (A=0,B=+-1) : the two punctures (analytic in (A^2, 1-B); see report)

Because prolate-spheroidal coordinates are **orthogonal**, the axisymmetric
Laplacian has **no cross term**:

    Delta u = (1-A^2)^2/(b^2 Den) [ (1-A^2)^2/4 u_AA + (1-A^2)^2/(4A) u_A
                                    + (1-B^2) u_BB - 2B u_B ],
    Den = (1+A^2)^2 - B^2 (1-A^2)^2.

This is a single dense 2-D collocation operator (no mortar), spectrally
convergent for both punctures simultaneously.  Standalone: numpy + jax only.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from . import spectral   # reused verbatim: spectral.cheb for the Lobatto A-matrix


# --------------------------------------------------------------------------
# 1-D node sets and differentiation matrices
# --------------------------------------------------------------------------
def cheb01(N):
    """Chebyshev-Gauss-Lobatto nodes on [0,1] and the d/dA matrix.

    Built from ``spectral.cheb`` on [-1,1] (descending: x[0]=1), mapped by
    A=(x+1)/2 so A[0]=1 (infinity edge), A[N]=0 (inner-axis edge).  d/dA = 2 d/dx.
    """
    x, D = spectral.cheb(N)
    A = 0.5 * (np.asarray(x) + 1.0)
    DA = 2.0 * np.asarray(D)
    return A, DA


def gl_nodes_diffmat(N):
    """Gauss-Legendre nodes (interior, symmetric about 0) and a barycentric
    nodal differentiation matrix on [-1,1].  ``N`` nodes (degree N-1)."""
    B, _ = np.polynomial.legendre.leggauss(N)
    w = np.ones(N)
    for j in range(N):
        d = B[j] - B
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (B[i] - B[j])
        D[i, i] = -np.sum(D[i, :])
    return B, D


# --------------------------------------------------------------------------
# the ABT / prolate map and its closed-form inverse
# --------------------------------------------------------------------------
def abt_map(A, B, b):
    """(A,B) -> (rho, z).  A=1 maps to infinity (rho,z -> inf)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    den = 1.0 - A ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = b * 2.0 * A / den * np.sqrt(np.clip(1.0 - B ** 2, 0.0, None))
        z = b * (1.0 + A ** 2) * B / den
    return rho, z


def inverse_map(rho, z, b):
    """(rho, z) -> (A, B) via prolate xi,eta and A = sqrt((xi-1)/(xi+1)).

    xi = (r1 + r2)/(2b), eta = (z>0 ? +1 : -1)*... computed from r1,r2:
    r1 = dist to (0,0,+b), r2 = dist to (0,0,-b); z = b xi eta => eta = (r1 - r2)/(-2b)
    (since r1 - r2 = -2 b eta for the standard prolate relation).  Returns (A, B).
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    r1 = np.hypot(rho, z - b)        # to +b
    r2 = np.hypot(rho, z + b)        # to -b
    xi = (r1 + r2) / (2.0 * b)
    eta = (r2 - r1) / (2.0 * b)      # = B ; z = b xi eta is +ve for z>0
    A = np.sqrt(np.clip((xi - 1.0) / (xi + 1.0), 0.0, 1.0))
    return A, np.clip(eta, -1.0, 1.0)


# --------------------------------------------------------------------------
# the prolate-spheroidal axisymmetric Laplacian (dense 2-D collocation matrix)
# --------------------------------------------------------------------------
def build_grid(Na, Nb):
    """Return ``(A, B, DA1, DB1)``: Lobatto A on [0,1], GL interior B on [-1,1].

    ``A`` has Na+1 nodes (A[0]=1 infinity, A[Na]=0 inner axis); ``B`` has Nb GL
    nodes (interior).  ``DA1, DB1`` are the 1-D differentiation matrices.
    """
    A, DA1 = cheb01(Na)
    B, DB1 = gl_nodes_diffmat(Nb)
    return A, B, DA1, DB1


def _coeffs(Af, Bf, b):
    """Prolate Laplacian coefficients (alpha,p,gamma,q) at flattened nodes,
    with edge rows (A=0, A=1) given finite dummies (they become BC rows)."""
    edge = (Af <= 1e-14) | (Af >= 1.0 - 1e-14)
    Asafe = np.where(edge, 0.5, Af)
    oa2 = 1.0 - Asafe ** 2
    Den = (1.0 + Asafe ** 2) ** 2 - Bf ** 2 * oa2 ** 2
    pref = oa2 ** 2 / (b ** 2 * Den)
    alpha = pref * oa2 ** 2 / 4.0
    pcoef = pref * oa2 ** 2 / (4.0 * Asafe)
    gamma = pref * (1.0 - Bf ** 2)
    qcoef = pref * (-2.0 * Bf)
    for c in (alpha, pcoef, gamma, qcoef):
        c[edge] = 0.0
    return alpha, pcoef, gamma, qcoef


def laplacian_matrix(A, B, DA1, DB1, b):
    """Dense prolate-spheroidal Laplacian (no cross term) + node coordinates.

    Returns ``(Lap, rho, z, Af, Bf, DA, DB)`` where Lap is the
    ((Na+1)*Nb)x((Na+1)*Nb) operator on the flattened field U[i*Nb + j], and
    DA, DB are the 2-D first-derivative matrices (used for BC rows).
    """
    Na1, Nb1 = A.size, B.size
    IA, IB = np.eye(Na1), np.eye(Nb1)
    DA = np.kron(np.asarray(DA1), IB)
    DB = np.kron(IA, np.asarray(DB1))
    DAA = DA @ DA
    DBB = DB @ DB
    AA, BB = np.meshgrid(A, B, indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    rho, z = abt_map(Af, Bf, b)
    alpha, pcoef, gamma, qcoef = _coeffs(Af, Bf, b)
    Lap = (alpha[:, None] * DAA + pcoef[:, None] * DA
           + gamma[:, None] * DBB + qcoef[:, None] * DB)
    return Lap, rho, z, Af, Bf, DA, DB


def solve_equilibrated(M, rhs):
    """Solve M x = rhs with row equilibration (scale each row by 1/max|row|).

    The prolate operator's rows span a huge dynamic range (the (1-A^2)^4 factor
    is tiny near the A=1 infinity edge, the 1/A factor large near A=0), giving a
    raw condition number ~1e13-1e15.  Row equilibration is an exact rescaling of
    the equations (the unknown is unchanged) that collapses cond to ~1e4 and
    takes the solve to machine precision.
    """
    M = np.asarray(M)
    scale = np.max(np.abs(M), axis=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    return np.linalg.solve(M / scale[:, None], np.asarray(rhs) / scale)


def apply_bcs(Lap, A, B, DA):
    """Row-replace BC edges; return ``(M, interior_mask)``.

    A=1 (i=0): Dirichlet u=0 (identity row).
    A=0 (i=Na): Neumann d/dA u=0 (the DA row).
    interior_mask is True on PDE rows (RHS/source applied there).
    """
    Na1, Nb1 = A.size, B.size
    M = np.array(Lap, dtype=float)
    interior = np.ones(Na1 * Nb1, dtype=bool)
    for j in range(Nb1):
        r_inf = 0 * Nb1 + j               # A=1 (infinity)
        M[r_inf, :] = 0.0
        M[r_inf, r_inf] = 1.0
        interior[r_inf] = False
        r_ax = (Na1 - 1) * Nb1 + j        # A=0 (inner axis)
        M[r_ax, :] = DA[r_ax, :]
        interior[r_ax] = False
    return M, interior
