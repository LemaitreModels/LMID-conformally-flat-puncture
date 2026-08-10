"""Oracle-independent ID-quality check: FD constraints on a Cartesian grid (B1 §5).

Interpolate the LM-initial-data spectral conformal factor onto a uniform Cartesian
(evolution-type) grid, assemble the *physical* ADM data
``(gamma_ij = psi^4 delta_ij, K_ij = psi^{-2} Â_ij)``, and measure the
**finite-difference** Hamiltonian and momentum constraint violation as a function
of grid spacing ``h``.  Constraint-satisfying ID should show the violation
converge at the FD order (2nd-order central here) down to the spectral residual
floor — the standard "does it land cleanly on an evolution grid" test.

The Hamiltonian/momentum operators are assembled **generically** from the nodal
metric (FD Christoffels -> FD Ricci; FD divergence of K), i.e. exactly as an
evolution code's constraint monitor does — NO conformal-flatness shortcut is
used in the FD path.  The closed forms  R = -8 psi^{-5} Δpsi  and
D_j K^{ij} = psi^{-10} ∂_j Â^{ij}  are provided separately as independent
continuum cross-checks, and exact self-tests pin the FD order:

  * single Schwarzschild puncture (psi = 1+M/2r, K=0): H_continuum = 0 exactly,
    so H_FD is pure truncation -> O(h^2);
  * transverse Bowen–York Â with K=psi^{-2}Â: the momentum constraint holds for
    ANY psi (D_j K^{ij}=psi^{-10}∂_j Â^{ij}=0), so M_FD -> 0 at O(h^2).

All standalone (numpy); the LM-initial-data field enters only through
``solver.solver_abt.evaluate_field_phys``.
"""

from __future__ import annotations

import numpy as np

from ..solver import source
from ..solver import solver_abt as sa
from ..solver import operators_abt as ops


# --------------------------------------------------------------------------
# Vectorized arbitrary-point field evaluator (the per-point loop in
# solver_abt.evaluate_field_phys is far too slow for N^3 Cartesian points;
# this is the identical 2-D barycentric tensor product, fully vectorized).
# --------------------------------------------------------------------------
def _bary_matrix_vec(x_eval, x_nodes, w):
    """Barycentric interpolation matrix B (M, n): f(x_eval)=B@f(x_nodes)."""
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
    x_nodes = np.asarray(x_nodes, dtype=float)
    w = np.asarray(w, dtype=float)
    diff = x_eval[:, None] - x_nodes[None, :]                  # (M, n)
    hit = np.abs(diff) < 1e-13
    safe = np.where(hit, 1.0, diff)
    terms = w[None, :] / safe
    rows_hit = hit.any(axis=1)
    B = terms / terms.sum(axis=1, keepdims=True)
    if rows_hit.any():                                         # exact node hits
        Bhit = np.where(hit, 1.0, 0.0)
        B[rows_hit] = Bhit[rows_hit]
    return B


def evaluate_u_vec(prob, U, rho, z, b):
    """u(rho, z) at arbitrary points via the 2-D (A,B) barycentric tensor product
    (vectorized twin of ``solver_abt.evaluate_field_phys``)."""
    A_q, B_q = ops.inverse_map(np.asarray(rho, float), np.asarray(z, float), b)
    A_q = np.atleast_1d(A_q).ravel()
    B_q = np.atleast_1d(B_q).ravel()
    wA = sa._bary_weights(prob.A)
    wB = sa._bary_weights(prob.B)
    BA = _bary_matrix_vec(A_q, prob.A, wA)                     # (M, Na+1)
    BB = _bary_matrix_vec(B_q, prob.B, wB)                     # (M, Nb)
    Umat = np.asarray(U).reshape(prob.shape)
    return np.einsum("mi,mj,ij->m", BA, BB, Umat)


# --------------------------------------------------------------------------
# 3-D Bowen–York conformal tensor (punctures on the z-axis, momenta along z)
# --------------------------------------------------------------------------
def _A_single_3d(X, Y, Z, z0, Pz):
    """Single-puncture BY Â^ij (3x3 last axes) on a 3-D Cartesian grid; puncture
    at (0,0,z0), linear momentum (0,0,Pz)."""
    dx, dy, dz = X, Y, Z - z0
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    n = np.stack([dx / r, dy / r, dz / r], axis=-1)            # (...,3)
    Pv = np.array([0.0, 0.0, Pz])
    Pn = n @ Pv                                                # (...,)
    eye = np.eye(3)
    nn = n[..., :, None] * n[..., None, :]
    Pi_nj = np.einsum("k,...j->...kj", Pv, n)                  # P^i n^j
    Pj_ni = np.einsum("...k,j->...kj", n, Pv)                  # n^i P^j
    return (1.5 / r[..., None, None] ** 2) * (
        Pi_nj + Pj_ni - (eye - nn) * Pn[..., None, None])


def A_tensor_3d(X, Y, Z, b, P):
    """Summed BY Â^ij (LM-initial-data convention: A at +b, P_A=(0,0,-P); B at -b, +P)."""
    return _A_single_3d(X, Y, Z, +b, -P) + _A_single_3d(X, Y, Z, -b, +P)


# --------------------------------------------------------------------------
# Cartesian grid + field assembly
# --------------------------------------------------------------------------
def cartesian_grid(L, N):
    """Uniform grid on [-L,L]^3 with N points per axis; returns (x, X, Y, Z, h)."""
    x = np.linspace(-L, L, N)
    h = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return x, X, Y, Z, h


def psi_on_grid(prob, U, sl, X, Y, Z):
    """psi = psi_BL + u on a 3-D Cartesian grid (axisymmetric about z)."""
    rho = np.sqrt(X ** 2 + Y ** 2)
    u = evaluate_u_vec(prob, U, rho.ravel(), Z.ravel(), sl.b).reshape(X.shape)
    psiBL = np.asarray(source.psi_BL_2c(rho, Z, sl.b, sl.m_A, sl.m_B))
    return psiBL + u


# --------------------------------------------------------------------------
# 2nd-order central finite differences (np.gradient interior is 2nd order)
# --------------------------------------------------------------------------
def _d1(f, h, axis):
    return np.gradient(f, h, axis=axis, edge_order=2)


def _grad(f, h):
    """∂_k f  -> array (3, ...)."""
    return np.stack([_d1(f, h, k) for k in range(3)], axis=0)


# --------------------------------------------------------------------------
# Generic FD constraints (assembled as an evolution code would, no shortcut)
# --------------------------------------------------------------------------
def fd_constraints_generic(psi, A_tensor, h):
    """Return ``(H, Mvec)`` from the nodal physical data via generic FD.

    gamma_ij = psi^4 delta_ij (built as a full nodal tensor; off-diagonals are
    exact zeros), K_ij = psi^{-2} Â_ij.  Christoffels and Ricci are assembled by
    finite-differencing the nodal metric; the momentum constraint is the FD
    covariant divergence of K^{ij}.  H, Mvec are returned on the full grid
    (mask/exclude punctures + borders when norming).
    """
    psi = np.asarray(psi, dtype=float)
    f = psi ** 4                                               # gamma diagonal
    finv = 1.0 / f                                             # g^{ii} (diagonal)
    sh = psi.shape

    # metric and its first derivatives (only the diagonal f is nonzero)
    g = np.zeros((3, 3) + sh)
    for a in range(3):
        g[a, a] = f
    d1g = np.zeros((3, 3, 3) + sh)                             # d1g[k,a,b]=∂_k g_ab
    df = _grad(f, h)                                           # (3,...)
    for a in range(3):
        d1g[:, a, a] = df

    # inverse metric (diagonal, algebraic — not FD)
    ginv = np.zeros((3, 3) + sh)
    for a in range(3):
        ginv[a, a] = finv

    # Christoffel: Gamma_low[c,a,b]=1/2(∂_a g_cb+∂_b g_ca-∂_c g_ab)
    Glow = 0.5 * (np.einsum("acb...->cab...", d1g)            # ∂_a g_cb
                  + np.einsum("bca...->cab...", d1g)           # ∂_b g_ca
                  - d1g)                                       # -∂_c g_ab
    # Gamma_up[c,a,b]=g^{cd}Gamma_low[d,a,b]
    Gup = np.einsum("cd...,dab...->cab...", ginv, Glow)

    # dGamma[k,c,a,b] = ∂_k Gamma_up[c,a,b]
    dGamma = np.stack([np.stack([np.stack([np.stack([_d1(Gup[c, a, b], h, k)
                                                     for b in range(3)], 0)
                                           for a in range(3)], 0)
                                 for c in range(3)], 0)
                       for k in range(3)], 0)                  # (k,c,a,b,...)

    # R_ab = ∂_k Γ^k_ab - ∂_b Γ^k_ak + Γ^k_kl Γ^l_ab - Γ^k_bl Γ^l_ak
    term1 = np.einsum("kkab...->ab...", dGamma)                # ∂_k Γ^k_ab
    term2 = np.einsum("bkak...->ab...", dGamma)                # ∂_b Γ^k_ak
    term3 = np.einsum("kkl...,lab...->ab...", Gup, Gup)        # Γ^k_kl Γ^l_ab
    term4 = np.einsum("kbl...,lak...->ab...", Gup, Gup)        # Γ^k_bl Γ^l_ak
    Ric = term1 - term2 + term3 - term4
    Rscalar = np.einsum("ab...,ab...->...", ginv, Ric)         # g^{ab} R_ab

    # extrinsic curvature K_ij = psi^{-2} Â_ij ; K^{ij}=g^{ia}g^{jb}K_ab
    # (index-first layout (a,b,...) so einsum matches ginv/Gup conventions)
    Klow = np.einsum("...ab->ab...", psi[..., None, None] ** (-2.0) * np.asarray(A_tensor))
    Kup = np.einsum("ia...,jb...,ab...->ij...", ginv, ginv, Klow)
    Ktr = np.einsum("ab...,ab...->...", ginv, Klow)            # K = g^{ab}K_ab
    KK = np.einsum("ab...,ab...->...", Klow, Kup)              # K_ij K^ij

    H = Rscalar + Ktr ** 2 - KK

    # momentum: M^i = ∂_j K^{ij} + Γ^i_{jk}K^{kj} + Γ^j_{jk}K^{ik} - g^{ij}∂_j K
    dKup = np.stack([np.stack([np.stack([_d1(Kup[i, j], h, kk)
                                         for j in range(3)], 0)
                               for i in range(3)], 0)
                     for kk in range(3)], 0)                   # (k,i,j,...)
    divK = np.einsum("jij...->i...", dKup)                     # ∂_j K^{ij}
    christ1 = np.einsum("ijk...,kj...->i...", Gup, Kup)        # Γ^i_{jk}K^{kj}
    christ2 = np.einsum("jjk...,ik...->i...", Gup, Kup)        # Γ^j_{jk}K^{ik}
    dKtr = _grad(Ktr, h)                                       # (j,...)
    gradKtr = np.einsum("ij...,j...->i...", ginv, dKtr)        # g^{ij}∂_j K
    Mvec = divK + christ1 + christ2 - gradKtr                  # (i,...)
    return H, Mvec


# --------------------------------------------------------------------------
# Closed-form continuum cross-checks (conformally flat, maximal slicing)
# --------------------------------------------------------------------------
def _laplacian_compact(f, h):
    """Compact 2nd-order central Laplacian (3-point per axis); edges wrap (excluded
    by the border in ``interior_mask``)."""
    lap = np.zeros_like(f)
    for k in range(3):
        lap += (np.roll(f, -1, axis=k) - 2.0 * f + np.roll(f, 1, axis=k)) / h ** 2
    return lap


def fd_hamiltonian_conformal(psi, A2, h):
    """H = -8 psi^{-5} Δ_FD psi - psi^{-12} Â²  (conformal-flat closed form)."""
    psi = np.asarray(psi, dtype=float)
    lap = _laplacian_compact(psi, h)
    return -8.0 * psi ** (-5.0) * lap - psi ** (-12.0) * np.asarray(A2)


def fd_momentum_conformal(psi, A_tensor, h):
    """M^i = psi^{-10} ∂_j Â^{ij}  (conformal-flat closed form; ->0 since Â is
    flat-transverse)."""
    psi = np.asarray(psi, dtype=float)
    A = np.einsum("...ab->ab...", np.asarray(A_tensor))        # (a,b,...)
    divA = np.stack([sum(np.gradient(A[i, j], h, axis=j, edge_order=2)
                         for j in range(3)) for i in range(3)], 0)
    return psi[None, ...] ** (-10.0) * divA


# --------------------------------------------------------------------------
# Masking + norms + convergence driver
# --------------------------------------------------------------------------
def interior_mask(X, Y, Z, h, b, r_excl, border=2):
    """True on interior cells: drop a ``border``-cell frame and balls of physical
    radius ``r_excl`` around each puncture (z=±b on the z-axis)."""
    rA = np.sqrt(X ** 2 + Y ** 2 + (Z - b) ** 2)
    rB = np.sqrt(X ** 2 + Y ** 2 + (Z + b) ** 2)
    m = (rA > r_excl) & (rB > r_excl)
    sl = [slice(border, -border)] * 3
    frame = np.zeros(X.shape, dtype=bool)
    frame[tuple(sl)] = True
    return m & frame


def norms(field, mask):
    """(Linf, L2-RMS) of a scalar field over the mask."""
    v = np.asarray(field)[mask]
    return float(np.max(np.abs(v))), float(np.sqrt(np.mean(v ** 2)))


def vec_norms(vec, mask):
    """(Linf, L2-RMS) of |M| = sqrt(sum_i M^i M^i) over the mask (flat magnitude)."""
    mag = np.sqrt(sum(np.asarray(vec[i]) ** 2 for i in range(3)))
    return norms(mag, mask)
