"""ADM / quasi-local diagnostics for the head-on conformally-flat slice (B1 §3).

Everything here is re-derived from scratch (no formula taken from memory without
a unit-test against an analytic result) and is *additive*: it imports the frozen
P1 solver but does not modify it.

Conventions (conformally flat, maximal slicing, vacuum Bowen–York)
------------------------------------------------------------------
Physical 3-metric  gamma_ij = psi^4 f_ij  (f = flat).  Maximal slicing  K = 0,
so K_ij is trace-free.  The conformal (Bowen–York) tensor Â^ij = source.A2's
tensor enters with the **conformal weight**

    K^ij  = psi^{-10} Â^ij,        K_ij = psi^{-2} Â_ij,     (Â_ij = f f Â^ij)

so that  D_j K^ij = psi^{-10} \bar D_j Â^ij = 0  (momentum constraint holds for
ANY psi, since Â is flat-transverse), and

    K_ij K^ij = psi^{-12} Â_ij Â^ij        (Â² = source.A2_2c, flat contraction),

giving the Hamiltonian constraint  R = K_ij K^ij  ->  Δ psi = -1/8 psi^{-7} Â²
(== solver.source).  These weight relations are validated in
``tests/test_validation_adm.py`` against a direct gamma-raised contraction and
against ``source.A2_2c``.

ADM quantities (flat reference at infinity; gamma -> f, psi -> 1)
----------------------------------------------------------------
    M_ADM = -(1/2pi) ∮_∞ \bar∂_i psi dŜ^i  = (m_A+m_B) + 2 lim_{r->inf} r u_0(r),
    P_ADM^i = (1/8pi) ∮_∞ (K^ij - f^ij K) n_j dS  = sum_X P_X   (= 0, symmetric),
    P_X^i   = (1/8pi) ∮_{S_X} Â^ij n_j dS         (per-puncture Gauss law = P_X).

The per-puncture Gauss integral of the *conformal* Â over a small sphere is
exactly 8pi P_X (derived in ``by_momentum_gauss``); it needs no solve, so it is a
rigorous check on the source + the convention sign.

Quasi-local (AH-free) individual puncture ADM mass (Brandt–Brügmann 1997)
-------------------------------------------------------------------------
    M_X = m_X * (1 + u(x_X) + sum_{Y != X} m_Y/(2 r_XY)),

the bare mass rescaled by the regular part of psi at the puncture.  Reduces to
M_X -> m_X as the holes separate (isolated Schwarzschild puncture).  The genuine
apparent-horizon *area* mass + horizon spins need an AH finder; that is scoped as
a documented stretch (head-on non-spinning -> horizon spin ~ 0).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ..solver import source
from ..solver import solver_abt as sa


# --------------------------------------------------------------------------
# Bowen–York conformal tensor Â^ij (vectorized) and the K_ij <-> Â_ij relation
# --------------------------------------------------------------------------
def A_single_tensor_vec(rho, z, z0, Pvec):
    """Single-puncture BY conformal tensor Â^ij (Cartesian 3x3) at meridian
    points ``(rho, z)`` (phi=0 plane), puncture on the z-axis at ``z=z0`` with
    linear momentum ``Pvec=(Px,Py,Pz)``.

    Returns an array of shape ``(..., 3, 3)``.  Vectorized twin of
    ``source._A_single_tensor`` (cross-checked against it in the test).
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    Pvec = np.asarray(Pvec, dtype=float)
    # displacement d = x - x0, x = (rho, 0, z), x0 = (0,0,z0)
    dx, dy, dz = rho, np.zeros_like(rho), z - z0
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    n = np.stack([dx / r, dy / r, dz / r], axis=-1)        # (...,3)
    Pn = n @ Pvec                                          # (...,)
    eye = np.eye(3)
    nn = n[..., :, None] * n[..., None, :]                 # n^i n^j  (...,3,3)
    Pi_nj = np.einsum("i,...j->...ij", Pvec, n)            # P^i n^j
    Pj_ni = np.einsum("...i,j->...ij", n, Pvec)            # n^i P^j = P^j n^i
    proj = eye - nn                                        # delta^ij - n^i n^j
    A = (3.0 / (2.0 * r[..., None, None] ** 2)) * (
        Pi_nj + Pj_ni - proj * Pn[..., None, None])
    return A


def A_tensor_2c(rho, z, b, P):
    """Summed BY conformal tensor Â^ij = Â_A + Â_B (Cartesian 3x3) at meridian
    points, PARASOL convention (A at +b with P_A=(0,0,-P); B at -b, P_B=(0,0,+P)).
    """
    A_A = A_single_tensor_vec(rho, z, +b, (0.0, 0.0, -P))
    A_B = A_single_tensor_vec(rho, z, -b, (0.0, 0.0, +P))
    return A_A + A_B


def physical_K_lower(psi, A_tensor):
    """K_ij = psi^{-2} Â_ij  (Cartesian; Â_ij = Â^ij since flat).  Shape (...,3,3)."""
    psi = np.asarray(psi, dtype=float)
    return psi[..., None, None] ** (-2.0) * np.asarray(A_tensor, dtype=float)


def KK_physical(psi, A2):
    """K_ij K^ij = psi^{-12} Â²  (Â² = flat contraction source.A2_2c)."""
    psi = np.asarray(psi, dtype=float)
    return psi ** (-12.0) * np.asarray(A2, dtype=float)


# --------------------------------------------------------------------------
# psi on the grid (solved) at arbitrary meridian points
# --------------------------------------------------------------------------
def psi_at(prob, U, rho, z, sl):
    """Full conformal factor psi = psi_BL + u at meridian points ``(rho, z)``."""
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    u = np.asarray(sa.evaluate_field_phys(prob, U, rho, z, sl.b))
    psiBL = np.asarray(source.psi_BL_2c(rho, z, sl.b, sl.m_A, sl.m_B))
    return psiBL + u


# --------------------------------------------------------------------------
# ADM mass — spectral boundary extraction (primary), monopole-tail + surface
# integral (independent cross-checks)
# --------------------------------------------------------------------------
def adm_mass_spectral(prob, U, sl):
    """M_ADM = (m_A + m_B) + 2c, with c read SPECTRALLY at the A=1 (infinity) edge.

    As A->1 the ABT map gives r ~ b/(1-A), so a 1/r-decaying correction behaves as
    u ~ c (1-A)/b for ANY B (higher multipoles decay faster, ~(1-A)^{>1}, and have
    zero A-derivative at A=1).  Hence  ∂_A u|_{A=1} = -c/b  identically in B, and

        M_ADM = (m_A + m_B) - 2b * <∂_A u>_{A=1} .

    Reading the boundary A-derivative spectrally (DA1 @ U at the A=1 node, averaged
    over B) avoids the r-amplification of the field's absolute error that limits the
    monopole-tail fit, and converges spectrally to the true ADM mass (validated to
    ~1e-11 vs TwoPunctures at 64x44).  This is the same construction TwoPunctures
    uses (admMass = mp+mm - 4b*v at A=1).
    """
    DA = np.asarray(prob.DA1)                              # d/dA on the A-nodes
    Umat = np.asarray(U).reshape(prob.shape)               # (Na+1, Nb)
    dUdA_at_inf = (DA @ Umat)[0, :]                        # A[0]=1 (infinity edge)
    c = -sl.b * float(np.mean(dUdA_at_inf))
    return float(sl.M + 2.0 * c)


def adm_mass_monopole(prob, U, sl, radii=None, n_mu=48, n_fit=5):
    """M_ADM = (m_A+m_B) + 2 c,  c = lim_{r->inf} r u_0(r).

    u_0(r) = (1/2)∫_{-1}^{1} u(r, mu) dmu  is the ell=0 (monopole) projection of
    the regular correction; r u_0(r) -> c with a 1/r tail.  We GL-quadrature the
    monopole at several large radii and fit r u_0 = c + d/r + e/r^2 (Richardson),
    returning M_ADM = (m_A+m_B) + 2 c.
    """
    mu, w = np.polynomial.legendre.leggauss(n_mu)          # exact monopole
    if radii is None:
        radii = sl.b * np.array([40.0, 60.0, 90.0, 135.0, 200.0])
    radii = np.asarray(radii, dtype=float)
    ru0 = np.empty(radii.size)
    for k, r in enumerate(radii):
        rho = r * np.sqrt(np.clip(1.0 - mu ** 2, 0.0, None))
        z = r * mu
        u = np.asarray(sa.evaluate_field_phys(prob, U, rho, z, sl.b))
        ru0[k] = r * 0.5 * np.sum(w * u)                   # r * u_0(r)
    # fit r u0 = c + d t + e t^2 with t = 1/r over the n_fit farthest radii
    k = min(n_fit, radii.size)
    rs = radii[-k:]
    y = ru0[-k:]
    t = 1.0 / rs
    deg = 2 if k >= 3 else 1
    coef = np.polyfit(t, y, deg)
    c = coef[-1]                                           # value at t=0 (r->inf)
    return float(sl.M + 2.0 * c)


def adm_mass_surface(prob, U, sl, R=None, n_mu=64):
    """M_ADM = -(1/2pi) ∮_{S_R} ∂_r psi dŜ at a large coordinate radius R.

    Cross-check for ``adm_mass_monopole``.  Uses an analytic ∂_r psi_BL plus a
    centred FD ∂_r u; integrates  -R^2 ∫_{-1}^{1} ∂_r psi dmu  (the 2pi from the
    azimuth cancels the 1/2pi prefactor).  Has an O(1/R) higher-multipole tail,
    so it is read at large R (and may be extrapolated by the caller).
    """
    if R is None:
        R = 120.0 * sl.b
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    sinth = np.sqrt(np.clip(1.0 - mu ** 2, 0.0, None))
    dR = 1e-3 * R
    # centred FD of the full psi in r at fixed mu
    def psi_on_sphere(rr):
        rho = rr * sinth
        z = rr * mu
        u = np.asarray(sa.evaluate_field_phys(prob, U, rho, z, sl.b))
        psiBL = np.asarray(source.psi_BL_2c(rho, z, sl.b, sl.m_A, sl.m_B))
        return psiBL + u
    dpsi_dr = (psi_on_sphere(R + dR) - psi_on_sphere(R - dR)) / (2.0 * dR)
    integral = R ** 2 * np.sum(w * dpsi_dr)                # ∫ ∂_r psi * R^2 dmu
    return float(-integral)                                # -(1/2pi)∮ ; 2pi cancels


# --------------------------------------------------------------------------
# ADM linear momentum — total at infinity and per-puncture Gauss law
# --------------------------------------------------------------------------
def _gauss_Az_nj(center_z, radius, b, P, n_mu=64):
    """∮_{sphere(center_z, radius)} Â^{zj} n_j dS  for the summed BY tensor.

    Axisymmetric: the integrand's z-component is azimuth-independent, so the
    azimuth gives a factor 2pi and we GL-quadrature in mu' = cos(theta') about
    the sphere centre.  Returns the scalar surface integral (= 8pi P_z when the
    sphere encloses exactly one puncture of z-momentum P_z).
    """
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    sinth = np.sqrt(np.clip(1.0 - mu ** 2, 0.0, None))
    rho = radius * sinth
    z = center_z + radius * mu
    A = A_tensor_2c(rho, z, b, P)                          # (n_mu,3,3)
    # outward normal of THIS sphere: n = (sin, 0, cos)
    n = np.stack([sinth, np.zeros_like(mu), mu], axis=-1)  # (n_mu,3)
    Az_j_nj = np.einsum("ki,ki->k", A[:, 2, :], n)         # Â^{z j} n_j
    # ∮ = 2pi ∫_{-1}^{1} (Â^{zj}n_j) radius^2 dmu
    return float(2.0 * np.pi * radius ** 2 * np.sum(w * Az_j_nj))


def by_momentum_gauss(b, P, which="A", radius=None, n_mu=64):
    """Per-puncture ADM linear momentum P_X^z via the Gauss law
    (1/8pi) ∮_{S_X} Â^{zj} n_j dS = P_X^z  (analytic; no solve needed).

    ``which`` in {"A","B"}; sphere radius defaults to b/2 (encloses only that
    puncture).  Expected: P_A^z = -P, P_B^z = +P.
    """
    if radius is None:
        radius = 0.5 * b
    center_z = +b if which == "A" else -b
    return _gauss_Az_nj(center_z, radius, b, P, n_mu) / (8.0 * np.pi)


def adm_linear_momentum_total(b, P, R=None, n_mu=96):
    """Total ADM linear momentum P_ADM^z = (1/8pi)∮_∞ Â^{zj} n_j dS.

    For the symmetric head-on pair this is 0 (P_A + P_B = 0); computed from the
    analytic Â at a large coordinate radius (psi^{-10} -> 1 at infinity).
    """
    if R is None:
        R = 200.0 * b
    return _gauss_Az_nj(0.0, R, b, P, n_mu) / (8.0 * np.pi)


# --------------------------------------------------------------------------
# Quasi-local (AH-free) individual puncture ADM masses
# --------------------------------------------------------------------------
def puncture_adm_mass(prob, U, sl, which="A"):
    """Individual puncture ADM mass  M_X = m_X (1 + u(x_X) + sum_{Y!=X} m_Y/(2 r_XY)).

    (Brandt–Brügmann 1997: the bare mass rescaled by the regular part of psi at
    the puncture.)  r_AB = 2b.  Reduces to m_X as b -> inf.
    """
    if which == "A":
        m_self, m_other, z_self = sl.m_A, sl.m_B, +sl.b
    else:
        m_self, m_other, z_self = sl.m_B, sl.m_A, -sl.b
    # regular part of psi at the puncture = 1 + u(x_X) + m_other/(2 r_XY);
    # u is smooth (C^inf) at the puncture, evaluated via the field interpolant.
    u_val = float(np.asarray(sa.evaluate_field_phys(prob, U, np.array([0.0]),
                                                    np.array([z_self]), sl.b))[0])
    reg = 1.0 + u_val + m_other / (2.0 * (2.0 * sl.b))
    return float(m_self * reg)
