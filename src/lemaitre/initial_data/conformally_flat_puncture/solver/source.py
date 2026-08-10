"""LM-initial-data-2C — two-centre Bowen–York source, two-puncture Brill–Lindquist
background (Phase A, plan.md §12.7).

Two punctures on the z-axis at ``z = +b`` (puncture A) and ``z = -b`` (puncture B),
masses ``m_A, m_B``, carrying *anti-parallel infall* linear momenta

    P_A = (0, 0, -P)      (A at +b moves toward the origin)
    P_B = (0, 0, +P)      (B at -b moves toward the origin)

The two-puncture conformal factor is the linear superposition

    psi_BL = 1 + m_A/(2 r_A) + m_B/(2 r_B),     r_X = |x - x_X|,

and the conformal extrinsic curvature is the *sum* of two single-puncture
Bowen–York linear-momentum tensors, each centred on its own puncture,

    Â^{ij} = Â_A^{ij} + Â_B^{ij},
    Â_X^{ij} = (3 / 2 r_X^2) [ P_X^i n_X^j + P_X^j n_X^i
                               - (delta^{ij} - n_X^i n_X^j)(P_X . n_X) ],
    n_X^i = (x - x_X)^i / r_X.

The sum is transverse (∂_j Â^{ij} = 0) for any separation — verified to ~1e-15
in ``A2_2c``'s companion test — so the **momentum constraint is analytic** and we
solve only the Hamiltonian (Lichnerowicz) constraint

    Δu = -1/8 (psi_BL + u)^{-7} Â_{ij}Â^{ij},     u -> 0 at infinity.

The contraction Â² = Â_{ij}Â^{ij} of the *sum* carries a genuine **cross term**
2 Â_A:Â_B that is O(1) (≈ the on-axis diagonal at the midpoint, growing as
b -> 0, never vanishing). It couples the angular modes densely — this is the
entire reason two-centre data is a genuinely coupled problem, and it is derived
and unit-tested directly here.

All public arrays are float64; everything is a pure function of arrays.  The
functions take **cylindrical (rho, z)** so they are grid-agnostic (any patch
supplies its own node positions).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # mandatory: float64 everywhere

import numpy as np
import jax.numpy as jnp


# --------------------------------------------------------------------------
# Geometry helpers (cylindrical rho>=0, axis at rho=0; punctures at z=±b)
# --------------------------------------------------------------------------
def radii(rho, z, b):
    """Distances ``r_A = |x - (0,0,+b)|`` and ``r_B = |x - (0,0,-b)|``.

    Returns ``(r_A, r_B)`` with the same shape as the broadcast of (rho, z).
    """
    rho = jnp.asarray(rho)
    z = jnp.asarray(z)
    r_A = jnp.sqrt(rho ** 2 + (z - b) ** 2)
    r_B = jnp.sqrt(rho ** 2 + (z + b) ** 2)
    return r_A, r_B


# --------------------------------------------------------------------------
# Two-puncture Brill–Lindquist background and its parameter derivatives
# --------------------------------------------------------------------------
def psi_BL_2c(rho, z, b, m_A, m_B):
    """Two-puncture conformal factor ``1 + m_A/(2 r_A) + m_B/(2 r_B)``.

    Only ever evaluated away from the punctures (their r=0 rows are BC rows /
    masked).  Each 1/r_X singularity is carried analytically here, so the
    correction ``u`` is smooth (C^∞) at *both* punctures.
    """
    r_A, r_B = radii(rho, z, b)
    return 1.0 + m_A / (2.0 * r_A) + m_B / (2.0 * r_B)


def dpsiBL_dmA(rho, z, b):
    """∂ψ_BL/∂m_A = 1/(2 r_A)."""
    r_A, _ = radii(rho, z, b)
    return 1.0 / (2.0 * r_A)


def dpsiBL_dmB(rho, z, b):
    """∂ψ_BL/∂m_B = 1/(2 r_B)."""
    _, r_B = radii(rho, z, b)
    return 1.0 / (2.0 * r_B)


# --------------------------------------------------------------------------
# Summed Bowen–York Â²  (the crux source; closed form via dot products)
# --------------------------------------------------------------------------
def _single_AA(r_X, a_X, P2):
    """Single-puncture self-contraction Â_X:Â_X = (9 / 2 r_X^4)(P^2 + 2 a_X^2).

    ``a_X = P_X . n_X`` is the radial momentum component; ``P2 = |P_X|^2``.
    Reduces to the familiar (9 P^2 / 2 r^4)(1 + 2 mu^2) when P is along z
    (a_X = P mu).  Derived in the module docstring / report.
    """
    return 9.0 / (2.0 * r_X ** 4) * (P2 + 2.0 * a_X ** 2)


def _cross_MAMB(p, gamma, alpha, beta, a_A, a_B):
    """The cross contraction M_A:M_B of the two raw BY shape tensors.

    With M_X^{ij} = P_X^i n_X^j + P_X^j n_X^i - (delta^{ij} - n_X^i n_X^j) a_X,

        M_A:M_B = 2 p gamma + 2 alpha beta - 3 a_A a_B
                  + 2 a_B alpha gamma + 2 a_A beta gamma + a_A a_B gamma^2,

    where  p = P_A·P_B,  gamma = n_A·n_B,  alpha = P_A·n_B,  beta = P_B·n_A,
    a_A = P_A·n_A,  a_B = P_B·n_B.  (Reduces to 2P^2 + 4a^2 when A == B,
    matching ``_single_AA``'s ``P2 + 2a^2`` ×2 — verified in the docstring.)
    Then Â_A:Â_B = (9 / 4 r_A^2 r_B^2) M_A:M_B.
    """
    return (2.0 * p * gamma + 2.0 * alpha * beta - 3.0 * a_A * a_B
            + 2.0 * a_B * alpha * gamma + 2.0 * a_A * beta * gamma
            + a_A * a_B * gamma ** 2)


def _dotproducts(rho, z, b, P):
    """All six dot products needed for the summed-BY contraction at (rho, z).

    Works in the φ=0 meridian plane (axisymmetry): x=(rho,0,z),
    x-A=(rho,0,z-b), x-B=(rho,0,z+b); P_A=(0,0,-P), P_B=(0,0,+P).
    Returns ``(r_A, r_B, p, gamma, alpha, beta, a_A, a_B, P2)``.
    """
    rho = jnp.asarray(rho)
    z = jnp.asarray(z)
    r_A, r_B = radii(rho, z, b)
    sA = z - b                      # (x-A)_z
    sB = z + b                      # (x-B)_z
    P2 = P ** 2
    a_A = -P * sA / r_A             # P_A·n_A = (-P)(sA)/r_A
    a_B = P * sB / r_B             # P_B·n_B = (+P)(sB)/r_B
    alpha = -P * sB / r_B           # P_A·n_B = (-P)(sB)/r_B   (= -a_B)
    beta = P * sA / r_A             # P_B·n_A = (+P)(sA)/r_A   (= -a_A)
    # n_A·n_B = [rho^2 + (z-b)(z+b)] / (r_A r_B)
    gamma = (rho ** 2 + sA * sB) / (r_A * r_B)
    p = -P2                          # P_A·P_B = (0,0,-P)·(0,0,+P)
    return r_A, r_B, p, gamma, alpha, beta, a_A, a_B, P2


def A2_2c(rho, z, b, P):
    """Summed Bowen–York Â_{ij}Â^{ij} at cylindrical (rho, z).

    Â² = Â_A:Â_A + Â_B:Â_B + 2 Â_A:Â_B  (self_A + self_B + cross).
    Pure function of arrays; same shape as broadcast(rho, z).
    """
    r_A, r_B, p, gamma, alpha, beta, a_A, a_B, P2 = _dotproducts(rho, z, b, P)
    self_A = _single_AA(r_A, a_A, P2)
    self_B = _single_AA(r_B, a_B, P2)
    cross = (9.0 / (2.0 * r_A ** 2 * r_B ** 2)) * _cross_MAMB(
        p, gamma, alpha, beta, a_A, a_B)            # = 2 Â_A:Â_B
    return self_A + self_B + cross


def A2_2c_parts(rho, z, b, P):
    """Same as ``A2_2c`` but returns ``(self_A, self_B, cross)`` separately.

    Used by the M0-A puncture-regularity check (the self-term -> r_A^3 while the
    cross-term -> r_A^5 once multiplied by psi_BL^{-7} ~ r_A^7).
    """
    r_A, r_B, p, gamma, alpha, beta, a_A, a_B, P2 = _dotproducts(rho, z, b, P)
    self_A = _single_AA(r_A, a_A, P2)
    self_B = _single_AA(r_B, a_B, P2)
    cross = (9.0 / (2.0 * r_A ** 2 * r_B ** 2)) * _cross_MAMB(
        p, gamma, alpha, beta, a_A, a_B)
    return self_A, self_B, cross


# --------------------------------------------------------------------------
# Raw Bowen–York tensor contraction (independent oracle for the M0-A test)
# --------------------------------------------------------------------------
def _A_single_tensor(x_vec, x0_vec, P_vec):
    """Single-puncture BY tensor Â_X^{ij} (3x3 Cartesian) at point x_vec."""
    d = np.asarray(x_vec, dtype=float) - np.asarray(x0_vec, dtype=float)
    r = np.linalg.norm(d)
    n = d / r
    Pn = float(np.asarray(P_vec, dtype=float) @ n)
    proj = np.eye(3) - np.outer(n, n)
    return (3.0 / (2.0 * r ** 2)) * (
        np.outer(P_vec, n) + np.outer(n, P_vec) - proj * Pn)


def A2_raw_2c_at_point(x_vec, b, P):
    """Direct Â_{ij}Â^{ij} of the *summed* BY tensor at a Cartesian point.

    Builds Â = Â_A + Â_B as full 3x3 tensors (A at +b, P_A=-P ẑ; B at -b,
    P_B=+P ẑ) and returns sum_ij Â^{ij}Â^{ij}.  Independent of the closed
    form ``A2_2c`` — used only to numerically verify it.
    """
    A = np.array([0.0, 0.0, b])
    B = np.array([0.0, 0.0, -b])
    P_A = np.array([0.0, 0.0, -P])
    P_B = np.array([0.0, 0.0, P])
    Aij = (_A_single_tensor(x_vec, A, P_A)
           + _A_single_tensor(x_vec, B, P_B))
    return float(np.sum(Aij * Aij))


def divergence_raw_2c_at_point(x_vec, b, P, h=1e-5):
    """∂_j Â^{ij} of the summed BY tensor at x_vec (central finite difference).

    Returns the 3-vector ‖∂_j Â^{ij}‖ components; should be ~0 (transverse) for
    every separation.  Used by the M0-A transversality bonus check.
    """
    A = np.array([0.0, 0.0, b])
    B = np.array([0.0, 0.0, -b])
    P_A = np.array([0.0, 0.0, -P])
    P_B = np.array([0.0, 0.0, P])

    def Aij(x):
        return (_A_single_tensor(x, A, P_A) + _A_single_tensor(x, B, P_B))

    x_vec = np.asarray(x_vec, dtype=float)
    div = np.zeros(3)
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        dA = (Aij(x_vec + e) - Aij(x_vec - e)) / (2.0 * h)   # ∂_j Â^{ij}
        div += dA[:, j]
    return div


# ==========================================================================
# Aligned-spin Bowen–York extension  (Milestone P2)
# ==========================================================================
# Each puncture additionally carries a spin ``S_X`` along the collision (z)
# axis.  The full conformal extrinsic curvature is the SUM of the linear-
# momentum tensors (above) and the per-puncture spin tensors
#
#     Â_{S,X}^{ij} = (3 / r_X^3) (eps^{ikl} S_{X,k} n_{X,l} n_X^j
#                                 + eps^{jkl} S_{X,k} n_{X,l} n_X^i)
#                  = (3 / r_X^3) (v_X^i n_X^j + v_X^j n_X^i),   v_X = S_X × n_X,
#
# so the total is Â = Â_{P,A} + Â_{P,B} + Â_{S,A} + Â_{S,B}.  Each spin piece is
# transverse and traceless on its own, so the momentum constraint stays analytic
# (∂_j Â^{ij} = 0 for the full sum — verified to machine precision by autodiff).
#
# DERIVATION of the full contraction Â² = Â_{ij}Â^{ij} (re-derived from scratch;
# verified vs a raw 3×3 tensor contraction ≤1e-12 AND sympy = 0 exactly — see
# tests/test_source_spin.py):  for ALIGNED spins (S_X ∥ z) and ON-AXIS momenta
# (P_X ∥ z), the spin vector v_X = S_X × n_X is purely AZIMUTHAL (⊥ the meridian
# plane that contains P_X, n_X and z).  Therefore EVERY momentum–spin contraction
# vanishes IDENTICALLY:
#   * the per-puncture "P×S" cross 2 Â_{P,X}:Â_{S,X} ∝ (P_X × S_X)·n_X = 0
#     (P_X ∥ S_X ⇒ P_X × S_X = 0);
#   * the cross-puncture ones Â_{P,X}:Â_{S,Y} = 0 because z, n_A, n_B are coplanar
#     (both punctures on the z-axis) ⇒ (n_A × n_B)·z = 0.
# Hence the spin enters Â² ONLY through the self-spin and spin–spin-cross terms,
# and the momentum sector (``A2_2c``) is UNCHANGED:
#
#   Δ(Â²)_spin = 18 S_A² ρ²/r_A^8 + 18 S_B² ρ²/r_B^8
#                + 36 S_A S_B ρ²(ρ² + s_A s_B)/(r_A^5 r_B^5),   s_X = z ∓ b.
#
# (Using 1−μ_X² = ρ²/r_X² and γ−μ_Aμ_B = ρ²/(r_A r_B) for the closed-form simpli-
# fication; μ_X = (z∓b)/r_X.)  Every term carries an explicit ρ² ⇒ the aligned-
# spin source VANISHES ON THE z-AXIS (Â_S² = 18 S² sin²θ/r^6), as it must.
#
# PARITY (z→−z, the load-bearing R1 check):  Δ(Â²)_spin is even in z iff
# |S_A| = |S_B| (the self-spin terms swap r_A↔r_B); the spin–spin cross is ALWAYS
# even; the (vanishing) P×S term cannot break parity.  So the full source is z-even
# iff (m_A = m_B) AND (|S_A| = |S_B|).  Unequal spin magnitudes (e.g. a single
# spinning puncture) GENUINELY populate odd-ℓ angular modes — these are
# representable by the production full Gauss–Legendre angular grid (no even-only
# restriction in the ABT two-centre base), so no basis change is needed.
# --------------------------------------------------------------------------
def A2_spin_extra(rho, z, b, S_A, S_B):
    """Aligned-spin addition Δ(Â²)_spin to the momentum-only ``A2_2c`` (closed form).

    Pure function of arrays; same shape as broadcast(rho, z).  Exactly ``0.0``
    when ``S_A == S_B == 0`` (every term carries a spin factor), so it can be
    added to ``A2_2c`` without changing the no-spin result.
    """
    rho = jnp.asarray(rho)
    z = jnp.asarray(z)
    r_A, r_B = radii(rho, z, b)
    sA = z - b
    sB = z + b
    rho2 = rho ** 2
    self_A = 18.0 * S_A ** 2 * rho2 / r_A ** 8
    self_B = 18.0 * S_B ** 2 * rho2 / r_B ** 8
    cross = 36.0 * S_A * S_B * rho2 * (rho2 + sA * sB) / (r_A ** 5 * r_B ** 5)
    return self_A + self_B + cross


def A2_2c_spin(rho, z, b, P, S_A=0.0, S_B=0.0):
    """Summed Bowen–York Â² with aligned spins ``S_A, S_B`` along z (full source).

    Â² = [momentum-only A2_2c]  +  Δ(Â²)_spin.  Reduces **bit-for-bit** to
    ``A2_2c(rho, z, b, P)`` when ``S_A == S_B == 0`` (short-circuit), so the
    P1 no-spin path is reproduced exactly.
    """
    base = A2_2c(rho, z, b, P)
    if S_A == 0.0 and S_B == 0.0:
        return base
    return base + A2_spin_extra(rho, z, b, S_A, S_B)


def A2_2c_spin_parts(rho, z, b, P, S_A=0.0, S_B=0.0):
    """``(momentum_A2, self_spin_A, self_spin_B, spin_cross)`` of the full source.

    Used by the per-puncture regularity check (each spin part × ψ_BL^{-7} → 0 at
    the punctures).
    """
    mom = A2_2c(rho, z, b, P)
    rho = jnp.asarray(rho)
    z = jnp.asarray(z)
    r_A, r_B = radii(rho, z, b)
    sA = z - b
    sB = z + b
    rho2 = rho ** 2
    self_A = 18.0 * S_A ** 2 * rho2 / r_A ** 8
    self_B = 18.0 * S_B ** 2 * rho2 / r_B ** 8
    cross = 36.0 * S_A * S_B * rho2 * (rho2 + sA * sB) / (r_A ** 5 * r_B ** 5)
    return mom, self_A, self_B, cross


# --------------------------------------------------------------------------
# Raw spin tensor + full-sum oracles (independent verification of the closed form)
# --------------------------------------------------------------------------
def _A_single_spin_tensor(x_vec, x0_vec, S_vec):
    """Single-puncture BY SPIN tensor Â_S^{ij} (3x3 Cartesian) at ``x_vec``.

    Â_S^{ij} = (3/r^3)(v^i n^j + v^j n^i),  v = S × n,  n = (x - x0)/r.
    """
    d = np.asarray(x_vec, dtype=float) - np.asarray(x0_vec, dtype=float)
    r = np.linalg.norm(d)
    n = d / r
    v = np.cross(np.asarray(S_vec, dtype=float), n)
    return (3.0 / r ** 3) * (np.outer(v, n) + np.outer(n, v))


def A2_raw_2c_spin_at_point(x_vec, b, P, S_A, S_B):
    """Direct Â_{ij}Â^{ij} of the summed BY tensor (momentum + aligned spin).

    Builds Â = Â_{P,A} + Â_{P,B} + Â_{S,A} + Â_{S,B} as full 3x3 tensors (A at
    +b, P_A=-Pẑ, S_A=S_Aẑ; B at -b, P_B=+Pẑ, S_B=S_Bẑ) and returns sum_ij Â².
    Independent of the closed form ``A2_2c_spin`` — used only to verify it.
    """
    A = np.array([0.0, 0.0, b])
    B = np.array([0.0, 0.0, -b])
    Aij = (_A_single_tensor(x_vec, A, [0.0, 0.0, -P])
           + _A_single_tensor(x_vec, B, [0.0, 0.0, P])
           + _A_single_spin_tensor(x_vec, A, [0.0, 0.0, S_A])
           + _A_single_spin_tensor(x_vec, B, [0.0, 0.0, S_B]))
    return float(np.sum(Aij * Aij))


def _A_full_tensor_jax(x_vec, b, P, S_A, S_B):
    """Full summed BY tensor Â^{ij} (momentum + spin) as a jax (3,3) at x.

    A differentiable twin of ``A2_raw_2c_spin_at_point``'s tensor, used by
    :func:`divergence_2c_spin_autodiff` for exact (autodiff) transversality.
    """
    x = jnp.asarray(x_vec, dtype=float)
    A = jnp.array([0.0, 0.0, b])
    B = jnp.array([0.0, 0.0, -b])

    def mom(x0, Pv):
        d = x - x0
        r = jnp.linalg.norm(d)
        n = d / r
        Pv = jnp.asarray(Pv)
        Pn = Pv @ n
        proj = jnp.eye(3) - jnp.outer(n, n)
        return (3.0 / (2.0 * r ** 2)) * (jnp.outer(Pv, n) + jnp.outer(n, Pv) - proj * Pn)

    def spin(x0, Sv):
        d = x - x0
        r = jnp.linalg.norm(d)
        n = d / r
        v = jnp.cross(jnp.asarray(Sv), n)
        return (3.0 / r ** 3) * (jnp.outer(v, n) + jnp.outer(n, v))

    return (mom(A, [0.0, 0.0, -P]) + mom(B, [0.0, 0.0, P])
            + spin(A, [0.0, 0.0, S_A]) + spin(B, [0.0, 0.0, S_B]))


def divergence_2c_spin_autodiff(x_vec, b, P, S_A, S_B):
    """∂_j Â^{ij} of the full summed BY tensor via jax autodiff (EXACT).

    Returns the 3-vector; ~machine zero (transverse) for any (P, S_A, S_B), since
    each Bowen–York piece is transverse.  Used by the P2 transversality gate
    (≤1e-14), tighter than the finite-difference ``divergence_raw_2c_at_point``.
    """
    jac = jax.jacfwd(lambda x: _A_full_tensor_jax(x, b, P, S_A, S_B))(
        jnp.asarray(x_vec, dtype=float))
    # jac[i,j,k] = ∂_k Â^{ij};  divergence^i = sum_j ∂_j Â^{ij} = sum_j jac[i,j,j]
    return np.array(jnp.einsum("ijj->i", jac))
