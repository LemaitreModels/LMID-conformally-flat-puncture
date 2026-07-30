"""PARASOL-3D — non-axisymmetric ADM diagnostics (Test E).

Add-only sibling of :mod:`parasol.solver.diagnostics` / :mod:`parasol.validation.adm`
for the first non-axisymmetric solver (``solver_3d``).  Two observables:

* **ADM angular momentum** ``J^i`` — the genuinely-3-D quantity.  Computed from
  the conformal Bowen–York tensor ``Â^{ij}`` (analytic, no solve needed, since
  ``J`` is fixed by ``K^{ij}=ψ^{-10}Â^{ij} -> Â^{ij}`` at infinity) by the York
  surface integral

      J^i = (1/8π) ε^{ijk} ∮_{S_R} x_j Â_{kl} n^l dS ,

  a full 2-D sphere quadrature (Gauss–Legendre in μ=cosθ × trapezoid in φ).  For
  a single puncture this gives **exactly** the closed-form Bowen–York result

      J^i = Σ_X ( S_X^i + (x_X × P_X)^i ) ,

  the spin part being R-independent (the ``Â_S∼1/r³`` tensor integrates with no
  tail) and the momentum part contributing the orbital term ``x_X × P_X`` (a 1/R
  tail, so read at large R / extrapolated).  Both are provided: the surface
  integral is the diagnostic, :func:`adm_J_closed_form` the rigorous cross-check,
  and TwoPunctures' reported ``J`` the external oracle.

* **ADM mass** ``M_ADM`` — the m=0 (φ-averaged) monopole, identical spectral
  boundary extraction to :func:`adm.adm_mass_spectral`; higher azimuthal modes
  decay faster than 1/r and do not feed the monopole, so the φ-average is exact.

The same PARASOL z-axis convention as the solver: punctures A at +b, B at −b,
per-puncture momentum/spin VECTORS.  Standalone: numpy + the sibling source_3d.
"""

from __future__ import annotations

import numpy as np

from . import source_3d


# Levi-Civita symbol ε^{ijk}
_EPS = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _EPS[_i, _j, _k] = 1.0
    _EPS[_i, _k, _j] = -1.0


# --------------------------------------------------------------------------
# ADM angular momentum from the conformal Bowen–York tensor
# --------------------------------------------------------------------------
def adm_J_surface(b, P_A_vec, P_B_vec, S_A_vec, S_B_vec,
                  R=None, n_mu=64, n_phi=64):
    """ADM angular momentum ``J^i`` via the York surface integral at radius R.

        J^i = (1/8π) ε^{ijk} ∮_{S_R} x_j (Â_{kl} n^l) dS ,

    over a CoM-centred coordinate sphere, with the analytic summed Bowen–York
    tensor ``Â`` (momentum + spin) of both punctures.  Returns the 3-vector
    (PARASOL frame).  ``R`` defaults to ``200·b`` (the spin part is exact at any
    R; a large R suppresses the momentum/orbital 1/R tail).
    """
    if R is None:
        R = 200.0 * (abs(b) if b else 1.0)
    mu, w = np.polynomial.legendre.leggauss(int(n_mu))         # ∫_{-1}^{1} dμ
    phi = 2.0 * np.pi * np.arange(int(n_phi)) / int(n_phi)     # trapezoid in φ
    dphi = 2.0 * np.pi / int(n_phi)
    sinth = np.sqrt(np.clip(1.0 - mu ** 2, 0.0, None))

    MU, PH = np.meshgrid(mu, phi, indexing="ij")               # (n_mu, n_phi)
    ST = np.sqrt(np.clip(1.0 - MU ** 2, 0.0, None))
    W = np.broadcast_to(w[:, None], MU.shape)                  # μ-weight per node
    # outward unit normal n and Cartesian point x = R n
    n = np.stack([ST * np.cos(PH), ST * np.sin(PH), MU], axis=-1).reshape(-1, 3)
    x = R * n
    weight = (W * dphi).reshape(-1)                            # dμ dφ quadrature weight

    T = source_3d.A_full_tensor_vec(x, b, P_A_vec, P_B_vec, S_A_vec, S_B_vec)  # (N,3,3)
    An = np.einsum("pkl,pl->pk", T, n)                         # Â_{kl} n^l   (N,3)
    integ = np.einsum("ijk,pj,pk->pi", _EPS, x, An)            # ε^{ijk} x_j (Ân)_k
    J = (R ** 2 / (8.0 * np.pi)) * np.einsum("p,pi->i", weight, integ)
    return J


def adm_J_surface_extrap(b, P_A_vec, P_B_vec, S_A_vec, S_B_vec,
                         radii=None, **kw):
    """``J^i`` extrapolated to R→∞ (Richardson in 1/R) — robust to the orbital tail.

    Evaluates :func:`adm_J_surface` at several radii and fits ``J(R)=J∞+c/R`` per
    component.  For pure-spin data every radius already agrees (no tail), so this
    equals :func:`adm_J_surface`; it matters only when ``x_X × P_X ≠ 0``.
    """
    if radii is None:
        radii = (abs(b) if b else 1.0) * np.array([60.0, 100.0, 160.0, 250.0])
    radii = np.asarray(radii, dtype=float)
    Js = np.array([adm_J_surface(b, P_A_vec, P_B_vec, S_A_vec, S_B_vec,
                                 R=float(R), **kw) for R in radii])     # (Nr,3)
    t = 1.0 / radii
    out = np.empty(3)
    for i in range(3):
        coef = np.polyfit(t, Js[:, i], min(2, radii.size - 1))
        out[i] = coef[-1]                                      # value at t=0 (R→∞)
    return out


def adm_J_closed_form(b, P_A_vec, P_B_vec, S_A_vec, S_B_vec):
    """Analytic Bowen–York ``J^i = Σ_X (S_X + x_X × P_X)`` (PARASOL frame).

    Punctures A at ``x_A=(0,0,+b)``, B at ``x_B=(0,0,-b)``.  The rigorous
    cross-check for the surface integral and the same quantity TwoPunctures
    reports (``J = Cross[r1,p1]+Cross[r2,p2]+s1+s2``).
    """
    xA = np.array([0.0, 0.0, float(b)])
    xB = np.array([0.0, 0.0, -float(b)])
    S_A = np.asarray(S_A_vec, dtype=float)
    S_B = np.asarray(S_B_vec, dtype=float)
    P_A = np.asarray(P_A_vec, dtype=float)
    P_B = np.asarray(P_B_vec, dtype=float)
    return S_A + S_B + np.cross(xA, P_A) + np.cross(xB, P_B)


# --------------------------------------------------------------------------
# ADM mass (m=0 monopole; spectral boundary extraction)
# --------------------------------------------------------------------------
def adm_mass_spectral_3d(prob, U, sl):
    """``M_ADM = (m_A+m_B) + 2c`` with ``c`` read spectrally from the m=0 mode.

    As in :func:`adm.adm_mass_spectral`: at the A=1 (infinity) edge the ABT map
    gives ``r ∼ b/(1−A)``, so a 1/r monopole obeys ``∂_A u|_{A=1} = −c/b``
    identically in B and ``M_ADM = (m_A+m_B) − 2b ⟨∂_A u_0⟩_{A=1}``.  The m=0
    (φ-averaged) field is the only azimuthal mode with a 1/r tail, so the
    φ-average is the exact monopole.
    """
    DA = np.asarray(prob.DA1)
    U3 = np.asarray(U).reshape(prob.shape)                     # (Na+1, Nb, Nφ)
    u0 = U3.mean(axis=2)                                       # m=0 mode (Na+1, Nb)
    dUdA_at_inf = (DA @ u0)[0, :]                              # A[0]=1 edge
    c = -sl.b * float(np.mean(dUdA_at_inf))
    return float(sl.M + 2.0 * c)
