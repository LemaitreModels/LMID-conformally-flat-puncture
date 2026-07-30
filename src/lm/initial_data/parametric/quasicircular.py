"""LM-initial-data — quasi-circular (QC) puncture momenta from PN closed forms.

**Phase P0 de-risk** of the "beyond head-on" extension (see
``notes/qc_extension_plan.md``): turn the free momentum knob into a *deterministic*
function of ``(b, q, spins)`` set by a post-Newtonian quasi-circularity condition,
so every surrogate node is astrophysical (inspiral/merger) rather than head-on
infall.  This module is **add-only and self-contained** — it computes the momenta
only; it does NOT touch ``theta_to_slice3d`` (that wiring is Phase P1).

    qc_momenta(b, m_A, m_B, S_A_vec, S_B_vec) -> (P_A_vec, P_B_vec)

Physics — the closed-form PN quasi-circular momenta
---------------------------------------------------
For a near-circular orbit the *tangential* momentum is fixed by the circular-orbit
condition of the PN Hamiltonian and the *radial* momentum by radiation reaction
(the standard puncture-ID procedure of Husa, Hannam, González, Sperhake &
Brügmann, PRD **77**, 044037 (2008), and Walther, Brügmann & Müller, PRD **79**,
124040 (2009)).

* **Tangential momentum** — the non-spinning 3PN closed form as an explicit series
  in ``x = M/D`` (D = coordinate separation between the punctures), Walther,
  Brügmann & Müller (2009) Eq. (45):

      p_t = μ [ x^{1/2}
                + 2 x^{3/2}
                + (1/16)(42 − 43ν) x^{5/2}
                + (1/128)(480 + (163π² − 4556)ν + 104ν²) x^{7/2} ] ,

  with  M = m_A + m_B,  μ = m_A m_B / M,  ν = μ/M  (symmetric mass ratio),
  D = 2b.  Leading (Newtonian) order  **p_t → μ √(M/D) = μ √(M/2b)**.
  The Newtonian and 1PN coefficients were independently cross-verified by
  converting the frequency-parametrised series of the 2024 revisit (Healy et al.,
  arXiv:2406.11564, Eqs. 37 & 89) into the separation series — the ν-terms cancel
  to give the coefficient 2 exactly, and the 1PN separation relation reproduces
  the standard ADM-coordinate result r/M = (MΩ)^{-2/3} − (1 − ν/3).  The 2PN/3PN
  coefficients are Walther Eq. (45) as transcribed (do not affect the P0 gates —
  the oracle J validates p_t self-consistently and the Newtonian anchor pins the
  leading term; higher-PN accuracy only tightens the residual eccentricity).

* **Leading spin-orbit correction** (aligned spins), from Healy et al. (2024)
  Eq. (37), converted to the separation parametrisation ((MΩ)^{4/3} → x² at
  leading order):

      p_t^{SO} = −μ · [2/(3(1+q)²)] · [(4+3q) χ₁ + q(3+4q) χ₂] · x² ,

  with Healy's convention **q = m₂/m₁ ≤ 1, body 1 = the LARGER hole** (the
  paper restates this in the q = m_A/m_B ≥ 1 convention; identical physics), and
  χ_i = (S_i · L̂)/m_i² the dimensionless spin projected on the orbital angular
  momentum.  Test-mass limit q→0: the (4+3q)χ₁ term (larger hole) dominates —
  the physically-correct frame-dragging behaviour.  This is a 1.5PN correction
  and is added on top of the non-spinning series.

* **Radial momentum** (radiation reaction), leading (Peters 1964) order — the
  Walther Eq. (51) procedure ``p_r = μ ṙ`` at leading order, with the quadrupole
  inspiral rate ṙ = dD/dt = −(64/5) μ M²/D³ :

      p_r = −(64/5) μ² M² / D³        (per-puncture, infall; → 0 as b → ∞).

  The full 3.5PN radial momentum (Walther Eq. 51 evaluated with the high-order
  flux) is a Phase-P4 refinement; the leading term suffices for the P0
  "eccentricity proxy small" gate (|p_r|/p_t ∝ ν (M/D)^{5/2} → 0).

Geometry / convention (risk R2, PINNED)
---------------------------------------
LM-initial-data punctures on the **z-axis** at A=(0,0,+b), B=(0,0,−b).  The tangential
momentum is placed along **x** (anti-symmetric), the small radial momentum along
**z** (infall):

    P_A = ( +p_t, 0, −p_r ),     P_B = ( −p_t, 0, +p_r ).

Hence the orbital plane is **x–z** and the orbital angular momentum
L = x_A×P_A + x_B×P_B = (0, 2b·p_t, 0) is along **+y**.  Therefore

    * "aligned spin"  = the  **S_y**  component  (χ_aligned = S_y/m²);
    * "in-plane / precessing"  = S_x, S_z.

Net linear momentum P_A + P_B = 0 exactly (equal-and-opposite momenta), i.e. the
CoM frame.  The radial momentum is parallel to the position vectors, so it does
not contribute to J: J_orbital = (0, 2b·p_t, 0) regardless of p_r.

Standalone: numpy + math + jax (x64 on import).  No solver coupling.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: F401  (kept for downstream/interop consistency)

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------
# Scalar PN pieces (functions of the coordinate separation D = 2b)
# --------------------------------------------------------------------------
def pt_nonspinning(b: float, m_A: float, m_B: float, pn_order: int = 3) -> float:
    """Non-spinning quasi-circular tangential momentum ``p_t`` (per puncture).

    Walther, Brügmann & Müller (2009) Eq. (45), a series in ``x = M/D``,
    D = 2b.  ``pn_order`` truncates the series:
        0 → Newtonian (x^{1/2}),  1 → +1PN,  2 → +2PN,  3 → +3PN (default).
    """
    if pn_order not in (0, 1, 2, 3):
        raise ValueError(f"pn_order must be 0..3, got {pn_order}")
    M = m_A + m_B
    mu = m_A * m_B / M
    nu = mu / M
    D = 2.0 * b
    x = M / D
    series = math.sqrt(x)                                        # Newtonian
    if pn_order >= 1:
        series += 2.0 * x ** 1.5                                 # 1PN
    if pn_order >= 2:
        series += (1.0 / 16.0) * (42.0 - 43.0 * nu) * x ** 2.5   # 2PN
    if pn_order >= 3:
        c3 = 480.0 + (163.0 * math.pi ** 2 - 4556.0) * nu + 104.0 * nu ** 2
        series += (1.0 / 128.0) * c3 * x ** 3.5                  # 3PN
    return mu * series


def pt_spin_orbit(b: float, m_A: float, m_B: float,
                  chi_A_aligned: float, chi_B_aligned: float) -> float:
    """Leading (1.5PN) spin-orbit correction to ``p_t`` (aligned spins).

    Healy et al. (2024, arXiv:2406.11564) Eq. (37) spin-orbit term, converted to
    ``x = M/D`` at leading order.  ``chi_X_aligned = (S_X·L̂)/m_X²`` is the
    dimensionless spin projected on the orbital angular momentum (= S_Xy/m_X² in
    the LM-initial-data frame, where L∥y).  Evaluated in Healy's internal convention
    q = m₂/m₁ ≤ 1 with body 1 = the LARGER hole, computed here directly from the
    masses (m1=max, m2=min) — this is INDEPENDENT of the surrogate's global
    q = m_A/m_B ∈ [1,3], so it is symmetric under a consistent relabel of (A,B).
    The paper Eq. (P_t^SO) states the algebraically identical form in the
    q = m_A/m_B ≥ 1 convention; the two agree to machine precision.
    """
    M = m_A + m_B
    mu = m_A * m_B / M
    D = 2.0 * b
    x = M / D
    # body 1 = larger, body 2 = smaller (Healy convention q = m2/m1 <= 1)
    if m_A >= m_B:
        m1, m2, chi1, chi2 = m_A, m_B, chi_A_aligned, chi_B_aligned
    else:
        m1, m2, chi1, chi2 = m_B, m_A, chi_B_aligned, chi_A_aligned
    q = m2 / m1
    coeff = (2.0 / (3.0 * (1.0 + q) ** 2)) * ((4.0 + 3.0 * q) * chi1
                                              + q * (3.0 + 4.0 * q) * chi2)
    return -mu * coeff * x ** 2


def pr_radial(b: float, m_A: float, m_B: float) -> float:
    """Leading-order radiation-reaction radial momentum MAGNITUDE ``|p_r|``.

    Peters (1964) quadrupole inspiral: ṙ = dD/dt = −(64/5) μ M²/D³, and
    p_r = μ ṙ (Walther Eq. 51 at leading order), so |p_r| = (64/5) μ² M²/D³.
    Positive; the sign convention (infall) is applied when assembling the vectors.
    """
    M = m_A + m_B
    mu = m_A * m_B / M
    D = 2.0 * b
    return (64.0 / 5.0) * mu ** 2 * M ** 2 / D ** 3


def qc_scalar_momenta(b: float, m_A: float, m_B: float,
                      chi_A_aligned: float = 0.0, chi_B_aligned: float = 0.0,
                      *, pn_order: int = 3, spin_orbit: bool = True,
                      radial: bool = True) -> Tuple[float, float]:
    """Scalar quasi-circular ``(p_t, p_r)`` (per-puncture momentum magnitudes).

    ``p_t`` = non-spinning 3PN + (optional) leading spin-orbit; ``p_r`` = leading
    radiation-reaction magnitude (0 if ``radial=False``).
    """
    p_t = pt_nonspinning(b, m_A, m_B, pn_order=pn_order)
    if spin_orbit and (chi_A_aligned or chi_B_aligned):
        p_t += pt_spin_orbit(b, m_A, m_B, chi_A_aligned, chi_B_aligned)
    p_r = pr_radial(b, m_A, m_B) if radial else 0.0
    return p_t, p_r


# --------------------------------------------------------------------------
# The deliverable — full puncture momentum vectors
# --------------------------------------------------------------------------
def qc_momenta(b: float, m_A: float, m_B: float,
               S_A_vec: Sequence[float] = (0.0, 0.0, 0.0),
               S_B_vec: Sequence[float] = (0.0, 0.0, 0.0),
               *, pn_order: int = 3, spin_orbit: bool = True,
               radial: bool = True) -> Tuple[Vec3, Vec3]:
    """Quasi-circular puncture linear momenta ``(P_A_vec, P_B_vec)`` (LM-initial-data frame).

    Punctures at A=(0,0,+b), B=(0,0,−b).  Returns the tangential-along-x /
    radial-along-z momenta of a near-circular orbit with orbital angular momentum
    L along **+y**:

        P_A = ( +p_t, 0, −p_r ),     P_B = ( −p_t, 0, +p_r ),

    so ``adm_J_closed_form`` → (0, 2b·p_t + S_Ay + S_By, 0).  The spins are inputs
    (their **y**-components feed the aligned spin-orbit correction to p_t); they
    are NOT returned — the caller sets them on the ``Slice3D`` directly.

    Parameters
    ----------
    b        : half-separation (D = 2b).
    m_A,m_B  : bare puncture masses (A at +b, B at −b).
    S_A_vec,S_B_vec : Cartesian spin 3-vectors (LM-initial-data frame).  Only the aligned
               (y) component enters the leading spin-orbit correction.
    pn_order : non-spinning p_t truncation (0..3, default 3).
    spin_orbit : include the leading (1.5PN) aligned spin-orbit correction.
    radial   : include the (small) radiation-reaction radial momentum.
    """
    m_A = float(m_A)
    m_B = float(m_B)
    chi_A_aligned = float(S_A_vec[1]) / m_A ** 2      # (S_A·ŷ)/m_A²
    chi_B_aligned = float(S_B_vec[1]) / m_B ** 2      # (S_B·ŷ)/m_B²
    p_t, p_r = qc_scalar_momenta(b, m_A, m_B, chi_A_aligned, chi_B_aligned,
                                 pn_order=pn_order, spin_orbit=spin_orbit,
                                 radial=radial)
    P_A_vec = (float(p_t), 0.0, float(-p_r))
    P_B_vec = (float(-p_t), 0.0, float(p_r))
    return P_A_vec, P_B_vec


# --------------------------------------------------------------------------
# Diagnostics — Newtonian anchor and the eccentricity proxy
# --------------------------------------------------------------------------
def pt_newtonian(b: float, m_A: float, m_B: float) -> float:
    """The Newtonian anchor ``μ √(M/D) = μ √(M/2b)`` (per-puncture)."""
    M = m_A + m_B
    mu = m_A * m_B / M
    D = 2.0 * b
    return mu * math.sqrt(M / D)


def eccentricity_proxy(b: float, m_A: float, m_B: float,
                       chi_A_aligned: float = 0.0, chi_B_aligned: float = 0.0,
                       **kw) -> float:
    """Quasi-circularity proxy ``|p_r| / p_t`` (small ⇒ near-circular; → 0 as b→∞)."""
    p_t, p_r = qc_scalar_momenta(b, m_A, m_B, chi_A_aligned, chi_B_aligned, **kw)
    return abs(p_r) / p_t


def orbital_angular_momentum(b: float, p_t: float) -> float:
    """Orbital angular momentum magnitude ``L = 2b·p_t`` (along +y in LM-initial-data)."""
    return 2.0 * b * p_t
