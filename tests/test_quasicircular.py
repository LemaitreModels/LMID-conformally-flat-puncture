"""Phase-P0 gates for the quasi-circular (QC) momentum condition.

De-risks ``parametric/quasicircular.py`` in isolation (no ``theta_to_slice3d``
wiring — that is Phase P1).  Three validation axes from the standing plan
(``notes/qc_extension_plan.md``):

  (a) large-b **Newtonian anchor**  p_t → μ √(M/D) = μ √(M/2b);
  (b) the **L-along-y convention** (risk R2): P_A=(+p_t,0,−p_r), P_B=(−p_t,0,+p_r)
      ⇒ ADM J = (0, 2b·p_t + S_y, 0) along **y**, zero net linear momentum, and
      the **TwoPunctures oracle** confirms it (slow, skips if the binary is absent);
  (c) the **eccentricity proxy** |p_r|/p_t is small and → 0 as b grows.

Fast tests are pure mapping/analytic (no solve, no oracle); the oracle J check is
``slow`` and skips cleanly when the TwoPunctures binary is not built.
"""

import math

import numpy as np
import pytest

from lemaitre.initial_data.parametric import quasicircular as qc
from lemaitre.initial_data.solver import diagnostics_3d as d3
from lemaitre.initial_data.validation import twopunctures as tp, conventions as cv

_oracle = pytest.mark.skipif(not tp.available(),
                             reason="TwoPunctures binary not built (see build.sh)")


# ==========================================================================
# (a) Newtonian anchor
# ==========================================================================
def test_newtonian_leading_order_exact():
    """pn_order=0 IS the Newtonian anchor μ√(M/D) = μ√(M/2b) to machine precision,
    for equal and unequal masses."""
    M = 1.0
    for q in (1.0, 1.5, 3.0, 8.0):
        m_A, m_B = M * q / (1 + q), M / (1 + q)
        mu = m_A * m_B / M
        for b in (2.0, 5.0, 25.0):
            D = 2.0 * b
            expect = mu * math.sqrt(M / D)
            got = qc.pt_nonspinning(b, m_A, m_B, pn_order=0)
            assert abs(got - expect) < 1e-15 * expect
            # and the dedicated anchor helper agrees
            assert abs(qc.pt_newtonian(b, m_A, m_B) - expect) < 1e-15 * expect


def test_pt_converges_to_newtonian_at_large_b():
    """The full 3PN p_t → the Newtonian anchor as b→∞, with the PN correction
    shrinking monotonically (∝ M/D ∝ 1/b at leading correction)."""
    m_A = m_B = 0.5
    ratios = []
    for b in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        ptN = qc.pt_newtonian(b, m_A, m_B)
        pt3 = qc.pt_nonspinning(b, m_A, m_B, pn_order=3)
        ratios.append(pt3 / ptN)
    ratios = np.array(ratios)
    assert ratios[-1] < 1.02 and ratios[0] > 1.05        # converging toward 1
    assert np.all(np.diff(ratios) < 0)                   # monotone decrease
    assert abs(ratios[-1] - 1.0) < 0.02
    # the leading PN correction halves when b doubles (∝ 1/b): ratio-1 ~ 2·(M/D)
    corr = ratios - 1.0
    assert abs(corr[-2] / corr[-1] - 2.0) < 0.05         # b: 80→160


def test_pn_order_series_structure():
    """Each PN order adds the expected (M/D)^{n+1/2} term (Walther Eq. 45)."""
    m_A = m_B = 0.5
    M, mu, nu, b = 1.0, 0.25, 0.25, 8.0
    x = M / (2.0 * b)
    p0 = qc.pt_nonspinning(b, m_A, m_B, 0)
    p1 = qc.pt_nonspinning(b, m_A, m_B, 1)
    p2 = qc.pt_nonspinning(b, m_A, m_B, 2)
    p3 = qc.pt_nonspinning(b, m_A, m_B, 3)
    assert abs((p1 - p0) - mu * 2.0 * x ** 1.5) < 1e-15
    assert abs((p2 - p1) - mu * (1 / 16) * (42 - 43 * nu) * x ** 2.5) < 1e-15
    c3 = 480 + (163 * math.pi ** 2 - 4556) * nu + 104 * nu ** 2
    assert abs((p3 - p2) - mu * (1 / 128) * c3 * x ** 3.5) < 1e-14


# ==========================================================================
# (b) L-along-y convention (risk R2) + zero net linear momentum
# ==========================================================================
def test_momentum_vectors_convention_and_zero_net():
    """qc_momenta gives tangential-x / radial-z anti-symmetric momenta with zero
    net linear momentum."""
    b, m_A, m_B = 5.0, 0.6, 0.4
    P_A, P_B = qc.qc_momenta(b, m_A, m_B)
    p_t, p_r = qc.qc_scalar_momenta(b, m_A, m_B)
    assert P_A == (p_t, 0.0, -p_r)
    assert P_B == (-p_t, 0.0, p_r)
    assert np.allclose(np.array(P_A) + np.array(P_B), 0.0, atol=0.0)  # exact
    # tangential along x, radial along z, nothing along y
    assert P_A[1] == 0.0 and P_B[1] == 0.0


def test_orbital_angular_momentum_along_plus_y():
    """L = x_A×P_A + x_B×P_B = (0, 2b·p_t, 0) along +y (the R2 convention)."""
    b, m_A, m_B = 5.0, 0.5, 0.5
    P_A, P_B = qc.qc_momenta(b, m_A, m_B)
    p_t, _ = qc.qc_scalar_momenta(b, m_A, m_B)
    J = d3.adm_J_closed_form(b, P_A, P_B, (0, 0, 0), (0, 0, 0))
    assert abs(J[0]) < 1e-14 and abs(J[2]) < 1e-14           # nothing off y
    assert abs(J[1] - 2.0 * b * p_t) < 1e-14                 # = 2b·p_t
    assert J[1] > 0.0                                        # +y
    assert abs(J[1] - qc.orbital_angular_momentum(b, p_t)) < 1e-15


def test_aligned_spin_is_S_y_and_adds_to_Jy():
    """Aligned spin = the S_y component: it feeds the SO correction and adds
    S_Ay+S_By to J_y, keeping J along y."""
    b, m_A, m_B = 6.0, 0.5, 0.5
    S_A, S_B = (0.0, 0.3, 0.0), (0.0, 0.1, 0.0)
    P_A, P_B = qc.qc_momenta(b, m_A, m_B, S_A, S_B)
    p_t, _ = qc.qc_scalar_momenta(b, m_A, m_B, S_A[1] / m_A ** 2, S_B[1] / m_B ** 2)
    J = d3.adm_J_closed_form(b, P_A, P_B, S_A, S_B)
    assert abs(J[0]) < 1e-14 and abs(J[2]) < 1e-14           # still along y
    assert abs(J[1] - (2.0 * b * p_t + S_A[1] + S_B[1])) < 1e-13


def test_in_plane_spin_tilts_J_off_y():
    """In-plane spin (S_x or S_z) does NOT feed the aligned SO correction and
    tilts J off y — pinning aligned=S_y, precessing=S_x,S_z."""
    b, m_A, m_B = 6.0, 0.5, 0.5
    # S_z is in-plane (orbital plane is x–z): J acquires a z-component = S_z
    S_A = (0.0, 0.0, 0.25)
    P_A, P_B = qc.qc_momenta(b, m_A, m_B, S_A, (0, 0, 0))
    P_A0, _ = qc.qc_momenta(b, m_A, m_B)                     # no-spin momenta
    assert P_A[0] == P_A0[0]                                 # S_z leaves p_t untouched
    J = d3.adm_J_closed_form(b, P_A, P_B, S_A, (0, 0, 0))
    assert abs(J[2] - 0.25) < 1e-14                          # J_z = S_z
    assert abs(J[1] - 2.0 * b * (-P_B[0])) < 1e-13           # J_y = orbital only


def test_spin_orbit_reduces_pt_for_prograde():
    """Prograde (aligned, +y) spin lowers the tangential momentum needed for a
    circular orbit; retrograde raises it; equal-mass SO is symmetric under A↔B."""
    b, m_A, m_B = 6.0, 0.5, 0.5
    p0 = qc.pt_nonspinning(b, m_A, m_B)
    pt_pro, _ = qc.qc_scalar_momenta(b, m_A, m_B, +0.5, +0.5)
    pt_ret, _ = qc.qc_scalar_momenta(b, m_A, m_B, -0.5, -0.5)
    assert pt_pro < p0 < pt_ret
    # A↔B symmetry at equal mass
    ab = qc.pt_spin_orbit(b, m_A, m_B, 0.4, 0.1)
    ba = qc.pt_spin_orbit(b, m_A, m_B, 0.1, 0.4)
    assert abs(ab - ba) < 1e-15
    # test-mass limit: the LARGER hole's spin dominates the SO correction
    big, small = 0.9, 0.1
    so_big = abs(qc.pt_spin_orbit(b, big, small, 1.0, 0.0))   # spin on larger
    so_small = abs(qc.pt_spin_orbit(b, big, small, 0.0, 1.0))  # spin on smaller
    assert so_big > so_small


# ==========================================================================
# (c) eccentricity proxy
# ==========================================================================
def test_eccentricity_proxy_small_and_decreasing():
    """|p_r|/p_t is small and → 0 as b grows (leading RR ∝ ν (M/D)^{5/2})."""
    m_A = m_B = 0.5
    bs = np.array([2.5, 5.0, 10.0, 20.0, 40.0])
    ep = np.array([qc.eccentricity_proxy(b, m_A, m_B) for b in bs])
    assert np.all(ep < 0.05)                                 # small everywhere
    assert np.all(np.diff(ep) < 0)                           # monotone decrease
    # ~(M/D)^{5/2} scaling: b→2b divides the proxy by ~2^{5/2}≈5.66
    assert abs(ep[-2] / ep[-1] - 2 ** 2.5) / 2 ** 2.5 < 0.05


def test_radial_momentum_sign_and_leading_form():
    """p_r is the Peters leading-order infall magnitude (64/5)μ²M²/D³, applied as
    infall (P_A_z<0, P_B_z>0), and vanishes when radial=False."""
    b, m_A, m_B = 5.0, 0.5, 0.5
    M, mu, D = 1.0, 0.25, 10.0
    assert abs(qc.pr_radial(b, m_A, m_B) - (64 / 5) * mu ** 2 * M ** 2 / D ** 3) < 1e-16
    P_A, P_B = qc.qc_momenta(b, m_A, m_B)
    assert P_A[2] < 0.0 and P_B[2] > 0.0                     # infall
    P_A0, P_B0 = qc.qc_momenta(b, m_A, m_B, radial=False)
    assert P_A0[2] == 0.0 and P_B0[2] == 0.0
    # radial momentum contributes nothing to J (parallel to position)
    J_r = d3.adm_J_closed_form(b, (0, 0, P_A[2]), (0, 0, P_B[2]), (0, 0, 0), (0, 0, 0))
    assert np.allclose(J_r, 0.0, atol=1e-15)


# ==========================================================================
# (b) TwoPunctures oracle cross-check (slow; skips if the binary is absent)
# ==========================================================================
# J is computed by TwoPunctures analytically from the puncture parameters
# (J = Σ Cross[r,p]+s), so it is resolution-independent — a modest grid suffices.
_QR = np.array([0.5, 1.0, 1.5])
_QZ = np.array([0.3, -0.2, 0.4])
_QP = np.array([0.0, 1.0, 2.0])


@_oracle
@pytest.mark.slow
@pytest.mark.parametrize("b,m_A,m_B,S_A,S_B", [
    (5.0, 0.5, 0.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),      # equal-mass, non-spinning
    (6.0, 0.6, 0.4, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),      # unequal-mass q=1.5
    (6.0, 0.5, 0.5, (0.0, 0.3, 0.0), (0.0, 0.1, 0.0)),      # aligned spin (S_y)
])
def test_oracle_J_is_2b_pt_along_y(b, m_A, m_B, S_A, S_B):
    """TwoPunctures reports ADM J = (0, 2b·p_t + S_Ay + S_By, 0) along +y for the
    QC data, and net linear momentum is zero by construction."""
    P_A, P_B = qc.qc_momenta(b, m_A, m_B, S_A, S_B)
    p_t, _ = qc.qc_scalar_momenta(b, m_A, m_B, S_A[1] / m_A ** 2, S_B[1] / m_B ** 2)
    res = tp.solve_parasol_points_3d(b, m_A, m_B, P_A, P_B, S_A, S_B,
                                     _QR, _QZ, _QP, nA=32, nB=32, nphi=8)
    J_par = np.array(cv.tp_vec_to_parasol(res.J))            # TP native → PARASOL
    J_expect = np.array([0.0, 2.0 * b * p_t + S_A[1] + S_B[1], 0.0])
    print(f"[QC] b={b} m=({m_A},{m_B}) S=({S_A},{S_B})  J_oracle(PARASOL)={J_par} "
          f"expect={J_expect}")
    assert np.max(np.abs(J_par - J_expect)) < 1e-9, f"J off: {J_par} vs {J_expect}"
    # dominantly along y (orbital + aligned spin), no tilt
    assert abs(J_par[0]) < 1e-9 and abs(J_par[2]) < 1e-9
    assert J_par[1] > 0.1
    # zero net linear momentum (by construction; TP does not report P_ADM)
    assert np.allclose(np.array(P_A) + np.array(P_B), 0.0, atol=0.0)
