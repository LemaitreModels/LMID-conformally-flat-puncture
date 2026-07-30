"""Test E — non-axisymmetric PARASOL data vs the TwoPunctures oracle + ADM J.

Cross-checks the first non-axisymmetric (Fourier-φ) two-centre Bowen–York data
(``solver_3d``) against TwoPunctures, and validates the 3-D ADM angular-momentum
diagnostic (``diagnostics_3d``).  Mirrors the B1 axisymmetric harness
(``test_validation_twopunctures.py``): the oracle-dependent comparisons are
``slow`` and skip cleanly when the binary is absent, so the analytic checks (the
J surface integral, frame rotations) still run everywhere.

The oracle binary was extended for Test E to accept per-puncture momentum/spin
VECTORS (the ``argc>=24`` override path of ``src/main.c``); the existing
axisymmetric (scalar) wrapper path is byte-identical, so B1 is unaffected.

Convention: PARASOL punctures on the z-axis at ±b; the PARASOL→TP frame is the
single proper rotation z^P→x^TP (``conventions.parasol_vec_to_tp``,
``parasol_point_to_tp_3d``), applied to query points AND momentum/spin vectors.
"""

import numpy as np
import pytest

from lm.initial_data.solver import solver_3d as s3, source, diagnostics_3d as d3
from lm.initial_data.solver.solver_3d import Slice3D
from lm.initial_data.validation import twopunctures as tp, conventions as cv

_oracle = pytest.mark.skipif(not tp.available(),
                             reason="TwoPunctures binary not built (see build.sh)")

# A genuinely non-axisymmetric slice: a single MISALIGNED spin on puncture A
# (z-aligned Sz + transverse Sx) breaks axisymmetry (populates m≠0), head-on
# z-momenta.  J should be tilted off the collision axis: J = (Sx, 0, Sz).
B, MA, MB, P = 1.5, 0.5, 0.5, 0.5
S_A_VEC = (0.3, 0.0, 0.2)
S_B_VEC = (0.0, 0.0, 0.0)
P_A_VEC = (0.0, 0.0, -P)
P_B_VEC = (0.0, 0.0, +P)

# shared query points with genuine φ-content
QR = np.array([0.4, 0.8, 0.6, 1.2, 2.0]) * B
QZ = np.array([0.6, 0.0, -0.5, 0.3, 0.4]) * B
QP = np.array([0.0, 1.0, 2.0, 0.5, 2.5])


# ==========================================================================
# Analytic (no oracle) — the ADM-J surface integral and the frame rotations.
# These always run.
# ==========================================================================
def test_adm_J_surface_matches_closed_form():
    """York surface integral of the BY tensor == Σ_X(S_X + x_X×P_X) to machine.

    Pure-spin (orbital 0) is exact at any R; an off-axis momentum adds the
    orbital term x_X×P_X, recovered (with extrapolation) just as exactly.
    """
    # (1) misaligned spin, on-axis momenta (orbital = 0)
    Jcf = d3.adm_J_closed_form(B, P_A_VEC, P_B_VEC, S_A_VEC, S_B_VEC)
    Jsi = d3.adm_J_surface(B, P_A_VEC, P_B_VEC, S_A_VEC, S_B_VEC)
    assert np.max(np.abs(Jcf - Jsi)) < 1e-12, f"spin J off: {Jcf} vs {Jsi}"
    # (2) off-axis momentum -> nonzero orbital angular momentum
    Pax_A, Pax_B = (0.1, 0.0, -P), (-0.1, 0.0, P)
    Jcf2 = d3.adm_J_closed_form(B, Pax_A, Pax_B, S_A_VEC, S_B_VEC)
    Jext = d3.adm_J_surface_extrap(B, Pax_A, Pax_B, S_A_VEC, S_B_VEC)
    assert np.max(np.abs(Jcf2 - Jext)) < 1e-9, f"orbital J off: {Jcf2} vs {Jext}"
    # the orbital term genuinely contributes (test has teeth)
    assert abs(Jcf2[1]) > 0.1, f"orbital term vanished: {Jcf2}"


def test_adm_J_misaligned_spin_is_tilted():
    """The misaligned-spin J is tilted off the collision (z) axis: J=(Sx,0,Sz)."""
    J = d3.adm_J_closed_form(B, P_A_VEC, P_B_VEC, S_A_VEC, S_B_VEC)
    assert abs(J[0] - 0.3) < 1e-12 and abs(J[2] - 0.2) < 1e-12
    assert abs(J[1]) < 1e-12
    assert abs(J[0]) > 0.1 and abs(J[2]) > 0.1, "J not genuinely tilted"


def test_frame_rotation_consistency():
    """parasol_vec_to_tp is a proper rotation, round-trips, and is consistent
    with the established axisymmetric scalar map (parasol_to_tp)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=3)
        rt = np.array(cv.tp_vec_to_parasol(cv.parasol_vec_to_tp(v)))
        assert np.max(np.abs(rt - v)) < 1e-14
    # consistency: PARASOL z-momentum (0,0,-P) -> TP (-P,0,0) (= parasol_to_tp)
    assert cv.parasol_vec_to_tp((0.0, 0.0, -P)) == (-P, 0.0, 0.0)
    # z-aligned spin (0,0,S) -> TP (S,0,0) (= parasol_to_tp's par_S_plus)
    assert cv.parasol_vec_to_tp((0.0, 0.0, 0.7)) == (0.7, 0.0, 0.0)
    # the 3-D point map reduces to the φ=0 meridian map
    p0 = cv.parasol_point_to_tp_3d(0.6, 1.1, 0.0)
    assert np.allclose(p0, cv.parasol_point_to_tp(0.6, 1.1))


# ==========================================================================
# Oracle cross-checks (slow, skip if the binary is absent)
# ==========================================================================
@pytest.fixture(scope="module")
def tp_misaligned():
    """One TwoPunctures solve at the misaligned-spin slice (vector data)."""
    return tp.solve_parasol_points_3d(
        B, MA, MB, P_A_VEC, P_B_VEC, S_A_VEC, S_B_VEC, QR, QZ, QP,
        nA=64, nB=64, nphi=12)


@pytest.fixture(scope="module")
def parasol_misaligned_fine():
    prob = s3.make_problem(Na=56, Nb=40, Nphi=10)
    sl = Slice3D(b=B, m_A=MA, m_B=MB, P_A_vec=P_A_VEC, P_B_vec=P_B_VEC,
                 S_A_vec=S_A_VEC, S_B_vec=S_B_VEC)
    U, info = s3.newton_solve(prob, sl, tol=1e-11, max_iter=50)
    return prob, U, sl, info


@_oracle
@pytest.mark.slow
def test_psi_agreement_misaligned_spin_spectral(tp_misaligned):
    """ψ_PARASOL → ψ_TP SPECTRALLY for a genuinely non-axisymmetric slice.

    The agreement drops monotonically as the meridian/φ resolution grows; it
    floors at ~1e-7, set by the modified-Newton φ-mode-iteration residual
    (``solver_3d``), not the oracle.  (B1's full-Newton axisymmetric path
    reached ~5e-8; the 3-D minimal-break solver is one order looser.)
    """
    res = tp_misaligned
    sl = Slice3D(b=B, m_A=MA, m_B=MB, P_A_vec=P_A_VEC, P_B_vec=P_B_VEC,
                 S_A_vec=S_A_VEC, S_B_vec=S_B_VEC)
    errs = []
    for (Na, Nb, Nphi) in [(40, 28, 8), (48, 34, 8), (56, 40, 10)]:
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
        U, info = s3.newton_solve(prob, sl, tol=1e-11, max_iter=50)
        u = np.asarray(s3.evaluate_field(prob, U, QR, QZ, QP, B))
        psi = np.asarray(source.psi_BL_2c(QR, QZ, B, MA, MB)) + u
        e = float(np.max(np.abs(psi - res.psi)))
        errs.append(e)
        print(f"[E] Na={Na} Nb={Nb} Nphi={Nphi} its={info.iters} "
              f"resid={info.residual_norm:.2e}  |dpsi|={e:.3e}")
    errs = np.array(errs)
    assert np.all(np.diff(errs) < 0), f"ψ not converging: {errs}"
    assert errs[0] / errs[-1] > 5.0, f"ψ convergence too weak: {errs}"
    assert errs[-1] < 1e-6, f"ψ floor {errs[-1]:.2e} above 1e-6"


@_oracle
@pytest.mark.slow
def test_total_adm_mass_agreement_3d(tp_misaligned, parasol_misaligned_fine):
    """Total ADM mass agrees with TwoPunctures to the spectral floor (~1e-9)."""
    prob, U, sl, info = parasol_misaligned_fine
    M = d3.adm_mass_spectral_3d(prob, U, sl)
    assert abs(M - tp_misaligned.E) / tp_misaligned.E < 1e-8


@_oracle
@pytest.mark.slow
def test_adm_J_vector_agreement_3d(tp_misaligned):
    """The ADM J vector agrees with TwoPunctures (rotated to the PARASOL frame),
    and is genuinely nonzero / tilted off the collision axis."""
    res = tp_misaligned
    J_tp_parasol = np.array(cv.tp_vec_to_parasol(res.J))     # TP native -> PARASOL
    J_par = d3.adm_J_closed_form(B, P_A_VEC, P_B_VEC, S_A_VEC, S_B_VEC)
    J_surf = d3.adm_J_surface(B, P_A_VEC, P_B_VEC, S_A_VEC, S_B_VEC)
    print(f"[E] J TP(native)={res.J}  ->PARASOL={J_tp_parasol}  "
          f"closed={J_par}  surface={J_surf}")
    assert np.max(np.abs(J_par - J_tp_parasol)) < 1e-9, "J vs TP disagree"
    assert np.max(np.abs(J_surf - J_tp_parasol)) < 1e-9, "J surface vs TP disagree"
    # genuinely tilted (both the collision-axis and transverse components nonzero)
    assert abs(J_tp_parasol[0]) > 0.1 and abs(J_tp_parasol[2]) > 0.1
    assert np.max(np.abs(res.J)) > 0.1, "TP reported zero J"


@_oracle
@pytest.mark.slow
def test_aligned_spin_axisymmetric_regression():
    """Aligned spin (S∥z) is axisymmetric: the 3-D field stays φ-independent and
    ψ still matches TwoPunctures — the B1 regression with the vector interface."""
    Sz = 0.2
    S_A, S_B = (0.0, 0.0, Sz), (0.0, 0.0, 0.0)
    sl = Slice3D(b=B, m_A=MA, m_B=MB, P_A_vec=P_A_VEC, P_B_vec=P_B_VEC,
                 S_A_vec=S_A, S_B_vec=S_B)
    prob = s3.make_problem(Na=56, Nb=40, Nphi=8)
    U, info = s3.newton_solve(prob, sl, tol=1e-11, max_iter=50)
    Uarr = np.asarray(U)
    # aligned spin -> axisymmetric: φ-variation at the spectral floor
    phi_var = np.max(np.abs(Uarr - Uarr.mean(axis=2, keepdims=True))) / np.max(np.abs(Uarr))
    print(f"[E] aligned-spin φ-variation = {phi_var:.2e}")
    assert phi_var < 1e-9, f"aligned spin not axisymmetric: {phi_var:.2e}"
    # ψ matches TP (vector data with only the aligned component)
    res = tp.solve_parasol_points_3d(B, MA, MB, P_A_VEC, P_B_VEC, S_A, S_B,
                                     QR, QZ, QP, nA=64, nB=64, nphi=4)
    u = np.asarray(s3.evaluate_field(prob, U, QR, QZ, QP, B))
    psi = np.asarray(source.psi_BL_2c(QR, QZ, B, MA, MB)) + u
    assert np.max(np.abs(psi - res.psi)) < 1e-6
    # J is purely along the collision axis (= Sz), no tilt
    J = d3.adm_J_closed_form(B, P_A_VEC, P_B_VEC, S_A, S_B)
    assert abs(J[2] - Sz) < 1e-12 and abs(J[0]) < 1e-12 and abs(J[1]) < 1e-12
    assert np.max(np.abs(cv.tp_vec_to_parasol(res.J) - J)) < 1e-9
