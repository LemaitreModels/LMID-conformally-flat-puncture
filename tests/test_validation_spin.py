"""P2 bonus — external validation of the ALIGNED-SPIN slice against TwoPunctures.

A second external-validation point (after B1's head-on): the aligned-spin
Bowen–York source is solved by the same real TwoPunctures oracle natively
(``par_S_plus/minus``), so LM-initial-data's spinning ψ and the angular momentum J can
be checked against an independent field-standard code.

Skipped cleanly if the compiled TwoPunctures binary is absent.  Marked ``slow``
(each TwoPunctures solve is ~10-30 s).  The binary must be the spin-enabled
build (``main.c`` reads optional ``SA SB`` argv; rebuild via build.sh's clang
step) — older binaries ignore the extra args and would (correctly) report a
spin mismatch, caught by the J check below.
"""

import numpy as np
import pytest

from lm.initial_data.solver import solver_abt as sa, source
from lm.initial_data.validation import adm, conventions, twopunctures as tp

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not tp.available(),
                       reason="TwoPunctures binary not built (see build.sh)"),
]

B, MA, MB, P = 3.0, 0.5, 0.5, 0.5
SA, SB = 0.3, 0.3                       # equal aligned spins (z-even slice)


def test_conventions_spin_mapping():
    """The S->par_S axis mapping is pinned (collision z -> TP x; sign + axis)."""
    par = conventions.lm_initial_data_to_tp(B, MA, MB, P, S_A=SA, S_B=SB)
    assert par.par_S_plus == (SA, 0.0, 0.0)
    assert par.par_S_minus == (SB, 0.0, 0.0)
    # no-spin default still maps to zero spin (B1 unchanged)
    assert conventions.lm_initial_data_to_tp(B, MA, MB, P).par_S_plus == (0.0, 0.0, 0.0)


def test_spin_angular_momentum_vs_tp():
    """TwoPunctures reports J = (S_A+S_B, 0, 0) along the collision (x) axis —
    pins the spin magnitude AND the (proper-rotation) sign of the S->par_S map."""
    res = tp.solve_tp(B, MA, MB, P, np.array([[B, 0, 0]]),
                      nA=48, nB=48, nphi=4, S_A=SA, S_B=SB)
    assert abs(res.J[0] - (SA + SB)) < 1e-9, f"J_x={res.J[0]} != {SA+SB}"
    assert abs(res.J[1]) < 1e-12 and abs(res.J[2]) < 1e-12
    # anti-aligned: J cancels
    res2 = tp.solve_tp(B, MA, MB, P, np.array([[B, 0, 0]]),
                       nA=48, nB=48, nphi=4, S_A=SA, S_B=-SA)
    assert abs(res2.J[0]) < 1e-9, f"anti-aligned J_x={res2.J[0]} != 0"


def test_spin_axisymmetry_nphi():
    """Aligned spin keeps the data axisymmetric: nphi=4 reproduces nphi=12 in ψ
    (the m=0-only Fourier content), validating the 2-D (axisymmetric) treatment."""
    rho = np.array([0.5, 1.0]) * B
    z = np.array([0.8, -0.4]) * B
    r4 = tp.solve_lm_initial_data_points(B, MA, MB, P, rho, z, nA=48, nB=48, nphi=4,
                                 S_A=SA, S_B=SB)
    r12 = tp.solve_lm_initial_data_points(B, MA, MB, P, rho, z, nA=48, nB=48, nphi=12,
                                  S_A=SA, S_B=SB)
    assert np.max(np.abs(r4.psi - r12.psi)) < 1e-10


def test_spin_psi_agreement_vs_tp():
    """LM-initial-data's spinning ψ agrees with TwoPunctures at shared meridian points.

    The agreement floors at the (steeper) spin two-centre spatial floor, not the
    head-on 1e-12 (PAPER_PLAN R7); both spectral codes converge to the same
    analytic solution at the well-resolved interior probes.
    """
    prob = sa.make_problem(Na=64, Nb=44, P=P)
    sl = sa.Slice(B, MA, MB, S_A=SA, S_B=SB)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=30)
    assert info.residual_norm < 1e-7, f"LM-initial-data spin solve ‖R‖={info.residual_norm:.2e}"
    rho = np.array([0.30, 0.60, 0.30, 0.90, 2.0]) * B
    z = np.array([1.10, 0.50, -1.10, 0.60, 0.40]) * B
    res = tp.solve_lm_initial_data_points(B, MA, MB, P, rho, z, nA=64, nB=64, nphi=4,
                                  S_A=SA, S_B=SB)
    u = np.asarray(sa.evaluate_field_phys(prob, U, rho, z, B))
    psi = np.asarray(source.psi_BL_2c(rho, z, B, MA, MB)) + u
    err = np.max(np.abs(psi - res.psi))
    print(f"\n[P2] spinning ψ: max|ψ_LM-initial-data - ψ_TP| = {err:.3e}")
    assert err < 1e-6, f"spinning ψ agreement {err:.2e}"
