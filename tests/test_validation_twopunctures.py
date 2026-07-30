"""B1 Step 4 — agreement with the external TwoPunctures oracle.

Skipped cleanly if the compiled TwoPunctures binary is absent (build it with
``~/.cache/bbhfm/parasol_tp_oracle/build.sh`` or set ``LM_TP_BIN``), so the
oracle-independent B1 deliverables still run everywhere.  Marked ``slow`` (each
TwoPunctures solve is ~10-30 s).
"""

import numpy as np
import pytest

from lm.initial_data.solver import solver_abt as sa, source
from lm.initial_data.validation import adm, twopunctures as tp

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not tp.available(),
                       reason="TwoPunctures binary not built (see build.sh)"),
]

B, MA, MB, P = 3.0, 0.5, 0.5, 0.5


@pytest.fixture(scope="module")
def lm_initial_data_solve():
    prob = sa.make_problem(Na=64, Nb=44, P=P)
    sl = sa.Slice(B, MA, MB)
    U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=30)
    assert info.residual_norm < 1e-9
    return prob, U, sl


def test_psi_agreement_on_shared_points(lm_initial_data_solve):
    """psi agrees with TwoPunctures at shared points to the two-centre floor."""
    prob, U, sl = lm_initial_data_solve
    rho = np.array([0.30, 0.60, 0.30, 0.90, 2.0]) * B
    z = np.array([1.10, 0.50, -1.10, 0.60, 0.40]) * B
    res = tp.solve_lm_initial_data_points(B, MA, MB, P, rho, z, nA=64, nB=64, nphi=4)
    u = np.asarray(sa.evaluate_field_phys(prob, U, rho, z, B))
    psi = np.asarray(source.psi_BL_2c(rho, z, B, MA, MB)) + u
    assert np.max(np.abs(psi - res.psi)) < 5e-8


def test_total_adm_mass_agreement(lm_initial_data_solve):
    prob, U, sl = lm_initial_data_solve
    res = tp.solve_tp(B, MA, MB, P, np.array([[B, 0, 0]]), nA=64, nB=64, nphi=4)
    M = adm.adm_mass_spectral(prob, U, sl)      # spectral boundary extraction
    assert abs(M - res.E) / res.E < 1e-9        # ~1e-11 achieved at 64x44


def test_individual_puncture_mass_agreement(lm_initial_data_solve):
    prob, U, sl = lm_initial_data_solve
    res = tp.solve_tp(B, MA, MB, P, np.array([[B, 0, 0]]), nA=64, nB=64, nphi=4)
    MA_ = adm.puncture_adm_mass(prob, U, sl, "A")
    assert abs(MA_ - res.mp_adm) / res.mp_adm < 5e-4


def test_no_angular_momentum_head_on(lm_initial_data_solve):
    """Head-on data carries no angular momentum (TwoPunctures J = 0)."""
    res = tp.solve_tp(B, MA, MB, P, np.array([[B, 0, 0]]), nA=48, nB=48, nphi=4)
    assert np.max(np.abs(res.J)) < 1e-12
    # LM-initial-data convention check: per-puncture linear momenta are exactly +-P
    assert abs(adm.by_momentum_gauss(B, P, "A") - (-P)) < 1e-9
    assert abs(adm.by_momentum_gauss(B, P, "B") - (+P)) < 1e-9
