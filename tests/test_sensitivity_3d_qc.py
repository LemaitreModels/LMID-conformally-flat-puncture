"""H5c acceptance — the quasi-circular (QC) certified-ID tangent ``dU/dθ_k``
(the QC extension of H5a; GRADIENT_ENHANCED_PLAN.md §4 H5 on the QC family).

The paper's astrophysical family is quasi-circular: the momenta are the PN
function ``quasicircular.qc_momenta(b, masses, spins)``, so a surrogate axis moves
them and the tangent must carry that chain rule.  Gates:
  * the jnp twin ``_qc_momenta_jax`` reproduces ``qc_momenta`` **bit-for-bit**;
  * ``certified_tangent_3d_qc`` matches the central FD of the certified QC solve to
    the FD-oracle floor, on ≥3 held-out QC slices per active axis (``b``, ``q``, a
    spin axis) — and the **direct** (H5a) tangent is materially WRONG for ``b``/``q``
    (missing the chain rule), the reason the QC tangent is needed;
  * a **planar** spin axis (``S_y=0``) has ``dP/dθ=0`` so the QC tangent reduces to
    the direct one; the **aligned** ``S_Ay`` activates the spin-orbit chain rule.

Standalone (numpy/scipy/jax); reuses the committed ``solver_3d``/``source_3d``/
``quasicircular`` and the H5a ``sensitivity_3d`` verbatim.
"""

import numpy as np
import pytest

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3d
from lm.initial_data.parametric import quasicircular as qcmod
from lm.initial_data.applications import sensitivity_3d as s3d
from lm.initial_data.applications import sensitivity_3d_qc as qc


M_TOT = 1.0
NA, NB, NPHI = 16, 12, 6
FIXED = {"qc": 1.0}


@pytest.fixture(scope="module")
def prob():
    return s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)


# ==========================================================================
# H5c-U1 — the jnp qc-momenta twin reproduces the numpy closed form bit-for-bit
# ==========================================================================
def test_qc_momenta_jax_bitforbit():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(8):
        b = rng.uniform(1.8, 6.0)
        q = rng.uniform(1.0, 3.0)
        mA, mB = q / (1 + q), 1.0 / (1 + q)
        SA = rng.uniform(-0.2, 0.2, 3)
        SB = rng.uniform(-0.2, 0.2, 3)
        PA, PB = qcmod.qc_momenta(b, mA, mB, SA, SB)
        ref = np.array([*PA, *PB])
        got = np.asarray(qc._qc_momenta_jax(np.array([b, mA, mB, *SA, *SB])))
        worst = max(worst, float(np.max(np.abs(got - ref))))
    assert worst < 1e-13, worst


# ==========================================================================
# H5c-T1 — QC tangent vs central FD of the certified QC solve (the chain rule)
# ==========================================================================
QC_CASES = [
    (["b", "q"], [np.array([3.1, 1.6]), np.array([2.4, 2.2]), np.array([2.0, 1.2])]),
    (["b", "S_x"], [np.array([2.6, 0.18]), np.array([3.0, 0.05]), np.array([2.2, 0.25])]),
]


def test_qc_tangent_vs_fd(prob):
    for active, thetas in QC_CASES:
        for theta in thetas:
            sl = p3d.theta_to_slice3d(theta, active, M_TOT, FIXED)
            U, info = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
            assert info.residual_norm < 1e-9
            asm = s3.assemble(prob, sl)
            for name in active:
                fd = qc.fd_tangent_3d_qc(prob, active, name, theta, M_TOT,
                                         fixed=FIXED, h=1e-4, solver="modified")
                t = np.asarray(qc.certified_tangent_3d_qc(prob, U, sl, name, M_TOT,
                                                          asm=asm, jac="nk"))
                rel = np.max(np.abs(t - fd)) / max(np.max(np.abs(fd)), 1e-30)
                assert rel < 1e-6, (active, name, theta, rel)


def test_qc_q_tangent_chi_nonzero(prob):
    """The q-tangent must carry the chi-fixed mass->spin chain (regression).

    On the q-axis the dimensionless spin ``chi`` is the fixed box coordinate, so the
    PHYSICAL spin ``S_X = chi_X m_X^2`` moves with the masses:
    ``dS_X/dq = chi_X d(m_X^2)/dq``.  The certified QC q-tangent must carry that
    chain in BOTH the qc-momenta spin-orbit term AND the Bowen-York spin source.
    ``test_qc_tangent_vs_fd`` only exercises ``q`` with the spins at their default
    zero (chi=0), where this chain is a no-op — so it never caught the chi-rebuild
    q-tangent bug.  With the chain dropped the q-tangent is O(1)-wrong at chi!=0
    (~8x vs FD) and exact only at chi=0; this test has teeth on that."""
    active = ["b", "q", "chi_Ay", "chi_By"]
    thetas = [np.array([2.5, 1.8, 0.4, -0.3]),
              np.array([4.5, 1.2, 0.6, 0.2]),
              np.array([3.0, 2.2, -0.5, 0.5])]
    for theta in thetas:
        sl = p3d.theta_to_slice3d(theta, active, M_TOT, FIXED)
        U, info = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
        assert info.residual_norm < 1e-9
        asm = s3.assemble(prob, sl)
        fd = qc.fd_tangent_3d_qc(prob, active, "q", theta, M_TOT,
                                 fixed=FIXED, h=1e-4, solver="modified")
        t = np.asarray(qc.certified_tangent_3d_qc(prob, U, sl, "q", M_TOT,
                                                  asm=asm, jac="nk"))
        rel = np.max(np.abs(t - fd)) / max(np.max(np.abs(fd)), 1e-30)
        assert rel < 1e-6, (theta.tolist(), rel)


def test_direct_tangent_is_wrong_for_qc_b_and_q(prob):
    """The H5a direct (fixed-momentum) tangent misses the qc chain rule → it is
    materially wrong for the QC ``b``/``q`` axes (the reason the QC tangent exists)."""
    active, theta = ["b", "q"], np.array([2.6, 1.7])
    sl = p3d.theta_to_slice3d(theta, active, M_TOT, FIXED)
    U, _ = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    asm = s3.assemble(prob, sl)
    for name in ("b", "q"):
        fd = qc.fd_tangent_3d_qc(prob, active, name, theta, M_TOT, fixed=FIXED,
                                 h=1e-4, solver="modified")
        t_dir = np.asarray(s3d.certified_tangent_3d(prob, U, sl, name, M_TOT, asm=asm, jac="nk"))
        rel_dir = np.max(np.abs(t_dir - fd)) / max(np.max(np.abs(fd)), 1e-30)
        assert rel_dir > 0.1, (name, rel_dir)     # the chain rule is decisive


# ==========================================================================
# H5c-U2 — planar spin: dP/dθ=0 ⇒ QC tangent reduces to the direct tangent
# ==========================================================================
def test_planar_spin_reduces_to_direct(prob):
    active, theta = ["b", "S_x"], np.array([2.6, 0.18])
    sl = p3d.theta_to_slice3d(theta, active, M_TOT, FIXED)
    U, _ = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    asm = s3.assemble(prob, sl)
    dP_A, dP_B = qc.dP_dtheta_qc(sl, "S_x", M_TOT)
    assert np.max(np.abs(dP_A)) < 1e-13 and np.max(np.abs(dP_B)) < 1e-13   # planar ⇒ no momenta change
    t_qc = np.asarray(qc.certified_tangent_3d_qc(prob, U, sl, "S_x", M_TOT, asm=asm, jac="nk"))
    t_dir = np.asarray(s3d.certified_tangent_3d(prob, U, sl, "S_x", M_TOT, asm=asm, jac="nk"))
    assert np.max(np.abs(t_qc - t_dir)) < 1e-12


# ==========================================================================
# H5c-U3 — aligned S_Ay: the spin-orbit chain rule is active (QC ≠ direct)
# ==========================================================================
def test_aligned_spin_orbit_chain_rule(prob):
    active, theta = ["b", "S_Ay"], np.array([3.0, 0.12])
    sl = p3d.theta_to_slice3d(theta, active, M_TOT, FIXED)
    U, _ = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    asm = s3.assemble(prob, sl)
    dP_A, _ = qc.dP_dtheta_qc(sl, "S_Ay", M_TOT)
    assert np.linalg.norm(dP_A) > 1e-4               # aligned spin moves the momenta
    fd = qc.fd_tangent_3d_qc(prob, active, "S_Ay", theta, M_TOT, fixed=FIXED,
                             h=1e-4, solver="modified")
    t_qc = np.asarray(qc.certified_tangent_3d_qc(prob, U, sl, "S_Ay", M_TOT, asm=asm, jac="nk"))
    t_dir = np.asarray(s3d.certified_tangent_3d(prob, U, sl, "S_Ay", M_TOT, asm=asm, jac="nk"))
    rel_qc = np.max(np.abs(t_qc - fd)) / max(np.max(np.abs(fd)), 1e-30)
    rel_dir = np.max(np.abs(t_dir - fd)) / max(np.max(np.abs(fd)), 1e-30)
    assert rel_qc < 1e-6, rel_qc
    assert rel_dir > 10 * rel_qc                     # direct misses the spin-orbit chain


# ==========================================================================
# H5c-U4 — dP/dθ conventions
# ==========================================================================
def test_dP_dtheta_conventions(prob):
    active, theta = ["b", "q"], np.array([2.6, 1.7])
    sl = p3d.theta_to_slice3d(theta, active, M_TOT, FIXED)
    for name in ("b", "q"):
        dP_A, dP_B = qc.dP_dtheta_qc(sl, name, M_TOT)
        assert np.linalg.norm(dP_A) > 1e-4           # b,q move the QC momenta
        assert np.allclose(dP_A, -dP_B)              # anti-symmetric (CoM frame)
        assert abs(dP_A[1]) < 1e-14                  # momenta stay in the x–z plane


# ==========================================================================
# H5c-T2 — add-only / standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lm.initial_data.applications.sensitivity_3d_qc as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm", "import context", "torch"):
        assert forbidden not in src, forbidden
