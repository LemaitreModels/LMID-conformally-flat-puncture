"""Guard: ``qc_chi_tangent.tangent_qc_chi`` must stay a PASS-THROUGH to
``sensitivity_3d_qc.certified_tangent_3d_qc`` — no correction added.

Since commit 25d120e the base tangent already carries the held-chi mass->spin chain
on the q axis, so any term added by this wrapper would DOUBLE-COUNT and silently
poison the q-axis Hermite interpolant (~2.7e-2 vs the correct ~2.6e-7).  These tests
have teeth on that: they fail bit-for-bit if a correction is re-introduced, and they
exercise the q axis at chi != 0 (where the old double-count was O(1)-wrong) plus b and
the spin axes.

Standalone (numpy/scipy/jax); reuses the committed solver + the base QC tangent.
"""
import numpy as np
import pytest

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3d
from lm.initial_data.applications import sensitivity_3d_qc as qc
from lm.initial_data.pipeline.qc_chi_tangent import tangent_qc_chi

M_TOT = 1.0
NA, NB, NPHI = 16, 12, 6
FIXED = {"qc": 1.0}
ACTIVE = ["b", "q", "chi_Ay", "chi_By"]
# chi != 0 on both holes — where the old wrapper double-counted the q chain O(1).
THETAS = [np.array([2.5, 1.8, 0.4, -0.3]),
          np.array([4.5, 1.2, 0.6, 0.2]),
          np.array([3.0, 2.2, -0.5, 0.5])]


@pytest.fixture(scope="module")
def prob():
    return s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)


def test_wrapper_is_passthrough(prob):
    """tangent_qc_chi == certified_tangent_3d_qc for EVERY axis (to the GMRES noise
    floor).  The wrapper adds nothing, so the only difference is two independent
    inner-GMRES solves of the same system (multithreaded → ~1e-11 relative jitter,
    NOT bit-identical).  A re-introduced correction would blow the q axis to ~1e0
    relative — ~9 orders above this floor, so the teeth are intact."""
    worst = {name: 0.0 for name in ACTIVE}
    for theta in THETAS:
        sl = p3d.theta_to_slice3d(theta, ACTIVE, M_TOT, FIXED)
        U, info = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
        assert info.residual_norm < 1e-9
        asm = s3.assemble(prob, sl)
        for name in ACTIVE:
            base = np.asarray(qc.certified_tangent_3d_qc(prob, U, sl, name, M_TOT,
                                                         asm=asm, jac="nk"))
            wrap = np.asarray(tangent_qc_chi(prob, U, sl, name, M_TOT,
                                             asm=asm, jac="nk"))
            rel = np.max(np.abs(wrap - base)) / max(np.max(np.abs(base)), 1e-30)
            worst[name] = max(worst[name], float(rel))
    for name in ACTIVE:
        assert worst[name] < 1e-7, (name, worst[name])


def test_wrapper_q_tangent_matches_fd_chi_nonzero(prob):
    """End-to-end: the wrapper q-tangent (== base) matches the FD oracle at chi != 0.
    A re-introduced double-count would blow this to O(1) (the historical ~8x)."""
    for theta in THETAS:
        sl = p3d.theta_to_slice3d(theta, ACTIVE, M_TOT, FIXED)
        U, info = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
        assert info.residual_norm < 1e-9
        asm = s3.assemble(prob, sl)
        fd = qc.fd_tangent_3d_qc(prob, ACTIVE, "q", theta, M_TOT,
                                 fixed=FIXED, h=1e-4, solver="modified")
        t = np.asarray(tangent_qc_chi(prob, U, sl, "q", M_TOT, asm=asm, jac="nk"))
        rel = np.max(np.abs(t - fd)) / max(np.max(np.abs(fd)), 1e-30)
        assert rel < 1e-6, (theta.tolist(), rel)
