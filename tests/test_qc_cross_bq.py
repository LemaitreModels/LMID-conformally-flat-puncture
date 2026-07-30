"""PARASOL M1 gate — the 5 new b/q second-order cross tangents
(:mod:`applications.sensitivity_3d_cross_bq`).

Teeth:
  * reduce-to-committed: the spin pair (chi_Ay,chi_By) through the dispatcher is
    BIT-FOR-BIT the committed ``sensitivity_3d_cross.cross_tangent_3d_qc``;
  * FD gate: each of the 5 new pairs matches a central FD of the first tangent
    along the other axis (FD-limited, ~1e-6);
  * autodiff gate: the closed-form ∂²Â²/∂θ_i∂θ_j and ∂²R/∂U² match jax autodiff of
    the analytic source, and the operator scales exactly as 1/b².

Standalone (numpy/scipy/jax); reuses the committed 3-D solver + first/second
tangents.  Small grid (Na=16,Nb=12,Nphi=6) — a genuinely non-axisymmetric QC slice.
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.solver import solver_3d_nk as s3nk
from lm.initial_data.solver import source_3d
from lm.initial_data.parametric.parametric_nd_3d import theta_to_slice3d
from lm.initial_data.applications import sensitivity_3d_qc as s3dqc
from lm.initial_data.applications import sensitivity_3d_cross as cross
from lm.initial_data.applications import sensitivity_3d_cross_bq as cbq

M_TOT = 1.0
NA, NB, NPHI = 16, 12, 6
FIXED = {"qc": 1.0}
ACTIVE = ["b", "q", "chi_Ay", "chi_By"]
THETA = np.array([3.5, 2.0, 0.4, 0.3])
NEW_PAIRS = [("b", "q"), ("b", "chi_Ay"), ("b", "chi_By"),
             ("q", "chi_Ay"), ("q", "chi_By")]


@pytest.fixture(scope="module")
def setup():
    prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)
    sl = theta_to_slice3d(THETA, ACTIVE, M_TOT, FIXED)
    U, _ = s3nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=40)
    U = np.asarray(U).reshape(prob.Ntot2d, prob.Nphi)
    asm = s3.assemble(prob, sl)
    return prob, sl, U, asm


def test_reduce_to_committed_spin_pair(setup):
    prob, sl, U, asm = setup
    a = cbq.cross_tangent_3d_qc_bq(prob, U, sl, "chi_Ay", "chi_By", M_TOT,
                                   asm=asm, jac="nk")
    b = cross.cross_tangent_3d_qc(prob, U, sl, "chi_Ay", "chi_By", M_TOT,
                                  asm=asm, jac="nk")
    assert np.max(np.abs(np.asarray(a) - np.asarray(b))) == 0.0  # bit-for-bit


@pytest.mark.parametrize("ni,nj", NEW_PAIRS)
def test_fd_gate(setup, ni, nj):
    prob, sl, U, asm = setup
    ana = np.asarray(cbq.cross_tangent_3d_qc_bq(prob, U, sl, ni, nj, M_TOT,
                                                asm=asm, jac="nk"))
    fd = cross.fd_cross_tangent_3d_qc(prob, ACTIVE, ni, nj, THETA, M_TOT,
                                      fixed=FIXED, h=1e-3)
    rel = np.linalg.norm(ana - fd) / (np.linalg.norm(fd) + 1e-300)
    assert rel < 5e-6, f"({ni},{nj}) relerr={rel:.3e}"


@pytest.mark.parametrize("ni,nj", NEW_PAIRS)
def test_symmetry(setup, ni, nj):
    prob, sl, U, asm = setup
    a = np.asarray(cbq.cross_tangent_3d_qc_bq(prob, U, sl, ni, nj, M_TOT,
                                              asm=asm, jac="nk"))
    b = np.asarray(cbq.cross_tangent_3d_qc_bq(prob, U, sl, nj, ni, M_TOT,
                                              asm=asm, jac="nk"))
    assert np.max(np.abs(a - b)) < 1e-9


def _a2_hessian(sl, asm, phi, node, k):
    b0 = sl.b
    X0 = np.array([asm.rho[node] * np.cos(phi[k]),
                   asm.rho[node] * np.sin(phi[k]), asm.z[node]])
    X0j = jnp.asarray(X0)

    def A2(theta):
        b, q, cAy, cBy = theta
        mA = M_TOT * q / (1.0 + q); mB = M_TOT / (1.0 + q)
        S_A = jnp.array([0.0, cAy * mA ** 2, 0.0])
        S_B = jnp.array([0.0, cBy * mB ** 2, 0.0])
        args = jnp.concatenate([jnp.array([b, mA, mB]), S_A, S_B])
        P = s3dqc._qc_momenta_jax(args)
        T = source_3d._A_full_tensor_jax_vec((b / b0) * X0j, b, P[0:3], P[3:6],
                                             S_A, S_B)
        return jnp.sum(T * T)

    t0 = jnp.array([sl.b, sl.m_A / sl.m_B, float(sl.S_A_vec[1]) / sl.m_A ** 2,
                    float(sl.S_B_vec[1]) / sl.m_B ** 2])
    return np.asarray(jax.jacfwd(jax.jacfwd(A2))(t0))


@pytest.mark.parametrize("ni,nj", NEW_PAIRS + [("chi_Ay", "chi_By")])
def test_autodiff_source_hessian(setup, ni, nj):
    prob, sl, U, asm = setup
    node = int(np.where(asm.interior)[0][asm.interior.sum() // 2])
    k = NPHI // 2
    H = _a2_hessian(sl, asm, prob.phi, node, k)
    idx = {"b": 0, "q": 1, "chi_Ay": 2, "chi_By": 3}
    _, _, A2_ij = cbq._source_second_derivs_bq(asm, prob.phi, sl, ni, nj, M_TOT)
    ana = float(A2_ij[node, k]); ad = float(H[idx[ni], idx[nj]])
    rel = abs(ana - ad) / (abs(ad) + 1e-300)
    assert rel < 1e-9, f"({ni},{nj}) analytic={ana:.6e} autodiff={ad:.6e} rel={rel:.3e}"


def test_operator_scaling(setup):
    prob, sl, U, asm = setup
    b1 = sl.b * 1.37
    sl1 = theta_to_slice3d(np.array([b1, 2.0, 0.4, 0.3]), ACTIVE, M_TOT, FIXED)
    asm1 = s3.assemble(prob, sl1)
    s0 = (cbq._lap_nodal(asm, prob, U) * sl.b ** 2)[asm.interior]
    s1 = (cbq._lap_nodal(asm1, prob, U) * b1 ** 2)[asm.interior]
    rel = np.max(np.abs(s0 - s1)) / (np.max(np.abs(s0)) + 1e-300)
    assert rel < 1e-9, f"operator 1/b^2 scaling rel={rel:.3e}"
