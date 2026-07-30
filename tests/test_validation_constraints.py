"""B1 Step 5 — oracle-independent FD constraint check (Cartesian evolution grid).

Exact self-tests pin the 2nd-order FD convergence:
  * single Schwarzschild puncture (psi=1+M/2r, K=0): H_continuum=0 exactly, so
    the generic FD Hamiltonian is pure truncation -> O(h^2);
  * transverse Bowen–York Â with K=psi^{-2}Â and ANY psi: the momentum constraint
    holds identically (D_j K^{ij}=psi^{-10}∂_j Â^{ij}=0), so M_FD -> O(h^2);
  * the generic FD Ricci and the conformal closed form R=-8psi^{-5}Δpsi agree.
Plus the real binary: LM-initial-data ID lands on the grid with O(h^2) constraint decay.
"""

import numpy as np
import pytest

from lm.initial_data.solver import solver_abt as sa, source
from lm.initial_data.validation import constraints as cst


def _order(hs, errs):
    """Least-squares log-log slope (FD convergence order)."""
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def _vec_evaluator_consistency():
    """The vectorized field evaluator must match solver_abt.evaluate_field_phys."""
    prob = sa.make_problem(Na=28, Nb=20, P=0.5)
    sl = sa.Slice(3.0, 0.5, 0.5)
    U, _ = sa.newton_solve(prob, sl, tol=1e-10, max_iter=20)
    rho = np.array([0.5, 1.2, 2.0, 4.0])
    z = np.array([1.0, -0.5, 2.0, 0.3])
    u_loop = np.asarray(sa.evaluate_field_phys(prob, U, rho, z, sl.b))
    u_vec = cst.evaluate_u_vec(prob, U, rho, z, sl.b)
    assert np.max(np.abs(u_loop - u_vec)) < 1e-12


def test_vectorized_evaluator_matches_solver():
    _vec_evaluator_consistency()


# --------------------------------------------------------------------------
# Schwarzschild self-test: H_FD is pure truncation -> O(h^2)
# --------------------------------------------------------------------------
def test_hamiltonian_FD_order_schwarzschild():
    M, L = 1.0, 4.0
    hs, errs, errs_conf = [], [], []
    for N in (24, 32, 48):
        x, X, Y, Z, h = cst.cartesian_grid(L, N)
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        r = np.where(r == 0, 1e-30, r)
        psi = 1.0 + M / (2.0 * r)
        A0 = np.zeros(X.shape + (3, 3))            # K=0 (time symmetric)
        H, _ = cst.fd_constraints_generic(psi, A0, h)
        Hc = cst.fd_hamiltonian_conformal(psi, np.zeros(X.shape), h)
        mask = cst.interior_mask(X, Y, Z, h, b=0.0, r_excl=0.8)
        hs.append(h)
        errs.append(cst.norms(H, mask)[0])
        errs_conf.append(cst.norms(Hc, mask)[0])
    hs, errs, errs_conf = np.array(hs), np.array(errs), np.array(errs_conf)
    assert _order(hs, errs) > 1.6, (hs, errs)            # ~2nd order (generic Ricci)
    assert errs[-1] < errs[0]                            # monotone decrease
    # the conformal closed form (R=-8psi^-5 Δpsi) ALSO converges at 2nd order
    assert _order(hs, errs_conf) > 1.6, (hs, errs_conf)
    assert errs_conf[-1] < errs_conf[0]


def test_generic_ricci_matches_analytic_R():
    """The generic FD Ricci reproduces the ANALYTIC scalar curvature of a smooth
    conformal metric gamma=psi^4 delta: R = -8 psi^{-5} Δpsi, converging O(h^2).

    Smooth manufactured psi = 1 + a exp(-(r/s)^2) (no puncture), so the continuum
    R is known in closed form and the generic FD Ricci is validated against ground
    truth (not merely against the conformal FD)."""
    a, s, L = 0.3, 2.0, 6.0
    hs, errs, rels = [], [], []
    for N in (24, 32, 48, 64):
        x, X, Y, Z, h = cst.cartesian_grid(L, N)
        r2 = X ** 2 + Y ** 2 + Z ** 2
        psi = 1.0 + a * np.exp(-r2 / s ** 2)
        lap_psi = a * np.exp(-r2 / s ** 2) * (4.0 * r2 / s ** 4 - 6.0 / s ** 2)
        R_analytic = -8.0 * psi ** (-5.0) * lap_psi          # K=0 -> H=R
        H, _ = cst.fd_constraints_generic(psi, np.zeros(X.shape + (3, 3)), h)
        mask = cst.interior_mask(X, Y, Z, h, b=0.0, r_excl=0.0)
        hs.append(h)
        errs.append(cst.norms(H - R_analytic, mask)[1])      # RMS
        rels.append(cst.norms(H - R_analytic, mask)[1] / cst.norms(R_analytic, mask)[1])
    hs, errs = np.array(hs), np.array(errs)
    assert _order(hs, errs) > 1.7, (hs, errs)        # 2nd-order to the true R
    assert rels[-1] < 0.025, rels                    # ~1.7% rel-RMS at N=64


# --------------------------------------------------------------------------
# Momentum self-test: transverse Â => M=0 for ANY psi -> O(h^2)
# --------------------------------------------------------------------------
def test_momentum_FD_order_transverse():
    b, mA, mB, P, L = 2.0, 0.5, 0.5, 0.5, 5.0
    hs, errs = [], []
    for N in (28, 40, 56):
        x, X, Y, Z, h = cst.cartesian_grid(L, N)
        rho = np.sqrt(X ** 2 + Y ** 2)
        psi = np.asarray(source.psi_BL_2c(rho, Z, b, mA, mB))   # u=0 (analytic)
        A = cst.A_tensor_3d(X, Y, Z, b, P)
        _, Mvec = cst.fd_constraints_generic(psi, A, h)
        mask = cst.interior_mask(X, Y, Z, h, b=b, r_excl=0.7)
        hs.append(h)
        errs.append(cst.vec_norms(Mvec, mask)[0])
    hs, errs = np.array(hs), np.array(errs)
    assert _order(hs, errs) > 1.6, (hs, errs)
    assert errs[-1] < errs[0]


def test_momentum_conformal_closed_form_small():
    """psi^{-10} ∂_j Â^{ij} -> 0 (Â flat-transverse); FD residual is truncation."""
    b, P, L, N = 2.0, 0.5, 5.0, 48
    x, X, Y, Z, h = cst.cartesian_grid(L, N)
    rho = np.sqrt(X ** 2 + Y ** 2)
    psi = np.asarray(source.psi_BL_2c(rho, Z, b, 0.5, 0.5))
    A = cst.A_tensor_3d(X, Y, Z, b, P)
    Mc = cst.fd_momentum_conformal(psi, A, h)
    mask = cst.interior_mask(X, Y, Z, h, b=b, r_excl=0.7)
    assert cst.vec_norms(Mc, mask)[0] < 1e-1     # truncation only; ->0 with h


# --------------------------------------------------------------------------
# The real binary: LM-initial-data ID lands on a Cartesian grid with O(h^2) decay
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_binary_constraints_converge():
    b, mA, mB, P, L = 3.0, 0.5, 0.5, 0.5, 9.0
    prob = sa.make_problem(Na=52, Nb=36, P=P)
    sl = sa.Slice(b, mA, mB)
    U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=25)
    assert info.residual_norm < 1e-9
    hs, eH, eM = [], [], []
    for N in (40, 56, 72, 88):                   # fine enough for the asymptotic regime
        x, X, Y, Z, h = cst.cartesian_grid(L, N)
        psi = cst.psi_on_grid(prob, U, sl, X, Y, Z)
        A = cst.A_tensor_3d(X, Y, Z, b, P)
        H, Mvec = cst.fd_constraints_generic(psi, A, h)
        mask = cst.interior_mask(X, Y, Z, h, b=b, r_excl=1.5)
        hs.append(h)
        eH.append(cst.norms(H, mask)[1])         # L2-RMS
        eM.append(cst.vec_norms(Mvec, mask)[1])
    hs = np.array(hs)
    assert _order(hs, np.array(eH)) > 1.7, (hs, eH)     # ~2nd order (H ~2.1)
    assert _order(hs, np.array(eM)) > 1.7, (hs, eM)     # ~2nd order (M ~2.0)
