"""B3 acceptance — differentiability ``∂ID/∂θ`` (PAPER_PLAN §5 B3, claim 3).

JAX gradients of the certified/interpolated initial data w.r.t. the physical
parameters θ=(q,b,χ_A,χ_B), with two concrete uses:

  (a) a **gradient-based parameter solve** — hit a target ADM mass + spin by
      (damped) Gauss–Newton on the *analytic* ``∂F/∂θ`` (``applications.sensitivity``;
      the differentiable cousin of B2's gradient-free Broyden loop);
  (b) **sensitivity fields** ``∂ψ/∂χ_A`` (figure (viii)).

Gates (PAPER_PLAN §5 B3):
  * analytic gradients match finite differences to FD accuracy (O(h²));
  * — ideally — the surrogate gradient matches the *certified-ID* sensitivity
    (the solver's implicit-function tangent ``solver_abt.tangent_b/q`` and the new
    ``sensitivity.tangent_chi``);
  * the gradient-based target solve converges in few steps, certified ‖R‖∞ ≤ 1e-10;
  * ≥1 sensitivity figure produced (driver ``run_b3.py``).

Standalone (jax/numpy); reuses the frozen solver/validation/parametric and the B2
control module verbatim.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from lemaitre.initial_data.solver import solver_abt as sa
from lemaitre.initial_data.solver import source
from lemaitre.initial_data.parametric import parametric_nd_2c as p3
from lemaitre.initial_data.applications import control as ctl
from lemaitre.initial_data.applications import sensitivity as sen


P = 0.5
M_TOT = 1.0
FIXED_CHI = {"q": 1.5, "b": 4.0}        # T2-style spin problem (q=1.5 breaks χ_A↔χ_B)
RANGE_CHI = (0.0, 0.6)
BOX_CHI = np.array([[0.0, 0.0], [0.6, 0.6]])
THETA_STAR_CHI = np.array([0.52, 0.13])  # off-node interior target (B2's T2 θ*)
START_CHI = np.array([0.40, 0.15])       # B2's documented sensible start


# --------------------------------------------------------------------------
# Module-scoped fixtures (build the interpolants once)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prob():
    return sa.make_problem(Na=28, Nb=20, P=P)


@pytest.fixture(scope="module")
def ps_chi(prob):
    """(χ_A, χ_B) interpolant at q=1.5, b=4 (the spin target-solve / sensitivity grid)."""
    return p3.from_problem_nd(
        prob, [{"name": "chi_A", "min": 0.0, "max": 0.6, "Q": 8},
               {"name": "chi_B", "min": 0.0, "max": 0.6, "Q": 8}],
        M_tot=M_TOT, fixed=FIXED_CHI).build(tol=1e-12, max_iter=20)


@pytest.fixture(scope="module")
def ps_bq(prob):
    """(b, q) interpolant — for the b/q IFT-tangent cross-checks (no spin)."""
    return p3.from_problem_nd(
        prob, [{"name": "b", "min": 3.0, "max": 12.0, "Q": 12},
               {"name": "q", "min": 1.0, "max": 3.0, "Q": 10}],
        M_tot=M_TOT).build(tol=1e-12, max_iter=20)


# ==========================================================================
# B3-U1 — the JAX observable map F_jax reproduces B2's numpy ADM observables
# ==========================================================================
def test_F_jax_matches_numpy_observables(prob, ps_chi):
    cn, fixed = ("chi_A", "chi_B"), FIXED_CHI
    for theta in (np.array([0.413, 0.227]), np.array([0.137, 0.523]),
                  np.array([0.55, 0.05])):
        sl = p3.theta_to_slice(theta, cn, M_TOT, fixed)
        U_int = ps_chi.evaluate(theta)                # node-safe numpy interpolant
        for names in (("J", "M_ADM"), ("M_A", "M_B")):
            F = np.asarray(sen.build_F_jax(ps_chi, cn, names, prob, M_TOT, fixed)(
                jnp.asarray(theta)))
            ref = ctl.evaluate_observables(prob, U_int, sl, names)   # B2's validation.adm
            assert np.max(np.abs(F - ref)) < 1e-11, (names, F, ref)


# ==========================================================================
# B3-U2 — analytic ∂F/∂θ vs central finite differences (O(h²) — the FD gate)
# ==========================================================================
def test_dF_dtheta_vs_fd(prob, ps_chi):
    cn, fixed = ("chi_A", "chi_B"), FIXED_CHI
    F = sen.build_F_jax(ps_chi, cn, ("J", "M_ADM"), prob, M_TOT, fixed)
    theta = np.array([0.413, 0.227])                 # off-node
    Jac = np.asarray(sen.jacobian_F(F)(jnp.asarray(theta)))
    h = 1e-6
    fd = np.zeros((2, 2))
    for k in range(2):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        fd[:, k] = (np.asarray(F(jnp.asarray(tp))) - np.asarray(F(jnp.asarray(tm)))) / (2 * h)
    assert np.max(np.abs(Jac - fd)) / np.max(np.abs(fd)) < 1e-6
    # the analytic J-row is exactly (m_A², m_B²) (J = χ_A m_A² + χ_B m_B², linear)
    m_A, m_B = M_TOT * 1.5 / 2.5, M_TOT / 2.5
    assert np.allclose(Jac[0], [m_A ** 2, m_B ** 2], atol=1e-12)


def test_dU_dtheta_vs_fd(prob, ps_chi):
    """jacfwd of the full nodal field ``evaluate_jax`` vs central FD (O(h²))."""
    theta = np.array([0.371, 0.289])
    dU = sen.nodal_dU_dtheta(ps_chi, theta)          # (Na+1, Nb, 2)
    h = 1e-6
    for k in range(2):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        fd = (np.asarray(ps_chi.evaluate_jax(tp)) - np.asarray(ps_chi.evaluate_jax(tm))) / (2 * h)
        rel = np.max(np.abs(dU[..., k] - fd)) / max(np.max(np.abs(fd)), 1e-30)
        assert rel < 1e-6, (k, rel)


# ==========================================================================
# B3-U3 — the new analytic ∂Â²/∂S closed form vs autodiff of the source
# ==========================================================================
def test_dA2_spin_dS_vs_autodiff():
    b, Pmom, S_A, S_B = 4.0, 0.5, 0.18, 0.11
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.2, 5.0, 9)
    z = rng.uniform(-5.0, 5.0, 9)
    dSA_ad = np.array([float(jax.grad(lambda SA: source.A2_2c_spin(r, zz, b, Pmom, SA, S_B))(S_A))
                       for r, zz in zip(rho, z)])
    dSB_ad = np.array([float(jax.grad(lambda SB: source.A2_2c_spin(r, zz, b, Pmom, S_A, SB))(S_B))
                       for r, zz in zip(rho, z)])
    dSA_cf = sen.dA2_spin_dS(rho, z, b, S_A, S_B, "chi_A")
    dSB_cf = sen.dA2_spin_dS(rho, z, b, S_B, S_A, "chi_B")
    assert np.max(np.abs(dSA_cf - dSA_ad)) < 1e-12
    assert np.max(np.abs(dSB_cf - dSB_ad)) < 1e-12


# ==========================================================================
# B3-U4 — the surrogate gradient IS the certified-ID sensitivity (IFT tangents)
# ==========================================================================
def test_surrogate_grad_matches_certified_tangent_chi(prob, ps_chi):
    cn, fixed = ("chi_A", "chi_B"), FIXED_CHI
    theta = np.array([0.413, 0.227])                 # off-node, χ>0 (non-trivial tangent)
    dU = sen.nodal_dU_dtheta(ps_chi, theta)
    sl = p3.theta_to_slice(theta, cn, M_TOT, fixed)
    U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=25)
    assert info.residual_norm < 1e-9
    tA = sen.tangent_chi(prob, U, sl, "chi_A", M_TOT)
    tB = sen.tangent_chi(prob, U, sl, "chi_B", M_TOT)
    relA = np.max(np.abs(dU[..., 0] - tA)) / np.max(np.abs(tA))
    relB = np.max(np.abs(dU[..., 1] - tB)) / np.max(np.abs(tB))
    # interp-accuracy-limited agreement (the surrogate gradient == certified tangent)
    assert relA < 1e-4, relA
    assert relB < 1e-4, relB
    # certified_tangent dispatch reproduces tangent_chi exactly
    assert np.array_equal(sen.certified_tangent(prob, U, sl, "chi_A", M_TOT), tA)


def test_surrogate_grad_matches_certified_tangent_bq(prob, ps_bq):
    cn = ("b", "q")
    theta = np.array([6.337, 1.713])                 # off-node
    dU = sen.nodal_dU_dtheta(ps_bq, theta)
    sl = p3.theta_to_slice(theta, cn, M_TOT, None)
    U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=25)
    assert info.residual_norm < 1e-9
    tb = sa.tangent_b(prob, U, sl)
    tq = sa.tangent_q(prob, U, sl, M_TOT)
    relb = np.max(np.abs(dU[..., 0] - tb)) / np.max(np.abs(tb))
    relq = np.max(np.abs(dU[..., 1] - tq)) / np.max(np.abs(tq))
    assert relb < 1e-3, relb
    assert relq < 1e-3, relq


def test_tangent_chi_zero_at_no_spin(prob):
    """At χ=0 the spin source is 0 and ∂Â²/∂χ=0 ⇒ ∂U/∂χ=0 (first order)."""
    sl = p3.theta_to_slice(np.array([0.0, 0.0]), ("chi_A", "chi_B"), M_TOT, {"q": 1.5, "b": 4.0})
    U, _ = sa.newton_solve(prob, sl, tol=1e-12, max_iter=25)
    tA = sen.tangent_chi(prob, U, sl, "chi_A", M_TOT)
    assert np.max(np.abs(tA)) < 1e-13


# ==========================================================================
# B3-T1 — gradient-based target solve  (χ_A,χ_B)->(J,M_ADM)  (use (a), headline)
# ==========================================================================
def test_gradient_target_solve_spin(prob, ps_chi):
    cn, tn, fixed = ("chi_A", "chi_B"), ("J", "M_ADM"), FIXED_CHI
    F = sen.build_F_jax(ps_chi, cn, tn, prob, M_TOT, fixed)

    # (i) surrogate round-trip: invert the differentiable map to its own θ*
    target_s = np.asarray(F(jnp.asarray(THETA_STAR_CHI)))
    r1 = sen.gauss_newton_target(ps_chi, cn, tn, prob, target_s, START_CHI,
                                 M_tot=M_TOT, fixed=fixed, box=BOX_CHI, tol_ctrl=1e-11)
    assert r1.converged
    assert r1.steps <= 12
    assert np.max(np.abs(r1.theta - THETA_STAR_CHI)) < 1e-8
    assert r1.certified_residual <= 1e-10               # certified at the solution (R7)
    # the outer loop used the EXACT analytic Jacobian (no FD sweep) and only the
    # FINAL polish is a certified solve
    assert r1.n_jac <= 12

    # (ii) certified target (true forward map via B2 make_target): recover θ* to
    # the surrogate's interpolation accuracy, certified ‖R‖∞ ≤ 1e-10 at the answer
    cp = ctl.ControlProblem(prob, cn, tn, M_tot=M_TOT, fixed=fixed, box=BOX_CHI,
                            interpolant=ps_chi)
    target_c = ctl.make_target(cp, THETA_STAR_CHI)
    r2 = sen.gauss_newton_target(ps_chi, cn, tn, prob, target_c, START_CHI,
                                 M_tot=M_TOT, fixed=fixed, box=BOX_CHI, tol_ctrl=1e-11)
    assert r2.converged
    assert np.max(np.abs(r2.theta - THETA_STAR_CHI)) < 1e-5    # interp-limited
    assert r2.certified_residual <= 1e-10


# ==========================================================================
# B3-T2 — second target  (q,b)->(M_A,M_B)  converges from a generic start
# ==========================================================================
def test_gradient_target_solve_mass(prob, ps_bq):
    # ps_bq has axis order (b, q) — the control vars must match that order
    cn, tn = ("b", "q"), ("M_A", "M_B")
    box = np.array([[3.0, 1.0], [12.0, 3.0]])
    theta_star = np.array([3.6, 2.2])                  # (b, q)
    cp = ctl.ControlProblem(prob, cn, tn, M_tot=M_TOT, box=box, interpolant=ps_bq)
    target = ctl.make_target(cp, theta_star)
    r = sen.gauss_newton_target(ps_bq, cn, tn, prob, target, np.array([8.0, 1.3]),
                                M_tot=M_TOT, box=box, tol_ctrl=1e-9)
    assert r.converged
    assert r.steps <= 15
    assert np.max(np.abs(r.theta - theta_star)) < 1e-3        # interp-limited (soft b)
    assert r.certified_residual <= 1e-10


# ==========================================================================
# B3-T3 — sensitivity field ∂ψ/∂χ_A (figure (viii)): FD cross-check + physics
# ==========================================================================
def test_sensitivity_field_fd_and_physics(prob, ps_chi):
    cn, fixed, b = ("chi_A", "chi_B"), FIXED_CHI, 4.0
    theta = np.array([0.413, 0.227])

    # (i) FD cross-check at a physical point: ∂ψ/∂χ_A = ∂u/∂χ_A (ψ_BL χ-independent)
    rho0, z0 = np.array([0.5]), np.array([3.0])
    da = sen.sensitivity_psi(ps_chi, cn, theta, "chi_A", rho0, z0, prob, M_TOT, fixed)[0]
    h = 1e-6

    def psi_pt(th):
        u = sa.evaluate_field_phys(prob, ps_chi.evaluate(th), rho0, z0, b)[0]
        sl = p3.theta_to_slice(th, cn, M_TOT, fixed)
        return float(source.psi_BL_2c(rho0, z0, b, sl.m_A, sl.m_B)[0]) + u

    tp, tm = theta.copy(), theta.copy()
    tp[0] += h
    tm[0] -= h
    fd = (psi_pt(tp) - psi_pt(tm)) / (2 * h)
    assert abs(da - fd) / abs(fd) < 1e-6

    # (ii) ψ_BL is χ-independent ⇒ ∂ψ/∂χ_A == the ABT-interp of nodal ∂u/∂χ_A
    dU_nodal = sen.nodal_dU_dtheta(ps_chi, theta)[..., 0]
    direct = np.asarray(sa.evaluate_field_phys(prob, dU_nodal, rho0, z0, b))[0]
    assert abs(da - direct) < 1e-12

    # (iii) physics: ∂ψ/∂χ_A concentrates near puncture A (+b), not B (−b)
    near_A = abs(sen.sensitivity_psi(ps_chi, cn, theta, "chi_A",
                                     np.array([0.3]), np.array([+b]), prob, M_TOT, fixed)[0])
    near_B = abs(sen.sensitivity_psi(ps_chi, cn, theta, "chi_A",
                                     np.array([0.3]), np.array([-b]), prob, M_TOT, fixed)[0])
    assert near_A > 5.0 * near_B


def test_sensitivity_q_includes_background(prob, ps_bq):
    """∂ψ/∂q carries the analytic ψ_BL piece (chart is q-independent); FD-checked."""
    cn = ("b", "q")
    theta = np.array([6.337, 1.713])
    rho0, z0 = np.array([1.2]), np.array([2.0])
    da = sen.sensitivity_psi(ps_bq, cn, theta, "q", rho0, z0, prob, M_TOT, None)[0]
    h = 1e-6

    def psi_pt(th):
        b = th[0]                                       # chart depends only on b (fixed here)
        u = sa.evaluate_field_phys(prob, ps_bq.evaluate(th), rho0, z0, b)[0]
        sl = p3.theta_to_slice(th, cn, M_TOT, None)
        return float(source.psi_BL_2c(rho0, z0, b, sl.m_A, sl.m_B)[0]) + u

    tp, tm = theta.copy(), theta.copy()
    tp[1] += h                                          # perturb q (index 1)
    tm[1] -= h
    fd = (psi_pt(tp) - psi_pt(tm)) / (2 * h)
    assert abs(da - fd) / abs(fd) < 1e-5


def test_sensitivity_b_rejected(prob, ps_bq):
    """∂ψ/∂b is rejected by the numpy figure path (the ABT chart moves with b)."""
    with pytest.raises(ValueError):
        sen.sensitivity_psi(ps_bq, ("b", "q"), np.array([6.337, 1.713]), "b",
                            np.array([1.2]), np.array([2.0]), prob, M_TOT, None)


# ==========================================================================
# B3-T4 — add-only / standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lemaitre.initial_data.applications.sensitivity as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm", "import context", "torch"):
        assert forbidden not in src, forbidden
