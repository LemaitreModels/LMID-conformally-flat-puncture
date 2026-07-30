"""M1-A acceptance — two-centre Newton solve on the single A-centred grid.

Primary correctness gate: the P=0 time-symmetric two-puncture slice is an EXACT
fixed point (Â=0 => u≡0 to machine zero).  Plus: Newton converges to
‖R‖∞ ≤ 1e-10 for P≠0, the analytic Jacobian matches autodiff, max|u| ~ P^2, and
the equal-mass solution is (approximately, on this B-limited grid) z-even.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from lemaitre.initial_data.solver import solver as psolver
from lemaitre.initial_data.solver import diagnostics as diag
from lemaitre.initial_data.solver.solver import Slice


# --------------------------------------------------------------------------
# M1-A.0 — PRIMARY GATE: P=0 is an EXACT fixed point (u ≡ 0)
# --------------------------------------------------------------------------
def test_P0_exact_fixed_point():
    """P=0 (time-symmetric Brill–Lindquist) => u ≡ 0 to machine zero."""
    prob = psolver.make_problem(N=28, L=2.0, L_theta=8, P=0.0)
    sl = Slice(b=3.0, m_A=0.5, m_B=0.5)
    U, info = psolver.newton_solve(prob, sl, tol=1e-14, max_iter=10)
    assert float(jnp.max(jnp.abs(U))) == 0.0, \
        f"P=0 gave nonzero u (max|u|={float(jnp.max(jnp.abs(U))):.2e})"
    assert info.residual_norm == 0.0, f"P=0 residual {info.residual_norm:.2e} != 0"
    # idempotent: re-solving from the (zero) solution stays zero
    U2, info2 = psolver.newton_solve(prob, sl, U0=U, tol=1e-14, max_iter=5)
    assert float(jnp.max(jnp.abs(U2))) == 0.0


def test_P0_fixed_point_unequal_mass():
    """The P=0 fixed point holds for unequal masses too (pure BL, u≡0)."""
    prob = psolver.make_problem(N=24, L=2.0, L_theta=8, P=0.0)
    sl = Slice(b=2.5, m_A=0.7, m_B=0.3)
    U, info = psolver.newton_solve(prob, sl, tol=1e-14, max_iter=10)
    assert float(jnp.max(jnp.abs(U))) == 0.0
    assert info.residual_norm == 0.0


# --------------------------------------------------------------------------
# M1-A.1 — Newton converges for P != 0
# --------------------------------------------------------------------------
def test_newton_converges_equal_mass():
    """Equal-mass head-on, P!=0: ‖R‖∞ ≤ 1e-10 from a cold start, few iters."""
    prob = psolver.make_problem(N=32, L=2.0, L_theta=8, P=0.5)
    sl = Slice(b=3.0, m_A=0.5, m_B=0.5)
    U, info = psolver.newton_solve(prob, sl, tol=1e-10, max_iter=20)
    assert info.converged, f"Newton did not converge: ‖R‖={info.residual_norm:.2e}"
    assert info.residual_norm <= 1e-10, f"‖R‖∞ = {info.residual_norm:.2e}"
    assert info.iters <= 12, f"took {info.iters} iters"
    # genuinely nontrivial solution (the infall correction)
    assert float(jnp.max(jnp.abs(U))) > 1e-3


def test_newton_quadratic_history():
    prob = psolver.make_problem(N=32, L=2.0, L_theta=8, P=0.5)
    sl = Slice(b=3.0, m_A=0.5, m_B=0.5)
    _U, info = psolver.newton_solve(prob, sl, tol=1e-10, max_iter=20)
    h = info.history
    assert h[-1] <= 1e-10
    drops = [h[i] / h[i + 1] for i in range(len(h) - 1)]
    assert max(drops) > 1e2, f"no fast (quadratic) step (max drop {max(drops):.1e})"


# --------------------------------------------------------------------------
# M1-A.2 — analytic Jacobian matches autodiff (tiny grid)
# --------------------------------------------------------------------------
def test_analytic_jacobian_matches_jacfwd():
    prob = psolver.make_problem(N=8, L=1.5, L_theta=4, P=0.7)
    sl = Slice(b=2.0, m_A=0.6, m_B=0.4)
    rng = np.random.default_rng(3)
    U = jnp.asarray(0.01 * rng.normal(size=prob.shape))   # generic point

    J_analytic = psolver.jacobian(prob, U, sl)

    def res_flat(uvec):
        return psolver.residual(prob, uvec.reshape(prob.shape), sl).ravel()

    J_ad = jax.jacfwd(res_flat)(U.ravel())
    err = float(jnp.max(jnp.abs(J_analytic - J_ad)))
    scale = float(jnp.max(jnp.abs(J_ad)))
    assert err / scale < 1e-9, f"analytic vs AD Jacobian rel error {err/scale:.2e}"


# --------------------------------------------------------------------------
# M1-A.3 — P -> 0  =>  max|u| ~ P^2
# --------------------------------------------------------------------------
def test_u_quadratic_in_P():
    """‖u‖ ~ P^2 for small P (the source ∝ Â² ∝ P², cross term included).

    Uses the global modal amplitude ‖U‖_F (robust to the B-under-resolved spike)
    and small P (deep in the linear regime; at larger P nonlinear saturation
    pulls the halving ratio below 4).
    """
    Ps = [0.2, 0.1, 0.05]
    amp = []
    for P in Ps:
        prob = psolver.make_problem(N=32, L=2.0, L_theta=8, P=P)
        sl = Slice(b=3.0, m_A=0.5, m_B=0.5)
        U, info = psolver.newton_solve(prob, sl, tol=1e-10, max_iter=20)
        assert info.converged
        amp.append(float(jnp.linalg.norm(U)))
    amp = np.array(amp)
    r1, r2 = amp[0] / amp[1], amp[1] / amp[2]
    assert amp[-1] < amp[0]
    assert 3.6 < r1 < 4.3, f"‖U‖ ratio (P/2) = {r1:.3f}, expected ~4"
    assert 3.6 < r2 < 4.3, f"‖U‖ ratio (P/2) = {r2:.3f}, expected ~4"


# --------------------------------------------------------------------------
# M1-A.4 — B-limitation evidence: the equal-mass z-asymmetry is O(10%)
# --------------------------------------------------------------------------
def test_equal_mass_B_limitation_evidence():
    """The single A-centred grid is B-limited: the *exactly z-even* equal-mass
    slice is reproduced with an O(10%) z-asymmetry, concentrated near B.

    The physical equal-mass head-on slice is exactly even in z.  On the
    A-centred grid the discrete solution is well-resolved near A but poorly near
    B (the under-resolved interior axis point r_A=2b, μ_A=-1, where the
    Legendre-in-μ_A expansion of B's near-zone has a pole touching μ_A=-1).  We
    quantify the asymmetry near A vs near B — concrete, measured motivation for
    the M2-A mortar/ball grid.  (M1-A's correctness gate is the P=0 fixed point
    above; this is a documented limitation, not a correctness failure.)
    """
    prob = psolver.make_problem(N=40, L=2.0, L_theta=10, P=0.5)
    sl = Slice(b=3.0, m_A=0.5, m_B=0.5)
    U, info = psolver.newton_solve(prob, sl, tol=1e-10, max_iter=20)
    assert info.converged

    def to_Acoords(rho, z):
        rA = np.hypot(rho, z - sl.b)
        muA = (z - sl.b) / rA
        return rA, muA

    # sample points straddling A and B; |u(rho,z)-u(rho,-z)| measures the
    # A-vs-B resolution mismatch (the exact solution is z-even => 0).
    rho = np.array([0.2, 0.5, 1.0, 2.0, 0.5, 1.0])
    z = np.array([1.0, 1.5, 0.5, 1.0, 2.5, 3.0])
    rA1, mu1 = to_Acoords(rho, z)
    rA2, mu2 = to_Acoords(rho, -z)
    u_pos = diag.evaluate_field_points(prob, U, rA1, mu1)
    u_neg = diag.evaluate_field_points(prob, U, rA2, mu2)
    asym = np.max(np.abs(u_pos - u_neg)) / np.max(np.abs(u_pos))
    print(f"\n[M1-A] equal-mass z-asymmetry (A-centred grid, B-limited) = {asym:.3e}")
    # the solve is sane (bounded), but the B-limitation is real and O(10%)
    assert asym < 0.6, f"asymmetry {asym:.3e} unexpectedly large (solver bug?)"
    assert asym > 0.05, f"asymmetry {asym:.3e}: expected the B-limitation to show"
