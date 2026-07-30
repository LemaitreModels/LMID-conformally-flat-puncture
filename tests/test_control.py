"""B2 acceptance — accelerated parameter control (the payoff; rebuttal to R5).

A Mendes-style Broyden control loop on ``G(θ)=F(θ)−target`` (``applications.control``)
over two representative head-on(+aligned-spin) targets, run **cold** (each solve
from scratch) vs **warm** (each solve seeded by the P3 interpolant ``ps.evaluate``).
Gates (PAPER_PLAN §5 B2):

  * the control loop converges to the target (and the **cold** loop reproduces the
    directly-targeted free data — the known-answer round-trip);
  * warm-starting reduces inner Newton iterations / solver work by a clear factor;
  * the certified constraint residual ``‖R‖∞ ≤ 1e-10`` is met at **every** control
    step in **both** modes (the speed-up is never bought with accuracy);
  * the outer loop is identical across modes (``calls_match``) so the reduction is
    a clean per-call factor.

Two targets (control vars are a subset of θ=(q,b,χ_A,χ_B), so the P3 interpolant
warm-starts directly):
  T1  control (q,b)        -> (M_A, M_B)   individual ADM masses (no spin; canonical
                                            mass control).  θ* near the small-b corner.
  T2  control (χ_A,χ_B)    -> (J, M_ADM)   total spin + total ADM mass (unequal mass
                                            q=1.5 breaks the χ_A↔χ_B swap degeneracy
                                            so the round-trip is unique; off-corner θ*).

Standalone (jax/numpy); reuses the frozen solver/validation/parametric verbatim.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from lm.initial_data.solver import solver_abt as sa
from lm.initial_data.validation import adm
from lm.initial_data.parametric import parametric_nd_2c as p3
from lm.initial_data.applications import control as ctl


P = 0.5
TOL_INNER = 1e-10          # the certified-residual gate
TOL_CTRL = 1e-9            # outer control-loop convergence


# --------------------------------------------------------------------------
# Module-scoped fixtures: build the spatial Problem + the two interpolants once
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prob():
    return sa.make_problem(Na=36, Nb=24, P=P)


@pytest.fixture(scope="module")
def cp_mass(prob):
    """T1: control (q,b) -> (M_A, M_B), no spin."""
    ranges = {"q": (1.0, 3.0), "b": (3.0, 12.0)}
    ps = ctl.build_interpolant(prob, ("q", "b"), ranges, {"q": 8, "b": 10}, M_tot=1.0)
    cp = ctl.ControlProblem(prob, ("q", "b"), ("M_A", "M_B"), M_tot=1.0,
                            box=ctl.box_from_ranges(("q", "b"), ranges), interpolant=ps)
    theta_star = np.array([2.2, 3.6])      # off-centre, toward the small-b/unequal corner
    target = ctl.make_target(cp, theta_star)
    return dict(cp=cp, theta_star=theta_star, target=target, start=np.array([1.3, 8.0]),
               ranges=ranges)


@pytest.fixture(scope="module")
def cp_spin(prob):
    """T2: control (χ_A,χ_B) -> (J, M_ADM), unequal mass q=1.5 (unique round-trip)."""
    ranges = {"chi_A": (0.0, 0.6), "chi_B": (0.0, 0.6)}
    fixed = {"q": 1.5, "b": 4.0}
    ps = ctl.build_interpolant(prob, ("chi_A", "chi_B"), ranges, {"chi_A": 8, "chi_B": 8},
                               M_tot=1.0, fixed=fixed)
    cp = ctl.ControlProblem(prob, ("chi_A", "chi_B"), ("J", "M_ADM"), M_tot=1.0, fixed=fixed,
                            box=ctl.box_from_ranges(("chi_A", "chi_B"), ranges), interpolant=ps)
    theta_star = np.array([0.55, 0.10])    # off-corner: high χ_A, low χ_B (single-spin-ish)
    target = ctl.make_target(cp, theta_star)
    return dict(cp=cp, theta_star=theta_star, target=target, start=np.array([0.40, 0.15]),
               ranges=ranges)


# ==========================================================================
# B2-U1 — the observable map F reuses validation.adm verbatim (sanity)
# ==========================================================================
def test_observable_map_matches_adm(prob):
    sl = p3.theta_to_slice([1.7, 5.0], ("q", "b"), M_tot=1.0)
    U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=30)
    F = ctl.evaluate_observables(prob, U, sl, ("M_A", "M_B", "M_ADM"))
    assert F[0] == adm.puncture_adm_mass(prob, U, sl, "A")
    assert F[1] == adm.puncture_adm_mass(prob, U, sl, "B")
    assert F[2] == adm.adm_mass_spectral(prob, U, sl)
    # J is analytic in the spins
    sl2 = p3.theta_to_slice([0.4, 0.2], ("chi_A", "chi_B"), M_tot=1.0, fixed={"q": 1.5, "b": 4.0})
    assert ctl.OBSERVABLES["J"](prob, None, sl2) == pytest.approx(sl2.S_A + sl2.S_B, abs=0)


# ==========================================================================
# B2-U2 — the warm-start mechanism: interpolant guess cuts inner Newton iters
# ==========================================================================
def test_interp_guess_cuts_inner_iters(cp_mass):
    cp = cp_mass["cp"]
    for theta in (np.array([1.37, 5.2]), np.array([2.3, 4.1]), np.array([1.1, 9.7])):
        sl = cp.slice_at(theta)
        _, ic = sa.newton_solve(cp.prob, sl, U0=None, tol=TOL_INNER, max_iter=30)
        g = np.asarray(cp.interpolant.evaluate(theta))
        _, iw = sa.newton_solve(cp.prob, sl, U0=g, tol=TOL_INNER, max_iter=30)
        assert iw.iters < ic.iters, f"warm {iw.iters} !< cold {ic.iters} at {theta}"
        assert iw.residual_norm <= TOL_INNER and ic.residual_norm <= TOL_INNER


# ==========================================================================
# B2-T1 — known-answer round-trip on the COLD loop (the cold-loop sanity)
# ==========================================================================
def test_roundtrip_cold_mass(cp_mass):
    cp, ts, tgt, start = cp_mass["cp"], cp_mass["theta_star"], cp_mass["target"], cp_mass["start"]
    r = ctl.broyden_control(cp, start, tgt, mode="cold", tol_ctrl=TOL_CTRL, tol_inner=TOL_INNER)
    assert r["converged"], f"cold loop did not converge: ||G||={r['ctrl_residual']:.2e}"
    assert np.max(np.abs(r["theta"] - ts)) < 1e-5, f"recovered {r['theta']} != {ts}"
    assert r["counters"].max_resid <= TOL_INNER


def test_roundtrip_cold_spin(cp_spin):
    cp, ts, tgt, start = cp_spin["cp"], cp_spin["theta_star"], cp_spin["target"], cp_spin["start"]
    r = ctl.broyden_control(cp, start, tgt, mode="cold", tol_ctrl=TOL_CTRL, tol_inner=TOL_INNER)
    assert r["converged"], f"cold loop did not converge: ||G||={r['ctrl_residual']:.2e}"
    assert np.max(np.abs(r["theta"] - ts)) < 1e-4, f"recovered {r['theta']} != {ts}"
    assert r["counters"].max_resid <= TOL_INNER


# ==========================================================================
# B2-T2 — warm-start reduces inner iters / LU solves; certified at every step;
#         identical outer path (calls_match); same converged free data
# ==========================================================================
@pytest.mark.parametrize("fixture_name", ["cp_mass", "cp_spin"])
def test_warm_start_speedup_and_certification(fixture_name, request):
    bundle = request.getfixturevalue(fixture_name)
    cp, tgt, start = bundle["cp"], bundle["target"], bundle["start"]
    res = ctl.run_comparison(cp, start, tgt, modes=("cold", "interp", "continuation"),
                             tol_ctrl=TOL_CTRL, tol_inner=TOL_INNER)
    cold, warm = res["cold"], res["interp"]
    fac = res["factors"]["interp"]

    # both modes converge to the target ...
    assert cold["converged"] and warm["converged"]
    # ... and to the SAME free data (the surrogate changes only the inner guess)
    assert np.max(np.abs(cold["theta"] - warm["theta"])) < 1e-6

    # certified constraint residual met at EVERY solver call in BOTH modes
    assert cold["counters"].max_resid <= TOL_INNER
    assert warm["counters"].max_resid <= TOL_INNER

    # the outer loop is identical across modes -> same number of solver calls
    assert res["factors"]["calls_match"], f"calls differ: {res['factors']['calls']}"

    # clear reduction: inner Newton iterations and (dominant-cost) LU solves
    assert fac["inner_iters"] >= 2.0, f"inner-iter factor {fac['inner_iters']:.2f} < 2"
    assert fac["lu_solves"] >= 3.0, f"LU-solve factor {fac['lu_solves']:.2f} < 3"
    # wall-clock improves too (lenient: contention-robust)
    assert fac["wall_clock"] > 1.2, f"wall-clock factor {fac['wall_clock']:.2f} <= 1.2"


# ==========================================================================
# B2-T3 — scattered parameter SURVEY: the global interpolant beats continuation
# (continuation has no smooth march for unconnected points -> ~ as bad as cold)
# ==========================================================================
def test_scattered_survey_interp_beats_continuation(cp_mass):
    cp = cp_mass["cp"]
    rng = np.random.default_rng(0)
    thetas = [np.array([rng.uniform(1.0, 3.0), rng.uniform(3.0, 12.0)]) for _ in range(12)]
    c_cold = ctl.survey_cost(cp, thetas, mode="cold", tol_inner=TOL_INNER)
    c_cont = ctl.survey_cost(cp, thetas, mode="continuation", tol_inner=TOL_INNER)
    c_intp = ctl.survey_cost(cp, thetas, mode="interp", tol_inner=TOL_INNER)
    assert c_cold.max_resid <= TOL_INNER and c_intp.max_resid <= TOL_INNER
    # the interpolant gives a global warm start: clearly fewer inner iters than cold
    assert c_intp.inner_iters < 0.6 * c_cold.inner_iters
    # and strictly better than continuation (which barely helps for scattered points)
    assert c_intp.inner_iters < c_cont.inner_iters
    # continuation barely beats cold for unconnected points (the surrogate's edge)
    assert c_cont.inner_iters > 0.7 * c_cold.inner_iters


# ==========================================================================
# B2-T4 — the interpolant warm-start does not change the physical answer
#         (cold and warm converged free data agree to the control tolerance)
# ==========================================================================
def test_cold_warm_consistency_spin(cp_spin):
    cp, tgt, start, ts = cp_spin["cp"], cp_spin["target"], cp_spin["start"], cp_spin["theta_star"]
    rc = ctl.broyden_control(cp, start, tgt, mode="cold", tol_ctrl=TOL_CTRL, tol_inner=TOL_INNER)
    rw = ctl.broyden_control(cp, start, tgt, mode="interp", tol_ctrl=TOL_CTRL, tol_inner=TOL_INNER)
    assert rc["converged"] and rw["converged"]
    assert np.max(np.abs(rc["theta"] - rw["theta"])) < 1e-6
    assert np.max(np.abs(rw["theta"] - ts)) < 1e-4


# ==========================================================================
# B2-T5 — the unequal-mass choice for T2 is load-bearing: (J, M_ADM) is
#         χ_A↔χ_B-symmetric at equal mass (2-to-1, round-trip non-unique) and
#         injective at q=1.5.  A deterministic teeth test (no basin dependence).
# ==========================================================================
def test_spin_target_degeneracy_needs_unequal_mass(prob):
    theta_a = np.array([0.55, 0.10])
    theta_b = np.array([0.10, 0.55])           # the χ_A↔χ_B swap

    def F_at(q, theta):
        cp = ctl.ControlProblem(prob, ("chi_A", "chi_B"), ("J", "M_ADM"), M_tot=1.0,
                                fixed={"q": q, "b": 4.0})
        F, _, _ = ctl.solve_and_observe(cp, theta, None, 1e-12, 40)
        return F

    # EQUAL mass: the target is symmetric under the swap -> the map is 2-to-1
    F_eq_a, F_eq_b = F_at(1.0, theta_a), F_at(1.0, theta_b)
    assert np.max(np.abs(F_eq_a - F_eq_b)) < 1e-9, \
        f"equal-mass (J,M_ADM) not swap-symmetric: {F_eq_a} vs {F_eq_b}"

    # UNEQUAL mass q=1.5: the swap is broken (J distinguishes χ_A from χ_B) -> injective
    F_uq_a, F_uq_b = F_at(1.5, theta_a), F_at(1.5, theta_b)
    # the J component alone separates by (χ_A−χ_B)(m_A²−m_B²) = 0.45·0.20 = 0.090
    assert abs(F_uq_a[0] - F_uq_b[0]) > 0.05, "q=1.5 J did not break the swap symmetry"
    assert np.max(np.abs(F_uq_a - F_uq_b)) > 0.05, "q=1.5 target not injective under swap"
