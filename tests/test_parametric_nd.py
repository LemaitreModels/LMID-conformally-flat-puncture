"""P3 acceptance — multi-dimensional parametric collocation + the analyticity walls.

The N-D tensor-product Chebyshev-in-parameter layer (``parametric_nd.py``) wired
to the validated ABT two-centre spin solver (``parametric_nd_2c.py``) over
``θ = (q, b, χ_A, χ_B)``.  Gates (PAPER_PLAN §5 P3):

  * joint held-out error ≤ 1e-8 at practical Q;
  * per-axis rates match Bernstein to ≲ 10–15% (b: a-priori merger b=0, reproduces
    P1; χ: a single inferred χ*≈1.5 beyond extremal — the soft spin wall);
  * ``evaluate_polished`` certifies ‖R‖∞ ≤ 1e-10 at arbitrary off-node 4-D θ;
  * the "handful of solves" warm-start property preserved (boustrophedon march);
  * cost scaling Q^d.

Guarded (add-only): a single active axis reduces **bit-for-bit** to the P1 1-D
sweep (U_nodes + interpolant byte-identical); the per-b cache is byte-identical to
``solver_abt.assemble``.  The held-out error measured here is interpolation-IN-
PARAMETER error (interpolant and direct solve share the frozen grid → spatial
error cancels, R7).
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from lm.initial_data.solver import solver_abt as sa
from lm.initial_data.parametric import parametric_nd as pnd
from lm.initial_data.parametric import parametric_nd_2c as p3
from lm.initial_data.parametric import parametric_2c as p2c


# ==========================================================================
# P3-U1 — boustrophedon snake order: Hamiltonian path, single-step adjacency
# ==========================================================================
def test_snake_order_adjacency():
    for shape in [(5,), (3, 4), (4, 3, 2), (2, 3, 2, 2), (15, 13, 11, 11)]:
        order = pnd.snake_order(shape)
        n = int(np.prod(shape))
        assert len(order) == n, f"{shape}: wrong length"
        assert len(set(order)) == n, f"{shape}: not a permutation"
        for a, b in zip(order, order[1:]):
            diff = [abs(x - y) for x, y in zip(a, b)]
            assert sum(1 for d in diff if d != 0) == 1 and max(diff) == 1, (
                f"{shape}: {a}->{b} not single-step adjacent")
        # first node = all-max-index corner (= all-q_min for descending CGL nodes)
        assert order[0] == tuple(s - 1 for s in shape)


def test_snake_single_axis_matches_argsort():
    """For one axis the snake = [Q, Q-1, ..., 0]: descending index = ascending
    value, i.e. the exact 1-D ParametricSolver.build march (np.argsort of the
    descending CGL nodes).  This is what makes the bit-for-bit reduction hold."""
    for Q in (4, 7, 12):
        order = pnd.snake_order((Q + 1,))
        assert order == [(i,) for i in range(Q, -1, -1)]


# ==========================================================================
# P3-U2 — bit-for-bit reduction of a single active axis to the P1 1-D sweep
# ==========================================================================
def test_b_axis_reduces_bit_for_bit():
    prob = sa.make_problem(Na=24, Nb=18, P=0.5)
    Q = 8
    ps1d = p2c.from_problem_b(prob, 0.5, 0.5, 3.0, 12.0, Q).build(tol=1e-12, max_iter=20)
    psnd = p3.from_problem_nd(prob, [{"name": "b", "min": 3.0, "max": 12.0, "Q": Q}],
                              M_tot=1.0, use_cache=False).build(tol=1e-12, max_iter=20)
    assert np.array_equal(ps1d.q_nodes, psnd.nodes[0])
    assert np.array_equal(ps1d.weights, psnd.weights[0])
    assert np.array_equal(ps1d.U_nodes, psnd.U_nodes), "U_nodes not bit-for-bit"
    # interpolant byte-identical at off-node b (the successive-tensordot formula)
    for bq in (4.137, 6.339, 9.853):
        assert np.array_equal(ps1d.evaluate(bq), psnd.evaluate([bq])), f"evaluate b={bq}"


def test_q_axis_reduces_bit_for_bit():
    prob = sa.make_problem(Na=24, Nb=18, P=0.5)
    Q = 8
    ps1d = p2c.from_problem_q(prob, b=4.0, M_tot=1.0, q_min=1.0, q_max=3.0, Q=Q).build(
        tol=1e-12, max_iter=20)
    psnd = p3.from_problem_nd(prob, [{"name": "q", "min": 1.0, "max": 3.0, "Q": Q}],
                              M_tot=1.0, fixed={"b": 4.0}, use_cache=False).build(
        tol=1e-12, max_iter=20)
    assert np.array_equal(ps1d.U_nodes, psnd.U_nodes), "q-axis U_nodes not bit-for-bit"
    for qq in (1.274, 1.742, 2.706):
        assert np.array_equal(ps1d.evaluate(qq), psnd.evaluate([qq]))


# ==========================================================================
# P3-U3 — the per-b cache is byte-identical to solver_abt.assemble
# ==========================================================================
def test_cache_assembly_byte_identical():
    prob = sa.make_problem(Na=20, Nb=16, P=0.5)
    cache = {}
    for sl in [sa.Slice(4.0, 0.6, 0.4, 0.1, 0.0), sa.Slice(4.0, 0.5, 0.5, 0.2, 0.2),
               sa.Slice(7.3, 0.5, 0.5, 0.0, 0.0)]:
        a_ref = sa.assemble(prob, sl)
        a_cac = p3.assemble_cached(prob, sl, cache)
        for fld in ("M0", "interior", "rho", "z", "psi", "A2"):
            assert np.array_equal(getattr(a_ref, fld), getattr(a_cac, fld)), \
                f"cached {fld} differs at b={sl.b}"


def test_build_cache_on_equals_off():
    prob = sa.make_problem(Na=20, Nb=16, P=0.5)
    axes = [{"name": "b", "min": 3.0, "max": 12.0, "Q": 5},
            {"name": "chi_A", "min": 0.0, "max": 0.6, "Q": 4}]
    off = p3.from_problem_nd(prob, axes, use_cache=False).build(tol=1e-12, max_iter=20)
    on = p3.from_problem_nd(prob, axes, use_cache=True).build(tol=1e-12, max_iter=20)
    assert np.array_equal(off.U_nodes, on.U_nodes), "cache on != off"


# ==========================================================================
# P3-U4 — N-D tensor-product barycentric: exact at nodes; reduces to 1-D
# ==========================================================================
def test_nd_barycentric_exact_at_nodes():
    prob = sa.make_problem(Na=20, Nb=16, P=0.5)
    ps = p3.from_problem_nd(prob, [{"name": "b", "min": 3, "max": 12, "Q": 5},
                                   {"name": "chi_A", "min": 0, "max": 0.6, "Q": 4},
                                   {"name": "chi_B", "min": 0, "max": 0.6, "Q": 3}],
                            use_cache=True).build(tol=1e-12, max_iter=20)
    assert ps.U_nodes.shape == (6, 5, 4, prob.A.size, prob.B.size)
    worst = 0.0
    for i, bn in enumerate(ps.nodes[0]):
        for j, cn in enumerate(ps.nodes[1]):
            for k, dn in enumerate(ps.nodes[2]):
                worst = max(worst, float(np.max(np.abs(
                    ps.evaluate([bn, cn, dn]) - ps.U_nodes[i, j, k]))))
    assert worst < 1e-13, f"exact-node N-D interp err {worst:.2e}"


# ==========================================================================
# P3-U5 — χ→S convention + theta_to_slice (D1/D2); S=0 -> P1 Slice
# ==========================================================================
def test_theta_to_slice_convention():
    # equal mass q=1, M=1 -> m_A=m_B=0.5; S = chi*m^2 = chi*0.25
    sl = p3.theta_to_slice([1.0, 4.0, 0.8, 0.4], ("q", "b", "chi_A", "chi_B"), M_tot=1.0)
    assert sl.b == 4.0 and abs(sl.m_A - 0.5) < 1e-15 and abs(sl.m_B - 0.5) < 1e-15
    assert abs(sl.S_A - 0.8 * 0.25) < 1e-15 and abs(sl.S_B - 0.4 * 0.25) < 1e-15
    # unequal mass q=2, M=1 -> m_A=2/3, m_B=1/3; S_X = chi_X m_X^2
    sl2 = p3.theta_to_slice([2.0, 5.0, 0.5, 0.5], ("q", "b", "chi_A", "chi_B"))
    assert abs(sl2.m_A - 2.0 / 3.0) < 1e-15 and abs(sl2.m_B - 1.0 / 3.0) < 1e-15
    assert abs(sl2.S_A - 0.5 * (2 / 3) ** 2) < 1e-15
    # chi=0 reduces to the P1 (no-spin) Slice exactly
    sl0 = p3.theta_to_slice([1.0, 4.0, 0.0, 0.0], ("q", "b", "chi_A", "chi_B"))
    assert sl0 == sa.Slice(4.0, 0.5, 0.5)


# ==========================================================================
# P3-U6 — off-node holdout guard has teeth
# ==========================================================================
def test_offnode_guard():
    axes = [{"name": "b", "min": 3, "max": 12, "Q": 8},
            {"name": "chi_A", "min": 0, "max": 0.6, "Q": 6}]
    pts = p3.holdout_points_nd(axes, n_points=6)
    p3.assert_off_node(pts, axes)                       # passes for generic points
    nodes, _ = p3.cheb_param_nodes(3, 12, 8)
    on = [np.array([float(nodes[2]), 0.281])]           # exactly on a b-node
    with pytest.raises(AssertionError):
        p3.assert_off_node(on, axes)


# ==========================================================================
# P3-U7 — JAX differentiability of the interpolant (the B3 hook) vs FD
# ==========================================================================
def test_evaluate_jax_grad_vs_fd():
    prob = sa.make_problem(Na=20, Nb=16, P=0.5)
    ps = p3.from_problem_nd(prob, [{"name": "b", "min": 3, "max": 12, "Q": 6},
                                   {"name": "chi_A", "min": 0, "max": 0.6, "Q": 5}],
                            use_cache=True).build(tol=1e-12, max_iter=20)
    # interpolant evaluate_jax matches the numpy evaluate off-node
    theta = np.array([6.339, 0.281])
    assert float(np.max(np.abs(np.asarray(ps.evaluate_jax(theta)) - ps.evaluate(theta)))) < 1e-12

    def f(th):
        return jnp.sum(ps.evaluate_jax(th))
    th0 = jnp.asarray(theta)
    g = np.asarray(jax.grad(f)(th0))
    h = 1e-6
    fd = np.array([(float(f(th0.at[k].add(h))) - float(f(th0.at[k].add(-h)))) / (2 * h)
                   for k in range(2)])
    assert np.max(np.abs(g - fd)) / np.max(np.abs(fd)) < 1e-6


# ==========================================================================
# P3-T1 — per-axis held-out convergence + the merger b-wall (reproduces P1)
# ==========================================================================
def test_per_axis_convergence_and_b_wall():
    prob = sa.make_problem(Na=36, Nb=24, P=0.5)
    Qs = [4, 8, 12, 16]
    # b axis reproduces the P1 headline numbers exactly
    rows_b, _ = p3.held_out_convergence_1axis(prob, "b", 3.0, 12.0, Qs)
    eb = np.array([r[1] for r in rows_b])
    assert abs(eb[2] - 1.538e-8) < 1e-10 and abs(eb[3] - 2.072e-10) < 1e-11, \
        f"b-axis does not reproduce P1: {eb}"
    assert np.all(np.diff(eb) < 0) and eb[0] / eb[-1] > 1e3
    rate_b = p3.geometric_rate([r[0] for r in rows_b], eb)
    assert abs(rate_b - p3.bernstein_rate_from_zero(3, 12)) / p3.bernstein_rate_from_zero(3, 12) < 0.15
    # q and both spin axes converge exponentially
    for name, fixed, lo, hi in [("q", {"b": 4.0}, 1.0, 3.0),
                                ("chi_A", {"b": 4.0, "chi_B": 0.0}, 0.0, 0.8),
                                ("chi_B", {"b": 4.0, "chi_A": 0.0}, 0.0, 0.8)]:
        rows, _ = p3.held_out_convergence_1axis(prob, name, lo, hi, Qs, fixed=fixed)
        e = np.array([r[1] for r in rows])
        assert np.all(np.diff(e) < 0), f"{name} not monotone: {e}"
        assert e[0] / e[-1] > 1e4, f"{name} not exponential: {e}"
    # the b-wall: rate degrades toward the merger, tracking the b=0 Bernstein pred
    wall = p3.analyticity_wall_b(prob, [3.0, 1.5], 12.0, Qs, fit_window=(4, 16))
    assert wall[0]["rate"] > wall[1]["rate"] + 0.05, "b-wall rate did not degrade"
    for w in wall:
        assert abs(w["rate"] - w["rate_pred"]) / w["rate_pred"] < 0.15


# ==========================================================================
# P3-T2 — the spin χ-wall: a soft, far wall (single χ* ≈ 1.5, beyond extremal)
# ==========================================================================
def test_chi_wall_soft_and_bernstein_consistent():
    prob = sa.make_problem(Na=36, Nb=24, P=0.5)
    Qs = [4, 8, 12, 16]
    wall = p3.analyticity_wall_chi(prob, [0.4, 0.6, 0.8], Qs, b=4.0, fit_window=(4, 12))
    # spin converges much faster than the merger (rate >> b's ~0.49)
    for w in wall:
        assert w["rate"] > 0.6, f"chi_max={w['chi_max']} rate {w['rate']:.3f} unexpectedly slow"
    # the inferred nearest singularity is beyond the extremal range and CONSISTENT
    chistars = np.array([w["chi_star"] for w in wall])
    assert np.all(chistars > 1.0), f"chi* not beyond extremal: {chistars}"
    chi_star = float(np.mean(chistars))
    assert (chistars.max() - chistars.min()) / chi_star < 0.15, \
        f"chi* not consistent across intervals: {chistars}"
    # a single fixed chi* reproduces every measured rate to <=15% (the soft wall)
    for w in wall:
        pred = p3.bernstein_rate(chi_star, 0.0, w["chi_max"])
        assert abs(w["rate"] - pred) / pred < 0.15, (
            f"chi_max={w['chi_max']}: rate {w['rate']:.3f} vs chi*={chi_star:.2f} "
            f"pred {pred:.3f}")
    # rate degrades as chi_max grows over the physical range (the wall direction)
    assert wall[0]["rate"] > wall[-1]["rate"], "chi-wall rate did not degrade with chi_max"


# ==========================================================================
# P3-T3 — certified evaluation at arbitrary off-node 4-D θ (≤1e-10)
# ==========================================================================
def test_certified_polish_4d():
    prob = sa.make_problem(Na=28, Nb=20, P=0.5)
    axes = [{"name": "b", "min": 3.0, "max": 12.0, "Q": 10},
            {"name": "q", "min": 1.0, "max": 3.0, "Q": 8},
            {"name": "chi_A", "min": 0.0, "max": 0.6, "Q": 6},
            {"name": "chi_B", "min": 0.0, "max": 0.6, "Q": 6}]
    ps = p3.from_problem_nd(prob, axes).build(tol=1e-12, max_iter=20)
    pts = p3.holdout_points_nd(axes, n_points=5)
    p3.assert_off_node(pts, axes)
    worst = 0.0
    for theta in pts:
        _U, info = ps.evaluate_polished(theta, newton_steps=2)
        assert info.residual_norm <= 1e-10, f"theta={theta}: ||R||={info.residual_norm:.2e}"
        assert info.iters <= 2
        worst = max(worst, info.residual_norm)
    print(f"\n[P3-T3] worst certified ||R||_inf over off-node 4-D theta = {worst:.2e}")


# ==========================================================================
# P3-T4 — warm-start "handful of solves": boustrophedon march keeps iters low
# ==========================================================================
def test_warm_start_handful():
    prob = sa.make_problem(Na=28, Nb=20, P=0.5)
    axes = [{"name": "b", "min": 3.0, "max": 12.0, "Q": 6},
            {"name": "chi_A", "min": 0.0, "max": 0.6, "Q": 5}]
    ps = p3.from_problem_nd(prob, axes).build(tol=1e-12, max_iter=20)
    n_nodes = ps.n_nodes
    mean_iters = float(np.sum(ps.iters)) / n_nodes
    # warm-started Newton stays a handful per node (vs the cold floor ~6-8)
    assert mean_iters < 6.0, f"mean iters/node {mean_iters:.1f} (warm start not effective)"
    # all nodes converged to the residual floor
    assert float(np.max(ps.residuals)) < 1e-8


# ==========================================================================
# P3-T5 — JOINT 4-D held-out convergence reaches ≤ 1e-8 (slow)
# ==========================================================================
@pytest.mark.slow
def test_joint_held_out_below_1e8():
    prob = sa.make_problem(Na=28, Nb=20, P=0.5)
    axes_tmpl = [{"name": "b", "min": 3.0, "max": 12.0},
                 {"name": "q", "min": 1.0, "max": 3.0},
                 {"name": "chi_A", "min": 0.0, "max": 0.6},
                 {"name": "chi_B", "min": 0.0, "max": 0.6}]
    # coarse isotropic decay (shows the joint exponential trend) ...
    rows, theta = p3.held_out_convergence_joint(prob, axes_tmpl, [(4, 4, 4, 4), (6, 6, 6, 6)])
    assert rows[1][2] < rows[0][2], f"joint not decreasing: {[r[2] for r in rows]}"
    # ... and an anisotropic grid (b/q richer, spin sparser) certifies <=1e-8
    rows_an, _ = p3.held_out_convergence_joint(
        prob, axes_tmpl, [(14, 13, 9, 9)], theta_holdout=theta)
    err = rows_an[0][2]
    print(f"\n[P3-T5] joint anisotropic ({rows_an[0][1]} nodes) held-out err = {err:.2e}")
    assert err <= 1e-8, f"joint held-out err {err:.2e} > 1e-8"


# ==========================================================================
# P3-T6 — cost scaling Q^d + the Smolyak/sparse crossover (D9)
# ==========================================================================
def test_cost_scaling_and_smolyak():
    # representative per-axis models (rate, log10C) — orders match the measured ones
    models = {"b": (0.49, -1.84), "q": (0.59, -1.36),
              "chi_A": (0.77, -1.35), "chi_B": (0.77, -1.35)}
    table, Qa, Qi = p3.cost_table(models, eps=1e-9)
    # anisotropic node count is strictly smaller than the isotropic Q^d
    for row in table:
        assert row["n_aniso"] <= row["n_iso"]
    # slow merger axis needs the most nodes, fast spin axes the fewest
    assert Qa["b"] > Qa["chi_A"]
    # the Q^d explosion: 4-D tensor is far larger than 1-D
    assert table[-1]["n_iso"] > 50 * table[0]["n_iso"]
    # Smolyak grows much slower than the full tensor in high d (the crossover)
    tens = (Qi + 1) ** 5
    smol = p3.smolyak_points(5, 5)
    assert smol < tens, f"Smolyak {smol} not < tensor {tens} at d=5"
