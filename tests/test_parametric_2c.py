"""M3-A acceptance — the headline two-centre PARAMETRIC milestone.

The §5 Chebyshev-in-parameter collocation layer (``parametric.py``) wired to the
validated ABT / prolate-spheroidal two-centre spatial solver (``solver_abt.py``)
via ``parametric_2c.py``.  Gates (plan §12.7 M3-A; realistic two-centre tols):

  * held-out parametric convergence in the half-separation ``b`` is EXPONENTIAL,
    reaching <= 1e-8 by Q_b ~ 16 (q=1, P=0.5 fixed, frozen grid);
  * the **analyticity wall** — the geometric rate is measured at two ``b_min`` and
    DEGRADES as the merger ``b -> 0`` is approached, tracking the b=0 Bernstein
    (nearest-singularity) prediction;
  * ``evaluate_polished`` certifies ||R||_inf <= 1e-10 at off-node ``b`` in <=2
    Newton steps;
  * the secondary mass-ratio (q) sweep converges exponentially;
  * frozen topology: the nodal field keeps shape (Na+1, Nb) at every node and the
    barycentric interpolant reproduces node values exactly.

The held-out error measured here is interpolation-IN-PARAMETER error: the
interpolant and the direct solve share the *same* frozen grid, so the spatial
discretisation error cancels in their difference and the interp error can fall
well below the single-solve spatial floor (~1e-9 at b~4) toward the ~1e-8 target.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from lemaitre.initial_data.solver import solver_abt as sa
from lemaitre.initial_data.parametric import parametric_2c as p2c
from lemaitre.initial_data.parametric import parametric
from lemaitre.initial_data.solver import diagnostics as diag


def _assert_off_node(hold, p_min, p_max, Qs, gap_min=1e-4):
    """Held-out points must be genuinely OFF every sweep's CGL nodes; otherwise
    the interpolation error collapses to ~0 and the exponential-decay check is
    vacuous (an on-node sample would look like *better* convergence).  Adversarial
    -review hardening — guards the convergence tests directly, not just by the
    node-avoiding holdout fractions.  ``gap_min=1e-4`` is ~1e9x the exact-node
    tolerance (1e-13), so it passes the generic production fractions (min gap
    ~8e-4) while catching deliberate on-node sampling (gaps ~1e-13)."""
    for Q in Qs:
        nodes, _ = parametric.cheb_param_nodes(p_min, p_max, Q)
        gap = min(float(np.min(np.abs(b - nodes))) for b in hold)
        assert gap > gap_min, f"held-out point within {gap:.1e} of a Q={Q} CGL node"


# --------------------------------------------------------------------------
# M3-A.1 — held-out parametric convergence in b (the money plot)
# --------------------------------------------------------------------------
def test_held_out_b_convergence():
    prob = sa.make_problem(Na=36, Nb=24, P=0.5)
    Qs = [4, 8, 12, 16]
    rows, hold = p2c.held_out_convergence_b(prob, 0.5, 0.5, 3.0, 12.0, Qs)
    _assert_off_node(hold, 3.0, 12.0, Qs)        # the measured decay is genuine interp error
    diag.convergence_table(rows, ["Q_b", "heldOutErr", "sweepIters"],
                           title="\n[M3-A.1] held-out parametric convergence in b "
                                 "(q=1, P=0.5, b in [3,12], Na=36 Nb=24)")
    errs = np.array([r[1] for r in rows])
    assert np.all(np.diff(errs) < 0), f"not monotone: {errs}"
    assert errs[0] / errs[-1] > 1e3, f"not exponential: {errs}"
    assert errs[-1] <= 1e-8, f"held-out err @ Q={Qs[-1]} = {errs[-1]:.2e} (target <=1e-8)"


# --------------------------------------------------------------------------
# M3-A.2 — the analyticity wall: rate degrades as b_min -> 0
# --------------------------------------------------------------------------
def test_analyticity_wall():
    prob = sa.make_problem(Na=36, Nb=24, P=0.5)
    Qs = [4, 8, 12, 16]
    wall = p2c.analyticity_wall(prob, 0.5, 0.5, [3.0, 1.5], 12.0, Qs)
    for w in wall:
        diag.convergence_table(list(zip(w["Qs"], w["errs"])), ["Q_b", "heldOutErr"],
                               title=f"\n[M3-A.2] b_min={w['b_min']}  "
                                     f"rate={w['rate']:.3f} dec/Q  "
                                     f"(b=0 Bernstein pred {w['rate_pred']:.3f})")
    w_far, w_near = wall[0], wall[1]      # b_min = 3.0, 1.5
    # the wall: smaller b_min (closer to the merger singularity) converges slower
    assert w_far["rate"] > w_near["rate"] + 0.05, (
        f"rate did not degrade: b_min=3 -> {w_far['rate']:.3f}, "
        f"b_min=1.5 -> {w_near['rate']:.3f}")
    # both measured rates track the nearest-singularity (b=0) Bernstein prediction
    for w in wall:
        rel = abs(w["rate"] - w["rate_pred"]) / w["rate_pred"]
        assert rel < 0.30, (f"b_min={w['b_min']}: measured rate {w['rate']:.3f} "
                            f"vs b=0 prediction {w['rate_pred']:.3f} (rel {rel:.2f})")


# --------------------------------------------------------------------------
# M3-A.3 — certified evaluation: ||R|| <= 1e-10 at off-node b in <=2 steps
# --------------------------------------------------------------------------
def test_evaluate_polished():
    prob = sa.make_problem(Na=36, Nb=24, P=0.5)
    ps = p2c.from_problem_b(prob, 0.5, 0.5, 3.0, 12.0, 16).build()
    worst = 0.0
    for b in p2c.holdout_points(3.0, 12.0):
        _U, info = ps.evaluate_polished(float(b), newton_steps=2)
        assert info.residual_norm <= 1e-10, (
            f"b={b:.3f}: certified ||R||={info.residual_norm:.2e} (>1e-10)")
        assert info.iters <= 2, f"b={b:.3f} took {info.iters} steps (>2)"
        worst = max(worst, info.residual_norm)
    print(f"\n[M3-A.3] worst certified ||R||_inf over off-node b = {worst:.2e} "
          f"(gate <=1e-10, in <=2 Newton steps)")


# --------------------------------------------------------------------------
# M3-A.4 — secondary q-sweep convergence (the 2-D capability)
# --------------------------------------------------------------------------
def test_q_sweep_convergence():
    prob = sa.make_problem(Na=36, Nb=24, P=0.5)
    Qs = [4, 8, 12]
    rows, hold = p2c.held_out_convergence_q(prob, b=4.0, M_tot=1.0,
                                            q_min=1.0, q_max=3.0, Qs=Qs)
    _assert_off_node(hold, 1.0, 3.0, Qs)         # the measured decay is genuine interp error
    diag.convergence_table(rows, ["Q_q", "heldOutErr", "sweepIters"],
                           title="\n[M3-A.4] held-out parametric convergence in q "
                                 "(b=4, M=1, q in [1,3])")
    errs = np.array([r[1] for r in rows])
    assert np.all(np.diff(errs) < 0), f"q-sweep not monotone: {errs}"
    assert errs[0] / errs[-1] > 1e3, f"q-sweep not exponential: {errs}"
    assert errs[-1] <= 1e-7, f"q held-out err @ Q={Qs[-1]} = {errs[-1]:.2e}"


# --------------------------------------------------------------------------
# M3-A.5 — frozen topology + exact-node interpolation
# --------------------------------------------------------------------------
def test_frozen_topology_and_exact_nodes():
    Na, Nb = 32, 22
    prob = sa.make_problem(Na=Na, Nb=Nb, P=0.5)
    ps = p2c.from_problem_b(prob, 0.5, 0.5, 3.0, 12.0, 6).build()
    # every node field has the same (Na+1, Nb) shape (np.stack succeeded)
    assert ps.U_nodes.shape == (ps.Q + 1, Na + 1, Nb), ps.U_nodes.shape
    assert ps.evaluate(5.0).shape == (Na + 1, Nb)
    # barycentric interpolant reproduces node values exactly (exact-node guard)
    for i in range(ps.Q + 1):
        bi = float(ps.q_nodes[i])
        err = float(np.max(np.abs(ps.evaluate(bi) - ps.U_nodes[i])))
        assert err < 1e-13, f"exact-node interp at b={bi:.3f}: {err:.2e}"


# --------------------------------------------------------------------------
# M3-A.6 — sanity: the interp goes BELOW the single-grid spatial floor
# (proves the held-out error is interpolation-in-parameter, not spatial)
# --------------------------------------------------------------------------
def test_interp_below_spatial_floor():
    """A coarse fixed grid has a spatial discretisation error ~1e-6-1e-5 vs a
    high-res reference, yet the held-out interpolation error on that SAME coarse
    grid drops far below it (~1e-9) — confirming the spatial error cancels in the
    interpolant-vs-direct comparison."""
    Na, Nb, P, b0 = 24, 18, 0.5, 6.339      # a generic off-node b in [3,12]
    coarse = sa.make_problem(Na=Na, Nb=Nb, P=P)
    fine = sa.make_problem(Na=48, Nb=32, P=P)
    sl = sa.Slice(b0, 0.5, 0.5)
    # spatial discretisation error of the coarse grid (vs the fine reference)
    qp = np.array([[2.0, 6.5], [3.5, 0.0], [9.0, 1.0]])
    Uc, _ = sa.newton_solve(coarse, sl, tol=1e-12, max_iter=25)
    Uf, _ = sa.newton_solve(fine, sl, tol=1e-12, max_iter=25)
    spatial_err = float(np.max(np.abs(
        sa.evaluate_field_phys(coarse, Uc, qp[:, 0], qp[:, 1], b0)
        - sa.evaluate_field_phys(fine, Uf, qp[:, 0], qp[:, 1], b0))))
    # held-out interpolation error on the same coarse grid at Q=16
    ps = p2c.from_problem_b(coarse, 0.5, 0.5, 3.0, 12.0, 16).build()
    interp_err = float(np.max(np.abs(ps.evaluate(b0) - Uc)))
    print(f"\n[M3-A.6] coarse-grid spatial err = {spatial_err:.2e}, "
          f"held-out interp err = {interp_err:.2e}")
    assert interp_err < spatial_err, (
        f"interp err {interp_err:.2e} not below spatial floor {spatial_err:.2e}")
    assert interp_err < 1e-7
