"""Fast wiring gates for the genuine-8-D (both full spin vectors) extension of
``parametric_nd_3d.theta_to_slice3d`` — no solver, pure mapping.

Guards two things: (a) the existing single-spin families are byte-for-byte
unchanged (the full-spin branch only triggers when a generic-spin name is
active/fixed), and (b) the full-spin branch builds the correct S_A/S_B vectors
and plugs into the Smolyak builder.
"""
import numpy as np

from lemaitre.initial_data.parametric import parametric_nd_3d as p3
from lemaitre.initial_data.parametric import parametric_nd_smolyak as sm


def test_planar_single_spin_family_unchanged():
    # polar family (b, S_mag, theta_S, q): S_Ay and both S_B components stay 0
    sl = p3.theta_to_slice3d([2.3, 0.3, 40.0, 1.7], ["b", "S_mag", "theta_S", "q"])
    th = np.deg2rad(40.0)
    assert sl.S_A_vec == (0.3 * np.sin(th), 0.0, 0.3 * np.cos(th))
    assert sl.S_B_vec == (0.0, 0.0, 0.0)
    assert sl.P_A_vec == (0.0, 0.0, -0.5) and sl.P_B_vec == (0.0, 0.0, 0.5)
    # Cartesian single-spin family (S_x, S_z) also unaffected
    sl2 = p3.theta_to_slice3d([0.2, 0.1], ["S_x", "S_z"], fixed={"b": 2.0})
    assert sl2.S_A_vec == (0.2, 0.0, 0.1) and sl2.S_B_vec == (0.0, 0.0, 0.0)


def test_full_spin_branch_builds_both_vectors():
    names = ["b", "q", "S_Ax", "S_Ay", "S_Az", "S_Bx", "S_By", "S_Bz"]
    sl = p3.theta_to_slice3d([2.0, 1.5, 0.3, 0.25, 0.2, -0.2, 0.3, -0.15], names)
    assert np.allclose(sl.S_A_vec, (0.3, 0.25, 0.2))
    assert np.allclose(sl.S_B_vec, (-0.2, 0.3, -0.15))
    assert abs(sl.m_A - 1.5 / 2.5) < 1e-14 and abs(sl.m_B - 1.0 / 2.5) < 1e-14
    # a fixed generic-spin component also triggers the full-spin branch
    sl2 = p3.theta_to_slice3d([2.0], ["b"], fixed={"S_By": 0.1})
    assert sl2.S_B_vec == (0.0, 0.1, 0.0)


def test_full_spin_box_plugs_into_smolyak_builder():
    # a synthetic solve_fn: the mapping/box must build a valid Smolyak grid at d=8
    box = [{"name": n, "min": lo, "max": hi} for n, lo, hi in [
        ("b", 1.5, 4.0), ("q", 1.0, 3.0),
        ("S_Ax", -0.4, 0.4), ("S_Ay", -0.4, 0.4), ("S_Az", -0.4, 0.4),
        ("S_Bx", -0.4, 0.4), ("S_By", -0.4, 0.4), ("S_Bz", -0.4, 0.4)]]
    axes = [(a["min"], a["max"]) for a in box]

    class _Info:
        iters, residual_norm = 2, 1e-15

    def solve_fn(theta, guess, tol, max_iter):
        return np.array(float(np.sum(np.sin(theta)))), _Info()

    s = sm.SmolyakSolverND(solve_fn, axes).build_isotropic(2)
    from lemaitre.initial_data.parametric.parametric_nd_2c import smolyak_points
    assert s.n_solver_nodes == smolyak_points(8, 2) == 145
    assert np.isfinite(float(s.evaluate([2.0, 1.5, 0.1, -0.1, 0.2, 0.0, 0.3, -0.2])))
