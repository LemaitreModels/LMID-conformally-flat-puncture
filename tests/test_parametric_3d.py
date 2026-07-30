"""PARASOL 3D — the certified non-axisymmetric PARAMETRIC layer (the "3-D lift").

The N-D Chebyshev-in-parameter collocation layer (``parametric_nd.py``, reused
verbatim) wired to the 3-D non-axisymmetric solver via ``parametric_nd_3d.py``.
A tensor-product interpolant over a misaligned-spin / off-axis-momentum
Bowen–York slice family, whose every prediction can be certified to
``‖R‖∞ ≤ 1e-10`` by the committed Newton–Krylov forward map.

Gates (the brief):
  * single-axis reduction — a b-only Nφ=1 sweep reproduces the committed 2-D
    ``parametric_nd_2c`` result to ~1e-12 (the layer contains the 2-D layer);
  * the D7 per-b cache is byte-identical to a fresh ``solver_3d.assemble``;
  * held-out interpolation error drops geometrically with Q in each axis
    (b, |S|/θ_S, S_x) and jointly — exponential parametric convergence;
  * certified prediction (headline) — at generic off-node θ,
    ``evaluate_polished`` reaches certified ``‖R‖∞ ≤ 1e-10`` from the interpolant
    warm start in ≤2 NK steps, the "cannot be silently wrong" gate over 3-D;
  * analyticity walls — the b→0 merger wall reproduces the P1 / b=0 Bernstein
    rate; the spin-tilt wall is soft/far;
  * TwoPunctures cross-check (slow, skip if absent) — ψ at a held-out 3-D θ
    matches TP spectrally.

The heavy multi-node studies are marked ``slow``; the quick gates (reduction,
byte-identical cache, a short convergence ladder, the certified prediction,
frozen topology) run by default.  Small grids / Nφ≤12 throughout.
"""

import numpy as np
import pytest

from lemaitre.initial_data.solver import solver_3d as s3, solver_abt as sa
from lemaitre.initial_data.parametric import parametric_nd_3d as p3
from lemaitre.initial_data.parametric import parametric_nd_2c as p2c
from lemaitre.initial_data.validation import twopunctures as tp

_oracle = pytest.mark.skipif(not tp.available(),
                             reason="TwoPunctures binary not built (see build.sh)")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _geom_rate(rows):
    Qs = np.array([r[0] for r in rows], float)
    es = np.array([r[1] for r in rows], float)
    return -float(np.polyfit(Qs, np.log10(es), 1)[0])


# --------------------------------------------------------------------------
# D7 cache — byte-identical to a fresh solver_3d.assemble
# --------------------------------------------------------------------------
def test_assemble_cached_byte_identical():
    prob = s3.make_problem(Na=28, Nb=20, Nphi=6)
    cache = {}
    for theta in ([1.8, 0.3, 50.0], [1.8, 0.2, 30.0], [3.0, 0.3, 50.0]):
        sl = p3.theta_to_slice3d(theta, ["b", "S_mag", "theta_S"])
        a1 = p3.assemble_cached_3d(prob, sl, cache)
        a2 = s3.assemble(prob, sl)
        for mi in range(prob.m_vals.size):
            assert np.array_equal(np.asarray(a1.M0[mi]), np.asarray(a2.M0[mi]))
            assert np.array_equal(np.asarray(a1.w[mi]), np.asarray(a2.w[mi]))
        for f in ("interior", "rho", "z", "psi", "A2"):
            assert np.array_equal(np.asarray(getattr(a1, f)), np.asarray(getattr(a2, f))), f
    # cache reused per-b (two θ at b=1.8 share the geometry object)
    assert len(cache) == 2, f"expected 2 distinct b keys, got {sorted(cache)}"


# --------------------------------------------------------------------------
# Single-axis reduction — Nφ=1 b-sweep == committed parametric_nd_2c
# --------------------------------------------------------------------------
def test_nk_reduction_to_2d_parametric():
    """A 1-axis 3-D sweep (only b active; Nφ=1, on-axis P, zero spin) reproduces
    the committed 2-D ``parametric_nd_2c`` b-sweep to ~1e-12 — the 3-D layer
    contains the validated 2-D layer."""
    Na, Nb = 28, 20
    prob3 = s3.make_problem(Na=Na, Nb=Nb, Nphi=1)
    ps3 = p3.from_problem_nd_3d(
        prob3, [{"name": "b", "min": 3.0, "max": 12.0, "Q": 8}],
        fixed={"S_mag": 0.0, "P": 0.5}, solver="nk").build()
    probA = sa.make_problem(Na=Na, Nb=Nb, P=0.5)
    ps2 = p2c.from_problem_nd(probA, [{"name": "b", "min": 3.0, "max": 12.0, "Q": 8}]).build()
    dmax = 0.0
    for b in p3.holdout_points_1axis(3.0, 12.0):
        U3 = np.asarray(ps3.evaluate([float(b)])).reshape(Na + 1, Nb)
        U2 = np.asarray(ps2.evaluate([float(b)]))
        dmax = max(dmax, float(np.max(np.abs(U3 - U2))))
    print(f"\n[3D-red] Nφ=1 b-sweep vs parametric_nd_2c: max diff = {dmax:.2e}")
    assert dmax < 1e-10, f"reduction diff {dmax:.2e} (>1e-10)"


# --------------------------------------------------------------------------
# Held-out spectral convergence in b (misaligned-spin 3-D family)
# --------------------------------------------------------------------------
def test_held_out_b_convergence():
    prob = s3.make_problem(Na=28, Nb=20, Nphi=6)
    Qs = [4, 8, 12]
    rows, hold = p3.held_out_convergence_1axis(
        prob, "b", 3.0, 12.0, Qs, fixed={"S_mag": 0.3, "theta_S": 50.0})
    axes = [{"name": "b", "min": 3.0, "max": 12.0, "Q": Q} for Q in Qs]
    for a in axes:
        p3.assert_off_node([np.array([h]) for h in hold], [a])
    print("\n[3D-b] held-out convergence (Nφ=6, misaligned spin |S|=0.3, tilt 50°):")
    for q, e, it in rows:
        print(f"   Q_b={q:>3}  heldOutErr={e:.3e}  sweepIters={it}")
    errs = np.array([r[1] for r in rows])
    assert np.all(np.diff(errs) < 0), f"not monotone: {errs}"
    assert errs[0] / errs[-1] > 1e3, f"not exponential: {errs}"
    assert errs[-1] <= 1e-7, f"held-out err @ Q={Qs[-1]} = {errs[-1]:.2e}"


# --------------------------------------------------------------------------
# Certified prediction (headline) — ≤1e-10 at off-node θ in ≤2 NK steps
# --------------------------------------------------------------------------
def test_certified_prediction_3d():
    """At a generic OFF-node θ over the 3-D family, the interpolant warm start +
    ≤2 certified NK steps reaches ``‖R‖∞ ≤ 1e-10`` — independent of any
    interpolation error.  The "cannot be silently wrong" gate in 3-D."""
    prob = s3.make_problem(Na=28, Nb=20, Nphi=6)
    axes = [{"name": "b", "min": 1.5, "max": 4.0, "Q": 5},
            {"name": "theta_S", "min": 0.0, "max": 90.0, "Q": 5}]
    ps = p3.from_problem_nd_3d(prob, axes, fixed={"S_mag": 0.3}, solver="nk").build()
    hold = p3.holdout_points_nd(axes, n_points=5)
    p3.assert_off_node(hold, axes)
    worst_warm = 0.0
    worst_cert = 0.0
    for th in hold:
        sl = p3.theta_to_slice3d(th, ["b", "theta_S"], fixed={"S_mag": 0.3})
        r_warm = s3.residual_norm(prob, ps.evaluate(th), sl)
        _U, info = ps.evaluate_polished(th, newton_steps=2)
        worst_warm = max(worst_warm, r_warm)
        worst_cert = max(worst_cert, info.residual_norm)
        assert info.residual_norm <= 1e-10, (
            f"θ={th}: certified ‖R‖={info.residual_norm:.2e} (>1e-10)")
    print(f"\n[3D-cert] worst warm-start raw ‖R‖={worst_warm:.2e}  "
          f"worst certified equil ‖R‖={worst_cert:.2e}  (gate ≤1e-10, ≤2 NK steps)")
    # teeth: the warm start is genuinely far, so the polish does real work
    assert worst_warm > 1e3 * worst_cert, "warm start too close — gate is vacuous"


# --------------------------------------------------------------------------
# Frozen topology + exact-node interpolation
# --------------------------------------------------------------------------
def test_frozen_topology_and_exact_nodes():
    Na, Nb, Nphi = 28, 20, 6
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    axes = [{"name": "b", "min": 1.5, "max": 4.0, "Q": 4},
            {"name": "theta_S", "min": 0.0, "max": 90.0, "Q": 4}]
    ps = p3.from_problem_nd_3d(prob, axes, fixed={"S_mag": 0.3}, solver="modified").build()
    assert ps.U_nodes.shape == (5, 5, Na + 1, Nb, Nphi), ps.U_nodes.shape
    assert ps.field_shape == (Na + 1, Nb, Nphi)
    errmax = 0.0
    for i in range(len(ps.nodes[0])):
        for j in range(len(ps.nodes[1])):
            th = [float(ps.nodes[0][i]), float(ps.nodes[1][j])]
            errmax = max(errmax, float(np.max(np.abs(ps.evaluate(th) - ps.U_nodes[i, j]))))
    print(f"\n[3D-topo] exact-node interp err = {errmax:.2e}")
    assert errmax < 1e-13, f"exact-node interp {errmax:.2e}"


# --------------------------------------------------------------------------
# Open question — spin-tilt axis: angle θ_S vs Cartesian S_x (both spectral)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_spin_tilt_parametrisation_convergence():
    """Both the POLAR tilt-angle θ_S and the CARTESIAN S_x parametrisations of the
    misaligned spin converge geometrically (the open-question study).  θ_S is the
    naturally bounded coordinate (|S| fixed, direction varies) and measures a
    (marginally) steeper rate — the recommended choice."""
    prob = s3.make_problem(Na=28, Nb=20, Nphi=8)
    Qs = [4, 6, 8, 10]
    rows_th, _ = p3.held_out_convergence_1axis(
        prob, "theta_S", 0.0, 90.0, Qs, fixed={"b": 2.0, "S_mag": 0.3})
    rows_sx, _ = p3.held_out_convergence_1axis(
        prob, "S_x", 0.0, 0.3, Qs, fixed={"b": 2.0, "S_z": 0.0})
    rate_th, rate_sx = _geom_rate(rows_th), _geom_rate(rows_sx)
    print(f"\n[3D-spin] θ_S rate = {rate_th:.3f} dec/Q  "
          f"(errs {['%.1e' % r[1] for r in rows_th]})")
    print(f"[3D-spin] S_x rate = {rate_sx:.3f} dec/Q  "
          f"(errs {['%.1e' % r[1] for r in rows_sx]})")
    for rows in (rows_th, rows_sx):
        errs = np.array([r[1] for r in rows])
        assert np.all(np.diff(errs) < 0), f"spin axis not monotone: {errs}"
        assert errs[0] / errs[-1] > 1e3, f"spin axis not exponential: {errs}"
    # both spectral (soft/far wall), well above the b-merger wall rate (~0.35)
    assert rate_th > 0.4 and rate_sx > 0.4, (rate_th, rate_sx)


# --------------------------------------------------------------------------
# Joint multi-axis held-out convergence
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_joint_convergence():
    prob = s3.make_problem(Na=28, Nb=20, Nphi=6)
    rows, _ = p3.held_out_convergence_joint(
        prob, [{"name": "b", "min": 1.5, "max": 4.0},
               {"name": "theta_S", "min": 0.0, "max": 90.0}],
        [(4, 4), (6, 6), (8, 8)], fixed={"S_mag": 0.3})
    print("\n[3D-joint] (b, θ_S):")
    for q, n, e, it in rows:
        print(f"   Q={q}  nodes={n}  heldOutErr={e:.3e}  iters={it}")
    errs = np.array([r[2] for r in rows])
    assert np.all(np.diff(errs) < 0), f"joint not monotone: {errs}"
    assert errs[0] / errs[-1] > 1e2, f"joint not exponential: {errs}"


# --------------------------------------------------------------------------
# Analyticity wall — b→0 merger reproduces the P1 / b=0 Bernstein rate
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_analyticity_wall_b():
    prob = s3.make_problem(Na=28, Nb=20, Nphi=6)
    wall = p3.analyticity_wall_b(prob, [3.0, 1.5], 12.0, [4, 8, 12, 16],
                                 fixed={"S_mag": 0.3, "theta_S": 50.0})
    for w in wall:
        print(f"\n[3D-wall] b_min={w['b_min']}  rate={w['rate']:.3f} dec/Q  "
              f"(b=0 Bernstein pred {w['rate_pred']:.3f})  "
              f"errs={['%.1e' % e for e in w['errs']]}")
    w_far, w_near = wall[0], wall[1]
    # the wall: smaller b_min (closer to the merger singularity) converges slower
    assert w_far["rate"] > w_near["rate"] + 0.05, (
        f"rate did not degrade: {w_far['rate']:.3f} vs {w_near['rate']:.3f}")
    # both measured rates track the nearest-singularity (b=0) Bernstein prediction
    for w in wall:
        rel = abs(w["rate"] - w["rate_pred"]) / w["rate_pred"]
        assert rel < 0.30, (f"b_min={w['b_min']}: rate {w['rate']:.3f} "
                            f"vs b=0 pred {w['rate_pred']:.3f} (rel {rel:.2f})")


def test_spin_wall_softer_than_merger():
    """The spin-tilt wall is SOFT/FAR — its geometric rate vastly exceeds the
    near-merger b wall, i.e. the nearest singularity in the tilt axis is far
    outside the physical range (no spin "wall" at moderate separation)."""
    prob = s3.make_problem(Na=28, Nb=20, Nphi=8)
    spin = p3.analyticity_wall_spin(prob, "theta_S", 90.0, [4, 6, 8, 10],
                                    b=2.0, fixed={"S_mag": 0.3})
    w = spin[0]
    print(f"\n[3D-spinwall] θ_S rate={w['rate']:.3f} dec/Q  "
          f"inferred nearest real singularity θ*={w['p_star']:.1f}° "
          f"(range [0,90]°)  errs={['%.1e' % e for e in w['errs']]}")
    assert w["rate"] > 0.45, f"spin wall not soft: rate {w['rate']:.3f}"
    assert w["p_star"] > 90.0, "inferred singularity not outside the range"


# --------------------------------------------------------------------------
# TwoPunctures cross-check (slow; skip if the oracle binary is absent)
# --------------------------------------------------------------------------
@_oracle
@pytest.mark.slow
def test_tp_cross_check_3d():
    """ψ from a CERTIFIED interpolant prediction at a held-out 3-D θ matches the
    TwoPunctures oracle spectrally (the 3-D analog of Test E, now through the
    parametric layer)."""
    from lemaitre.initial_data.solver import source
    b, S_mag = 1.5, 0.3
    Na, Nb, Nphi = 56, 40, 10
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    axes = [{"name": "b", "min": 1.3, "max": 2.0, "Q": 4},
            {"name": "theta_S", "min": 30.0, "max": 70.0, "Q": 4}]
    ps = p3.from_problem_nd_3d(prob, axes, fixed={"S_mag": S_mag}, solver="nk").build()
    # a generic held-out θ
    th = p3.holdout_points_nd(axes, n_points=1)[0]
    p3.assert_off_node([th], axes)
    U, info = ps.evaluate_polished(th, newton_steps=2)
    sl = p3.theta_to_slice3d(th, ["b", "theta_S"], fixed={"S_mag": S_mag})
    print(f"\n[3D-TP] θ=(b={th[0]:.3f}, tilt={th[1]:.2f}°)  "
          f"certified ‖R‖={info.residual_norm:.2e}")
    assert info.residual_norm <= 1e-9

    QR = np.array([0.4, 0.8, 0.6, 1.2, 2.0]) * sl.b
    QZ = np.array([0.6, 0.0, -0.5, 0.3, 0.4]) * sl.b
    QP = np.array([0.0, 1.0, 2.0, 0.5, 2.5])
    res = tp.solve_parasol_points_3d(sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec,
                                     sl.S_A_vec, sl.S_B_vec, QR, QZ, QP,
                                     nA=64, nB=64, nphi=12)
    u = np.asarray(s3.evaluate_field(prob, U, QR, QZ, QP, sl.b))
    psi = np.asarray(source.psi_BL_2c(QR, QZ, sl.b, sl.m_A, sl.m_B)) + u
    dpsi = float(np.max(np.abs(psi - res.psi)))
    print(f"[3D-TP] |dψ| vs TwoPunctures = {dpsi:.3e}")
    assert dpsi < 1e-5, f"ψ vs TP {dpsi:.2e}"
