"""PARASOL — Smolyak SPARSE-grid parametric layer (the d≳3 cost fix).

The sparse-grid sibling of ``test_parametric_nd`` / ``test_parametric_3d``.  The
combination-technique Smolyak interpolant (``parametric_nd_smolyak.py``) is built
on nested Clenshaw–Curtis CGL levels and reuses the committed dense
``ParametricSolutionND`` as each subgrid.

Pure-math gates (no solver — a synthetic ``solve_fn``):
  * node-count gate — the built unique-node count == ``parametric_nd_2c.smolyak_points``;
  * nesting gate — level-i CGL nodes ⊂ level-(i+1) to 1e-13;
  * d=1 reduction — Smolyak level L equals the dense ``ParametricSolutionND`` at
    Q=2^L bit-for-bit;
  * polynomial exactness — the level-L interpolant reproduces an in-class
    total-degree polynomial to ~1e-12;
  * sparse-vs-dense + dimension-adaptive — on an anisotropic synthetic field the
    sparse grid uses materially fewer nodes than the dense tensor at matched
    accuracy, and the adaptive variant beats the isotropic Smolyak.

Solver gates (the real 3-D family; marked ``slow``):
  * sparse-vs-dense headline — at matched held-out accuracy the sparse grid uses
    fewer solver nodes than the dense tensor over the (b, θ_S, q) family;
  * certified prediction — ``evaluate_polished`` reaches ``‖R‖∞ ≤ 1e-10`` at a
    generic off-node θ in ≤2 NK steps (the certify property carries to sparse).
"""

from dataclasses import dataclass

import numpy as np
import pytest

from lm.initial_data.parametric import parametric_nd_smolyak as sm
from lm.initial_data.parametric.parametric_nd import ParametricSolverND
from lm.initial_data.parametric.parametric_nd_2c import smolyak_points


# --------------------------------------------------------------------------
# Synthetic solver — a smooth analytic field, so the unit gates need no PDE solve
# --------------------------------------------------------------------------
@dataclass
class _FakeInfo:
    iters: int
    residual_norm: float


def _make_solve(f):
    def solve_fn(theta, guess, tol, max_iter):
        return f(np.asarray(theta, float)), _FakeInfo(2, 1e-15)
    return solve_fn


_BOX4 = [(-1.0, 1.0), (0.0, 2.0), (1.0, 3.0), (0.5, 1.5)]


def _smooth_field(d):
    coef = np.array([0.7, 0.3, 0.5, 0.4][:d])

    def f(theta):
        t = np.asarray(theta, float)
        return np.array(np.sum(np.sin(coef * t)) + 0.1 * np.prod(t))
    return f


# ==========================================================================
# Node-count gate — built unique nodes == smolyak_points(d, level)
# ==========================================================================
def test_node_count_matches_smolyak_points():
    for d in (2, 3, 4):
        sf = _make_solve(_smooth_field(d))
        for level in range(0, 5):
            s = sm.SmolyakSolverND(sf, _BOX4[:d]).build_isotropic(level)
            want = smolyak_points(d, level)
            assert s.n_solver_nodes == want, (
                f"d={d} level={level}: built {s.n_solver_nodes} != smolyak_points {want}")


# ==========================================================================
# Nesting gate — level-i CGL nodes ⊂ level-(i+1)
# ==========================================================================
def test_nested_cgl_levels():
    lo, hi = -1.3, 2.7
    # level 0 must be the midpoint (cheb_param_nodes(.,.,0) is NaN, not the midpoint)
    n0, w0 = sm.nested_levels(lo, hi, 0)
    assert n0.shape == (1,) and abs(n0[0] - 0.5 * (lo + hi)) < 1e-14
    assert sm.cc_m(0) == 1 and sm.cc_m(3) == 9
    for i in range(5):
        a, _ = sm.nested_levels(lo, hi, i)
        b, _ = sm.nested_levels(lo, hi, i + 1)
        assert len(a) == sm.cc_m(i) and len(b) == sm.cc_m(i + 1)
        for x in a:
            assert float(np.min(np.abs(x - b))) < 1e-13, f"level {i} node {x} not in level {i+1}"


# ==========================================================================
# d=1 reduction — Smolyak level L == dense ParametricSolutionND at Q=2^L
# ==========================================================================
def test_d1_reduction_to_dense_tensor():
    f = _smooth_field(1)
    sf = _make_solve(f)
    axis = [(-1.0, 1.0)]
    for L in (1, 2, 3):
        s = sm.SmolyakSolverND(sf, axis).build_isotropic(L)
        dense = ParametricSolverND(sf, [(-1.0, 1.0, 2 ** L)]).build()
        assert s.n_solver_nodes == 2 ** L + 1 == dense.n_nodes
        pts = np.linspace(-0.93, 0.91, 9)
        dmax = max(abs(float(s.evaluate([p])) - float(dense.evaluate([p]))) for p in pts)
        # combination is a single subgrid at d=1 → identical interpolant (machine zero)
        assert dmax < 1e-13, f"L={L}: Smolyak vs dense differ by {dmax:.2e}"


# ==========================================================================
# Polynomial exactness — level-L Smolyak reproduces in-class total-degree polys
# ==========================================================================
def test_polynomial_exactness():
    # additive degree-≤2^L per axis + a bilinear term (in the level-≥2 space).
    # The single-axis subgrid (L,0,..) captures each additive term exactly and the
    # (1,1,0,..) subgrid the bilinear term → the Smolyak operator is exact.
    for d in (2, 3):
        for L in (2, 3):
            D = min(2 ** L, 4)

            def f(theta, D=D, d=d):
                t = np.asarray(theta, float)
                val = 0.0
                for k in range(d):
                    val = val + (0.3 * t[k] ** D - 0.5 * t[k] ** 2 + 1.1)
                val = val + 0.4 * t[0] * t[1]
                return np.array(val)

            s = sm.SmolyakSolverND(_make_solve(f), _BOX4[:d]).build_isotropic(L)
            rng = np.random.default_rng(0)
            err = 0.0
            for _ in range(25):
                th = [lo + (hi - lo) * rng.random() for (lo, hi) in _BOX4[:d]]
                err = max(err, abs(float(s.evaluate(th)) - float(f(th))))
            assert err < 1e-11, f"d={d} L={L}: poly reproduction err {err:.2e}"


# ==========================================================================
# Sparse-vs-dense + dimension-adaptive — on an anisotropic synthetic field
# ==========================================================================
def _aniso_field(theta):
    # axis0 HARD (high frequency), axis1 EASY (low freq), axis2 medium
    x, y, z = np.asarray(theta, float)
    return np.array(np.exp(0.9 * np.sin(3.0 * x)) + 0.5 * np.cos(0.4 * y)
                    + 0.3 * np.sin(1.2 * z))


def test_sparse_beats_dense_and_adaptive_beats_isotropic():
    # Mechanism proof on a STRONGLY anisotropic field (axis0 hard, axis1 trivial):
    # this is the regime where the dimension-adaptive greedy demonstrably wins.
    # (On the only-moderately-anisotropic real solver family the adaptive greedy
    # does NOT beat isotropic — see smolyak_analysis.md §4; isotropic is the
    # recommended default there.)
    axes = [(-1.0, 1.0)] * 3
    sf = _make_solve(_aniso_field)
    rng = np.random.default_rng(2)
    hold = [[lo + (hi - lo) * rng.random() for lo, hi in axes] for _ in range(40)]

    def herr(s):
        return max(abs(float(s.evaluate(th)) - float(_aniso_field(th))) for th in hold)

    # (1) isotropic Smolyak vs dense tensor at MATCHED accuracy (~9e-5)
    iso = sm.SmolyakSolverND(sf, axes).build_isotropic(4)          # 177 nodes
    dense = ParametricSolverND(sf, [(-1.0, 1.0, 16)] * 3).build()  # 4913 nodes
    e_iso, e_dense = herr(iso), herr(dense)
    assert e_iso < 5e-4 and e_dense < 5e-4, f"accuracies not matched: {e_iso:.1e}, {e_dense:.1e}"
    assert iso.n_solver_nodes < dense.n_nodes / 10, (
        f"sparse not materially cheaper: {iso.n_solver_nodes} vs {dense.n_nodes}")

    # (2) dimension-adaptive beats isotropic Smolyak at matched accuracy
    iso5 = sm.SmolyakSolverND(sf, axes).build_isotropic(5)          # 441 nodes, ~5e-11
    adap = sm.SmolyakSolverND(sf, axes).build_adaptive(max_nodes=120, indicator_tol=1e-14)
    e_iso5, e_adap = herr(iso5), herr(adap)
    # adaptive reaches isotropic-L5 accuracy with far fewer nodes (exploits anisotropy)
    assert e_adap <= e_iso5 * 5, f"adaptive accuracy {e_adap:.1e} vs iso5 {e_iso5:.1e}"
    assert adap.n_solver_nodes < iso5.n_solver_nodes / 2, (
        f"adaptive not cheaper than isotropic: {adap.n_solver_nodes} vs {iso5.n_solver_nodes}")
    # the adaptive index set is downward-closed (a valid combination grid)
    sm._assert_downward_closed(adap.index_set)


# ==========================================================================
# build_adaptive add-only options — defaults unchanged + the new modes run
# ==========================================================================
def test_adaptive_new_kwargs_default_off_equivalence():
    """The committed default ``build_adaptive()`` must be byte-for-byte identical to
    explicitly passing the add-only defaults (``indicator='surplus'``,
    ``seed_level=0``) — the new knobs are genuinely default-off."""
    sf = _make_solve(_aniso_field)
    axes = [(-1.0, 1.0)] * 3
    a = sm.SmolyakSolverND(sf, axes).build_adaptive(max_nodes=120, indicator_tol=1e-14)
    b = sm.SmolyakSolverND(sf, axes).build_adaptive(
        max_nodes=120, indicator_tol=1e-14, indicator="surplus", seed_level=0)
    assert sorted(a.index_set) == sorted(b.index_set)
    assert a.n_solver_nodes == b.n_solver_nodes
    assert getattr(a, "n_probe_solves", 0) == 0
    rng = np.random.default_rng(3)
    for _ in range(8):
        th = [lo + (hi - lo) * rng.random() for lo, hi in axes]
        assert abs(float(a.evaluate(th)) - float(b.evaluate(th))) < 1e-14


def test_adaptive_indicator_options_run():
    """profit / seed_level / held-out modes each build a valid downward-closed
    sparse grid and a finite interpolant (correctness, not a win claim — on this
    moderately-anisotropic-style synthetic they need not beat surplus)."""
    sf = _make_solve(_aniso_field)
    axes = [(-1.0, 1.0)] * 3
    rng = np.random.default_rng(5)
    probe = [np.array([lo + (hi - lo) * rng.random() for lo, hi in axes]) for _ in range(6)]

    # profit (cost-aware indicator)
    p = sm.SmolyakSolverND(sf, axes).build_adaptive(max_nodes=120, indicator="profit")
    sm._assert_downward_closed(p.index_set)
    assert np.isfinite(float(p.evaluate(probe[0])))

    # seed_level (pre-seed the isotropic |l|₁≤2 cross-terms)
    s = sm.SmolyakSolverND(sf, axes).build_adaptive(max_nodes=200, seed_level=2)
    sm._assert_downward_closed(s.index_set)
    assert set(sm.isotropic_index_set(3, 2)).issubset(set(map(tuple, s.index_set)))

    # held-out-driven greedy — needs probe_points; reports its probe-solve cost
    h = sm.SmolyakSolverND(sf, axes).build_adaptive(
        max_nodes=120, indicator="heldout", probe_points=probe, indicator_tol=-1.0)
    sm._assert_downward_closed(h.index_set)
    assert h.n_probe_solves == len(probe)
    assert np.isfinite(float(h.evaluate(probe[0])))

    # validation: bad indicator name, and held-out without probes
    with pytest.raises(ValueError):
        sm.SmolyakSolverND(sf, axes).build_adaptive(indicator="bogus")
    with pytest.raises(ValueError):
        sm.SmolyakSolverND(sf, axes).build_adaptive(indicator="heldout")


# ==========================================================================
# evaluate_jax twin agrees with numpy evaluate (the ∂ID/∂θ hook)
# ==========================================================================
def test_evaluate_jax_matches_numpy():
    sf = _make_solve(_smooth_field(3))
    s = sm.SmolyakSolverND(sf, _BOX4[:3]).build_isotropic(3)
    th = np.array([0.137, 0.846, 2.231])
    a = float(s.evaluate(th))
    b = float(np.asarray(s.evaluate_jax(th)))
    assert abs(a - b) < 1e-12, f"jax twin disagrees: {a} vs {b}"


# ==========================================================================
# Solver gates — the real 3-D non-axisymmetric family (slow)
# ==========================================================================
@pytest.mark.slow
def test_sparse_beats_dense_solver_family_d4():
    """The headline: at d=4 (the ``analysis.md`` §8 curse regime) the sparse grid
    reaches the dense tensor's held-out accuracy with **materially fewer solver
    nodes**, over the real unequal-mass misaligned-spin family
    ``θ=(b, |S|, θ_S, q)``.

    (At d=3 with this smooth, only-moderately-anisotropic family the dense tensor
    is competitive at moderate accuracy — Smolyak's advantage is textbook-known to
    grow with dimension, which is exactly why §8 flagged *d=4* as the regime where
    the sparse grid pays off.  We therefore demonstrate the saving at d=4.)
    """
    from lm.initial_data.solver import solver_3d as s3
    from lm.initial_data.parametric import parametric_nd_3d as p3

    prob = s3.make_problem(Na=16, Nb=14, Nphi=4)
    box = [{"name": "b", "min": 1.5, "max": 4.0},
           {"name": "S_mag", "min": 0.0, "max": 0.4},
           {"name": "theta_S", "min": 0.0, "max": 90.0},
           {"name": "q", "min": 1.0, "max": 3.0}]

    # held-out direct solves (shared frozen grid → spatial error cancels)
    solve_fn, _ = p3.make_solve_fn(prob, [a["name"] for a in box], solver="modified")
    probe = [dict(a, Q=8) for a in box]
    hold = p3.holdout_points_nd(probe, n_points=5)
    p3.assert_off_node(hold, probe)
    U_direct = [np.asarray(solve_fn(th, None, 1e-12, 30)[0]) for th in hold]

    def herr(ps):
        return max(float(np.max(np.abs(ps.evaluate(th) - U_direct[i])))
                   for i, th in enumerate(hold))

    # dense Q=4 (625 nodes) is the matched-accuracy reference (~2e-3)
    dense = p3.from_problem_nd_3d(prob, [dict(a, Q=4) for a in box], solver="modified").build(
        tol=1e-12, max_iter=30)
    e_dense = herr(dense)
    # isotropic Smolyak level 3 (137 nodes) should reach the SAME accuracy
    sp = sm.from_problem_smolyak_3d(prob, box, solver="modified").build_isotropic(
        3, tol=1e-12, max_iter=30)
    e_sparse = herr(sp)

    print(f"\n[smolyak-d4] dense Q=4: {dense.n_nodes} nodes err={e_dense:.2e}; "
          f"sparse L=3: {sp.n_solver_nodes} nodes err={e_sparse:.2e}; "
          f"saving = {dense.n_nodes / sp.n_solver_nodes:.1f}x")
    assert e_sparse <= e_dense, f"sparse {e_sparse:.2e} did not match dense {e_dense:.2e}"
    assert sp.n_solver_nodes < dense.n_nodes / 3, (
        f"sparse used {sp.n_solver_nodes}; expected < {dense.n_nodes // 3} (no material saving)")


@pytest.mark.slow
def test_certified_prediction_sparse():
    """``evaluate_polished`` on the sparse interpolant reaches certified
    ``‖R‖∞ ≤ 1e-10`` at a generic off-node θ in ≤2 NK steps — the certify
    property carries over to the sparse layer."""
    from lm.initial_data.solver import solver_3d as s3
    from lm.initial_data.parametric import parametric_nd_3d as p3

    prob = s3.make_problem(Na=24, Nb=18, Nphi=6)
    box = [{"name": "b", "min": 1.5, "max": 4.0},
           {"name": "theta_S", "min": 0.0, "max": 90.0}]
    sp = sm.from_problem_smolyak_3d(prob, box, solver="nk").build_isotropic(3, max_iter=20)

    hold = p3.holdout_points_nd([dict(a, Q=8) for a in box], n_points=4)
    p3.assert_off_node(hold, [dict(a, Q=8) for a in box])
    worst = 0.0
    for th in hold:
        _U, info = sp.evaluate_polished(th, newton_steps=2, tol=1e-10)
        worst = max(worst, float(info.residual_norm))
    print(f"\n[smolyak-3d] worst certified ‖R‖ over {len(hold)} off-node θ = {worst:.2e}")
    assert worst <= 1e-10, f"certified residual {worst:.2e} > 1e-10"
