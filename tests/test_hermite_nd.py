"""H2 acceptance — N-D gradient-enhanced (hard-axis-only) Hermite interpolant.

The N-D milestone of ``GRADIENT_ENHANCED_PLAN.md`` §4 (H2).  Two tiers:

  * **fast, pure-interpolant** gates (no solver) — the reduce-to-committed and
    node-exactness / gradient-only properties, on analytic test fields;
  * **slow, solver-backed** gates on ``solver_abt`` at the 44×32 grid — the
    productized Phase-0 Q1 field-rate win and the unchanged ``evaluate_polished``
    certification guarantee.

Gates (§4 H2):
  (a) reduces **bit-for-bit** to :class:`hermite.HermiteSolution1D` for ``d==1``;
  (b) reduces **bit-for-bit** to :class:`parametric_nd.ParametricSolutionND` on the
      value-only axes when zero axes are enhanced;
  (c) held-out field-error **rate beats value-only** on the enhanced axis (Phase-0
      Q1, held-out-vs-same-grid-solve protocol on ``solver_abt`` at 44×32);
  (d) ``evaluate_polished`` certifies random off-node points to ``‖R‖∞ ≤ 1e-10``.

Plus: gradient-only (no-mixed-partial) construction is **exact for additive fields
and node-exact always**; save/load round-trips bit-for-bit.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from lemaitre.initial_data.parametric.parametric import cheb_param_nodes, ParametricSolution
from lemaitre.initial_data.parametric.parametric_nd import ParametricSolutionND
from lemaitre.initial_data.parametric.hermite import HermiteSolution1D, cardinal_deriv_at_nodes
from lemaitre.initial_data.parametric.hermite_nd import (
    HermiteSolutionND,
    HermiteSolverND,
    load_hermite_nd,
    from_problem_nd_hermite,
)


FIELD_SHAPE = (2, 3)


# --------------------------------------------------------------------------
# Analytic test-field helpers (mirrors tests/test_hermite.py's _poly_field)
# --------------------------------------------------------------------------
def _poly_field(deg, lo, hi, seed=0):
    """Degree-``deg`` polynomial field (per FIELD_SHAPE cell) in the normalized
    variable ``s∈[-1,1]`` so monomials stay O(1); returns ``(f, fp)``."""
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal((deg + 1,) + FIELD_SHAPE)
    powers = np.arange(deg, -1, -1)
    a = 2.0 / (hi - lo)
    c = (lo + hi) / (hi - lo)

    def f(x):
        s = a * float(x) - c
        return np.tensordot(s ** powers, coef, axes=(0, 0))

    def fp(x):
        s = a * float(x) - c
        dp = np.where(powers > 0, powers * s ** np.clip(powers - 1, 0, None), 0.0)
        return a * np.tensordot(dp, coef, axes=(0, 0))

    return f, fp


def _off_node_grid_probes(axes, n_per=5, guard=1e-3):
    """A few length-d query points, each component ≥ guard from any CGL node."""
    fracs = np.array([0.137, 0.371, 0.523, 0.689, 0.853, 0.293, 0.611])
    pts = []
    for i in range(n_per):
        theta = []
        for k, (lo, hi, Q) in enumerate(axes):
            fr = fracs[(i + k) % len(fracs)]
            theta.append(lo + fr * (hi - lo))
        pts.append(np.array(theta, dtype=float))
    # off-node guard
    for (lo, hi, Q), k in zip(axes, range(len(axes))):
        nodes, _ = cheb_param_nodes(lo, hi, Q)
        for th in pts:
            assert float(np.min(np.abs(th[k] - nodes))) > guard
    return pts


def _build_1d_pair(lo, hi, Q, seed):
    """A HermiteSolution1D and the equivalent d=1 HermiteSolutionND from the same
    analytic (value, tangent) node data."""
    f, fp = _poly_field(2 * Q + 1, lo, hi, seed=seed)
    nodes, weights = cheb_param_nodes(lo, hi, Q)
    U = np.stack([f(t) for t in nodes])
    dU = np.stack([fp(t) for t in nodes])
    her1d = HermiteSolution1D.from_nodes(lo, hi, Q, U, dU)
    hernd = HermiteSolutionND(
        axes=[(lo, hi, Q)], nodes=[nodes], weights=[weights],
        U_nodes=U, dU_nodes=dU[:, None, ...],           # (Q+1, d=1, *field)
        cvec=[cardinal_deriv_at_nodes(nodes)], enhanced=(0,),
        iters=np.zeros(nodes.size, int), residuals=np.zeros(nodes.size))
    return her1d, hernd, f, fp, nodes


# ==========================================================================
# H2-U1 (gate a) — d==1 reduces bit-for-bit to HermiteSolution1D
# ==========================================================================
@pytest.mark.parametrize("lo,hi,Q", [(-1.6, 1.6, 5), (1.5, 4.0, 6)])
def test_reduce_to_hermite1d_bitforbit(lo, hi, Q):
    her1d, hernd, f, fp, nodes = _build_1d_pair(lo, hi, Q, seed=Q)
    probes = _off_node_grid_probes([(lo, hi, Q)], n_per=7)
    for th in probes:
        # value: bit-for-bit
        assert np.array_equal(hernd.evaluate(th), her1d.evaluate(float(th[0])))
        # jax value: bit-for-bit
        a = np.asarray(hernd.evaluate_jax(jnp.asarray(th)))
        b = np.asarray(her1d.evaluate_jax(jnp.asarray(float(th[0]))))
        assert np.array_equal(a, b)
    # and both still reproduce the degree-2Q+1 poly exactly
    for th in probes:
        assert np.max(np.abs(hernd.evaluate(th) - f(th[0]))) < 1e-11


# ==========================================================================
# H2-U2 (gate b) — zero enhanced axes reduces bit-for-bit to ParametricSolutionND
# ==========================================================================
def _build_2d_value_only(seed=0):
    a0 = (-1.0, 2.0, 6)
    a1 = (1.5, 4.0, 5)
    n0, w0 = cheb_param_nodes(*a0)
    n1, w1 = cheb_param_nodes(*a1)
    f0, f0p = _poly_field(4, a0[0], a0[1], seed=seed)
    f1, f1p = _poly_field(4, a1[0], a1[1], seed=seed + 1)
    U = np.empty((n0.size, n1.size) + FIELD_SHAPE)
    dU = np.zeros((n0.size, n1.size, 2) + FIELD_SHAPE)   # arbitrary (unused, enh=())
    for i, t0 in enumerate(n0):
        for j, t1 in enumerate(n1):
            U[i, j] = f0(t0) * f1(t1)
    iters = np.zeros((n0.size, n1.size), int)
    resid = np.zeros((n0.size, n1.size))
    dense = ParametricSolutionND(axes=[a0, a1], nodes=[n0, n1], weights=[w0, w1],
                                 U_nodes=U, iters=iters, residuals=resid)
    hernd = HermiteSolutionND(axes=[a0, a1], nodes=[n0, n1], weights=[w0, w1],
                              U_nodes=U, dU_nodes=dU,
                              cvec=[cardinal_deriv_at_nodes(n0), cardinal_deriv_at_nodes(n1)],
                              enhanced=(), iters=iters, residuals=resid)
    return dense, hernd, [a0, a1]


def test_reduce_to_parametric_nd_zero_enhanced_bitforbit():
    dense, hernd, axes = _build_2d_value_only()
    probes = _off_node_grid_probes(axes, n_per=7)
    for th in probes:
        assert np.array_equal(hernd.evaluate(th), dense.evaluate(th))
        a = np.asarray(hernd.evaluate_jax(jnp.asarray(th)))
        b = np.asarray(dense.evaluate_jax(jnp.asarray(th)))
        assert np.array_equal(a, b)
    # at grid nodes both take the exact-node branch → stored value
    n0, n1 = hernd.nodes
    for i in (0, n0.size // 2, n0.size - 1):
        for j in (0, n1.size - 1):
            th = np.array([n0[i], n1[j]])
            assert np.array_equal(hernd.evaluate(th), dense.evaluate(th))


# ==========================================================================
# H2-U3 — gradient-only construction: EXACT for additive fields, node-exact always
# ==========================================================================
def _build_2d_enhanced(field_kind, Q0=5, Q1=4, seed=3):
    """2-D grid, BOTH axes enhanced.  ``field_kind='additive'`` ⇒ mixed partial 0
    (gradient-only is exact); ``'product'`` ⇒ nonzero mixed partial (node-exact
    only)."""
    a0 = (-1.0, 2.0, Q0)
    a1 = (0.5, 3.0, Q1)
    n0, w0 = cheb_param_nodes(*a0)
    n1, w1 = cheb_param_nodes(*a1)
    f0, f0p = _poly_field(2 * Q0 + 1, a0[0], a0[1], seed=seed)
    f1, f1p = _poly_field(2 * Q1 + 1, a1[0], a1[1], seed=seed + 1)
    U = np.empty((n0.size, n1.size) + FIELD_SHAPE)
    dU = np.empty((n0.size, n1.size, 2) + FIELD_SHAPE)
    for i, t0 in enumerate(n0):
        for j, t1 in enumerate(n1):
            if field_kind == "additive":
                U[i, j] = f0(t0) + f1(t1)
                dU[i, j, 0] = f0p(t0)
                dU[i, j, 1] = f1p(t1)
            else:                                        # product
                U[i, j] = f0(t0) * f1(t1)
                dU[i, j, 0] = f0p(t0) * f1(t1)
                dU[i, j, 1] = f0(t0) * f1p(t1)
    her = HermiteSolutionND(axes=[a0, a1], nodes=[n0, n1], weights=[w0, w1],
                            U_nodes=U, dU_nodes=dU,
                            cvec=[cardinal_deriv_at_nodes(n0), cardinal_deriv_at_nodes(n1)],
                            enhanced=(0, 1),
                            iters=np.zeros((n0.size, n1.size), int),
                            residuals=np.zeros((n0.size, n1.size)))

    def truth(th):
        return (f0(th[0]) + f1(th[1])) if field_kind == "additive" else f0(th[0]) * f1(th[1])

    return her, truth, [a0, a1]


def test_gradient_only_exact_for_additive_field():
    """Additive field (zero mixed partial): gradient-only Hermite over BOTH
    enhanced axes reproduces it to interpolation floor — validates the tensor
    product and that dropping the (vanishing) mixed partial is exact here."""
    her, truth, axes = _build_2d_enhanced("additive")
    for th in _off_node_grid_probes(axes, n_per=7):
        assert np.max(np.abs(her.evaluate(th) - truth(th))) < 1e-11


def test_multi_enhanced_node_exact_even_nonseparable():
    """With TWO enhanced axes and a nonseparable (product) field the gradient-only
    interpolant is still exact AT every node (h_i(θ_j)=δ_ij, ĥ_i(θ_j)=0)."""
    her, truth, axes = _build_2d_enhanced("product")
    n0, n1 = her.nodes
    for i in range(n0.size):
        for j in range(n1.size):
            th = np.array([n0[i], n1[j]])
            assert np.max(np.abs(her.evaluate(th) - her.U_nodes[i, j])) == 0.0
            assert np.max(np.abs(her.evaluate(th) - truth(th))) < 1e-11


def test_single_enhanced_axis_exact_2d():
    """One enhanced axis + one value-only axis: the tensor product is EXACT (no
    mixed partial needed).  A field of Hermite-degree 2Q0+1 in the enhanced axis
    and Lagrange-degree Q1 in the value-only axis is reproduced exactly."""
    a0 = (-1.0, 2.0, 5)               # enhanced (degree 2*5+1 = 11)
    a1 = (0.5, 3.0, 4)               # value-only (degree 4 = Q1)
    n0, w0 = cheb_param_nodes(*a0)
    n1, w1 = cheb_param_nodes(*a1)
    f0, f0p = _poly_field(2 * a0[2] + 1, a0[0], a0[1], seed=11)
    g1, _ = _poly_field(a1[2], a1[0], a1[1], seed=12)     # value-only: degree Q1
    U = np.empty((n0.size, n1.size) + FIELD_SHAPE)
    dU = np.zeros((n0.size, n1.size, 2) + FIELD_SHAPE)
    for i, t0 in enumerate(n0):
        for j, t1 in enumerate(n1):
            U[i, j] = f0(t0) * g1(t1)
            dU[i, j, 0] = f0p(t0) * g1(t1)               # dU/dθ0 (enhanced)
            # dU[..,1] unused (axis 1 value-only)
    her = HermiteSolutionND(axes=[a0, a1], nodes=[n0, n1], weights=[w0, w1],
                            U_nodes=U, dU_nodes=dU,
                            cvec=[cardinal_deriv_at_nodes(n0), cardinal_deriv_at_nodes(n1)],
                            enhanced=(0,),
                            iters=np.zeros((n0.size, n1.size), int),
                            residuals=np.zeros((n0.size, n1.size)))
    for th in _off_node_grid_probes([a0, a1], n_per=7):
        assert np.max(np.abs(her.evaluate(th) - f0(th[0]) * g1(th[1]))) < 1e-11


# ==========================================================================
# H2-U4 — persistence round-trips bit-for-bit
# ==========================================================================
def test_save_load_roundtrip_bitforbit(tmp_path):
    her, truth, axes = _build_2d_enhanced("product", seed=9)
    her.iters = np.full(her.U_nodes.shape[:2], 3, dtype=int)
    her.residuals = np.full(her.U_nodes.shape[:2], 1e-11)
    path = her.save(tmp_path / "h2_model",
                    meta={"axis_names": ["b", "chi_B"], "note": "H2 test"})
    back = load_hermite_nd(path)

    assert back.axes == her.axes
    assert tuple(back.enhanced) == tuple(her.enhanced)
    for a, b in [(back.U_nodes, her.U_nodes), (back.dU_nodes, her.dU_nodes),
                 (back.iters, her.iters), (back.residuals, her.residuals)]:
        assert np.array_equal(a, b)
    for kk in range(her.d):
        assert np.array_equal(back.nodes[kk], her.nodes[kk])
        assert np.array_equal(back.weights[kk], her.weights[kk])
        assert np.array_equal(back.cvec[kk], her.cvec[kk])
    assert back.meta["kind"] == "hermite_nd"
    assert back.meta["axis_names"] == ["b", "chi_B"]
    for th in _off_node_grid_probes(axes, n_per=5):
        assert np.array_equal(back.evaluate(th), her.evaluate(th))


def test_load_rejects_wrong_kind(tmp_path):
    """A dense ParametricSolutionND artifact must not load as a hermite_nd."""
    dense, _, _ = _build_2d_value_only()
    p = dense.save(tmp_path / "dense")
    with pytest.raises(ValueError):
        load_hermite_nd(p)


# ==========================================================================
# H2-T1 — add-only / standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lemaitre.initial_data.parametric.hermite_nd as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm", "import context", "torch"):
        assert forbidden not in src, forbidden


# ==========================================================================
# Solver-backed gates (slow) — on solver_abt at the 44×32 grid
# ==========================================================================
_AXIS_B = dict(name="b", lo=1.5, hi=4.0, fixed={"q": 1.0, "chi_A": 0.0, "chi_B": 0.0})
M_TOT = 1.0
P_MOM = 0.5


def _holdout_b(n=15):
    fr = (np.arange(1, n + 1) - 0.5) / n
    fr = (fr + 0.0173 * np.sin(3.0 * np.arange(1, n + 1))) % 1.0
    fr = np.clip(fr, 0.03, 0.97)
    return _AXIS_B["lo"] + fr * (_AXIS_B["hi"] - _AXIS_B["lo"])


def _fit_rate(Qs, errs, floor=3e-9):
    Qs = np.asarray(Qs, float)
    errs = np.asarray(errs, float)
    m = errs > floor
    if m.sum() < 2:
        return np.nan
    return -float(np.polyfit(Qs[m], np.log10(errs[m]), 1)[0])


def _relL2(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.slow
def test_held_out_rate_beats_value_only_b():
    """Gate (c): productized Phase-0 Q1.  Build value-only AND Hermite (enhanced
    ``b``) from the SAME node set at several Q; compare held-out relative-L2 field
    error vs a direct same-grid solve.  Hermite's geometric rate must materially
    beat value-only, and Hermite must be no worse at every Q."""
    from lemaitre.initial_data.solver import solver_abt as sa
    prob = sa.make_problem(Na=44, Nb=32, P=P_MOM)

    hold = _holdout_b(n=15)
    from lemaitre.initial_data.parametric.parametric_nd_2c import theta_to_slice
    U_true = {}
    for p in hold:
        sl = theta_to_slice([float(p)], ["b"], M_TOT, _AXIS_B["fixed"])
        Up, _ = sa.newton_solve(prob, sl, tol=1e-12, max_iter=25)
        U_true[float(p)] = np.asarray(Up)

    Qs = [2, 3, 4, 5, 6, 7]
    val_med, her_med = [], []
    for Q in Qs:
        axes = [{"name": "b", "min": _AXIS_B["lo"], "max": _AXIS_B["hi"], "Q": Q}]
        her = from_problem_nd_hermite(prob, axes, enhanced=["b"], M_tot=M_TOT,
                                      fixed=_AXIS_B["fixed"], use_cache=True).build(
            tol=1e-12, max_iter=20)
        # value-only interpolant from the SAME solved nodes (fair comparison)
        dense = ParametricSolutionND(
            axes=her.axes, nodes=her.nodes, weights=her.weights,
            U_nodes=her.U_nodes, iters=her.iters, residuals=her.residuals)
        ev = [_relL2(dense.evaluate([float(p)]), U_true[float(p)]) for p in hold]
        eh = [_relL2(her.evaluate([float(p)]), U_true[float(p)]) for p in hold]
        val_med.append(float(np.median(ev)))
        her_med.append(float(np.median(eh)))

    # (i) Hermite never worse at any Q
    for ev, eh in zip(val_med, her_med):
        assert eh <= ev * (1 + 1e-6), (eh, ev)
    # (ii) geometric rate materially better (Phase-0 measured 2.22x on b)
    rate_v = _fit_rate(Qs, val_med)
    rate_h = _fit_rate(Qs, her_med)
    assert rate_h > rate_v, (rate_h, rate_v)
    assert rate_h > 1.4 * rate_v, (rate_h, rate_v)
    # (iii) clear field-accuracy separation at the finer end
    assert min(her_med) < 1e-8, min(her_med)
    assert min(her_med) < 0.01 * val_med[-1], (min(her_med), val_med[-1])


@pytest.mark.slow
def test_evaluate_polished_certifies():
    """Gate (d): the certification guarantee is unchanged — the Hermite object is
    only a guess; ``evaluate_polished`` (Newton on the real solver) drives random
    off-node points to ``‖R‖∞ ≤ 1e-10``."""
    from lemaitre.initial_data.solver import solver_abt as sa
    prob = sa.make_problem(Na=44, Nb=32, P=P_MOM)
    axes = [{"name": "b", "min": _AXIS_B["lo"], "max": _AXIS_B["hi"], "Q": 6}]
    her = from_problem_nd_hermite(prob, axes, enhanced=["b"], M_tot=M_TOT,
                                  fixed=_AXIS_B["fixed"], use_cache=True).build(
        tol=1e-12, max_iter=20)
    rng = np.random.default_rng(1)
    pts = _AXIS_B["lo"] + rng.uniform(0.05, 0.95, 6) * (_AXIS_B["hi"] - _AXIS_B["lo"])
    for p in pts:
        U, info = her.evaluate_polished([float(p)], newton_steps=5, tol=1e-12)
        assert info.residual_norm <= 1e-10, (float(p), info.residual_norm)
