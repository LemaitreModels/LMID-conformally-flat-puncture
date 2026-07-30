"""H1 acceptance — 1-D gradient-enhanced (Hermite) interpolant.

The foundation milestone of ``GRADIENT_ENHANCED_PLAN.md`` §4 (H1).  Pure
interpolant math (no solver needed): all fields here are analytic test fields, so
the checks are fast and standalone.  Follows ``tests/test_sensitivity.py`` style
with explicit tolerances.

Gates (§4 H1):
  * reproduces a degree-2Q+1 polynomial in θ — value AND derivative — to ≤1e-12 at
    strictly OFF-node points (both the numpy analytic grad and jax autodiff);
  * matches value and first derivative at every node bit-for-bit;
  * reduce-to-committed: Taylor order=0 returns U_node exactly, and on a
    value-only node set the Hermite and committed barycentric interpolants of the
    SAME analytic field agree as Q→large;
  * save/load round-trips bit-for-bit.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from lemaitre.initial_data.parametric.parametric import cheb_param_nodes, ParametricSolution
from lemaitre.initial_data.parametric.hermite import (
    HermiteSolution1D,
    hermite_cardinal,
    cardinal_deriv_at_nodes,
    taylor_predict,
    load_hermite,
)


# --------------------------------------------------------------------------
# Helpers: build a per-cell polynomial / analytic field on a small field shape
# --------------------------------------------------------------------------
FIELD_SHAPE = (2, 3)


def _poly_field(deg, lo, hi, seed=0):
    """A degree-``deg`` polynomial field: each of the FIELD_SHAPE cells is an
    independent polynomial in θ.  Returns ``(f, fp)`` with ``f(x)``/``fp(x)`` →
    ``FIELD_SHAPE`` arrays (value and θ-derivative).

    Built in the normalized variable ``s(θ)=(2θ−(lo+hi))/(hi−lo)∈[-1,1]`` so the
    monomials stay O(1) regardless of the interval — a degree-``deg`` polynomial
    in ``s`` is a degree-``deg`` polynomial in ``θ``, so this still exercises
    degree-``2Q+1`` exactness, but keeps absolute tolerances meaningful (raw
    ``θ^deg`` on ``[1.5,4]`` would be ~1e8 and swamp a 1e-12 absolute check)."""
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal((deg + 1,) + FIELD_SHAPE)     # highest power first
    powers = np.arange(deg, -1, -1)
    a = 2.0 / (hi - lo)                                       # ds/dθ
    c = (lo + hi) / (hi - lo)

    def f(x):
        s = a * np.asarray(x, float) - c
        return np.tensordot(s ** powers, coef, axes=(0, 0))

    def fp(x):
        s = a * np.asarray(x, float) - c
        dp = np.where(powers > 0, powers * s ** np.clip(powers - 1, 0, None), 0.0)
        return a * np.tensordot(dp, coef, axes=(0, 0))        # chain rule ds/dθ

    return f, fp


def _analytic_field(x):
    """A smooth, genuinely analytic (non-polynomial) field and its derivative."""
    x = np.asarray(x, float)
    base = np.exp(-0.7 * x) * np.cos(1.3 * x)
    dbase = np.exp(-0.7 * x) * (-0.7 * np.cos(1.3 * x) - 1.3 * np.sin(1.3 * x))
    scale = np.array([[1.0, -0.5, 0.3], [0.7, 1.2, -0.9]])          # FIELD_SHAPE
    return base * scale, dbase * scale


def _off_node_probes(lo, hi, nodes, n=37, guard=1e-3):
    """Evenly-spaced probes in ``(lo, hi)`` all at least ``guard`` from any node."""
    xs = np.linspace(lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo), n)
    keep = [x for x in xs if np.min(np.abs(x - nodes)) > guard]
    return np.array(keep)


def _build_hermite_from_field(lo, hi, Q, f, fp):
    nodes, _ = cheb_param_nodes(lo, hi, Q)
    U = np.stack([f(t) for t in nodes])
    dU = np.stack([fp(t) for t in nodes])
    return HermiteSolution1D.from_nodes(lo, hi, Q, U, dU)


# ==========================================================================
# H1-U1 — degree-2Q+1 polynomial reproduction (value AND derivative), off-node
# ==========================================================================
@pytest.mark.parametrize("lo,hi,Q", [(-1.6, 1.6, 5), (1.5, 4.0, 6)])
def test_hermite_reproduces_poly_2Qp1_off_node(lo, hi, Q):
    deg = 2 * Q + 1
    f, fp = _poly_field(deg, lo, hi, seed=Q)
    sol = _build_hermite_from_field(lo, hi, Q, f, fp)
    xs = _off_node_probes(lo, hi, sol.nodes)
    assert xs.size >= 10

    # value
    verr = max(np.max(np.abs(sol.evaluate(float(x)) - f(x))) for x in xs)
    assert verr < 1e-12, verr

    # derivative — numpy analytic path (evaluate_grad)
    gerr_np = max(np.max(np.abs(sol.evaluate_grad(float(x)) - fp(x))) for x in xs)
    assert gerr_np < 1e-12, gerr_np

    # derivative — jax autodiff path (the exposed-gradient hook used by the apps)
    jgrad = jax.jacfwd(sol.evaluate_jax)
    gerr_jax = max(np.max(np.abs(np.asarray(jgrad(jnp.asarray(float(x)))) - fp(x)))
                   for x in xs)
    assert gerr_jax < 1e-12, gerr_jax


def test_numpy_and_jax_value_agree_off_node():
    lo, hi, Q = -1.6, 1.6, 6
    f, fp = _poly_field(2 * Q + 1, lo, hi, seed=3)
    sol = _build_hermite_from_field(lo, hi, Q, f, fp)
    for x in _off_node_probes(lo, hi, sol.nodes, n=11):
        a = sol.evaluate(float(x))
        b = np.asarray(sol.evaluate_jax(jnp.asarray(float(x))))
        assert np.max(np.abs(a - b)) < 1e-13


def test_evaluate_batch_matches_scalar_off_node():
    lo, hi, Q = 1.5, 4.0, 5
    f, fp = _poly_field(2 * Q + 1, lo, hi, seed=7)
    sol = _build_hermite_from_field(lo, hi, Q, f, fp)
    xs = _off_node_probes(lo, hi, sol.nodes, n=9)
    batch = sol.evaluate(xs)                              # (len(xs), *field)
    assert batch.shape == (xs.size,) + FIELD_SHAPE
    for j, x in enumerate(xs):
        assert np.array_equal(batch[j], sol.evaluate(float(x)))


# ==========================================================================
# H1-U2 — value AND first derivative match at every node, bit-for-bit
# ==========================================================================
def test_matches_value_and_deriv_at_nodes_bitforbit():
    lo, hi, Q = -1.6, 1.6, 6
    f, fp = _poly_field(2 * Q + 1, lo, hi, seed=1)
    nodes, _ = cheb_param_nodes(lo, hi, Q)
    U = np.stack([f(t) for t in nodes])
    dU = np.stack([fp(t) for t in nodes])
    sol = HermiteSolution1D.from_nodes(lo, hi, Q, U, dU)
    for i in range(nodes.size):
        # value: node-safe branch returns the stored field exactly
        assert np.array_equal(sol.evaluate(float(nodes[i])), U[i])
        # derivative: node-safe grad returns the stored tangent exactly
        assert np.array_equal(sol.evaluate_grad(float(nodes[i])), dU[i])


def test_hermite_cardinal_delta_properties():
    """The cardinal pair satisfies h_i(θ_j)=δ_ij, ĥ_i(θ_j)=0, h_i'(θ_j)=0,
    ĥ_i'(θ_j)=δ_ij — checked at strictly off-node probes reconstructing the
    identity, plus the c_i node-set formula."""
    lo, hi, Q = -1.0, 2.0, 5
    nodes, weights = cheb_param_nodes(lo, hi, Q)
    card = hermite_cardinal(nodes, weights)
    # c_i matches the standalone accessor and the sum formula
    assert np.allclose(card.cvec, cardinal_deriv_at_nodes(nodes), atol=0, rtol=0)
    # Σ_i h_i(θ) ≡ 1 (partition of unity for the value cardinals) off-node
    for x in _off_node_probes(lo, hi, nodes, n=7):
        assert abs(np.sum(card.h(float(x))) - 1.0) < 1e-12
    # h,ĥ callables reproduce a linear field f(x)=a+b x with tangent b exactly
    a, b = 0.37, -1.21
    U = a + b * nodes
    dU = np.full_like(nodes, b)
    for x in _off_node_probes(lo, hi, nodes, n=7):
        val = float(card.h(float(x)) @ U + card.hhat(float(x)) @ dU)
        assert abs(val - (a + b * x)) < 1e-12


# ==========================================================================
# H1-U3 — Taylor predictor (single-node degenerate mode)
# ==========================================================================
def test_taylor_order0_returns_node_value_exactly():
    rng = np.random.default_rng(0)
    U = rng.standard_normal(FIELD_SHAPE)
    dU = rng.standard_normal(FIELD_SHAPE)
    out = taylor_predict(2.5, U, dU, dtheta=0.31, order=0)
    assert np.array_equal(out, U)


def test_taylor_order1_first_order():
    rng = np.random.default_rng(2)
    U = rng.standard_normal(FIELD_SHAPE)
    dU = rng.standard_normal(FIELD_SHAPE)
    dth = 0.137
    out = taylor_predict(2.5, U, dU, dtheta=dth, order=1)
    assert np.allclose(out, U + dU * dth, atol=0, rtol=0)


def test_taylor_order2_matches_quadratic():
    """With exact 1st+2nd derivatives, order=2 reproduces a quadratic exactly."""
    a, b, c = 1.3, -0.7, 0.45
    node = 2.0
    for dth in (0.0, 0.05, -0.2, 0.4):
        U = np.array([a + b * node + c * node ** 2])
        d1 = np.array([b + 2 * c * node])
        d2 = np.array([2 * c])
        out = taylor_predict(node, U, [d1, d2], dtheta=dth, order=2)
        truth = a + b * (node + dth) + c * (node + dth) ** 2
        assert abs(float(out[0]) - truth) < 1e-13


def test_taylor_insufficient_derivs_raises():
    U = np.zeros(FIELD_SHAPE)
    dU = np.ones(FIELD_SHAPE)
    with pytest.raises(ValueError):
        taylor_predict(2.0, U, dU, dtheta=0.1, order=2)   # only 1 derivative given


# ==========================================================================
# H1-U4 — reduce-to-committed: value-only barycentric agreement as Q→large
# ==========================================================================
def test_reduce_to_committed_value_only_node_set():
    """On the SAME analytic field, the Hermite interpolant and the committed
    value-only barycentric ``ParametricSolution`` both converge spectrally; their
    difference shrinks as Q grows, and Hermite is never worse (higher order)."""
    lo, hi = -1.0, 2.0
    # dense held-out truth
    xs_truth = _off_node_probes(lo, hi, cheb_param_nodes(lo, hi, 30)[0], n=41)

    def relL2(approx_list, truth_list):
        a = np.concatenate([np.ravel(v) for v in approx_list])
        t = np.concatenate([np.ravel(v) for v in truth_list])
        return float(np.linalg.norm(a - t) / np.linalg.norm(t))

    truth = [_analytic_field(x)[0] for x in xs_truth]
    err_val, err_her, diff_vh = [], [], []
    Qs = [4, 8, 16]
    for Q in Qs:
        nodes, weights = cheb_param_nodes(lo, hi, Q)
        U = np.stack([_analytic_field(t)[0] for t in nodes])
        dU = np.stack([_analytic_field(t)[1] for t in nodes])
        # committed value-only interpolant (constructed directly, no solver)
        ps = ParametricSolution(q_min=lo, q_max=hi, Q=Q, q_nodes=nodes, weights=weights,
                                U_nodes=U, iters=[0] * nodes.size,
                                residuals=[0.0] * nodes.size)
        her = HermiteSolution1D.from_nodes(lo, hi, Q, U, dU)
        v = [ps.evaluate(float(x)) for x in xs_truth]
        h = [her.evaluate(float(x)) for x in xs_truth]
        err_val.append(relL2(v, truth))
        err_her.append(relL2(h, truth))
        diff_vh.append(relL2(v, h))

    # both converge; Hermite at least as accurate at every Q (higher per-node order)
    for ev, eh in zip(err_val, err_her):
        assert eh <= ev * (1 + 1e-9), (eh, ev)
    # error decreases with Q for both (spectral)
    assert err_val[-1] < err_val[0] and err_her[-1] < err_her[0]
    # at large Q both are converged and therefore agree
    assert err_val[-1] < 1e-6, err_val
    assert err_her[-1] < 1e-9, err_her
    assert diff_vh[-1] < 1e-6, diff_vh


def test_reduce_to_committed_1d_barycentric_at_nodes():
    """Sanity: at CGL nodes the committed 1-D barycentric and the Hermite value
    both return the stored value (the node-safe branch)."""
    lo, hi, Q = -1.0, 2.0, 6
    nodes, weights = cheb_param_nodes(lo, hi, Q)
    U = np.stack([_analytic_field(t)[0] for t in nodes])
    dU = np.stack([_analytic_field(t)[1] for t in nodes])
    ps = ParametricSolution(q_min=lo, q_max=hi, Q=Q, q_nodes=nodes, weights=weights,
                            U_nodes=U, iters=[0] * nodes.size,
                            residuals=[0.0] * nodes.size)
    her = HermiteSolution1D.from_nodes(lo, hi, Q, U, dU)
    for i in range(nodes.size):
        assert np.array_equal(ps.evaluate(float(nodes[i])), her.evaluate(float(nodes[i])))


# ==========================================================================
# H1-U5 — persistence round-trips bit-for-bit
# ==========================================================================
def test_save_load_roundtrip_bitforbit(tmp_path):
    lo, hi, Q = 1.5, 4.0, 6
    f, fp = _poly_field(2 * Q + 1, lo, hi, seed=9)
    nodes, _ = cheb_param_nodes(lo, hi, Q)
    U = np.stack([f(t) for t in nodes])
    dU = np.stack([fp(t) for t in nodes])
    sol = HermiteSolution1D.from_nodes(
        lo, hi, Q, U, dU,
        iters=np.array([3] * nodes.size), residuals=np.full(nodes.size, 1e-11))
    path = sol.save(tmp_path / "h1_model", meta={"axis_name": "b", "note": "H1 test"})
    back = load_hermite(path)

    assert back.lo == sol.lo and back.hi == sol.hi and back.Q == sol.Q
    for a, b in [(back.nodes, sol.nodes), (back.weights, sol.weights),
                 (back.U_nodes, sol.U_nodes), (back.dU_nodes, sol.dU_nodes),
                 (back.cvec, sol.cvec), (back.iters, sol.iters),
                 (back.residuals, sol.residuals)]:
        assert np.array_equal(a, b)
    assert back.meta["kind"] == "hermite1d"
    assert back.meta["axis_name"] == "b"
    # evaluate matches bit-for-bit off-node
    for x in _off_node_probes(lo, hi, sol.nodes, n=5):
        assert np.array_equal(back.evaluate(float(x)), sol.evaluate(float(x)))
        assert np.array_equal(back.evaluate_grad(float(x)), sol.evaluate_grad(float(x)))


def test_load_rejects_wrong_kind(tmp_path):
    """A dense ParametricSolutionND artifact must not load as a hermite1d."""
    from lemaitre.initial_data.parametric.parametric_nd import ParametricSolutionND
    nodes, weights = cheb_param_nodes(1.0, 3.0, 4)
    U = np.stack([np.ones(FIELD_SHAPE) * t for t in nodes])
    dense = ParametricSolutionND(
        axes=[(1.0, 3.0, 4)], nodes=[nodes], weights=[weights],
        U_nodes=U, iters=np.zeros(nodes.size, int), residuals=np.zeros(nodes.size))
    p = dense.save(tmp_path / "dense")
    with pytest.raises(ValueError):
        load_hermite(p)


# ==========================================================================
# H1-T1 — add-only / standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lemaitre.initial_data.parametric.hermite as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm", "import context", "torch"):
        assert forbidden not in src, forbidden
