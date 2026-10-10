"""Acceptance tests for the pool-weight evaluation path (``parametric/pooled.py``).

The pooled path re-associates the combination sum into GEMVs against the
deduplicated node pool, and **it is what ``evaluate`` does** since the default
flip (2026-08-18); the direct combination sum is retained as
``evaluate_subgrid_sum``, which is the reference every equivalence gate here
compares against.  Its failure mode is silent: a wrong subgrid→pool index map
produces a smooth, plausible, WRONG field that still certifies (the polish
repairs any guess), so the certification gate cannot catch it — only a field
comparison can.  What must hold, on every member of the combination family
(value-only / gradient-enhanced / full-bilinear cross / both POD wrappers):

* the pooled path agrees with the reference ``evaluate_subgrid_sum`` to
  roundoff at generic θ — same interpolant, different summation order;
* the default ``evaluate`` really is the pooled path, so that an accidental
  revert is a failure here and not a silent change of every figure;
* the exact-node guard fires at the same θ and returns the same values;
* the jax twin agrees off-node, and so does its ``jacfwd`` (the exposed
  gradient the applications differentiate);
* a batch evaluation equals the stacked single-point evaluations;
* the value weights sum to 1 (the combination technique reproduces constants,
  ``Σ_l c_l = 1`` — a partition-of-unity invariant of the weight vector).

The models here are built on an ANALYTIC solve_fn (no elliptic solver), so the
whole file runs in seconds; the shipped-model equivalence + held-out
field-error gate is a producer-scale run recorded in the findings, not a test.
"""
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lemaitre.initial_data.conformally_flat_puncture.parametric import (
    parametric_nd_smolyak as smol,
    hermite_smolyak as hsm,
    hermite_smolyak_cross as hsc,
    hermite_smolyak_pod as hsp,
    hermite_smolyak_pod_cross as hspc,
)
from lemaitre.initial_data.conformally_flat_puncture.parametric.pooled import (
    pooled_evaluator,
)

AXES = [(2.0, 3.2), (0.0, 0.3), (-0.5, 0.5)]
ENH = (0, 2)
LEVEL = 3
FIELD_SHAPE = (3, 2)


# --------------------------------------------------------------------------
# analytic stand-in for the solver: smooth field + exact tangents/cross
# --------------------------------------------------------------------------
def _field(theta):
    b, s, c = theta
    base = np.array([[np.sin(b) * np.cosh(s), b * c],
                     [np.exp(0.3 * c) + s * b, np.cos(2.0 * s + c)],
                     [b ** 2 * c, s + 0.1 * b * c ** 2]])
    return base


def _tangent(theta):
    h = 1e-6
    cols = []
    for k in range(3):
        tp = np.array(theta, float); tp[k] += h
        tm = np.array(theta, float); tm[k] -= h
        cols.append((_field(tp) - _field(tm)) / (2 * h))
    return np.stack(cols, axis=0)


def _cross_pair(theta, e0, e1):
    h = 1e-5
    t = np.array(theta, float)
    tpp = t.copy(); tpp[[e0, e1]] += [h, h]
    tpm = t.copy(); tpm[[e0, e1]] += [h, -h]
    tmp = t.copy(); tmp[[e0, e1]] += [-h, h]
    tmm = t.copy(); tmm[[e0, e1]] += [-h, -h]
    return (_field(tpp) - _field(tpm) - _field(tmp) + _field(tmm)) / (4 * h * h)


class _Info:
    iters = 1
    residual_norm = 1e-14
    converged = True


def _solve_fn(theta, guess, tol, max_iter):
    return _field(np.asarray(theta, float)), _Info()


def _tangent_fn(theta, U):
    return _tangent(np.asarray(theta, float))


# --------------------------------------------------------------------------
# the five family members
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def value_model():
    return smol.SmolyakSolverND(_solve_fn, AXES).build_isotropic(LEVEL)


@pytest.fixture(scope="module")
def hermite_model():
    return hsm.HermiteSmolyakSolverND(
        _solve_fn, AXES, _tangent_fn, enhanced_axes=ENH).build_isotropic(LEVEL)


@pytest.fixture(scope="module")
def cross_model(hermite_model):
    pairs = hsc._global_pairs(ENH)
    pool = {}
    for key, (theta, U, dU, it, rs) in hermite_model._dedup_pool().items():
        cross = np.stack([_cross_pair(theta, e0, e1) for (e0, e1) in pairs])
        pool[key] = (theta, U, dU, cross, it, rs)
    return hsc.build_cross_from_pool(AXES, hermite_model.index_set, ENH, pool)


@pytest.fixture(scope="module")
def pod_model(hermite_model):
    pod, _ = hsp.build_pod_hermite_smolyak(hermite_model, r=5)
    return pod


@pytest.fixture(scope="module")
def pod_cross_model(cross_model):
    pod, _ = hspc.build_pod_hermite_smolyak_cross(cross_model, r=5)
    return pod


ALL = ["value_model", "hermite_model", "cross_model", "pod_model", "pod_cross_model"]


def _models(request, names=ALL):
    return [(n, request.getfixturevalue(n)) for n in names]


def _rng_thetas(n, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([a[0] for a in AXES])
    hi = np.array([a[1] for a in AXES])
    return lo + (0.05 + 0.9 * rng.random((n, 3))) * (hi - lo)


def _scale(model):
    th = _rng_thetas(1, seed=99)[0]
    return float(np.max(np.abs(model.evaluate(th)))) or 1.0


# ==========================================================================
# P1 — pooled == oracle at generic θ, on every family member
# ==========================================================================
@pytest.mark.parametrize("name", ALL)
def test_pooled_matches_oracle(request, name):
    model = request.getfixturevalue(name)
    ev = pooled_evaluator(model)
    sc = _scale(model)
    worst = 0.0
    for th in _rng_thetas(25):
        a = np.asarray(model.evaluate_subgrid_sum(th))
        b = np.asarray(ev.evaluate(th))
        worst = max(worst, float(np.max(np.abs(a - b))) / sc)
    assert worst < 1e-12, f"{name}: pooled vs oracle rel diff {worst:.2e}"


# ==========================================================================
# P2 — the exact-node guard fires at the same θ with the same values
# ==========================================================================
@pytest.mark.parametrize("name", ALL)
def test_pooled_node_guard(request, name):
    model = request.getfixturevalue(name)
    ev = pooled_evaluator(model)
    sc = _scale(model)
    # hit level-2 nodes on axes 0 and 2 (one enhanced, one value axis at that
    # level), keep axis 1 generic — the guard must fire per axis
    n0, _ = smol.nested_levels(*AXES[0], 2)
    n2, _ = smol.nested_levels(*AXES[2], 2)
    for th in [np.array([n0[1], 0.17, 0.123]),
               np.array([2.345, 0.21, n2[2]]),
               np.array([n0[2], 0.11, n2[0]])]:
        a = np.asarray(model.evaluate_subgrid_sum(th))
        b = np.asarray(ev.evaluate(th))
        assert float(np.max(np.abs(a - b))) / sc < 1e-12


# ==========================================================================
# P3 — jax twin and its jacfwd agree off-node
# ==========================================================================
@pytest.mark.parametrize("name", ALL)
def test_pooled_jax_matches(request, name):
    model = request.getfixturevalue(name)
    ev = pooled_evaluator(model)
    sc = _scale(model)
    for th in _rng_thetas(5, seed=1):
        a = np.asarray(model.evaluate_jax_subgrid_sum(jnp.asarray(th)))
        b = np.asarray(ev.evaluate_jax(jnp.asarray(th)))
        assert float(np.max(np.abs(a - b))) / sc < 1e-12


def test_pooled_jacfwd_matches(pod_cross_model):
    """The exposed gradient (what qc_targeting/qc_effpot differentiate)."""
    ev = pooled_evaluator(pod_cross_model)
    Ja = jax.jacfwd(pod_cross_model.evaluate_jax_subgrid_sum)
    Jb = jax.jacfwd(ev.evaluate_jax)
    for th in _rng_thetas(3, seed=2):
        a = np.asarray(Ja(jnp.asarray(th)))
        b = np.asarray(Jb(jnp.asarray(th)))
        den = max(float(np.max(np.abs(a))), 1e-30)
        assert float(np.max(np.abs(a - b))) / den < 1e-10


# ==========================================================================
# P4 — batch == stacked singles
# ==========================================================================
@pytest.mark.parametrize("name", ["cross_model", "pod_cross_model"])
def test_pooled_batch_matches_singles(request, name):
    model = request.getfixturevalue(name)
    ev = pooled_evaluator(model)
    thetas = _rng_thetas(7, seed=3)
    batch = ev.evaluate_batch(thetas)
    singles = np.stack([np.asarray(ev.evaluate(th)) for th in thetas])
    assert np.allclose(batch, singles, rtol=0, atol=1e-13 * _scale(model))


# ==========================================================================
# P5 — the value weights are a partition of unity (Σ_l c_l = 1)
# ==========================================================================
def test_pooled_weights_partition_of_unity(cross_model):
    ev = pooled_evaluator(cross_model)
    for th in _rng_thetas(5, seed=4):
        Wv, _, _ = ev.inner.weights(th) if hasattr(ev, "inner") else ev.weights(th)
        assert abs(float(np.sum(Wv)) - 1.0) < 1e-12


# ==========================================================================
# P7 — the DEFAULT evaluate is the pooled path, and the reference still differs
# ==========================================================================
@pytest.mark.parametrize("name", ALL)
def test_default_evaluate_is_the_pooled_path(request, name):
    """``evaluate`` must route through the pool weights, not the direct sum.

    This is the gate that makes the default flip visible: it fails if anyone
    reverts ``evaluate`` to the subgrid sum, which would otherwise silently
    change the guess behind every figure and every certified query.  The
    containers delegate straight to the cached evaluator, so the agreement is
    exact; the POD wrappers decode through their own numpy ``mean + Φ·c``, so
    they are held to roundoff instead.
    """
    model = request.getfixturevalue(name)
    ev = pooled_evaluator(model)
    sc = _scale(model)
    for th in _rng_thetas(5, seed=7):
        d = float(np.max(np.abs(np.asarray(model.evaluate(th))
                                - np.asarray(ev.evaluate(th)))))
        assert d / sc < 1e-14, f"{name}: evaluate is not the pooled path ({d:.2e})"


@pytest.mark.parametrize("name", ALL)
def test_reference_path_is_still_reachable(request, name):
    """The retained direct sum stays callable and stays a *different* summation.

    Reachability is the point (the historical figure data was produced with it);
    it is deliberately NOT asserted bit-equal to the default, because the whole
    reason the flip needed a figure re-run is that it is not.
    """
    model = request.getfixturevalue(name)
    sc = _scale(model)
    for th in _rng_thetas(3, seed=8):
        a = np.asarray(model.evaluate_subgrid_sum(th))
        b = np.asarray(model.evaluate(th))
        assert a.shape == b.shape
        assert float(np.max(np.abs(a - b))) / sc < 1e-12


# ==========================================================================
# P6 — evaluate_pooled convenience is wired on the containers
# ==========================================================================
def test_evaluate_pooled_methods(hermite_model, pod_cross_model):
    th = _rng_thetas(1, seed=5)[0]
    a = np.asarray(hermite_model.evaluate(th))
    b = np.asarray(hermite_model.evaluate_pooled(th))
    assert np.allclose(a, b, rtol=0, atol=1e-12 * _scale(hermite_model))
    a = np.asarray(pod_cross_model.evaluate(th))
    b = np.asarray(pod_cross_model.evaluate_pooled(th))
    assert np.allclose(a, b, rtol=0, atol=1e-12 * _scale(pod_cross_model))


# --------------------------------------------------------------------------
# the device-constant caches must survive a first use under jax.jit
# --------------------------------------------------------------------------
def _fresh(name):
    """A NEW instance of a family member: the module fixtures are shared, so
    their caches may already have been filled eagerly by an earlier test."""
    value = smol.SmolyakSolverND(_solve_fn, AXES).build_isotropic(2)
    herm = hsm.HermiteSmolyakSolverND(_solve_fn, AXES, _tangent_fn,
                                      enhanced_axes=ENH).build_isotropic(2)
    if name == "value_model":
        return value
    if name == "hermite_model":
        return herm
    pool = {}
    for key, (theta, U, dU, it, rs) in herm._dedup_pool().items():
        cross = np.stack([_cross_pair(theta, e0, e1) for (e0, e1) in hsc._global_pairs(ENH)])
        pool[key] = (theta, U, dU, cross, it, rs)
    cross = hsc.build_cross_from_pool(AXES, herm.index_set, ENH, pool)
    if name == "cross_model":
        return cross
    if name == "pod_model":
        return hsp.build_pod_hermite_smolyak(herm, r=5)[0]
    return hspc.build_pod_hermite_smolyak_cross(cross, r=5)[0]


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("route", ["evaluate_jax", "evaluate_jax_subgrid_sum"])
def test_jax_constants_cached_under_jit_are_concrete(name, route):
    """First use INSIDE a jit trace, then a second jitted function.

    The containers upload their node tensors once and cache them (``_jax_consts``).
    If that first upload happened while a ``jax.jit`` was tracing, the cache kept the
    trace's tracers, and the next jitted function died with ``UnexpectedTracerError``
    — which is how ``qc_effpot.circular_gradient`` (``dV`` traced first, then
    ``d2V``) failed on every freshly loaded model.  Nothing caught it because every
    test reached the cache eagerly first.
    """
    model = _fresh(name)
    f = getattr(model, route)
    th = jnp.asarray(_rng_thetas(1, seed=11)[0])
    val = jax.jit(lambda t: jnp.sum(f(t)))
    grad = jax.jit(jax.grad(lambda t: jnp.sum(f(t))))
    v = float(val(th))                       # fills the cache inside a trace
    g = np.asarray(grad(th))                 # a second trace reads it
    assert np.isfinite(v) and np.all(np.isfinite(g))
    assert np.isclose(v, float(np.sum(np.asarray(f(th)))), rtol=1e-12, atol=0)


def test_dense_and_1d_hermite_caches_under_jit():
    """The same contract for the dense container (``qc_effpot``'s model type) and the
    1-D Hermite one, which the Smolyak family does not exercise."""
    from lemaitre.initial_data.conformally_flat_puncture.parametric import parametric_nd as pnd
    from lemaitre.initial_data.conformally_flat_puncture.parametric import hermite as h1
    from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric import (
        cheb_param_nodes)
    nodes, weights = cheb_param_nodes(1.0, 2.0, 6)
    dense = pnd.ParametricSolutionND(
        axes=[(1.0, 2.0, 6)], nodes=[nodes], weights=[weights],
        U_nodes=np.stack([np.sin(x) * np.ones((3, 2)) for x in nodes]),
        iters=np.zeros(nodes.size, dtype=int), residuals=np.zeros(nodes.size))
    herm1 = h1.HermiteSolution1D.from_nodes(
        1.0, 2.0, 6, np.stack([np.sin(x) * np.ones((3, 2)) for x in nodes]),
        np.stack([np.cos(x) * np.ones((3, 2)) for x in nodes]))
    for f, th in ((dense.evaluate_jax, jnp.asarray([1.37])), (herm1.evaluate_jax, 1.37)):
        v = float(jax.jit(lambda t: jnp.sum(f(t)))(th))
        g = jax.jit(jax.grad(lambda t: jnp.sum(f(t))))(th)
        assert np.isfinite(v) and np.all(np.isfinite(np.asarray(g)))


def test_standalone_imports():
    import importlib
    m = importlib.import_module(
        "lemaitre.initial_data.conformally_flat_puncture.parametric.pooled")
    assert hasattr(m, "pooled_evaluator")
