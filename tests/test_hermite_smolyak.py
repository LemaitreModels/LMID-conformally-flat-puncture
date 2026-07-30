"""H5b acceptance — gradient-enhanced SPARSE (Hermite-Smolyak) collocation
(GRADIENT_ENHANCED_PLAN.md §4 H5b).

Gates:
  * **reduce-to-committed**: with zero enhanced axes the interpolant reduces
    **bit-for-bit** to :class:`parametric_nd_smolyak.SmolyakSolutionND` (H3's
    value-only limit, now inside the combination sum);
  * **node-exact for values** everywhere, and **for the enhanced-axis tangent at
    the genuinely Hermite-resolved nodes** (level ≥1 on the enhanced axis) — the
    level-0 (value-only) nodes for that axis are interpolated by design (R7/R4);
  * ``evaluate_polished`` certifies random off-node points to ``‖R‖∞ ≤ 1e-10``
    (unchanged, with the NK solver);
  * ``save``/``load`` round-trips **bit-for-bit** (value + tangent + enhanced set).

Standalone (numpy/scipy/jax); reuses the committed Smolyak primitives, the H2
``HermiteSolutionND`` subgrid, and the H5a ``sensitivity_3d`` tangent verbatim.
"""

import os
import tempfile

import numpy as np
import pytest

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3d
from lm.initial_data.parametric import parametric_nd_smolyak as smol
from lm.initial_data.parametric import hermite_smolyak as hsm
from lm.initial_data.parametric.parametric_nd_smolyak import nested_levels


M_TOT = 1.0
FIXED = {"P": 0.5, "q": 1.0}
AXES = [{"name": "b", "min": 2.0, "max": 3.2},
        {"name": "S_x", "min": 0.0, "max": 0.3}]
LEVEL = 3


@pytest.fixture(scope="module")
def prob():
    return s3.make_problem(Na=16, Nb=12, Nphi=6)


@pytest.fixture(scope="module")
def holdout():
    return p3d.holdout_points_nd([dict(a, Q=8) for a in AXES], n_points=5)


@pytest.fixture(scope="module")
def sm(prob):
    """Committed value-only Smolyak (the reduce-to-committed oracle)."""
    return smol.from_problem_smolyak_3d(prob, AXES, M_tot=M_TOT, fixed=FIXED,
                                        solver="modified").build_isotropic(
        LEVEL, tol=1e-12, max_iter=40)


@pytest.fixture(scope="module")
def he0(prob):
    """Hermite-Smolyak with NO enhanced axis (the value-only limit)."""
    return hsm.from_problem_hermite_smolyak_3d(
        prob, AXES, enhanced=(), M_tot=M_TOT, fixed=FIXED, solver="modified"
    ).build_isotropic(LEVEL, tol=1e-12, max_iter=40)


@pytest.fixture(scope="module")
def he(prob):
    """Hermite-Smolyak enhanced on the hard axis ``b`` (NK solve + NK tangent)."""
    return hsm.from_problem_hermite_smolyak_3d(
        prob, AXES, enhanced=["b"], M_tot=M_TOT, fixed=FIXED, solver="nk",
        tangent_jac="nk").build_isotropic(LEVEL, tol=1e-12, max_iter=20)


# ==========================================================================
# H5b-U1 — reduce-to-committed: enhanced=() is bit-for-bit SmolyakSolutionND
# ==========================================================================
def test_reduces_to_smolyak_bitforbit(sm, he0, holdout):
    assert he0.n_solver_nodes == sm.n_solver_nodes
    assert he0.enhanced == ()
    for th in holdout:
        diff = np.max(np.abs(he0.evaluate(th) - sm.evaluate(th)))
        assert diff == 0.0, (th, diff)               # exact same float ops
    # the jax twin agrees too (off-node)
    for th in holdout:
        dj = np.max(np.abs(np.asarray(he0.evaluate_jax(th)) - np.asarray(sm.evaluate_jax(th))))
        assert dj < 1e-13, (th, dj)


# ==========================================================================
# H5b-U2 — node-exact for VALUES (every sparse-grid node, any level)
# ==========================================================================
def test_value_node_exact(he):
    pool = he._dedup_pool()
    worst = 0.0
    for key, (theta, U, dU, it, rs) in pool.items():
        worst = max(worst, float(np.max(np.abs(he.evaluate(theta) - U))))
    assert worst < 1e-13, worst


# ==========================================================================
# H5b-U3 — node-exact for the ENHANCED-AXIS TANGENT at b-resolved nodes
# ==========================================================================
def test_enhanced_tangent_node_reproduction(he):
    """At nodes where ``b`` is Hermite-resolved (level ≥1) the combination's
    b-derivative reproduces the stored tangent — central FD converges to it ∝h²
    (node-exact, not interp-floored).  Use the max-b subgrid ``(LEVEL, 0)``: its b
    nodes are all level-LEVEL, at the single S_x midpoint."""
    b_nodes, _ = nested_levels(AXES[0]["min"], AXES[0]["max"], LEVEL)   # level-LEVEL b CGL
    sx_mid, _ = nested_levels(AXES[1]["min"], AXES[1]["max"], 0)        # S_x level-0 midpoint
    pool = he._dedup_pool()
    tested = 0
    for b in b_nodes:
        if abs(b - AXES[0]["min"]) < 1e-9 or abs(b - AXES[0]["max"]) < 1e-9:
            continue                                   # skip box edges (one-sided FD)
        theta = np.array([b, sx_mid[0]])
        key = smol._node_key(theta)
        assert key in pool, theta
        dUdb = pool[key][2][0]                         # stored dU/db at the node (dU stack, axis b)
        rels = []
        for h in (1e-4, 5e-5):
            up = he.evaluate(np.array([b + h, sx_mid[0]]))
            dn = he.evaluate(np.array([b - h, sx_mid[0]]))
            fd = (up - dn) / (2 * h)
            rels.append(np.max(np.abs(fd - dUdb)) / max(np.max(np.abs(dUdb)), 1e-30))
        assert rels[0] < 1e-6, (b, rels)               # matches stored tangent (FD floor)
        assert rels[1] < 0.45 * rels[0] + 1e-12, (b, rels)  # ∝h² → node-exact, not floored
        tested += 1
    assert tested >= 2


def test_value_only_level0_axis_is_interpolated(he):
    """The level-0 (value-only) node of the enhanced axis is NOT tangent-exact —
    the level-0 decision (R7/R4): the enhanced axis is value-only at level 0, so a
    node living only on that factor has an interpolated (not stored) b-derivative.
    (Documents the design; the value there is still node-exact.)"""
    # a node with S_x at the finest level (only reachable via the (0, LEVEL) subgrid,
    # where b is level 0) — value node-exact but b-tangent interpolated
    sx_fine, _ = nested_levels(AXES[1]["min"], AXES[1]["max"], LEVEL)
    b_mid, _ = nested_levels(AXES[0]["min"], AXES[0]["max"], 0)
    pool = he._dedup_pool()
    # find an S_x fine node that is NOT also a coarse (level≤LEVEL-1) node
    sx_coarse = set(np.round(nested_levels(AXES[1]["min"], AXES[1]["max"], LEVEL - 1)[0], 12))
    cand = [s for s in sx_fine if round(s, 12) not in sx_coarse]
    theta = np.array([b_mid[0], cand[0]])
    key = smol._node_key(theta)
    assert key in pool
    _theta, U, dU, _it, _rs = pool[key]
    dUdb = dU[0]                                       # stored dU/db at the node
    # value is node-exact even here
    assert np.max(np.abs(he.evaluate(theta) - U)) < 1e-13
    # b-tangent is interpolated (materially off the stored tangent) — by design
    fd = (he.evaluate(np.array([theta[0] + 1e-4, theta[1]]))
          - he.evaluate(np.array([theta[0] - 1e-4, theta[1]]))) / 2e-4
    rel = np.max(np.abs(fd - dUdb)) / max(np.max(np.abs(dUdb)), 1e-30)
    assert rel > 1e-5, rel                             # NOT node-exact (value-only level 0)


# ==========================================================================
# H5b-T1 — certified polish (unchanged guarantee, NK solver)
# ==========================================================================
def test_certified_polish(he, holdout):
    worst = 0.0
    for th in holdout[:4]:
        U, info = he.evaluate_polished(th, newton_steps=2, tol=1e-10)
        worst = max(worst, info.residual_norm)
    assert worst <= 1e-10, worst


# ==========================================================================
# H5b-T2 — save / load round-trips bit-for-bit (value + tangent + enhanced set)
# ==========================================================================
def test_save_load_roundtrip(he, holdout):
    with tempfile.TemporaryDirectory() as td:
        path = he.save(os.path.join(td, "he.npz"), meta={"note": "h5b test"})
        he2 = hsm.load_hermite_smolyak(path)
        assert he2.enhanced == he.enhanced
        assert he2.n_solver_nodes == he.n_solver_nodes
        for th in holdout:
            assert np.max(np.abs(he.evaluate(th) - he2.evaluate(th))) == 0.0, th
        # the tangent stack survived (a reloaded subgrid's dU matches)
        p1, p2 = he._dedup_pool(), he2._dedup_pool()
        assert set(p1) == set(p2)
        for k in p1:
            assert np.max(np.abs(p1[k][2] - p2[k][2])) < 1e-15   # dU stack
        assert he2.meta["note"] == "h5b test"


# ==========================================================================
# H5b-U4 — anisotropic build certifies (spends levels on the hard axis)
# ==========================================================================
def test_anisotropic_build(prob):
    her = hsm.from_problem_hermite_smolyak_3d(
        prob, AXES, enhanced=["b"], M_tot=M_TOT, fixed=FIXED, solver="nk",
        tangent_jac="nk").build_anisotropic(3.0, weights=[1.0, 2.0], tol=1e-12, max_iter=20)
    assert her.n_solver_nodes > 0 and her.enhanced == (0,)
    th = p3d.holdout_points_nd([dict(a, Q=8) for a in AXES], n_points=1)[0]
    U, info = her.evaluate_polished(th, newton_steps=3, tol=1e-10)
    assert info.residual_norm <= 1e-10


# ==========================================================================
# H5b-U5 — wiring validates the enhanced-axis names
# ==========================================================================
def test_enhanced_axis_validation(prob):
    with pytest.raises(ValueError):
        hsm.from_problem_hermite_smolyak_3d(prob, AXES, enhanced=["not_an_axis"],
                                            M_tot=M_TOT, fixed=FIXED)


# ==========================================================================
# H5b-T3 — add-only / standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lm.initial_data.parametric.hermite_smolyak as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm", "import context", "torch"):
        assert forbidden not in src, forbidden
