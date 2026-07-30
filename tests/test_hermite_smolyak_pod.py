"""H5d acceptance — POD (reduced-basis) compression of the gradient-enhanced
SPARSE (Hermite-Smolyak) surrogate (GRADIENT_ENHANCED_PLAN.md §4 H5d, R5).

All gates on the **quasi-circular (QC)** family (the paper's astrophysical family;
head-on is out), built via the QC convenience wiring
``sensitivity_3d_qc.from_problem_hermite_smolyak_3d_qc`` (auto-plugs the QC
chain-rule tangent).  Gates:

  * **faithfulness** — ``pod.evaluate == mean + ΦΦᵀ(full_hermite_smolyak − mean)``
    to roundoff (the POD projection identity, in coeff space);
  * the reduced-basis Hermite-Smolyak **reproduces the full Hermite-Smolyak** at
    full rank (interp floor) and to the **truncation tail** at a gradient-safe rank;
  * the **exposed gradient is preserved to ~1e-6** (the full sparse gradient
    projected onto ``Φ``);
  * ``evaluate_polished`` still **certifies to ‖R‖∞ ≤ 1e-10** (unchanged);
  * ``save``/``load`` round-trips **bit-for-bit**;
  * the derivative fields' POD rank **barely grows** over the value-only basis (R5).

Standalone (numpy/scipy/jax); reuses the committed Smolyak/Hermite/POD layers and
the H5c QC tangent verbatim.
"""

import os
import tempfile

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from lemaitre.initial_data.solver import solver_3d as s3
from lemaitre.initial_data.parametric import parametric_nd_3d as p3d
from lemaitre.initial_data.applications import sensitivity_3d_qc as qc
from lemaitre.initial_data.parametric.hermite_smolyak_pod import (
    pod_basis_pool,
    build_pod_hermite_smolyak,
    project_hermite_smolyak_pod,
    PODHermiteSmolyak,
    load_pod_hermite_smolyak,
)
from lemaitre.initial_data.parametric.hermite_pod import rank_for_tail


M_TOT = 1.0
FIXED = {"qc": 1.0}
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
def model(prob):
    """The full QC Hermite-Smolyak (enhanced on the hard axis ``b``, NK solve + NK
    QC chain-rule tangent) — the corpus this milestone compresses."""
    return qc.from_problem_hermite_smolyak_3d_qc(
        prob, AXES, enhanced=["b"], M_tot=M_TOT, fixed=FIXED, solver="nk",
        tangent_jac="nk").build_isotropic(LEVEL, tol=1e-12, max_iter=20)


def _relL2(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# ==========================================================================
# H5d-U1 — POD faithfulness: PODHermiteSmolyak == POD projection of the full
# ==========================================================================
def test_pod_faithfulness_bitlevel(model, holdout):
    """``pod.evaluate(θ) == mean + ΦΦᵀ(full(θ) − mean)`` to roundoff — the linear
    POD projection commutes with the (linear) combination-technique Hermite
    interpolation *and* the combination reproduces constants (Σc_l=1)."""
    Phi, mean, diag = pod_basis_pool(model, tail=1e-6)
    pod = project_hermite_smolyak_pod(model, Phi, mean)
    worst = 0.0
    for th in holdout:
        full = np.asarray(model.evaluate(th)).reshape(-1)
        proj = mean + Phi @ (Phi.T @ (full - mean))
        got = np.asarray(pod.evaluate(th)).reshape(-1)
        worst = max(worst, np.max(np.abs(got - proj)))
    assert worst < 1e-9, worst


# ==========================================================================
# H5d-U2 — reproduces the full Hermite-Smolyak at full rank (interp floor)
# ==========================================================================
def test_pod_reproduces_full_at_full_rank(model, holdout):
    _, _, diag = pod_basis_pool(model, tail=1e-12)
    r_full = len(diag["s"])
    Phi, mean, _ = pod_basis_pool(model, r=r_full)
    pod = project_hermite_smolyak_pod(model, Phi, mean)
    worst = max(_relL2(pod.evaluate(th), model.evaluate(th)) for th in holdout)
    assert worst < 1e-8, worst


# ==========================================================================
# H5d-U3 — reproduces the full Hermite-Smolyak to the truncation tail
# ==========================================================================
def test_pod_reproduces_full_to_truncation_tail(model, holdout):
    """At a gradient-safe (tail=1e-8) truncated rank the reduced-basis
    Hermite-Smolyak reproduces the full sparse interpolant to ~the tail (the real
    deliverable, mirroring the solver-backed H3 gate)."""
    pod, diag = build_pod_hermite_smolyak(model, tail=1e-8, solve_fn=model._solve_fn)
    worst = max(_relL2(pod.evaluate(th), model.evaluate(th)) for th in holdout)
    assert worst < 1e-6, (worst, pod.r)


# ==========================================================================
# H5d-U4 — exposed parameter gradient preserved to ~1e-6
# ==========================================================================
def test_pod_gradient_preserved(model, holdout):
    """The exposed gradient of the reduced-basis model is the full sparse gradient
    projected onto ``Φ`` (``P_r·∂U/∂θ``): the projection identity holds to roundoff
    and the truncation loss is ≤ 1e-6 at a gradient-safe rank."""
    pod, diag = build_pod_hermite_smolyak(model, tail=1e-8, solve_fn=model._solve_fn)
    Phi = pod.Phi
    worst_match = worst_loss = 0.0
    for th in holdout:
        thj = jnp.asarray(th, dtype=jnp.float64)
        Jfull = np.asarray(jax.jacfwd(lambda t: model.evaluate_jax(t).reshape(-1))(thj))
        Jpod = np.asarray(jax.jacfwd(lambda t: pod.evaluate_jax(t).reshape(-1))(thj))
        Jproj = Phi @ (Phi.T @ Jfull)
        worst_match = max(worst_match, np.linalg.norm(Jpod - Jproj) / np.linalg.norm(Jfull))
        worst_loss = max(worst_loss, np.linalg.norm(Jfull - Jproj) / np.linalg.norm(Jfull))
    assert worst_match < 1e-9, worst_match
    assert worst_loss < 1e-6, worst_loss


# ==========================================================================
# H5d-U5 — the derivative fields share the value basis; rank barely grows (R5)
# ==========================================================================
def test_pod_rank_barely_grows(model):
    """The stacked value+derivative POD rank of the sparse pool **barely grows** over
    the value-only rank — the R5 storage mitigation.  Adding the enhanced-axis
    tangent ``dU/db`` (a second corpus of the SAME size as the value corpus) grows
    the rank by only ~30% (ratio ≈ 1.29 at the measured tails), NOT the ~100% a
    fully-independent corpus would give, and the value-only rank-``r`` basis already
    captures ``dU/db`` to ~6.5e-6 — the derivatives live in essentially the same
    low-rank spatial subspace as ``U``."""
    _, _, diag = pod_basis_pool(model, tail=1e-6)
    for tail in (1e-4, 1e-6):
        rv = diag["rank_value"][tail]
        rs = diag["rank_stacked"][tail]
        assert rs <= 1.5 * rv, (tail, rs, rv)         # ≪ the 2× of an independent corpus
    resid = diag["dU_on_value_basis_resid"]
    assert max(resid) < 2e-5, resid


# ==========================================================================
# H5d-T1 — certified polish (unchanged guarantee, NK solver)
# ==========================================================================
def test_certified_polish(model, holdout):
    pod, _ = build_pod_hermite_smolyak(model, tail=1e-8, solve_fn=model._solve_fn)
    worst = 0.0
    for th in holdout[:3]:
        U, info = pod.evaluate_polished(th, newton_steps=3, tol=1e-10)
        worst = max(worst, info.residual_norm)
    assert worst <= 1e-10, worst


# ==========================================================================
# H5d-T2 — save / load round-trips bit-for-bit
# ==========================================================================
def test_save_load_roundtrip(model, holdout):
    pod, _ = build_pod_hermite_smolyak(model, tail=1e-6, solve_fn=model._solve_fn)
    with tempfile.TemporaryDirectory() as td:
        path = pod.save(os.path.join(td, "h5d_pod.npz"),
                        meta={"axis_names": ["b", "S_x"], "note": "h5d test"})
        back = load_pod_hermite_smolyak(path)
        assert back.r == pod.r
        assert tuple(back.enhanced) == tuple(pod.enhanced)
        assert back.n_solver_nodes == pod.n_solver_nodes
        assert np.array_equal(back.Phi, pod.Phi)
        assert np.array_equal(back.mean, pod.mean)
        assert back.meta["kind"] == "pod_hermite_smolyak"
        assert back.meta["axis_names"] == ["b", "S_x"]
        for th in holdout:
            assert np.array_equal(np.asarray(back.evaluate(th)),
                                  np.asarray(pod.evaluate(th))), th
        # the tangent stack survived (a reloaded coeff subgrid's dU matches)
        p1, p2 = pod.coeff_model._dedup_pool(), back.coeff_model._dedup_pool()
        assert set(p1) == set(p2)
        for k in p1:
            assert np.max(np.abs(p1[k][2] - p2[k][2])) < 1e-15   # dU coeff stack


def test_load_rejects_wrong_kind(model):
    """A raw Hermite-Smolyak artifact must not load as a pod_hermite_smolyak."""
    with tempfile.TemporaryDirectory() as td:
        p = model.save(os.path.join(td, "plain_hsm.npz"))
        with pytest.raises(ValueError):
            load_pod_hermite_smolyak(p)


# ==========================================================================
# H5d-U6 — enhanced=() reduces to the value-only sparse POD re-encoding
# ==========================================================================
def test_value_only_pool_basis(model):
    """With ``include_derivatives=False`` the pool basis is the value-only POD; the
    reduced model then re-encodes only ``U`` (the value-only ``PODSmolyak`` analog),
    and the projection identity still holds."""
    Phi, mean, diag = pod_basis_pool(model, tail=1e-6, include_derivatives=False)
    pod = project_hermite_smolyak_pod(model, Phi, mean)
    th = p3d.holdout_points_nd([dict(a, Q=8) for a in AXES], n_points=1)[0]
    full = np.asarray(model.evaluate(th)).reshape(-1)
    proj = mean + Phi @ (Phi.T @ (full - mean))
    assert np.max(np.abs(np.asarray(pod.evaluate(th)).reshape(-1) - proj)) < 1e-9


# ==========================================================================
# H5d-T3 — add-only / standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lemaitre.initial_data.parametric.hermite_smolyak_pod as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm", "import context", "torch"):
        assert forbidden not in src, forbidden
