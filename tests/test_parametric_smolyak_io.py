"""LM-initial-data — persistence (save/load) layer for the certified parametric surrogates.

A built surrogate becomes a reusable on-disk artifact ("we provide the solution,
no need to solve").  These gates cover BOTH the sparse Smolyak sparse-grid model
(``SmolyakSolutionND.save`` / ``load_smolyak``) and the dense tensor model
(``ParametricSolutionND.save`` / ``load_parametric``):

  * round-trip bit-for-bit — a reloaded model's ``evaluate`` matches the original
    to machine zero at random θ; structural fields preserved;
  * sparse dedup — the stored node count == ``n_solver_nodes`` (the deduplicated
    pool, well below the as-stored-per-subgrid tensors);
  * metadata — ``kind``/``axis_names``/``box``/resolution/``format_version`` round-trip;
  * no-solver load — ``evaluate`` works; ``evaluate_polished`` raises until a
    solver is attached (``attach_solve_fn_3d`` reaches certified ‖R‖∞≤1e-10, slow);
  * robustness — a truncated/garbage file, or a kind/format-version mismatch, is a
    clear error, never a silent mis-read.

Numpy-only serialization (no pickle, no new deps).  Fast gates use a synthetic
``solve_fn`` (a smooth analytic field), exactly like ``test_parametric_smolyak``.
"""

import os

import numpy as np
import pytest

from lm.initial_data.parametric import parametric_nd_smolyak as sm
from lm.initial_data.parametric.parametric_nd import (
    ParametricSolverND, ParametricSolutionND, load_parametric, attach_solve_fn_3d,
)


# --------------------------------------------------------------------------
# Synthetic solver — a smooth analytic field (no PDE solve for the unit gates)
# --------------------------------------------------------------------------
class _FakeInfo:
    def __init__(self, iters, residual_norm):
        self.iters = iters
        self.residual_norm = residual_norm


def _make_solve(f):
    def solve_fn(theta, guess, tol, max_iter):
        return f(np.asarray(theta, float)), _FakeInfo(2, 1e-15)
    return solve_fn


_BOX4 = [(-1.0, 1.0), (0.0, 2.0), (1.0, 3.0), (0.5, 1.5)]


def _vec_field(d):
    """A smooth VECTOR field (non-trivial field_shape → exercises N×*field_shape)."""
    coef = np.array([0.7, 0.3, 0.5, 0.4][:d])

    def f(theta):
        t = np.asarray(theta, float)
        s = np.sum(np.sin(coef * t)) + 0.1 * np.prod(t)
        return np.array([s, np.cos(s), 0.25 * s * s])   # field_shape = (3,)
    return f


def _rand_theta(box, n, seed):
    rng = np.random.default_rng(seed)
    return [np.array([lo + (hi - lo) * rng.random() for (lo, hi) in box]) for _ in range(n)]


# ==========================================================================
# Sparse round-trip — bit-for-bit evaluate + structural fields preserved
# ==========================================================================
def test_smolyak_roundtrip_bitforbit(tmp_path):
    d = 3
    sf = _make_solve(_vec_field(d))
    orig = sm.SmolyakSolverND(sf, _BOX4[:d]).build_isotropic(3)
    p = orig.save(tmp_path / "sparse", meta={"axis_names": ["b", "theta_S", "q"]})
    assert os.path.exists(p)

    loaded = sm.load_smolyak(p)
    # structural fields
    assert sorted(map(tuple, loaded.index_set)) == sorted(map(tuple, orig.index_set))
    assert loaded.coeffs == orig.coeffs
    assert loaded.n_solver_nodes == orig.n_solver_nodes
    assert loaded.field_shape == orig.field_shape
    assert loaded._solve_fn is None
    # bit-for-bit evaluate at ~20 random off-node θ
    worst = 0.0
    for th in _rand_theta(_BOX4[:d], 20, seed=11):
        worst = max(worst, float(np.max(np.abs(loaded.evaluate(th) - orig.evaluate(th)))))
    assert worst == 0.0, f"sparse round-trip not bit-for-bit: {worst:.2e}"
    # jax twin also matches the original
    th = _rand_theta(_BOX4[:d], 1, seed=99)[0]
    assert float(np.max(np.abs(np.asarray(loaded.evaluate_jax(th))
                               - np.asarray(orig.evaluate_jax(th))))) < 1e-13


# ==========================================================================
# Sparse dedup — stored rows == n_solver_nodes; file << as-stored-per-subgrid
# ==========================================================================
def test_smolyak_dedup(tmp_path):
    d = 4

    # a field big enough that per-node field bytes dominate the file (so the
    # dedup saving is visible on disk, not swamped by θ/index arrays + zip overhead)
    def big_field(theta):
        t = np.asarray(theta, float)
        return np.outer(np.sin(t), np.cos(t)).reshape(-1)   # field_shape = (d*d,) = (16,)

    orig = sm.SmolyakSolverND(_make_solve(big_field), _BOX4[:d]).build_isotropic(3)
    p = orig.save(tmp_path / "sparse_d4")

    data = np.load(p, allow_pickle=False)
    n_stored = data["node_U"].shape[0]
    assert n_stored == orig.n_solver_nodes, (
        f"stored {n_stored} node fields != n_solver_nodes {orig.n_solver_nodes}")
    assert data["node_thetas"].shape == (orig.n_solver_nodes, d)

    # the file stores the DEDUPLICATED pool (n_solver_nodes fields), not the sum
    # over the overlapping subgrids (Σ_i ∏_k m(l_k) fields — much larger).
    field_bytes = int(np.prod(orig.field_shape)) * 8
    as_stored_per_subgrid = sum(int(np.prod([len(n) for n in sub.nodes]))
                                for sub in orig.subgrids)
    assert orig.n_solver_nodes < as_stored_per_subgrid, (
        "dedup is vacuous here — pick a level with subgrid overlap")
    on_disk = os.path.getsize(p)
    assert on_disk < as_stored_per_subgrid * field_bytes, (
        f"file {on_disk}B not below as-stored-per-subgrid "
        f"{as_stored_per_subgrid}×{field_bytes}B — dedup not effective")
    # and it is within a small factor of the deduplicated field payload itself
    assert on_disk < 2.0 * orig.n_solver_nodes * field_bytes + 32768


# ==========================================================================
# Dense round-trip — bit-for-bit evaluate + nodes/weights/axes preserved
# ==========================================================================
def test_dense_roundtrip_bitforbit(tmp_path):
    d = 3
    sf = _make_solve(_vec_field(d))
    axes = [(lo, hi, 5) for (lo, hi) in _BOX4[:d]]
    orig = ParametricSolverND(sf, axes).build()
    p = orig.save(tmp_path / "dense", meta={"axis_names": ["b", "theta_S", "q"]})
    assert os.path.exists(p)

    loaded = load_parametric(p)
    assert loaded.axes == orig.axes
    assert loaded.field_shape == orig.field_shape
    assert loaded._solve_fn is None
    for k in range(d):
        assert np.array_equal(loaded.nodes[k], orig.nodes[k])
        assert np.array_equal(loaded.weights[k], orig.weights[k])
    assert np.array_equal(loaded.U_nodes, orig.U_nodes)
    assert np.array_equal(loaded.iters, orig.iters)
    assert np.array_equal(loaded.residuals, orig.residuals)
    worst = 0.0
    for th in _rand_theta(_BOX4[:d], 20, seed=7):
        worst = max(worst, float(np.max(np.abs(loaded.evaluate(th) - orig.evaluate(th)))))
    assert worst == 0.0, f"dense round-trip not bit-for-bit: {worst:.2e}"


# ==========================================================================
# Metadata round-trip for both kinds
# ==========================================================================
def test_metadata_roundtrip(tmp_path):
    d = 3
    sf = _make_solve(_vec_field(d))
    meta_common = dict(axis_names=["b", "theta_S", "q"],
                       box=[[1.5, 4.0], [0.0, 90.0], [1.0, 3.0]],
                       Na=44, Nb=32, Nphi=8, solver="nk", tol=1e-12, note="prod")

    sp = sm.SmolyakSolverND(sf, _BOX4[:d]).build_isotropic(2)
    ps = sp.save(tmp_path / "s", meta=dict(meta_common, level=2))
    ms = sm.load_smolyak(ps).meta
    assert ms["kind"] == "smolyak" and ms["format_version"] == 1
    assert ms["axis_names"] == meta_common["axis_names"]
    assert ms["box"] == meta_common["box"] and ms["level"] == 2
    assert (ms["Na"], ms["Nb"], ms["Nphi"], ms["solver"]) == (44, 32, 8, "nk")

    dn = ParametricSolverND(sf, [(lo, hi, 4) for (lo, hi) in _BOX4[:d]]).build()
    pd = dn.save(tmp_path / "d", meta=dict(meta_common, Q=[4, 4, 4]))
    md = load_parametric(pd).meta
    assert md["kind"] == "dense" and md["format_version"] == 1
    assert md["axis_names"] == meta_common["axis_names"] and md["Q"] == [4, 4, 4]


# ==========================================================================
# No-solver load — evaluate works; evaluate_polished raises until attached
# ==========================================================================
def test_no_solver_evaluate_polished_raises(tmp_path):
    d = 2
    sf = _make_solve(_vec_field(d))
    sp = sm.SmolyakSolverND(sf, _BOX4[:d]).build_isotropic(2)
    dn = ParametricSolverND(sf, [(lo, hi, 4) for (lo, hi) in _BOX4[:d]]).build()
    ps, pd = sp.save(tmp_path / "s"), dn.save(tmp_path / "d")
    ls, ld = sm.load_smolyak(ps), load_parametric(pd)

    th = _rand_theta(_BOX4[:d], 1, seed=3)[0]
    assert np.all(np.isfinite(ls.evaluate(th)))
    assert np.all(np.isfinite(ld.evaluate(th)))
    with pytest.raises(RuntimeError):
        ls.evaluate_polished(th)
    with pytest.raises(RuntimeError):
        ld.evaluate_polished(th)


# ==========================================================================
# Robustness — garbage / truncated file, kind / format-version mismatch
# ==========================================================================
def test_robustness_bad_files(tmp_path):
    # not an npz at all
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"this is not an npz file at all")
    with pytest.raises(ValueError):
        sm.load_smolyak(junk)
    with pytest.raises(ValueError):
        load_parametric(junk)

    # a valid npz but missing the meta_json marker
    nometa = tmp_path / "nometa.npz"
    np.savez(nometa, foo=np.arange(3))
    with pytest.raises(ValueError):
        load_parametric(nometa)

    # truncated valid artifact
    sf = _make_solve(_vec_field(2))
    good = sm.SmolyakSolverND(sf, _BOX4[:2]).build_isotropic(2).save(tmp_path / "good")
    raw = open(good, "rb").read()
    trunc = tmp_path / "trunc.npz"
    trunc.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(Exception):
        sm.load_smolyak(trunc)


def test_kind_mismatch_reported(tmp_path):
    d = 3
    sf = _make_solve(_vec_field(d))
    sp = sm.SmolyakSolverND(sf, _BOX4[:d]).build_isotropic(2)
    dn = ParametricSolverND(sf, [(lo, hi, 4) for (lo, hi) in _BOX4[:d]]).build()
    ps, pd = sp.save(tmp_path / "s"), dn.save(tmp_path / "d")

    # loading a dense file as sparse (and vice versa) must report kind mismatch
    with pytest.raises(ValueError, match="kind mismatch"):
        sm.load_smolyak(pd)
    with pytest.raises(ValueError, match="kind mismatch"):
        load_parametric(ps)


# ==========================================================================
# Solver gate — attach a real 3-D solver, reach certified ‖R‖∞ ≤ 1e-10 (slow)
# ==========================================================================
@pytest.mark.slow
def test_attach_solve_fn_certifies_both(tmp_path):
    """After a no-solver load, ``attach_solve_fn_3d`` lets ``evaluate_polished``
    reach certified ``‖R‖∞ ≤ 1e-10`` at a generic off-node θ — for BOTH the sparse
    and dense reloaded models (a small real 3-D grid)."""
    from lm.initial_data.solver import solver_3d as s3
    from lm.initial_data.parametric import parametric_nd_3d as p3

    prob = s3.make_problem(Na=24, Nb=18, Nphi=6)
    box = [{"name": "b", "min": 1.5, "max": 4.0},
           {"name": "theta_S", "min": 0.0, "max": 90.0}]
    names = [a["name"] for a in box]

    sp = sm.from_problem_smolyak_3d(prob, box, solver="nk").build_isotropic(3, max_iter=20)
    dn = p3.from_problem_nd_3d(prob, [dict(a, Q=4) for a in box], solver="nk").build(
        tol=1e-12, max_iter=20)
    ps, pd = sp.save(tmp_path / "s"), dn.save(tmp_path / "d")

    hold = p3.holdout_points_nd([dict(a, Q=8) for a in box], n_points=3)
    p3.assert_off_node(hold, [dict(a, Q=8) for a in box])

    for loader, path in ((sm.load_smolyak, ps), (load_parametric, pd)):
        model = loader(path)
        with pytest.raises(RuntimeError):
            model.evaluate_polished(hold[0])          # no solver yet
        attach_solve_fn_3d(model, prob, names, solver="nk")
        worst = max(float(model.evaluate_polished(th, newton_steps=2, tol=1e-10)[1].residual_norm)
                    for th in hold)
        assert worst <= 1e-10, f"{loader.__name__}: certified ‖R‖ {worst:.2e} > 1e-10"
