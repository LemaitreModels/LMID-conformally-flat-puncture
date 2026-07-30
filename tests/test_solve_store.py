"""PARASOL — the persistent, content-addressed SOLVE STORE (``solve_store.py``).

The store turns the overlap between surrogate builds (nested CC levels,
dense↔sparse, re-runs, box extensions) into a shared, growing asset: a physical
slice is solved once and reused thereafter.  These gates cover the reuse
arithmetic and the keying (the correctness of "what counts as the same solve"):

Fast gates (a synthetic counting ``base_solve_fn`` — no PDE solve):
  * **nesting reuse (headline)** — build isotropic L=3 into a fresh store
    (misses==137), then L=4 against the same store (misses==264, hits==137, size==401);
  * **identity reuse** — rebuild the identical surrogate → misses==0 and the
    interpolant is bit-for-bit identical;
  * **key correctness / anti-staleness** — same active-θ but different ``fixed``
    (|S|), a different grid, or a different code_tag must NOT collide (no false hit);
  * **atomicity / robustness** — a truncated/garbage file is a miss (no crash),
    the atomic write leaves no partial ``.npz``, and the resid≤tol gate holds.

Slow gate (the real 3-D solver):
  * **solver-cross reuse** — a ``modified``-built store is reused by an ``nk``
    build (hits>0), and ``evaluate_polished`` certifies ‖R‖∞≤1e-10 on the
    modified-solved field (guards the "exclude solver type from the key" decision).
"""

import os

import numpy as np
import pytest

from lm.initial_data.parametric import solve_store as ss
from lm.initial_data.parametric.parametric_nd import ParametricSolverND
from lm.initial_data.parametric.parametric_nd_smolyak import SmolyakSolverND
from lm.initial_data.parametric.parametric_nd_2c import smolyak_points
from lm.initial_data.parametric.parametric_nd_3d import theta_to_slice3d


# --------------------------------------------------------------------------
# Synthetic counting solver — a smooth analytic field, no PDE solve
# --------------------------------------------------------------------------
class _FakeInfo:
    def __init__(self, iters, residual_norm):
        self.iters = iters
        self.residual_norm = residual_norm


def _make_counter_solve(f, resid=1e-15):
    """Return ``(solve_fn, counter)`` where ``counter['n']`` is the number of
    genuine base solves (store misses)."""
    counter = {"n": 0}

    def solve_fn(theta, guess, tol, max_iter):
        counter["n"] += 1
        return np.asarray(f(np.asarray(theta, float)), dtype=float), _FakeInfo(2, resid)

    return solve_fn, counter


def _smooth_field(d):
    coef = np.array([0.7, 0.3, 0.5, 0.4][:d])

    def f(theta):
        t = np.asarray(theta, float)
        return np.array(np.sum(np.sin(coef * t)) + 0.1 * np.prod(t))

    return f


# Axes whose θ→Slice3D map is INJECTIVE over the box (all linear / monotone):
# b, P, P_x, q → no physical-slice degeneracy (unlike polar spin at |S|=0), so the
# unique-node count equals the unique-physical-slice count and the reuse
# arithmetic is exact.
_BOX4 = [{"name": "b", "min": 1.5, "max": 4.0},
         {"name": "P", "min": 0.1, "max": 0.5},
         {"name": "P_x", "min": 0.05, "max": 0.3},
         {"name": "q", "min": 1.0, "max": 3.0}]
_ACTIVE4 = [a["name"] for a in _BOX4]
_SPEC4 = [(a["min"], a["max"]) for a in _BOX4]
_GRID = (16, 14, 4)


def _fresh_store(tmp_path, code_tag="testtag", grid=_GRID, sub="store"):
    return ss.SolveStore(os.path.join(str(tmp_path), sub), grid_meta=grid, code_tag=code_tag)


# ==========================================================================
# Nesting reuse (the headline)
# ==========================================================================
def test_nesting_reuse_headline(tmp_path):
    assert smolyak_points(4, 3) == 137 and smolyak_points(4, 4) == 401

    store = _fresh_store(tmp_path)
    # resid=1e-9 is the bug-triggering value: a loose modified-Newton monitor on a
    # deeply-converged field.  Under the old strict gate (resid ≤ request tol=1e-12)
    # this gave 0 hits / 401 misses; reuse_tol=1e-6 admits it → 137 hits.
    base, counter = _make_counter_solve(_smooth_field(4), resid=1e-9)
    wrapped = ss.wrap_solve_fn(base, store, _ACTIVE4)

    # L=3 into a fresh store: every node is a miss
    SmolyakSolverND(wrapped, _SPEC4).build_isotropic(3)
    assert store.n_misses == smolyak_points(4, 3) == 137
    assert store.n_hits == 0
    assert store.n_entries == 137
    assert counter["n"] == 137

    # L=4 against the SAME store: the 137 L=3 nodes are reused (nesting), 264 new
    store.reset_stats()
    counter["n"] = 0
    SmolyakSolverND(wrapped, _SPEC4).build_isotropic(4)
    assert store.n_hits == 137
    assert store.n_misses == 401 - 137 == 264
    assert counter["n"] == 264
    assert store.n_entries == 401


# ==========================================================================
# Identity reuse — rebuild the identical surrogate → 0 misses, bit-identical
# ==========================================================================
def test_identity_reuse_bit_for_bit(tmp_path):
    store = _fresh_store(tmp_path)
    base, counter = _make_counter_solve(_smooth_field(4))
    wrapped = ss.wrap_solve_fn(base, store, _ACTIVE4)

    sol1 = SmolyakSolverND(wrapped, _SPEC4).build_isotropic(3)
    n0 = store.n_entries

    store.reset_stats()
    counter["n"] = 0
    sol2 = SmolyakSolverND(wrapped, _SPEC4).build_isotropic(3)
    assert store.n_misses == 0
    assert counter["n"] == 0
    assert store.n_hits == 137
    assert store.n_entries == n0                    # no new files

    # the interpolant is bit-for-bit identical (store round-trips float64 exactly)
    rng = np.random.default_rng(7)
    for _ in range(6):
        th = np.array([a["min"] + (a["max"] - a["min"]) * rng.random() for a in _BOX4])
        assert np.array_equal(sol1.evaluate(th), sol2.evaluate(th))


# ==========================================================================
# Key correctness / anti-staleness — no false collisions
# ==========================================================================
def test_key_distinguishes_fixed_grid_codetag():
    th = np.array([2.0, 1.5])
    sl = theta_to_slice3d(th, ["b", "q"])
    k = ss.slice_key(sl, (16, 14, 4), "abc")

    # different grid → different key
    assert ss.slice_key(sl, (20, 16, 8), "abc") != k
    # different code_tag → different key
    assert ss.slice_key(sl, (16, 14, 4), "def") != k

    # different `fixed` inactive knob (|S|) → different physical slice → different key
    sl_hi = theta_to_slice3d(th, ["b", "q"], fixed={"S_mag": 0.3})
    sl_lo = theta_to_slice3d(th, ["b", "q"], fixed={"S_mag": 0.1})
    assert ss.slice_key(sl_hi, (16, 14, 4), "abc") != ss.slice_key(sl_lo, (16, 14, 4), "abc")


def test_no_false_hit_across_grid_and_codetag(tmp_path):
    sl = theta_to_slice3d(np.array([2.0, 1.5]), ["b", "q"])
    U = np.ones((5, 4, 3))
    # two stores sharing the SAME directory but a different grid / code_tag
    root = os.path.join(str(tmp_path), "shared")
    s_g1 = ss.SolveStore(root, grid_meta=(16, 14, 4), code_tag="t")
    s_g2 = ss.SolveStore(root, grid_meta=(20, 16, 8), code_tag="t")
    s_c2 = ss.SolveStore(root, grid_meta=(16, 14, 4), code_tag="OTHER")

    s_g1.put(sl, U, 3, 1e-14)
    assert s_g1.get(sl, 1e-12) is not None       # same grid+tag → hit
    assert s_g2.get(sl, 1e-12) is None           # different grid → miss (no stale reuse)
    assert s_c2.get(sl, 1e-12) is None           # different code_tag → miss


def test_no_false_hit_full_build_different_fixed(tmp_path):
    box = [{"name": "b", "min": 1.5, "max": 4.0},
           {"name": "q", "min": 1.0, "max": 3.0}]
    active = [a["name"] for a in box]
    spec = [(a["min"], a["max"]) for a in box]
    store = _fresh_store(tmp_path, sub="fixed")
    base, counter = _make_counter_solve(_smooth_field(2))

    n = smolyak_points(2, 2)
    # build at |S|=0.3
    w_hi = ss.wrap_solve_fn(base, store, active, fixed={"S_mag": 0.3})
    SmolyakSolverND(w_hi, spec).build_isotropic(2)
    assert store.n_entries == n

    # same active θ nodes, different |S| → all NEW physical slices, no false hits
    store.reset_stats()
    w_lo = ss.wrap_solve_fn(base, store, active, fixed={"S_mag": 0.1})
    SmolyakSolverND(w_lo, spec).build_isotropic(2)
    assert store.n_hits == 0
    assert store.n_misses == n
    assert store.n_entries == 2 * n


# ==========================================================================
# reuse_tol admission gate — decoupled from the caller's solve tol
# ==========================================================================
def test_reuse_tol_admission_regression(tmp_path):
    """The regression that would have caught the bug: admission is gated on the
    store-level ``reuse_tol``, NOT the caller's aspirational solve tol.  An entry
    with resid between the old build tol (1e-12) and reuse_tol (here 1e-9) MUST be
    reused; an entry worse than reuse_tol (1e-3) must NOT be."""
    store = _fresh_store(tmp_path, sub="reuse")
    assert store.reuse_tol == 1e-6                # default

    # a deeply-converged field whose loose monitor reads 1e-9 (the modified case)
    sl_ok = theta_to_slice3d(np.array([2.0, 1.5]), ["b", "q"])
    store.put(sl_ok, np.ones((3, 3, 3)), 4, 1e-9)
    # request tol 1e-12 (< stored resid) MUST still hit — tol no longer gates reuse
    got = store.get(sl_ok, 1e-12)
    assert got is not None and got[1] == 4 and got[2] == 1e-9
    assert store.get(sl_ok) is not None           # tol omitted → same admission
    assert store.get(sl_ok, 1e-15) is not None    # even a tighter request → hit

    # a genuinely failed/diverged solve (resid > reuse_tol) is stored but NOT reused
    sl_bad = theta_to_slice3d(np.array([3.0, 2.0]), ["b", "q"])
    store.put(sl_bad, np.ones((3, 3, 3)), 30, 1e-3)
    assert store.get(sl_bad, 1e-12) is None
    assert store.get(sl_bad, 1e-2) is None         # request tol does not admit it either

    # an explicit tighter reuse_tol re-tightens admission (1e-9 entry now rejected)
    store.reuse_tol = 1e-10
    assert store.get(sl_ok) is None


# ==========================================================================
# Atomicity / robustness
# ==========================================================================
def test_corrupt_file_is_a_miss_and_atomic_write(tmp_path):
    sl = theta_to_slice3d(np.array([2.5, 2.0]), ["b", "q"])
    store = _fresh_store(tmp_path, sub="robust")
    path = store._path(store.key(sl))

    # a truncated / garbage file at the expected path → miss, no crash
    with open(path, "wb") as fh:
        fh.write(b"\x93NUMPY not really an npz \x00\x01\x02")
    assert store.get(sl, 1e-12) is None

    # a subsequent atomic put overwrites it and reads back cleanly
    U = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    store.put(sl, U, 5, 1e-15)
    got = store.get(sl, 1e-12)
    assert got is not None
    assert np.array_equal(got[0], U) and got[1] == 5

    # the atomic write left no partial temp behind; exactly one addressed entry
    names = os.listdir(store.root_dir)
    assert not any(n.startswith("_tmp_") for n in names)
    assert store.n_entries == 1


# ==========================================================================
# Solver-cross reuse (slow, real solver) — the "exclude solver from key" gate
# ==========================================================================
@pytest.mark.slow
def test_solver_cross_reuse_certifies(tmp_path):
    from lm.initial_data.solver import solver_3d as s3
    from lm.initial_data.parametric import parametric_nd_3d as p3

    prob = s3.make_problem(Na=24, Nb=18, Nphi=6)
    box = [{"name": "b", "min": 1.5, "max": 4.0},
           {"name": "theta_S", "min": 0.0, "max": 90.0}]
    fixed = {"S_mag": 0.3}
    store = ss.SolveStore(os.path.join(str(tmp_path), "cross"),
                          grid_meta=(prob.Na, prob.Nb, prob.Nphi), code_tag="cross")
    assert store.reuse_tol == 1e-6

    # Build to the ASPIRATIONAL tol=1e-12 (the real production value).  The
    # modified solver's nodal-residual monitor only reaches ~5e-11, but its FIELD
    # is converged — reuse is gated on reuse_tol=1e-6, not the request tol, so the
    # nk build reuses these fields.  Under the old strict gate this was 0 hits.
    ss.from_problem_smolyak_3d_cached(prob, box, store=store, fixed=fixed,
                                      solver="modified").build_isotropic(3, tol=1e-12,
                                                                         max_iter=20)
    n_after_modified = store.n_entries
    assert n_after_modified > 0

    # build with the certified NK solver against the SAME store → the modified
    # fields are reused (bit-identical converged field; solver excluded from key)
    store.reset_stats()
    sol_nk = ss.from_problem_smolyak_3d_cached(prob, box, store=store, fixed=fixed,
                                               solver="nk").build_isotropic(3, tol=1e-12,
                                                                            max_iter=20)
    print(f"\n[solve-store cross] nk build: {store.n_hits} hits / {store.n_misses} misses")
    assert store.n_hits > 0                        # was 0 under the strict-gate bug
    assert store.n_entries == n_after_modified     # nk added nothing (all reused)

    # the modified-solved surrogate is still certifiable: evaluate_polished reaches
    # ‖R‖∞ ≤ 1e-10 at generic off-node θ in ≤2 NK steps
    hold = p3.holdout_points_nd([dict(a, Q=8) for a in box], n_points=3)
    p3.assert_off_node(hold, [dict(a, Q=8) for a in box])
    worst = 0.0
    for th in hold:
        _U, info = sol_nk.evaluate_polished(th, newton_steps=2, tol=1e-10)
        worst = max(worst, float(info.residual_norm))
    print(f"[solve-store cross] worst certified ‖R‖ = {worst:.2e}")
    assert worst <= 1e-10
