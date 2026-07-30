"""LM-initial-data — a persistent, content-addressed SOLVE STORE.

Every ``build_*`` sweep today re-solves its own fresh in-memory node pool.
Building the isotropic Smolyak levels ``L=2,3,4,5`` separately costs
``41+137+401+1105 = 1684`` solves even though the *union* of their nodes is only
``1105`` — the CC doubling levels are nested (level ``i ⊂ i+1``), the CGL nodes
double-nest, and dense↔sparse / adaptive / re-runs all revisit the same physical
slices.  A durable node store turns that overlap into a shared, growing asset:
once a physical slice is solved (on the laptop or the cluster), any later
surrogate build reuses it instead of re-solving.

This module is **ADD-ONLY**.  It reuses ``theta_to_slice3d`` /
``make_solve_fn`` / ``SmolyakSolverND`` / ``ParametricSolverND`` / ``_finalize``
verbatim; it does not touch the pool / finalize / build logic.  A
:class:`SolveStore` is a directory of one ``.npz`` per solved slice, keyed by the
SHA-1 of the *physical slice* (the rounded :class:`~lm.initial_data.solver.solver_3d.Slice3D`
fields) together with the frozen grid ``(Na,Nb,Nφ)`` and a ``code_tag`` (git
short hash).  :func:`wrap_solve_fn` slots the store between the parametric layer
and the raw ``solve_fn``: a store *hit* returns the cached field, a *miss* runs
the base solve and files the result.

Key design decisions (grounded in the committed facts):

  * **The key is the physical slice, not the active-θ vector.**  Two builds with
    different active axes / different ``fixed`` inactive knobs that nevertheless
    land on the SAME physical slice share the solve; two builds at the same θ but
    a different ``fixed`` (e.g. ``|S|=0.3`` vs ``0.1``) do NOT collide.
  * **Solver type and tol are EXCLUDED from the key.**  The converged field is
    bit-identical between ``solver="modified"`` and ``solver="nk"`` (the NK report
    — NK *reproduces*, never improves, the modified field), so a ``modified``-built
    store is reused by an ``nk`` build (and stays certifiable).  ``tol`` is not in
    the key either.
  * **Reuse admission is gated on a store-level ``reuse_tol``, NOT the caller's
    solve tol.**  The stored field IS the converged solution; the residual gate
    exists only to reject a *failed/diverged* solve, never to enforce the caller's
    aspirational tolerance.  Coupling admission to the request tol defeated the
    store's entire purpose: modified-Newton's ``info.residual_norm`` is a loose
    *monitor* (~1e-9) even when its field is bit-identical to NK's (the loose
    number is a monitor artifact, not field error), while builds request
    ``tol=1e-12`` — so ``1e-9 ≤ 1e-12`` is false and *every* lookup missed (an
    8-D spin8 L=4 build reported 0 hits / 3937 misses).  :meth:`SolveStore.get`
    now admits a cached entry iff ``stored_resid ≤ self.reuse_tol`` (default
    ``1e-6``), independent of any per-request ``tol`` — admitting the deeply
    converged modified/NK fields while still rejecting genuine failures.
  * **The grid and code_tag ARE in the key.**  ``U`` depends on the full physical
    slice and the frozen ``(Na,Nb,Nφ)`` grid, so a different grid or a different
    code revision is a genuine miss, never a stale reuse.
  * **Atomic, lock-free writes.**  ``put`` writes to a temp file in the same
    directory and ``os.replace``s it into place, so concurrent cluster workers are
    safe: a duplicate compute is last-write-wins on a bit-identical field
    (harmless).  Reads tolerate missing/corrupt files (treated as a miss, never a
    crash), so a half-written file from a killed worker cannot poison a build.

Numpy-only serialization (no pickle, no new deps).  Each file is
self-describing / auditable: it stores ``U`` plus a JSON metadata blob (the slice
fields, grid, residual, iters, code_tag).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..solver.solver_3d import Slice3D
from .parametric_nd_3d import theta_to_slice3d, make_solve_fn
from .parametric_nd import _git_commit


# 40-hex-char content-addressed filenames; the temp files never match this.
_KEY_RE = re.compile(r"^[0-9a-f]{40}\.npz$")


# --------------------------------------------------------------------------
# Cached-info shim — the (U, info) contract a store hit must honour
# --------------------------------------------------------------------------
class _CachedInfo:
    """Minimal ``NewtonInfo`` stand-in for a store hit.

    ``_solve_pool`` / ``ParametricSolverND.build`` read only ``info.iters`` and
    ``info.residual_norm`` off the returned info object, so those are the only
    fields a cached solve needs to expose.  ``converged`` is provided too (the
    stored residual already passed the ``≤ tol`` gate in :meth:`SolveStore.get`).
    """

    __slots__ = ("iters", "residual_norm", "converged", "cached")

    def __init__(self, iters: int, residual_norm: float):
        self.iters = int(iters)
        self.residual_norm = float(residual_norm)
        self.converged = True
        self.cached = True


# --------------------------------------------------------------------------
# Canonical slice key
# --------------------------------------------------------------------------
def _round12(x) -> float:
    """Round to 12 dp and normalise ``-0.0`` → ``0.0`` (matches ``_node_key``)."""
    return round(float(x), 12) + 0.0


def slice_payload(sl: Slice3D, grid_meta: Sequence[int], code_tag: str) -> dict:
    """The canonical (JSON-serialisable) content of the key + audit record.

    The 15 physical scalars of the slice (b, m_A, m_B, and the four 3-vectors),
    each rounded to 12 dp, plus the frozen grid ``(Na,Nb,Nφ)`` and the
    ``code_tag``.  This same dict is both hashed into the key and stored (as
    ``meta_json``) so a file is self-describing.
    """
    fields = [_round12(sl.b), _round12(sl.m_A), _round12(sl.m_B)]
    for vec in (sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec):
        fields.extend(_round12(c) for c in vec)
    return {
        "fields": fields,
        "grid": [int(g) for g in grid_meta],
        "code_tag": str(code_tag),
    }


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def slice_key(sl: Slice3D, grid_meta: Sequence[int], code_tag: str) -> str:
    """SHA-1 of the canonical bytes of ``slice_payload`` (a 40-hex-char string)."""
    return hashlib.sha1(_canonical_bytes(slice_payload(sl, grid_meta, code_tag))).hexdigest()


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------
class SolveStore:
    """A content-addressed directory of per-solve ``.npz`` files.

    Parameters
    ----------
    root_dir
        Directory holding the per-slice files (created if absent).
    grid_meta
        ``(Na, Nb, Nφ)`` of the frozen grid — part of the key (``U`` depends on it).
    code_tag
        Provenance tag folded into the key; defaults to the git short hash
        (``"unknown"`` if git is unavailable).  A different tag ⇒ a genuine miss.
    """

    def __init__(self, root_dir, grid_meta: Sequence[int],
                 code_tag: Optional[str] = None, reuse_tol: float = 1e-6):
        self.root_dir = str(root_dir)
        self.grid_meta = tuple(int(g) for g in grid_meta)
        self.code_tag = _git_commit() if code_tag is None else str(code_tag)
        # Reuse-admission threshold, DECOUPLED from any caller solve tol: a stored
        # entry is reusable iff its achieved residual ≤ reuse_tol.  Loose enough to
        # admit deeply-converged modified/NK fields (~1e-9, a monitor artifact) but
        # tight enough to reject a failed/diverged solve.
        self.reuse_tol = float(reuse_tol)
        os.makedirs(self.root_dir, exist_ok=True)
        self.n_hits = 0
        self.n_misses = 0

    # ----- keys / paths -----
    def key(self, sl: Slice3D) -> str:
        return slice_key(sl, self.grid_meta, self.code_tag)

    def _path(self, key: str) -> str:
        return os.path.join(self.root_dir, key + ".npz")

    # ----- stats -----
    def reset_stats(self) -> None:
        self.n_hits = 0
        self.n_misses = 0

    @property
    def n_entries(self) -> int:
        """Number of solved slices on disk (content-addressed files only)."""
        try:
            names = os.listdir(self.root_dir)
        except OSError:
            return 0
        return sum(1 for n in names if _KEY_RE.match(n))

    def __len__(self) -> int:
        return self.n_entries

    # ----- read -----
    def get(self, sl: Slice3D, tol: float = None) -> Optional[Tuple[np.ndarray, int, float]]:
        """Return ``(U, iters, resid)`` for ``sl`` iff a stored, adequately-converged
        solve exists; otherwise ``None``.

        Admission is gated on ``stored_resid ≤ self.reuse_tol`` — the store-level
        threshold — and is **independent of the per-request ``tol``**.  The ``tol``
        argument is retained only for signature compatibility (callers pass their
        aspirational solve tol); it does NOT control reuse, and there is
        deliberately no ``max(tol, reuse_tol)`` fallback (that would reintroduce
        the strict-gate bug that defeated cross-build reuse).  The stored field is
        the converged solution; ``reuse_tol`` only screens out failed/diverged
        solves.

        A missing key, an unreadable/corrupt/half-written file, or a stored
        residual worse than ``reuse_tol`` all count as a miss (never raises).
        """
        path = self._path(self.key(sl))
        if not os.path.exists(path):
            self.n_misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                U = np.asarray(data["U"], dtype=float)
                resid = float(np.asarray(data["resid"]))
                iters = int(np.asarray(data["iters"]))
        except Exception:
            # truncated / garbage / concurrently-being-written file → treat as miss
            self.n_misses += 1
            return None
        if resid <= self.reuse_tol:
            self.n_hits += 1
            return U, iters, resid
        self.n_misses += 1
        return None

    # ----- write (atomic) -----
    def put(self, sl: Slice3D, U, iters: int, resid: float,
            solver: Optional[str] = None) -> str:
        """Atomically file a solved slice.  Last-write-wins (bit-identical field).

        ``solver``/``iters`` are recorded in ``meta_json`` for auditing but are NOT
        in the key — the converged field is solver-independent (that cross-solver
        reuse is a feature).  ``resid`` is the achieved residual; an entry with
        ``resid > reuse_tol`` is stored but not reused (a later tighter solve can
        overwrite it).
        """
        key = self.key(sl)
        path = self._path(key)
        meta = slice_payload(sl, self.grid_meta, self.code_tag)
        meta["resid"] = float(resid)
        meta["iters"] = int(iters)
        meta["solver"] = None if solver is None else str(solver)
        # write to a unique temp in the SAME dir (so os.replace is atomic on-fs),
        # ending in .npz so np.savez does not append a second suffix.
        fd, tmp = tempfile.mkstemp(prefix="_tmp_", suffix=".npz", dir=self.root_dir)
        os.close(fd)
        try:
            np.savez(tmp,
                     U=np.asarray(U, dtype=float),
                     iters=np.asarray(int(iters), dtype=np.int64),
                     resid=np.asarray(float(resid), dtype=float),
                     meta_json=np.array(json.dumps(meta)))
            os.replace(tmp, path)
        except Exception:
            # never leave a partial temp behind on a failed write
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return path


# --------------------------------------------------------------------------
# solve_fn wrapper — slot the store between the parametric layer and the solver
# --------------------------------------------------------------------------
def wrap_solve_fn(base_solve_fn, store: SolveStore, active_names: Sequence[str],
                  M_tot: float = 1.0, fixed: Optional[Dict[str, float]] = None,
                  reuse_tol: Optional[float] = None, solver: Optional[str] = None):
    """Wrap ``base_solve_fn`` so solves route through ``store``.

    Same ``solve_fn(theta, guess, tol, max_iter) -> (U, info)`` contract.  On a
    store hit the cached ``(U, _CachedInfo)`` is returned (no solve); on a miss
    the base solver runs (keeping the warm-start ``guess``) and the result is
    filed.  The physical slice is reconstructed with the committed
    ``theta_to_slice3d`` (so the key is the physical slice, not θ).

    Reuse admission is governed by the store's ``reuse_tol`` (see
    :meth:`SolveStore.get`), NOT the per-request ``tol``.  Passing ``reuse_tol``
    here overrides ``store.reuse_tol`` for this store.  ``solver`` is recorded in
    each new entry's ``meta_json`` for auditing (not in the key).
    """
    active_names = list(active_names)
    if reuse_tol is not None:
        store.reuse_tol = float(reuse_tol)

    def solve_fn(theta, guess, tol, max_iter):
        sl = theta_to_slice3d(theta, active_names, M_tot, fixed)
        hit = store.get(sl, tol)
        if hit is not None:
            U, iters, resid = hit
            return U, _CachedInfo(iters, resid)
        U, info = base_solve_fn(theta, guess, tol, max_iter)
        store.put(sl, np.asarray(U), info.iters, info.residual_norm, solver=solver)
        return U, info

    return solve_fn


# --------------------------------------------------------------------------
# Store coercion helper
# --------------------------------------------------------------------------
def _as_store(store, prob, code_tag: Optional[str] = None,
              reuse_tol: Optional[float] = None) -> SolveStore:
    """Accept a :class:`SolveStore` or a root path; build one from ``prob``'s grid.

    For an existing store, ``reuse_tol`` (if given) overrides its threshold; for a
    freshly-built store it sets the threshold (default ``1e-6``).
    """
    if isinstance(store, SolveStore):
        if reuse_tol is not None:
            store.reuse_tol = float(reuse_tol)
        return store
    kw = {} if reuse_tol is None else {"reuse_tol": float(reuse_tol)}
    return SolveStore(store, grid_meta=(prob.Na, prob.Nb, prob.Nphi),
                      code_tag=code_tag, **kw)


# --------------------------------------------------------------------------
# Cached builders — mirror the committed from_problem_* wiring, + the store
# --------------------------------------------------------------------------
def from_problem_smolyak_3d_cached(prob, axes: Sequence[dict], *, store,
                                   M_tot: float = 1.0,
                                   fixed: Optional[Dict[str, float]] = None,
                                   use_cache: bool = True, solver: str = "nk",
                                   gmres_rtol: float = 1e-4, code_tag=None,
                                   reuse_tol: Optional[float] = None,
                                   retry_tol: Optional[float] = None):
    """``from_problem_smolyak_3d`` with ``solve_fn`` routed through a
    :class:`SolveStore`.

    Identical to the committed builder except the base ``solve_fn`` from
    ``make_solve_fn`` is wrapped by :func:`wrap_solve_fn`.  ``store`` may be a
    :class:`SolveStore` or a root path (a store is then built from ``prob``'s
    grid).  ``reuse_tol`` sets/overrides the store's reuse-admission threshold.
    Returns a ``SmolyakSolverND`` — call ``.build_isotropic`` /
    ``.build_adaptive`` / ... as usual.
    """
    from .parametric_nd_smolyak import SmolyakSolverND

    store = _as_store(store, prob, code_tag, reuse_tol)
    active_names = [a["name"] for a in axes]
    base_solve_fn, _ = make_solve_fn(prob, active_names, M_tot=M_tot, fixed=fixed,
                                     use_cache=use_cache, solver=solver,
                                     gmres_rtol=gmres_rtol, retry_tol=retry_tol)
    solve_fn = wrap_solve_fn(base_solve_fn, store, active_names, M_tot=M_tot,
                             fixed=fixed, solver=solver)
    spec = [(a["min"], a["max"]) for a in axes]
    return SmolyakSolverND(solve_fn, spec)


def from_problem_nd_3d_cached(prob, axes: Sequence[dict], *, store,
                              M_tot: float = 1.0,
                              fixed: Optional[Dict[str, float]] = None,
                              use_cache: bool = True, solver: str = "nk",
                              gmres_rtol: float = 1e-4, code_tag=None,
                              reuse_tol: Optional[float] = None,
                              retry_tol: Optional[float] = None):
    """``from_problem_nd_3d`` with ``solve_fn`` routed through a :class:`SolveStore`.

    Identical to the committed builder except the base ``solve_fn`` is wrapped by
    :func:`wrap_solve_fn`.  ``store`` may be a :class:`SolveStore` or a root path.
    ``reuse_tol`` sets/overrides the store's reuse-admission threshold.  Returns a
    ``ParametricSolverND`` — call ``.build`` as usual.
    """
    from .parametric_nd import ParametricSolverND

    store = _as_store(store, prob, code_tag, reuse_tol)
    active_names = [a["name"] for a in axes]
    base_solve_fn, _ = make_solve_fn(prob, active_names, M_tot=M_tot, fixed=fixed,
                                     use_cache=use_cache, solver=solver,
                                     gmres_rtol=gmres_rtol, retry_tol=retry_tol)
    solve_fn = wrap_solve_fn(base_solve_fn, store, active_names, M_tot=M_tot,
                             fixed=fixed, solver=solver)
    spec = [(a["min"], a["max"], a["Q"]) for a in axes]
    return ParametricSolverND(solve_fn, spec)
