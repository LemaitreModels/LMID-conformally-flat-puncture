"""PARASOL — POD (reduced-basis) compression of the gradient-enhanced SPARSE
(Hermite-Smolyak) surrogate (H5d).

The sparse (Smolyak) sibling of the dense POD re-encoding
(:mod:`hermite_pod`, :class:`hermite_pod.PODHermiteND`), and the
gradient-enhanced sibling of the committed value-only sparse POD
(:class:`experiments.ml.pod_surrogate.PODSmolyak`).  It closes the R5 storage
cost of the gradient enhancement on the sparse path: the H5b/H5c Hermite-Smolyak
pool carries ``(1 + n_enhanced)×`` fields per node (the value ``U`` plus one
certified tangent ``dU/dθ_k`` per globally-enhanced axis), and H3 already
measured that those derivative fields live in **essentially the same low-rank
spatial basis as** ``U`` (the ``θ→field`` map is analytic, so its parameter
tangent lives in the solution-manifold tangent space).  So one POD basis ``Φ``,
built from the **stacked value+derivative corpus of the sparse pool**, compresses
both ``U`` and every ``dU/dθ_k`` by the same ``nfeat/r`` factor.

Construction (the direct analog of ``PODSmolyak`` / ``PODHermiteND``):

  1. **One basis from the pool.**  ``pod_basis_pool`` SVDs the stacked
     value+derivative corpus of the deduplicated node pool, **reusing H3's
     :func:`hermite_pod.pod_basis` verbatim** — it is handed a synthetic
     single-axis :class:`hermite_nd.HermiteSolutionND` whose "grid" is the pool
     and whose enhanced-axis derivative corpora are the pool's ``dU/dθ_k`` (so
     ``pod_basis``'s own ``_flatten_corpus`` builds exactly
     ``[ (U−mean)ᵀ | (dU/dθ_{e_1})ᵀ | … ]``).  The full diagnostics (value vs
     stacked rank tables, the "derivatives share the value basis" residual) come
     for free — the R5 measurement.
  2. **Project each subgrid to coeff space.**  Each Hermite subgrid is projected
     with **:func:`hermite_pod.project_hermite_pod` verbatim** (value coeff
     ``(U−mean)·Φ``, tangent coeff ``dU/dθ_k·Φ``), and its ``.coeff_hermite`` (an
     ``r``-dim :class:`hermite_nd.HermiteSolutionND`) becomes the coeff-space
     subgrid — exactly as ``PODSmolyak`` replaces each value subgrid with an
     ``r``-dim ``ParametricSolutionND``.
  3. **Combine + decode.**  The coeff subgrids are wrapped in the committed
     :class:`hermite_smolyak.HermiteSmolyakSolutionND` (its
     ``Σ_l c_l·sub_l.evaluate`` combination reused verbatim, now in coeff space),
     and ``evaluate`` decodes ``u = mean + Φ·c``.

By linearity of barycentric/Hermite interpolation *and* the combination-technique
property ``Σ_l c_l = 1`` (constants reproduced), ``evaluate`` is bit-identical (to
roundoff) to the POD projection ``mean + ΦΦᵀ(full_hermite_smolyak − mean)`` of the
full sparse Hermite interpolant, so the H5b node-exactness / reduce-to-committed
properties and the certified polish all carry over; the exposed parameter gradient
of the compressed model is the full sparse gradient projected onto ``Φ`` (preserved
to the truncation tail).

**Add-only.**  Reuses :mod:`hermite_pod` (``pod_basis``, ``project_hermite_pod``,
``rank_for_tail``, ``randomized_svd`` — verbatim), :mod:`hermite_smolyak`
(``HermiteSmolyakSolutionND``, ``HermiteSmolyakSolverND`` — the combination
container + the loader ``_finalize``, verbatim), :mod:`hermite_nd`
(``HermiteSolutionND``), and the ``parametric_nd`` persistence helpers verbatim;
never edits a committed module.  Certification is unchanged — the compressed object
is only a *guess*; ``evaluate_polished`` reuses the committed ``solve_fn`` →
``newton_solve``.

Standalone: numpy + jax + the sibling ``parametric`` modules.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric_nd import (              # persistence helpers, reused verbatim
    FORMAT_VERSION,
    _pack_meta,
    _unpack_meta,
    _git_commit,
    _load_npz,
    _check_meta,
)
from .parametric_nd_smolyak import _node_key           # node keying (verbatim)
from .hermite import cardinal_deriv_at_nodes           # node-set primitive (verbatim)
from .hermite_nd import HermiteSolutionND              # the H2 interpolant (verbatim)
from .hermite_smolyak import (                          # the H5b sparse layer (verbatim)
    HermiteSmolyakSolutionND,
    HermiteSmolyakSolverND,
)
from .hermite_pod import (                              # the H3 POD machinery (verbatim)
    pod_basis,
    project_hermite_pod,
    rank_for_tail,
    randomized_svd,
)


# --------------------------------------------------------------------------
# POD basis from the stacked value+derivative corpus of a SPARSE pool
# --------------------------------------------------------------------------
def _pool_corpus_hermite(model: HermiteSmolyakSolutionND) -> HermiteSolutionND:
    """A synthetic single-axis :class:`hermite_nd.HermiteSolutionND` over the
    deduplicated node pool of ``model``, so :func:`hermite_pod.pod_basis` (which
    reads ``.U_nodes``/``.dU_nodes``/``.enhanced``/``.field_shape``) builds the
    stacked value+derivative POD basis of the *sparse pool* directly.

    The fake grid has ``d=1`` with ``N`` pool nodes; ``dU_nodes`` carries ONLY the
    globally-enhanced axes' tangents stacked on axis 1, and ``enhanced`` is
    ``(0, …, n_enh−1)`` so ``pod_basis._flatten_corpus`` pulls exactly the
    ``dU/dθ_k`` corpora the interpolant consumes.  ``cvec`` is unused by
    ``pod_basis`` (no ``evaluate`` is called on the fake), so it is left zero.
    """
    pool = model._dedup_pool()                          # key -> (theta, U, dU, iters, resid)
    keys = list(pool)
    N = len(keys)
    fs = model.field_shape
    Us = np.stack([np.asarray(pool[k][1], dtype=float) for k in keys])      # (N, *fs)
    enh = tuple(int(e) for e in model.enhanced)
    if enh:
        dUs = np.stack([np.stack([np.asarray(pool[k][2][e], dtype=float) for e in enh],
                                 axis=0) for k in keys])                    # (N, n_enh, *fs)
    else:
        dUs = np.zeros((N, 0) + tuple(fs), dtype=float)
    fake_nodes = np.arange(N, dtype=float)
    return HermiteSolutionND(
        axes=[(0.0, 1.0, 0)], nodes=[fake_nodes], weights=[np.ones(N)],
        U_nodes=Us, dU_nodes=dUs, cvec=[np.zeros(N)],
        enhanced=tuple(range(len(enh))),
        iters=np.zeros(N, dtype=int), residuals=np.zeros(N))


def pod_basis_pool(model: HermiteSmolyakSolutionND, *, r: Optional[int] = None,
                   tail: Optional[float] = None, include_derivatives: bool = True,
                   randomized: bool = False, seed: int = 0):
    """POD spatial modes ``Φ`` from the stacked value+derivative corpus of the
    sparse pool of a :class:`hermite_smolyak.HermiteSmolyakSolutionND`.

    Thin wrapper that hands the pool to :func:`hermite_pod.pod_basis` (reused
    **verbatim**) via :func:`_pool_corpus_hermite`.  Returns ``(Phi, mean, diag)``
    with the identical ``diag`` structure — ``s``/``s_value`` singular values,
    ``rank_stacked``/``rank_value`` tables, and ``dU_on_value_basis_resid`` (the
    R5 "derivatives share the value basis" residual)."""
    fake = _pool_corpus_hermite(model)
    return pod_basis(fake, r=r, tail=tail, include_derivatives=include_derivatives,
                     randomized=randomized, seed=seed)


# --------------------------------------------------------------------------
# The POD (reduced-basis) gradient-enhanced Hermite-Smolyak surrogate
# --------------------------------------------------------------------------
class PODHermiteSmolyak:
    """Reduced-basis (POD) re-encoding of a
    :class:`hermite_smolyak.HermiteSmolyakSolutionND`.

    Interpolates the length-``r`` POD **coefficient** vectors with the identical
    H5b combination machinery — an internal
    :class:`hermite_smolyak.HermiteSmolyakSolutionND` over the coeff space (its
    subgrids are the coeff-space :class:`hermite_nd.HermiteSolutionND` from
    :func:`hermite_pod.project_hermite_pod`, reused verbatim) — and decodes
    ``u = mean + Φ·c``.  By linearity + ``Σ_l c_l = 1`` this is bit-identical (to
    roundoff) to the POD projection of the full sparse Hermite interpolant, so the
    node-exactness / reduce-to-committed properties and the certified polish carry
    over; the exposed parameter gradient is the full sparse gradient projected onto
    ``Φ`` (preserved to the truncation tail).
    """

    def __init__(self, coeff_model: HermiteSmolyakSolutionND, Phi, mean, field_shape,
                 _solve_fn: Optional[Callable] = None):
        self.coeff_model = coeff_model          # HermiteSmolyakSolutionND over r-dim coeffs
        self.Phi = np.asarray(Phi, dtype=float)              # (nfeat, r)
        self.mean = np.asarray(mean, dtype=float)            # (nfeat,)
        self.field_shape = tuple(int(x) for x in field_shape)
        self._Phi_j = jnp.asarray(self.Phi)
        self._mean_j = jnp.asarray(self.mean)
        self._solve_fn = _solve_fn

    @property
    def d(self) -> int:
        return self.coeff_model.d

    @property
    def r(self) -> int:
        return int(self.Phi.shape[1])

    @property
    def enhanced(self):
        return self.coeff_model.enhanced

    @property
    def n_solver_nodes(self) -> int:
        return self.coeff_model.n_solver_nodes

    @property
    def n_nodes(self) -> int:
        return self.coeff_model.n_solver_nodes

    # ----- decode: interpolate the coeffs, then u = mean + Φ·c -----
    def evaluate(self, theta):
        """``ũ(θ)`` decoded from the interpolated POD coefficients (numpy, node-safe)."""
        c = np.asarray(self.coeff_model.evaluate(theta)).reshape(-1)   # (r,)
        u = self.mean + self.Phi @ c
        return u.reshape(self.field_shape)

    def evaluate_jax(self, theta):
        """``jnp`` twin of :meth:`evaluate` — the exposed-gradient hook
        (``jax.jacfwd`` gives ``P_r·∂U/∂θ``, the full sparse gradient projected
        onto ``Φ``).  Must NOT be queried exactly at a node."""
        c = jnp.reshape(self.coeff_model.evaluate_jax(theta), (-1,))
        u = self._mean_j + self._Phi_j @ c
        return jnp.reshape(u, self.field_shape)

    # ----- certified evaluation (unchanged; decode → committed solve_fn) -----
    def evaluate_polished(self, theta, newton_steps: int = 2, tol: float = 1e-12):
        """POD-decoded sparse-Hermite guess + 1–2 Newton steps → certified
        ``‖R‖≤tol``.

        Certification is unchanged: the compressed object is only a *guess*; the
        attached ``solve_fn`` → ``newton_solve`` is the certificate."""
        if self._solve_fn is None:
            raise RuntimeError(
                "no solve_fn attached; build via build_pod_hermite_smolyak with a "
                "solver-backed HermiteSmolyakSolutionND, or reattach a solve_fn")
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(np.asarray(theta, dtype=float), guess, tol, newton_steps)
        return U, info

    # ----- persistence: store the DEDUPLICATED coeff pool (numpy-only .npz) -----
    def save(self, path, *, meta=None, coeff_dtype=np.float64, mode_dtype=np.float64):
        """Persist to a single ``.npz`` (numpy-only, no pickle).  Round-trips
        bit-for-bit via :func:`load_pod_hermite_smolyak`.

        Stores the **deduplicated** coeff node pool — ``node_thetas`` (N×d),
        ``node_U`` (N×r), ``node_dU`` (N×d×r), ``node_iters`` (N), ``node_resids``
        (N) — plus ``Phi`` (nfeat×r), ``mean`` (nfeat), ``index_set`` (M×d),
        ``axes`` (d×2), ``enhanced`` (global indices), ``field_shape``, ``r`` and
        ``meta_json``.  The combination coefficients are recomputed on load
        (:meth:`hermite_smolyak.HermiteSmolyakSolverND._finalize`).

        ``Φ`` and the coeff pool are only a *warm start* for the certified polish,
        so they need not be float64 (the manuscript O1 note): ``mode_dtype`` /
        ``coeff_dtype`` may be ``float32`` to halve the on-disk footprint (reload
        upcasts to float64, so ``evaluate`` still runs in float64).  Defaults
        preserve the float64 artifact byte-for-byte.  ``mean`` (nfeat, negligible)
        stays float64.
        """
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        cm = self.coeff_model
        pool = cm._dedup_pool()                  # key -> (theta, U(r), dU(d,r), iters, resid)
        keys = list(pool)
        node_thetas = np.array([pool[k][0] for k in keys], dtype=float)          # N×d
        node_U = np.array([pool[k][1] for k in keys]).astype(coeff_dtype)        # N×r
        node_dU = np.array([pool[k][2] for k in keys]).astype(coeff_dtype)       # N×d×r
        node_iters = np.array([pool[k][3] for k in keys], dtype=np.int64)        # N
        node_resids = np.array([pool[k][4] for k in keys], dtype=float)          # N
        index_set = np.array([[int(x) for x in l] for l in cm.index_set], dtype=np.int64)
        axes = np.array([[float(lo), float(hi)] for (lo, hi) in cm.axes], dtype=float)
        full_meta = {"d": int(self.d), "r": int(self.r),
                     "n_solver_nodes": int(cm.n_solver_nodes),
                     "git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION      # authoritative
        full_meta["kind"] = "pod_hermite_smolyak"
        np.savez(path,
                 Phi=self.Phi.astype(mode_dtype), mean=self.mean,
                 node_thetas=node_thetas, node_U=node_U, node_dU=node_dU,
                 node_iters=node_iters, node_resids=node_resids,
                 index_set=index_set, axes=axes,
                 enhanced=np.asarray(sorted(int(e) for e in cm.enhanced), dtype=np.int64),
                 field_shape=np.asarray(self.field_shape, dtype=np.int64),
                 r=np.asarray(self.r, dtype=np.int64),
                 meta_json=_pack_meta(full_meta))
        return path


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def project_hermite_smolyak_pod(model: HermiteSmolyakSolutionND, Phi: np.ndarray,
                                mean: np.ndarray, *,
                                solve_fn: Optional[Callable] = None) -> PODHermiteSmolyak:
    """Project a solver-backed
    :class:`hermite_smolyak.HermiteSmolyakSolutionND` onto the POD basis ``Φ``
    (``mean`` the value-mean) → a :class:`PODHermiteSmolyak`.

    Each Hermite subgrid is projected with :func:`hermite_pod.project_hermite_pod`
    (verbatim); its ``.coeff_hermite`` (the ``r``-dim
    :class:`hermite_nd.HermiteSolutionND`) becomes the coeff-space subgrid, and the
    coeff subgrids are wrapped in a coeff-space
    :class:`hermite_smolyak.HermiteSmolyakSolutionND` with the SAME index set /
    combination coefficients / enhanced set as ``model``."""
    coeff_subs = [project_hermite_pod(sub, Phi, mean).coeff_hermite
                  for sub in model.subgrids]
    coeff_model = HermiteSmolyakSolutionND(
        axes=list(model.axes), index_set=[tuple(l) for l in model.index_set],
        coeffs=list(model.coeffs), subgrids=coeff_subs, enhanced=tuple(model.enhanced),
        n_solver_nodes=model.n_solver_nodes, total_iters=model.total_iters,
        _solve_fn=None)
    sf = solve_fn if solve_fn is not None else getattr(model, "_solve_fn", None)
    return PODHermiteSmolyak(coeff_model, Phi, mean, model.field_shape, _solve_fn=sf)


def build_pod_hermite_smolyak(model: HermiteSmolyakSolutionND, *, r: Optional[int] = None,
                              tail: Optional[float] = None, include_derivatives: bool = True,
                              randomized: bool = False, seed: int = 0,
                              solve_fn: Optional[Callable] = None):
    """Convenience: :func:`pod_basis_pool` then :func:`project_hermite_smolyak_pod`.

    Returns ``(pod, diag)`` — the :class:`PODHermiteSmolyak` and the
    ``pod_basis_pool`` diagnostics (rank tables, share-the-basis residuals)."""
    Phi, mean, diag = pod_basis_pool(model, r=r, tail=tail,
                                     include_derivatives=include_derivatives,
                                     randomized=randomized, seed=seed)
    pod = project_hermite_smolyak_pod(model, Phi, mean, solve_fn=solve_fn)
    return pod, diag


# --------------------------------------------------------------------------
# Persistence: load a compressed sparse gradient-enhanced surrogate
# --------------------------------------------------------------------------
def load_pod_hermite_smolyak(path) -> PODHermiteSmolyak:
    """Load a :class:`PODHermiteSmolyak` saved by :meth:`PODHermiteSmolyak.save`.

    Rebuilds the coeff node pool (value + tangent) by re-keying ``node_thetas``,
    constructs a solver-less ``HermiteSmolyakSolverND(solve_fn=None, …,
    enhanced_axes=…)``, and returns its ``_finalize(index_set, pool)`` (the
    committed combination assembly with **zero solves**), wrapped with ``Φ``/
    ``mean``.  ``evaluate`` / ``evaluate_jax`` work immediately;
    ``evaluate_polished`` raises until a solver is attached.  Parsed metadata is
    stored on the returned object as ``.meta``.
    """
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "pod_hermite_smolyak")
    try:
        Phi = np.asarray(data["Phi"], dtype=float)
        mean = np.asarray(data["mean"], dtype=float)
        field_shape = tuple(int(x) for x in np.asarray(data["field_shape"], dtype=np.int64))
        node_thetas = np.asarray(data["node_thetas"], dtype=float)      # N×d
        node_U = np.asarray(data["node_U"], dtype=float)                # N×r
        node_dU = np.asarray(data["node_dU"], dtype=float)              # N×d×r
        node_iters = np.asarray(data["node_iters"])
        node_resids = np.asarray(data["node_resids"], dtype=float)
        index_set = [tuple(int(x) for x in row) for row in np.asarray(data["index_set"])]
        axes = [(float(a[0]), float(a[1])) for a in np.asarray(data["axes"], dtype=float)]
        enhanced = tuple(int(e) for e in np.asarray(data["enhanced"], dtype=np.int64))
        pool: Dict[tuple, tuple] = {}
        for i in range(node_thetas.shape[0]):
            key = _node_key(node_thetas[i])
            pool[key] = (np.asarray(node_U[i], dtype=float),
                         np.asarray(node_dU[i], dtype=float),
                         int(node_iters[i]), float(node_resids[i]))
        solver = HermiteSmolyakSolverND(solve_fn=None, axes=axes, tangent_fn=None,
                                        enhanced_axes=enhanced)
        coeff_model = solver._finalize(index_set, pool)   # combination coeffs recomputed
        pod = PODHermiteSmolyak(coeff_model, Phi, mean, field_shape, _solve_fn=None)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt PARASOL pod_hermite_smolyak surrogate '{path}': {e}")
    pod.meta = meta
    return pod
