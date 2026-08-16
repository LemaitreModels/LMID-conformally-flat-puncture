"""LM-initial-data — POD (reduced-basis) compression of the gradient-enhanced SPARSE
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
  2. **Project the pool to coeff space.**  Each pool node's value and tangent
     stack are projected once (value coeff ``(U−mean)·Φ``, tangent coeff
     ``dU/dθ_k·Φ`` — the same projections ``hermite_pod.project_hermite_pod``
     applies), and the ``r``-dim coeff-space subgrids are assembled from the
     projected pool by the committed combination machinery — exactly as
     ``PODSmolyak`` replaces each value subgrid with an ``r``-dim
     ``ParametricSolutionND``, minus the 6.5× re-projection a per-subgrid pass
     would pay.
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

**Reuses** :mod:`hermite_pod` (``pod_basis`` — verbatim), :mod:`hermite_smolyak`
(``HermiteSmolyakSolutionND``, ``HermiteSmolyakSolverND`` — the combination
container + the loader ``_finalize``, verbatim), :mod:`hermite_nd`
(``HermiteSolutionND``), and the ``parametric_nd`` persistence helpers verbatim;
Certification is unchanged — the compressed object
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
    _git_commit,
    _load_model_npz,
    _save_npz,
)
from .parametric_nd_smolyak import _node_key           # node keying (verbatim)
from .certification import CertifiedEvaluateMixin   # the residual gate (one place)
from .hermite_nd import HermiteSolutionND              # the H2 interpolant (verbatim)
from .hermite_smolyak import (                          # the H5b sparse layer (verbatim)
    HermiteSmolyakSolutionND,
    HermiteSmolyakSolverND,
)
from .hermite_pod import pod_basis                      # the H3 POD machinery (verbatim)


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
                   randomized: bool = False, seed: int = 0,
                   value_diagnostics: bool = True):
    """POD spatial modes ``Φ`` from the stacked value+derivative corpus of the
    sparse pool of a :class:`hermite_smolyak.HermiteSmolyakSolutionND`.

    Thin wrapper that hands the pool to :func:`hermite_pod.pod_basis` (reused
    **verbatim**) via :func:`_pool_corpus_hermite`.  Returns ``(Phi, mean, diag)``
    with the identical ``diag`` structure — ``s``/``s_value`` singular values,
    ``rank_stacked``/``rank_value`` tables, and ``dU_on_value_basis_resid`` (the
    R5 "derivatives share the value basis" residual; the value-only half is
    skipped with ``value_diagnostics=False`` on memory-bound builds)."""
    fake = _pool_corpus_hermite(model)
    return pod_basis(fake, r=r, tail=tail, include_derivatives=include_derivatives,
                     randomized=randomized, seed=seed,
                     value_diagnostics=value_diagnostics)


# --------------------------------------------------------------------------
# The POD (reduced-basis) decode wrapper shared by the gradient-only and cross
# compressed models
# --------------------------------------------------------------------------
class _PODSmolyakBase(CertifiedEvaluateMixin):
    """Decode wrapper around a coeff-space combination model: interpolate the
    length-``r`` POD coefficient vectors with the committed combination
    machinery and decode ``u = mean + Φ·c``.

    Shared by :class:`PODHermiteSmolyak` and
    :class:`hermite_smolyak_pod_cross.PODHermiteSmolyakCross`, which used to be
    ~90 % copy-paste of each other; they now differ only in persistence and in
    which coeff-space container they wrap."""

    def __init__(self, coeff_model, Phi, mean, field_shape,
                 _solve_fn: Optional[Callable] = None):
        self.coeff_model = coeff_model          # combination model over r-dim coeffs
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
    def coeffs(self, theta):
        """The interpolated length-``r`` POD coefficient vector at ``θ``."""
        return np.asarray(self.coeff_model.evaluate(theta)).reshape(-1)   # (r,)

    def evaluate(self, theta):
        """``ũ(θ)`` decoded from the interpolated POD coefficients (numpy, node-safe)."""
        u = self.mean + self.Phi @ self.coeffs(theta)
        return u.reshape(self.field_shape)

    def evaluate_jax(self, theta):
        """``jnp`` twin of :meth:`evaluate` — the exposed-gradient hook
        (``jax.jacfwd`` gives ``P_r·∂U/∂θ``, the full sparse gradient projected
        onto ``Φ``).  Must NOT be queried exactly at a node."""
        c = jnp.reshape(self.coeff_model.evaluate_jax(theta), (-1,))
        u = self._mean_j + self._Phi_j @ c
        return jnp.reshape(u, self.field_shape)

    # ----- pool-weight path (opt-in; the subgrid sum stays the oracle) -----
    def evaluate_pooled(self, theta):
        """``ũ(θ)`` with the coeff interpolation on the pool-weight path
        (:mod:`.pooled`) and the same ``mean + Φ·c`` decode.  Agrees with
        :meth:`evaluate` to roundoff; node-safe."""
        from .pooled import pooled_evaluator
        return pooled_evaluator(self).evaluate(theta)


class PODHermiteSmolyak(_PODSmolyakBase):
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

    _solve_fn_hint = ("build via build_pod_hermite_smolyak with a "
                      "solver-backed HermiteSmolyakSolutionND, or reattach a solve_fn")

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
        so they need not be float64 (the paper O1 note): ``mode_dtype`` /
        ``coeff_dtype`` may be ``float32`` to halve the on-disk footprint (reload
        upcasts to float64, so ``evaluate`` still runs in float64).  Defaults
        preserve the float64 artifact byte-for-byte.  ``mean`` (nfeat, negligible)
        stays float64.
        """
        cm = self.coeff_model
        pool = cm._dedup_pool()                  # key -> (theta, U(r), dU(d,r), iters, resid)
        keys = list(pool)
        arrays = dict(
            Phi=self.Phi.astype(mode_dtype), mean=self.mean,
            node_thetas=np.array([pool[k][0] for k in keys], dtype=float),        # N×d
            node_U=np.array([pool[k][1] for k in keys]).astype(coeff_dtype),      # N×r
            node_dU=np.array([pool[k][2] for k in keys]).astype(coeff_dtype),     # N×d×r
            node_iters=np.array([pool[k][3] for k in keys], dtype=np.int64),      # N
            node_resids=np.array([pool[k][4] for k in keys], dtype=float),        # N
            index_set=np.array([[int(x) for x in l] for l in cm.index_set],
                               dtype=np.int64),
            axes=np.array([[float(lo), float(hi)] for (lo, hi) in cm.axes],
                          dtype=float),
            enhanced=np.asarray(sorted(int(e) for e in cm.enhanced), dtype=np.int64),
            field_shape=np.asarray(self.field_shape, dtype=np.int64),
            r=np.asarray(self.r, dtype=np.int64))
        return _save_npz(path, arrays,
                         {"d": int(self.d), "r": int(self.r),
                          "n_solver_nodes": int(cm.n_solver_nodes),
                          "git_commit": _git_commit()},
                         meta, kind="pod_hermite_smolyak")


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def project_hermite_smolyak_pod(model: HermiteSmolyakSolutionND, Phi: np.ndarray,
                                mean: np.ndarray, *,
                                solve_fn: Optional[Callable] = None) -> PODHermiteSmolyak:
    """Project a solver-backed
    :class:`hermite_smolyak.HermiteSmolyakSolutionND` onto the POD basis ``Φ``
    (``mean`` the value-mean) → a :class:`PODHermiteSmolyak`.

    The **pool** is projected once — value coeff ``(U−mean)·Φ``, tangent coeffs
    ``dU/dθ_k·Φ`` (same projections :func:`hermite_pod.project_hermite_pod`
    applies) — and the coeff-space subgrids are then assembled from the
    projected pool by the committed combination machinery.  Projecting per
    subgrid instead, as this used to, re-projects every pool node once per
    subgrid it appears in (a 6.5× redundancy on the shipped 8-D corpus) and
    materialises a full-rank coefficient copy per subgrid along the way."""
    Phi = np.asarray(Phi, dtype=float)
    mean = np.asarray(mean, dtype=float)
    nfeat = int(np.prod(model.field_shape)) if model.field_shape else 1
    pool = model._dedup_pool()               # key -> (theta, U, dU, iters, resid)
    cpool = {}
    for key, (theta, U, dU, it, rs) in pool.items():
        Uc = (np.asarray(U, dtype=float).reshape(nfeat) - mean) @ Phi
        dUc = np.asarray(dU, dtype=float).reshape(model.d, nfeat) @ Phi
        cpool[key] = (Uc, dUc, it, rs)
    solver = HermiteSmolyakSolverND(solve_fn=None, axes=list(model.axes),
                                    tangent_fn=None,
                                    enhanced_axes=tuple(model.enhanced))
    coeff_model = solver._finalize([tuple(l) for l in model.index_set], cpool)
    sf = solve_fn if solve_fn is not None else getattr(model, "_solve_fn", None)
    return PODHermiteSmolyak(coeff_model, Phi, mean, model.field_shape, _solve_fn=sf)


def build_pod_hermite_smolyak(model: HermiteSmolyakSolutionND, *, r: Optional[int] = None,
                              tail: Optional[float] = None, include_derivatives: bool = True,
                              randomized: bool = False, seed: int = 0,
                              value_diagnostics: bool = True,
                              solve_fn: Optional[Callable] = None):
    """Convenience: :func:`pod_basis_pool` then :func:`project_hermite_smolyak_pod`.

    Returns ``(pod, diag)`` — the :class:`PODHermiteSmolyak` and the
    ``pod_basis_pool`` diagnostics (rank tables, share-the-basis residuals)."""
    Phi, mean, diag = pod_basis_pool(model, r=r, tail=tail,
                                     include_derivatives=include_derivatives,
                                     randomized=randomized, seed=seed,
                                     value_diagnostics=value_diagnostics)
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
    def build(data, meta):
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
        return PODHermiteSmolyak(coeff_model, Phi, mean, field_shape, _solve_fn=None)

    return _load_model_npz(path, "pod_hermite_smolyak", build)
