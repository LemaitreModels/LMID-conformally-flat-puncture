"""LM-initial-data — POD (reduced-basis) compression of the FULL-BILINEAR (cross) sparse
Hermite-Smolyak surrogate.

The cross-carrying sibling of the committed :mod:`hermite_smolyak_pod`
(``PODHermiteSmolyak``).  The gradient-only POD compresses ``(1 + n_enh)`` fields
per node (value + one certified tangent per enhanced axis) onto ONE spatial POD
basis ``Φ`` — the derivatives share the value basis (the analytic ``θ→field`` map's
tangents live in the solution-manifold tangent space).  The full-bilinear cross
model (:mod:`hermite_smolyak_cross`) adds ``C(n_enh,2)`` mixed second-partials
``∂²U/∂θ_i∂θ_j``; those are also tangents of the same manifold, so they compress
onto the SAME ``Φ``.  So this re-encodes value + first tangents + cross onto one
basis and interpolates the length-``r`` coeff vectors with the committed
cross combination machinery (:class:`hermite_smolyak_cross.HermiteSmolyakCrossSolutionND`).

Construction (direct analog of ``PODHermiteSmolyak``):
  1. ``pod_basis_pool_cross`` SVDs the stacked ``[U | dU_e… | cross_p…]`` corpus of
     the deduplicated pool (reusing ``hermite_pod.pod_basis`` verbatim via a fake
     single-axis corpus whose "enhanced derivative" slots carry BOTH the first
     tangents and the cross fields).
  2. Each cross subgrid is projected onto ``Φ`` — value/first-tangent coeffs via
     ``hermite_pod.project_hermite_pod`` (verbatim), the cross coeffs by the same
     ``cross·Φ`` projection — giving a coeff-space
     :class:`hermite_smolyak_cross.HermiteCrossSolutionND`.
  3. The coeff subgrids are combined by the committed cross combination
     (``Σ_l c_l·sub_l.evaluate`` in coeff space) and decoded ``u = mean + Φ·c``.

By linearity of the (Hermite/barycentric) interpolation + the combination-technique
``Σ_l c_l = 1`` property, ``evaluate`` is bit-identical (to roundoff) to the POD
projection ``mean + ΦΦᵀ(full_cross_interpolant − mean)`` of the full cross
interpolant, so the reduce-to-committed / node-exactness properties carry over and
the certified polish is unchanged (the compressed object is only a *guess*).

**Add-only.**  Reuses ``hermite_pod`` (``pod_basis``, ``project_hermite_pod``,
``rank_for_tail``), ``hermite_smolyak_cross`` (``HermiteCrossSolutionND``,
``HermiteSmolyakCrossSolutionND``, ``build_cross_from_pool``, ``_global_pairs``),
and the ``parametric_nd`` persistence helpers — all verbatim; never edits a
committed module.

Standalone: numpy + jax + the sibling ``parametric`` modules.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric_nd import (FORMAT_VERSION, _pack_meta, _unpack_meta,
                            _git_commit, _load_npz, _check_meta)
from .parametric_nd_smolyak import _node_key
from .hermite_nd import HermiteSolutionND
from .hermite_pod import pod_basis, project_hermite_pod
from .hermite_smolyak_cross import (HermiteCrossSolutionND,
                                    HermiteSmolyakCrossSolutionND,
                                    build_cross_from_pool, _global_pairs)


# --------------------------------------------------------------------------
# POD basis from the stacked value + first-tangent + CROSS corpus of the pool
# --------------------------------------------------------------------------
def _pool_corpus_hermite_cross(model: HermiteSmolyakCrossSolutionND) -> HermiteSolutionND:
    """A synthetic single-axis :class:`hermite_nd.HermiteSolutionND` whose stacked
    corpus is ``[U | dU_e (enhanced) | cross_p (pairs)]`` of the cross model's pool,
    so :func:`hermite_pod.pod_basis` builds ``Φ`` from value + first tangents + the
    cross fields (all live in the same manifold-tangent spatial span)."""
    pool = model.pool                                   # key -> (theta, U, dU, cross, iters, resid)
    keys = list(pool)
    N = len(keys)
    fs = model.field_shape
    enh = tuple(int(e) for e in model.enhanced)
    npair = len(model.cross_pairs_global)
    Us = np.stack([np.asarray(pool[k][1], dtype=float) for k in keys])         # (N, *fs)
    cols = []
    for e in enh:
        cols.append(np.stack([np.asarray(pool[k][2][e], dtype=float) for k in keys]))
    for p in range(npair):
        cols.append(np.stack([np.asarray(pool[k][3][p], dtype=float) for k in keys]))
    if cols:
        dUs = np.stack(cols, axis=1)                    # (N, n_enh+npair, *fs)
    else:
        dUs = np.zeros((N, 0) + tuple(fs), dtype=float)
    fake_nodes = np.arange(N, dtype=float)
    return HermiteSolutionND(
        axes=[(0.0, 1.0, 0)], nodes=[fake_nodes], weights=[np.ones(N)],
        U_nodes=Us, dU_nodes=dUs, cvec=[np.zeros(N)],
        enhanced=tuple(range(len(enh) + npair)),
        iters=np.zeros(N, dtype=int), residuals=np.zeros(N))


def pod_basis_pool_cross(model: HermiteSmolyakCrossSolutionND, *,
                         r: Optional[int] = None, tail: Optional[float] = None,
                         include_derivatives: bool = True, randomized: bool = False,
                         seed: int = 0):
    """POD spatial modes ``Φ`` from the value + first-tangent + cross corpus of the
    cross model's pool (thin wrapper on :func:`hermite_pod.pod_basis`)."""
    fake = _pool_corpus_hermite_cross(model)
    return pod_basis(fake, r=r, tail=tail, include_derivatives=include_derivatives,
                     randomized=randomized, seed=seed)


# --------------------------------------------------------------------------
# The POD (reduced-basis) full-bilinear cross Hermite-Smolyak surrogate
# --------------------------------------------------------------------------
class PODHermiteSmolyakCross:
    """Reduced-basis (POD) re-encoding of a
    :class:`hermite_smolyak_cross.HermiteSmolyakCrossSolutionND`.

    Interpolates the length-``r`` POD coefficient vectors with the committed cross
    combination machinery (coeff-space
    :class:`hermite_smolyak_cross.HermiteSmolyakCrossSolutionND`) and decodes
    ``u = mean + Φ·c``."""

    def __init__(self, coeff_model: HermiteSmolyakCrossSolutionND, Phi, mean,
                 field_shape, _solve_fn: Optional[Callable] = None):
        self.coeff_model = coeff_model
        self.Phi = np.asarray(Phi, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
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

    def coeffs(self, theta):
        """The interpolated length-``r`` POD coefficient vector at ``θ``."""
        return np.asarray(self.coeff_model.evaluate(theta)).reshape(-1)

    def evaluate(self, theta):
        c = self.coeffs(theta)
        return (self.mean + self.Phi @ c).reshape(self.field_shape)

    def evaluate_jax(self, theta):
        c = jnp.reshape(self.coeff_model.evaluate_jax(theta), (-1,))
        return jnp.reshape(self._mean_j + self._Phi_j @ c, self.field_shape)

    def evaluate_polished(self, theta, newton_steps: int = 2, tol: float = 1e-12):
        if self._solve_fn is None:
            raise RuntimeError("no solve_fn attached; build via build_pod_hermite_smolyak_cross "
                               "with a solver-backed model, or reattach a solve_fn")
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(np.asarray(theta, dtype=float), guess, tol, newton_steps)
        return U, info

    # ----- persistence (numpy-only .npz) -----
    def save(self, path, *, meta=None, coeff_dtype=np.float64, mode_dtype=np.float64,
             slim=True):
        """Persist the deduplicated coeff pool (value + first tangents + cross, in
        POD-coeff space) + ``Φ``/``mean``.  Round-trips via
        :func:`load_pod_hermite_smolyak_cross`.

        ``slim`` (default) stores the tangent block ONLY for the enhanced axes,
        which is **lossless**: :meth:`hermite_smolyak_cross.HermiteCrossSolutionND.evaluate`
        and its jax twin read ``dU_nodes`` only at ``self.enhanced``, so the other
        ``d - n_enh`` blocks are dead weight — they inflated the stored size by
        ``(1+d+npair)/(1+n_enh+npair)`` (1.5x at d=4, 2.5x at d=8) without ever
        being read.  The loader re-expands them as zeros, so the in-memory object
        and every evaluation are bit-for-bit identical either way.

        ``slim=False`` writes the historical full-``d`` layout.  Both layouts load;
        the file records which one it uses in ``meta['dU_layout']``.
        """
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        cm = self.coeff_model
        pool = cm.pool                            # key -> (theta, U(r), dU(d,r), cross(np,r), it, rs)
        keys = list(pool)
        node_thetas = np.array([pool[k][0] for k in keys], dtype=float)
        node_U = np.array([pool[k][1] for k in keys]).astype(coeff_dtype)
        node_dU = np.array([pool[k][2] for k in keys]).astype(coeff_dtype)
        enh_sorted = sorted(int(e) for e in cm.enhanced)
        if slim:                                  # drop the never-read tangent blocks
            node_dU = node_dU[:, enh_sorted, :]
        node_cross = np.array([pool[k][3] for k in keys]).astype(coeff_dtype)
        node_iters = np.array([pool[k][4] for k in keys], dtype=np.int64)
        node_resids = np.array([pool[k][5] for k in keys], dtype=float)
        index_set = np.array([[int(x) for x in l] for l in cm.index_set], dtype=np.int64)
        axes = np.array([[float(lo), float(hi)] for (lo, hi) in cm.axes], dtype=float)
        cross_pairs = np.array([[int(a), int(b)] for (a, b) in cm.cross_pairs_global],
                               dtype=np.int64).reshape(-1, 2)
        full_meta = {"d": int(self.d), "r": int(self.r),
                     "n_solver_nodes": int(cm.n_solver_nodes), "git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION
        full_meta["kind"] = "pod_hermite_smolyak_cross"
        full_meta["dU_layout"] = "enhanced" if slim else "full"
        np.savez(path, Phi=self.Phi.astype(mode_dtype), mean=self.mean,
                 node_thetas=node_thetas, node_U=node_U, node_dU=node_dU,
                 node_cross=node_cross, node_iters=node_iters, node_resids=node_resids,
                 index_set=index_set, axes=axes,
                 enhanced=np.asarray(enh_sorted, dtype=np.int64),
                 cross_pairs=cross_pairs,
                 field_shape=np.asarray(self.field_shape, dtype=np.int64),
                 r=np.asarray(self.r, dtype=np.int64), meta_json=_pack_meta(full_meta))
        return path


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def _project_cross_subgrid(sub: HermiteCrossSolutionND, Phi, mean) -> HermiteCrossSolutionND:
    """Project one cross subgrid onto ``Φ`` — value/first-tangent coeffs via
    ``hermite_pod.project_hermite_pod`` (verbatim), cross coeffs by ``cross·Φ`` —
    returning a coeff-space :class:`HermiteCrossSolutionND` (field_shape == (r,))."""
    base = project_hermite_pod(sub, Phi, mean).coeff_hermite     # U_nodes(r), dU_nodes(d,r)
    d = sub.d
    grid = sub.U_nodes.shape[:d]
    N = int(np.prod(grid)) if grid else 1
    nfeat = int(np.prod(sub.field_shape)) if sub.field_shape else 1
    Phi = np.asarray(Phi, dtype=float)
    r = Phi.shape[1]
    npair = len(sub.cross_pairs)
    if npair:
        cx = np.asarray(sub.cross_nodes, dtype=float)            # (grid, npair, *fs)
        cross_coeff = np.empty(grid + (npair, r), dtype=float)
        for j in range(npair):
            Cj = np.take(cx, j, axis=d).reshape(N, nfeat)        # (N, nfeat)
            cross_coeff[..., j, :] = (Cj @ Phi).reshape(grid + (r,))
    else:
        cross_coeff = np.empty(grid + (0, r), dtype=float)
    return HermiteCrossSolutionND(
        axes=list(base.axes), nodes=list(base.nodes), weights=list(base.weights),
        U_nodes=base.U_nodes, dU_nodes=base.dU_nodes, cvec=list(base.cvec),
        enhanced=tuple(base.enhanced), iters=base.iters, residuals=base.residuals,
        _solve_fn=None, cross_nodes=cross_coeff, cross_pairs=tuple(sub.cross_pairs))


def project_hermite_smolyak_cross_pod(model: HermiteSmolyakCrossSolutionND, Phi, mean, *,
                                      solve_fn: Optional[Callable] = None
                                      ) -> PODHermiteSmolyakCross:
    """Project a cross model onto ``Φ`` → a :class:`PODHermiteSmolyakCross`."""
    coeff_subs = [_project_cross_subgrid(sub, Phi, mean) for sub in model.subgrids]
    # coeff-space pool (for save/load): value+dU+cross projected per node
    Phi = np.asarray(Phi, dtype=float); mean = np.asarray(mean, dtype=float)
    r = Phi.shape[1]
    fs = model.field_shape
    nfeat = int(np.prod(fs))
    cpool: Dict[tuple, tuple] = {}
    for k, (th, U, dU, cross, it, rs) in model.pool.items():
        Uc = (np.asarray(U, float).reshape(nfeat) - mean) @ Phi
        dUc = np.stack([np.asarray(dU[a], float).reshape(nfeat) @ Phi for a in range(model.d)])
        Cc = (np.stack([np.asarray(cross[p], float).reshape(nfeat) @ Phi
                        for p in range(len(model.cross_pairs_global))])
              if len(model.cross_pairs_global) else np.zeros((0, r)))
        cpool[k] = (th, Uc, dUc, Cc, it, rs)
    coeff_model = HermiteSmolyakCrossSolutionND(
        axes=list(model.axes), index_set=[tuple(l) for l in model.index_set],
        coeffs=list(model.coeffs), subgrids=coeff_subs, enhanced=tuple(model.enhanced),
        n_solver_nodes=model.n_solver_nodes, total_iters=model.total_iters,
        _solve_fn=None, cross_pairs_global=tuple(model.cross_pairs_global), pool=cpool)
    sf = solve_fn if solve_fn is not None else getattr(model, "_solve_fn", None)
    return PODHermiteSmolyakCross(coeff_model, Phi, mean, model.field_shape, _solve_fn=sf)


def build_pod_hermite_smolyak_cross(model: HermiteSmolyakCrossSolutionND, *,
                                    r: Optional[int] = None, tail: Optional[float] = None,
                                    include_derivatives: bool = True,
                                    randomized: bool = False, seed: int = 0,
                                    solve_fn: Optional[Callable] = None):
    """Convenience: :func:`pod_basis_pool_cross` then
    :func:`project_hermite_smolyak_cross_pod`.  Returns ``(pod, diag)``."""
    Phi, mean, diag = pod_basis_pool_cross(model, r=r, tail=tail,
                                           include_derivatives=include_derivatives,
                                           randomized=randomized, seed=seed)
    pod = project_hermite_smolyak_cross_pod(model, Phi, mean, solve_fn=solve_fn)
    return pod, diag


def truncate_pod_cross(pod: PODHermiteSmolyakCross, r_new: int,
                       solve_fn: Optional[Callable] = None) -> PODHermiteSmolyakCross:
    """A rank-``r_new`` truncation of ``pod`` (slice ``Φ`` and the coeff pool to the
    leading ``r_new`` modes) — the memory/accuracy-sweep primitive.  ``r_new ==
    pod.r`` returns an equivalent model."""
    cm = pod.coeff_model
    r_new = int(min(r_new, pod.r))
    Phi = pod.Phi[:, :r_new]
    pool = {k: (v[0], np.asarray(v[1])[:r_new], np.asarray(v[2])[:, :r_new],
                np.asarray(v[3])[:, :r_new] if v[3].size else v[3][:, :r_new],
                v[4], v[5]) for k, v in cm.pool.items()}
    coeff_model = build_cross_from_pool(cm.axes, cm.index_set, tuple(cm.enhanced), pool)
    sf = solve_fn if solve_fn is not None else pod._solve_fn
    return PODHermiteSmolyakCross(coeff_model, Phi, pod.mean, pod.field_shape, _solve_fn=sf)


# --------------------------------------------------------------------------
# Persistence: load a compressed cross surrogate
# --------------------------------------------------------------------------
def load_pod_hermite_smolyak_cross(path) -> PODHermiteSmolyakCross:
    """Load a :class:`PODHermiteSmolyakCross` saved by :meth:`.save` (zero solves)."""
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "pod_hermite_smolyak_cross")
    try:
        Phi = np.asarray(data["Phi"], dtype=float)
        mean = np.asarray(data["mean"], dtype=float)
        field_shape = tuple(int(x) for x in np.asarray(data["field_shape"], dtype=np.int64))
        node_thetas = np.asarray(data["node_thetas"], dtype=float)
        node_U = np.asarray(data["node_U"], dtype=float)
        node_dU = np.asarray(data["node_dU"], dtype=float)
        node_cross = np.asarray(data["node_cross"], dtype=float)
        node_iters = np.asarray(data["node_iters"])
        node_resids = np.asarray(data["node_resids"], dtype=float)
        index_set = [tuple(int(x) for x in row) for row in np.asarray(data["index_set"])]
        axes = [(float(a[0]), float(a[1])) for a in np.asarray(data["axes"], dtype=float)]
        enhanced = tuple(int(e) for e in np.asarray(data["enhanced"], dtype=np.int64))
        # Slim files store the tangent block only for the enhanced axes; re-expand to
        # the full (N, d, r) the evaluator indexes by GLOBAL axis index.  The restored
        # blocks are the ones evaluate() never reads, so this is exact, not a default.
        # Files written before the layout key are full-d.
        if meta.get("dU_layout", "full") == "enhanced":
            d = int(meta["d"])
            full = np.zeros((node_dU.shape[0], d) + node_dU.shape[2:], dtype=float)
            full[:, list(enhanced), :] = node_dU
            node_dU = full
        pool: Dict[tuple, tuple] = {}
        for i in range(node_thetas.shape[0]):
            pool[_node_key(node_thetas[i])] = (
                node_thetas[i], np.asarray(node_U[i], dtype=float),
                np.asarray(node_dU[i], dtype=float), np.asarray(node_cross[i], dtype=float),
                int(node_iters[i]), float(node_resids[i]))
        coeff_model = build_cross_from_pool(axes, index_set, enhanced, pool)
        pod = PODHermiteSmolyakCross(coeff_model, Phi, mean, field_shape, _solve_fn=None)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt LM-initial-data pod_hermite_smolyak_cross surrogate '{path}': {e}")
    pod.meta = meta
    return pod
