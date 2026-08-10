"""LM-initial-data — FULL BILINEAR (cross-enhanced) Hermite–Smolyak collocation.

The cross-carrying sibling of the committed gradient-only Hermite–Smolyak layer
(:mod:`hermite_smolyak`).  The committed enhanced-axis tensor product
(:meth:`hermite_nd.HermiteSolutionND.evaluate`) keeps, on an enhanced axis ``k``,
the value cardinal ``h`` acting on the value and on the OTHER enhanced axes'
first-derivative accumulators, plus ``ĥ`` on axis ``k``'s own derivative — but it
**drops the mixed cross term** ``ĥ_{e0}·ĥ_{e1}·∂²U/∂θ_{e0}∂θ_{e1}`` between two
simultaneously-enhanced axes (the documented "gradient-only" R3 fallback).

For two enhanced axes there is exactly one such mixed partial (``C(2,2)=1``).  This
module adds it: it carries a **mixed accumulator** per enhanced *pair* and, when a
subgrid Hermite-processes enhanced axis ``k``, updates the other enhanced axis's
first-derivative accumulator with ``+ ĥ_k · (mixed)`` — the honest tensor product
of the two 1-D Hermite operators, i.e. the **full bilinear** Hermite on the
enhanced subspace.  Because the combination technique is a sum of tensor-product
subgrids, doing the full tensor product per subgrid gives the full bilinear sparse
interpolant with no change to the combination coefficients (``evaluate =
Σ_l c_l·sub_l.evaluate`` is inherited verbatim).

**Reduce-to-committed (the M2 gate).**  With no active cross pair (``cross_pairs``
empty on every subgrid — e.g. a subgrid with ≤1 enhanced axis at level ≥1, or the
whole model built with the cross disabled) the evaluation is **bit-for-bit**
:meth:`hermite_nd.HermiteSolutionND.evaluate` /
:meth:`hermite_smolyak.HermiteSmolyakSolutionND.evaluate` (identical float ops:
the mixed accumulators are simply absent).

**Scope.**  Full bilinear = full tensor product, hence EXACT, for **≤2 enhanced
axes** (the shipped ``enh=[χ_Ay, χ_By]`` target).  For ``n>2`` enhanced axes this
carries all pairwise (second-order) crosses but not the ``≥3``-order mixed
partials (a documented pairwise-bilinear truncation of the ``2^n`` full product).

**Add-only.**  Subclasses / reuses the committed
:class:`hermite_nd.HermiteSolutionND`,
:class:`hermite_smolyak.HermiteSmolyakSolutionND` and the committed Smolyak
primitives / persistence helpers **verbatim**; never edits a committed module.  The
per-node cross field is supplied by the build driver (from
``applications.sensitivity_3d_cross``); this module only stores, combines and
persists it.  Certification is unchanged (``evaluate_polished`` reuses the attached
``solve_fn`` → ``newton_solve``).

Standalone: numpy + jax + the sibling ``parametric`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric_nd import FORMAT_VERSION, _pack_meta, _git_commit
from .parametric_nd_smolyak import combination_coeffs, _node_key
from .hermite_nd import HermiteSolutionND
from .hermite_smolyak import HermiteSmolyakSolutionND, HermiteSmolyakSolverND
from .hermite import cardinal_deriv_at_nodes, _hermite_bases_np, _hermite_bases_jax


# ==========================================================================
# 1.  Cross-enhanced subgrid  (full bilinear on the enhanced axes)
# ==========================================================================
@dataclass
class HermiteCrossSolutionND(HermiteSolutionND):
    """A :class:`hermite_nd.HermiteSolutionND` that ALSO carries the per-node mixed
    second partials of the enhanced pairs and evaluates the **full bilinear**
    Hermite (parent + the dropped ``ĥ·ĥ·∂²U`` cross term).

    Extra fields
    ------------
    cross_nodes : ``(n_0, …, n_{d-1}, n_pairs, *field_shape)`` — the stored mixed
        partial ``∂²U/∂θ_{e0}∂θ_{e1}`` for each active pair (aligned with
        :attr:`cross_pairs`); ``n_pairs`` may be 0 (then evaluate == parent).
    cross_pairs : list of ``(e0, e1)`` GLOBAL axis-index pairs (``e0<e1``) that are
        Hermite-active on THIS subgrid (both axes enhanced with level ≥1).
    """

    cross_nodes: np.ndarray = None
    cross_pairs: Tuple[Tuple[int, int], ...] = ()

    # ----- full-bilinear tensor-product interpolant (numpy, node-safe) -----
    def evaluate(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if theta.shape[0] != self.d:
            raise ValueError(f"theta has {theta.shape[0]} comps; expected d={self.d}")
        enh = set(int(e) for e in self.enhanced)
        accV = self.U_nodes
        accD = {e: np.take(self.dU_nodes, e, axis=self.d) for e in enh}
        # mixed accumulators keyed by the sorted enhanced pair
        accD2 = {tuple(pr): np.take(self.cross_nodes, pi, axis=self.d)
                 for pi, pr in enumerate(self.cross_pairs)}
        for k in range(self.d):
            nodes_k = self.nodes[k]
            diff = theta[k] - nodes_k
            hit = np.isclose(diff, 0.0, atol=1e-13)
            if np.any(hit):                                    # exact-node guard
                i = int(np.argmax(hit))
                accV = accV[i]
                accD = {e: accD[e][i] for e in accD if e != k}
                accD2 = {pr: v[i] for pr, v in accD2.items() if k not in pr}
            elif k in enh:
                h, hh, _, _ = _hermite_bases_np(theta[k], nodes_k,
                                                self.weights[k], self.cvec[k])
                newV = (np.tensordot(h, accV, axes=(0, 0))
                        + np.tensordot(hh, accD[k], axes=(0, 0)))
                newD = {}
                for e in accD:
                    if e == k:
                        continue
                    pr = (min(e, k), max(e, k))
                    val = np.tensordot(h, accD[e], axes=(0, 0))
                    if pr in accD2:                            # the FULL cross term
                        val = val + np.tensordot(hh, accD2[pr], axes=(0, 0))
                    newD[e] = val
                newD2 = {pr: np.tensordot(h, v, axes=(0, 0))
                         for pr, v in accD2.items() if k not in pr}
                accV, accD, accD2 = newV, newD, newD2
            else:                                              # value-only barycentric
                t = self.weights[k] / diff
                s = t.sum()
                accV = np.tensordot(t, accV, axes=(0, 0)) / s
                accD = {e: np.tensordot(t, accD[e], axes=(0, 0)) / s for e in accD}
                accD2 = {pr: np.tensordot(t, v, axes=(0, 0)) / s
                         for pr, v in accD2.items()}
        return accV

    # ----- JAX-differentiable twin (branchless, off-node) -----
    def evaluate_jax(self, theta):
        theta = jnp.asarray(theta)
        enh = set(int(e) for e in self.enhanced)
        accV = jnp.asarray(self.U_nodes)
        dU = jnp.asarray(self.dU_nodes)
        accD = {e: jnp.take(dU, e, axis=self.d) for e in enh}
        cross = jnp.asarray(self.cross_nodes) if len(self.cross_pairs) else None
        accD2 = {tuple(pr): jnp.take(cross, pi, axis=self.d)
                 for pi, pr in enumerate(self.cross_pairs)}
        for k in range(self.d):
            nodes_k = jnp.asarray(self.nodes[k])
            diff = theta[k] - nodes_k
            if k in enh:
                h, hh = _hermite_bases_jax(theta[k], nodes_k,
                                           jnp.asarray(self.weights[k]),
                                           jnp.asarray(self.cvec[k]))
                newV = (jnp.tensordot(h, accV, axes=(0, 0))
                        + jnp.tensordot(hh, accD[k], axes=(0, 0)))
                newD = {}
                for e in accD:
                    if e == k:
                        continue
                    pr = (min(e, k), max(e, k))
                    val = jnp.tensordot(h, accD[e], axes=(0, 0))
                    if pr in accD2:
                        val = val + jnp.tensordot(hh, accD2[pr], axes=(0, 0))
                    newD[e] = val
                newD2 = {pr: jnp.tensordot(h, v, axes=(0, 0))
                         for pr, v in accD2.items() if k not in pr}
                accV, accD, accD2 = newV, newD, newD2
            else:
                t = jnp.asarray(self.weights[k]) / diff
                s = jnp.sum(t)
                accV = jnp.tensordot(t, accV, axes=(0, 0)) / s
                accD = {e: jnp.tensordot(t, accD[e], axes=(0, 0)) / s for e in accD}
                accD2 = {pr: jnp.tensordot(t, v, axes=(0, 0)) / s
                         for pr, v in accD2.items()}
        return accV


# ==========================================================================
# 2.  Cross-enhanced Smolyak combination solution
# ==========================================================================
@dataclass
class HermiteSmolyakCrossSolutionND(HermiteSmolyakSolutionND):
    """Combination-technique sparse interpolant whose subgrids are
    :class:`HermiteCrossSolutionND` (full bilinear on the enhanced axes).

    ``evaluate``/``evaluate_jax``/``evaluate_polished`` are inherited verbatim
    (``Σ_l c_l · sub_l.evaluate``); only persistence carries the extra per-node
    cross field.  :attr:`cross_pairs_global` are the GLOBAL enhanced pairs stored
    at the node level; :attr:`pool` is the deduplicated node pool
    (``key -> (theta, U, dU, cross, iters, resid)``) the model was built from
    (used for a clean round-trip)."""

    cross_pairs_global: Tuple[Tuple[int, int], ...] = ()
    pool: Optional[Dict[tuple, tuple]] = field(repr=False, default=None)

    def save(self, path, *, meta=None):
        """Persist to a single ``.npz`` (numpy-only).  Stores the deduplicated
        node pool — ``node_thetas`` (N×d), ``node_U`` (N×\\*fs), ``node_dU``
        (N×d×\\*fs), ``node_cross`` (N×n_pairs×\\*fs), ``node_iters`` (N),
        ``node_resids`` (N) — plus ``index_set``, ``axes``, ``enhanced``,
        ``cross_pairs`` (n_pairs×2 GLOBAL indices), ``field_shape`` and
        ``meta_json``.  Round-trips bit-for-bit via
        :func:`load_hermite_smolyak_cross`."""
        if self.pool is None:
            raise RuntimeError("no node pool attached; build via the cross builder")
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        keys = list(self.pool)
        node_thetas = np.array([self.pool[k][0] for k in keys], dtype=float)
        node_U = np.array([self.pool[k][1] for k in keys], dtype=float)
        node_dU = np.array([self.pool[k][2] for k in keys], dtype=float)
        node_cross = np.array([self.pool[k][3] for k in keys], dtype=float)
        node_iters = np.array([self.pool[k][4] for k in keys], dtype=np.int64)
        node_resids = np.array([self.pool[k][5] for k in keys], dtype=float)
        index_set = np.array([[int(x) for x in l] for l in self.index_set], dtype=np.int64)
        axes = np.array([[float(lo), float(hi)] for (lo, hi) in self.axes], dtype=float)
        cross_pairs = np.array([[int(a), int(b)] for (a, b) in self.cross_pairs_global],
                               dtype=np.int64).reshape(-1, 2)
        full_meta = {"d": int(self.d), "n_solver_nodes": int(self.n_solver_nodes),
                     "git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION
        full_meta["kind"] = "hermite_smolyak_cross"
        np.savez(path,
                 node_thetas=node_thetas, node_U=node_U, node_dU=node_dU,
                 node_cross=node_cross, node_iters=node_iters, node_resids=node_resids,
                 index_set=index_set, axes=axes,
                 enhanced=np.asarray(sorted(int(e) for e in self.enhanced), dtype=np.int64),
                 cross_pairs=cross_pairs,
                 field_shape=np.asarray(self.field_shape, dtype=np.int64),
                 meta_json=_pack_meta(full_meta))
        return path


# ==========================================================================
# 3.  Builder: assemble the cross combination interpolant from a node pool
# ==========================================================================
def _global_pairs(enhanced: Sequence[int]) -> List[Tuple[int, int]]:
    """All sorted GLOBAL enhanced-axis pairs (``C(|enhanced|,2)`` of them)."""
    return [tuple(sorted(pr)) for pr in combinations(sorted(int(e) for e in enhanced), 2)]


class HermiteSmolyakCrossSolverND(HermiteSmolyakSolverND):
    """Cross-enhanced sibling of :class:`hermite_smolyak.HermiteSmolyakSolverND`.

    Reuses the committed level-0 / index-set / combination machinery verbatim; the
    only differences are that :meth:`_assemble_subgrid` produces a
    :class:`HermiteCrossSolutionND` (carrying the active pairs' cross fields) and
    the node pool value is ``(U, dU, cross, iters, resid)`` (``cross`` shaped
    ``(n_global_pairs, *field)``)."""

    def _assemble_subgrid(self, l, pool) -> HermiteCrossSolutionND:
        nodes, weights = self._subgrid_nodes(l)
        shape = tuple(len(n) for n in nodes)
        first = pool[next(iter(pool))]
        field_shape = first[1].shape                       # (theta, U, ...) -> U is [1]
        global_pairs = _global_pairs(self.enhanced)
        # active pairs on THIS subgrid: both axes enhanced at level >= 1
        sub_enh = set(self._subgrid_enhanced(l))
        active = [pr for pr in global_pairs if pr[0] in sub_enh and pr[1] in sub_enh]
        gp_index = {pr: i for i, pr in enumerate(global_pairs)}

        U_nodes = np.empty(shape + field_shape, dtype=float)
        dU_nodes = np.empty(shape + (self.d,) + field_shape, dtype=float)
        cross_nodes = np.empty(shape + (len(active),) + field_shape, dtype=float)
        iters = np.zeros(shape, dtype=int)
        resids = np.zeros(shape, dtype=float)
        for idx in np.ndindex(*shape):
            key = _node_key([nodes[k][idx[k]] for k in range(self.d)])
            _theta, Ua, dU, cross, it, rs = pool[key]     # pool value: 6-tuple
            U_nodes[idx] = Ua
            dU_nodes[idx] = dU
            for j, pr in enumerate(active):
                cross_nodes[idx + (j,)] = cross[gp_index[pr]]
            iters[idx] = it
            resids[idx] = rs
        axes_meta = [(lo, hi, len(nodes[k]) - 1) for k, (lo, hi) in enumerate(self.axes)]
        cvec = [cardinal_deriv_at_nodes(n) for n in nodes]
        return HermiteCrossSolutionND(
            axes=axes_meta, nodes=nodes, weights=weights, U_nodes=U_nodes,
            dU_nodes=dU_nodes, cvec=cvec, enhanced=self._subgrid_enhanced(l),
            iters=iters, residuals=resids, _solve_fn=None,
            cross_nodes=cross_nodes, cross_pairs=tuple(active))

    def _finalize(self, index_set, pool) -> HermiteSmolyakCrossSolutionND:
        coeffs = combination_coeffs(index_set)
        kept = sorted(coeffs)
        subgrids = [self._assemble_subgrid(l, pool) for l in kept]
        total_iters = int(sum(v[4] for v in pool.values()))    # v = (theta,U,dU,cross,iters,resid)
        return HermiteSmolyakCrossSolutionND(
            axes=self.axes, index_set=[tuple(l) for l in index_set],
            coeffs=[coeffs[l] for l in kept], subgrids=subgrids,
            enhanced=self.enhanced, n_solver_nodes=len(pool),
            total_iters=total_iters, _solve_fn=self.solve_fn,
            cross_pairs_global=tuple(_global_pairs(self.enhanced)),
            pool=dict(pool))


def build_cross_from_pool(axes: Sequence[Tuple[float, float]],
                          index_set: Sequence[tuple],
                          enhanced: Sequence[int],
                          pool: Dict[tuple, tuple],
                          solve_fn: Optional[Callable] = None
                          ) -> HermiteSmolyakCrossSolutionND:
    """Assemble a :class:`HermiteSmolyakCrossSolutionND` from an already-solved node
    pool ``key -> (theta, U, dU, cross, iters, resid)`` (``cross`` =
    ``(n_global_pairs, *field)``), reusing the combination-technique assembly with
    **zero solves**.

    This is the Milestone-3 entry: the driver loads the shipped model's
    ``node_U``/``node_dU``, computes the per-node cross via
    ``applications.sensitivity_3d_cross``, and hands the pool here."""
    solver = HermiteSmolyakCrossSolverND(solve_fn=solve_fn, axes=axes,
                                         tangent_fn=None, enhanced_axes=enhanced)
    return solver._finalize([tuple(int(x) for x in l) for l in index_set], pool)


# ==========================================================================
# 4.  Persistence: load a cross-enhanced surrogate saved by .save
# ==========================================================================
def load_hermite_smolyak_cross(path) -> HermiteSmolyakCrossSolutionND:
    """Load a :class:`HermiteSmolyakCrossSolutionND` saved by :meth:`.save`.

    Rebuilds the node pool (value + tangent + cross) and returns the finalized
    combination interpolant with **zero solves** (``evaluate``/``evaluate_jax``
    work immediately; ``evaluate_polished`` raises until a solver is attached).
    Parsed metadata is stored on the returned object as ``.meta``."""
    from .parametric_nd import _load_npz, _unpack_meta, _check_meta
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "hermite_smolyak_cross")
    try:
        node_thetas = np.asarray(data["node_thetas"], dtype=float)
        node_U = np.asarray(data["node_U"], dtype=float)
        node_dU = np.asarray(data["node_dU"], dtype=float)
        node_cross = np.asarray(data["node_cross"], dtype=float)
        node_iters = np.asarray(data["node_iters"])
        node_resids = np.asarray(data["node_resids"], dtype=float)
        index_set = [tuple(int(x) for x in row) for row in np.asarray(data["index_set"])]
        axes = [(float(a[0]), float(a[1])) for a in np.asarray(data["axes"], dtype=float)]
        enhanced = tuple(int(e) for e in np.asarray(data["enhanced"], dtype=np.int64))
        pool: Dict[tuple, tuple] = {}
        for i in range(node_thetas.shape[0]):
            key = _node_key(node_thetas[i])
            pool[key] = (node_thetas[i], np.asarray(node_U[i], dtype=float),
                         np.asarray(node_dU[i], dtype=float),
                         np.asarray(node_cross[i], dtype=float),
                         int(node_iters[i]), float(node_resids[i]))
        sol = build_cross_from_pool(axes, index_set, enhanced, pool, solve_fn=None)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt LM-initial-data hermite_smolyak_cross surrogate '{path}': {e}")
    sol.meta = meta
    return sol
