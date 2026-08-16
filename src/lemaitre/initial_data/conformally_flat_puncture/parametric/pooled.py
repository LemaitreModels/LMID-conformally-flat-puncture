"""LM-initial-data — pool-weight evaluation of the combination-technique family.

Every combination model evaluates as ``Σ_l c_l · sub_l.evaluate(θ)``, and every
subgrid interpolant is *linear* in its stored fields with rank-1 per-axis
weights: a value axis contracts with the normalized barycentric vector ``ℓ``, a
Hermite axis with the cardinal pair ``(h, ĥ)``.  Distributing the subgrid sum
through that linearity, the whole model is a handful of GEMVs against the
**deduplicated** node pool::

    U(θ) = W_V(θ)·U_pool + Σ_e W_e(θ)·dU_pool[:,e] + Σ_p W_p(θ)·cross_pool[:,p]

where the pool-node weight vectors ``W`` are accumulated from tiny per-subgrid
outer products of 1-D bases through precomputed subgrid→pool index maps.  The
point is the redundancy this removes: the shipped 8-D model expands its 15,713
unique pool nodes into 101,575 subgrid node-slots (6.5×), and the subgrid-sum
path streams every slot on every query, while this path streams each pool field
exactly once (and a batch of queries becomes one GEMM).

**The subgrid-sum ``evaluate`` stays the oracle.**  The pooled path performs
the same contractions in a different summation order, so it agrees with the
oracle to roundoff but not bit-for-bit — it is therefore *opt-in*
(:func:`pooled_evaluator` / ``model.evaluate_pooled``), and it is gated by
equivalence tests against the oracle at random and at on-node θ plus a held-out
field-error run on the shipped models.  A wrong index map would still certify
(the polish repairs any guess), which is exactly why the gates compare fields,
not certificates.

Node-hit semantics mirror the oracle exactly: a θ-component within the
oracle's ``np.isclose(…, atol=1e-13)`` window of a node turns that axis's value
basis into the one-hot selector and its Hermite tangent basis into zero.  The
jax twin is branchless and, like every ``evaluate_jax`` in this package, must
not be queried exactly at a node.

Standalone: numpy + jax + the sibling ``parametric`` modules.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric_nd_smolyak import _node_key
from .hermite import _hermite_bases_np, _hermite_bases_jax

#: the oracle's exact-node window (`np.isclose(diff, 0, atol=1e-13)` with rtol
#: irrelevant against 0) — restated here as a plain comparison so the pooled
#: path hits the guard at exactly the same θ.
_NODE_ATOL = 1e-13


class _SubTerm:
    """One subgrid's contribution plan: its combination coefficient, its
    per-axis basis keys, and where its slots scatter in each term group."""

    __slots__ = ("coeff", "axis_keys", "n_slots", "val_slice",
                 "tan_slices", "cross_slices")

    def __init__(self, coeff, axis_keys, n_slots):
        self.coeff = float(coeff)
        self.axis_keys = axis_keys          # [(k, m), ...] length d
        self.n_slots = int(n_slots)
        self.val_slice = None               # slice into the value segment
        self.tan_slices = {}                # e -> slice into axis-e segment
        self.cross_slices = {}              # global pair index -> slice


class PooledEvaluator:
    """Pool-weight evaluator over a combination model's subgrids.

    Built once per model (cached by :func:`pooled_evaluator`): a one-time pass
    keys every subgrid slot into the deduplicated pool and copies each pool
    field once; every query is then 1-D basis evaluations + per-subgrid outer
    products + one ``bincount`` scatter + one GEMV per stored field block.
    """

    def __init__(self, model):
        d = int(model.d)
        self.d = d
        self.field_shape = tuple(model.field_shape)
        F = int(np.prod(self.field_shape)) if self.field_shape else 1
        enhanced = tuple(sorted(int(e) for e in getattr(model, "enhanced", ())))
        pairs = tuple(tuple(int(x) for x in p)
                      for p in getattr(model, "cross_pairs_global", ()))
        self.enhanced = enhanced
        self.pairs = pairs

        # ---- pass 1: key every subgrid slot into the pool; record index maps ----
        key_to_row: Dict[tuple, int] = {}
        sub_rows: List[np.ndarray] = []
        terms: List[_SubTerm] = []
        self._axes: Dict[Tuple[int, int], tuple] = {}   # (k, m) -> (nodes, weights, cvec|None)
        for c, sub in zip(model.coeffs, model.subgrids):
            nodes = sub.nodes
            shape = tuple(len(n) for n in nodes)
            axis_keys = []
            for k in range(d):
                m = shape[k]
                keyk = (k, m)
                hermite_role = (k in enhanced) and (m > 1)
                # the level-0 rule makes the role a function of (axis, level):
                # every subgrid at this (k, m) must agree, or the shared basis
                # table would silently serve the wrong cardinal family.
                sub_enh = set(int(e) for e in getattr(sub, "enhanced", ()))
                if hermite_role != (k in sub_enh):
                    raise ValueError(
                        f"subgrid enhancement disagrees with the level-0 rule at "
                        f"axis {k} (m={m}): model-enhanced={enhanced}, "
                        f"subgrid-enhanced={tuple(sorted(sub_enh))}")
                if keyk not in self._axes:
                    cvec = np.asarray(sub.cvec[k], float) if hermite_role else None
                    self._axes[keyk] = (np.asarray(nodes[k], float),
                                        np.asarray(sub.weights[k], float), cvec)
                axis_keys.append(keyk)
            thetas = np.stack(np.meshgrid(*nodes, indexing="ij"),
                              axis=-1).reshape(-1, d)
            rows = np.empty(thetas.shape[0], dtype=np.int64)
            for i in range(thetas.shape[0]):
                key = _node_key(thetas[i])
                row = key_to_row.get(key)
                if row is None:
                    row = len(key_to_row)
                    key_to_row[key] = row
                rows[i] = row
            sub_rows.append(rows)
            terms.append(_SubTerm(c, axis_keys, rows.size))
        N = len(key_to_row)
        self.n_pool = N

        # ---- pass 2: copy each pool field exactly once (first subgrid wins;
        # the subgrids were assembled from one pool, so all copies agree) ----
        V = np.empty((N, F))
        filled = np.zeros(N, dtype=bool)
        n_enh = len(enhanced)
        dU = np.empty((N, n_enh, F)) if n_enh else None
        npair = len(pairs)
        cross = np.zeros((N, npair, F)) if npair else None
        cross_filled = np.zeros((N, npair), dtype=bool) if npair else None
        pair_index = {p: i for i, p in enumerate(pairs)}
        for rows, sub in zip(sub_rows, model.subgrids):
            new = ~filled[rows]                     # rows are unique within a subgrid
            if np.any(new):
                V[rows[new]] = np.asarray(sub.U_nodes, float).reshape(-1, F)[new]
                if n_enh:
                    flat_dU = np.asarray(sub.dU_nodes, float).reshape(-1, self.d, F)
                    dU[rows[new]] = flat_dU[new][:, list(enhanced), :]
                filled[rows[new]] = True
            for j, p in enumerate(getattr(sub, "cross_pairs", ())):
                pi = pair_index[tuple(int(x) for x in p)]
                flat_c = np.asarray(sub.cross_nodes, float).reshape(-1, len(sub.cross_pairs), F)
                todo = ~cross_filled[rows, pi]
                if np.any(todo):
                    cross[rows[todo], pi] = flat_c[todo, j]
                    cross_filled[rows[todo], pi] = True
        self._V = V
        self._dU = dU
        self._cross = cross

        # ---- term groups: concatenated index arrays, one scatter per group ----
        def _cat(selected):
            offs, cat = 0, []
            slices = []
            for t, rows in selected:
                slices.append((t, slice(offs, offs + rows.size)))
                cat.append(rows)
                offs += rows.size
            return (np.concatenate(cat) if cat else np.empty(0, np.int64)), slices

        self._val_rows, val_slices = _cat(list(zip(terms, sub_rows)))
        for t, sl in val_slices:
            t.val_slice = sl
        self._tan_rows: Dict[int, np.ndarray] = {}
        for e in enhanced:
            sel = [(t, rows) for t, rows, sub in zip(terms, sub_rows, model.subgrids)
                   if e in set(int(x) for x in getattr(sub, "enhanced", ()))]
            rows_e, slices = _cat(sel)
            self._tan_rows[e] = rows_e
            for t, sl in slices:
                t.tan_slices[e] = sl
        self._cross_rows: Dict[int, np.ndarray] = {}
        for pi, p in enumerate(pairs):
            sel = [(t, rows) for t, rows, sub in zip(terms, sub_rows, model.subgrids)
                   if tuple(p) in set(tuple(int(x) for x in q)
                                      for q in getattr(sub, "cross_pairs", ()))]
            rows_p, slices = _cat(sel)
            self._cross_rows[pi] = rows_p
            for t, sl in slices:
                t.cross_slices[pi] = sl
        self._terms = terms
        self._jax_consts_cache = None

    # ------------------------------------------------------------------
    # per-query 1-D bases (numpy, node-safe — mirrors the oracle's guard)
    # ------------------------------------------------------------------
    def _bases_np(self, theta):
        out = {}
        for (k, m), (nodes, weights, cvec) in self._axes.items():
            diff = theta[k] - nodes
            hit = np.abs(diff) <= _NODE_ATOL
            if np.any(hit):
                one = np.zeros(m)
                one[int(np.argmax(hit))] = 1.0
                out[(k, m)] = (one, np.zeros(m) if cvec is not None else None)
            elif cvec is not None:
                h, hh, _, _ = _hermite_bases_np(theta[k], nodes, weights, cvec)
                out[(k, m)] = (h, hh)
            else:
                t = weights / diff
                out[(k, m)] = (t / t.sum(), None)
        return out

    @staticmethod
    def _outer(coeff, factors):
        w = coeff
        for f in factors:
            w = np.multiply.outer(w, f)
        return np.asarray(w).ravel()

    def weights(self, theta):
        """``(W_V, {e: W_e}, {p: W_p})`` pool-weight vectors at one θ (numpy)."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if theta.shape[0] != self.d:
            raise ValueError(f"theta has {theta.shape[0]} comps; expected d={self.d}")
        B = self._bases_np(theta)
        seg_val = np.empty(self._val_rows.size)
        seg_tan = {e: np.empty(r.size) for e, r in self._tan_rows.items()}
        seg_cross = {pi: np.empty(r.size) for pi, r in self._cross_rows.items()}
        for t in self._terms:
            base = [B[kk][0] for kk in t.axis_keys]
            seg_val[t.val_slice] = self._outer(t.coeff, base)
            for e, sl in t.tan_slices.items():
                fac = list(base)
                fac[e] = B[t.axis_keys[e]][1]
                seg_tan[e][sl] = self._outer(t.coeff, fac)
            for pi, sl in t.cross_slices.items():
                e0, e1 = self.pairs[pi]
                fac = list(base)
                fac[e0] = B[t.axis_keys[e0]][1]
                fac[e1] = B[t.axis_keys[e1]][1]
                seg_cross[pi][sl] = self._outer(t.coeff, fac)
        N = self.n_pool
        Wv = np.bincount(self._val_rows, seg_val, minlength=N)
        Wd = {e: np.bincount(self._tan_rows[e], seg_tan[e], minlength=N)
              for e in seg_tan}
        Wc = {pi: np.bincount(self._cross_rows[pi], seg_cross[pi], minlength=N)
              for pi in seg_cross}
        return Wv, Wd, Wc

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def evaluate(self, theta):
        """``U(θ)`` by pool-weight GEMVs (numpy, node-safe)."""
        Wv, Wd, Wc = self.weights(theta)
        out = Wv @ self._V
        for ei, e in enumerate(self.enhanced):
            out += Wd[e] @ self._dU[:, ei, :]
        for pi in Wc:
            out += Wc[pi] @ self._cross[:, pi, :]
        return out.reshape(self.field_shape)

    def evaluate_batch(self, thetas):
        """``U`` at a batch of query points — the weights become a ``(B, N)``
        matrix and every field block is hit with one GEMM."""
        thetas = np.asarray(thetas, dtype=float)
        if thetas.ndim != 2 or thetas.shape[1] != self.d:
            raise ValueError(f"thetas must be (B, {self.d}); got {thetas.shape}")
        rows = [self.weights(th) for th in thetas]
        WV = np.stack([r[0] for r in rows])
        out = WV @ self._V
        for ei, e in enumerate(self.enhanced):
            out += np.stack([r[1][e] for r in rows]) @ self._dU[:, ei, :]
        for pi in range(len(self.pairs)):
            out += np.stack([r[2][pi] for r in rows]) @ self._cross[:, pi, :]
        return out.reshape((thetas.shape[0],) + self.field_shape)

    # ------------------------------------------------------------------
    # jax twin (branchless, off-node; differentiable in θ; jit-friendly)
    # ------------------------------------------------------------------
    def _jax_consts(self):
        c = self._jax_consts_cache
        if c is None:
            c = (jnp.asarray(self._V),
                 None if self._dU is None else jnp.asarray(self._dU),
                 None if self._cross is None else jnp.asarray(self._cross))
            self._jax_consts_cache = c
        return c

    def _bases_jax(self, theta):
        out = {}
        for (k, m), (nodes, weights, cvec) in self._axes.items():
            if cvec is not None:
                h, hh = _hermite_bases_jax(theta[k], jnp.asarray(nodes),
                                           jnp.asarray(weights), jnp.asarray(cvec))
                out[(k, m)] = (h, hh)
            else:
                t = jnp.asarray(weights) / (theta[k] - jnp.asarray(nodes))
                out[(k, m)] = (t / jnp.sum(t), None)
        return out

    @staticmethod
    def _outer_jax(coeff, factors):
        w = jnp.asarray(coeff)
        for f in factors:
            w = jnp.tensordot(w, f, axes=0)
        return jnp.ravel(w)

    def evaluate_jax(self, theta):
        """``jnp`` twin of :meth:`evaluate` — same off-node contract as every
        ``evaluate_jax`` in this package; wrap with
        :func:`parametric_nd.jax_evaluator` to compile it once per model."""
        theta = jnp.asarray(theta)
        B = self._bases_jax(theta)
        V, dU, cross = self._jax_consts()
        N = self.n_pool

        def scatter(rows, segs):
            return jnp.zeros(N).at[rows].add(jnp.concatenate(segs))

        seg_val, seg_tan, seg_cross = [], {e: [] for e in self.enhanced}, \
            {pi: [] for pi in range(len(self.pairs))}
        for t in self._terms:
            base = [B[kk][0] for kk in t.axis_keys]
            seg_val.append(self._outer_jax(t.coeff, base))
            for e in t.tan_slices:
                fac = list(base)
                fac[e] = B[t.axis_keys[e]][1]
                seg_tan[e].append(self._outer_jax(t.coeff, fac))
            for pi in t.cross_slices:
                e0, e1 = self.pairs[pi]
                fac = list(base)
                fac[e0] = B[t.axis_keys[e0]][1]
                fac[e1] = B[t.axis_keys[e1]][1]
                seg_cross[pi].append(self._outer_jax(t.coeff, fac))
        out = scatter(self._val_rows, seg_val) @ V
        for ei, e in enumerate(self.enhanced):
            if seg_tan[e]:
                out += scatter(self._tan_rows[e], seg_tan[e]) @ dU[:, ei, :]
        for pi in range(len(self.pairs)):
            if seg_cross[pi]:
                out += scatter(self._cross_rows[pi], seg_cross[pi]) @ cross[:, pi, :]
        return jnp.reshape(out, self.field_shape)


class _PODPooled:
    """Pooled evaluator of a POD wrapper: pooled coeff interpolation + decode
    ``u = mean + Φ·c`` (the decode is unchanged from the wrapper)."""

    def __init__(self, pod):
        self.inner = PooledEvaluator(pod.coeff_model)
        self.Phi = pod.Phi
        self.mean = pod.mean
        self.field_shape = tuple(pod.field_shape)
        self._Phi_j = jnp.asarray(self.Phi)
        self._mean_j = jnp.asarray(self.mean)

    def evaluate(self, theta):
        c = np.asarray(self.inner.evaluate(theta)).reshape(-1)
        return (self.mean + self.Phi @ c).reshape(self.field_shape)

    def evaluate_batch(self, thetas):
        C = self.inner.evaluate_batch(np.asarray(thetas, float))
        C = C.reshape(C.shape[0], -1)
        return (self.mean + C @ self.Phi.T).reshape((C.shape[0],) + self.field_shape)

    def evaluate_jax(self, theta):
        c = jnp.reshape(self.inner.evaluate_jax(theta), (-1,))
        return jnp.reshape(self._mean_j + self._Phi_j @ c, self.field_shape)


def pooled_evaluator(model):
    """The pool-weight evaluator of ``model``, built once and cached on it.

    ``model`` may be any combination container
    (:class:`~.parametric_nd_smolyak.SmolyakSolutionND` and subclasses) or a POD
    wrapper around one (``coeff_model``/``Phi``/``mean``).  The returned object
    exposes ``evaluate`` / ``evaluate_batch`` / ``evaluate_jax``.
    """
    ev = model.__dict__.get("_pooled_evaluator")
    if ev is None:
        if hasattr(model, "coeff_model") and hasattr(model, "Phi"):
            ev = _PODPooled(model)
        else:
            ev = PooledEvaluator(model)
        model.__dict__["_pooled_evaluator"] = ev
    return ev
