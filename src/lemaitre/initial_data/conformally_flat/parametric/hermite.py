"""LM-initial-data — gradient-enhanced (Hermite) collocation in parameter space (H1).

The 1-D **foundation** of the gradient-enhanced surrogate
(``GRADIENT_ENHANCED_PLAN.md`` §2/§4).  The committed value-only interpolant
(:class:`parametric.ParametricSolution` / :class:`parametric_nd.ParametricSolutionND`)
stores only the solved field ``U_i`` at each parameter node and combines nodes
with Lagrange weights.  This module adds the **gradient-enhanced (Hermite)**
interpolant that matches value **and** first parameter-derivative at every node:

    ũ(θ) = Σ_i [ h_i(θ) U_i + ĥ_i(θ) U_i' ],
        h_i(θ_j)=δ_ij,  h_i'(θ_j)=0,  ĥ_i(θ_j)=0,  ĥ_i'(θ_j)=δ_ij,

a polynomial of degree ``2Q+1`` on ``Q+1`` CGL nodes.  ``U_i' ≡ dU/dθ|_{θ_i}`` is
the certified implicit-function tangent (``applications.sensitivity.certified_tangent``)
— one extra back-solve against the node's already-factored Jacobian, per §0.

Phase 0 (``reports/hermite_derisk/``) returned **GO**: Hermite ~doubles the
per-node field-error rate on the hard axes (b, χ_B) and its exposed gradient
matches the certified tangent to ~1e-10, at fixed node count.

**Scope (H1): the 1-D interpolant only** — axis-agnostic (a plain Hermite in one
parameter; the hard-axis-only *selection* and the N-D tensor layer are H2).  This
is the degenerate ``d=1`` object that ``HermiteSolutionND`` (H2) will build on.

**Add-only.**  Reuses ``parametric.cheb_param_nodes`` and the ``parametric_nd``
persistence helpers (``_pack_meta``/``_unpack_meta``/``_git_commit``/``_load_npz``/
``_check_meta``/``FORMAT_VERSION``) **verbatim**; never edits a committed module.
Certification is unchanged — the Hermite object is still only a *guess*;
``evaluate_polished`` (Newton) remains the certificate (reused, not reimplemented).

Standalone: numpy + jax + the sibling ``parametric``/``parametric_nd`` modules.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric import cheb_param_nodes            # reused verbatim (the 1-D CGL layer)
from .parametric_nd import (                        # persistence helpers, reused verbatim
    FORMAT_VERSION,
    _pack_meta,
    _unpack_meta,
    _git_commit,
    _load_npz,
    _check_meta,
)


# --------------------------------------------------------------------------
# Node-set quantities: barycentric weights and the cardinal-derivative vector
# --------------------------------------------------------------------------
def _bary_weights(nodes: np.ndarray) -> np.ndarray:
    """General barycentric weights ``w_i = 1/∏_{j≠i}(θ_i−θ_j)``.

    The barycentric quotient is invariant to an overall rescaling of the
    weights, so this agrees with :func:`parametric.cheb_param_nodes`' closed-form
    CGL weights up to a constant.  For a CGL node set prefer passing the
    ``cheb_param_nodes`` weights directly (better conditioned at large Q); this
    fallback is fine for modest Q and for non-CGL node sets.
    """
    nodes = np.asarray(nodes, dtype=float)
    n = nodes.size
    w = np.ones(n)
    for i in range(n):
        d = nodes[i] - nodes
        d[i] = 1.0
        w[i] = 1.0 / np.prod(d)
    return w


def cardinal_deriv_at_nodes(nodes: np.ndarray) -> np.ndarray:
    """``c_i = ℓ_i'(θ_i) = Σ_{j≠i} 1/(θ_i−θ_j)`` — a node-set-only quantity.

    This is the diagonal of the barycentric differentiation matrix; it enters the
    Hermite cardinal ``h_i = (1 − 2 c_i (θ−θ_i)) ℓ_i²``.
    """
    nodes = np.asarray(nodes, dtype=float)
    n = nodes.size
    c = np.zeros(n)
    idx = np.arange(n)
    for i in range(n):
        d = nodes[i] - nodes
        d[i] = 1.0
        c[i] = np.sum(1.0 / d[idx != i])
    return c


# --------------------------------------------------------------------------
# Hermite cardinal basis (barycentric-stable form) + analytic derivative
# --------------------------------------------------------------------------
HermiteCardinal = namedtuple(
    "HermiteCardinal", ["h", "hhat", "dh", "dhhat", "cvec", "weights"]
)


def _hermite_bases_np(theta, nodes, weights, cvec):
    """``(h_i, ĥ_i, h_i', ĥ_i')`` at scalar ``theta`` — OFF-node numpy.

    All four length-``n`` arrays via the barycentric-stable Hermite form
    ``h_i=(1−2c_i d_i)ℓ_i²``, ``ĥ_i=d_i ℓ_i²`` (``d_i=θ−θ_i``), and their exact
    analytic θ-derivatives.  ``theta`` must be strictly off every node (the
    barycentric quotient is 0/0 at a node — the caller node-guards).
    """
    theta = float(theta)
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    cvec = np.asarray(cvec, dtype=float)
    d = theta - nodes                    # (n,)  all nonzero off-node
    t = weights / d
    S = t.sum()
    ell = t / S                          # ℓ_i
    ell2 = ell ** 2
    h = (1.0 - 2.0 * cvec * d) * ell2
    hh = d * ell2
    # analytic derivatives:  ℓ_i' = ℓ_i(−1/d_i − S'/S),  S' = −Σ t_j/d_j
    Sp = -(t / d).sum()
    ellp = ell * (-1.0 / d - Sp / S)
    ell2p = 2.0 * ell * ellp
    dh = (-2.0 * cvec) * ell2 + (1.0 - 2.0 * cvec * d) * ell2p
    dhh = ell2 + d * ell2p
    return h, hh, dh, dhh


def _hermite_bases_jax(theta, nodes, weights, cvec):
    """``(h_i, ĥ_i)`` at scalar ``theta`` — branchless jnp (differentiable, OFF-node).

    The value bases only; the derivative is obtained by ``jax`` autodiff of the
    assembled field (see :meth:`HermiteSolution1D.evaluate_jax`), so it stays
    consistent with the numpy analytic form to roundoff.
    """
    theta = jnp.asarray(theta)
    nodes = jnp.asarray(nodes)
    weights = jnp.asarray(weights)
    cvec = jnp.asarray(cvec)
    d = theta - nodes
    t = weights / d
    ell = t / jnp.sum(t)
    ell2 = ell ** 2
    h = (1.0 - 2.0 * cvec * d) * ell2
    hh = d * ell2
    return h, hh


def hermite_cardinal(nodes, weights=None):
    """Hermite cardinal pair ``(h_i, ĥ_i)`` (and derivatives) on a node set.

    Returns a :class:`HermiteCardinal` of **callables** ``h(θ)``, ``hhat(θ)``,
    ``dh(θ)``, ``dhhat(θ)`` — each mapping a scalar ``θ`` to a length-``Q+1``
    array over the nodes — plus the node-set vectors ``cvec`` (``c_i``) and
    ``weights`` (barycentric).  ``h_i(θ_j)=δ_ij``, ``ĥ_i(θ_j)=0``,
    ``h_i'(θ_j)=0``, ``ĥ_i'(θ_j)=δ_ij`` by construction.

    The callables use the branchless barycentric form and are valid **off-node**
    only (0/0 at a node); the node-safe assembly lives in
    :class:`HermiteSolution1D`.  If ``weights`` is None the general barycentric
    weights are computed (:func:`_bary_weights`); for a CGL node set pass
    :func:`parametric.cheb_param_nodes`' weights for best conditioning.
    """
    nodes = np.asarray(nodes, dtype=float)
    if weights is None:
        weights = _bary_weights(nodes)
    weights = np.asarray(weights, dtype=float)
    cvec = cardinal_deriv_at_nodes(nodes)

    def h(theta):
        return _hermite_bases_np(theta, nodes, weights, cvec)[0]

    def hhat(theta):
        return _hermite_bases_np(theta, nodes, weights, cvec)[1]

    def dh(theta):
        return _hermite_bases_np(theta, nodes, weights, cvec)[2]

    def dhhat(theta):
        return _hermite_bases_np(theta, nodes, weights, cvec)[3]

    return HermiteCardinal(h=h, hhat=hhat, dh=dh, dhhat=dhhat, cvec=cvec, weights=weights)


# --------------------------------------------------------------------------
# Single-node Taylor predictor (the degenerate one-node special case)
# --------------------------------------------------------------------------
def _as_deriv_list(dU):
    """Coerce the tangent argument into a list ``[U', U'', …]`` of derivative fields.

    A plain array is the single first tangent (order-1); a list/tuple of arrays
    supplies successive derivatives for higher-order Taylor expansion.
    """
    if isinstance(dU, (list, tuple)):
        return [np.asarray(d, dtype=float) for d in dU]
    return [np.asarray(dU, dtype=float)]


def taylor_predict(node, U, dU, dtheta, order: int = 1):
    """Single-node Taylor predictor ``U(θ_i + δθ) ≈ Σ_{m=0}^{order} U^{(m)} δθ^m/m!``.

    The degenerate one-node special case of the Hermite interpolant (§2), kept as
    a lightweight/diagnostic mode.  Valid inside the analyticity-wall radius; it
    collapses near a hard real singularity (``b→0``), so it is a fallback, not the
    model — the multi-node :class:`HermiteSolution1D` is the surrogate.

    Parameters
    ----------
    node : float
        The expansion point ``θ_i`` (informational; the offset ``δθ`` is passed
        explicitly).
    U : array
        The stored value field ``U_i``.
    dU : array or sequence of arrays
        First tangent ``U_i'`` (``order=1``), or ``[U', U'', …]`` for higher order.
    dtheta : float
        The offset ``δθ = θ − θ_i``.
    order : int
        Taylor order.  ``order=0`` returns ``U_i`` exactly.
    """
    U = np.asarray(U, dtype=float)
    if order == 0:
        return U.copy()
    derivs = _as_deriv_list(dU)
    if len(derivs) < order:
        raise ValueError(
            f"taylor_predict(order={order}) needs {order} derivative field(s); "
            f"got {len(derivs)}")
    out = U.astype(float, copy=True)
    fact = 1.0
    dpow = 1.0
    for m in range(1, order + 1):
        fact *= m
        dpow *= float(dtheta)
        out = out + derivs[m - 1] * (dpow / fact)
    return out


# --------------------------------------------------------------------------
# 1-D gradient-enhanced (Hermite) parametric solution container
# --------------------------------------------------------------------------
@dataclass
class HermiteSolution1D:
    """Gradient-enhanced 1-D Hermite interpolant, mirroring the
    :class:`parametric.ParametricSolution` API but carrying per-node **values**
    ``U_i`` *and* **tangents** ``U_i'``.

    Fields
    ------
    lo, hi, Q : the parameter range and degree (``Q+1`` nodes).
    nodes, weights : CGL nodes (descending value) and barycentric weights, aligned.
    U_nodes : (Q+1, *field_shape) converged fields, aligned to ``nodes``.
    dU_nodes : (Q+1, *field_shape) certified tangents ``dU/dθ`` at the nodes.
    cvec : (Q+1,) cardinal-derivative vector ``c_i`` (node-set only).
    iters, residuals : optional per-node Newton provenance (aligned; may be None).
    """

    lo: float
    hi: float
    Q: int
    nodes: np.ndarray            # (Q+1,)  CGL order (descending value)
    weights: np.ndarray          # (Q+1,)  barycentric weights, aligned
    U_nodes: np.ndarray          # (Q+1, *field_shape)  values, aligned
    dU_nodes: np.ndarray         # (Q+1, *field_shape)  tangents dU/dθ, aligned
    cvec: np.ndarray             # (Q+1,)  cardinal-derivative vector c_i
    iters: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    _solve_fn: Callable = field(repr=False, default=None)

    # ---- construction from a CGL box (reuses cheb_param_nodes verbatim) ----
    @classmethod
    def from_nodes(cls, lo, hi, Q, U_nodes, dU_nodes, *,
                   iters=None, residuals=None, solve_fn=None):
        """Build on the CGL nodes of ``[lo, hi]`` at degree ``Q`` from stacked
        node values/tangents (both ``(Q+1, *field_shape)``, aligned to the
        ``cheb_param_nodes`` CGL order)."""
        nodes, weights = cheb_param_nodes(float(lo), float(hi), int(Q))
        U_nodes = np.asarray(U_nodes, dtype=float)
        dU_nodes = np.asarray(dU_nodes, dtype=float)
        if U_nodes.shape[0] != nodes.size or dU_nodes.shape[0] != nodes.size:
            raise ValueError(
                f"U_nodes/dU_nodes leading dim must be {nodes.size} (Q+1); "
                f"got {U_nodes.shape[0]}/{dU_nodes.shape[0]}")
        if U_nodes.shape != dU_nodes.shape:
            raise ValueError("U_nodes and dU_nodes must have identical shape")
        return cls(lo=float(lo), hi=float(hi), Q=int(Q), nodes=nodes, weights=weights,
                   U_nodes=U_nodes, dU_nodes=dU_nodes,
                   cvec=cardinal_deriv_at_nodes(nodes),
                   iters=iters, residuals=residuals, _solve_fn=solve_fn)

    @property
    def field_shape(self):
        return self.U_nodes.shape[1:]

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.size)

    # ---- gradient-enhanced (Hermite) interpolant: value ----
    def evaluate(self, theta):
        """``ũ(θ)`` via the Hermite cardinals.  Scalar or batch ``θ`` (numpy,
        node-safe).  Returns the field for a scalar query, or a stacked
        ``(len(θ), *field_shape)`` array for a batch (mirrors
        :meth:`parametric.ParametricSolution.evaluate`)."""
        te = np.atleast_1d(np.asarray(theta, dtype=float))
        fshape = self.field_shape
        out = np.empty((te.shape[0],) + fshape)
        for j, tj in enumerate(te):
            diff = tj - self.nodes
            hit = np.isclose(diff, 0.0, atol=1e-13)
            if np.any(hit):                              # exact-node guard → stored value
                out[j] = self.U_nodes[int(np.argmax(hit))]
            else:
                h, hh, _, _ = _hermite_bases_np(tj, self.nodes, self.weights, self.cvec)
                out[j] = (np.tensordot(h, self.U_nodes, axes=(0, 0))
                          + np.tensordot(hh, self.dU_nodes, axes=(0, 0)))
        return out[0] if np.ndim(theta) == 0 else out

    # ---- gradient-enhanced interpolant: exposed derivative (node-safe numpy) ----
    def evaluate_grad(self, theta):
        """``dũ/dθ`` via the analytic Hermite-cardinal derivatives (numpy,
        node-safe).  Scalar or batch ``θ``.  At a node it returns the stored
        tangent ``U_i'`` exactly (the cardinal property ``ũ'(θ_i)=U_i'``); this
        is the node-safe companion of ``jax.jacfwd(evaluate_jax)`` (which is 0/0
        exactly at a node)."""
        te = np.atleast_1d(np.asarray(theta, dtype=float))
        fshape = self.field_shape
        out = np.empty((te.shape[0],) + fshape)
        for j, tj in enumerate(te):
            diff = tj - self.nodes
            hit = np.isclose(diff, 0.0, atol=1e-13)
            if np.any(hit):                              # exact-node guard → stored tangent
                out[j] = self.dU_nodes[int(np.argmax(hit))]
            else:
                _, _, dh, dhh = _hermite_bases_np(tj, self.nodes, self.weights, self.cvec)
                out[j] = (np.tensordot(dh, self.U_nodes, axes=(0, 0))
                          + np.tensordot(dhh, self.dU_nodes, axes=(0, 0)))
        return out[0] if np.ndim(theta) == 0 else out

    # ---- JAX-differentiable interpolant (the exposed-gradient hook) ----
    def evaluate_jax(self, theta):
        """``jnp`` twin of :meth:`evaluate` — branchless and differentiable in the
        scalar ``θ`` (the exposed-gradient hook for the applications).  Must NOT
        be queried exactly at a node (the barycentric quotient is removable there
        but not finite for jax; use :meth:`evaluate_grad` at nodes)."""
        h, hh = _hermite_bases_jax(theta, self.nodes, self.weights, self.cvec)
        U = jnp.asarray(self.U_nodes)
        dU = jnp.asarray(self.dU_nodes)
        return jnp.tensordot(h, U, axes=(0, 0)) + jnp.tensordot(hh, dU, axes=(0, 0))

    # ---- certified evaluation (unchanged; reuses the attached solve_fn) ----
    def evaluate_polished(self, theta, newton_steps: int = 2, tol: float = 1e-12):
        """Hermite prediction + 1–2 Newton steps → certified ``‖R‖≤tol`` at ``θ``.

        The Hermite object is only a *guess*; certification is unchanged from the
        committed path (the attached ``solve_fn`` → ``newton_solve``).  Returns
        ``(U, info)`` with ``info.residual_norm`` the certified constraint
        residual, independent of any interpolation error."""
        if self._solve_fn is None:
            raise RuntimeError("no solve_fn attached; pass solve_fn= to from_nodes / the builder")
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(float(theta), guess, tol, newton_steps)
        return U, info

    # ---- persistence (numpy-only .npz; reuses the parametric_nd helpers) ----
    def save(self, path, *, meta=None):
        """Persist to a single ``.npz`` (numpy-only, no pickle).  Round-trips
        bit-for-bit via :func:`load_hermite`.  The reloaded object is a standalone
        predictor: ``evaluate``/``evaluate_grad`` need only numpy + the parametric
        modules (``evaluate_polished`` needs a reattached ``solve_fn``)."""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        arrays = {
            "nodes": np.asarray(self.nodes, dtype=float),
            "weights": np.asarray(self.weights, dtype=float),
            "U_nodes": np.asarray(self.U_nodes, dtype=float),
            "dU_nodes": np.asarray(self.dU_nodes, dtype=float),
            "cvec": np.asarray(self.cvec, dtype=float),
            "box": np.array([float(self.lo), float(self.hi), float(self.Q)], dtype=float),
            "field_shape": np.asarray(self.field_shape, dtype=np.int64),
        }
        if self.iters is not None:
            arrays["iters"] = np.asarray(self.iters, dtype=np.int64)
        if self.residuals is not None:
            arrays["residuals"] = np.asarray(self.residuals, dtype=float)
        full_meta = {"git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION        # authoritative
        full_meta["kind"] = "hermite1d"
        arrays["meta_json"] = _pack_meta(full_meta)
        np.savez(path, **arrays)
        return path


def load_hermite(path) -> "HermiteSolution1D":
    """Load a :class:`HermiteSolution1D` saved by :meth:`HermiteSolution1D.save`.

    Reconstructs with ``_solve_fn=None`` (``evaluate``/``evaluate_grad`` work
    immediately; ``evaluate_polished`` raises until a solver is attached).  The
    parsed metadata is stored on the returned object as ``.meta``.
    """
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "hermite1d")
    try:
        lo, hi, Q = (float(data["box"][0]), float(data["box"][1]),
                     int(round(float(data["box"][2]))))
        nodes = np.asarray(data["nodes"], dtype=float)
        weights = np.asarray(data["weights"], dtype=float)
        U_nodes = np.asarray(data["U_nodes"], dtype=float)
        dU_nodes = np.asarray(data["dU_nodes"], dtype=float)
        cvec = np.asarray(data["cvec"], dtype=float)
        iters = np.asarray(data["iters"]) if "iters" in data else None
        residuals = np.asarray(data["residuals"], dtype=float) if "residuals" in data else None
        sol = HermiteSolution1D(lo=lo, hi=hi, Q=Q, nodes=nodes, weights=weights,
                                U_nodes=U_nodes, dU_nodes=dU_nodes, cvec=cvec,
                                iters=iters, residuals=residuals, _solve_fn=None)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt LM-initial-data hermite surrogate '{path}': {e}")
    sol.meta = meta
    return sol
