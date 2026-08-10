"""LM-initial-data — N-D gradient-enhanced (hard-axis-only) Hermite surrogate (H2).

The N-D lift of the 1-D foundation :mod:`hermite` (H1), following the
**anisotropic** route of ``GRADIENT_ENHANCED_PLAN.md`` §2/§4 H2: enhance only the
1–2 *hard* axes (near-merger separation ``b``, small-hole spin ``χ_B``/``S_By``)
with 1-D Hermite, and keep the easy axes (``q``, ``S_Ay``) **value-only**
barycentric.  This is a tensor product of 1-D operators:

  * a **value-only** axis contracts with the committed barycentric weights
    (bit-for-bit :class:`parametric_nd.ParametricSolutionND`);
  * an **enhanced** axis contracts with the Hermite cardinal pair
    ``(h_i, ĥ_i)`` (reused **verbatim** from :mod:`hermite`), consuming both the
    stored value ``U_i`` and the stored parameter-tangent ``dU/dθ_k|_i``.

**The mixed-partial decision (stated, per the prompt).**  A *full* tensor-product
Hermite over ``n`` enhanced axes needs mixed partials up to order ``n``
(``2^n`` fields/node — the R3 cost).  With **≤1 enhanced axis** the tensor product
is **exact and needs no mixed partial** (the value-only axes are linear and
commute with the Hermite operator, so ``dU/dθ_k`` contracted through them is
exactly ``∂/∂θ_k`` of the value-only interpolant).  With **≥2 enhanced axes** the
only missing ingredient is the single mixed partial ``∂²U/∂θ_i∂θ_j`` between the
enhanced axes — a **second-order** implicit-function tangent that **does not exist
anywhere in the committed tree** (``sensitivity.certified_tangent`` and
``solver_abt.tangent_{b,q}``/``tangent_chi`` are all first-order).  Deriving it
would be new physics, outside the add-only scope (§6) and the R3 mitigation.  So
H2 builds the **gradient-only** form (first derivatives, **no mixed partial** —
the §2 fallback): the interpolant carries one derivative accumulator per enhanced
axis and drops the ``ĥ·(mixed)`` cross-term.  This is:

  * **exact** for ≤1 enhanced axis (all four H2 acceptance gates use ≤1 enhanced
    axis — the productized Phase-0 Q1 rate win is a single-enhanced-axis sweep);
  * **node-exact for values and for the tangent along each enhanced axis** for any
    number of enhanced axes (``h_i(θ_j)=δ_ij``, ``ĥ_i(θ_j)=0``);
  * a documented first-order approximation of the *off-node cross-curvature* for
    ≥2 simultaneously-enhanced axes.

Both value-only and single-enhanced-axis reductions are **bit-for-bit** (same
float ops, same reused primitives) — the H2 reduce-to-committed gates.

**Add-only.**  Reuses ``parametric_nd`` (``tensor_param_nodes``/``snake_order`` and
the persistence helpers) and ``hermite`` (``cardinal_deriv_at_nodes``,
``_hermite_bases_np``/``_hermite_bases_jax``) **verbatim**; never edits a committed
module.  Certification is unchanged — the Hermite object is only a *guess*;
``evaluate_polished`` reuses the committed ``solve_fn`` → ``newton_solve``.

Standalone: numpy + jax + the sibling ``parametric``/``parametric_nd``/``hermite``
modules.  The ABT two-centre wiring (``θ=(q,b,χ_A,χ_B)`` + per-axis certified
tangents) lives in :func:`from_problem_nd_hermite`, which imports
``parametric_nd_2c`` and ``applications.sensitivity`` (reused verbatim).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric_nd import (              # reused verbatim (the N-D primitives + IO)
    tensor_param_nodes,
    snake_order,
    FORMAT_VERSION,
    _pack_meta,
    _unpack_meta,
    _git_commit,
    _load_npz,
    _check_meta,
)
from .hermite import (                    # reused verbatim (the H1 1-D primitives)
    cardinal_deriv_at_nodes,
    _hermite_bases_np,
    _hermite_bases_jax,
)


# --------------------------------------------------------------------------
# N-D gradient-enhanced (Hermite) parametric solution container
# --------------------------------------------------------------------------
@dataclass
class HermiteSolutionND:
    """Value-only axes ⊗ Hermite-enhanced axes (§2/§4 H2).

    Mirrors :class:`parametric_nd.ParametricSolutionND` (``evaluate``,
    ``evaluate_jax``, ``evaluate_polished``, ``save``) but carries, per node, the
    value ``U_i`` **and** the full per-axis certified tangent stack ``dU/dθ_k|_i``
    (all ``d`` axes; only the ``enhanced`` axes' tangents enter the interpolant).

    Fields
    ------
    axes : ``[(p_min, p_max, Q), ...]`` — one tuple per active dimension.
    nodes, weights : per-axis CGL nodes (descending value) and barycentric weights.
    U_nodes : ``(n_0, …, n_{d-1}, *field_shape)`` — converged values.
    dU_nodes : ``(n_0, …, n_{d-1}, d, *field_shape)`` — certified tangents
        ``dU/dθ_k`` (the axis-``k`` slice is ``dU/dθ_k`` at every node).
    cvec : list of per-axis cardinal-derivative vectors ``c_i`` (node-set only).
    enhanced : tuple of **axis indices** carrying the Hermite (gradient) enhancement.
    iters, residuals : per-node Newton provenance (aligned).
    """

    axes: List[Tuple[float, float, int]]
    nodes: List[np.ndarray]
    weights: List[np.ndarray]
    U_nodes: np.ndarray
    dU_nodes: np.ndarray
    cvec: List[np.ndarray]
    enhanced: Tuple[int, ...]
    iters: np.ndarray
    residuals: np.ndarray
    _solve_fn: Callable = field(repr=False, default=None)

    @property
    def d(self) -> int:
        return len(self.axes)

    @property
    def field_shape(self):
        return self.U_nodes.shape[self.d:]

    @property
    def n_nodes(self) -> int:
        return int(np.prod([len(n) for n in self.nodes]))

    # ----- tensor-product gradient-enhanced interpolant (numpy, node-safe) -----
    def evaluate(self, theta):
        """``ũ(θ)`` by successive 1-D contraction — value-only axes barycentric,
        enhanced axes Hermite (gradient-only for ≥2 enhanced axes).  ``θ`` is a
        length-``d`` vector (single query point).

        Reduces **bit-for-bit** to
        :meth:`parametric_nd.ParametricSolutionND.evaluate` when ``enhanced`` is
        empty, and to :meth:`hermite.HermiteSolution1D.evaluate` for ``d == 1``
        with axis 0 enhanced (identical float ops, reused primitives)."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if theta.shape[0] != self.d:
            raise ValueError(f"theta has {theta.shape[0]} comps; expected d={self.d}")
        enh = set(int(e) for e in self.enhanced)
        accV = self.U_nodes
        # one derivative accumulator per enhanced axis: dU/dθ_e contracted through
        # the axes processed so far (starts as the stored axis-e tangent tensor).
        accD = {e: np.take(self.dU_nodes, e, axis=self.d) for e in enh}
        for k in range(self.d):
            nodes_k = self.nodes[k]
            diff = theta[k] - nodes_k
            hit = np.isclose(diff, 0.0, atol=1e-13)
            if np.any(hit):                                   # exact-node guard (per axis)
                i = int(np.argmax(hit))
                accV = accV[i]
                accD = {e: accD[e][i] for e in accD if e != k}
            elif k in enh:
                h, hh, _, _ = _hermite_bases_np(theta[k], nodes_k,
                                                self.weights[k], self.cvec[k])
                newV = (np.tensordot(h, accV, axes=(0, 0))
                        + np.tensordot(hh, accD[k], axes=(0, 0)))
                # gradient-only: the value cardinal h acts on the OTHER enhanced
                # axes' derivative accumulators; the ĥ·(mixed) cross-term is dropped.
                accD = {e: np.tensordot(h, accD[e], axes=(0, 0))
                        for e in accD if e != k}
                accV = newV
            else:                                             # value-only barycentric
                t = self.weights[k] / diff
                s = t.sum()
                accV = np.tensordot(t, accV, axes=(0, 0)) / s
                accD = {e: np.tensordot(t, accD[e], axes=(0, 0)) / s for e in accD}
        return accV

    # ----- JAX-differentiable interpolant (branchless, off-node; grad hook) -----
    def evaluate_jax(self, theta):
        """``jnp`` twin of :meth:`evaluate` — branchless and differentiable in the
        parameter vector ``θ`` (the exposed-gradient hook for the applications:
        ``jax.jacfwd(evaluate_jax)`` is ``∂U/∂θ`` of the surrogate).  Must NOT be
        queried exactly at a node (the barycentric quotient is 0/0 there).

        Reduces bit-for-bit to the committed ``ParametricSolutionND.evaluate_jax``
        (no enhanced axes) / ``HermiteSolution1D.evaluate_jax`` (``d==1``, axis 0
        enhanced)."""
        theta = jnp.asarray(theta)
        enh = set(int(e) for e in self.enhanced)
        accV = jnp.asarray(self.U_nodes)
        dU = jnp.asarray(self.dU_nodes)
        accD = {e: jnp.take(dU, e, axis=self.d) for e in enh}
        for k in range(self.d):
            nodes_k = jnp.asarray(self.nodes[k])
            diff = theta[k] - nodes_k
            if k in enh:
                h, hh = _hermite_bases_jax(theta[k], nodes_k,
                                           jnp.asarray(self.weights[k]),
                                           jnp.asarray(self.cvec[k]))
                newV = (jnp.tensordot(h, accV, axes=(0, 0))
                        + jnp.tensordot(hh, accD[k], axes=(0, 0)))
                accD = {e: jnp.tensordot(h, accD[e], axes=(0, 0))
                        for e in accD if e != k}
                accV = newV
            else:
                t = jnp.asarray(self.weights[k]) / diff
                s = jnp.sum(t)
                accV = jnp.tensordot(t, accV, axes=(0, 0)) / s
                accD = {e: jnp.tensordot(t, accD[e], axes=(0, 0)) / s for e in accD}
        return accV

    # ----- certified evaluation (unchanged; reuses the attached solve_fn) -----
    def evaluate_polished(self, theta, newton_steps: int = 2, tol: float = 1e-12):
        """Hermite prediction + 1–2 Newton steps → certified ``‖R‖≤tol`` at ``θ``.

        The Hermite object is only a *guess*; certification is unchanged from the
        committed path (the attached ``solve_fn`` → ``newton_solve``).  Returns
        ``(U, info)`` with ``info.residual_norm`` the certified constraint
        residual, independent of any interpolation error."""
        if self._solve_fn is None:
            raise RuntimeError(
                "no solve_fn attached; build via HermiteSolverND / from_problem_nd_hermite")
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(np.asarray(theta, dtype=float), guess, tol, newton_steps)
        return U, info

    # ----- persistence (numpy-only .npz; reuses the parametric_nd helpers) -----
    def save(self, path, *, meta=None):
        """Persist to a single ``.npz`` (numpy-only, no pickle).  Round-trips
        bit-for-bit via :func:`load_hermite_nd`.  The reloaded object is a
        standalone predictor: ``evaluate``/``evaluate_jax`` need only numpy/jax +
        the parametric modules (``evaluate_polished`` needs a reattached
        ``solve_fn``)."""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        d = self.d
        arrays = {}
        for k in range(d):
            arrays[f"nodes_{k}"] = np.asarray(self.nodes[k], dtype=float)
            arrays[f"weights_{k}"] = np.asarray(self.weights[k], dtype=float)
            arrays[f"cvec_{k}"] = np.asarray(self.cvec[k], dtype=float)
        arrays["U_nodes"] = np.asarray(self.U_nodes, dtype=float)
        arrays["dU_nodes"] = np.asarray(self.dU_nodes, dtype=float)
        arrays["iters"] = np.asarray(self.iters, dtype=np.int64)
        arrays["residuals"] = np.asarray(self.residuals, dtype=float)
        arrays["axes"] = np.array([[float(lo), float(hi), float(Q)]
                                   for (lo, hi, Q) in self.axes], dtype=float)
        arrays["enhanced"] = np.asarray(sorted(int(e) for e in self.enhanced),
                                        dtype=np.int64)
        arrays["field_shape"] = np.asarray(self.field_shape, dtype=np.int64)
        full_meta = {"d": int(d), "git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION      # authoritative
        full_meta["kind"] = "hermite_nd"
        arrays["meta_json"] = _pack_meta(full_meta)
        np.savez(path, **arrays)
        return path


def load_hermite_nd(path) -> "HermiteSolutionND":
    """Load a :class:`HermiteSolutionND` saved by :meth:`HermiteSolutionND.save`.

    Reconstructs with ``_solve_fn=None`` (``evaluate``/``evaluate_jax`` work
    immediately; ``evaluate_polished`` raises until a solver is attached).  The
    parsed metadata is stored on the returned object as ``.meta``.
    """
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "hermite_nd")
    try:
        d = int(meta["d"])
        nodes = [np.asarray(data[f"nodes_{k}"], dtype=float) for k in range(d)]
        weights = [np.asarray(data[f"weights_{k}"], dtype=float) for k in range(d)]
        cvec = [np.asarray(data[f"cvec_{k}"], dtype=float) for k in range(d)]
        axes = [(float(a[0]), float(a[1]), int(round(float(a[2]))))
                for a in np.asarray(data["axes"], dtype=float)]
        enhanced = tuple(int(e) for e in np.asarray(data["enhanced"], dtype=np.int64))
        U_nodes = np.asarray(data["U_nodes"], dtype=float)
        dU_nodes = np.asarray(data["dU_nodes"], dtype=float)
        iters = np.asarray(data["iters"])
        residuals = np.asarray(data["residuals"], dtype=float)
        sol = HermiteSolutionND(axes=axes, nodes=nodes, weights=weights,
                                U_nodes=U_nodes, dU_nodes=dU_nodes, cvec=cvec,
                                enhanced=enhanced, iters=iters, residuals=residuals,
                                _solve_fn=None)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt LM-initial-data hermite_nd surrogate '{path}': {e}")
    sol.meta = meta
    return sol


# --------------------------------------------------------------------------
# Driver: snake march (reused) + store the full per-node tangent stack
# --------------------------------------------------------------------------
class HermiteSolverND:
    """Drives the boustrophedon continuation sweep (``parametric_nd.snake_order`` /
    ``tensor_param_nodes``, reused verbatim) and builds the N-D Hermite interpolant,
    **storing the full per-node tangent stack** the ``tangent_fn`` returns.

    Parameters
    ----------
    solve_fn : ``solve_fn(theta_vec, guess, tol, max_iter) -> (U, info)`` — the
        committed Newton-from-warm-start (reused verbatim from the wiring).
    axes : ``[(p_min, p_max, Q), ...]`` — the active axis boxes.
    tangent_fn : ``tangent_fn(theta_vec, U) -> (d, *field_shape)`` — the certified
        per-axis tangent stack at ``(θ, U)`` (composed from
        ``sensitivity.certified_tangent`` per active axis in
        :func:`from_problem_nd_hermite`).
    enhanced_axes : indices of the Hermite-enhanced axes (default: none →
        value-only, reduces bit-for-bit to ``ParametricSolverND``'s output).
    """

    def __init__(self, solve_fn: Callable,
                 axes: Sequence[Tuple[float, float, int]],
                 tangent_fn: Callable,
                 enhanced_axes: Sequence[int] = ()):
        self.solve_fn = solve_fn
        self.axes = [tuple(a) for a in axes]
        self.d = len(self.axes)
        self.tangent_fn = tangent_fn
        self.enhanced = tuple(sorted(int(e) for e in enhanced_axes))

    def build(self, tol: float = 1e-12, max_iter: int = 20,
              use_tangent: bool = True) -> HermiteSolutionND:
        """Snake-march the tensor grid; at each converged node store the value and
        the full ``(d, *field)`` certified tangent stack.  The tangent is computed
        once per node (``d`` back-solves against the node's Jacobian, per §0) and
        reused as the predictor for the next (Hamming-adjacent) node."""
        nodes, weights = tensor_param_nodes(self.axes)
        shape = tuple(len(n) for n in nodes)
        order = snake_order(shape)

        U_nodes = None
        dU_nodes = None
        field_shape = None
        iters = np.zeros(shape, dtype=int)
        resids = np.zeros(shape, dtype=float)
        guess = None
        theta_prev = None
        tang_prev = None
        for idx in order:
            theta = np.array([nodes[k][idx[k]] for k in range(self.d)], dtype=float)
            g = guess
            if use_tangent and guess is not None and tang_prev is not None:
                step = theta - theta_prev                      # (d,)
                g = np.asarray(guess) + np.tensordot(step, tang_prev, axes=(0, 0))
            U, info = self.solve_fn(theta, g, tol, max_iter)
            Ua = np.asarray(U)
            tang = np.asarray(self.tangent_fn(theta, Ua))      # (d, *field)
            if U_nodes is None:
                field_shape = Ua.shape
                if tang.shape != (self.d,) + field_shape:
                    raise ValueError(
                        f"tangent_fn returned {tang.shape}; expected {(self.d,) + field_shape}")
                U_nodes = np.empty(shape + field_shape, dtype=float)
                dU_nodes = np.empty(shape + (self.d,) + field_shape, dtype=float)
            U_nodes[idx] = Ua
            dU_nodes[idx] = tang
            iters[idx] = info.iters
            resids[idx] = info.residual_norm
            guess = jnp.asarray(Ua)
            theta_prev = theta
            tang_prev = tang

        cvec = [cardinal_deriv_at_nodes(n) for n in nodes]
        return HermiteSolutionND(
            axes=self.axes, nodes=nodes, weights=weights,
            U_nodes=U_nodes, dU_nodes=dU_nodes, cvec=cvec, enhanced=self.enhanced,
            iters=iters, residuals=resids, _solve_fn=self.solve_fn)


# --------------------------------------------------------------------------
# ABT two-centre wiring: solve_fn (reused) + per-axis certified tangent_fn
# --------------------------------------------------------------------------
def from_problem_nd_hermite(prob, axes: Sequence[dict], enhanced: Sequence[str] = (),
                            M_tot: float = 1.0, fixed=None, use_cache: bool = True):
    """A :class:`HermiteSolverND` over the ABT two-centre solver ``solver_abt``.

    ``axes = [{name,min,max,Q}, ...]`` (subset/ordering of
    ``parametric_nd_2c.AXIS_NAMES``); ``enhanced`` lists the **names** of the
    Hermite-enhanced axes (e.g. ``["b", "chi_B"]`` — the §4/Phase-0 hard axes).

    The ``solve_fn`` is reused **verbatim** from
    ``parametric_nd_2c.from_problem_nd`` (its Newton-from-warm-start closure); the
    ``tangent_fn`` composes ``applications.sensitivity.certified_tangent`` over the
    active axes (the IFT tangent ``J·dU/dθ=−∂R/∂θ``, reused verbatim) into the full
    ``(d, *field)`` stack the builder stores.  Per §0/Phase-0 the certified tangent
    on the committed **modified-Newton** nodes is floor-insensitive (≈2e-14), so
    **no NK build is needed**.

    Add-only: imports ``parametric_nd_2c`` and ``applications.sensitivity`` (both
    reused verbatim); defines no new physics.
    """
    from . import parametric_nd_2c as p3
    from ..applications import sensitivity as sen

    active_names = [a["name"] for a in axes]
    name_to_idx = {n: i for i, n in enumerate(active_names)}
    enhanced_idx = [name_to_idx[n] for n in enhanced]

    # solve_fn reused verbatim (the committed ParametricSolverND wiring)
    solver_nd = p3.from_problem_nd(prob, axes, M_tot=M_tot, fixed=fixed,
                                   use_cache=use_cache)
    solve_fn = solver_nd.solve_fn

    def tangent_fn(theta_vec, U):
        """Full per-axis certified tangent stack ``(d, *field)`` at ``(θ, U)``.

        One assembly is shared across the ``d`` axes (the geometry is common at a
        node); each ``certified_tangent`` back-solves against the node Jacobian."""
        sl = p3.theta_to_slice(theta_vec, active_names, M_tot, fixed)
        asm = sen.sa.assemble(prob, sl)                    # shared node assembly
        stack = [np.asarray(sen.certified_tangent(prob, U, sl, name, M_tot, asm=asm))
                 for name in active_names]
        return np.stack(stack, axis=0)

    spec = [(a["min"], a["max"], a["Q"]) for a in axes]
    return HermiteSolverND(solve_fn, spec, tangent_fn, enhanced_axes=enhanced_idx)
