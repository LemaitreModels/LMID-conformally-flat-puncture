"""PARASOL — gradient-enhanced (Hermite) SPARSE-grid collocation (H5b).

The sparse (Smolyak) sibling of the dense gradient-enhanced layer
(:mod:`hermite_nd`), and the gradient-enhanced sibling of the committed
value-only Smolyak layer (:mod:`parametric_nd_smolyak`).  H3 proved the
value-only Hermite interpolant telescopes into the combination technique
**bit-for-bit**; H5a delivered the missing ``solver_3d`` certified tangent
(:func:`applications.sensitivity_3d.certified_tangent_3d`).  H5b is the sparse
plumbing that rolls the *gradient* enhancement onto the sparse path, changed in
exactly the three places called out in ``GRADIENT_ENHANCED_PLAN.md`` §4 H5b:

  (i)   the node pool stores the per-node **tangent stack** ``(U, dU, iters,
        resid)`` (``dU`` is the full ``(d, *field)`` certified tangent from the
        ``tangent_fn``, one shared assembly + ``d`` back-solves per node, per §0);
  (ii)  each subgrid is assembled as a :class:`hermite_nd.HermiteSolutionND`
        (Hermite-enhanced on the hard axes) instead of a
        :class:`parametric_nd.ParametricSolutionND`;
  (iii) ``evaluate = Σ_l c_l·sub_l.evaluate(θ)`` is UNCHANGED (same signature,
        same combination coefficients — H3 proved the value-only limit is
        bit-for-bit ``SmolyakSolutionND``).

**The level-0 decision (H3 blocker (ii) / R7 — committed default).**  A fixed
per-axis 1-D operator sequence ``{I_l}`` keeps the combination telescoping
consistent.  For an enhanced axis: ``I_0`` is **value-only** (the level-0 single
*midpoint* node, a constant factor — one node cannot Hermite-interpolate, and
enhancing it would inject the fragile 1-node Taylor of R4); ``I_l`` for ``l ≥ 1``
is **Hermite** on the genuine ``≥3``-node CGL factor.  So a subgrid ``l``
Hermite-enhances axis ``k`` iff ``k`` is globally enhanced **and** ``l_k ≥ 1`` —
:meth:`HermiteSmolyakSolverND._subgrid_enhanced`.  With no globally-enhanced axis
every subgrid is value-only and the whole object reduces bit-for-bit to
:class:`parametric_nd_smolyak.SmolyakSolutionND`.

**Add-only.**  Reuses the committed Smolyak primitives (``nested_levels``,
``isotropic_index_set``/``anisotropic_index_set``, ``combination_coeffs``,
``_node_key``, ``_assert_downward_closed``), ``hermite_nd.HermiteSolutionND``,
``hermite.cardinal_deriv_at_nodes``, ``parametric_nd.snake_order`` and the
persistence helpers **verbatim**; never edits a committed module.  Certification
is unchanged — the Hermite-Smolyak object is only a *guess*;
``evaluate_polished`` reuses the committed ``solve_fn`` → ``newton_solve``.

Standalone: numpy + jax + the sibling ``parametric`` modules and (for the 3-D
wiring) ``applications.sensitivity_3d``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric_nd import snake_order, FORMAT_VERSION, _pack_meta, _git_commit
from .parametric_nd_smolyak import (          # committed sparse primitives (verbatim)
    nested_levels,
    isotropic_index_set,
    anisotropic_index_set,
    combination_coeffs,
    _node_key,
    _assert_downward_closed,
)
from .hermite_nd import HermiteSolutionND     # the H2 gradient-enhanced subgrid
from .hermite import cardinal_deriv_at_nodes  # node-set cardinal-derivative vector


# --------------------------------------------------------------------------
# Sparse gradient-enhanced solution container (combination of Hermite subgrids)
# --------------------------------------------------------------------------
@dataclass
class HermiteSmolyakSolutionND:
    """Combination-technique sparse interpolant with Hermite-enhanced subgrids:
    ``Σ_i c_i · subgrid_i.evaluate(θ)`` where each ``subgrid`` is a
    :class:`hermite_nd.HermiteSolutionND` (value-only on the easy axes,
    Hermite-enhanced on the globally-enhanced hard axes that carry ``l_k ≥ 1``).

    Mirrors :class:`parametric_nd_smolyak.SmolyakSolutionND` field-for-field; the
    only additions are :attr:`enhanced` (the GLOBAL enhanced-axis indices) and the
    subgrids' stored per-node tangent stacks.
    """
    axes: List[Tuple[float, float]]        # [(p_min, p_max), ...] (no Q — levels)
    index_set: List[tuple]                 # the admissible multi-index set (downward-closed)
    coeffs: List[int]                      # combination coefficient per kept subgrid
    subgrids: List[HermiteSolutionND]      # the kept (nonzero-coeff) Hermite subgrids
    enhanced: Tuple[int, ...]              # GLOBAL Hermite-enhanced axis indices
    n_solver_nodes: int                    # unique solver calls (sparse-grid node count)
    total_iters: int                       # total Newton iters over the node pool
    _solve_fn: Callable = field(repr=False, default=None)

    @property
    def d(self) -> int:
        return len(self.axes)

    @property
    def field_shape(self):
        return self.subgrids[0].field_shape

    @property
    def n_nodes(self) -> int:
        return self.n_solver_nodes

    # ----- combination-technique interpolant (identical signature to Smolyak) -----
    def evaluate(self, theta):
        """``U(θ) = Σ_i c_i · subgrid_i.evaluate(θ)``.  ``θ`` is a length-d vector.

        Bit-for-bit :meth:`parametric_nd_smolyak.SmolyakSolutionND.evaluate` when
        ``enhanced`` is empty (each subgrid then reduces to the barycentric
        contraction, H3/H2)."""
        out = None
        for c, sub in zip(self.coeffs, self.subgrids):
            v = sub.evaluate(theta)
            out = c * v if out is None else out + c * v
        return out

    # ----- JAX-differentiable twin (exposed-gradient hook; not at a node) -----
    def evaluate_jax(self, theta):
        out = None
        for c, sub in zip(self.coeffs, self.subgrids):
            v = sub.evaluate_jax(theta)
            out = c * v if out is None else out + c * v
        return out

    # ----- certified evaluation (the "cannot be silently wrong" gate) -----
    def evaluate_polished(self, theta, newton_steps: int = 2, tol: float = 1e-12):
        """Sparse Hermite prediction + 1–2 Newton steps → certified ‖R‖≤tol at θ.

        Unchanged from the committed path: the sparse Hermite object is only a
        *guess*; the attached ``solve_fn`` → ``newton_solve`` is the certificate."""
        if self._solve_fn is None:
            raise RuntimeError("no solve_fn attached; build via HermiteSmolyakSolverND / "
                               "from_problem_hermite_smolyak_3d")
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(np.asarray(theta, dtype=float), guess, tol, newton_steps)
        return U, info

    # ----- persistence: store the DEDUPLICATED node pool (value + tangent) -----
    def _dedup_pool(self) -> Dict[tuple, tuple]:
        """Reconstruct ``key -> (theta, U, dU, iters, resid)`` from the kept subgrids.

        As in the value-only Smolyak, the union of the kept (nonzero-coeff)
        subgrids' nodes is exactly the solver-node pool (nested CC levels), so the
        stored artifact is the deduplicated pool (~``n_solver_nodes`` value +
        ``d`` tangent fields each), not the overlapping per-subgrid tensors."""
        pool: Dict[tuple, tuple] = {}
        for sub in self.subgrids:
            nodes = sub.nodes
            shape = tuple(len(n) for n in nodes)
            for idx in np.ndindex(*shape):
                theta = np.array([nodes[k][idx[k]] for k in range(self.d)], dtype=float)
                key = _node_key(theta)
                if key in pool:
                    continue
                pool[key] = (theta,
                             np.asarray(sub.U_nodes[idx], dtype=float),
                             np.asarray(sub.dU_nodes[idx], dtype=float),   # (d, *field)
                             int(sub.iters[idx]), float(sub.residuals[idx]))
        return pool

    def save(self, path, *, meta=None):
        """Persist to a single ``.npz`` (numpy-only, no pickle).  Round-trips
        bit-for-bit via :func:`load_hermite_smolyak`.

        Stores the **deduplicated** node pool — ``node_thetas`` (N×d), ``node_U``
        (N×\\*fs), ``node_dU`` (N×d×\\*fs), ``node_iters`` (N), ``node_resids`` (N)
        — plus ``index_set`` (M×d), ``axes`` (d×2), ``enhanced`` (global indices),
        ``field_shape``, and ``meta_json``.  The combination coefficients are
        recomputed on load (``combination_coeffs``)."""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        pool = self._dedup_pool()
        keys = list(pool)
        node_thetas = np.array([pool[k][0] for k in keys], dtype=float)     # N×d
        node_U = np.array([pool[k][1] for k in keys], dtype=float)          # N×*fs
        node_dU = np.array([pool[k][2] for k in keys], dtype=float)         # N×d×*fs
        node_iters = np.array([pool[k][3] for k in keys], dtype=np.int64)   # N
        node_resids = np.array([pool[k][4] for k in keys], dtype=float)     # N
        index_set = np.array([[int(x) for x in l] for l in self.index_set], dtype=np.int64)
        axes = np.array([[float(lo), float(hi)] for (lo, hi) in self.axes], dtype=float)
        full_meta = {"d": int(self.d), "n_solver_nodes": int(self.n_solver_nodes),
                     "git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION      # authoritative
        full_meta["kind"] = "hermite_smolyak"
        np.savez(path,
                 node_thetas=node_thetas, node_U=node_U, node_dU=node_dU,
                 node_iters=node_iters, node_resids=node_resids,
                 index_set=index_set, axes=axes,
                 enhanced=np.asarray(sorted(int(e) for e in self.enhanced), dtype=np.int64),
                 field_shape=np.asarray(self.field_shape, dtype=np.int64),
                 meta_json=_pack_meta(full_meta))
        return path


# --------------------------------------------------------------------------
# Builder: solve the shared node pool once (value + tangent), assemble subgrids
# --------------------------------------------------------------------------
class HermiteSmolyakSolverND:
    """Drives the sparse-grid continuation sweep and builds the gradient-enhanced
    combination interpolant.

    Mirrors :class:`parametric_nd_smolyak.SmolyakSolverND`; the differences are the
    node pool storing the tangent stack (needs a ``tangent_fn``) and the subgrids
    being :class:`hermite_nd.HermiteSolutionND` (enhanced per the level-0 rule).

    Parameters
    ----------
    solve_fn : ``solve_fn(theta_vec, guess, tol, max_iter) -> (U, info)`` (the same
        contract as the committed layers).
    axes : ``[(p_min, p_max), ...]`` (no Q — Smolyak uses doubling levels).
    tangent_fn : ``tangent_fn(theta_vec, U) -> (d, *field_shape)`` — the certified
        per-axis tangent stack (composed from ``sensitivity_3d.certified_tangent_3d``
        in :func:`from_problem_hermite_smolyak_3d`).
    enhanced_axes : GLOBAL indices of the Hermite-enhanced axes (default: none →
        value-only, reduces bit-for-bit to ``SmolyakSolverND``'s output).
    """

    def __init__(self, solve_fn: Callable, axes: Sequence[Tuple[float, float]],
                 tangent_fn: Optional[Callable] = None,
                 enhanced_axes: Sequence[int] = ()):
        self.solve_fn = solve_fn
        self.axes = [tuple(a) for a in axes]
        self.d = len(self.axes)
        self.tangent_fn = tangent_fn
        self.enhanced = tuple(sorted(int(e) for e in enhanced_axes))

    # ----- nested per-axis nodes/weights for one multi-index -----
    def _subgrid_nodes(self, l: Sequence[int]):
        nodes, weights = [], []
        for k, (lo, hi) in enumerate(self.axes):
            n, w = nested_levels(lo, hi, int(l[k]))
            nodes.append(n)
            weights.append(w)
        return nodes, weights

    # ----- the level-0 decision: enhance axis k in subgrid l iff l_k >= 1 -----
    def _subgrid_enhanced(self, l: Sequence[int]) -> Tuple[int, ...]:
        return tuple(k for k in self.enhanced if int(l[k]) >= 1)

    # ----- solve every UNIQUE node of a set of subgrids, warm-started -----
    def _solve_pool(self, index_set, tol, max_iter, pool=None, guess0=None, verbose=False):
        """Solve each unique node across the subgrids in ``index_set`` exactly once,
        storing ``key -> (U, dU, iters, resid)`` (``dU`` = the ``(d,*field)``
        certified tangent stack from ``tangent_fn``).

        Coarse→fine (by ``|l|₁``), snake-march within each subgrid; shared nodes
        (the nesting payoff) are solved once and their tangent computed once."""
        pool = {} if pool is None else pool
        guess = guess0
        for l in sorted(index_set, key=lambda t: sum(t)):
            nodes, _ = self._subgrid_nodes(l)
            shape = tuple(len(n) for n in nodes)
            for idx in snake_order(shape):
                theta = np.array([nodes[k][idx[k]] for k in range(self.d)], dtype=float)
                key = _node_key(theta)
                if key in pool:
                    guess = jnp.asarray(pool[key][0])
                    continue
                U, info = self.solve_fn(theta, guess, tol, max_iter)
                Ua = np.asarray(U)
                if self.tangent_fn is not None:
                    dU = np.asarray(self.tangent_fn(theta, Ua))          # (d, *field)
                    if dU.shape != (self.d,) + Ua.shape:
                        raise ValueError(
                            f"tangent_fn returned {dU.shape}; expected "
                            f"{(self.d,) + Ua.shape}")
                else:
                    dU = np.zeros((self.d,) + Ua.shape)
                pool[key] = (Ua, dU, int(info.iters), float(info.residual_norm))
                guess = jnp.asarray(Ua)
        if verbose:
            print(f"[hermite-smolyak] pool: {len(pool)} unique nodes")
        return pool

    # ----- assemble one subgrid's HermiteSolutionND from the solved pool -----
    def _assemble_subgrid(self, l, pool) -> HermiteSolutionND:
        nodes, weights = self._subgrid_nodes(l)
        shape = tuple(len(n) for n in nodes)
        field_shape = pool[next(iter(pool))][0].shape
        U_nodes = np.empty(shape + field_shape, dtype=float)
        dU_nodes = np.empty(shape + (self.d,) + field_shape, dtype=float)
        iters = np.zeros(shape, dtype=int)
        resids = np.zeros(shape, dtype=float)
        for idx in np.ndindex(*shape):
            key = _node_key([nodes[k][idx[k]] for k in range(self.d)])
            Ua, dU, it, rs = pool[key]
            U_nodes[idx] = Ua
            dU_nodes[idx] = dU
            iters[idx] = it
            resids[idx] = rs
        axes_meta = [(lo, hi, len(nodes[k]) - 1) for k, (lo, hi) in enumerate(self.axes)]
        cvec = [cardinal_deriv_at_nodes(n) for n in nodes]
        return HermiteSolutionND(
            axes=axes_meta, nodes=nodes, weights=weights, U_nodes=U_nodes,
            dU_nodes=dU_nodes, cvec=cvec, enhanced=self._subgrid_enhanced(l),
            iters=iters, residuals=resids, _solve_fn=None)

    # ----- assemble the combination interpolant from a set + a solved pool -----
    def _finalize(self, index_set, pool) -> HermiteSmolyakSolutionND:
        coeffs = combination_coeffs(index_set)
        kept = sorted(coeffs)                          # deterministic order
        subgrids = [self._assemble_subgrid(l, pool) for l in kept]
        total_iters = int(sum(v[2] for v in pool.values()))
        return HermiteSmolyakSolutionND(
            axes=self.axes, index_set=[tuple(l) for l in index_set],
            coeffs=[coeffs[l] for l in kept], subgrids=subgrids,
            enhanced=self.enhanced, n_solver_nodes=len(pool),
            total_iters=total_iters, _solve_fn=self.solve_fn)

    # ----- public builders -----------------------------------------------
    def build_isotropic(self, level: int, tol: float = 1e-12, max_iter: int = 20,
                         verbose: bool = False) -> HermiteSmolyakSolutionND:
        """Classic isotropic Smolyak at total level ``level`` (``|l|₁ ≤ level``);
        the unique-node count is ``parametric_nd_2c.smolyak_points(d, level)``."""
        index_set = isotropic_index_set(self.d, level)
        pool = self._solve_pool(index_set, tol, max_iter, verbose=verbose)
        return self._finalize(index_set, pool)

    def build_anisotropic(self, level: float, weights: Optional[Sequence[float]] = None,
                          caps: Optional[Sequence[int]] = None, tol: float = 1e-12,
                          max_iter: int = 20, verbose: bool = False) -> HermiteSmolyakSolutionND:
        """Weighted-simplex sparse grid ``Σ_k w_k l_k ≤ level`` (optional per-axis
        ``caps``) — the cheap static way to spend levels on the hard axes."""
        index_set = anisotropic_index_set(self.d, level, weights=weights, caps=caps)
        pool = self._solve_pool(index_set, tol, max_iter, verbose=verbose)
        return self._finalize(index_set, pool)

    def build_from_index_set(self, index_set: Sequence[tuple], tol: float = 1e-12,
                             max_iter: int = 20, verbose: bool = False) -> HermiteSmolyakSolutionND:
        """Build from an arbitrary **downward-closed** multi-index set (escape hatch)."""
        index_set = [tuple(int(x) for x in l) for l in index_set]
        _assert_downward_closed(index_set)
        pool = self._solve_pool(index_set, tol, max_iter, verbose=verbose)
        return self._finalize(index_set, pool)


# --------------------------------------------------------------------------
# Wiring: build a HermiteSmolyakSolverND around the 3-D non-axisymmetric solver
# --------------------------------------------------------------------------
def from_problem_hermite_smolyak_3d(prob, axes: Sequence[dict], enhanced: Sequence[str] = (),
                                    M_tot: float = 1.0, fixed: Optional[Dict[str, float]] = None,
                                    use_cache: bool = True, solver: str = "nk",
                                    gmres_rtol: float = 1e-4, tangent_jac: str = "nk",
                                    tangent_fn: Optional[Callable] = None
                                    ) -> HermiteSmolyakSolverND:
    """A :class:`HermiteSmolyakSolverND` over the 3-D non-axisymmetric ``solver_3d``.

    ``axes = [{name,min,max}, ...]`` (NO Q — Smolyak uses doubling *levels*; subset/
    ordering of ``parametric_nd_3d.AXIS_NAMES_3D``); ``enhanced`` lists the **names**
    of the Hermite-enhanced axes (e.g. ``["b", "S_x"]`` — the hard axes).

    The ``solve_fn`` is reused **verbatim** from
    ``parametric_nd_3d.make_solve_fn`` (its Newton-from-warm-start closure + D7
    per-b cache).  The per-node tangent stack comes from ``tangent_fn(θ, U) →
    (d, *field)`` — pass one explicitly to use a different tangent (e.g. the
    **quasi-circular** chain-rule tangent for the QC family); when ``tangent_fn``
    is ``None`` the default composes ``applications.sensitivity_3d.certified_tangent_3d``
    over the active axes (the H5a IFT tangent ``J·dU/dθ=−∂R/∂θ``, one shared
    per-slice assembly across the ``d`` axes).  ``tangent_jac='nk'`` (default) is the
    accurate full-J tangent solve (a genuinely non-axisymmetric slice needs it — the
    block-diagonal ``'modified'`` route drops the φ-mode-coupling).

    **The default tangent is the DIRECT (fixed-physical-momentum) interpretation**
    that matches ``certified_tangent_3d``'s signature.  For the **QC family**
    (``fixed={"qc": 1.0}``) the physical momenta depend on ``(b, masses, spins)``
    via ``quasicircular.qc_momenta``, so the tangent must add that chain rule — the
    default here would be silently wrong.  So the default path **raises** if the QC
    flag is set with no explicit ``tangent_fn``; pass the QC chain-rule tangent
    (``sensitivity_3d_qc``) instead.

    Add-only: imports ``parametric_nd_3d``, ``solver_3d``, and
    ``applications.sensitivity_3d`` (all reused verbatim); defines no new physics.
    """
    from .parametric_nd_3d import make_solve_fn, theta_to_slice3d
    from ..solver import solver_3d as s3
    from ..applications import sensitivity_3d as s3d

    active_names = [a["name"] for a in axes]
    name_to_idx = {n: i for i, n in enumerate(active_names)}
    for n in enhanced:
        if n not in name_to_idx:
            raise ValueError(f"enhanced axis {n!r} not among active axes {active_names}")
    enhanced_idx = [name_to_idx[n] for n in enhanced]

    solve_fn, _ = make_solve_fn(prob, active_names, M_tot=M_tot, fixed=fixed,
                                use_cache=use_cache, solver=solver, gmres_rtol=gmres_rtol)

    if tangent_fn is None:
        if fixed is not None and fixed.get("qc", 0.0):
            raise ValueError(
                "QC family (fixed={'qc':...}) needs the qc-momenta chain-rule "
                "tangent; pass tangent_fn= (e.g. from applications.sensitivity_3d_qc) "
                "— the default direct tangent would be silently wrong for QC.")

        def tangent_fn(theta_vec, U):
            """Full per-axis certified tangent stack ``(d, *field)`` at ``(θ, U)``.

            One ``s3.assemble`` is shared across the ``d`` axes (the geometry is
            common at a node); each ``certified_tangent_3d`` back-solves against it."""
            sl = theta_to_slice3d(theta_vec, active_names, M_tot, fixed)
            asm = s3.assemble(prob, sl)                    # shared node assembly
            stack = [np.asarray(s3d.certified_tangent_3d(prob, U, sl, name, M_tot,
                                                         asm=asm, jac=tangent_jac))
                     for name in active_names]
            return np.stack(stack, axis=0)

    spec = [(a["min"], a["max"]) for a in axes]
    return HermiteSmolyakSolverND(solve_fn, spec, tangent_fn, enhanced_axes=enhanced_idx)


# --------------------------------------------------------------------------
# Persistence: load a sparse gradient-enhanced surrogate saved by .save
# --------------------------------------------------------------------------
def load_hermite_smolyak(path) -> HermiteSmolyakSolutionND:
    """Load a :class:`HermiteSmolyakSolutionND` saved by :meth:`.save`.

    Rebuilds the node pool (value + tangent) by re-keying ``node_thetas``,
    constructs a solver-less ``HermiteSmolyakSolverND(solve_fn=None, …,
    enhanced_axes=…)``, and returns ``solver._finalize(index_set, pool)`` — reusing
    the combination-technique assembly with **zero solves**.  ``evaluate`` /
    ``evaluate_jax`` work immediately; ``evaluate_polished`` raises until a solver
    is attached.  Parsed metadata is stored on the returned object as ``.meta``.
    """
    from .parametric_nd import _load_npz, _unpack_meta, _check_meta
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "hermite_smolyak")
    try:
        node_thetas = np.asarray(data["node_thetas"], dtype=float)     # N×d
        node_U = np.asarray(data["node_U"], dtype=float)               # N×*fs
        node_dU = np.asarray(data["node_dU"], dtype=float)             # N×d×*fs
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
        sol = solver._finalize(index_set, pool)      # combination coeffs recomputed
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt PARASOL hermite_smolyak surrogate '{path}': {e}")
    sol.meta = meta
    return sol
