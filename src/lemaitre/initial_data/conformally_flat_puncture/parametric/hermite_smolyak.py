"""LM-initial-data — gradient-enhanced (Hermite) SPARSE-grid collocation.

The sparse (Smolyak) sibling of the dense gradient-enhanced layer
(:mod:`hermite_nd`), and the gradient-enhanced sibling of the committed
value-only Smolyak layer (:mod:`parametric_nd_smolyak`).  Two earlier results make
it possible: the value-only Hermite interpolant telescopes into the combination
technique **bit-for-bit** (:mod:`hermite_pod`, ``value_only_combination``), and the
``solver_3d`` certified tangent exists
(:func:`applications.sensitivity_3d.certified_tangent_3d`).  This module is the
sparse plumbing that rolls the *gradient* enhancement onto the sparse path, and it
differs from the value-only layer in exactly three places:

  (i)   the node pool stores the per-node **tangent stack** ``(U, dU, iters,
        resid)`` (``dU`` is the full ``(d, *field)`` certified tangent from the
        ``tangent_fn``, one shared assembly + ``d`` back-solves per node);
  (ii)  each subgrid is assembled as a :class:`hermite_nd.HermiteSolutionND`
        (Hermite-enhanced on the hard axes) instead of a
        :class:`parametric_nd.ParametricSolutionND`;
  (iii) ``evaluate = Σ_l c_l·sub_l.evaluate(θ)`` is UNCHANGED (same signature,
        same combination coefficients — the value-only limit is bit-for-bit
        ``SmolyakSolutionND``).

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

**Reuses** the Smolyak primitives (``nested_levels``,
``isotropic_index_set``/``anisotropic_index_set``, ``combination_coeffs``,
``_node_key``, ``_assert_downward_closed``), ``hermite_nd.HermiteSolutionND``,
``hermite.cardinal_deriv_at_nodes``, ``parametric_nd.snake_order`` and the
persistence helpers **verbatim**.  Certification
is unchanged — the Hermite-Smolyak object is only a *guess*;
``evaluate_polished`` reuses the committed ``solve_fn`` → ``newton_solve``.

Standalone: numpy + jax + the sibling ``parametric`` modules and (for the 3-D
wiring) ``applications.sensitivity_3d``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from .parametric_nd import _git_commit, _load_model_npz, _save_npz
from .parametric_nd_smolyak import (
    # the two committed sparse bases this module subclasses, plus the sparse
    # primitives — the primitives are re-exports kept for existing importers
    SmolyakSolutionND,
    SmolyakSolverND,
    nested_levels,                # noqa: F401  (re-export)
    isotropic_index_set,          # noqa: F401  (re-export)
    anisotropic_index_set,        # noqa: F401  (re-export)
    combination_coeffs,           # noqa: F401  (re-export)
    _node_key,
    _assert_downward_closed,      # noqa: F401  (re-export)
)
from .hermite_nd import HermiteSolutionND     # the H2 gradient-enhanced subgrid
from .hermite import cardinal_deriv_at_nodes  # node-set cardinal-derivative vector


# --------------------------------------------------------------------------
# Sparse gradient-enhanced solution container (combination of Hermite subgrids)
# --------------------------------------------------------------------------
@dataclass
class HermiteSmolyakSolutionND(SmolyakSolutionND):
    """Combination-technique sparse interpolant with Hermite-enhanced subgrids:
    ``Σ_i c_i · subgrid_i.evaluate(θ)`` where each ``subgrid`` is a
    :class:`hermite_nd.HermiteSolutionND` (value-only on the easy axes,
    Hermite-enhanced on the globally-enhanced hard axes that carry ``l_k ≥ 1``).

    Subclasses :class:`parametric_nd_smolyak.SmolyakSolutionND`:
    ``evaluate``/``evaluate_jax``/``evaluate_polished`` and the pool machinery
    are inherited **verbatim** (the combination sum is agnostic to what a
    subgrid stores), so the value-only reduction is bit-for-bit by
    construction.  The only additions are :attr:`enhanced` (the GLOBAL
    enhanced-axis indices) and the per-node tangent stack in the pool record.
    """
    #: GLOBAL Hermite-enhanced axis indices.  Dataclass inheritance forces a
    #: default (the parent's ``_solve_fn`` has one); ``()`` = value-only.
    enhanced: Tuple[int, ...] = ()

    _solve_fn_hint = ("build via HermiteSmolyakSolverND / "
                      "from_problem_hermite_smolyak_3d")

    # ----- persistence: store the DEDUPLICATED node pool (value + tangent) -----
    def _pool_record(self, theta, sub, idx) -> tuple:
        """``(theta, U, dU, iters, resid)`` — the gradient-enhanced pool record
        (``dU`` is the ``(d, *field)`` certified tangent stack)."""
        return (theta,
                np.asarray(sub.U_nodes[idx], dtype=float),
                np.asarray(sub.dU_nodes[idx], dtype=float),   # (d, *field)
                int(sub.iters[idx]), float(sub.residuals[idx]))

    def save(self, path, *, meta=None):
        """Persist to a single ``.npz`` (numpy-only, no pickle).  Round-trips
        bit-for-bit via :func:`load_hermite_smolyak`.

        Stores the **deduplicated** node pool — ``node_thetas`` (N×d), ``node_U``
        (N×\\*fs), ``node_dU`` (N×d×\\*fs), ``node_iters`` (N), ``node_resids`` (N)
        — plus ``index_set`` (M×d), ``axes`` (d×2), ``enhanced`` (global indices),
        ``field_shape``, and ``meta_json``.  The combination coefficients are
        recomputed on load (``combination_coeffs``)."""
        pool = self._dedup_pool()
        keys = list(pool)
        arrays = dict(
            node_thetas=np.array([pool[k][0] for k in keys], dtype=float),     # N×d
            node_U=np.array([pool[k][1] for k in keys], dtype=float),          # N×*fs
            node_dU=np.array([pool[k][2] for k in keys], dtype=float),         # N×d×*fs
            node_iters=np.array([pool[k][3] for k in keys], dtype=np.int64),   # N
            node_resids=np.array([pool[k][4] for k in keys], dtype=float),     # N
            index_set=np.array([[int(x) for x in l] for l in self.index_set],
                               dtype=np.int64),
            axes=np.array([[float(lo), float(hi)] for (lo, hi) in self.axes],
                          dtype=float),
            enhanced=np.asarray(sorted(int(e) for e in self.enhanced), dtype=np.int64),
            field_shape=np.asarray(self.field_shape, dtype=np.int64))
        return _save_npz(path, arrays,
                         {"d": int(self.d), "n_solver_nodes": int(self.n_solver_nodes),
                          "git_commit": _git_commit()},
                         meta, kind="hermite_smolyak")


# --------------------------------------------------------------------------
# Builder: solve the shared node pool once (value + tangent), assemble subgrids
# --------------------------------------------------------------------------
class HermiteSmolyakSolverND(SmolyakSolverND):
    """Drives the sparse-grid continuation sweep and builds the gradient-enhanced
    combination interpolant.

    Subclasses :class:`parametric_nd_smolyak.SmolyakSolverND`: the node march
    (``_solve_pool``), the nested-level plumbing and the public builders
    (``build_isotropic``/``build_anisotropic``/``build_from_index_set``) are
    inherited **verbatim**; the overrides are the pool record (the tangent stack
    joins the value, :meth:`_node_record`) and the subgrid/solution assembly
    (:class:`hermite_nd.HermiteSolutionND` subgrids, enhanced per the level-0
    rule).  The adaptive greedy stays value-only — its surplus indicator scores
    plain field values, so :meth:`build_adaptive` raises here.

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

    _log_tag = "hermite-smolyak"

    def __init__(self, solve_fn: Callable, axes: Sequence[Tuple[float, float]],
                 tangent_fn: Optional[Callable] = None,
                 enhanced_axes: Sequence[int] = ()):
        super().__init__(solve_fn, axes)
        self.tangent_fn = tangent_fn
        self.enhanced = tuple(sorted(int(e) for e in enhanced_axes))

    # ----- the level-0 decision: enhance axis k in subgrid l iff l_k >= 1 -----
    def _subgrid_enhanced(self, l: Sequence[int]) -> Tuple[int, ...]:
        return tuple(k for k in self.enhanced if int(l[k]) >= 1)

    # ----- pool record: the tangent stack joins the value -----
    def _node_record(self, theta, Ua, info) -> tuple:
        """``(U, dU, iters, resid)`` — ``dU`` the ``(d, *field)`` certified
        tangent stack from ``tangent_fn``, computed once per unique node (shared
        nodes are the nesting payoff)."""
        if self.tangent_fn is not None:
            dU = np.asarray(self.tangent_fn(theta, Ua))          # (d, *field)
            if dU.shape != (self.d,) + Ua.shape:
                raise ValueError(
                    f"tangent_fn returned {dU.shape}; expected "
                    f"{(self.d,) + Ua.shape}")
        else:
            dU = np.zeros((self.d,) + Ua.shape)
        return (Ua, dU, int(info.iters), float(info.residual_norm))

    def build_adaptive(self, *args, **kwargs):
        raise NotImplementedError(
            "the dimension-adaptive greedy is value-only (its surplus indicator "
            "stores value records); build the value model adaptively with "
            "SmolyakSolverND, then re-solve its index set here with tangents")

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
    def _make_solution(self, index_set, coeffs, subgrids, pool,
                       total_iters) -> HermiteSmolyakSolutionND:
        return HermiteSmolyakSolutionND(
            axes=self.axes, index_set=index_set, coeffs=coeffs, subgrids=subgrids,
            enhanced=self.enhanced, n_solver_nodes=len(pool),
            total_iters=total_iters, _solve_fn=self.solve_fn)


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

    Imports ``parametric_nd_3d``, ``solver_3d``, and
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

            One assembly is shared across the ``d`` axes (the geometry is common
            at a node) — and, when the solve at this θ just ran, it is the
            solve's OWN assembly (``make_solve_fn`` exposes it as
            ``solve_fn.last_asm``): a bare re-assembly per node used to cost
            more than the tangent solves themselves over a large corpus.  The
            tangent solve handles either representation (``_tangent_solve_nk``
            reads ``asm.sep`` when present)."""
            sl = theta_to_slice3d(theta_vec, active_names, M_tot, fixed)
            key = tuple(float(x) for x in np.asarray(theta_vec).ravel())
            last = getattr(solve_fn, "last_asm", None)
            asm = last[1] if (last is not None and last[0] == key) else None
            if asm is not None and tangent_jac != "nk" and asm.M0 is None:
                asm = None      # the modified route needs the dense per-m blocks
            if asm is None:
                asm = s3.assemble(prob, sl)                # cold path (no solve ran)
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
    def build(data, meta):
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
        return solver._finalize(index_set, pool)     # combination coeffs recomputed

    return _load_model_npz(path, "hermite_smolyak", build)
