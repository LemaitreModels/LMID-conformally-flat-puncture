"""PARASOL — multi-dimensional spectral collocation in parameter space (P3).

The N-D lift of :mod:`parametric` (the 1-D Chebyshev-in-parameter layer, reused
**verbatim** — its ``cheb_param_nodes`` is imported, not re-implemented).  This
module adds, solver-agnostically:

  * :func:`tensor_param_nodes` — a product of per-axis CGL node/weight sets;
  * :func:`snake_order` — the **boustrophedon** (reflected-serpentine) traversal
    of a tensor grid, so consecutive nodes are Hamming-adjacent and every node
    warm-starts from a genuine neighbour;
  * :class:`ParametricSolutionND` — the N-D container with **tensor-product
    barycentric interpolation** by *successive 1-D contraction* (reduces
    bit-for-bit to the 1-D :class:`parametric.ParametricSolution` for one axis),
    a branchless JAX twin ``evaluate_jax`` for ``∂ID/∂θ`` (B3), and the certified
    ``evaluate_polished``;
  * :class:`ParametricSolverND` — drives the snake march and builds the interpolant.

Solver-agnostic: it needs only
  * ``solve_fn(theta_vec, guess, tol, max_iter) -> (U, info)``  (Newton from a warm start)
  * a flatten convention — ``U`` is an array, interpolated elementwise.
The ABT two-centre wiring (``θ=(q,b,χ_A,χ_B) -> Slice -> newton_solve``) lives in
``parametric_nd_2c.py``; this file knows nothing of the physics.

Design (see reports/P3/analysis.md §0):
  D4 — successive-tensordot barycentric, exact for one axis (bit-for-bit);
  D5 — snake march, per-axis forward = descending index = ascending value, so a
       single active axis matches the 1-D ``np.argsort`` march exactly.

Standalone: numpy + jax + the sibling ``parametric`` module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric import cheb_param_nodes   # reused verbatim (the 1-D CGL layer)


# --------------------------------------------------------------------------
# Tensor-product parameter nodes (product of per-axis CGL sets)
# --------------------------------------------------------------------------
def tensor_param_nodes(axes: Sequence[Tuple[float, float, int]]):
    """Per-axis CGL nodes/weights for a tensor grid.

    ``axes = [(p_min, p_max, Q), ...]`` (one tuple per active dimension).
    Returns ``(nodes, weights)`` — lists of arrays, one per axis, each in the
    native ``cheb_param_nodes`` CGL order (descending value).  The full tensor
    grid has ``∏_k (Q_k + 1)`` points; this returns the 1-D factors (the grid is
    never materialised as a flat list of points — :func:`snake_order` walks the
    multi-index).
    """
    nodes, weights = [], []
    for (lo, hi, Q) in axes:
        n, w = cheb_param_nodes(lo, hi, Q)
        nodes.append(n)
        weights.append(w)
    return nodes, weights


# --------------------------------------------------------------------------
# Boustrophedon (reflected-serpentine) traversal of a tensor grid
# --------------------------------------------------------------------------
def snake_order(shape: Sequence[int]) -> List[tuple]:
    """Serpentine multi-index order over a grid of ``shape`` (D5).

    Consecutive multi-indices differ in **exactly one axis by exactly one step**
    (a Hamiltonian path on the grid graph), so each node warm-starts from a true
    neighbour.  The per-axis forward direction is **descending index**
    ``[n-1, …, 0]``; because ``cheb_param_nodes`` returns nodes in *descending
    value*, descending index = *ascending value*.  Hence for a single axis this
    is ``[(Q,), (Q-1,), …, (0,)]`` — identical to the 1-D
    ``ParametricSolver.build`` march (``np.argsort`` of descending nodes), which
    is what makes the single-axis reduction bit-for-bit.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) == 1:
        return [(i,) for i in range(shape[0] - 1, -1, -1)]
    sub = snake_order(shape[1:])
    out: List[tuple] = []
    for k, i in enumerate(range(shape[0] - 1, -1, -1)):   # outer axis, descending index
        seq = sub if k % 2 == 0 else sub[::-1]            # reflect every other block
        for s in seq:
            out.append((i,) + s)
    return out


# --------------------------------------------------------------------------
# N-D parametric solution container + tensor-product barycentric interpolant
# --------------------------------------------------------------------------
@dataclass
class ParametricSolutionND:
    axes: List[Tuple[float, float, int]]   # [(p_min, p_max, Q), ...]
    nodes: List[np.ndarray]                # per-axis CGL nodes (descending value)
    weights: List[np.ndarray]              # per-axis barycentric weights, aligned
    U_nodes: np.ndarray                    # (n_0, …, n_{d-1}, *field_shape)
    iters: np.ndarray                      # (n_0, …, n_{d-1}) Newton iters per node
    residuals: np.ndarray                  # (n_0, …, n_{d-1}) ||R||_inf per node
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

    # ----- tensor-product barycentric interpolant (D4) -----
    def evaluate(self, theta):
        """U(θ) by successive 1-D barycentric contraction.  ``θ`` is a length-d
        vector (single query point).  Reduces bit-for-bit to the 1-D
        ``parametric.ParametricSolution.evaluate`` for ``d == 1``."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if theta.shape[0] != self.d:
            raise ValueError(f"theta has {theta.shape[0]} comps; expected d={self.d}")
        V = self.U_nodes
        for k in range(self.d):
            nodes_k = self.nodes[k]
            diff = theta[k] - nodes_k
            hit = np.isclose(diff, 0.0, atol=1e-13)
            if np.any(hit):                                   # exact-node guard (per axis)
                V = V[int(np.argmax(hit))]                    # select slice (drops axis 0)
            else:
                t = self.weights[k] / diff
                V = np.tensordot(t, V, axes=(0, 0)) / t.sum() # identical to the 1-D formula
        return V

    # ----- JAX-differentiable interpolant (B3 hook) -----
    def evaluate_jax(self, theta):
        """``jnp`` twin of :meth:`evaluate` — branchless and differentiable in θ
        (the ``∂ID/∂θ`` hook for B3).  Must NOT be queried exactly at a node
        (the barycentric quotient is removable there but not finite for jax)."""
        theta = jnp.asarray(theta)
        V = jnp.asarray(self.U_nodes)
        for k in range(self.d):
            nodes_k = jnp.asarray(self.nodes[k])
            t = jnp.asarray(self.weights[k]) / (theta[k] - nodes_k)
            V = jnp.tensordot(t, V, axes=(0, 0)) / jnp.sum(t)
        return V

    # ----- certified evaluation (§5.5) -----
    def evaluate_polished(self, theta, newton_steps: int = 2, tol: float = 1e-12):
        """Barycentric prediction + 1–2 Newton steps → certified ‖R‖≤tol at θ.

        Returns ``(U, info)``; ``info.residual_norm`` is the certified constraint
        residual at θ, independent of any interpolation error (R7)."""
        if self._solve_fn is None:
            raise RuntimeError("no solve_fn attached; build via ParametricSolverND/from_problem_nd")
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(np.asarray(theta, dtype=float), guess, tol, newton_steps)
        return U, info

    # ----- persistence (numpy-only .npz; no pickle, no new deps) -----
    def save(self, path, *, meta=None):
        """Persist this dense tensor surrogate to a single ``.npz`` (numpy-only,
        no pickle).  Round-trips bit-for-bit via :func:`load_parametric`.

        The stored artifact is a **standalone predictor**: after a reload
        ``evaluate`` works with only numpy + the parametric modules (no solver,
        no jax solver).  ``evaluate_polished`` needs a ``solve_fn`` — reattach one
        with :func:`attach_solve_fn_3d`.

        The dense model is one full tensor, so there is no dedup issue: per-axis
        nodes/weights are stored ragged (``nodes_0…nodes_{d-1}`` /
        ``weights_0…``) alongside the ``U_nodes`` tensor, ``iters``, ``residuals``,
        the ``(lo,hi,Q)`` axis meta, ``field_shape``, and a ``meta_json`` 0-d
        string.  ``meta`` merges caller fields (axis_names, box, Q-per-axis,
        Na/Nb/Nφ, solver, tol, note, …); ``format_version``/``kind`` are set
        authoritatively.
        """
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        d = self.d
        arrays = {}
        for k in range(d):
            arrays[f"nodes_{k}"] = np.asarray(self.nodes[k], dtype=float)
            arrays[f"weights_{k}"] = np.asarray(self.weights[k], dtype=float)
        arrays["U_nodes"] = np.asarray(self.U_nodes, dtype=float)
        arrays["iters"] = np.asarray(self.iters, dtype=np.int64)
        arrays["residuals"] = np.asarray(self.residuals, dtype=float)
        arrays["axes"] = np.array([[float(lo), float(hi), float(Q)]
                                   for (lo, hi, Q) in self.axes], dtype=float)
        arrays["field_shape"] = np.asarray(self.field_shape, dtype=np.int64)
        full_meta = {"d": int(d), "git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION      # authoritative
        full_meta["kind"] = "dense"
        arrays["meta_json"] = _pack_meta(full_meta)
        np.savez(path, **arrays)
        return path


# --------------------------------------------------------------------------
# Driver: snake march over the tensor grid, warm-starting Newton each step
# --------------------------------------------------------------------------
class ParametricSolverND:
    """Drives the boustrophedon continuation sweep and builds the N-D interpolant."""

    def __init__(self, solve_fn: Callable, axes: Sequence[Tuple[float, float, int]],
                 tangent_fn: Optional[Callable] = None):
        # solve_fn(theta_vec, guess, tol, max_iter) -> (U, info); guess=None => cold start
        self.solve_fn = solve_fn
        self.axes = [tuple(a) for a in axes]
        self.d = len(self.axes)
        self.tangent_fn = tangent_fn

    def build(self, use_tangent: bool = False, tol: float = 1e-12,
              max_iter: int = 20) -> ParametricSolutionND:
        nodes, weights = tensor_param_nodes(self.axes)
        shape = tuple(len(n) for n in nodes)
        order = snake_order(shape)

        U_nodes = None
        field_shape = None
        iters = np.zeros(shape, dtype=int)
        resids = np.zeros(shape, dtype=float)
        guess = None
        theta_prev = None
        for idx in order:
            theta = np.array([nodes[k][idx[k]] for k in range(self.d)], dtype=float)
            g = guess
            if use_tangent and guess is not None and self.tangent_fn is not None:
                dU = self.tangent_fn(theta_prev, guess)       # dU/dθ at previous node
                g = guess + np.asarray((theta - theta_prev) @ np.atleast_2d(dU))
            U, info = self.solve_fn(theta, g, tol, max_iter)
            Ua = np.asarray(U)
            if U_nodes is None:
                field_shape = Ua.shape
                U_nodes = np.empty(shape + field_shape, dtype=float)
            U_nodes[idx] = Ua
            iters[idx] = info.iters
            resids[idx] = info.residual_norm
            guess = jnp.asarray(Ua)
            theta_prev = theta

        return ParametricSolutionND(
            axes=self.axes, nodes=nodes, weights=weights,
            U_nodes=U_nodes, iters=iters, residuals=resids,
            _solve_fn=self.solve_fn,
        )


# --------------------------------------------------------------------------
# Persistence layer (numpy-only .npz; no pickle, no new deps) — ADD-ONLY
# --------------------------------------------------------------------------
# A built surrogate becomes a reusable on-disk artifact ("we provide the
# solution, no need to solve").  Shared by the dense ``ParametricSolutionND``
# (this module) and the sparse ``SmolyakSolutionND`` (parametric_nd_smolyak),
# which imports the helpers below.  Serialization is numpy-only (parallel
# arrays + a JSON string), so a reloaded model is a standalone predictor that
# needs only numpy + the parametric modules for ``evaluate``.
FORMAT_VERSION = 1


def _git_commit() -> str:
    """Short HEAD hash for provenance, or ``"unknown"`` if unavailable."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _pack_meta(meta: dict):
    """Serialize a metadata dict to a 0-d unicode array (no pickle)."""
    import json
    return np.array(json.dumps(meta))


def _unpack_meta(data) -> dict:
    import json
    if "meta_json" not in data:
        raise ValueError("not a PARASOL surrogate artifact: missing 'meta_json'")
    raw = data["meta_json"]
    try:
        s = raw.item() if hasattr(raw, "item") else str(raw)
        return json.loads(s)
    except Exception as e:  # corrupt / truncated meta blob
        raise ValueError(f"corrupt PARASOL meta_json: {e}")


def _load_npz(path):
    """``np.load`` (no pickle) with a clear error on a garbage/truncated file."""
    path = str(path)
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"could not read PARASOL surrogate '{path}': {e}")


def _check_meta(meta: dict, expected_kind: str):
    """Report a format-version or kind mismatch clearly (never silently mis-read)."""
    fv = meta.get("format_version")
    if fv != FORMAT_VERSION:
        raise ValueError(
            f"unsupported PARASOL format_version {fv!r} "
            f"(this build reads {FORMAT_VERSION})")
    kind = meta.get("kind")
    if kind != expected_kind:
        raise ValueError(
            f"kind mismatch: file is {kind!r} but this loader expects {expected_kind!r}")


def load_parametric(path) -> "ParametricSolutionND":
    """Load a dense :class:`ParametricSolutionND` saved by :meth:`ParametricSolutionND.save`.

    Reconstructs with ``_solve_fn=None`` — ``evaluate`` works immediately (numpy
    only); ``evaluate_polished`` raises ``RuntimeError`` until a solver is
    attached with :func:`attach_solve_fn_3d`.  The parsed metadata is stored on
    the returned object as ``.meta``.
    """
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "dense")
    try:
        d = int(meta["d"])
        nodes = [np.asarray(data[f"nodes_{k}"], dtype=float) for k in range(d)]
        weights = [np.asarray(data[f"weights_{k}"], dtype=float) for k in range(d)]
        axes = [(float(a[0]), float(a[1]), int(round(float(a[2]))))
                for a in np.asarray(data["axes"], dtype=float)]
        U_nodes = np.asarray(data["U_nodes"], dtype=float)
        iters = np.asarray(data["iters"])
        residuals = np.asarray(data["residuals"], dtype=float)
        sol = ParametricSolutionND(axes=axes, nodes=nodes, weights=weights,
                                   U_nodes=U_nodes, iters=iters, residuals=residuals,
                                   _solve_fn=None)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt PARASOL dense surrogate '{path}': {e}")
    sol.meta = meta
    return sol


def attach_solve_fn_3d(sol, prob, axis_names, *, M_tot: float = 1.0, fixed=None,
                       use_cache: bool = True, solver: str = "nk",
                       gmres_rtol: float = 1e-4):
    """Attach a 3-D ``solve_fn`` to a loaded surrogate so ``evaluate_polished``
    works (certified ``‖R‖∞ ≤ 1e-10``).

    Works for **both** :class:`ParametricSolutionND` and
    ``parametric_nd_smolyak.SmolyakSolutionND`` (both expose the same
    ``_solve_fn`` slot + ``evaluate_polished`` contract).  Uses the SAME
    ``parametric_nd_3d.make_solve_fn`` wiring the ``from_problem_*_3d`` builders
    use, so a reloaded model reaches the identical certified prediction.

    ``axis_names`` must match the axis order the model was built with (the box
    order).  This is the ONLY part of the persistence layer that needs jax / the
    solver — plain ``evaluate`` on a loaded model needs only numpy + the
    parametric modules (a standalone predictor).
    """
    from .parametric_nd_3d import make_solve_fn
    solve_fn, _ = make_solve_fn(prob, list(axis_names), M_tot=M_tot, fixed=fixed,
                                use_cache=use_cache, solver=solver, gmres_rtol=gmres_rtol)
    sol._solve_fn = solve_fn
    return sol
