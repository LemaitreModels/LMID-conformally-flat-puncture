"""PARASOL — spectral collocation in parameter space (§5 of plan.md).

This is the headline contribution.  The map  q |-> u(.,.;q)  is analytic on a
range with no coincidence/merger singularity (a single puncture is analytic for
all m>0), so a global Chebyshev interpolant in q converges **exponentially** in
the number of parameter nodes Q.

The layer is *solver-agnostic*: it only needs
  * solve_fn(q, guess, tol, max_iter) -> (U, info)   (Newton from an optional warm start)
  * (optional) tangent_fn(q, U) -> dU/dq             (continuation predictor)
  * a flatten convention — here U is just an array, interpolated elementwise.
A two-puncture head-on solver (parameter = mass ratio) would reuse this file
verbatim; only the injected callables change.  ``from_problem`` wires the
single-puncture solver in solver.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp


# --------------------------------------------------------------------------
# Parameter nodes (Chebyshev–Gauss–Lobatto on [q_min, q_max])
# --------------------------------------------------------------------------
def cheb_param_nodes(q_min: float, q_max: float, Q: int):
    """CGL nodes on [q_min,q_max] and their barycentric weights.

    xi_i = cos(i pi/Q), i=0..Q, mapped affinely to [q_min,q_max].
    Barycentric weights lambda_i = (-1)^i * delta_i, delta = 1/2 at i in {0,Q}
    (standard CGL weights — independent of the affine map).  Returns
    ``(q_nodes, weights)`` in CGL order (descending q).
    """
    i = np.arange(Q + 1)
    xi = np.cos(np.pi * i / Q)
    q = 0.5 * (q_max + q_min) + 0.5 * (q_max - q_min) * xi
    lam = (-1.0) ** i
    lam[0] *= 0.5
    lam[-1] *= 0.5
    return q, lam


# --------------------------------------------------------------------------
# Parametric solution container + sweep
# --------------------------------------------------------------------------
@dataclass
class ParametricSolution:
    q_min: float
    q_max: float
    Q: int
    q_nodes: np.ndarray          # (Q+1,)  CGL order
    weights: np.ndarray          # (Q+1,)  barycentric weights, aligned
    U_nodes: np.ndarray          # (Q+1, *field_shape)  converged fields, aligned
    iters: list                  # Newton iters per node (aligned to q_nodes)
    residuals: list              # ||R||_inf per node (aligned)
    _solve_fn: Callable = field(repr=False, default=None)

    # ----- barycentric interpolant in q -----
    def evaluate(self, q):
        """U(q) via barycentric interpolation.  Scalar or batch q."""
        qe = np.atleast_1d(np.asarray(q, dtype=float))
        fshape = self.U_nodes.shape[1:]
        out = np.empty((qe.shape[0],) + fshape)
        for j, qj in enumerate(qe):
            diff = qj - self.q_nodes
            hit = np.isclose(diff, 0.0, atol=1e-13)
            if np.any(hit):                                  # exact-node guard
                out[j] = self.U_nodes[int(np.argmax(hit))]
            else:
                t = self.weights / diff
                out[j] = np.tensordot(t, self.U_nodes, axes=(0, 0)) / t.sum()
        return out[0] if np.ndim(q) == 0 else out

    # ----- certified evaluation (§5.5) -----
    def evaluate_polished(self, q, newton_steps: int = 2, tol: float = 1e-12):
        """Barycentric prediction + 1-2 Newton steps -> certified ||R||<=tol.

        Returns ``(U, info)`` where ``info.residual_norm`` is the certified
        constraint residual at q, independent of any interpolation error.
        """
        if self._solve_fn is None:
            raise RuntimeError("no solve_fn attached; build via ParametricSolver/from_problem")
        guess = jnp.asarray(self.evaluate(q))
        U, info = self._solve_fn(float(q), guess, tol, newton_steps)
        return U, info


class ParametricSolver:
    """Drives the continuation sweep and builds the interpolant."""

    def __init__(self, solve_fn: Callable, q_min: float, q_max: float, Q: int,
                 tangent_fn: Optional[Callable] = None):
        # solve_fn(q, guess, tol, max_iter) -> (U, info); guess=None => cold start
        self.solve_fn = solve_fn
        self.tangent_fn = tangent_fn
        self.q_min = q_min
        self.q_max = q_max
        self.Q = Q

    def build(self, use_tangent: bool = False, tol: float = 1e-12,
              max_iter: int = 20) -> ParametricSolution:
        """Solve at every CGL node, warm-starting Newton along the parameter march."""
        q_nodes, weights = cheb_param_nodes(self.q_min, self.q_max, self.Q)
        Q1 = q_nodes.shape[0]
        order = np.argsort(q_nodes)                          # march by increasing q

        U_nodes = [None] * Q1
        iters = [None] * Q1
        resids = [None] * Q1
        guess = None
        q_prev = None
        for step, idx in enumerate(order):
            q = float(q_nodes[idx])
            g = guess
            if use_tangent and guess is not None and self.tangent_fn is not None:
                dU = self.tangent_fn(q_prev, guess)          # dU/dq at previous node
                g = guess + (q - q_prev) * dU                # linear predictor
            U, info = self.solve_fn(q, g, tol, max_iter)
            U_nodes[idx] = np.asarray(U)
            iters[idx] = info.iters
            resids[idx] = info.residual_norm
            guess = jnp.asarray(U)
            q_prev = q

        return ParametricSolution(
            q_min=self.q_min, q_max=self.q_max, Q=self.Q,
            q_nodes=q_nodes, weights=weights,
            U_nodes=np.stack(U_nodes), iters=iters, residuals=resids,
            _solve_fn=self.solve_fn,
        )


# --------------------------------------------------------------------------
# Convenience: wire the single-puncture solver (solver.py)
# --------------------------------------------------------------------------
def from_problem(prob, q_min: float, q_max: float, Q: int):
    """Build a ParametricSolver around the single-puncture spatial solver.

    q == m (bare puncture mass); the spatial Problem is m-independent (fixed L),
    so it is built once and reused at every parameter node.
    """
    from ..solver import solver

    def solve_fn(q, guess, tol, max_iter):
        return solver.newton_solve(prob, m=q, U0=guess, tol=tol, max_iter=max_iter)

    def tangent_fn(q, U):
        return solver.tangent(prob, jnp.asarray(U), m=q)

    return ParametricSolver(solve_fn, q_min, q_max, Q, tangent_fn=tangent_fn)
