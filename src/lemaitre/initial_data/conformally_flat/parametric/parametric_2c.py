"""LM-initial-data-2C — wire the §5 parametric layer to the ABT two-centre solver (M3-A).

The headline milestone.  ``parametric.py`` (the solver-agnostic Chebyshev-in-
parameter collocation layer) is reused **verbatim**; this module only injects the
two callables it needs —

  * ``solve_fn(param, guess, tol, max_iter) -> (U, info)``   (Newton from a warm start)
  * ``tangent_fn(param, U) -> dU/dparam``                    (continuation predictor)

— for the two physical sweeps of Phase A:

  * ``from_problem_b``  — sweep the half-separation ``b`` at fixed masses (the
    primary "money plot"; the analog of the single-centre mass sweep).
  * ``from_problem_q``  — sweep the mass ratio ``q = m_A/m_B`` at fixed ``b`` and
    total mass ``M`` (the secondary sweep, for the 2-D capability).

**Frozen topology (load-bearing).** A single ``Problem`` (the b/mass-independent
(A,B) ABT grid) is built once with fixed ``Na, Nb, P`` and reused at every
parameter node; ``b`` enters only the operator scale ``Lap ∝ 1/b²`` and the
analytic source, never the node count.  Hence the nodal field ``U[i,j]`` keeps
the **same shape (Na+1, Nb) and the same coordinate meaning at every node**, so
``parametric.ParametricSolver.build``'s ``np.stack(U_nodes)`` and the elementwise
barycentric interpolant are valid.  This is exactly why the puncture-fitted ABT
grid is ideal for the b-sweep: the b-dependence at a fixed abstract node
``(A_i, B_j)`` is smooth and analytic for ``b`` bounded away from the merger
``b → 0``.

The held-out parametric error measured below is **interpolation-in-parameter**
error only: ``ps.evaluate(b)`` and a direct ``newton_solve`` at the same ``b``
live on the *same* fixed grid, so the spatial error cancels in their difference.

Standalone: numpy + jax + the sibling modules (solver_abt, parametric).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..solver import solver_abt as sa
from . import parametric


# --------------------------------------------------------------------------
# Wiring: build a ParametricSolver around the ABT two-centre spatial solver
# --------------------------------------------------------------------------
def from_problem_b(prob, m_A, m_B, b_min, b_max, Q):
    """Sweep the half-separation ``b`` at fixed masses (q = m_A/m_B fixed).

    ``prob`` is the frozen ABT grid (``solver_abt.make_problem``).  The parameter
    is ``b``; masses are held fixed, so the dimensionless knobs (q, p_A) move
    only through ``b/M`` and ``P/m_A``.
    """
    def solve_fn(b, guess, tol, max_iter):
        return sa.newton_solve(prob, sa.Slice(float(b), m_A, m_B),
                               U0=guess, tol=tol, max_iter=max_iter)

    def tangent_fn(b, U):
        return sa.tangent_b(prob, np.asarray(U), sa.Slice(float(b), m_A, m_B))

    return parametric.ParametricSolver(solve_fn, b_min, b_max, Q, tangent_fn=tangent_fn)


def from_problem_q(prob, b, M_tot, q_min, q_max, Q):
    """Sweep the mass ratio ``q = m_A/m_B`` at fixed ``b`` and total mass ``M``.

    ``m_A = M q/(1+q)``, ``m_B = M/(1+q)`` so ``m_A + m_B = M`` is fixed; only the
    background ψ_BL depends on q (Â² is mass-independent), exactly the structure
    ``tangent_q`` exploits.
    """
    def solve_fn(q, guess, tol, max_iter):
        mA = M_tot * float(q) / (1.0 + float(q))
        mB = M_tot / (1.0 + float(q))
        return sa.newton_solve(prob, sa.Slice(b, mA, mB),
                               U0=guess, tol=tol, max_iter=max_iter)

    def tangent_fn(q, U):
        mA = M_tot * float(q) / (1.0 + float(q))
        mB = M_tot / (1.0 + float(q))
        return sa.tangent_q(prob, np.asarray(U), sa.Slice(b, mA, mB), M_tot)

    return parametric.ParametricSolver(solve_fn, q_min, q_max, Q, tangent_fn=tangent_fn)


# --------------------------------------------------------------------------
# Held-out sample points (generic — never coincide with any CGL node set)
# --------------------------------------------------------------------------
# Irrational-ish fractions of the interval; cos-spaced CGL nodes never land on
# these, so the comparison measures pure interpolation-in-parameter error.
_HOLDOUT_FRACS = np.array([0.137, 0.371, 0.523, 0.689, 0.853])


def holdout_points(p_min, p_max, fracs=None):
    """Generic off-node parameter samples inside ``[p_min, p_max]``."""
    fr = _HOLDOUT_FRACS if fracs is None else np.asarray(fracs, dtype=float)
    return p_min + fr * (p_max - p_min)


# --------------------------------------------------------------------------
# Study 1 — held-out parametric convergence in b (the money plot)
# --------------------------------------------------------------------------
def held_out_convergence_b(prob, m_A, m_B, b_min, b_max, Qs, holdout=None,
                           use_tangent=False, tol=1e-12, max_iter=20):
    """Held-out interpolation-in-b error vs the number of CGL nodes Q.

    For each ``Q`` in ``Qs`` build the b-sweep, then compare ``ps.evaluate(b)`` to
    a *direct* ``newton_solve`` at each held-out ``b`` (both on the same frozen
    grid -> spatial error cancels).  The error is the max over held-out points of
    ``max|U_interp - U_direct|``.

    Returns ``(rows, holdout)`` with ``rows = [(Q, heldout_err, sweep_iters), ...]``.
    """
    hold = holdout_points(b_min, b_max) if holdout is None else np.asarray(holdout, float)
    # direct (reference) solves at the held-out b's — computed once
    U_direct = {}
    for b in hold:
        Ub, _ = sa.newton_solve(prob, sa.Slice(float(b), m_A, m_B),
                                tol=tol, max_iter=25)
        U_direct[float(b)] = np.asarray(Ub)

    rows = []
    for Q in Qs:
        ps = from_problem_b(prob, m_A, m_B, b_min, b_max, Q).build(
            use_tangent=use_tangent, tol=tol, max_iter=max_iter)
        e = max(float(np.max(np.abs(ps.evaluate(float(b)) - U_direct[float(b)])))
                for b in hold)
        rows.append((Q, e, int(sum(ps.iters))))
    return rows, hold


# --------------------------------------------------------------------------
# Study 2 — the analyticity wall: geometric rate vs b_min
# --------------------------------------------------------------------------
def geometric_rate(Qs, errs, q_lo=None, q_hi=None):
    """Decades of held-out error lost per unit Q  (slope of log10(err) vs Q).

    Fit only over a clean monotone-decay window ``[q_lo, q_hi]`` (defaults to the
    full range) so a high-Q conditioning floor does not contaminate the slope.
    Returns ``-slope`` (positive = decades gained per Q).
    """
    Qs = np.asarray(Qs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    mask = np.ones(Qs.shape, dtype=bool)
    if q_lo is not None:
        mask &= Qs >= q_lo
    if q_hi is not None:
        mask &= Qs <= q_hi
    mask &= errs > 0
    slope = np.polyfit(Qs[mask], np.log10(errs[mask]), 1)[0]
    return -slope


def bernstein_rate_from_zero(b_min, b_max):
    """Predicted decades/Q from the nearest analyticity-breaking point at b=0.

    A Chebyshev interpolant on ``[b_min, b_max]`` converges like ``ρ^{-Q}`` where
    ``ρ`` is the Bernstein parameter of the largest ellipse (foci at the interval
    endpoints) excluding the nearest complex singularity.  Modelling the merger
    ``b → 0`` as that singularity, its image under the affine map to ``[-1,1]`` is
    ``ξ0 = (2·0 - (b_max+b_min))/(b_max - b_min) < -1`` and
    ``ρ = |ξ0| + sqrt(ξ0² - 1)``.  Returns ``log10(ρ)`` (decades/Q).

    This is a *first-model* prediction (treats the wall as a point singularity at
    b=0); the measured rate is compared to it in ``run.py``/the tests.
    """
    xi0 = (2.0 * 0.0 - (b_max + b_min)) / (b_max - b_min)
    rho = abs(xi0) + np.sqrt(xi0 ** 2 - 1.0)
    return float(np.log10(rho))


def analyticity_wall(prob, m_A, m_B, b_mins, b_max, Qs, holdout_fracs=None,
                     fit_window=None, tol=1e-12, max_iter=20):
    """Measure the geometric convergence rate at several ``b_min`` (the wall study).

    For each ``b_min`` in ``b_mins`` run :func:`held_out_convergence_b` on
    ``[b_min, b_max]`` (held-out points at fixed *fractions* of each interval so
    every range is probed the same way), measure the geometric rate, and pair it
    with the b=0 Bernstein prediction.

    Returns a list of dicts, one per b_min:
        {b_min, Qs, errs, iters, rate, rate_pred}.
    """
    out = []
    for b_min in b_mins:
        hold = holdout_points(b_min, b_max, holdout_fracs)
        rows, _ = held_out_convergence_b(prob, m_A, m_B, b_min, b_max, Qs,
                                         holdout=hold, tol=tol, max_iter=max_iter)
        Q_arr = [r[0] for r in rows]
        e_arr = [r[1] for r in rows]
        it_arr = [r[2] for r in rows]
        q_lo, q_hi = (None, None) if fit_window is None else fit_window
        rate = geometric_rate(Q_arr, e_arr, q_lo, q_hi)
        out.append(dict(b_min=float(b_min), Qs=Q_arr, errs=e_arr, iters=it_arr,
                        rate=float(rate),
                        rate_pred=bernstein_rate_from_zero(b_min, b_max)))
    return out


# --------------------------------------------------------------------------
# Study 3 — secondary q-sweep convergence
# --------------------------------------------------------------------------
def held_out_convergence_q(prob, b, M_tot, q_min, q_max, Qs, holdout=None,
                           use_tangent=False, tol=1e-12, max_iter=20):
    """Held-out interpolation-in-q error vs Q (fixed b, fixed total mass M)."""
    hold = holdout_points(q_min, q_max) if holdout is None else np.asarray(holdout, float)

    def slq(q):
        return sa.Slice(b, M_tot * q / (1.0 + q), M_tot / (1.0 + q))

    U_direct = {}
    for q in hold:
        Uq, _ = sa.newton_solve(prob, slq(float(q)), tol=tol, max_iter=25)
        U_direct[float(q)] = np.asarray(Uq)

    rows = []
    for Q in Qs:
        ps = from_problem_q(prob, b, M_tot, q_min, q_max, Q).build(
            use_tangent=use_tangent, tol=tol, max_iter=max_iter)
        e = max(float(np.max(np.abs(ps.evaluate(float(q)) - U_direct[float(q)])))
                for q in hold)
        rows.append((Q, e, int(sum(ps.iters))))
    return rows, hold
