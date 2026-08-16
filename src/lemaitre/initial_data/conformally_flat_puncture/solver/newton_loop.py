"""The one Newton loop — shared driver for every Newton solve in the package.

``solver_abt.newton_solve`` (2-D), ``solver_3d.newton_solve`` (modified Newton)
and ``solver_3d_nk.newton_solve_nk`` (the certified Newton–Krylov) run the same
loop skeleton: measure the residual, keep the best iterate, stop on tolerance or
stagnation, step.  It used to be written out separately in each of them (and
again in the fig04 field-error producer), so a fix to the *stopping policy* had
to land four times and could drift.  This module owns that skeleton; the solvers
supply only their monitor and their step.

Stopping policy (the two deliberate choices)
--------------------------------------------
* **Solve past the tolerance, certify against it.**  The loop breaks on an
  internal ``target_tol`` (default ``tol/10``), while ``converged`` — and every
  certification comparison downstream — is still judged against the caller's
  ``tol``.  Breaking on the first measurement below ``tol`` leaves the certified
  residual wherever that measurement happened to land, which can be within a few
  percent of the gate; a certificate that close to its threshold is fragile
  against any later ulp-level change in the arithmetic.  Newton is quadratic
  here, so the marginal iterates gain several orders in one extra step: one
  decade of target is enough to push every such point to the residual floor,
  and it costs that extra step *only* for measurements landing in
  ``[target_tol, tol)`` — points already below ``target_tol`` break exactly as
  before.

* **Stagnation needs two strikes.**  The loop stops early when the residual has
  stopped improving, so sweeps do not burn iterations grinding at the floor.
  A single non-halving measurement is not proof of that: the monitor can
  fluctuate at one iterate while the field is still converging, and stopping
  there has been observed to cost five orders of magnitude in the returned
  residual.  So a *strike* is a measurement that fails to halve the best
  residual so far, and only two consecutive strikes stop the loop; any
  measurement that halves the best resets the count.  Genuine floor behaviour
  still stops within two extra iterations; a one-off blip no longer does.

Budget semantics
----------------
``max_steps`` bounds the number of Newton steps, and **every step taken is
measured**: the loop records the residual of the guess and of each of the
``k <= max_steps`` iterates it produces, so ``history[k]`` is the residual
after exactly ``k`` steps and the final step's result is never computed and
then thrown away unmeasured (which is what a measure-then-step loop does when
it runs out of iterations one measurement short).

Standalone: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class LoopRun:
    """What one driven Newton loop did.

    ``U`` is the **best** measured iterate (not necessarily the last), and
    ``residual_norm`` its monitor value — the certified number.  ``iters`` is
    the number of residual measurements (``len(history)``; the number of Newton
    steps taken is ``iters - 1``).  ``converged`` is judged against the
    caller's ``tol``, not the internal target.
    """
    U: Any
    converged: bool
    iters: int
    residual_norm: float
    history: list


def newton_loop(U0, *,
                monitor_fn: Callable[[Any], tuple],
                step_fn: Callable[[Any, int, Any], Any],
                tol: float,
                max_steps: int,
                target_tol: Optional[float] = None,
                on_iterate: Optional[Callable[[int, float, Any], None]] = None,
                ) -> LoopRun:
    """Drive Newton from ``U0``; return the best iterate and its history.

    ``monitor_fn(U) -> (rn, aux)`` measures the residual norm the loop drives
    down; ``aux`` is passed to the step untouched, so a solver whose step wants
    the residual it just measured (e.g. the 2-D dense Newton) does not compute
    it twice.  ``step_fn(U, k, aux) -> U_new`` takes Newton step ``k`` (0-based:
    ``k`` steps have been taken when it is called).  ``on_iterate(k, rn, U)``,
    if given, sees every measured iterate — after ``k`` steps — before any
    stopping decision; instrumented wrappers (the fig04 field-error producer)
    hook here instead of re-implementing the loop.

    ``target_tol`` defaults to ``tol/10`` — see the module docstring for why
    the loop solves one decade past the tolerance it certifies against.
    """
    target = 0.1 * float(tol) if target_tol is None else float(target_tol)
    U = U0
    history: list = []
    best_U, best_rn = U, np.inf
    strikes = 0
    k = 0
    while True:
        rn, aux = monitor_fn(U)
        rn = float(rn)
        history.append(rn)
        if on_iterate is not None:
            on_iterate(k, rn, U)
        if rn < 0.5 * best_rn:
            strikes = 0
        else:
            strikes += 1
        if rn < best_rn:
            best_U, best_rn = U, rn
        if rn < target:
            break
        if strikes >= 2:                     # stagnation: two consecutive non-halvings
            break
        if k >= max_steps:                   # step budget spent (all steps measured)
            break
        U = step_fn(U, k, aux)
        k += 1
    return LoopRun(U=best_U, converged=bool(best_rn < tol), iters=len(history),
                   residual_norm=float(best_rn), history=history)
