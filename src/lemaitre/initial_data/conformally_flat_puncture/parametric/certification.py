"""The certification gate — the residual check that stands between a solve and a
returned datum.

The interpolant is only a *guess*; the certificate is the constraint residual of
the field actually handed back, measured after the last Newton step by the
genuine elliptic solver.  Every ``evaluate_polished`` in this package computes
that number and returns it as ``info.residual_norm``.  This module supplies the
one place where it is *enforced*, so the threshold, the comparison and the error
message exist once rather than nine times.

Two deliberate asymmetries:

* **The tolerance lives here, not in each signature.**  :data:`CERT_TOL` is the
  paper's gate.  Before it existed, every ``evaluate_polished`` defaulted to
  ``1e-12`` — below the equilibrated residual's roundoff floor (~2e-12 on the
  production grid), so ``info.converged`` was False on every default-tol query
  and the solver's own ``if rn < tol: break`` was dead code.  The status flag
  reported nothing and cost an extra Newton step to not report it.

* **The gate is opt-in (``strict=True``), not the default.**  A raise inside the
  library would be wrong for the callers that legitimately query *below* the
  gate: the polish-history producers sweep ``tol=1e-12`` precisely to watch the
  residual fall through 1e-10 and past it, and a sweep over hundreds of θ wants
  a bad point flagged, not the sweep lost.  So the library measures and reports;
  the gate is closed at the points where a datum leaves the package — the
  quasi-circular applications and the evolution export — which is where the
  paper's claim is made.

Callers that want the gate pass ``strict=True``; callers that want to see the
number compare ``info.residual_norm`` themselves.
"""

from __future__ import annotations

from typing import Any, Optional

# The paper's certification threshold on the *equilibrated* discrete constraint
# residual.  Chosen as the default `tol` of every `evaluate_polished`, so that
# `info.converged` means what the paper says it means.  It sits ~1.5 decades
# above the equilibrated roundoff floor and ~0.5 decades above the Newton-Krylov
# floor observed at the extreme corners of the 8-D box (~4e-11), which is the
# margin that keeps the gate from firing on arithmetic rather than on physics.
CERT_TOL = 1e-10


class CertificationError(RuntimeError):
    """A queried datum failed the ``‖R‖∞ ≤ tol`` gate and was therefore not returned.

    Raised only from a ``strict=True`` call.  The message carries the parameter,
    the achieved residual and the step count, because the actionable question
    after a failure is *where* in the box it happened — the known mode is a
    global-convergence stall at extreme corners (high ``q`` + strong spin at wide
    separation), for which ``parametric_nd_3d.make_solve_fn`` provides the
    ``retry_tol`` damped-Newton globalization.
    """

    def __init__(self, theta: Any, residual_norm: float, tol: float,
                 iters: Optional[int] = None):
        self.theta = theta
        self.residual_norm = float(residual_norm)
        self.tol = float(tol)
        self.iters = None if iters is None else int(iters)
        steps = "" if self.iters is None else f" after {self.iters} Newton steps"
        super().__init__(
            f"certification failed at theta={theta}: equilibrated "
            f"||R||_inf = {self.residual_norm:.3e} > tol = {self.tol:.1e}{steps}. "
            f"The datum was NOT returned. If this is an extreme corner of the box, "
            f"attach the solver with a damped-Newton globalization "
            f"(attach_solve_fn_3d(..., retry_tol=...)); otherwise raise newton_steps."
        )


def certified_return(U, info, theta, tol: float, strict: bool):
    """Return ``(U, info)``, or raise :class:`CertificationError` when ``strict``.

    The comparison is written ``not (r <= tol)`` rather than ``r > tol`` so that a
    NaN residual — a diverged solve, not merely an under-converged one — fails the
    gate instead of passing it silently.
    """
    if strict and not (float(info.residual_norm) <= float(tol)):
        raise CertificationError(theta, info.residual_norm, tol,
                                 getattr(info, "iters", None))
    return U, info
