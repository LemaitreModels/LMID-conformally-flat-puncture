"""The certification gate — the residual check that stands between a solve and a
returned datum.

The interpolant is only a *guess*; the certificate is the constraint residual of
the field actually handed back, measured after the last Newton step by the
genuine elliptic solver.  Every ``evaluate_polished`` in this package computes
that number and returns it as ``info.residual_norm``.  This module supplies the
one place where it is *enforced*, so the threshold, the comparison and the error
message exist once rather than nine times.

Two deliberate asymmetries:

* **The tolerance lives here, not in each signature.**  :data:`CERT_TOL_U` is the
  live gate and the default ``tol`` everywhere; :data:`CERT_TOL` is its
  ``v``-norm predecessor, kept at its published value and no longer the default.
  **Which norm a threshold is in is part of the threshold** — since 2026-08-25
  ``operators_3d.EQUIL_NORM_DEFAULT`` is ``"u"``, so a bare ``1e-10`` here would
  silently be four orders looser than the number it looks like.  Before either
  constant existed, every ``evaluate_polished`` defaulted to ``1e-12`` — below the
  equilibrated residual's roundoff floor (~2e-12 on the production grid), so
  ``info.converged`` was False on every default-tol query and the solver's own
  ``if rn < tol: break`` was dead code.  The status flag reported nothing and cost
  an extra Newton step to not report it.

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

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

#: **The historical ``v``-norm gate.  NOT the live threshold — see
#: :data:`CERT_TOL_U`.**
#:
#: ``1e-10`` on the *equilibrated* discrete constraint residual measured in the
#: ``v`` norm, i.e. against ``operators_3d``'s ``scales``.  It sits ~1.5 decades
#: above the equilibrated roundoff floor and ~0.5 decades above the Newton–Krylov
#: floor observed at the extreme corners of the 8-D box (~4e-11 — a ``v``-norm,
#: pre-2026-08-25 measurement, like every number in this paragraph), which was the
#: margin that kept it from firing on arithmetic rather than on physics.
#:
#: **It is kept, with its value and its name unchanged, because it is what the
#: submitted paper's eleven residual numbers were measured against.**  Frederik's
#: 2026-08-25 reversibility constraint is that an old constant keeps a distinct name
#: so no published number is orphaned.  Reading a ``u``-norm residual against this
#: number is the *mixing* failure the two-norm design exists to prevent: the two are
#: **not comparable**, and no single factor converts between them (the measured
#: ``v/u`` ratio on field-converged solves runs ``1×`` at ``Nφ = 1`` to ``6.6e+08``
#: at ``36×24×20``, because ``1/w_min`` bounds the ratio but is attained by the *row
#: scale*, not by any particular residual).
#:
#: Anything still comparing against this constant must say ``norm="v"`` at the
#: comparison and must be measuring a pre-2026-08-25 claim.
CERT_TOL = 1e-10

#: **The live certification gate: ``1e-11`` on the equilibrated residual in the
#: ``u`` norm.**  Frederik's value, 2026-08-25.
#:
#: The default ``tol`` of every ``evaluate_polished``, so ``info.converged`` means
#: what the papers say it means — now that :data:`~...solver.operators_3d.
#: EQUIL_NORM_DEFAULT` is ``"u"``, a ``tol`` in this module must be a ``u``-norm
#: number or the solve's own stopping test is read in the wrong units.
#:
#: **How the value was derived, and its window.**  The gate remains a
#: *constraint-residual bound*; truncation error justifies its VALUE and does not
#: become the claim.  A datum is worth certifying when its iteration error is
#: negligible against the error the grid itself carries, which bounds the gate from
#: **above**; what the solve attains bounds it from below.  Measured, that window is
#: ``[6.30e−13, 4.56e−07]`` — the ceiling is the *curved* leaf's, at the
#: quasi-circular box's best-resolved point ``(b, q, χ) = (10, 1, 0.9)`` where
#: ``e_trunc = [5.16e−07, 8.53e−07]`` at the shipped ``44×32×8``.  ``1e-11`` sits
#: ``16×`` above the worst achieved field-converged residual, ``4.6e+03×`` below the
#: conservative ceiling, and rejects the diverging population (``u ≈ 2.7e−02``) by
#: ``2.7e+09×``.
#:
#: **⚠ This is NOT a tightening of :data:`CERT_TOL`, and the sentence has to travel
#: with the number.**  At the production grid ``1e-11`` in ``u`` is ``≈1.7e−08`` in
#: ``v``-equivalent terms — **~170× LOOSER** than the old ``1e-10``-in-``v``.  No
#: value in the derived window is as tight as the old gate.  That is not a
#: weakening of the claim but a correction of the ruler: the ``v`` monitor was
#: rejecting six *field-converged* solves and certifying coarser ones, because it
#: carries a boundary factor that grows without bound in ``Nφ``.
#:
#: **What it does NOT say.**  It bounds the constraint violation, not the distance
#: to the continuum solution.  The curved leaf's shipped grid carries ``8.3–52 %``
#: field error at its laddered configuration and ``0.11 %`` at its best-resolved
#: corner, all of it certified.  Certification is not a field-accuracy claim.
CERT_TOL_U = 1e-11


#: Attribute a ``solve_fn`` carries to declare which norm its residual is in, and
#: therefore which threshold it must be read against.  Set by
#: :func:`parametric_nd_3d.make_solve_fn`, read by :func:`gate_for`.
SOLVE_FN_GATE_ATTR = "cert_tol"


def gate_for(solve_fn) -> float:
    """The certification gate appropriate to ``solve_fn``'s residual norm.

    **Why this is not a constant.**  Until 2026-08-25 there was one threshold
    because there was one norm.  There are now two, and which applies is decided by
    the *solver*, not by the caller or the container:

    * the **3-D Newton–Krylov** path reports the equilibrated residual in the ``u``
      norm (``operators_3d.EQUIL_NORM_DEFAULT``), so its gate is :data:`CERT_TOL_U`;
    * the **axisymmetric 2-D ABT** path (``solver_abt.newton_solve``) has no ``Nφ``
      and therefore no ``(1 − B²)^{−m/2}`` boundary factor at all — it *is* the
      ``Nφ = 1`` case, where the two norms are **bit-identical**.  The norm change is
      a no-op there, so its gate must not move either: it stays :data:`CERT_TOL`.

    **That second point cost a test, and it is the subtle half.**  The invariance
    that protects the axisymmetric sector protects the *comparison*; it says nothing
    about the *threshold* underneath it.  Applying the ``u`` gate ``1e-11`` there is
    not a change of units — it is a genuine, unjustified ``10×`` tightening, and it
    showed up as ``test_certified_polish_4d`` spending a third Newton step to reach
    an internal target a decade below where it was calibrated.

    A ``solve_fn`` declares its norm by carrying :data:`SOLVE_FN_GATE_ATTR`; one that
    does not is assumed axisymmetric, because that is the norm-invariant case and the
    conservative one — a solver whose residual really is in ``u`` and forgets to
    declare it gets a gate ``170×`` too *tight* in ``v``-equivalent terms and fails
    loudly, rather than one four orders too loose and passing silently.
    """
    return float(getattr(solve_fn, SOLVE_FN_GATE_ATTR, CERT_TOL))


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


class CertifiedEvaluateMixin:
    """The one ``evaluate_polished`` body every solution container shares.

    A container contributes ``evaluate`` (the guess) and ``_solve_fn`` (the
    certified Newton solve); this mixin supplies the polish step itself, so the
    tolerance default, the guess→solve→gate wiring and the NaN-safe comparison
    exist once rather than nine times.  Two per-class knobs:

    * ``_solve_fn_hint`` — appended to the no-solver error, so the message still
      names the class's own builder;
    * ``_solve_theta`` — how θ is passed to ``solve_fn`` (the 1-D containers'
      solvers take a scalar, the N-D ones a vector).
    """

    _solve_fn_hint = "attach one via the builder or parametric_nd.attach_solve_fn_3d"

    @staticmethod
    def _solve_theta(theta):
        return np.asarray(theta, dtype=float)

    def evaluate_polished(self, theta, newton_steps: int = 2, tol: Optional[float] = None,
                          strict: bool = False):
        """Interpolated prediction + 1–2 Newton steps → certified ``‖R‖≤tol`` at θ.

        The interpolant is only a *guess*; the attached ``solve_fn`` →
        ``newton_solve`` is the certificate.  Returns ``(U, info)`` with
        ``info.residual_norm`` the certified constraint residual at θ,
        independent of any interpolation error.  ``strict=True`` closes the
        gate: a datum that misses ``tol`` raises :class:`CertificationError`
        instead of being returned.

        **``tol=None`` means "this solver's gate", which is not one number** — see
        :func:`gate_for`.  A container is generic: the same class carries a 2-D ABT
        solve_fn or a 3-D Newton–Krylov one, and since 2026-08-25 those two report
        residuals in *different norms*.  So the threshold is a property of the
        attached solver, not of the container and not of this module.
        """
        if self._solve_fn is None:
            raise RuntimeError(f"no solve_fn attached; {self._solve_fn_hint}")
        tol = gate_for(self._solve_fn) if tol is None else tol
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(self._solve_theta(theta), guess, tol, newton_steps)
        return certified_return(U, info, theta, tol, strict)
