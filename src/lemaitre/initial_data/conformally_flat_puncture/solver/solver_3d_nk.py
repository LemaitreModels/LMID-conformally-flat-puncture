"""LM-initial-data-3D Newton–Krylov — the *certified* non-axisymmetric two-centre solve.

Sibling of ``solver_3d``.  The modified-Newton solver
(``solver_3d.newton_solve``) drops the φ-varying part of the source Jacobian: it
adds only the φ-AVERAGED diagonal ``d̄(A,B) = ⟨−⅞(ψ+u)⁻⁸Â²⟩_φ`` to each per-m
block.  The converged residual is therefore solver-limited — ``‖R‖∞`` RISES with
resolution as more azimuthal-mode content appears (the documented monitor
behaviour: 3e-9 → 1.1e-7 over the Block-B ladder, while the field keeps
converging).  This module restores the dropped mode-coupling with a true
**Newton–Krylov** solve, taking ``‖R‖∞`` to machine precision and giving the
3-D analog of the axisymmetric *certified polish*.

Design — matrix-free Newton with a block-diagonal preconditioner
----------------------------------------------------------------
The nonlinear source ``S = ⅛(ψ+u)⁻⁷Â²`` is node-diagonal in PHYSICAL (A,B,φ)
space; its Fréchet derivative ``D_nl = −⅞(ψ+u)⁻⁸Â²`` is likewise node-diagonal
in physical space, so multiplying by it is a *pointwise* product in φ — i.e. a
**convolution in the azimuthal mode m** that couples modes.

* **Full Jacobian action** on a nodal increment ``δu`` (real, shape
  ``(Ntot2d, Nφ)``), assembled in mode space exactly as ``residual_modes``:

      (J δu)_m = M0_m (δû_m / w_m)  +  interior · rfft_φ[ D_nl · δu ]_m ,

  with ``δû = rfft_φ(δu)``.  The first term reuses the per-m block operators
  ``asm.M0`` verbatim (linear, block-diagonal in m); the second is the
  mode-coupling the modified Newton drops.  The nodal action is the irfft.

* **Preconditioner** — two of them, chosen by :func:`choose_separable`:

  - *separable* (the default above Nφ=1): the exact inverse of the LINEAR block
    alone, by fast diagonalization of its 1-D Kronecker factors
    (``solver/separable.py``).  It drops the nonlinear diagonal, which is exactly
    what makes it independent of ``b``, of θ and of the iterate — so it is built
    ONCE per grid and there is nothing to refactor per Newton step or per
    parameter point.  Measured cost: one extra GMRES iteration per step, flat in
    Nφ up to 24.
  - *dense*: the EXISTING per-m block, i.e. exactly what
    ``solver_3d.newton_step`` solves: ``M̂_m = M0_m + diag(interior · d̄ · w_m)``,
    factored ONCE per Newton step (LU on the row-equilibrated block).  It is J
    minus only the mode-coupling.  At Nφ=1 there is no mode-coupling, so it IS J
    and GMRES converges in one iteration — which is why that case keeps it.

* **Outer loop**: classical Newton, ``δu = J⁻¹(−R)`` via ``scipy gmres`` with a
  ``LinearOperator`` for J and for ``M⁻¹``.  Quadratic convergence → ``‖R‖`` to
  machine.

With ``Nφ=1`` the preconditioner IS the full Jacobian (no mode-coupling), so the
GMRES solve is exact in one iteration and every step is **identical to the frozen
2-D Newton** — the axisymmetric-reduction gate.

Standalone: numpy + scipy + the sibling modules (solver_3d, operators_abt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, gmres

from . import operators_3d as ops3
from . import solver_3d as s3
from .solver_3d import Assembly3D, Problem3D, Slice3D  # noqa: F401  (re-export)


# --------------------------------------------------------------------------
# Row-equilibrated LU (the repeated-solve form of operators_abt.solve_equilibrated)
# --------------------------------------------------------------------------
def _lu_factor_equilibrated(M: np.ndarray):
    """Factor ``M`` with row equilibration; return ``(lu, piv, scale)``.

    Identical rescaling to ``operators_abt.solve_equilibrated`` (each row scaled
    by 1/max|row|, an exact rescaling of the equations) but factored once so the
    preconditioner apply is a cheap triangular solve, not a fresh LU.
    """
    M = np.asarray(M, dtype=float)
    scale = np.max(np.abs(M), axis=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    lu, piv = sla.lu_factor(M / scale[:, None])
    return lu, piv, scale


def _lu_solve_equilibrated(fac, rhs: np.ndarray) -> np.ndarray:
    """Apply the factorization to a real-or-complex right-hand side.

    The factors are real and the mode-space right-hand sides complex, so passing
    the complex vector straight to ``lu_solve`` promotes the *factors* — a fresh
    complex copy of an ``(Na·Nb)²`` array, and complex triangular solves, on
    every preconditioner apply.  Solving ``[Re rhs, Im rhs]`` as two real columns
    is the same arithmetic (the factors are real, so real and imaginary parts do
    not mix) in one real ``getrs`` call.  A mode whose imaginary part is exactly
    zero — m=0, and the Nyquist mode when Nφ is even — takes the one-column real
    solve, which is cheaper again and bit-identical to the axisymmetric path.
    """
    lu, piv, scale = fac
    y = np.asarray(rhs) / scale
    yr = ops3._real_part_if_real(y)
    if yr is not None:
        out = sla.lu_solve((lu, piv), yr)
        return out.astype(complex) if np.iscomplexobj(y) else out
    return ops3._join_ri(sla.lu_solve((lu, piv), ops3._split_ri(y)))


# --------------------------------------------------------------------------
# Which representation of the linear operator a solve should build
# --------------------------------------------------------------------------
def choose_separable(prob: Problem3D, separable, n_warmup: int = 0) -> bool:
    """Resolve ``separable=None`` (the default) into a decision, and say why.

    The separable route is the default because it is much cheaper — the operator
    is built once per grid rather than per parameter point, and there is no LU to
    refactor per Newton step.  Two cases deliberately opt back out:

    * **Nφ=1.**  With one azimuthal mode there is no mode-coupling, so the
      block-diagonal preconditioner IS the full Jacobian and GMRES converges in a
      single iteration — an exact solve.  The separable preconditioner drops the
      nonlinear diagonal, so it would take several iterations to reach the same
      answer.  Choosing the exact one here is not a special case for the tests'
      benefit: it is faster, and it keeps the axisymmetric reduction identical to
      the frozen 2-D Newton (``tests/test_solver_3d.py``).
    * **``n_warmup > 0``.**  The warm-up steps are ``solver_3d.newton_step``,
      whose Jacobian carries the nonlinear diagonal and so needs the dense blocks.

    Pass ``separable=True``/``False`` to force either route.  A caller that
    supplies its own ``asm`` bypasses this entirely — the assembly it built
    already fixes the representation.
    """
    if separable is not None:
        return bool(separable)
    if int(prob.Nphi) == 1:
        return False
    if n_warmup:
        return False
    return True


# --------------------------------------------------------------------------
# Residual monitors — raw nodal vs. equilibrated (the certified number)
# --------------------------------------------------------------------------
def _block_scales(asm: Assembly3D):
    """Per-m row-equilibration scale ``max|M0_m row|`` (the solve's row scaling).

    Depends only on the linear blocks, so it is computed with them by
    ``operators_3d.mode_operators_cached`` and carried on the assembly; this
    recomputes it only for an assembly built by hand (``asm.scales is None``),
    which would otherwise re-scan ~79 MiB of blocks per Newton iteration.
    """
    if getattr(asm, "scales", None) is not None:
        return asm.scales
    scales = []
    for mi in range(asm.m_vals.size):
        M0 = np.asarray(asm.M0[mi])
        s = np.max(np.abs(M0), axis=1)
        scales.append(np.where(s > 0.0, s, 1.0))
    return scales


def equil_residual_inf(asm: Assembly3D, U: np.ndarray, scales=None) -> float:
    """The **equilibrated** mode-space residual inf-norm — the certified monitor.

    ``max_m ‖R̂_m / scale_m‖∞`` with ``scale_m = max|M0_m row|`` (the same exact
    row rescaling :func:`operators_abt.solve_equilibrated` uses in the solve).
    The raw nodal residual ``solver_3d.nodal_residual_inf`` is dominated by
    floating-point roundoff in the stiff rows next to the inner axis (A→0, where
    the 1/A and m²/ρ² coefficients are enormous), so it floors well above machine
    precision and RISES with (Na, Nφ).  Dividing by the row scale removes that
    roundoff amplification, giving the residual in the well-conditioned norm the
    Newton solve actually drives to zero — the honest certified constraint
    residual.
    """
    Nphi = U.shape[1]
    if scales is None:
        scales = _block_scales(asm)
    Rm = s3.residual_modes(asm, np.asarray(U).reshape(asm.interior.size, Nphi))
    e = 0.0
    for mi in range(asm.m_vals.size):
        e = max(e, float(np.max(np.abs(Rm[:, mi]) / scales[mi])))
    return e


# --------------------------------------------------------------------------
# One Newton–Krylov step
# --------------------------------------------------------------------------
@dataclass
class _StepInfo:
    gmres_iters: int
    gmres_status: int
    residual_norm: float        # ‖R‖∞ BEFORE the step (nodal)


def newton_step_nk(asm: Assembly3D, U: np.ndarray,
                   gmres_rtol: float = 1e-4, gmres_atol: float = 1e-12,
                   gmres_restart: int = 50, gmres_maxiter: int = 60):
    """One true-Newton step ``U → U + δu`` with ``δu = J⁻¹(−R)`` via GMRES.

    ``U`` nodal (Ntot2d, Nφ).  Returns ``(U_new, _StepInfo)``.  The Jacobian is
    applied matrix-free in mode space (full mode-coupling); the block-diagonal
    modified-Newton operator preconditions it.

    ``gmres_rtol`` is the **inexact-Newton forcing term** η: GMRES reduces the
    *linear* residual to ``η‖R‖`` (relative to the current ‖R‖).  η≈1e-4 keeps
    every step single-digit GMRES iterations *and* gives near-quadratic Newton
    convergence.  A fixed-small absolute tolerance would instead demand the
    impossible (sub-roundoff) target ``rtol·‖R‖`` once ‖R‖ nears the
    discretisation floor, stalling GMRES at thousands of iterations.
    """
    Nphi = U.shape[1]
    Ntot = asm.interior.size
    Nm = asm.m_vals.size
    interior = asm.interior
    M0 = asm.M0
    w = asm.w

    # current residual (mode space, exact) and the physical-space source derivative
    Rm = s3.residual_modes(asm, U)
    R_node = np.fft.irfft(Rm, n=Nphi, axis=1)
    _, base = s3._nl_source(asm, U)
    D_nl = -0.875 * base ** (-8.0) * asm.A2          # (Ntot, Nφ) node-diagonal deriv
    d_bar = D_nl.mean(axis=1)                          # φ-average -> m=0 diagonal

    # The preconditioner.  Two routes, and the difference is what they include:
    #
    #  * separable (the DEFAULT — see choose_separable) — the LINEAR block alone,
    #    inverted exactly by fast diagonalization (solver/separable.py).  Dropping
    #    the nonlinear diagonal is what makes it b- and θ-independent, so there is
    #    nothing left to factor per step or per parameter point; it costs a GMRES
    #    iteration or two, and at Nφ=1 it is no longer the exact Jacobian.
    #  * dense — M̂_m = M0_m + diag(interior·d̄·w_m), i.e. exactly what
    #    solver_3d.newton_step solves, LU-factored once per Newton step.  It is J
    #    minus only the azimuthal mode-coupling, so at Nφ=1 it IS J and GMRES
    #    converges in one iteration.  The factorizations are ~80% of the step.
    #    choose_separable opts back into it at Nφ=1 and during warm-up.
    #
    # Which one this call gets was already decided by the caller: it is whichever
    # representation ``asm`` holds (``asm.sep is None`` => dense).
    sep = asm.sep
    facs = None
    if sep is None:
        facs = []
        for mi in range(Nm):
            Jm = np.array(M0[mi])
            di = np.where(interior, d_bar * w[mi], 0.0)
            Jm[np.diag_indices_from(Jm)] += di
            facs.append(_lu_factor_equilibrated(Jm))

    def _Jmatvec(dU_flat):
        dU = dU_flat.reshape(Ntot, Nphi)
        dUhat = np.fft.rfft(dU, axis=1)
        DdU_hat = np.fft.rfft(D_nl * dU, axis=1)        # the mode-coupling term
        out = np.empty((Ntot, Nm), dtype=complex)
        for mi in range(Nm):
            out[:, mi] = (s3.linear_apply(asm, mi, dUhat[:, mi] / w[mi])
                          + np.where(interior, DdU_hat[:, mi], 0.0))
        return np.fft.irfft(out, n=Nphi, axis=1).ravel()

    def _Minv(y_flat):
        y = y_flat.reshape(Ntot, Nphi)
        yhat = np.fft.rfft(y, axis=1)
        dUhat = np.empty((Ntot, Nm), dtype=complex)
        for mi in range(Nm):
            if sep is None:
                dvhat = _lu_solve_equilibrated(facs[mi], yhat[:, mi])
            else:
                dvhat = sep.solve(mi, yhat[:, mi].reshape(sep.Na1, sep.Nb1),
                                  asm.b).ravel()
            dUhat[:, mi] = w[mi] * dvhat                # δu_m = w · δv_m
        return np.fft.irfft(dUhat, n=Nphi, axis=1).ravel()

    n = Ntot * Nphi
    Jop = LinearOperator((n, n), matvec=_Jmatvec)
    Mop = LinearOperator((n, n), matvec=_Minv)
    b = -R_node.ravel()

    it_count = [0]

    def _cb(_pr):
        it_count[0] += 1

    dU_flat, status = gmres(Jop, b, M=Mop, rtol=gmres_rtol, atol=gmres_atol,
                            restart=gmres_restart, maxiter=gmres_maxiter,
                            callback=_cb, callback_type="pr_norm")
    dU = dU_flat.reshape(Ntot, Nphi)
    rn = float(np.max(np.abs(R_node)))
    return U + dU, _StepInfo(it_count[0], int(status), rn)


# --------------------------------------------------------------------------
# Newton–Krylov solve
# --------------------------------------------------------------------------
@dataclass
class NKInfo:
    converged: bool
    iters: int
    residual_norm: float        # the EQUILIBRATED residual (the certified number)
    raw_residual_norm: float    # the raw nodal inf-norm (roundoff-limited monitor)
    history: list               # equilibrated-residual history
    gmres_iters: list           # GMRES iterations used per Newton step


def newton_solve_nk(prob: Problem3D, sl: Slice3D, U0: Optional[np.ndarray] = None,
                    tol: float = 1e-10, max_iter: int = 20,
                    asm: Optional[Assembly3D] = None,
                    n_warmup: int = 0,
                    gmres_rtol: float = 1e-4,
                    verbose: bool = False,
                    separable: Optional[bool] = None):
    """Solve the non-axisymmetric two-centre Lichnerowicz equation by Newton–Krylov.

    Returns ``(U, NKInfo)`` with ``U`` shaped (Na+1, Nb, Nφ).  Convergence is
    driven on the **equilibrated** residual (:func:`equil_residual_inf`) — the
    well-conditioned norm the solve actually controls — which NK takes to machine
    precision and which, unlike the raw nodal monitor, does NOT rise with
    resolution.  ``info.residual_norm`` is that certified number;
    ``info.raw_residual_norm`` is the roundoff-limited raw nodal inf-norm (what
    the modified-Newton solver reports), kept for the before/after comparison.

    ``n_warmup`` optional cheap modified-Newton (``solver_3d.newton_step``) steps
    first — gets into the basin without a GMRES solve, then NK polishes.  Tracks
    the BEST iterate and stops once the equilibrated residual stagnates.

    ``separable`` picks the representation of the linear operator: the assembly is
    built without dense blocks and preconditioned by the exact inverse of the
    *linear* operator (``solver/separable.py``) instead of an LU of the linear
    operator plus the nonlinear diagonal.  Same equation, same residual, same
    certified monitor — only the linear algebra inside the step differs, and it
    typically costs one extra GMRES iteration per Newton step.  The default
    (``None``) is *auto*: separable except at Nφ=1 and with ``n_warmup``, where
    the dense blocks are needed or strictly better — see :func:`choose_separable`.
    Ignored when the caller supplies ``asm``, which already fixes the choice.
    """
    if asm is None:
        asm = s3.assemble(prob, sl,
                          separable=choose_separable(prob, separable, n_warmup))
    if asm.sep is not None and n_warmup:
        raise ValueError("n_warmup needs the dense blocks, and this assembly is "
                         "separable; rebuild it with assemble(..., separable=False)")
    shp = prob.shape
    scales = _block_scales(asm)
    U = (np.zeros((prob.Ntot2d, prob.Nphi)) if U0 is None
         else np.asarray(U0, dtype=float).reshape(prob.Ntot2d, prob.Nphi))

    history, gmres_iters = [], []
    best_U, best_rn = U, np.inf
    it = 0
    for it in range(1, max_iter + 1):
        rn = equil_residual_inf(asm, U, scales)
        history.append(rn)
        if rn < best_rn:
            best_U, best_rn = U, rn
        if verbose:
            gi = gmres_iters[-1] if gmres_iters else None
            print(f"  [NK] it={it:2d}  equil||R||={rn:.3e}  gmres={gi}")
        if rn < tol:
            break
        if it >= 3 and rn > 0.5 * history[-2]:        # stagnation near the floor
            break
        if it <= n_warmup:
            U, _ = s3.newton_step(asm, U)             # cheap modified-Newton warm-up
            gmres_iters.append(0)
        else:
            U, sinfo = newton_step_nk(asm, U, gmres_rtol=gmres_rtol)
            gmres_iters.append(sinfo.gmres_iters)

    raw = s3.nodal_residual_inf(asm, best_U)
    return best_U.reshape(shp), NKInfo(best_rn < tol, it, best_rn, raw,
                                       history, gmres_iters)


# --------------------------------------------------------------------------
# Certified polish (3-D analog of parametric.evaluate_polished)
# --------------------------------------------------------------------------
def evaluate_polished_nk(prob: Problem3D, sl: Slice3D, U_guess: np.ndarray,
                         newton_steps: int = 2, tol: float = 1e-10,
                         asm: Optional[Assembly3D] = None,
                         separable: Optional[bool] = None):
    """Warm guess + ≤``newton_steps`` NK-Newton steps → certified ``‖R‖∞`` ≤ tol.

    The 3-D certified-evaluation gate: ``info.residual_norm`` is the constraint
    residual at ``sl``, independent of how ``U_guess`` was produced (e.g. an
    interpolated / perturbed warm start).  ``max_iter`` is ``newton_steps+1`` so
    the residual AFTER the final step is the certified number (the loop measures
    the residual at the start of each iteration).
    """
    return newton_solve_nk(prob, sl, U0=U_guess, tol=tol,
                           max_iter=newton_steps + 1, asm=asm, gmres_rtol=1e-4,
                           separable=separable)
