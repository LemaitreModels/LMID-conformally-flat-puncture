"""PARASOL-3D — the ``solver_3d`` certified-ID parameter tangent ``dU/dθ_k``
(Milestone H5a; the gating de-risk of the gradient-enhanced **sparse** track).

The dense/axisymmetric path already exposes the implicit-function-theorem
tangent (``applications.sensitivity.certified_tangent`` → ``solver_abt.tangent_b/
tangent_q`` + ``tangent_chi``): the certified-ID sensitivity of the elliptic solve,
``J·(dU/dθ) = −∂R/∂θ``, one back-solve against the node's already-factored
Jacobian.  H3 decided the gradient enhancement onto the sparse (Smolyak-wrapped)
3-D family is blocked by two things; blocker (i) is that **no such tangent exists
for the non-axisymmetric ``solver_3d``**.  This module removes it.

The one genuinely-new content is the **analytic ``∂R/∂θ_k`` of the 3-D operator**:
  * the grid geometry ``∂/∂b`` — the prolate Laplacian and the ``1/ρ²``
    centrifugal term both scale as ``1/b²`` (``operators_abt._coeffs``,
    ``operators_3d``), so ``∂(L_m u_m)/∂b = −(2/b) L_m u_m`` (nodally
    ``−(2/b)·Δ_3D u`` on interior rows), exactly the ``solver_abt.tangent_b``
    geometry term lifted to 3-D;
  * the Bowen–York source ``∂Â²/∂θ`` — each per-puncture momentum/spin tensor is
    **linear** in its ``P``/``S`` vector (``source_3d._mom_tensor_vec`` /
    ``_spin_tensor_vec``), so ``∂Â^{ij}/∂θ`` is the *same* tensor evaluated with
    the derivative vector ``dP_X/dθ`` / ``dS_X/dθ`` (a chain rule that needs **no**
    finite differencing of the source), and ``∂Â²/∂θ = 2 Â^{ij} ∂Â_{ij}/∂θ``.
    For ``∂/∂b`` the tensors carry a clean scale law (``Â_P∝1/b²``, ``Â_S∝1/b³``
    at fixed ``(A,B)`` since the node/puncture positions scale as ``b``), giving
    ``∂Â^{ij}/∂b|_{P,S fixed} = −(2/b)Â_P^{ij} − (3/b)Â_S^{ij}`` — the general
    replacement for the head-on-only ``∂Â²/∂b = −4Â²/b`` of ``tangent_b``.
  * ``ψ_BL`` carries the same two pieces as the axisymmetric code:
    ``∂ψ_BL/∂b = −(ψ_BL−1)/b`` (fixed-node), ``∂ψ_BL/∂q`` via the mass map.

Because ``solver_3d`` **decouples per azimuthal mode ``m``**, the IFT solve is a
per-mode operation reusing the per-mode operators the Newton step already builds.
Two solve routes are provided (both take the shared per-slice assembly ``asm`` so
one assembly is amortised across all axes — the ``sensitivity.certified_tangent``
``asm=`` pattern):

  * ``jac="nk"`` (default) — the **full** Jacobian (the mode-coupling the modified
    Newton drops) via a GMRES solve preconditioned by the per-``m`` block, the
    tangent analog of ``solver_3d_nk.newton_step_nk``.  This is the route that
    reproduces the finite-difference of the certified solve to the residual floor
    for a genuinely non-axisymmetric slice.
  * ``jac="modified"`` — the cheap **block-diagonal** per-mode back-solve
    (``dv_m = operators_abt.solve_equilibrated(Ĵ_m, −∂R̂/∂θ|_m)``) reusing exactly
    the ``solver_3d.newton_step`` per-``m`` factored block.  For an axisymmetric /
    aligned slice (φ-independent source) it *is* the full Jacobian, so it matches
    ``jac="nk"`` bit-for-bit; the modified-vs-nk gap measures the dropped
    mode-coupling for a misaligned slice.

Add-only / standalone: imports the frozen ``solver_3d`` / ``solver_3d_nk`` /
``operators_abt`` / ``source`` / ``source_3d`` **verbatim** and defines no new
physics; the analytic derivatives are new functions here (no existing signature
changes).  numpy + scipy + the frozen siblings.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, gmres

from ..solver import solver_3d as s3
from ..solver import solver_3d_nk as s3nk  # noqa: F401  (reduce-to gate uses the NK node/solve)
from ..solver import operators_abt as ops
from ..solver import source
from ..solver import source_3d
from ..solver.solver_3d import Assembly3D, Problem3D, Slice3D


# the axes for which a direct (fixed-physical-vector) 3-D tangent is defined
TANGENT_AXES_3D = ("b", "q", "P", "P_x",
                   "S_x", "S_z", "S_mag", "theta_S",
                   "S_Ax", "S_Ay", "S_Az", "S_Bx", "S_By", "S_Bz",
                   # dimensionless-spin axes (chi = S/m^2); affine relabel of the
                   # S_* axes at fixed masses, dS_Xi/dchi_Xi = m_X^2 (chi rebuild).
                   "chi_Ax", "chi_Ay", "chi_Az", "chi_Bx", "chi_By", "chi_Bz")


# ==========================================================================
# 1.  Physical-vector derivatives  dP_X/dθ, dS_X/dθ  (the direct-axis chain rule)
# ==========================================================================
def _dvec_dtheta(sl: Slice3D, name: str) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Derivatives of the physical momentum/spin vectors w.r.t. axis ``name``.

    Returns ``(dP_A, dP_B, dS_A, dS_B)`` (each a length-3 ``np.ndarray``) for the
    **direct** (fixed-mass, head-on convention) interpretation that mirrors
    ``parametric_nd_3d.theta_to_slice3d`` in its non-QC branch and matches
    ``solver_abt.tangent_b/q``'s "momenta fixed" semantics.  ``b``/``q`` return
    all-zero vectors (their source effect is the ``Â²`` scale law / ``ψ_BL``,
    handled in :func:`_dA2_dtheta` / :func:`_dpsi_dtheta`).
    """
    z = np.zeros(3)
    dPA = np.zeros(3); dPB = np.zeros(3); dSA = np.zeros(3); dSB = np.zeros(3)
    S_A = np.asarray(sl.S_A_vec, dtype=float)
    if name in ("b", "q"):
        return dPA, dPB, dSA, dSB
    if name == "P":                       # head-on infall magnitude (P_A_z = −P)
        dPA[2] = -1.0; dPB[2] = +1.0
    elif name == "P_x":                   # transverse (orbital) momentum
        dPA[0] = +1.0; dPB[0] = -1.0
    elif name == "S_x":                   # spin on A, Cartesian x
        dSA[0] = 1.0
    elif name == "S_z":                   # spin on A, Cartesian z
        dSA[2] = 1.0
    elif name == "S_mag":                 # spin on A, polar magnitude
        Smag = float(np.hypot(S_A[0], S_A[2]))
        if Smag <= 0.0:
            raise ValueError("S_mag tangent needs a non-zero spin (direction "
                             "undefined at S_mag=0 from the slice alone)")
        dSA[:] = np.array([S_A[0], 0.0, S_A[2]]) / Smag
    elif name == "theta_S":               # spin on A, polar tilt (DEGREES)
        # S_x=Smag sinθ, S_z=Smag cosθ ⇒ dS_x/dθ=S_z, dS_z/dθ=−S_x (radians)
        dSA[:] = np.array([S_A[2], 0.0, -S_A[0]]) * (np.pi / 180.0)
    elif name in ("S_Ax", "S_Ay", "S_Az"):
        dSA["xyz".index(name[-1])] = 1.0
    elif name in ("S_Bx", "S_By", "S_Bz"):
        dSB["xyz".index(name[-1])] = 1.0
    elif name in ("chi_Ax", "chi_Ay", "chi_Az"):   # dimensionless spin on A
        # S_Ai = chi_Ai · m_A^2 (theta_to_slice3d), so dS_Ai/dchi_Ai = m_A^2 at
        # fixed masses; the chain then flows verbatim through _dA2_dtheta (direct
        # source) and sensitivity_3d_qc._dargs_dtheta (the QC-momenta chain).
        dSA["xyz".index(name[-1])] = float(sl.m_A) ** 2
    elif name in ("chi_Bx", "chi_By", "chi_Bz"):   # dimensionless spin on B
        dSB["xyz".index(name[-1])] = float(sl.m_B) ** 2
    else:
        raise ValueError(f"no 3-D tangent for axis {name!r}; "
                         f"available: {TANGENT_AXES_3D}")
    return dPA, dPB, dSA, dSB


# ==========================================================================
# 2.  Analytic  ∂Â²/∂θ  over the (A,B,φ) node cloud  (the genuinely-new content)
# ==========================================================================
def _dA2_dtheta(asm: Assembly3D, phi: np.ndarray, sl: Slice3D,
                name: str) -> np.ndarray:
    """``∂(Â_{ij}Â^{ij})/∂θ`` on the node cloud — shape ``(Ntot2d, Nφ)``.

    ``Â²(θ) = Â^{ij}Â_{ij}`` and ``∂Â²/∂θ = 2 Â^{ij} ∂Â_{ij}/∂θ``; because each
    Bowen–York tensor is linear in its ``P``/``S`` vector, ``∂Â_{ij}/∂θ`` is the
    same ``_mom_tensor_vec`` / ``_spin_tensor_vec`` built from the derivative
    vectors (:func:`_dvec_dtheta`).  For ``b`` the tensors scale as ``Â_P∝1/b²``,
    ``Â_S∝1/b³`` at fixed ``(A,B)`` (node & puncture positions ∝ b), so
    ``∂Â/∂b = −(2/b)Â_P − (3/b)Â_S``.  ``q`` leaves ``Â²`` unchanged (the BY
    tensors are mass-independent) in the direct interpretation.
    """
    rho = np.asarray(asm.rho, dtype=float).ravel()
    z = np.asarray(asm.z, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    finite = np.isfinite(rho) & np.isfinite(z)
    rho_s = np.where(finite, rho, 1.0)
    z_s = np.where(finite, z, 0.0)
    Ntot, Nphi = rho.size, phi.size
    xA = np.array([0.0, 0.0, sl.b])
    xB = np.array([0.0, 0.0, -sl.b])

    if name == "q":
        return np.zeros((Ntot, Nphi))

    if name == "b":
        out = np.empty((Ntot, Nphi))
        for k in range(Nphi):
            X = np.stack([rho_s * np.cos(phi[k]), rho_s * np.sin(phi[k]), z_s],
                         axis=1)
            TP = (source_3d._mom_tensor_vec(X, xA, sl.P_A_vec)
                  + source_3d._mom_tensor_vec(X, xB, sl.P_B_vec))
            TS = (source_3d._spin_tensor_vec(X, xA, sl.S_A_vec)
                  + source_3d._spin_tensor_vec(X, xB, sl.S_B_vec))
            T = TP + TS
            dT = -(2.0 / sl.b) * TP - (3.0 / sl.b) * TS
            a2 = 2.0 * np.sum(T * dT, axis=(1, 2))
            out[:, k] = np.where(finite & np.isfinite(a2), a2, 0.0)
        return out

    dPA, dPB, dSA, dSB = _dvec_dtheta(sl, name)
    out = np.empty((Ntot, Nphi))
    for k in range(Nphi):
        X = np.stack([rho_s * np.cos(phi[k]), rho_s * np.sin(phi[k]), z_s], axis=1)
        T = source_3d.A_full_tensor_vec(X, sl.b, sl.P_A_vec, sl.P_B_vec,
                                        sl.S_A_vec, sl.S_B_vec)
        dT = np.zeros_like(T)
        if np.any(dPA):
            dT = dT + source_3d._mom_tensor_vec(X, xA, dPA)
        if np.any(dPB):
            dT = dT + source_3d._mom_tensor_vec(X, xB, dPB)
        if np.any(dSA):
            dT = dT + source_3d._spin_tensor_vec(X, xA, dSA)
        if np.any(dSB):
            dT = dT + source_3d._spin_tensor_vec(X, xB, dSB)
        a2 = 2.0 * np.sum(T * dT, axis=(1, 2))
        out[:, k] = np.where(finite & np.isfinite(a2), a2, 0.0)
    return out


def _dpsi_dtheta(asm: Assembly3D, sl: Slice3D, name: str, M_tot: float) -> np.ndarray:
    """``∂ψ_BL/∂θ`` on the node cloud — shape ``(Ntot2d,)`` (φ-independent).

    ``b``: ``−(ψ_BL−1)/b`` (fixed-node scale law, as ``solver_abt.tangent_b``).
    ``q``: the mass-map piece ``dm_A/dq/(2r_A) + dm_B/dq/(2r_B)``.  Else 0.
    """
    if name == "b":
        return -(np.asarray(asm.psi, dtype=float) - 1.0) / sl.b
    if name == "q":
        q = sl.m_A / sl.m_B
        dmA = M_tot / (1.0 + q) ** 2
        dmB = -M_tot / (1.0 + q) ** 2
        finite = np.isfinite(asm.rho)
        rho_s = np.where(finite, asm.rho, 1.0)
        z_s = np.where(finite, asm.z, 0.0)
        inv2rA = np.where(finite, np.array(source.dpsiBL_dmA(rho_s, z_s, sl.b)), 0.0)
        inv2rB = np.where(finite, np.array(source.dpsiBL_dmB(rho_s, z_s, sl.b)), 0.0)
        return dmA * inv2rA + dmB * inv2rB
    return np.zeros(asm.psi.shape)


def dR_dtheta_node(prob: Problem3D, asm: Assembly3D, U: np.ndarray, sl: Slice3D,
                   name: str, M_tot: float) -> np.ndarray:
    """The analytic nodal residual derivative ``∂R/∂θ`` — shape ``(Ntot2d, Nφ)``.

    ``R_node = Δ_3D u + interior·S``, ``S = ⅛(ψ+u)^{-7}Â²``.  So on interior rows
        ``∂R/∂θ = [geometry ∂/∂b] + ⅛[ −7(ψ+u)^{-8} ∂ψ/∂θ Â² + (ψ+u)^{-7} ∂Â²/∂θ ]``
    and 0 on the (θ-independent) BC rows.  The geometry term (present only for the
    ``b`` axis) is ``−(2/b)·Δ_3D u`` reconstructed from the per-m linear operator.
    """
    U = np.asarray(U, dtype=float).reshape(prob.Ntot2d, prob.Nphi)
    interior = asm.interior
    base = asm.psi[:, None] + U
    dpsi = _dpsi_dtheta(asm, sl, name, M_tot)          # (Ntot2d,)
    dA2 = _dA2_dtheta(asm, prob.phi, sl, name)         # (Ntot2d, Nφ)
    dS_nl = 0.125 * (-7.0 * base ** (-8.0) * dpsi[:, None] * asm.A2
                     + base ** (-7.0) * dA2)
    dR_int = dS_nl
    if name == "b":
        # −(2/b)·Δ_3D u on interior rows: reconstruct the linear operator action
        Uhat = np.fft.rfft(U, axis=1)
        linhat = np.empty((prob.Ntot2d, asm.m_vals.size), dtype=complex)
        for mi in range(asm.m_vals.size):
            linhat[:, mi] = asm.M0[mi] @ (Uhat[:, mi] / asm.w[mi])
        lap_nodal = np.fft.irfft(linhat, n=prob.Nphi, axis=1)
        dR_int = dR_int + (-2.0 / sl.b) * lap_nodal
    return np.where(interior[:, None], dR_int, 0.0)


# ==========================================================================
# 3.  The per-mode tangent solves  (block-diagonal modified / full-J NK)
# ==========================================================================
def _tangent_solve_modified(asm: Assembly3D, U: np.ndarray,
                            dR_node: np.ndarray) -> np.ndarray:
    """``J_mod·dU = −∂R/∂θ`` — one per-mode back-solve against the ``newton_step``
    block ``Ĵ_m = M0_m + diag(interior·d̄·w_m)`` (``d̄`` = φ-averaged source deriv).

    The cheap plan-literal tangent: ``dv_m = operators_abt.solve_equilibrated(Ĵ_m,
    −∂R̂/∂θ|_m)`` reusing exactly the factored per-``m`` block the Newton step
    builds.  For a φ-independent source (axisymmetric / aligned) this IS the full
    Jacobian (``d̄ = D_nl``), so it matches :func:`_tangent_solve_nk` bit-for-bit.
    """
    Nphi = U.shape[1]
    _, base = s3._nl_source(asm, U)
    D_nl = -0.875 * base ** (-8.0) * asm.A2
    d_bar = D_nl.mean(axis=1)
    dRm = np.fft.rfft(dR_node, axis=1)
    dUhat = np.empty((asm.interior.size, asm.m_vals.size), dtype=complex)
    for mi in range(asm.m_vals.size):
        Jm = np.array(asm.M0[mi])
        di = np.where(asm.interior, d_bar * asm.w[mi], 0.0)
        Jm[np.diag_indices_from(Jm)] += di
        dvhat = ops.solve_equilibrated(Jm, -dRm[:, mi])
        dUhat[:, mi] = asm.w[mi] * dvhat
    return np.fft.irfft(dUhat, n=Nphi, axis=1)


def _tangent_solve_nk(asm: Assembly3D, U: np.ndarray, dR_node: np.ndarray,
                      gmres_rtol: float = 1e-11, gmres_atol: float = 1e-14,
                      gmres_restart: int = 60, gmres_maxiter: int = 200):
    """``J·dU = −∂R/∂θ`` with the **full** Jacobian (mode-coupling included) via a
    preconditioned GMRES — the tangent analog of ``solver_3d_nk.newton_step_nk``.

    ``(J δu)_m = M0_m(δû_m/w_m) + interior·rfft(D_nl·δu)_m`` (matrix-free, exact),
    preconditioned by the per-``m`` block ``Ĵ_m`` (a triangular solve on the
    row-equilibrated LU).  Returns ``(dU, gmres_iters)``.  With ``Nφ=1`` (or a
    φ-independent source) the preconditioner IS ``J`` and GMRES converges in one
    step — reducing to the block-diagonal solve.
    """
    Nphi = U.shape[1]
    Ntot = asm.interior.size
    Nm = asm.m_vals.size
    interior = asm.interior
    M0, w = asm.M0, asm.w

    _, base = s3._nl_source(asm, U)
    D_nl = -0.875 * base ** (-8.0) * asm.A2
    d_bar = D_nl.mean(axis=1)

    facs = []
    for mi in range(Nm):
        Jm = np.array(M0[mi])
        di = np.where(interior, d_bar * w[mi], 0.0)
        Jm[np.diag_indices_from(Jm)] += di
        facs.append(s3nk._lu_factor_equilibrated(Jm))

    def _Jmatvec(dU_flat):
        dU = dU_flat.reshape(Ntot, Nphi)
        dUhat = np.fft.rfft(dU, axis=1)
        DdU_hat = np.fft.rfft(D_nl * dU, axis=1)
        out = np.empty((Ntot, Nm), dtype=complex)
        for mi in range(Nm):
            out[:, mi] = (M0[mi] @ (dUhat[:, mi] / w[mi])
                          + np.where(interior, DdU_hat[:, mi], 0.0))
        return np.fft.irfft(out, n=Nphi, axis=1).ravel()

    def _Minv(y_flat):
        y = y_flat.reshape(Ntot, Nphi)
        yhat = np.fft.rfft(y, axis=1)
        dUhat = np.empty((Ntot, Nm), dtype=complex)
        for mi in range(Nm):
            dvhat = s3nk._lu_solve_equilibrated(facs[mi], yhat[:, mi])
            dUhat[:, mi] = w[mi] * dvhat
        return np.fft.irfft(dUhat, n=Nphi, axis=1).ravel()

    n = Ntot * Nphi
    Jop = LinearOperator((n, n), matvec=_Jmatvec)
    Mop = LinearOperator((n, n), matvec=_Minv)
    it_count = [0]

    def _cb(_pr):
        it_count[0] += 1

    dU_flat, _status = gmres(Jop, -np.asarray(dR_node).ravel(), M=Mop,
                             rtol=gmres_rtol, atol=gmres_atol,
                             restart=gmres_restart, maxiter=gmres_maxiter,
                             callback=_cb, callback_type="pr_norm")
    return dU_flat.reshape(Ntot, Nphi), it_count[0]


# ==========================================================================
# 4.  The deliverable — certified_tangent_3d
# ==========================================================================
def certified_tangent_3d(prob: Problem3D, U: np.ndarray, sl: Slice3D, name: str,
                         M_tot: float, asm: Optional[Assembly3D] = None, *,
                         jac: str = "nk", gmres_rtol: float = 1e-11,
                         return_iters: bool = False):
    """The certified-ID sensitivity ``dU/dθ_name`` of the 3-D solve (IFT tangent).

    Solves ``J·(dU/dθ) = −∂R/∂θ`` for axis ``name`` ∈ :data:`TANGENT_AXES_3D`,
    reusing the shared per-slice assembly ``asm`` (pass one ``s3.assemble(prob, sl)``
    and reuse it across every axis — the ``sensitivity.certified_tangent`` ``asm=``
    pattern).  ``jac="nk"`` (default) uses the full Jacobian (matches the FD of the
    certified solve to the residual floor); ``jac="modified"`` the cheap
    block-diagonal per-mode back-solve.  Returns ``dU`` shaped ``prob.shape``
    (``(Na+1, Nb, Nφ)``); with ``return_iters`` also the GMRES iteration count
    (``0`` for the modified route).
    """
    if asm is None:
        asm = s3.assemble(prob, sl)
    Uarr = np.asarray(U, dtype=float).reshape(prob.Ntot2d, prob.Nphi)
    dR = dR_dtheta_node(prob, asm, Uarr, sl, name, M_tot)
    if jac == "modified":
        dU = _tangent_solve_modified(asm, Uarr, dR)
        iters = 0
    elif jac == "nk":
        dU, iters = _tangent_solve_nk(asm, Uarr, dR, gmres_rtol=gmres_rtol)
    else:
        raise ValueError(f"jac must be 'nk' or 'modified', got {jac!r}")
    dU = dU.reshape(prob.shape)
    return (dU, iters) if return_iters else dU


# ==========================================================================
# 5.  Slice perturbation for the FD validation oracle (same convention as §1)
# ==========================================================================
def perturb_slice_3d(sl: Slice3D, name: str, dval: float, M_tot: float) -> Slice3D:
    """Return ``sl`` perturbed by ``dval`` along axis ``name`` (direct interp).

    The finite-difference companion of :func:`_dvec_dtheta` / :func:`_dpsi_dtheta`
    / :func:`_dA2_dtheta`: it perturbs exactly the physical quantity those
    derivatives are taken with respect to, so central FD of ``newton_solve`` at
    ``sl±h`` is the ground truth for :func:`certified_tangent_3d`.
    """
    PA = list(map(float, sl.P_A_vec)); PB = list(map(float, sl.P_B_vec))
    SA = list(map(float, sl.S_A_vec)); SB = list(map(float, sl.S_B_vec))
    b, m_A, m_B = sl.b, sl.m_A, sl.m_B
    if name == "b":
        b = sl.b + dval
    elif name == "q":
        M = sl.M
        q = (sl.m_A / sl.m_B) + dval
        m_A = M * q / (1.0 + q)
        m_B = M / (1.0 + q)
    elif name == "P":
        P0 = -PA[2]
        PA[2] = -(P0 + dval); PB[2] = +(P0 + dval)
    elif name == "P_x":
        Px0 = PA[0]
        PA[0] = Px0 + dval; PB[0] = -(Px0 + dval)
    elif name == "S_x":
        SA[0] += dval
    elif name == "S_z":
        SA[2] += dval
    elif name == "S_mag":
        Smag = float(np.hypot(SA[0], SA[2]))
        th = np.arctan2(SA[0], SA[2])
        SA[0] = (Smag + dval) * np.sin(th); SA[2] = (Smag + dval) * np.cos(th)
    elif name == "theta_S":
        Smag = float(np.hypot(SA[0], SA[2]))
        th = np.degrees(np.arctan2(SA[0], SA[2])) + dval
        SA[0] = Smag * np.sin(np.radians(th)); SA[2] = Smag * np.cos(np.radians(th))
    elif name in ("S_Ax", "S_Ay", "S_Az"):
        SA["xyz".index(name[-1])] += dval
    elif name in ("S_Bx", "S_By", "S_Bz"):
        SB["xyz".index(name[-1])] += dval
    elif name in ("chi_Ax", "chi_Ay", "chi_Az"):   # perturb dimensionless spin on A
        SA["xyz".index(name[-1])] += dval * float(m_A) ** 2
    elif name in ("chi_Bx", "chi_By", "chi_Bz"):   # perturb dimensionless spin on B
        SB["xyz".index(name[-1])] += dval * float(m_B) ** 2
    else:
        raise ValueError(f"no 3-D tangent for axis {name!r}")
    return Slice3D(b=b, m_A=m_A, m_B=m_B, P_A_vec=tuple(PA), P_B_vec=tuple(PB),
                   S_A_vec=tuple(SA), S_B_vec=tuple(SB))


def fd_tangent_3d(prob: Problem3D, sl: Slice3D, name: str, M_tot: float, *,
                  h: float = 1e-4, solver: str = "modified", tol: float = 1e-12,
                  max_iter: int = 40) -> np.ndarray:
    """Central FD of the certified solve ``d/dθ_name[newton_solve→U]`` (the oracle).

    ``solver='modified'`` uses ``solver_3d.newton_solve`` (the modified-Newton
    field the sweep stores); ``solver='nk'`` uses ``solver_3d_nk.newton_solve_nk``
    (residual to machine → a cleaner ground truth, the Q0/R2 node-floor check).
    Returns ``dU/dθ`` shaped ``prob.shape``.  **Validation only** (never the
    production tangent).
    """
    def solve(s):
        if solver == "nk":
            U, _ = s3nk.newton_solve_nk(prob, s, tol=tol, max_iter=max_iter)
        else:
            U, _ = s3.newton_solve(prob, s, tol=tol, max_iter=max_iter)
        return np.asarray(U).reshape(prob.shape)

    Up = solve(perturb_slice_3d(sl, name, +h, M_tot))
    Um = solve(perturb_slice_3d(sl, name, -h, M_tot))
    return (Up - Um) / (2.0 * h)
