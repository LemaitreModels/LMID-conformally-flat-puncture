"""PARASOL-3D — the QC certified-ID **second-order cross tangent**
``∂²U/∂θ_i∂θ_j`` between two enhanced spin axes (the mixed-partial the
gradient-only Hermite–Smolyak construction drops).

The second-order sibling of the first-order QC tangent
(:mod:`applications.sensitivity_3d_qc`).  The gradient-enhanced sparse model
(:mod:`parametric.hermite_smolyak`) carries only the first tangents ``dU/dθ_k``
and therefore drops the single mixed partial ``∂²U/∂θ_i∂θ_j`` between the two
enhanced axes.  This module supplies exactly that field, so a *full bilinear*
Hermite on the enhanced ``(χ_Ay, χ_By)`` subspace can be assembled.

Why this is a cheap analytic POST-PROCESS (not a bigger lift / not autodiff
through the solve)
-------------------------------------------------------------------------------
The ``solver_3d`` residual has the closed form (``solver_3d._nl_source``)

    R = Δ_3D u + interior · S,   S = ⅛ (ψ_BL(θ) + u)^{-7} Â²(θ).

Along the solution manifold ``R(U(θ), θ) ≡ 0``; differentiating twice gives the
**second-order forward-sensitivity equation** (the same factored Jacobian
``J = ∂R/∂U`` the first tangent uses, one extra back-solve):

    J · (∂²U/∂θ_i∂θ_j) = −[ ∂²R/∂θ_i∂θ_j
                            + (∂²R/∂θ_i∂U)(dU/dθ_j)
                            + (∂²R/∂θ_j∂U)(dU/dθ_i)
                            + (∂²R/∂U²)[dU/dθ_i, dU/dθ_j] ].

For the two **enhanced spin axes** ``χ_Ay, χ_By`` (at fixed ``b, q``) the only
θ-dependence of ``R`` is through ``Â²(θ)`` — ``ψ_BL`` and the grid geometry are
spin-independent (``∂ψ/∂χ = 0``, no ``∂/∂b`` geometry term).  So every needed
second derivative is a node-diagonal (physical-space) function of ``base = ψ+u``
times a source second derivative:

    ∂²R/∂U²        = interior · 7 base^{-9} Â²                         (= R_UU)
    ∂²R/∂U∂θ_i     = interior · (−7/8) base^{-8} (Â²)_i               (= R_Ui)
    ∂²R/∂θ_i∂θ_j   = interior · (1/8)  base^{-7} (Â²)_ij              (= R_ij)

with ``(Â²)_i = 2 Â^{kl} (dÂ_i)_{kl}`` (the first-order source derivative already
assembled in :func:`sensitivity_3d_qc.certified_tangent_3d_qc`) and

    (Â²)_ij = 2 (dÂ_i)^{kl}(dÂ_j)_{kl} + 2 Â^{kl} (d²Â_ij)_{kl}.

Each Bowen–York tensor is **linear** in its momentum/spin vector, so ``dÂ_i`` is
the same builder evaluated on the first-derivative vectors (spin chain
``_dvec_dtheta`` + the QC-momenta chain ``dP_dtheta_qc``), and ``d²Â_ij`` is the
builder on the **second-derivative** vectors — for spin axes ``d²S = 0`` and the
only nonzero piece is the momenta ``d²P`` from the ``qc_momenta`` Hessian
(``jax.jacfwd²`` of the closed-form PN momenta; ≈0 in practice since the PN
momenta have no ``χ_Ay·χ_By`` spin–spin cross term, but carried exactly so the
routine is bullet-proof and self-validating).  ``d²Â_ij`` therefore needs **no**
finite differencing of the source.

The back-solve reuses :func:`sensitivity_3d._tangent_solve_nk` /
``_tangent_solve_modified`` **verbatim** (same factored ``J``), and the two first
tangents ``dU/dθ_i``, ``dU/dθ_j`` are either supplied (the shipped model already
stores them in ``node_dU`` — the Milestone-3 post-process) or computed on demand
via :func:`sensitivity_3d_qc.certified_tangent_3d_qc`.

Scope / guard.  This routine is exact for a pair of **spin axes** (``chi_*`` /
Cartesian ``S_*``, at fixed ``b, q``): there the geometry/``ψ`` second-order
terms all vanish.  It **raises** for a pair involving ``b``, ``q``, ``S_mag`` or
``theta_S`` (which need extra geometry/``ψ``/nonlinear-vector second-order terms)
— outside the ``(χ_Ay, χ_By)`` target.

Add-only / standalone: imports the committed ``sensitivity_3d`` /
``sensitivity_3d_qc`` / ``solver_3d`` / ``source_3d`` **verbatim**; defines no new
physics beyond the second derivative of the (already-committed) PN-momenta twin.
numpy + jax.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..solver import solver_3d as s3
from ..solver import source_3d
from . import sensitivity_3d as s3d
from . import sensitivity_3d_qc as s3dqc


# axis pairs supported by the second-order cross tangent: pure spin axes only
# (the source-only case; no geometry / ψ / nonlinear-vector second-order terms).
_LINEAR_SPIN_AXES = ("S_x", "S_z",
                     "S_Ax", "S_Ay", "S_Az", "S_Bx", "S_By", "S_Bz",
                     "chi_Ax", "chi_Ay", "chi_Az", "chi_Bx", "chi_By", "chi_Bz")


# ==========================================================================
# 1.  Second-derivative of the physical momentum/spin vectors  d²P, d²S
# ==========================================================================
def _qc_momenta_hessian(sl: s3.Slice3D) -> np.ndarray:
    """``∂²(P_A, P_B)/∂args²`` (6×9×9) via forward-over-forward autodiff of the
    committed PN-momenta twin :func:`sensitivity_3d_qc._qc_momenta_jax`.

    ``args = [b, m_A, m_B, S_Ax, S_Ay, S_Az, S_Bx, S_By, S_Bz]``; exact (not
    finite-differenced).  ``H[p, a, b] = ∂²P_p/∂args_a∂args_b``."""
    args = jnp.asarray(s3dqc._qc_args(sl))
    return np.asarray(jax.jacfwd(jax.jacfwd(s3dqc._qc_momenta_jax))(args))


def dP2_dtheta_qc(sl: s3.Slice3D, name_i: str, name_j: str,
                  M_tot: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(d²P_A/∂θ_i∂θ_j, d²P_B/∂θ_i∂θ_j)`` (two 3-vectors) for QC spin axes.

    ``d²P/∂θ_i∂θ_j = (dargs/dθ_i)ᵀ H (dargs/dθ_j) + ∂P/∂args · (d²args/∂θ_i∂θ_j)``.
    For a pair of spin axes at fixed ``b, q`` the chain factors ``dargs/dθ`` are
    **constant** (a spin-vector component with a fixed ``m^2`` factor), so
    ``d²args/∂θ_i∂θ_j = 0`` and only the Hessian term survives.  This is ≈0 in
    practice (the committed PN momenta carry no ``χ_Ay·χ_By`` spin–spin term) but
    is computed exactly so the routine self-validates."""
    darg_i = s3dqc._dargs_dtheta(sl, name_i, M_tot)      # (9,)  constant for spin axes
    darg_j = s3dqc._dargs_dtheta(sl, name_j, M_tot)
    H = _qc_momenta_hessian(sl)                          # (6,9,9)
    dP2 = np.einsum("pab,a,b->p", H, darg_i, darg_j)     # (6,)
    return dP2[0:3], dP2[3:6]


# ==========================================================================
# 2.  First-derivative Bowen–York tensor dÂ over the (A,B,φ) node cloud
# ==========================================================================
def _dA_tensor_qc(X: np.ndarray, sl: s3.Slice3D, name: str, M_tot: float) -> np.ndarray:
    """The full QC first-derivative tensor ``dÂ_{ij} = ∂Â_{ij}/∂θ`` at Cartesian
    points ``X`` (Npts,3) — shape (Npts, 3, 3).

    ``dÂ = Â_S(x_X, dS_X) + Σ_Y Â_P(x_Y, dP_Y/dθ)`` — the spin chain
    (``sensitivity_3d._dvec_dtheta``) plus the QC momenta chain
    (``sensitivity_3d_qc.dP_dtheta_qc``); each Bowen–York tensor is linear in its
    vector, so its θ-derivative is the same builder on the derivative vector.
    Contracting ``2 Â^{kl} dÂ_{kl}`` reproduces
    ``certified_tangent_3d_qc``'s ``(Â²)_θ`` exactly (a numerical cross-check gate).
    """
    xA = np.array([0.0, 0.0, sl.b])
    xB = np.array([0.0, 0.0, -sl.b])
    _dPA_dir, _dPB_dir, dSA, dSB = s3d._dvec_dtheta(sl, name)   # spin-vector chain
    dP_A, dP_B = s3dqc.dP_dtheta_qc(sl, name, M_tot)            # QC momenta chain
    dT = np.zeros((X.shape[0], 3, 3))
    if np.any(dSA):
        dT = dT + source_3d._spin_tensor_vec(X, xA, dSA)
    if np.any(dSB):
        dT = dT + source_3d._spin_tensor_vec(X, xB, dSB)
    if np.any(dP_A):
        dT = dT + source_3d._mom_tensor_vec(X, xA, dP_A)
    if np.any(dP_B):
        dT = dT + source_3d._mom_tensor_vec(X, xB, dP_B)
    return dT


def _d2A_tensor_qc(X: np.ndarray, sl: s3.Slice3D, name_i: str, name_j: str,
                   M_tot: float) -> np.ndarray:
    """The QC second-derivative tensor ``d²Â_{ij} = ∂²Â_{ij}/∂θ_i∂θ_j`` at points
    ``X`` (Npts,3) — shape (Npts, 3, 3).

    For a pair of spin axes the spin second-derivative vanishes (``S`` is linear
    in its own axis, and ``χ_Ay``/``χ_By`` act on different punctures), so the
    only piece is the momenta ``d²P`` from :func:`dP2_dtheta_qc`; each Bowen–York
    momentum tensor is linear in ``P`` so ``d²Â`` is the builder on ``d²P``."""
    xA = np.array([0.0, 0.0, sl.b])
    xB = np.array([0.0, 0.0, -sl.b])
    dP2_A, dP2_B = dP2_dtheta_qc(sl, name_i, name_j, M_tot)
    d2T = np.zeros((X.shape[0], 3, 3))
    if np.any(dP2_A):
        d2T = d2T + source_3d._mom_tensor_vec(X, xA, dP2_A)
    if np.any(dP2_B):
        d2T = d2T + source_3d._mom_tensor_vec(X, xB, dP2_B)
    return d2T


def _source_second_derivs(asm: s3.Assembly3D, phi: np.ndarray, sl: s3.Slice3D,
                          name_i: str, name_j: str, M_tot: float):
    """The three source-tensor contractions over the node cloud — each (Ntot2d, Nφ):

        A2_i  = ∂Â²/∂θ_i          = 2 Â:dÂ_i
        A2_j  = ∂Â²/∂θ_j          = 2 Â:dÂ_j
        A2_ij = ∂²Â²/∂θ_i∂θ_j     = 2 dÂ_i:dÂ_j + 2 Â:d²Â_ij

    Non-finite rows (the A=1 infinity edge) are returned as 0 (BC rows, masked in
    the solve)."""
    rho = np.asarray(asm.rho, dtype=float).ravel()
    z = np.asarray(asm.z, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    finite = np.isfinite(rho) & np.isfinite(z)
    rho_s = np.where(finite, rho, 1.0)
    z_s = np.where(finite, z, 0.0)
    Ntot, Nphi = rho.size, phi.size
    A2_i = np.empty((Ntot, Nphi))
    A2_j = np.empty((Ntot, Nphi))
    A2_ij = np.empty((Ntot, Nphi))
    for k in range(Nphi):
        X = np.stack([rho_s * np.cos(phi[k]), rho_s * np.sin(phi[k]), z_s], axis=1)
        T = source_3d.A_full_tensor_vec(X, sl.b, sl.P_A_vec, sl.P_B_vec,
                                        sl.S_A_vec, sl.S_B_vec)
        dTi = _dA_tensor_qc(X, sl, name_i, M_tot)
        dTj = _dA_tensor_qc(X, sl, name_j, M_tot)
        d2Tij = _d2A_tensor_qc(X, sl, name_i, name_j, M_tot)
        a2i = 2.0 * np.sum(T * dTi, axis=(1, 2))
        a2j = 2.0 * np.sum(T * dTj, axis=(1, 2))
        a2ij = 2.0 * np.sum(dTi * dTj, axis=(1, 2)) + 2.0 * np.sum(T * d2Tij, axis=(1, 2))
        ok = finite
        A2_i[:, k] = np.where(ok & np.isfinite(a2i), a2i, 0.0)
        A2_j[:, k] = np.where(ok & np.isfinite(a2j), a2j, 0.0)
        A2_ij[:, k] = np.where(ok & np.isfinite(a2ij), a2ij, 0.0)
    return A2_i, A2_j, A2_ij


# ==========================================================================
# 3.  The deliverable — the second-order cross tangent  ∂²U/∂θ_i∂θ_j
# ==========================================================================
def cross_tangent_3d_qc(prob: s3.Problem3D, U: np.ndarray, sl: s3.Slice3D,
                        name_i: str, name_j: str, M_tot: float,
                        dU_i: Optional[np.ndarray] = None,
                        dU_j: Optional[np.ndarray] = None,
                        asm: Optional[s3.Assembly3D] = None, *,
                        jac: str = "nk", gmres_rtol: float = 1e-11,
                        return_iters: bool = False):
    """``∂²U/∂θ_i∂θ_j`` — the QC certified-ID second-order cross tangent.

    Solves ``J·U_ij = −[R_ij + R_Ui·U_j + R_Uj·U_i + R_UU·U_i·U_j]`` for the pair
    of spin axes ``(name_i, name_j)`` (both in :data:`_LINEAR_SPIN_AXES`; ``b``,
    ``q``, ``S_mag``, ``theta_S`` raise), reusing the shared per-slice assembly
    ``asm`` and the H5a full-``J`` (``jac='nk'``) / block-diagonal
    (``jac='modified'``) tangent solvers.

    ``dU_i``/``dU_j`` are the first tangents ``dU/dθ_i``/``dU/dθ_j`` — pass the
    shipped model's stored ``node_dU`` slices (the Milestone-3 post-process, no
    re-solve), or leave ``None`` to compute them here via
    ``certified_tangent_3d_qc`` (same ``jac``).  ``sl`` must be the physical QC
    slice (``theta_to_slice3d(..., fixed={'qc':1})``).  Returns ``U_ij`` shaped
    ``prob.shape``; with ``return_iters`` also the GMRES iteration count.
    """
    for nm in (name_i, name_j):
        if nm not in _LINEAR_SPIN_AXES:
            raise NotImplementedError(
                f"cross_tangent_3d_qc supports spin-axis pairs only "
                f"(got {nm!r}); axes b/q/S_mag/theta_S need extra geometry/ψ/"
                f"nonlinear-vector second-order terms — {_LINEAR_SPIN_AXES}")
    if asm is None:
        asm = s3.assemble(prob, sl)
    Uarr = np.asarray(U, dtype=float).reshape(prob.Ntot2d, prob.Nphi)

    if dU_i is None:
        dU_i = s3dqc.certified_tangent_3d_qc(prob, Uarr, sl, name_i, M_tot,
                                             asm=asm, jac=jac, gmres_rtol=gmres_rtol)
    if dU_j is None:
        dU_j = s3dqc.certified_tangent_3d_qc(prob, Uarr, sl, name_j, M_tot,
                                             asm=asm, jac=jac, gmres_rtol=gmres_rtol)
    Ui = np.asarray(dU_i, dtype=float).reshape(prob.Ntot2d, prob.Nphi)
    Uj = np.asarray(dU_j, dtype=float).reshape(prob.Ntot2d, prob.Nphi)

    base = asm.psi[:, None] + Uarr
    A2 = asm.A2
    A2_i, A2_j, A2_ij = _source_second_derivs(asm, prob.phi, sl, name_i, name_j, M_tot)

    R_ij = 0.125 * base ** (-7.0) * A2_ij
    R_Ui = -0.875 * base ** (-8.0) * A2_i
    R_Uj = -0.875 * base ** (-8.0) * A2_j
    R_UU = 7.0 * base ** (-9.0) * A2
    bracket = R_ij + R_Ui * Uj + R_Uj * Ui + R_UU * Ui * Uj
    dR_node = np.where(asm.interior[:, None], bracket, 0.0)   # J·U_ij = −dR_node

    if jac == "modified":
        Uij = s3d._tangent_solve_modified(asm, Uarr, dR_node)
        iters = 0
    elif jac == "nk":
        Uij, iters = s3d._tangent_solve_nk(asm, Uarr, dR_node, gmres_rtol=gmres_rtol)
    else:
        raise ValueError(f"jac must be 'nk' or 'modified', got {jac!r}")
    Uij = Uij.reshape(prob.shape)
    return (Uij, iters) if return_iters else Uij


# ==========================================================================
# 4.  FD validation oracle — central FD of the FIRST tangent (the M1 gate)
# ==========================================================================
def fd_cross_tangent_3d_qc(prob: s3.Problem3D, active_names, name_i: str, name_j: str,
                           theta, M_tot: float, *, fixed=None, h: float = 1e-4,
                           tol: float = 1e-12, max_iter: int = 40) -> np.ndarray:
    """Central FD ``∂/∂θ_j [ dU/dθ_i ]`` of the certified QC solve — the ground
    truth for :func:`cross_tangent_3d_qc`.

    Perturbs the ACTIVE parameter ``θ_j`` and rebuilds the QC slice via
    ``theta_to_slice3d`` (so the induced momenta move through ``qc_momenta``,
    exactly the dependence the cross tangent carries), re-solves ``U`` at
    ``θ±h·ê_j`` with the NK solver (residual to machine → a clean oracle), and
    finite-differences the first tangent ``dU/dθ_i`` computed there.
    **Validation only.**"""
    from ..parametric.parametric_nd_3d import theta_to_slice3d
    from ..solver import solver_3d_nk as s3nk

    active_names = list(active_names)
    fixed = dict(fixed or {})
    fixed.setdefault("qc", 1.0)
    kj = active_names.index(name_j)
    theta = np.asarray(theta, dtype=float)

    def dUi_at(th):
        sl = theta_to_slice3d(th, active_names, M_tot, fixed)
        U, _ = s3nk.newton_solve_nk(prob, sl, tol=tol, max_iter=max_iter)
        Ua = np.asarray(U)
        asm = s3.assemble(prob, sl)
        return np.asarray(s3dqc.certified_tangent_3d_qc(prob, Ua, sl, name_i, M_tot,
                                                        asm=asm, jac="nk"))

    tp = theta.copy(); tp[kj] += h
    tm = theta.copy(); tm[kj] -= h
    return (dUi_at(tp) - dUi_at(tm)) / (2.0 * h)
