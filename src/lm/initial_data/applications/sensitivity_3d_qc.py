"""PARASOL-3D — the quasi-circular (QC) certified-ID parameter tangent ``dU/dθ_k``.

The QC extension of the H5a direct tangent (:mod:`applications.sensitivity_3d`).
The paper's astrophysical family is the **quasi-circular** one: the puncture
momenta are NOT a free axis but the deterministic PN function
``quasicircular.qc_momenta(b, m_A, m_B, S_A, S_B)`` of the separation, masses, and
spins (``parametric_nd_3d.theta_to_slice3d`` with ``fixed={"qc": 1.0}``).  So a
surrogate axis ``θ`` (``b``, ``q``, a spin component) moves the physical momenta
through ``qc_momenta``, and the certified tangent must carry that **chain rule** —
the H5a direct (fixed-momentum) tangent is incomplete for QC.

The decomposition (no double-counting — the direct ``b`` tangent already carries
the fixed-vector ``Â²`` scale law, the QC piece adds the *momenta-change* source):

    ∂R/∂θ|_QC = ∂R/∂θ|_direct                                  (sensitivity_3d)
              + interior · ⅛(ψ+u)^{-7} · 2 Â^{ij} Σ_X Â_mom(x_X, dP_X/dθ)_{ij}

where ``dP_X/dθ`` is obtained by **autodiff of the closed-form PN momenta**
(``_qc_momenta_jax``, the jnp twin of ``qc_momenta``, cross-checked bit-for-bit)
w.r.t. its natural arguments ``(b, m_A, m_B, S_A, S_B)``, chained with the simple
``dargs/dθ`` factors (``db/dθ=1``; the mass map ``dm/dq``; the spin construction
``dS_X/dθ`` reused from ``sensitivity_3d._dvec_dtheta``).  Because each Bowen–York
momentum tensor is **linear** in ``P``, ``Σ_X Â_mom(x_X, dP_X/dθ)`` is the same
``source_3d._mom_tensor_vec`` built from the derivative vectors — no finite
differencing of the source.

Facts worth stating:
  * for a **planar** spin axis (``S_x``/``S_z``/``S_mag``/``theta_S`` — spin in the
    x–z plane, ``S_y=0``) the QC momenta are unchanged (``qc_momenta`` uses only the
    aligned ``S_y``), so ``dP/dθ=0`` and the QC tangent reduces to the direct spin
    tangent.  The chain rule bites for ``b``, ``q`` (always), and the aligned
    ``S_Ay``/``S_By``.

Add-only / standalone: imports the committed ``sensitivity_3d`` / ``solver_3d`` /
``source_3d`` / ``quasicircular`` **verbatim**; defines no new physics beyond the
jnp twin of the PN momenta.  numpy + jax.
"""

from __future__ import annotations

from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..solver import solver_3d as s3
from ..solver import source_3d
from . import sensitivity_3d as s3d


# the QC-family axes (momentum is determined, so no free "P"/"P_x")
QC_TANGENT_AXES = ("b", "q", "S_x", "S_z", "S_mag", "theta_S",
                   "S_Ax", "S_Ay", "S_Az", "S_Bx", "S_By", "S_Bz",
                   # dimensionless-spin axes (chi = S/m^2), chi rebuild
                   "chi_Ax", "chi_Ay", "chi_Az", "chi_Bx", "chi_By", "chi_Bz")


# ==========================================================================
# 1.  The jnp twin of quasicircular.qc_momenta (autodiff'd for dP/d(args))
# ==========================================================================
def _qc_momenta_jax(args):
    """``qc_momenta`` as a differentiable jnp map ``args -> (P_A, P_B) flat (6,)``.

    ``args = [b, m_A, m_B, S_Ax, S_Ay, S_Az, S_Bx, S_By, S_Bz]``.  Mirrors
    ``quasicircular.qc_momenta`` at its defaults (pn_order=3, spin_orbit + radial
    on): the non-spinning 3PN tangential series, the leading aligned spin-orbit
    correction (uses the ``S_y`` components), and the leading radiation-reaction
    radial momentum.  Cross-checked bit-for-bit against the numpy original in the
    test suite.  The spin-orbit term is ALWAYS added (it is ∝χ, so 0 at χ=0 — the
    numpy skip is a value no-op, but always-adding gives the correct ∂/∂S_y at
    χ=0)."""
    b, mA, mB, SAx, SAy, SAz, SBx, SBy, SBz = args
    M = mA + mB
    mu = mA * mB / M
    nu = mu / M
    D = 2.0 * b
    x = M / D
    series = (jnp.sqrt(x) + 2.0 * x ** 1.5
              + (1.0 / 16.0) * (42.0 - 43.0 * nu) * x ** 2.5
              + (1.0 / 128.0) * (480.0 + (163.0 * jnp.pi ** 2 - 4556.0) * nu
                                 + 104.0 * nu ** 2) * x ** 3.5)
    pt = mu * series
    # leading aligned spin-orbit (Healy convention q=m2/m1<=1, body 1 = larger)
    chiA = SAy / mA ** 2
    chiB = SBy / mB ** 2
    swap = mA >= mB
    m1 = jnp.where(swap, mA, mB)
    m2 = jnp.where(swap, mB, mA)
    chi1 = jnp.where(swap, chiA, chiB)
    chi2 = jnp.where(swap, chiB, chiA)
    qq = m2 / m1
    coeff = (2.0 / (3.0 * (1.0 + qq) ** 2)) * ((4.0 + 3.0 * qq) * chi1
                                               + qq * (3.0 + 4.0 * qq) * chi2)
    pt = pt - mu * coeff * x ** 2
    pr = (64.0 / 5.0) * mu ** 2 * M ** 2 / D ** 3
    return jnp.array([pt, 0.0, -pr, -pt, 0.0, pr])


def _qc_args(sl: s3.Slice3D) -> np.ndarray:
    return np.array([sl.b, sl.m_A, sl.m_B, *sl.S_A_vec, *sl.S_B_vec], dtype=float)


def qc_momenta_vector_jacobian(sl: s3.Slice3D) -> np.ndarray:
    """``∂(P_A, P_B)/∂args`` (6×9) at the slice via forward-mode autodiff of
    :func:`_qc_momenta_jax`.  Exact (not finite-differenced)."""
    args = jnp.asarray(_qc_args(sl))
    return np.asarray(jax.jacfwd(_qc_momenta_jax)(args))


def _dargs_dtheta(sl: s3.Slice3D, name: str, M_tot: float) -> np.ndarray:
    """``d args/dθ`` (9,) for QC axis ``name`` — the chain factors feeding the
    qc-momenta Jacobian.  ``b`` moves ``args[0]``; ``q`` moves the masses
    (``args[1:3]``) via the mass map; a spin axis moves the spin components
    (``args[3:9]``) via ``sensitivity_3d._dvec_dtheta`` (the direct spin chain)."""
    darg = np.zeros(9)
    if name == "b":
        darg[0] = 1.0
    elif name == "q":
        q = sl.m_A / sl.m_B
        dmA = M_tot / (1.0 + q) ** 2
        dmB = -M_tot / (1.0 + q) ** 2
        darg[1] = dmA
        darg[2] = dmB
        # chi is the fixed box coordinate on the q-axis, so the PHYSICAL spin
        # S_X = chi_X m_X^2 moves with the masses:
        #   dS_X/dq = chi_X d(m_X^2)/dq = (S_X/m_X^2) 2 m_X dm_X/dq = 2 S_X dm_X/(m_X dq).
        # (Omitting this is the chi-rebuild q-tangent bug — it is ∝chi, so 0 at
        # chi=0 and dominant near |chi|=0.99; feeds the qc-momenta spin-orbit chain.)
        S_A = np.asarray(sl.S_A_vec, dtype=float)
        S_B = np.asarray(sl.S_B_vec, dtype=float)
        darg[3:6] = 2.0 * S_A * dmA / sl.m_A
        darg[6:9] = 2.0 * S_B * dmB / sl.m_B
    else:
        _dPA, _dPB, dSA, dSB = s3d._dvec_dtheta(sl, name)   # spin-vector chain
        darg[3:6] = dSA
        darg[6:9] = dSB
    return darg


def dP_dtheta_qc(sl: s3.Slice3D, name: str, M_tot: float):
    """``(dP_A/dθ, dP_B/dθ)`` (two 3-vectors) for QC axis ``name``."""
    dP = qc_momenta_vector_jacobian(sl) @ _dargs_dtheta(sl, name, M_tot)
    return dP[0:3], dP[3:6]


# ==========================================================================
# 2.  The qc-momenta source  ∂Â²/∂θ|_momenta  (linear-tensor chain rule)
# ==========================================================================
def _dA2_dtheta_qc_momenta(asm: s3.Assembly3D, phi: np.ndarray, sl: s3.Slice3D,
                           dP_A: np.ndarray, dP_B: np.ndarray) -> np.ndarray:
    """``∂Â²/∂θ`` from the momenta CHANGE only — shape ``(Ntot2d, Nφ)``.

    ``2 Â^{ij} [Â_mom(x_A, dP_A) + Â_mom(x_B, dP_B)]_{ij}`` (the BY momentum tensor
    is linear in ``P``, so its θ-derivative is the same builder on the derivative
    vector)."""
    rho = np.asarray(asm.rho, dtype=float).ravel()
    z = np.asarray(asm.z, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    finite = np.isfinite(rho) & np.isfinite(z)
    rho_s = np.where(finite, rho, 1.0)
    z_s = np.where(finite, z, 0.0)
    Ntot, Nphi = rho.size, phi.size
    xA = np.array([0.0, 0.0, sl.b])
    xB = np.array([0.0, 0.0, -sl.b])
    out = np.empty((Ntot, Nphi))
    for k in range(Nphi):
        X = np.stack([rho_s * np.cos(phi[k]), rho_s * np.sin(phi[k]), z_s], axis=1)
        T = source_3d.A_full_tensor_vec(X, sl.b, sl.P_A_vec, sl.P_B_vec,
                                        sl.S_A_vec, sl.S_B_vec)
        dT = np.zeros_like(T)
        if np.any(dP_A):
            dT = dT + source_3d._mom_tensor_vec(X, xA, dP_A)
        if np.any(dP_B):
            dT = dT + source_3d._mom_tensor_vec(X, xB, dP_B)
        a2 = 2.0 * np.sum(T * dT, axis=(1, 2))
        out[:, k] = np.where(finite & np.isfinite(a2), a2, 0.0)
    return out


def _dA2_dtheta_qc_massspin(asm: s3.Assembly3D, phi: np.ndarray, sl: s3.Slice3D,
                            dS_A: np.ndarray, dS_B: np.ndarray) -> np.ndarray:
    """``∂Â²/∂θ`` from the mass-induced SPIN change only — shape ``(Ntot2d, Nφ)``.

    On the ``q`` axis the dimensionless spin ``chi`` is held fixed, so the physical
    spin ``S_X = chi_X m_X^2`` moves with the masses (``dS_X/dθ`` from
    :func:`_dargs_dtheta`).  :func:`sensitivity_3d._dA2_dtheta` returns 0 for ``q``
    (its fixed-physical-vector convention), so this mass→spin source is added here.
    Like the momenta source, the Bowen–York spin tensor is linear in ``S``, so its
    θ-derivative is the same ``source_3d._spin_tensor_vec`` built on the derivative
    vector: ``2 Â^{ij}[Â_spin(x_A, dS_A) + Â_spin(x_B, dS_B)]_{ij}``."""
    rho = np.asarray(asm.rho, dtype=float).ravel()
    z = np.asarray(asm.z, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    finite = np.isfinite(rho) & np.isfinite(z)
    rho_s = np.where(finite, rho, 1.0)
    z_s = np.where(finite, z, 0.0)
    Ntot, Nphi = rho.size, phi.size
    xA = np.array([0.0, 0.0, sl.b])
    xB = np.array([0.0, 0.0, -sl.b])
    out = np.empty((Ntot, Nphi))
    for k in range(Nphi):
        X = np.stack([rho_s * np.cos(phi[k]), rho_s * np.sin(phi[k]), z_s], axis=1)
        T = source_3d.A_full_tensor_vec(X, sl.b, sl.P_A_vec, sl.P_B_vec,
                                        sl.S_A_vec, sl.S_B_vec)
        dT = np.zeros_like(T)
        if np.any(dS_A):
            dT = dT + source_3d._spin_tensor_vec(X, xA, dS_A)
        if np.any(dS_B):
            dT = dT + source_3d._spin_tensor_vec(X, xB, dS_B)
        a2 = 2.0 * np.sum(T * dT, axis=(1, 2))
        out[:, k] = np.where(finite & np.isfinite(a2), a2, 0.0)
    return out


# ==========================================================================
# 3.  The deliverable — certified_tangent_3d_qc
# ==========================================================================
def certified_tangent_3d_qc(prob: s3.Problem3D, U: np.ndarray, sl: s3.Slice3D,
                            name: str, M_tot: float,
                            asm: Optional[s3.Assembly3D] = None, *,
                            jac: str = "nk", gmres_rtol: float = 1e-11,
                            return_iters: bool = False):
    """The QC certified-ID tangent ``dU/dθ_name`` (IFT solve ``J·dU=−∂R/∂θ``) for
    the quasi-circular family, carrying the ``qc_momenta`` chain rule.

    ``sl`` must be the physical QC slice (``theta_to_slice3d(..., fixed={"qc":1})``),
    so its momenta already equal ``qc_momenta(b, m_A, m_B, S_A, S_B)``.  ``∂R/∂θ`` is
    the committed direct residual derivative (:func:`sensitivity_3d.dR_dtheta_node`)
    plus the qc-momenta source; the solve reuses the shared assembly ``asm`` and the
    H5a tangent solvers.  Reduces to :func:`sensitivity_3d.certified_tangent_3d` when
    ``dP/dθ = 0`` (a planar spin axis)."""
    if asm is None:
        asm = s3.assemble(prob, sl)
    Uarr = np.asarray(U, dtype=float).reshape(prob.Ntot2d, prob.Nphi)
    dR = s3d.dR_dtheta_node(prob, asm, Uarr, sl, name, M_tot)     # direct part
    base = asm.psi[:, None] + Uarr
    dP_A, dP_B = dP_dtheta_qc(sl, name, M_tot)
    if np.any(dP_A) or np.any(dP_B):
        dA2 = _dA2_dtheta_qc_momenta(asm, prob.phi, sl, dP_A, dP_B)
        dS = 0.125 * base ** (-7.0) * dA2
        dR = dR + np.where(asm.interior[:, None], dS, 0.0)
    # mass-induced spin change: an axis that moves the masses at fixed chi makes
    # S_X = chi_X m_X^2 vary, so Â_S changes.  The direct dR_dtheta_node/_dA2_dtheta
    # skip this for 'q' (fixed-physical-vector convention), so add it here.  Only 'q'
    # moves the masses (b changes separation; spin axes fix the masses), and it is a
    # no-op at chi=0 (dS/dq ∝ chi, so the spin block of _dargs_dtheta is 0).
    if name == "q":
        darg = _dargs_dtheta(sl, name, M_tot)
        dS_A_vec, dS_B_vec = darg[3:6], darg[6:9]
        if np.any(dS_A_vec) or np.any(dS_B_vec):
            dA2s = _dA2_dtheta_qc_massspin(asm, prob.phi, sl, dS_A_vec, dS_B_vec)
            dSs = 0.125 * base ** (-7.0) * dA2s
            dR = dR + np.where(asm.interior[:, None], dSs, 0.0)
    if jac == "modified":
        dU = s3d._tangent_solve_modified(asm, Uarr, dR)
        iters = 0
    elif jac == "nk":
        dU, iters = s3d._tangent_solve_nk(asm, Uarr, dR, gmres_rtol=gmres_rtol)
    else:
        raise ValueError(f"jac must be 'nk' or 'modified', got {jac!r}")
    dU = dU.reshape(prob.shape)
    return (dU, iters) if return_iters else dU


# ==========================================================================
# 4.  FD validation oracle (central FD of the certified QC solve)
# ==========================================================================
def fd_tangent_3d_qc(prob: s3.Problem3D, active_names, name: str, theta, M_tot: float,
                     *, fixed=None, h: float = 1e-4, solver: str = "modified",
                     tol: float = 1e-12, max_iter: int = 40) -> np.ndarray:
    """Central FD of the certified QC solve ``d/dθ_name[newton_solve→U]`` — the
    ground truth for :func:`certified_tangent_3d_qc`.  Perturbs the ACTIVE
    parameter ``θ`` and rebuilds the QC slice via ``theta_to_slice3d`` (so the
    momenta move through ``qc_momenta``, exactly the dependence the tangent
    carries).  **Validation only.**"""
    from ..parametric.parametric_nd_3d import theta_to_slice3d
    from ..solver import solver_3d_nk as s3nk

    active_names = list(active_names)
    fixed = dict(fixed or {})
    fixed.setdefault("qc", 1.0)
    k = active_names.index(name)
    theta = np.asarray(theta, dtype=float)

    def solve(th):
        sl = theta_to_slice3d(th, active_names, M_tot, fixed)
        if solver == "nk":
            U, _ = s3nk.newton_solve_nk(prob, sl, tol=tol, max_iter=max_iter)
        else:
            U, _ = s3.newton_solve(prob, sl, tol=tol, max_iter=max_iter)
        return np.asarray(U).reshape(prob.shape)

    tp = theta.copy(); tp[k] += h
    tm = theta.copy(); tm[k] -= h
    return (solve(tp) - solve(tm)) / (2.0 * h)


# ==========================================================================
# 5.  QC convenience wiring — the tangent_fn + the auto-plugged sparse builder
# ==========================================================================
def qc_tangent_fn(prob: s3.Problem3D, active_names, M_tot: float = 1.0,
                  fixed: Optional[dict] = None, *, tangent_jac: str = "nk"):
    """The QC per-node **tangent stack** ``tangent_fn(θ, U) -> (d, *field)`` for the
    sparse (Hermite-Smolyak) gradient enhancement, carrying the ``qc_momenta`` chain
    rule via :func:`certified_tangent_3d_qc`.

    Mirrors the default ``tangent_fn`` inside
    ``hermite_smolyak.from_problem_hermite_smolyak_3d`` but swaps the direct
    (fixed-momentum) :func:`sensitivity_3d.certified_tangent_3d` for the QC
    chain-rule tangent — the direct default would be silently wrong for ``b``/``q``
    (H5c: 30–108% off).  ``fixed`` must carry the QC flag (``{"qc": 1.0}``); one
    ``s3.assemble`` is shared across the ``d`` axes at each node."""
    from ..parametric.parametric_nd_3d import theta_to_slice3d

    active_names = list(active_names)

    def tangent_fn(theta_vec, U):
        sl = theta_to_slice3d(theta_vec, active_names, M_tot, fixed)
        asm = s3.assemble(prob, sl)                    # shared node assembly
        stack = [np.asarray(certified_tangent_3d_qc(prob, U, sl, name, M_tot,
                                                    asm=asm, jac=tangent_jac))
                 for name in active_names]
        return np.stack(stack, axis=0)

    return tangent_fn


def from_problem_hermite_smolyak_3d_qc(prob, axes, enhanced=(), M_tot: float = 1.0,
                                       fixed: Optional[dict] = None, use_cache: bool = True,
                                       solver: str = "nk", gmres_rtol: float = 1e-4,
                                       tangent_jac: str = "nk"):
    """A :class:`hermite_smolyak.HermiteSmolyakSolverND` over the 3-D solver for the
    **quasi-circular** family, auto-plugging the QC chain-rule tangent.

    The committed ``hermite_smolyak.from_problem_hermite_smolyak_3d`` correctly
    **raises** on the QC flag (``fixed={"qc": ...}``) without an explicit
    ``tangent_fn`` (the direct default would be silently wrong).  This wrapper
    builds the QC tangent (:func:`qc_tangent_fn`) and passes it through, so QC users
    do not hand-build it.  ``fixed`` defaults to ``{"qc": 1.0}`` (the QC family).
    Everything else is forwarded verbatim.

    Add-only: imports the committed ``hermite_smolyak`` layer lazily (avoids an
    import cycle); defines no new physics.
    """
    from ..parametric.hermite_smolyak import from_problem_hermite_smolyak_3d

    fixed = dict(fixed or {})
    fixed.setdefault("qc", 1.0)
    active_names = [a["name"] for a in axes]
    tf = qc_tangent_fn(prob, active_names, M_tot, fixed, tangent_jac=tangent_jac)
    return from_problem_hermite_smolyak_3d(
        prob, axes, enhanced=enhanced, M_tot=M_tot, fixed=fixed, use_cache=use_cache,
        solver=solver, gmres_rtol=gmres_rtol, tangent_jac=tangent_jac, tangent_fn=tf)
