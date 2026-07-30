"""LM-initial-data-3D — the QC certified-ID **second-order cross tangent** for the FULL
4-axis quasi-circular family ``(b, q, χ_Ay, χ_By)``: the 5 pairwise mixed
partials ``∂²U/∂θ_i∂θ_j`` that touch ``b`` or ``q``.

The committed :mod:`applications.sensitivity_3d_cross` supplies ``∂²U/∂θ_i∂θ_j``
for a pair of **spin** axes only (``_LINEAR_SPIN_AXES`` — the source-only case, no
geometry/``ψ``/nonlinear-vector second-order terms).  A full pairwise-bilinear
4-axis model needs ``C(4,2)=6`` crosses; the shipped module covers only the
spin–spin ``(χ_Ay, χ_By)`` one.  This add-only module supplies the other five:

    (b, q),  (b, χ_Ay),  (b, χ_By),  (q, χ_Ay),  (q, χ_By).

Nothing here changes any committed behaviour.  The public entry point
:func:`cross_tangent_3d_qc_bq` is a **dispatcher**: for a pair of spin axes it
falls through to the committed :func:`sensitivity_3d_cross.cross_tangent_3d_qc`
**verbatim** (bit-for-bit reduce-to-committed); for any pair touching ``b`` or
``q`` it runs the extended analytic path below.

The second-order forward-sensitivity equation (same factored NK Jacobian
``J = ∂R/∂U`` the first tangent uses — one extra back-solve):

    J · U_ij = −[ R_ij + R_Ui·U_j + R_Uj·U_i + R_UU·(U_i, U_j) ]

with ``R_·`` the closed-form partial derivatives of

    R = L(b) u + interior · S,      S = ⅛ (ψ_BL(θ) + u)^{-7} Â²(θ),

and ``U_i = dU/dθ_i`` the first certified tangents
(:func:`sensitivity_3d_qc.certified_tangent_3d_qc`, which already supports ``b``
and ``q``).

The genuinely-new content vs the committed spin–spin case (all vanish for a spin
pair, so the general code reduces to the committed one analytically):

  * **Operator scale.**  The linear 3-D Laplacian ``L`` (prolate + ``1/ρ²``
    centrifugal) scales as ``1/b²`` (``operators_3d``), so ``∂L/∂b = −(2/b)L`` and
    the mixed ``∂²R/∂b∂U`` carries the OPERATOR term ``−(2/b)·Δ_3D`` acting on the
    other first tangent ``U_j`` (reconstructed from the per-``m`` blocks exactly as
    :func:`sensitivity_3d.dR_dtheta_node` does for the first order).  ``∂²L/∂b² is
    not needed (no ``(b,b)`` pair) and ``∂²L/∂b∂θ = 0`` for ``θ≠b``, so ``L``
    contributes nothing to ``R_ij``.

  * **ψ_BL derivatives.**  ``ψ_BL`` depends on ``b`` (scale law
    ``ψ_b=−(ψ−1)/b``) and ``q`` (mass map, ``sensitivity_3d._dpsi_dtheta``); the
    single nonzero cross is ``ψ_bq = −ψ_q/b`` (``ψ`` depends only on ``b,q`` so
    ``ψ_ij=0`` unless both axes are in ``{b,q}``).  These feed the extra
    ``ψ``-terms of ``R_ij`` (``56 g^{-9}ψ_iψ_j Â² − 7 g^{-8}(ψ_ij Â² + ψ_i Â²_j +
    ψ_j Â²_i)``) and ``R_Ui`` (``+7 g^{-9} ψ_i Â²``) — all zero for a spin pair.

  * **Second-order QC chain rule.**  The per-puncture momenta/masses depend on
    ``(b,q)``, so ``d²Â`` needs the second-derivative source vectors:
      - ``d²P`` gains a ``∂P/∂args·(d²args)`` term on top of the committed
        Hessian term ``(dargs_i)ᵀ H (dargs_j)`` — nonzero for ``(q,χ)`` pairs,
        where the mass map makes ``args`` (the physical spin ``S=χ m²``) curve;
      - ``d²S`` gains the mass-induced ``∂²(χ m_X²)/∂q∂χ = 2 m_X (dm_X/dq)`` for a
        ``(q, χ_X·)`` pair;
      - and when one axis is ``b`` the tensor's geometric scale (``Â_P∝1/b²``,
        ``Â_S∝1/b³`` at fixed grid ``(A,B)``, node/puncture positions ∝ ``b``)
        gives the cross terms ``−(2/b)Â_P(dP_other) − (3/b)Â_S(dS_other)``.

Every term is closed-form / node-diagonal in the source and the operator scale —
no finite differencing in the shipped tangent (FD is validation only).

Add-only / standalone: imports the committed ``sensitivity_3d`` /
``sensitivity_3d_qc`` / ``sensitivity_3d_cross`` / ``solver_3d`` / ``source_3d``
**verbatim**; defines no new physics beyond the (already-committed) PN-momenta
twin's second derivative.  numpy + jax.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ..solver import solver_3d as s3
from ..solver import source_3d
from . import sensitivity_3d as s3d
from . import sensitivity_3d_qc as s3dqc
from . import sensitivity_3d_cross as cross


# the six dimensionless-spin axes (chi = S/m^2)
_CHI_AXES = ("chi_Ax", "chi_Ay", "chi_Az", "chi_Bx", "chi_By", "chi_Bz")
# the spin axes for which the COMMITTED cross tangent is exact (dispatch target)
_LINEAR_SPIN_AXES = cross._LINEAR_SPIN_AXES
# the axes this module additionally supports as a cross member (touching b/q)
_BQ_SUPPORTED = ("b", "q") + _CHI_AXES


# ==========================================================================
# 1.  Second-order chain factors  d²args, d²P, d²S  (generalise the spin case)
# ==========================================================================
def _d2args_dtheta2(sl: s3.Slice3D, name_i: str, name_j: str,
                    M_tot: float) -> np.ndarray:
    """``d² args / dθ_i dθ_j`` (9,) where ``args=[b,m_A,m_B,S_Ax..S_Bz]`` — the
    second-order chain factor feeding the qc-momenta Jacobian.

    ``b`` is linear (``args[0]``) and the masses depend only on ``q`` (their
    ``(q,q)`` curvature is not a cross pair), so the ONLY nonzero cross entries are
    the spin slots ``S_X = χ_X m_X(q)²``: for a ``(q, χ_X·)`` pair
    ``∂²S/∂q∂χ = ∂(m_X²)/∂q = 2 m_X (dm_X/dq)``.  Zero for ``(b,q)``, ``(b,χ)`` and
    ``(χ,χ)`` (the committed spin-spin case)."""
    d2 = np.zeros(9)
    pair = (name_i, name_j)
    if "q" in pair:
        other = name_j if name_i == "q" else name_i
        if other in _CHI_AXES:
            q = sl.m_A / sl.m_B
            dmA = M_tot / (1.0 + q) ** 2
            dmB = -M_tot / (1.0 + q) ** 2
            X, comp = other[4], "xyz".index(other[-1])
            if X == "A":
                d2[3 + comp] = 2.0 * sl.m_A * dmA
            else:
                d2[6 + comp] = 2.0 * sl.m_B * dmB
    return d2


def dP2_dtheta_qc_general(sl: s3.Slice3D, name_i: str, name_j: str,
                          M_tot: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(d²P_A/∂θ_i∂θ_j, d²P_B/∂θ_i∂θ_j)`` for the QC family, general in ``b,q,χ``.

    ``d²P = (dargs_i)ᵀ H (dargs_j) + (∂P/∂args)·(d²args_ij)`` — the committed
    :func:`sensitivity_3d_cross.dP2_dtheta_qc` computes only the Hessian term (exact
    for spin pairs, where ``d²args=0``); this adds the ``Jac·d²args`` term needed
    when the mass map curves the physical spin (``(q,χ)`` pairs)."""
    darg_i = s3dqc._dargs_dtheta(sl, name_i, M_tot)          # (9,)
    darg_j = s3dqc._dargs_dtheta(sl, name_j, M_tot)
    H = cross._qc_momenta_hessian(sl)                        # (6,9,9)
    d2args = _d2args_dtheta2(sl, name_i, name_j, M_tot)      # (9,)
    Jac = s3dqc.qc_momenta_vector_jacobian(sl)               # (6,9)
    dP2 = np.einsum("pab,a,b->p", H, darg_i, darg_j) + Jac @ d2args
    return dP2[0:3], dP2[3:6]


# ==========================================================================
# 2.  First/second-derivative BY tensors over the node cloud (with b geometry)
# ==========================================================================
def _dA_tensor_bq(X: np.ndarray, sl: s3.Slice3D, name: str, M_tot: float) -> np.ndarray:
    """The full QC first-derivative tensor ``dÂ_{ij}=∂Â_{ij}/∂θ`` at points ``X``
    (Npts,3) — shape (Npts,3,3), general over ``b, q, χ``.

    ``dÂ = [b geometric scale] + Â_P(dP) + Â_S(dS)`` with ``dP`` the qc-momenta
    chain (:func:`sensitivity_3d_qc.dP_dtheta_qc`) and ``dS`` the direct/mass-induced
    spin chain (:func:`sensitivity_3d_qc._dargs_dtheta`).  For a ``χ`` axis (no
    geometry, ``dS=m²`` direct) this equals the committed
    :func:`sensitivity_3d_cross._dA_tensor_qc`; the ``b`` branch adds the scale law
    ``−(2/b)Â_P −(3/b)Â_S`` (fixed-vector) reproducing
    :func:`sensitivity_3d._dA2_dtheta`'s ``b`` contraction."""
    xA = np.array([0.0, 0.0, sl.b])
    xB = np.array([0.0, 0.0, -sl.b])
    dP_A, dP_B = s3dqc.dP_dtheta_qc(sl, name, M_tot)
    darg = s3dqc._dargs_dtheta(sl, name, M_tot)
    dS_A, dS_B = darg[3:6], darg[6:9]
    dT = np.zeros((X.shape[0], 3, 3))
    if name == "b":
        TP = (source_3d._mom_tensor_vec(X, xA, sl.P_A_vec)
              + source_3d._mom_tensor_vec(X, xB, sl.P_B_vec))
        TS = (source_3d._spin_tensor_vec(X, xA, sl.S_A_vec)
              + source_3d._spin_tensor_vec(X, xB, sl.S_B_vec))
        dT = dT - (2.0 / sl.b) * TP - (3.0 / sl.b) * TS
    if np.any(dP_A):
        dT = dT + source_3d._mom_tensor_vec(X, xA, dP_A)
    if np.any(dP_B):
        dT = dT + source_3d._mom_tensor_vec(X, xB, dP_B)
    if np.any(dS_A):
        dT = dT + source_3d._spin_tensor_vec(X, xA, dS_A)
    if np.any(dS_B):
        dT = dT + source_3d._spin_tensor_vec(X, xB, dS_B)
    return dT


def _d2A_tensor_bq(X: np.ndarray, sl: s3.Slice3D, name_i: str, name_j: str,
                   M_tot: float) -> np.ndarray:
    """The QC second-derivative tensor ``d²Â_{ij}=∂²Â_{ij}/∂θ_i∂θ_j`` at points
    ``X`` (Npts,3) — shape (Npts,3,3), general over ``b, q, χ``.

    ``d²Â = Â_P(d²P) + Â_S(d²S) + [b geometric cross terms]``.  The vector second
    derivatives ``d²P``/``d²S`` come from :func:`dP2_dtheta_qc_general` /
    :func:`_d2args_dtheta2`; when one axis is ``b`` the tensor's geometric scale
    (``Â_P∝1/b²``, ``Â_S∝1/b³``) contributes the cross term
    ``−(2/b)Â_P(dP_other) −(3/b)Â_S(dS_other)`` (the first-derivative vectors of the
    OTHER axis).  For a spin pair (neither ``b``, ``d²S=0``, ``d²P``→Hessian) this
    reduces to the committed :func:`sensitivity_3d_cross._d2A_tensor_qc`."""
    xA = np.array([0.0, 0.0, sl.b])
    xB = np.array([0.0, 0.0, -sl.b])
    dP2_A, dP2_B = dP2_dtheta_qc_general(sl, name_i, name_j, M_tot)
    d2args = _d2args_dtheta2(sl, name_i, name_j, M_tot)
    dS2_A, dS2_B = d2args[3:6], d2args[6:9]
    d2T = np.zeros((X.shape[0], 3, 3))
    if np.any(dP2_A):
        d2T = d2T + source_3d._mom_tensor_vec(X, xA, dP2_A)
    if np.any(dP2_B):
        d2T = d2T + source_3d._mom_tensor_vec(X, xB, dP2_B)
    if np.any(dS2_A):
        d2T = d2T + source_3d._spin_tensor_vec(X, xA, dS2_A)
    if np.any(dS2_B):
        d2T = d2T + source_3d._spin_tensor_vec(X, xB, dS2_B)
    # b geometric cross terms: −p/b · Â_c(dv_other) for the axis that IS b
    for a_name, other in ((name_i, name_j), (name_j, name_i)):
        if a_name == "b":
            dP_o_A, dP_o_B = s3dqc.dP_dtheta_qc(sl, other, M_tot)
            darg_o = s3dqc._dargs_dtheta(sl, other, M_tot)
            dS_o_A, dS_o_B = darg_o[3:6], darg_o[6:9]
            if np.any(dP_o_A):
                d2T = d2T - (2.0 / sl.b) * source_3d._mom_tensor_vec(X, xA, dP_o_A)
            if np.any(dP_o_B):
                d2T = d2T - (2.0 / sl.b) * source_3d._mom_tensor_vec(X, xB, dP_o_B)
            if np.any(dS_o_A):
                d2T = d2T - (3.0 / sl.b) * source_3d._spin_tensor_vec(X, xA, dS_o_A)
            if np.any(dS_o_B):
                d2T = d2T - (3.0 / sl.b) * source_3d._spin_tensor_vec(X, xB, dS_o_B)
    return d2T


def _source_second_derivs_bq(asm: s3.Assembly3D, phi: np.ndarray, sl: s3.Slice3D,
                             name_i: str, name_j: str, M_tot: float):
    """The three source contractions over the node cloud — each (Ntot2d, Nφ):

        A2_i  = ∂Â²/∂θ_i        = 2 Â:dÂ_i
        A2_j  = ∂Â²/∂θ_j        = 2 Â:dÂ_j
        A2_ij = ∂²Â²/∂θ_i∂θ_j   = 2 dÂ_i:dÂ_j + 2 Â:d²Â_ij

    with the ``b``-geometry-aware first/second tensors above.  Non-finite rows (the
    A=1 infinity edge) → 0 (BC rows, masked in the solve)."""
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
        dTi = _dA_tensor_bq(X, sl, name_i, M_tot)
        dTj = _dA_tensor_bq(X, sl, name_j, M_tot)
        d2Tij = _d2A_tensor_bq(X, sl, name_i, name_j, M_tot)
        a2i = 2.0 * np.sum(T * dTi, axis=(1, 2))
        a2j = 2.0 * np.sum(T * dTj, axis=(1, 2))
        a2ij = 2.0 * np.sum(dTi * dTj, axis=(1, 2)) + 2.0 * np.sum(T * d2Tij, axis=(1, 2))
        A2_i[:, k] = np.where(finite & np.isfinite(a2i), a2i, 0.0)
        A2_j[:, k] = np.where(finite & np.isfinite(a2j), a2j, 0.0)
        A2_ij[:, k] = np.where(finite & np.isfinite(a2ij), a2ij, 0.0)
    return A2_i, A2_j, A2_ij


# ==========================================================================
# 3.  ψ_BL cross derivative  and the linear-operator reconstruction
# ==========================================================================
def _psi_cross(asm: s3.Assembly3D, sl: s3.Slice3D, name_i: str, name_j: str,
               M_tot: float) -> np.ndarray:
    """``∂²ψ_BL/∂θ_i∂θ_j`` (Ntot2d,).  ``ψ_BL`` depends only on ``(b,q)``, so the
    only nonzero cross among the supported pairs is ``(b,q)``:
    ``ψ_bq = −ψ_q/b`` (``ψ_b=−(ψ−1)/b`` differentiated w.r.t. ``q`` at fixed ``b``,
    equivalently ``ψ_q`` differentiated w.r.t. ``b`` via the ``1/b`` scale)."""
    if {name_i, name_j} == {"b", "q"}:
        psi_q = s3d._dpsi_dtheta(asm, sl, "q", M_tot)
        return -psi_q / sl.b
    return np.zeros(asm.psi.shape)


def _lap_nodal(asm: s3.Assembly3D, prob: s3.Problem3D, V: np.ndarray) -> np.ndarray:
    """``Δ_3D V`` reconstructed from the per-``m`` linear blocks (the operator
    action ``L u``), exactly as :func:`sensitivity_3d.dR_dtheta_node`'s ``b``
    geometry term.  ``V`` is (Ntot2d, Nφ); returns (Ntot2d, Nφ)."""
    V = np.asarray(V, dtype=float).reshape(prob.Ntot2d, prob.Nphi)
    Vhat = np.fft.rfft(V, axis=1)
    linhat = np.empty((prob.Ntot2d, asm.m_vals.size), dtype=complex)
    for mi in range(asm.m_vals.size):
        linhat[:, mi] = asm.M0[mi] @ (Vhat[:, mi] / asm.w[mi])
    return np.fft.irfft(linhat, n=prob.Nphi, axis=1)


# ==========================================================================
# 4.  The deliverable — the dispatcher cross tangent  ∂²U/∂θ_i∂θ_j
# ==========================================================================
def cross_tangent_3d_qc_bq(prob: s3.Problem3D, U: np.ndarray, sl: s3.Slice3D,
                           name_i: str, name_j: str, M_tot: float,
                           dU_i: Optional[np.ndarray] = None,
                           dU_j: Optional[np.ndarray] = None,
                           asm: Optional[s3.Assembly3D] = None, *,
                           jac: str = "nk", gmres_rtol: float = 1e-11,
                           return_iters: bool = False):
    """``∂²U/∂θ_i∂θ_j`` — the 4-axis QC certified-ID second-order cross tangent.

    **Dispatcher.**  If both axes are spin axes (:data:`_LINEAR_SPIN_AXES`) this
    calls the committed :func:`sensitivity_3d_cross.cross_tangent_3d_qc`
    **verbatim** (bit-for-bit).  For any pair touching ``b`` or ``q`` it runs the
    extended analytic path, adding the operator-scale, ``ψ_BL`` and second-order QC
    chain terms.  Same interface as the committed routine: reuses the shared
    per-slice assembly ``asm``; the first tangents ``dU_i``/``dU_j`` may be supplied
    (the shipped model's stored ``node_dU``) or computed here via
    :func:`sensitivity_3d_qc.certified_tangent_3d_qc` (which supports ``b``/``q``).
    ``sl`` must be the physical QC slice.  Returns ``U_ij`` shaped ``prob.shape``;
    with ``return_iters`` also the GMRES iteration count.
    """
    if name_i in _LINEAR_SPIN_AXES and name_j in _LINEAR_SPIN_AXES:
        return cross.cross_tangent_3d_qc(
            prob, U, sl, name_i, name_j, M_tot, dU_i=dU_i, dU_j=dU_j, asm=asm,
            jac=jac, gmres_rtol=gmres_rtol, return_iters=return_iters)

    for nm in (name_i, name_j):
        if nm not in _BQ_SUPPORTED:
            raise NotImplementedError(
                f"cross_tangent_3d_qc_bq supports the 4-axis QC family "
                f"{_BQ_SUPPORTED} (got {nm!r})")
    if name_i == name_j:
        raise ValueError(f"cross tangent needs distinct axes (got {name_i!r} twice)")

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

    g = asm.psi[:, None] + Uarr                        # base = ψ+u
    A2 = asm.A2
    psi_i = s3d._dpsi_dtheta(asm, sl, name_i, M_tot)[:, None]   # (Ntot2d,1)
    psi_j = s3d._dpsi_dtheta(asm, sl, name_j, M_tot)[:, None]
    psi_ij = _psi_cross(asm, sl, name_i, name_j, M_tot)[:, None]
    A2_i, A2_j, A2_ij = _source_second_derivs_bq(asm, prob.phi, sl,
                                                 name_i, name_j, M_tot)

    g7, g8, g9 = g ** (-7.0), g ** (-8.0), g ** (-9.0)
    R_ij = 0.125 * (56.0 * g9 * psi_i * psi_j * A2
                    - 7.0 * g8 * (psi_ij * A2 + psi_i * A2_j + psi_j * A2_i)
                    + g7 * A2_ij)
    R_Ui = -0.875 * g8 * A2_i + 7.0 * g9 * psi_i * A2
    R_Uj = -0.875 * g8 * A2_j + 7.0 * g9 * psi_j * A2
    R_UU = 7.0 * g9 * A2
    bracket = R_ij + R_Ui * Uj + R_Uj * Ui + R_UU * Ui * Uj
    dR_node = np.where(asm.interior[:, None], bracket, 0.0)

    # linear-operator (∂²R/∂b∂U) terms: −(2/b)·Δ_3D acting on the OTHER first tangent
    if name_i == "b":
        dR_node = dR_node + np.where(asm.interior[:, None],
                                     -(2.0 / sl.b) * _lap_nodal(asm, prob, Uj), 0.0)
    if name_j == "b":
        dR_node = dR_node + np.where(asm.interior[:, None],
                                     -(2.0 / sl.b) * _lap_nodal(asm, prob, Ui), 0.0)

    if jac == "modified":
        Uij = s3d._tangent_solve_modified(asm, Uarr, dR_node)
        iters = 0
    elif jac == "nk":
        Uij, iters = s3d._tangent_solve_nk(asm, Uarr, dR_node, gmres_rtol=gmres_rtol)
    else:
        raise ValueError(f"jac must be 'nk' or 'modified', got {jac!r}")
    Uij = Uij.reshape(prob.shape)
    return (Uij, iters) if return_iters else Uij
