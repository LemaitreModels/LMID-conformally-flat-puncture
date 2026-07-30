"""PARASOL-3D — Fourier-in-φ extension of the ABT / prolate-spheroidal patch.

The first non-axisymmetric PARASOL operator.  The axisymmetric two-centre code
(``operators_abt.py``, ``solver_abt.py``) is kept byte-for-byte frozen and is the
regression oracle; this module is an **add-only sibling** that lifts the same
prolate-spheroidal patch to 3-D by appending an azimuthal Fourier axis.

The full 3-D flat Laplacian in prolate-spheroidal coordinates (φ the azimuthal
rotation angle, the meridian (ρ,z) part orthogonal and axisymmetric) is

    Δ u = Δ_axisym u + (1/ρ²) ∂²_φ u,

with the meridian operator exactly the dense prolate operator of
``operators_abt.laplacian_matrix``.  In a Fourier basis along φ (``Nφ``
equispaced collocation points, real FFT) the azimuthal term is **diagonal in
the azimuthal mode m**: ``∂²_φ → −m²``.  Hence the linear operator is
**block-diagonal — one 2-D (A,B) block per m**:

    L_m = Lap_axisym − m² diag(1/ρ²).

Each 2-D block is factored once (``operators_abt.solve_equilibrated``); the full
3-D dense matrix (size (Na·Nb·Nφ)²) is **never formed**.

``ρ = b·2A/(1−A²)·√(1−B²)`` so ``1/ρ²`` is singular on the axis (A=0, the inner
segment |z|≤b; and B=±1, the outer axis).  The GL B-nodes already avoid B=±1;
the A=0 edge is a BC row.  On the inner axis only the m=0 mode is regular, so we
impose ``u_m = 0`` (Dirichlet) at A=0 for m≠0 — consistent with the existing A=0
Neumann row used for m=0.

The prolate B-operator is the **associated-Legendre operator**, whose regular
m-mode solution carries the factor ``(1−B²)^{|m|/2}`` at the outer axis B=±1.
For ODD m=1 that is a ``(1−B²)^{1/2}`` branch point — polynomial collocation of
the field itself converges only algebraically there.  We therefore solve for the
**smooth** factored unknown ``v_m`` with ``u_m = (1−B²)^{|m|/2} v_m``
(``block_operator_m_v``/``mode_operators``): the singular factor's B-derivatives
are analytic and only ``v_m`` is differentiated numerically, restoring spectral
convergence for every m.  For m=0 the factor is 1 and ``v_m≡u_m`` (the original
operator bit-for-bit).

Standalone: numpy + jax + the frozen sibling ``operators_abt``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from . import operators_abt as ops


# --------------------------------------------------------------------------
# φ collocation grid and real-FFT azimuthal mode set
# --------------------------------------------------------------------------
def phi_grid(Nphi: int) -> np.ndarray:
    """``Nφ`` equispaced collocation points φ_k = 2π k/Nφ on [0, 2π)."""
    return 2.0 * np.pi * np.arange(Nphi) / Nphi


def fourier_modes(Nphi: int) -> np.ndarray:
    """Azimuthal mode indices produced by ``numpy.fft.rfft`` of length ``Nφ``.

    ``m = 0, 1, ..., Nφ//2`` (the real-FFT half-spectrum); ``∂²_φ → −m²``.
    """
    return np.arange(Nphi // 2 + 1)


def build_grid_3d(Na: int, Nb: int, Nphi: int):
    """Return ``(A, B, DA1, DB1, phi)``: the frozen ABT (A,B) grid + φ nodes.

    ``A, B, DA1, DB1`` come verbatim from ``operators_abt.build_grid`` so the
    meridian discretisation is identical to the 2-D code.
    """
    A, B, DA1, DB1 = ops.build_grid(Na, Nb)
    return A, B, DA1, DB1, phi_grid(Nphi)


# --------------------------------------------------------------------------
# The per-m 2-D block operator
# --------------------------------------------------------------------------
def axisym_blocks(A, B, DA1, DB1, b):
    """Meridian pieces shared by every azimuthal mode.

    Returns ``(Lap, rho, z, Af, Bf, DA, DB, inv_rho2)`` where ``Lap`` is the
    dense prolate-spheroidal axisymmetric Laplacian (``operators_abt``) and
    ``inv_rho2 = 1/ρ²`` on the flattened (Na+1, Nb) node set, finite on the
    interior and set to 0 on the A=1 (ρ=∞) and A=0 (ρ=0) edges (those rows are
    BC rows, so the dummy value is never used).
    """
    Lap, rho, z, Af, Bf, DA, DB = ops.laplacian_matrix(A, B, DA1, DB1, b)
    rho = np.asarray(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_rho2 = 1.0 / rho ** 2
    good = np.isfinite(inv_rho2) & (rho > 1e-14)
    inv_rho2 = np.where(good, inv_rho2, 0.0)
    return Lap, rho, z, Af, Bf, DA, DB, inv_rho2


def block_operator_m(Lap, inv_rho2, m: int) -> np.ndarray:
    """The raw (no-BC) 2-D operator for azimuthal mode m: ``Lap − m² diag(1/ρ²)``."""
    if m == 0:
        return np.array(Lap, dtype=float)
    L = np.array(Lap, dtype=float)
    L[np.diag_indices_from(L)] -= (m ** 2) * inv_rho2
    return L


# --------------------------------------------------------------------------
# Associated-Legendre basis factoring — spectral odd-m convergence
# --------------------------------------------------------------------------
# The prolate B-operator (1−B²)∂²_B − 2B∂_B − m²/(1−B²) is the **associated-
# Legendre operator** (the centrifugal coefficient m²·Den/(4A²) → m² as B→±1,
# since Den → 4A² there).  Its regular m-mode solution behaves as (1−B²)^{|m|/2}
# at the outer prolate axis B=±1.  For ODD m=1 that is a (1−B²)^{1/2} branch
# point, which polynomial (GL) collocation resolves only ALGEBRAICALLY.
#
# Fix: substitute  u_m(A,B) = (1−B²)^{|m|/2} · v_m(A,B), with v_m SMOOTH in B, and
# build the operator on v_m — differentiating only the smooth v numerically; the
# singular factor's B-derivatives are analytic.  The (1−B²)^{|m|/2−1} centrifugal
# singularity then cancels against the γ w'' term (exactly as B→±1, where
# Den/(4A²)→1), leaving a regular operator whose v converges SPECTRALLY.
#
# For m=0 the factor is w≡1 and v_m≡u_m, so the m=0 block is the original Lap
# bit-for-bit (and the axisymmetric reduction is unchanged).
def bc_factor(Bf, m: int):
    """The B-factor ``w=(1−B²)^{|m|/2}`` and its analytic B-derivatives ``w', w''``.

    ``Bf`` is the flattened B-node array.  For m=0 returns ``(1, 0, 0)``.
    """
    Bf = np.asarray(Bf, dtype=float)
    if m == 0:
        return np.ones_like(Bf), np.zeros_like(Bf), np.zeros_like(Bf)
    s = abs(int(m)) / 2.0
    oa = 1.0 - Bf ** 2
    w = oa ** s
    wp = s * oa ** (s - 1.0) * (-2.0 * Bf)
    wpp = (s * (s - 1.0) * oa ** (s - 2.0) * (4.0 * Bf ** 2)
           + s * oa ** (s - 1.0) * (-2.0))
    return w, wp, wpp


def block_operator_m_v(A, B, DA1, DB1, b, m: int):
    """The factored 2-D operator for azimuthal mode m, acting on ``v_m`` (no BC).

    Returns ``(Mv, w)`` where ``Mv @ v_m == L_m[(1−B²)^{|m|/2} v_m] = L_m u_m`` and
    ``w`` is the B-factor at the flattened nodes.  For m=0 ``Mv`` is the original
    prolate Laplacian (``w≡1``) bit-for-bit.
    """
    Na1, Nb1 = A.size, B.size
    IA, IB = np.eye(Na1), np.eye(Nb1)
    DA = np.kron(np.asarray(DA1), IB)
    DB = np.kron(IA, np.asarray(DB1))
    DAA = DA @ DA
    DBB = DB @ DB
    AA, BB = np.meshgrid(A, B, indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    rho, z = ops.abt_map(Af, Bf, b)
    alpha, pcoef, gamma, qcoef = ops._coeffs(Af, Bf, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_rho2 = 1.0 / rho ** 2
    inv_rho2 = np.where(np.isfinite(inv_rho2) & (rho > 1e-14), inv_rho2, 0.0)
    cent = -(int(m) ** 2) * inv_rho2                        # centrifugal coeff
    w, wp, wpp = bc_factor(Bf, m)
    # A-derivatives: w is constant in A, so it factors through (row scaling)
    Mv = (alpha * w)[:, None] * DAA + (pcoef * w)[:, None] * DA
    # B-derivatives: expand ∂_B(w v), ∂²_B(w v) analytically in w and numerically in v
    Mv = Mv + gamma[:, None] * (np.diag(wpp) + 2.0 * wp[:, None] * DB
                                + w[:, None] * DBB)
    Mv = Mv + qcoef[:, None] * (np.diag(wp) + w[:, None] * DB)
    Mv = Mv + np.diag(cent * w)                             # centrifugal: cent·w·v
    return Mv, w


def mode_operators(A, B, DA1, DB1, b, m_vals):
    """Per-mode BC-applied operators and B-factors for the whole mode set.

    Returns ``(M_bc_list, w_list, interior)``: for each m the factored operator
    with BC rows replaced (``apply_bcs_m``) and the node-array B-factor ``w``.
    ``interior`` is the (shared) PDE-row mask.  Unknown per mode is ``v_m`` with
    ``u_m = w · v_m``; for m=0 ``w≡1`` so it is the original u-solve.
    """
    DA = np.kron(np.asarray(DA1), np.eye(B.size))           # 2-D ∂/∂A (for the m=0 Neumann row)
    M_bc_list, w_list, interior = [], [], None
    for m in m_vals:
        Mv, w = block_operator_m_v(A, B, DA1, DB1, b, int(m))
        Mm, interior = apply_bcs_m(Mv, A, B, DA, int(m))
        M_bc_list.append(Mm)
        w_list.append(w)
    return M_bc_list, w_list, interior


def apply_bcs_m(Lap_m, A, B, DA, m: int):
    """Row-replace BC edges for mode m; return ``(M, interior_mask)``.

    A=1 (i=0): Dirichlet u_m=0 (identity row) — all m.
    A=0 (i=Na): m=0 Neumann d/dA u=0 (the DA row, as in the 2-D code);
                m≠0 Dirichlet u_m=0 (axis regularity).
    interior_mask is True on PDE rows (identical for every m).
    """
    if m == 0:
        return ops.apply_bcs(Lap_m, A, B, DA)
    Na1, Nb1 = A.size, B.size
    M = np.array(Lap_m, dtype=float)
    interior = np.ones(Na1 * Nb1, dtype=bool)
    for j in range(Nb1):
        r_inf = 0 * Nb1 + j               # A=1 (infinity): Dirichlet
        M[r_inf, :] = 0.0
        M[r_inf, r_inf] = 1.0
        interior[r_inf] = False
        r_ax = (Na1 - 1) * Nb1 + j        # A=0 (inner axis): Dirichlet for m≠0
        M[r_ax, :] = 0.0
        M[r_ax, r_ax] = 1.0
        interior[r_ax] = False
    return M, interior
