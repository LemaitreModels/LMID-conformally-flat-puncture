"""LM-initial-data-3D — non-axisymmetric two-centre Newton solve (Fourier-in-φ).

The first non-axisymmetric LM-initial-data solver.  The frozen axisymmetric path
(``solver_abt.py``) stays the regression oracle; this add-only sibling lifts it
to 3-D by a Fourier collocation in φ.

Unknown: the NODAL field ``U[i, j, k] = u(A_i, B_j, φ_k)`` (shape
(Na+1, Nb, Nφ)).  The Lichnerowicz equation

    Δ_3D u = −⅛ (ψ_BL + u)^{-7} Â²,     u → 0 at infinity,

splits into the per-m block-diagonal **linear** operator (``operators_3d``)
and the **nonlinear** source ``S = ⅛(ψ_BL+u)^{-7}Â²``, which is node-diagonal in
physical space but couples azimuthal modes (Â² is φ-dependent for a misaligned
spin / off-axis momentum).

Solver — **mode-iteration / modified Newton** (the handoff's fixed-point-first
route).  At each Newton step:

  * the residual is assembled **exactly** in mode space,
        R̂_m = M0_m Û_m + interior · Ŝ_m,
    where Û_m = rfft_φ(U), Ŝ_m = rfft_φ(S) and M0_m is the per-m block with BC
    rows replaced;
  * the linear solve uses the **block-diagonal** Jacobian
        Ĵ_m = M0_m + diag(interior · d̄),     d̄(A,B) = ⟨−⅞(ψ+u)^{-8}Â²⟩_φ,
    i.e. the m=0 (φ-averaged) part of the node-diagonal source derivative — the
    exact mode-to-mode-diagonal block of the true Jacobian.  Each block is
    factored independently (``operators_abt.solve_equilibrated``).

Because the residual is exact, the converged solution solves the full coupled
problem; only the dropped φ-varying part of the source Jacobian affects the
(linear) convergence *rate*, which is fast for the mild Bowen–York nonlinearity
of a minimal axisymmetry break.

With ``Nφ=1`` (only the m=0 mode, on-axis P, zero spin) every step is **identical
to the frozen 2-D Newton** — the axisymmetric-reduction gate.

Standalone: numpy + jax + the sibling modules (operators_3d, operators_abt,
source_3d, source).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from . import operators_3d as ops3
from . import operators_abt as ops
from . import source
from . import source_3d


# --------------------------------------------------------------------------
# Problem / Slice dataclasses (mirror solver_abt, add vector momenta/spins + Nφ)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Slice3D:
    """Physical parameters of one non-axisymmetric slice.

    ``P_A_vec, P_B_vec`` are the per-puncture linear-momentum vectors, ``S_A_vec,
    S_B_vec`` the spin vectors (both Cartesian, punctures on the z-axis at ±b).
    The axisymmetric head-on default (anti-parallel z-momenta, no spin) is
    obtained with ``P_A_vec=(0,0,-P)``, ``P_B_vec=(0,0,+P)``.
    """
    b: float
    m_A: float
    m_B: float
    P_A_vec: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    P_B_vec: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    S_A_vec: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    S_B_vec: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def M(self) -> float:
        return self.m_A + self.m_B

    @staticmethod
    def head_on(b, m_A, m_B, P):
        """Convenience: the axisymmetric head-on slice (anti-parallel z-momenta)."""
        return Slice3D(b=b, m_A=m_A, m_B=m_B,
                       P_A_vec=(0.0, 0.0, -P), P_B_vec=(0.0, 0.0, P))


@dataclass
class Problem3D:
    """The b/mass-independent (A,B,φ) grid (built once; frozen topology)."""
    Na: int
    Nb: int
    Nphi: int
    A: np.ndarray
    B: np.ndarray
    DA1: np.ndarray
    DB1: np.ndarray
    phi: np.ndarray
    m_vals: np.ndarray

    @property
    def shape(self):
        return (self.A.size, self.B.size, self.Nphi)

    @property
    def Ntot2d(self) -> int:
        return self.A.size * self.B.size


def make_problem(Na: int = 36, Nb: int = 24, Nphi: int = 8) -> Problem3D:
    A, B, DA1, DB1, phi = ops3.build_grid_3d(Na, Nb, Nphi)
    return Problem3D(Na=Na, Nb=Nb, Nphi=Nphi, A=A, B=B, DA1=DA1, DB1=DB1,
                     phi=phi, m_vals=ops3.fourier_modes(Nphi))


# --------------------------------------------------------------------------
# Per-slice assembly: per-m BC operators + analytic ψ_BL and Â² on the grid
# --------------------------------------------------------------------------
@dataclass
class Assembly3D:
    M0: list                  # per-m FACTORED block with BC rows replaced (acts on v_m)
    w: list                   # per-m B-factor (Ntot2d,) so u_m = w · v_m
    interior: np.ndarray      # bool mask (Ntot2d,), True on PDE rows (same all m)
    rho: np.ndarray           # (Ntot2d,)
    z: np.ndarray
    psi: np.ndarray           # (Ntot2d,) ψ_BL on the grid (A=1 -> 1)
    A2: np.ndarray            # (Ntot2d, Nφ) summed BY Â² on the φ-collocation grid
    m_vals: np.ndarray


def assemble(prob: Problem3D, sl: Slice3D) -> Assembly3D:
    Lap, rho, z, Af, Bf, DA, DB, inv_rho2 = ops3.axisym_blocks(
        prob.A, prob.B, prob.DA1, prob.DB1, sl.b)
    # per-m FACTORED operators (acting on v_m = u_m / w; spectral odd-m) + BC rows
    M0_list, w_list, interior = ops3.mode_operators(
        prob.A, prob.B, prob.DA1, prob.DB1, sl.b, prob.m_vals)
    # analytic ψ_BL (φ-independent) and the non-axisymmetric Â²
    finite = np.isfinite(rho)
    rho_s = np.where(finite, rho, 1.0)
    z_s = np.where(finite, z, 0.0)
    psi = np.array(source.psi_BL_2c(rho_s, z_s, sl.b, sl.m_A, sl.m_B))
    psi = np.where(finite, psi, 1.0)
    A2 = source_3d.A2_at_nodes_3d(rho, z, prob.phi, sl.b,
                                  sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec)
    return Assembly3D(M0=M0_list, w=w_list, interior=interior, rho=rho, z=z,
                      psi=psi, A2=A2, m_vals=prob.m_vals)


# --------------------------------------------------------------------------
# Residual (exact, mode space) and modified-Newton step
# --------------------------------------------------------------------------
def _nl_source(asm: Assembly3D, U: np.ndarray):
    """Nodal nonlinear source ``S = ⅛(ψ+u)^{-7}Â²`` and base ``ψ+u``.

    ``U`` is nodal (Ntot2d, Nφ); both returns are (Ntot2d, Nφ).
    """
    base = asm.psi[:, None] + U
    S_nl = 0.125 * base ** (-7.0) * asm.A2
    return S_nl, base


def residual_modes(asm: Assembly3D, U: np.ndarray):
    """Exact mode-space residual R̂_m (Ntot2d, Nm complex).

    R̂_m = M0_m v̂_m + interior · Ŝ_m, with v̂_m = Û_m / w (the factored unknown,
    smooth in B), Û = rfft_φ(U), Ŝ = rfft_φ(S_nl).  Since M0_m v̂_m = L_m u_m, this
    is the exact mode-space residual of the nonlinear PDE.
    """
    S_nl, _ = _nl_source(asm, U)
    Uhat = np.fft.rfft(U, axis=1)
    Shat = np.fft.rfft(S_nl, axis=1)
    Rm = np.empty_like(Uhat)
    interior = asm.interior
    for mi in range(asm.m_vals.size):
        vhat = Uhat[:, mi] / asm.w[mi]                     # u_m = w · v_m
        Rm[:, mi] = asm.M0[mi] @ vhat + np.where(interior, Shat[:, mi], 0.0)
    return Rm


def nodal_residual_inf(asm: Assembly3D, U: np.ndarray) -> float:
    """‖R‖_∞ as a nodal (physical) inf-norm — the convergence monitor.

    Reconstructs the nodal residual from the exact mode-space residual; for
    ``Nφ=1`` this equals the 2-D solver's ``residual_vec`` inf-norm.
    """
    Rm = residual_modes(asm, U)
    R_node = np.fft.irfft(Rm, n=U.shape[1], axis=1)
    return float(np.max(np.abs(R_node)))


def newton_step(asm: Assembly3D, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One modified-Newton step; returns ``(U_new, Rm)`` (Rm pre-step residual).

    Solves per mode for the factored increment δv_m (Ĵ_m δv_m = −R̂_m), then
    reconstructs δu_m = w · δv_m.  The source Jacobian in v-space is
    diag(interior · d̄ · w) (chain rule ∂u/∂v = w; d̄ = φ-averaged source deriv).
    """
    Nphi = U.shape[1]
    Rm = residual_modes(asm, U)
    S_nl, base = _nl_source(asm, U)
    D_nl = -0.875 * base ** (-8.0) * asm.A2            # node-diagonal source deriv
    d_bar = D_nl.mean(axis=1)                          # (Ntot2d,) = φ-average (m=0)
    dUhat = np.empty((asm.interior.size, asm.m_vals.size), dtype=complex)
    for mi in range(asm.m_vals.size):
        Jm = np.array(asm.M0[mi])
        di = np.where(asm.interior, d_bar * asm.w[mi], 0.0)
        Jm[np.diag_indices_from(Jm)] += di
        dvhat = ops.solve_equilibrated(Jm, -Rm[:, mi])
        dUhat[:, mi] = asm.w[mi] * dvhat               # δu_m = w · δv_m
    dU = np.fft.irfft(dUhat, n=Nphi, axis=1)
    return U + dU, Rm


# --------------------------------------------------------------------------
# Newton iteration
# --------------------------------------------------------------------------
@dataclass
class NewtonInfo:
    converged: bool
    iters: int
    residual_norm: float
    history: list


def newton_solve(prob: Problem3D, sl: Slice3D, U0: Optional[np.ndarray] = None,
                 tol: float = 1e-10, max_iter: int = 30,
                 asm: Optional[Assembly3D] = None):
    """Solve the non-axisymmetric two-centre Lichnerowicz equation at ``sl``.

    Returns ``(U, NewtonInfo)`` with ``U`` shaped (Na+1, Nb, Nφ).  Tracks the
    nodal residual inf-norm; returns the BEST iterate and stops once the
    residual stagnates (mirrors ``solver_abt.newton_solve``).
    """
    if asm is None:
        asm = assemble(prob, sl)
    shp = prob.shape
    U = (np.zeros((prob.Ntot2d, prob.Nphi)) if U0 is None
         else np.asarray(U0, dtype=float).reshape(prob.Ntot2d, prob.Nphi))

    history = []
    best_U, best_rn = U, np.inf
    it = 0
    for it in range(1, max_iter + 1):
        rn = nodal_residual_inf(asm, U)
        history.append(rn)
        if rn < best_rn:
            best_U, best_rn = U, rn
        if rn < tol:
            break
        if it >= 3 and rn > 0.5 * history[-2]:        # stagnation -> stop
            break
        U, _ = newton_step(asm, U)

    return best_U.reshape(shp), NewtonInfo(best_rn < tol, it, best_rn, history)


# --------------------------------------------------------------------------
# Field evaluation (2-D barycentric in (A,B) + trig interpolation in φ)
# --------------------------------------------------------------------------
def _bary_weights(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    w = np.ones(n)
    for j in range(n):
        d = x[j] - x
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    return w


def _interp1(xq, x, w, vals):
    d = xq - x
    hit = np.isclose(d, 0.0, atol=1e-13)
    if np.any(hit):
        return vals[int(np.argmax(hit))]
    t = w / d
    return (t @ vals) / t.sum()


def _fourier_interp(vals, phi_q):
    """Trig interpolation of equispaced periodic samples ``vals`` at ``phi_q``."""
    vals = np.asarray(vals, dtype=float)
    N = vals.size
    if N == 1:
        return float(vals[0])
    c = np.fft.fft(vals)
    m = np.fft.fftfreq(N) * N                          # integer wavenumbers
    return float(np.real(np.sum(c * np.exp(1j * m * phi_q)) / N))


def evaluate_field(prob: Problem3D, U, rho, z, phi, b):
    """Interpolate nodal U to physical (rho, z, phi) points (arrays, matched)."""
    U = np.asarray(U).reshape(prob.shape)
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    A_q, B_q = ops.inverse_map(rho, z, b)
    wA, wB = _bary_weights(prob.A), _bary_weights(prob.B)
    out = np.empty(A_q.shape[0])
    for k in range(A_q.shape[0]):
        # interpolate (A,B) on each φ-plane -> values at the φ nodes
        vals_phi = np.empty(prob.Nphi)
        for p in range(prob.Nphi):
            col = np.array([_interp1(B_q[k], prob.B, wB, U[i, :, p])
                            for i in range(prob.A.size)])
            vals_phi[p] = _interp1(A_q[k], prob.A, wA, col)
        out[k] = _fourier_interp(vals_phi, phi[k])
    return out


def residual_norm(prob: Problem3D, U, sl) -> float:
    asm = assemble(prob, sl)
    return nodal_residual_inf(asm, np.asarray(U).reshape(prob.Ntot2d, prob.Nphi))


# --------------------------------------------------------------------------
# Linear 3-D Poisson solve  Δ_3D u = S  (exercises the per-m block operator;
# the manufactured-solution gate, decoupled from the nonlinear source)
# --------------------------------------------------------------------------
def solve_poisson(prob: Problem3D, b: float, S_nodal: np.ndarray) -> np.ndarray:
    """Solve ``Δ_3D u = S`` with u→0 at infinity for a nodal source ``S``.

    ``S_nodal`` is (Na+1, Nb, Nφ) (or (Ntot2d, Nφ)); homogeneous BCs (A=1
    Dirichlet u=0; A=0 Neumann for m=0, Dirichlet for m≠0).  Block-diagonal in
    the azimuthal mode m, one ``solve_equilibrated`` per m.  Returns u of shape
    (Na+1, Nb, Nφ).
    """
    M0_list, w_list, interior = ops3.mode_operators(
        prob.A, prob.B, prob.DA1, prob.DB1, b, prob.m_vals)
    S = np.asarray(S_nodal, dtype=float).reshape(prob.Ntot2d, prob.Nphi)
    Shat = np.fft.rfft(S, axis=1)
    Uhat = np.empty((prob.Ntot2d, prob.m_vals.size), dtype=complex)
    for mi in range(prob.m_vals.size):
        rhs = np.where(interior, Shat[:, mi], 0.0)        # BC rows -> 0 (v=0 -> u=0)
        vhat = ops.solve_equilibrated(M0_list[mi], rhs)
        Uhat[:, mi] = w_list[mi] * vhat                   # u_m = w · v_m
    U = np.fft.irfft(Uhat, n=prob.Nphi, axis=1)
    return U.reshape(prob.shape)
