"""PARASOL-2C — two-centre head-on Newton solve on the ABT single patch (M2-A).

The production two-centre solver: the summed Bowen–York source (``source.py``,
reused verbatim) on the single-patch prolate-spheroidal / ABT grid
(``operators_abt.py``).  Both punctures are resolved spectrally on ONE dense 2-D
collocation operator — no mortar.

Unknown: the NODAL field ``U[i, j] = u(A_i, B_j)`` (shape (Na+1, Nb)), flattened
C-order ``vec(U)[i*Nb + j]``.  The Laplacian is the dense prolate operator; the
nonlinear source is node-diagonal, so

    R = M0 @ vec(U) + interior * (1/8 (psi_BL+u)^{-7} Â²)        [BC rows from M0]
    J = M0 + diag(interior * (-7/8 (psi_BL+u)^{-8} Â²)),

where ``M0`` is the Laplacian with BC rows row-replaced (A=1 Dirichlet u=0,
A=0 inner-axis Neumann).  Frozen topology: the (A,B) grid is fixed across the
whole sweep; the separation ``b`` enters only the operator scale (Lap ∝ 1/b²)
and the analytic source — never the node count.

Standalone: numpy + jax + the sibling modules (operators_abt, source, spectral).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from . import operators_abt as ops
from . import source
from . import spectral


@dataclass(frozen=True)
class Slice:
    """Physical parameters of one head-on slice (the sweep variables).

    ``S_A, S_B`` are the aligned (∥z) Bowen–York spins of punctures A (+b) and
    B (−b); they default to 0, so an unspinning ``Slice(b, m_A, m_B)`` is the P1
    head-on slice unchanged (and the source reduces to it bit-for-bit).
    """
    b: float
    m_A: float
    m_B: float
    S_A: float = 0.0
    S_B: float = 0.0

    @property
    def M(self) -> float:
        return self.m_A + self.m_B

    @property
    def q(self) -> float:
        return self.m_A / self.m_B


@dataclass
class Problem:
    """The b/mass-independent ABT grid (built once; frozen topology)."""
    Na: int
    Nb: int
    P: float
    A: np.ndarray
    B: np.ndarray
    DA1: np.ndarray
    DB1: np.ndarray

    @property
    def shape(self):
        return (self.A.size, self.B.size)

    @property
    def Ntot(self) -> int:
        return self.A.size * self.B.size


def make_problem(Na: int = 36, Nb: int = 24, P: float = 0.5) -> Problem:
    A, B, DA1, DB1 = ops.build_grid(Na, Nb)
    return Problem(Na=Na, Nb=Nb, P=P, A=A, B=B, DA1=DA1, DB1=DB1)


# --------------------------------------------------------------------------
# Per-slice assembly: the b-dependent operator + analytic source on the grid
# --------------------------------------------------------------------------
@dataclass
class Assembly:
    M0: np.ndarray            # Laplacian with BC rows replaced
    interior: np.ndarray      # bool mask, True on PDE rows
    rho: np.ndarray           # (Ntot,) physical radii
    z: np.ndarray
    psi: np.ndarray           # (Ntot,) psi_BL on the grid (finite; A=1 -> 1)
    A2: np.ndarray            # (Ntot,) summed BY Â^2 (finite; A=1 -> 0)


def assemble(prob: Problem, sl: Slice) -> Assembly:
    Lap, rho, z, Af, Bf, DA, DB = ops.laplacian_matrix(
        prob.A, prob.B, prob.DA1, prob.DB1, sl.b)
    M0, interior = ops.apply_bcs(Lap, prob.A, prob.B, DA)
    finite = np.isfinite(rho)
    rho_s = np.where(finite, rho, 1.0)
    z_s = np.where(finite, z, 0.0)
    psi = np.array(source.psi_BL_2c(rho_s, z_s, sl.b, sl.m_A, sl.m_B))
    # Aligned-spin source (P2): A2_2c_spin reduces bit-for-bit to A2_2c when
    # sl.S_A == sl.S_B == 0, so the no-spin assembly is unchanged.
    A2 = np.array(source.A2_2c_spin(rho_s, z_s, sl.b, prob.P, sl.S_A, sl.S_B))
    psi = np.where(finite, psi, 1.0)
    A2 = np.where(finite, A2, 0.0)
    return Assembly(M0=M0, interior=interior, rho=rho, z=z, psi=psi, A2=A2)


# --------------------------------------------------------------------------
# Residual, analytic Jacobian
# --------------------------------------------------------------------------
def residual_vec(asm: Assembly, u: np.ndarray) -> np.ndarray:
    base = asm.psi + u
    src = 0.125 * base ** (-7.0) * asm.A2            # +1/8 (psi+u)^-7 Â²
    return asm.M0 @ u + np.where(asm.interior, src, 0.0)


def jacobian_mat(asm: Assembly, u: np.ndarray) -> np.ndarray:
    base = asm.psi + u
    dsrc = -0.875 * base ** (-8.0) * asm.A2          # -7/8 (psi+u)^-8 Â²
    J = np.array(asm.M0)
    di = np.where(asm.interior, dsrc, 0.0)
    J[np.diag_indices_from(J)] += di
    return J


# --------------------------------------------------------------------------
# Newton iteration
# --------------------------------------------------------------------------
@dataclass
class NewtonInfo:
    converged: bool
    iters: int
    residual_norm: float
    history: list


def newton_solve(prob: Problem, sl: Slice, U0: Optional[np.ndarray] = None,
                 tol: float = 1e-10, max_iter: int = 15, asm: Optional[Assembly] = None):
    """Solve the two-centre Lichnerowicz equation at slice ``sl``.

    Returns ``(U, NewtonInfo)`` with ``U`` shaped (Na+1, Nb).  ``U0`` warm-start.
    Newton converges quadratically (with row-equilibrated linear solves the
    operator is well conditioned, cond ~1e4) then floors at the nonlinear-source
    residual floor (~1e-11 at moderate Na, growing mildly with resolution as the
    nodal source carries more high-frequency content).  We return the
    BEST iterate and stop once the residual stagnates, so the sweep stays cheap.
    """
    if asm is None:
        asm = assemble(prob, sl)
    n = prob.Ntot
    u = np.zeros(n) if U0 is None else np.asarray(U0, dtype=float).ravel()

    history = []
    best_u, best_rn = u, np.inf
    it = 0
    for it in range(1, max_iter + 1):
        R = residual_vec(asm, u)
        rn = float(np.max(np.abs(R)))
        history.append(rn)
        if rn < best_rn:
            best_u, best_rn = u, rn
        if rn < tol:
            break
        # stagnation: residual stopped improving (hit the floor) -> stop
        if it >= 3 and rn > 0.5 * history[-2]:
            break
        J = jacobian_mat(asm, u)
        du = ops.solve_equilibrated(J, -R)
        u = u + du

    return best_u.reshape(prob.shape), NewtonInfo(best_rn < tol, it, best_rn, history)


# --------------------------------------------------------------------------
# Parameter tangents dU/db, dU/dq (analytic source derivatives; predictor)
# --------------------------------------------------------------------------
def tangent_b(prob: Problem, U: np.ndarray, sl: Slice,
              asm: Optional[Assembly] = None) -> np.ndarray:
    """dU/db by implicit differentiation, J (dU/db) = -dR/db.

    Lap(b) = Lap_unit/b^2, psi_BL-1 ∝ 1/b, Â² ∝ 1/b^4, so analytically
      dR/db|_interior = (-2/b)(Lap@u) + 1/8[ -7(psi+u)^{-8}(-(psi-1)/b)Â²
                                             + (psi+u)^{-7}(-4Â²/b) ].
    BC rows are b-independent (dR/db=0 there).
    """
    if asm is None:
        asm = assemble(prob, sl)
    u = np.asarray(U, dtype=float).ravel()
    base = asm.psi + u
    lap_u = asm.M0 @ u                                # interior rows = Lap@u
    dpsi_db = -(asm.psi - 1.0) / sl.b
    dA2_db = -4.0 * asm.A2 / sl.b
    dsrc_db = 0.125 * (-7.0 * base ** (-8.0) * dpsi_db * asm.A2
                       + base ** (-7.0) * dA2_db)
    dR_db = np.where(asm.interior, (-2.0 / sl.b) * lap_u + dsrc_db, 0.0)
    J = jacobian_mat(asm, u)
    dU = ops.solve_equilibrated(J, -dR_db)
    return dU.reshape(prob.shape)


def tangent_q(prob: Problem, U: np.ndarray, sl: Slice, M_tot: float,
              asm: Optional[Assembly] = None) -> np.ndarray:
    """dU/dq at fixed total mass M and separation b (q=m_A/m_B).

    Only psi_BL depends on q (through m_A=Mq/(1+q), m_B=M/(1+q)); Â² is
    mass-independent.  dpsi/dq = (dm_A/dq)/(2 r_A) + (dm_B/dq)/(2 r_B).
    """
    if asm is None:
        asm = assemble(prob, sl)
    u = np.asarray(U, dtype=float).ravel()
    base = asm.psi + u
    q = sl.q
    dmA = M_tot * (1.0 / (1.0 + q) - q / (1.0 + q) ** 2)     # = M/(1+q)^2
    dmB = -M_tot / (1.0 + q) ** 2
    finite = np.isfinite(asm.rho)
    rho_s = np.where(finite, asm.rho, 1.0)
    z_s = np.where(finite, asm.z, 0.0)
    inv2rA = np.where(finite, np.array(source.dpsiBL_dmA(rho_s, z_s, sl.b)), 0.0)
    inv2rB = np.where(finite, np.array(source.dpsiBL_dmB(rho_s, z_s, sl.b)), 0.0)
    dpsi_dq = dmA * inv2rA + dmB * inv2rB
    dsrc_dq = 0.125 * (-7.0 * base ** (-8.0) * dpsi_dq * asm.A2)
    dR_dq = np.where(asm.interior, dsrc_dq, 0.0)
    J = jacobian_mat(asm, u)
    dU = ops.solve_equilibrated(J, -dR_dq)
    return dU.reshape(prob.shape)


# --------------------------------------------------------------------------
# Field evaluation (2-D barycentric interpolation A,B) and observables
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


def evaluate_field_AB(prob: Problem, U, A_q, B_q):
    """Interpolate nodal U to query (A_q, B_q) arrays via 2-D barycentric."""
    U = np.asarray(U).reshape(prob.shape)
    A_q = np.atleast_1d(np.asarray(A_q, dtype=float))
    B_q = np.atleast_1d(np.asarray(B_q, dtype=float))
    wA, wB = _bary_weights(prob.A), _bary_weights(prob.B)

    def interp1(xq, x, w, vals):
        d = xq - x
        hit = np.isclose(d, 0.0, atol=1e-13)
        if np.any(hit):
            return vals[int(np.argmax(hit))]
        t = w / d
        return float((t @ vals) / t.sum())

    out = np.empty(A_q.shape[0])
    for k in range(A_q.shape[0]):
        col = np.array([interp1(B_q[k], prob.B, wB, U[i, :]) for i in range(prob.A.size)])
        out[k] = interp1(A_q[k], prob.A, wA, col)
    return out


def evaluate_field_phys(prob: Problem, U, rho, z, b):
    """Interpolate nodal U to physical (rho, z) points (via the inverse map)."""
    A_q, B_q = ops.inverse_map(rho, z, b)
    return evaluate_field_AB(prob, U, A_q, B_q)


def residual_norm(prob: Problem, U, sl) -> float:
    asm = assemble(prob, sl)
    R = residual_vec(asm, np.asarray(U).ravel())
    return float(np.max(np.abs(R)))


def adm_mass(prob: Problem, U, sl) -> float:
    """ADM mass M_ADM = (m_A+m_B) + 2 lim_{r->inf} r u.

    Read the 1/r tail from the field along the outer axis (B=+1 edge ~ z>=b)
    just inside infinity: fit r*u ~ c + d/r over the farthest finite A-nodes.
    """
    U = np.asarray(U).reshape(prob.shape)
    # outer axis above A: use B closest to +1 (largest B node) and A near 1
    jmax = int(np.argmax(prob.B))
    rho, z = ops.abt_map(prob.A, np.full(prob.A.size, prob.B[jmax]), sl.b)
    r = np.hypot(rho, z)
    u_line = U[:, jmax]
    fin = np.isfinite(r) & (r > 2.0 * sl.b)
    r_f, u_f = r[fin], u_line[fin]
    order = np.argsort(r_f)
    r_f, u_f = r_f[order], u_f[order]
    k = min(4, r_f.size)
    rs = r_f[-k:]
    y = rs * u_f[-k:]
    t = 1.0 / rs
    tn = t / t.max()
    d, c = np.polyfit(tn, y, 1)
    return float(sl.M + 2.0 * c)
