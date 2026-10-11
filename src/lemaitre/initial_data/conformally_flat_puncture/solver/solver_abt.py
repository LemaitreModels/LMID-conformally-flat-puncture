"""LM-initial-data-2C — two-centre head-on Newton solve on the ABT single patch (M2-A).

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

On a ``"gauss"`` radial grid (``operators_abt.RADIAL_MAPS``) ``M0`` acts on
``v = u/(A − 1)`` and has no BC rows, so the Newton state is ``v`` (as the 3-D
solver's is ``v_m = u_m/w_m``) and ``u = (A − 1)·v`` is reconstructed by a multiply:

    R = M0 @ v + 1/8 (psi_BL + f v)^{-7} Â²,     J_v = M0 + diag(f · (-7/8 (psi_BL+u)^{-8} Â²)),

with ``f = A − 1`` at the nodes (``Assembly.w``).  The functions taking ``u`` keep
taking the physical field; only :func:`jacobian_mat` is in ``v`` there (``J_v δv = −R``,
``δu = f δv``).  This is what makes the 3-D m=0 block, and so the Nφ=1 reduction,
the same operator as this one on that grid too.

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
from . import operators_3d as ops3
from . import source
from . import spectral
from .newton_loop import newton_loop


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
    # angular node map (``operators_abt.B_MAPS``) and radial map
    # (``operators_abt.RADIAL_MAPS``); see ``solver_3d.Problem3D`` — the defaults are
    # identity (a hand-built Problem is the original grid), not the grid default
    bmap: str = "cos"
    radial: str = "lobatto"

    @property
    def shape(self):
        return (self.A.size, self.B.size)

    @property
    def Bs(self) -> np.ndarray:
        """The B nodes in the map's interpolation coordinate (``B`` for ``"cos"``)."""
        return ops.s_nodes(self.Nb, self.bmap) if self.bmap != "cos" else self.B

    @property
    def Ntot(self) -> int:
        return self.A.size * self.B.size

    @property
    def scheme(self):
        """``"paper-v1"``, ``"v2"`` (``operators_abt.SCHEMES``), or ``None`` (mixed)."""
        return ops.scheme_of(self.bmap, self.radial)


def make_problem(Na: int = 36, Nb: int = 24, P: float = 0.5, bmap=None,
                 radial=None, scheme=None) -> Problem:
    """The grid; ``scheme``/``bmap``/``radial`` as ``solver_3d.make_problem``."""
    bmap, radial = ops.resolve_scheme(scheme, bmap, radial)
    A, B, DA1, DB1 = ops.build_grid(Na, Nb, bmap, radial)
    return Problem(Na=Na, Nb=Nb, P=P, A=A, B=B, DA1=DA1, DB1=DB1, bmap=bmap,
                   radial=radial)


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
    # "gauss" radial grid only: the radial factor f = A − 1 at the flattened nodes,
    # u = w·v, and M0 acts on v.  None on "lobatto", where M0 acts on u.
    w: Optional[np.ndarray] = None


def radial_w(prob) -> Optional[np.ndarray]:
    """``A − 1`` at the flattened nodes of a ``"gauss"`` grid, else ``None``."""
    f = ops.radial_factor(prob.A)
    return None if f is None else np.repeat(f, prob.B.size)


def v_of_u(asm: Assembly, u: np.ndarray) -> np.ndarray:
    """The unknown ``M0`` acts on: ``u`` itself, or ``u/(A − 1)`` on a ``"gauss"`` grid."""
    return u if asm.w is None else u / asm.w


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
    return Assembly(M0=M0, interior=interior, rho=rho, z=z, psi=psi, A2=A2,
                    w=radial_w(prob))


# --------------------------------------------------------------------------
# Residual, analytic Jacobian
# --------------------------------------------------------------------------
def residual_vec(asm: Assembly, u: np.ndarray) -> np.ndarray:
    """The residual of the PHYSICAL field ``u`` (on a ``"gauss"`` grid ``M0`` is
    applied to ``u/(A − 1)``; see :func:`residual_vec_v` for the solver's form)."""
    base = asm.psi + u
    src = 0.125 * base ** (-7.0) * asm.A2            # +1/8 (psi+u)^-7 Â²
    return asm.M0 @ v_of_u(asm, u) + np.where(asm.interior, src, 0.0)


def residual_vec_v(asm: Assembly, v: np.ndarray):
    """``(R, u)`` from the unknown ``v`` of a ``"gauss"`` grid: ``u = (A − 1)·v`` by a
    multiply, ``R = M0 @ v + interior · src(u)``."""
    u = asm.w * v
    base = asm.psi + u
    src = 0.125 * base ** (-7.0) * asm.A2
    return asm.M0 @ v + np.where(asm.interior, src, 0.0), u


def jacobian_mat(asm: Assembly, u: np.ndarray) -> np.ndarray:
    """The Newton matrix at the physical field ``u``.

    It acts on ``δu`` on a ``"lobatto"`` grid and on ``δv`` (``δu = (A − 1)·δv``) on a
    ``"gauss"`` one, where the source derivative picks up the chain-rule factor
    ``∂u/∂v = A − 1``; every caller solves it and then multiplies by ``asm.w``
    (:func:`du_from_solve`).
    """
    base = asm.psi + u
    dsrc = -0.875 * base ** (-8.0) * asm.A2          # -7/8 (psi+u)^-8 Â²
    J = np.array(asm.M0)
    if asm.w is not None:
        dsrc = dsrc * asm.w
    di = np.where(asm.interior, dsrc, 0.0)
    J[np.diag_indices_from(J)] += di
    return J


def du_from_solve(asm: Assembly, d: np.ndarray) -> np.ndarray:
    """The physical increment from the solution of a :func:`jacobian_mat` system:
    ``d`` itself, or ``(A − 1)·d`` on a ``"gauss"`` grid."""
    return d if asm.w is None else asm.w * d


# --------------------------------------------------------------------------
# Newton iteration
# --------------------------------------------------------------------------
@dataclass
class NewtonInfo:
    converged: bool
    iters: int
    residual_norm: float
    history: list
    # which norm ``residual_norm`` and ``history`` are in: "raw" (max|R|, the
    # "lobatto" grid) or "u" (equilibrated, the "gauss" grid) — see newton_solve
    norm: str = "raw"


def newton_solve(prob: Problem, sl: Slice, U0: Optional[np.ndarray] = None,
                 tol: float = 1e-10, max_iter: int = 15, asm: Optional[Assembly] = None):
    """Solve the two-centre Lichnerowicz equation at slice ``sl``.

    Returns ``(U, NewtonInfo)`` with ``U`` shaped (Na+1, Nb).  ``U0`` warm-start.
    Newton converges quadratically (with row-equilibrated linear solves the
    operator is well conditioned, cond ~1e4) then floors at the nonlinear-source
    residual floor (~1e-11 at moderate Na, growing mildly with resolution as the
    nodal source carries more high-frequency content).  The loop itself — best
    iterate, an internal target one decade below ``tol``, the two-strike
    stagnation rule, ``max_iter`` as the step budget — is the shared driver in
    ``newton_loop.py``; ``converged`` is judged against the caller's ``tol``.
    """
    if asm is None:
        asm = assemble(prob, sl)
    n = prob.Ntot
    u = np.zeros(n) if U0 is None else np.asarray(U0, dtype=float).ravel()

    if asm.w is not None:
        # "gauss" radial grid: the state is v = u/(A − 1) (a warm start pays that one
        # division; a cold start none), u is rebuilt by a multiply, and the step
        # solves J_v δv = −R.
        #
        # The monitor is the EQUILIBRATED residual in the u norm — each row divided
        # by max_k |M0[i,k]|/|w_k| (operators_3d.u_row_scales, the 3-D solver's own
        # definition), so this solve is the Nφ=1 case of the 3-D one in norm as well
        # as in operator, and its gate is the 3-D one (certification.gate_for_problem).
        # Not the raw max|R| of the "lobatto" branch below: on this grid the
        # innermost row sits at A ~ 1/Na² and carries coefficients ~1e9–1e11, so the
        # raw residual's roundoff floor (~eps·row scale·|v|, measured 3e-9 – 3e-8 at
        # 28×20 and 44×32) lies above any certification threshold while the
        # equilibrated residual is at ~1e-17.
        su = ops3.u_row_scales([asm.M0], [asm.w])[0]

        def monitor_v(v):
            R, uu = residual_vec_v(asm, v)
            return float(np.max(np.abs(R) / su)), (R, uu)

        def step_v(v, _k, aux):
            R, uu = aux
            return v + ops.solve_equilibrated(jacobian_mat(asm, uu), -R)

        run = newton_loop(u / asm.w, monitor_fn=monitor_v, step_fn=step_v,
                          tol=tol, max_steps=max_iter)
        return ((asm.w * run.U).reshape(prob.shape),
                NewtonInfo(run.converged, run.iters, run.residual_norm, run.history,
                           norm="u"))

    def monitor(u):
        R = residual_vec(asm, u)
        return float(np.max(np.abs(R))), R

    def step(u, _k, R):
        J = jacobian_mat(asm, u)
        du = ops.solve_equilibrated(J, -R)
        return u + du

    run = newton_loop(u, monitor_fn=monitor, step_fn=step,
                      tol=tol, max_steps=max_iter)
    return run.U.reshape(prob.shape), NewtonInfo(run.converged, run.iters,
                                                 run.residual_norm, run.history)


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
    lap_u = asm.M0 @ v_of_u(asm, u)                   # interior rows = Lap@u
    dpsi_db = -(asm.psi - 1.0) / sl.b
    dA2_db = -4.0 * asm.A2 / sl.b
    dsrc_db = 0.125 * (-7.0 * base ** (-8.0) * dpsi_db * asm.A2
                       + base ** (-7.0) * dA2_db)
    dR_db = np.where(asm.interior, (-2.0 / sl.b) * lap_u + dsrc_db, 0.0)
    J = jacobian_mat(asm, u)
    dU = du_from_solve(asm, ops.solve_equilibrated(J, -dR_db))
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
    dU = du_from_solve(asm, ops.solve_equilibrated(J, -dR_dq))
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


def evaluate_field_AB(prob: Problem, U, A_q, B_q, s_q=None):
    """Interpolate nodal U to query (A_q, B_q) arrays via 2-D barycentric.

    On a mapped grid (``prob.bmap != "cos"``) the angular rule is barycentric in the
    map's coordinate ``s``; pass ``s_q`` (from ``operators_abt.inverse_map_s``) to
    avoid recovering it from ``B_q``, which loses accuracy next to the axis.

    On a ``"gauss"`` radial grid the interpolated polynomial is ``V = u/(A − 1)``,
    and ``u = (A_q − 1)·V`` at the query point (``solver_3d.evaluate_field``).
    """
    U = np.asarray(U).reshape(prob.shape)
    gauss = getattr(prob, "radial", "lobatto") == "gauss"
    if gauss:
        U = U / (prob.A - 1.0)[:, None]
    A_q = np.atleast_1d(np.asarray(A_q, dtype=float))
    B_q = np.atleast_1d(np.asarray(B_q, dtype=float))
    bmap = getattr(prob, "bmap", "cos")
    Bs = getattr(prob, "Bs", prob.B)
    if bmap != "cos":
        B_q = (ops.s_of_b(B_q, bmap) if s_q is None
               else np.atleast_1d(np.asarray(s_q, dtype=float)))
    wA, wB = _bary_weights(prob.A), _bary_weights(Bs)

    def interp1(xq, x, w, vals):
        d = xq - x
        hit = np.isclose(d, 0.0, atol=1e-13)
        if np.any(hit):
            return vals[int(np.argmax(hit))]
        t = w / d
        return float((t @ vals) / t.sum())

    out = np.empty(A_q.shape[0])
    for k in range(A_q.shape[0]):
        col = np.array([interp1(B_q[k], Bs, wB, U[i, :]) for i in range(prob.A.size)])
        out[k] = interp1(A_q[k], prob.A, wA, col)
    if gauss:
        out = (A_q - 1.0) * out
    return out


def evaluate_field_phys(prob: Problem, U, rho, z, b):
    """Interpolate nodal U to physical (rho, z) points (via the inverse map)."""
    bmap = getattr(prob, "bmap", "cos")
    if bmap == "cos":
        A_q, B_q = ops.inverse_map(rho, z, b)
        return evaluate_field_AB(prob, U, A_q, B_q)
    A_q, s_q = ops.inverse_map_s(rho, z, b, bmap)
    return evaluate_field_AB(prob, U, A_q, ops.b_of_s(s_q, bmap), s_q=s_q)


