"""LM-initial-data-2C — two-centre head-on Newton solve (Phase A, plan.md §12.7).

**Stage-1 discretization: a single puncture-A-centred spherical grid.**

The flat axisymmetric Laplacian, *centred on puncture A* (which sits on the
global symmetry axis at z=+b), is still **diagonal in the Legendre degree
ell_A** about A:

    (Δ u)_ell = u_ell'' + (2/r_A) u_ell' - ell(ell+1)/r_A^2 u_ell ,

so the radial operator stack and the block-diagonal Laplacian of the
single-centre solver are reused verbatim (``operators.build_laplacian``), and the
per-ℓ r_A=0 row-replacement *is* exactly puncture-A regularity.  What changes
relative to single-centre is **only the backbone and the source**:

  * ψ_BL is the two-term ``1 + m_A/(2 r_A) + m_B/(2 r_B)``  (``source.psi_BL_2c``);
  * Â² is the summed Bowen–York contraction with its O(1) cross term
    (``source.A2_2c``), evaluated on the grid's physical (rho, z) nodes.

Puncture B sits at the interior axis point (r_A=2b, μ_A=-1) — its 1/r_B
singularity lives in the analytic backbone (subtracted), so ``u`` is smooth
there.  This grid is mortar-free and reuses the single-centre machinery almost
verbatim, which makes it the natural stepping stone for the M0-A source
derivation and the M1-A P=0 exact-fixed-point gate (the highest-risk correctness
step).  It is, however, **B-limited** in spatial accuracy: a function with sharp
near-zone structure at B has a Legendre-in-μ_A pole that touches μ_A=-1 on the
shell r_A=2b, so the angular expansion there converges only slowly.  A
mortar/ball grid (M2-A) removes that limitation.

Modal radial unknowns ``U[a, k] = u_{ell_a}(r_{A,k})`` (shape (L_theta, N+1)),
flattened C-order to ``vec(U)[a*(N+1)+k]`` — identical ordering to single-centre.

Residual (interior radial nodes; BC rows give R = BC-residual, target 0):

    R[a,k] = (Δ u)_a[k] + (1/8) [ A @ ( (ψ_BL + u)^{-7} Â² ) ]_a[k] .

Analytic Jacobian (sign discipline — the +1/8 ψ^{-7} Â² source has u-derivative
-7/8 ψ^{-8} Â²):

    J = Lap_const  -  (7/8) [ per-radius block  A @ diag( (ψ+u)^{-8} Â² ) @ S ] .
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from . import spectral
from . import operators
from . import source


# --------------------------------------------------------------------------
# Slice configuration: the per-solve physical parameters (P fixed in Problem)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Slice:
    """Physical parameters of one head-on slice (the sweep variables)."""
    b: float            # half-separation; punctures at z = ±b
    m_A: float          # bare mass of puncture A (at +b)
    m_B: float          # bare mass of puncture B (at -b)

    @property
    def M(self) -> float:
        return self.m_A + self.m_B

    @property
    def q(self) -> float:
        return self.m_A / self.m_B


# --------------------------------------------------------------------------
# Problem: the b/mass-independent grid + operators (built once, frozen topology)
# --------------------------------------------------------------------------
@dataclass
class Problem:
    N: int
    L: float
    L_theta: int
    P: float
    J: int
    r: jnp.ndarray            # (N+1,)  r_A grid; r[0]=inf, r[N]=0
    r_safe: jnp.ndarray       # (N+1,)  r_A=0 and r_A=inf -> 1.0 dummy (masked)
    x: jnp.ndarray            # (N+1,)  CGL nodes; r_A = L(1+x)/(1-x)
    Dr: jnp.ndarray
    Dr2: jnp.ndarray
    ells: np.ndarray          # (L_theta,)
    mu: jnp.ndarray           # (J,)   GL nodes = cos(theta_A)
    w: jnp.ndarray            # (J,)
    sin_th: jnp.ndarray       # (J,)   sqrt(1 - mu^2)
    S: jnp.ndarray            # (J, L_theta)   synthesis
    A: jnp.ndarray            # (L_theta, J)   analysis
    A_list: list
    Lap_const: jnp.ndarray    # (M, M) block-diagonal Laplacian (b-independent)
    interior_mask: jnp.ndarray  # (N+1,)  0 at the two BC rows, 1 on interior

    @property
    def M(self) -> int:
        return self.L_theta * (self.N + 1)

    @property
    def shape(self):
        return (self.L_theta, self.N + 1)


def make_problem(N: int = 32, L: float = 2.0, L_theta: int = 8,
                 P: float = 0.5, J: Optional[int] = None) -> Problem:
    """Assemble the b/mass-independent grid + operators for the A-centred solve.

    Default ``L_theta=8`` (vs single-centre 6): puncture B's interior structure
    needs more even angular modes than a single puncture.  ``P`` is fixed across
    the entire parameter sweep (P=0 is degenerate, u≡0).
    """
    r, Dr, Dr2, x = spectral.radial_grid(N, L)
    mu, w, ells, S, A = spectral.legendre_transforms(L_theta, J)
    J_eff = int(mu.shape[0])
    A_list, Lap_const = operators.build_laplacian(ells, r, Dr, Dr2)

    r_np = np.asarray(r)
    r_safe_np = r_np.copy()
    r_safe_np[0] = 1.0                        # r_A=inf BC row dummy (masked)
    r_safe_np[-1] = 1.0                       # r_A=0   BC row dummy (masked)
    r_safe = jnp.asarray(r_safe_np)

    sin_th = jnp.asarray(np.sqrt(np.clip(1.0 - np.asarray(mu) ** 2, 0.0, None)))

    mask = np.ones(N + 1)
    mask[0] = 0.0                             # r_A = inf BC row
    mask[-1] = 0.0                            # r_A = 0   BC row
    interior_mask = jnp.asarray(mask)

    return Problem(
        N=N, L=L, L_theta=L_theta, P=P, J=J_eff,
        r=r, r_safe=r_safe, x=x, Dr=Dr, Dr2=Dr2, ells=ells, mu=mu, w=w,
        sin_th=sin_th, S=S, A=A, A_list=A_list, Lap_const=Lap_const,
        interior_mask=interior_mask,
    )


# --------------------------------------------------------------------------
# Physical-node geometry and the per-slice source arrays
# --------------------------------------------------------------------------
def grid_rho_z(prob: Problem, b: float):
    """Physical (rho, z) of every grid node, shape (J, N+1) each.

    Node (j, k) is at A-centred (r_A=r_safe[k], μ_A=mu[j]):
        z   = b + r_A μ_A ,     rho = r_A sqrt(1-μ_A^2).
    BC rows use the dummy r_safe=1.0 (their source is masked to 0).
    """
    r = prob.r_safe[None, :]                  # (1, N+1)
    mu = prob.mu[:, None]                      # (J, 1)
    sin_th = prob.sin_th[:, None]             # (J, 1)
    z = b + r * mu                             # (J, N+1)
    rho = r * sin_th                           # (J, N+1)
    return rho, z


def source_arrays(prob: Problem, sl: Slice):
    """Per-slice ``(psi_BL_grid, A2_grid)`` on the (J, N+1) grid.

    Both are masked to 0 on the two radial BC rows (so no inf/NaN leaks into the
    nonlinear source there).  ``psi_BL`` is left un-masked at the interior; the
    BC-row masking happens on the assembled source term in ``residual``.
    """
    rho, z = grid_rho_z(prob, sl.b)
    psi = source.psi_BL_2c(rho, z, sl.b, sl.m_A, sl.m_B)     # (J, N+1)
    A2 = source.A2_2c(rho, z, sl.b, prob.P)                  # (J, N+1)
    return psi, A2


# --------------------------------------------------------------------------
# Residual, analytic Jacobian, parameter tangent
# --------------------------------------------------------------------------
def residual(prob: Problem, U: jnp.ndarray, sl: Slice,
             psi=None, A2=None) -> jnp.ndarray:
    """R[a, k], shape (L_theta, N+1).  BC rows carry the (Lap) BC residual."""
    if psi is None or A2 is None:
        psi, A2 = source_arrays(prob, sl)
    u_nodal = prob.S @ U                          # (J, N+1)
    base = psi + u_nodal                           # (J, N+1)
    g_pt = 0.125 * base ** (-7.0) * A2             # +1/8 (ψ+u)^-7 Â²
    g_pt = g_pt * prob.interior_mask[None, :]      # zero at the two BC rows
    g_modal = prob.A @ g_pt                        # (L_theta, N+1)

    lap = (prob.Lap_const @ U.ravel()).reshape(prob.shape)
    return lap + g_modal


def jacobian(prob: Problem, U: jnp.ndarray, sl: Slice,
             psi=None, A2=None) -> jnp.ndarray:
    """Analytic Jacobian, shape (M, M).  J = Lap_const + Src(U)."""
    if psi is None or A2 is None:
        psi, A2 = source_arrays(prob, sl)
    u_nodal = prob.S @ U
    base = psi + u_nodal
    c = -0.875 * base ** (-8.0) * A2               # -7/8 (ψ+u)^-8 Â²
    c = c * prob.interior_mask[None, :]            # no source on BC rows

    SJ = jnp.einsum("aj,jk,jb->abk", prob.A, c, prob.S)   # (Lth, Lth, N+1)
    Np1 = prob.N + 1
    Src4 = jnp.einsum("abk,kK->akbK", SJ, jnp.eye(Np1))   # diagonal in radius
    Src = Src4.reshape(prob.M, prob.M)
    return prob.Lap_const + Src


# --------------------------------------------------------------------------
# Newton iteration
# --------------------------------------------------------------------------
@dataclass
class NewtonInfo:
    converged: bool
    iters: int
    residual_norm: float
    history: list


def newton_solve(prob: Problem, sl: Slice, U0: Optional[jnp.ndarray] = None,
                 tol: float = 1e-10, max_iter: int = 25):
    """Solve R(U) = 0 at the fixed slice ``sl`` with the analytic Jacobian.

    Returns ``(U, NewtonInfo)``.  ``U0`` warm-start (default: zeros).  Default
    ``tol=1e-10`` (the relaxed two-centre gate — the B-interior axis point lifts
    the conditioning floor above the single-centre 1e-12).
    """
    U = jnp.zeros(prob.shape) if U0 is None else jnp.asarray(U0)
    psi, A2 = source_arrays(prob, sl)             # b/mass/P fixed within the solve

    history = []
    rn = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        R = residual(prob, U, sl, psi, A2)
        rn = float(jnp.max(jnp.abs(R)))
        history.append(rn)
        if rn < tol:
            return U, NewtonInfo(True, it - 1, rn, history)
        Jm = jacobian(prob, U, sl, psi, A2)
        dU = jnp.linalg.solve(Jm, -R.ravel()).reshape(prob.shape)
        U = U + dU

    R = residual(prob, U, sl, psi, A2)
    rn = float(jnp.max(jnp.abs(R)))
    history.append(rn)
    return U, NewtonInfo(rn < tol, it, rn, history)


# --------------------------------------------------------------------------
# Parameter tangent dU/dparam (continuation predictor; autodiff source deriv)
# --------------------------------------------------------------------------
def tangent_b(prob: Problem, U: jnp.ndarray, sl: Slice) -> jnp.ndarray:
    """dU/db by implicit differentiation: J (dU/db) = -dR/db.

    Only the source depends on b; dR/db is taken by forward-mode autodiff of the
    residual w.r.t. the scalar b (a cheap predictor — not used for accuracy).
    """
    def R_of_b(bv):
        return residual(prob, U, Slice(bv, sl.m_A, sl.m_B)).ravel()

    dR_db = jax.jacfwd(R_of_b)(sl.b)
    Jm = jacobian(prob, U, sl)
    dU = jnp.linalg.solve(Jm, -dR_db).reshape(prob.shape)
    return dU


def tangent_q(prob: Problem, U: jnp.ndarray, sl: Slice, M_tot: float) -> jnp.ndarray:
    """dU/dq at fixed total mass M and separation b (q = m_A/m_B).

    m_A = M q/(1+q), m_B = M/(1+q); dR/dq by autodiff through those.
    """
    def R_of_q(qv):
        mA = M_tot * qv / (1.0 + qv)
        mB = M_tot / (1.0 + qv)
        return residual(prob, U, Slice(sl.b, mA, mB)).ravel()

    dR_dq = jax.jacfwd(R_of_q)(sl.q)
    Jm = jacobian(prob, U, sl)
    dU = jnp.linalg.solve(Jm, -dR_dq).reshape(prob.shape)
    return dU


# --------------------------------------------------------------------------
# Convenience: nodal field reconstruction and the ell=0 radial profile
# --------------------------------------------------------------------------
def to_nodal(prob: Problem, U: jnp.ndarray) -> jnp.ndarray:
    """u(r_{A,k}, μ_j) on the (J, N+1) grid from modal coefficients."""
    return prob.S @ U


def u0_profile(U: jnp.ndarray) -> jnp.ndarray:
    """The ell=0 radial mode u_0(r_{A,k}) (first modal row)."""
    return U[0]
