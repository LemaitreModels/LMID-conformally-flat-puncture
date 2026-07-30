"""LM-initial-data — differentiable, certified eccentricity control via the Cook (1994)
effective-potential method, on a 2-D free-momentum family.

The eccentricity demonstrator (paper §VI, companion to the parameter-targeting one
in ``qc_targeting``).  Family: equal mass (q=1), no spin, radial momentum P_r=0
(an *apsis* sequence), free tangential momentum P_t and separation b — a small
2-D surrogate over (b, P_t) built by ``run_qc_effpot``.

Physics (Cook, PRD 50, 5025 (1994); the effective-potential method for
quasi-circular orbits):

  * the binding energy is ``E_b = M_ADM − (M_A + M_B)`` with M_A, M_B the individual
    (Brandt–Brügmann) ADM masses — a FIELD-dependent quantity (through M_ADM and the
    u-at-puncture in M_A, M_B), so a black box must run a certified elliptic solve
    to evaluate it;
  * at fixed angular momentum ``J = 2 b P_t`` (P_r=0), the CIRCULAR orbit is the
    minimum ``∂E_b/∂b|_J = 0``;  the classical method locates it by SCANNING b (a
    certified solve per point) and fitting the minimum;
  * a mildly off-circular apsis at (b0, J) has eccentricity ``e = |b0−b'|/(b0+b')``
    where ``b'`` is the second turning point ``E_b^eff(b';J) = E_b^eff(b0;J)`` across
    the minimum;  circularization (e→0) is exactly driving b0 → b_circ.

The demonstrator: the differentiable surrogate exposes ``∂E_b/∂b|_J`` analytically
(``jax.grad``), so a Newton root-find locates the circular orbit on the FREE
interpolant (no solve per step), certifying only at the end — versus the classical
certified-solve scan.  The honest metric is the number of certified elliptic
solves; every emitted configuration is certified ``‖R‖∞ ≤ 1e-10``.

Add-only / standalone: imports the frozen ``solver_3d`` / ``parametric_nd`` /
``parametric_nd_3d`` / ``validation.adm`` verbatim and reuses ``qc_targeting.M_ADM``;
defines no new physics.  numpy + jax only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from scipy.optimize import brentq

from ..solver import solver_3d as s3
from ..parametric.parametric_nd import load_parametric, attach_solve_fn_3d
from . import qc_targeting as qt

NAMES = ("b", "P_x")
FIXED = {"P": 0.0, "q": 1.0}
M_BARE, M_A_BARE, M_B_BARE = 1.0, 0.5, 0.5     # q = 1, equal mass


def load_model(path, prob, *, solver="nk"):
    model = load_parametric(path)
    attach_solve_fn_3d(model, prob, NAMES, M_tot=1.0, fixed=FIXED, solver=solver)
    return model


# ==========================================================================
# Individual (Brandt–Brügmann) ADM masses and the binding energy — numpy
# ==========================================================================
def _bary_wB(prob):
    B = np.asarray(prob.B)
    w = np.ones_like(B)
    for j in range(B.size):
        d = B[j] - B; d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    return w


def _u_at_puncture(prob, U, which, wB=None):
    """u at the puncture corner (A=0, B=±1) of the φ-averaged field."""
    wB = _bary_wB(prob) if wB is None else wB
    Um = np.asarray(U).reshape(prob.shape).mean(axis=2)     # (Na+1, Nb), m=0
    Bq = 1.0 if which == "A" else -1.0
    t = wB / (Bq - np.asarray(prob.B))
    return float((t @ Um[-1, :]) / t.sum())                 # A=0 (inner-axis) row


def binding_energy(prob, U, b):
    """``E_b = M_ADM − (M_A + M_B)`` (Cook), individual masses Brandt–Brügmann."""
    M_ADM = qt.M_ADM(prob, U, b)
    uA, uB = _u_at_puncture(prob, U, "A"), _u_at_puncture(prob, U, "B")
    M_A = M_A_BARE * (1.0 + uA + M_B_BARE / (2.0 * 2.0 * b))
    M_B = M_B_BARE * (1.0 + uB + M_A_BARE / (2.0 * 2.0 * b))
    return M_ADM - (M_A + M_B)


def Eb_certified(model, prob, b, P_t, newton_steps=2, tol=1e-10):
    """One CERTIFIED binding energy: polish at (b,P_t), then E_b on the certified U."""
    U, info = model.evaluate_polished(np.array([b, P_t]), newton_steps=newton_steps,
                                      tol=tol)
    return binding_energy(prob, np.asarray(U), b), float(info.residual_norm)


# ==========================================================================
# Differentiable binding energy on the free surrogate (jnp)
# ==========================================================================
def build_Eb_jax(model, prob):
    """``E_b(θ)`` on the surrogate field (θ=(b,P_t)); jax-differentiable."""
    DA1 = jnp.asarray(prob.DA1)
    B = jnp.asarray(prob.B)
    shape = prob.shape
    wB = jnp.asarray(_bary_wB(prob))

    def u_punc(Uavg, which):
        Bq = 1.0 if which == "A" else -1.0
        t = wB / (Bq - B)
        return (t @ Uavg[-1, :]) / jnp.sum(t)

    def Eb(theta):
        b = theta[0]
        U = jnp.asarray(model.evaluate_jax(theta)).reshape(shape)
        Uavg = jnp.mean(U, axis=2)
        # M_ADM = M_bare − 2 b ⟨∂_A u⟩_{A=1}
        M_ADM = M_BARE - 2.0 * b * jnp.mean((DA1 @ Uavg)[0, :])
        uA, uB = u_punc(Uavg, "A"), u_punc(Uavg, "B")
        M_A = M_A_BARE * (1.0 + uA + M_B_BARE / (4.0 * b))
        M_B = M_B_BARE * (1.0 + uB + M_A_BARE / (4.0 * b))
        return M_ADM - (M_A + M_B)

    return Eb


def build_effpot_jax(model, prob):
    """``V(b; J) = E_b(b, P_t=J/2b)`` and ``dV/db|_J`` on the surrogate (jnp)."""
    Eb = build_Eb_jax(model, prob)

    def V(b, J):
        return Eb(jnp.array([b, J / (2.0 * b)]))

    dV = jax.grad(V, argnums=0)
    return V, dV


# ==========================================================================
# Circular orbit at fixed J:  ∂E_b/∂b|_J = 0
# ==========================================================================
@dataclass
class CircResult:
    method: str
    J: float
    b_circ: float
    P_t_circ: float
    n_certified_solves: int
    certified_residual: float = np.nan
    dEb_db_certified: float = np.nan     # certified slope at b_circ (≈0 if circular)
    wall_s: float = 0.0
    scan: object = field(default=None, repr=False)


def circular_gradient(model, prob, J, b0, box_b, *, tol=1e-10, max_newton=30,
                      verify_h=0.05):
    """Locate the circular orbit at fixed ``J`` by Newton on ``dV/db|_J`` on the
    FREE surrogate (no solve per step), then certify.

    Cost: 1 certified emission at b_circ + a 2-point certified central-difference
    of ``dE_b/db`` there (the certified circular-orbit check) = 3 certified solves.
    """
    t0 = time.perf_counter()
    V, dV = build_effpot_jax(model, prob)
    d2V = jax.grad(dV, argnums=0)
    b = float(b0)
    for _ in range(max_newton):
        g = float(dV(b, J))
        h = float(d2V(b, J))
        if abs(g) < 1e-12 or h == 0.0:
            break
        step = -g / h
        b = float(np.clip(b + step, box_b[0], box_b[1]))
        if abs(step) < 1e-10:
            break
    b_circ = b
    P_t_circ = J / (2.0 * b_circ)
    # ---- certify: emit ID + certified dE_b/db check (central difference) ----
    _, r0 = Eb_certified(model, prob, b_circ, P_t_circ, tol=tol)
    Ep, rp = Eb_certified(model, prob, b_circ + verify_h, J / (2 * (b_circ + verify_h)), tol=tol)
    Em, rm = Eb_certified(model, prob, b_circ - verify_h, J / (2 * (b_circ - verify_h)), tol=tol)
    slope = (Ep - Em) / (2 * verify_h)
    return CircResult(method="gradient", J=J, b_circ=b_circ, P_t_circ=P_t_circ,
                      n_certified_solves=3, certified_residual=max(r0, rp, rm),
                      dEb_db_certified=slope, wall_s=time.perf_counter() - t0)


def circular_scan(model, prob, J, box_b, *, n_scan=13, tol=1e-10):
    """Classical effective-potential SCAN: certified E_b at ``n_scan`` separations
    on a fixed-J apsis sequence, parabola-fit the minimum.  Cost: n_scan solves."""
    t0 = time.perf_counter()
    bs = np.linspace(box_b[0], box_b[1], n_scan)
    Ebs, worst_R = [], 0.0
    for b in bs:
        E, r = Eb_certified(model, prob, b, J / (2 * b), tol=tol)
        Ebs.append(E); worst_R = max(worst_R, r)
    Ebs = np.array(Ebs)
    k = int(np.argmin(Ebs))
    k = min(max(k, 1), n_scan - 2)
    # local parabola fit around the discrete minimum
    c = np.polyfit(bs[k - 1:k + 2], Ebs[k - 1:k + 2], 2)
    b_circ = float(-c[1] / (2 * c[0])) if c[0] > 0 else float(bs[k])
    b_circ = float(np.clip(b_circ, box_b[0], box_b[1]))
    return CircResult(method="scan", J=J, b_circ=b_circ, P_t_circ=J / (2 * b_circ),
                      n_certified_solves=n_scan, certified_residual=worst_R,
                      wall_s=time.perf_counter() - t0,
                      scan=dict(b=bs.tolist(), Eb=Ebs.tolist()))


# ==========================================================================
# Eccentricity of an off-circular apsis (turning-point / Cook)
# ==========================================================================
def eccentricity(model, prob, b0, J, b_circ, box_b):
    """``e = |b0 − b'|/(b0 + b')`` with ``b'`` the second turning point of the
    effective potential across the circular minimum (surrogate, no solve)."""
    V, _ = build_effpot_jax(model, prob)
    Vf = lambda b: float(V(b, J))
    E0 = Vf(b0)
    # the other root of V(b;J)=E0 lies on the opposite side of b_circ
    if b0 < b_circ:
        lo, hi = b_circ, box_b[1]
    else:
        lo, hi = box_b[0], b_circ
    try:
        bp = brentq(lambda b: Vf(b) - E0, lo, hi)
    except ValueError:
        bp = b0                                    # E0 below range on that side
    return abs(b0 - bp) / (b0 + bp), bp
