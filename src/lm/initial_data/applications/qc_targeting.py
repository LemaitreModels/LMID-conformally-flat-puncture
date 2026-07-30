"""PARASOL — gradient-based physical-parameter targeting on the QUASI-CIRCULAR model.

The paper's headline application, promoted from the axisymmetric prototype
(``applications/control.py`` = Broyden warm-start, ``applications/sensitivity.py``
= the differentiable ``∂ID/∂θ`` hook) onto the *shipped* 4-D quasi-circular model
(Smolyak L=4, axes ``(b, q, S_Ay, S_By)``, ``qc=1.0``).

The point the demonstrator makes — the rebuttal to "isn't this just warm-started
TwoPunctures?": to hit a *physical target* (a chosen ADM mass + angular momentum),

  * a **black-box** solver (cold or Broyden) must run **many certified elliptic
    solves** — it has no gradient, so it finite-differences the parameter Jacobian
    and iterates a certified solve per outer evaluation;
  * the **differentiable** surrogate runs the *entire* outer Gauss--Newton loop on
    the free interpolant (analytic ``∂F/∂θ`` via ``jax.jacfwd``, *no* solve per
    step) and certifies **once** at the end — hitting the target in ~one certified
    solve.

The honest metric is therefore the **number of certified elliptic solves** to
reach the target (each ``newton_solve_nk`` call asserts ``‖R‖∞ ≤ tol``, so the
speed-up is never bought with accuracy).  Because the surrogate observable is only
interpolation-accurate, the single certified polish is followed by a short
**certified last-mile** correction (surrogate-Jacobian Newton, each step
re-certified) that drives the *true* target residual — measured on the certified
field — to tolerance; this is what makes the emitted configuration hit the
physical target, not merely the interpolant's estimate of it.

QC-specific content vs the 2-centre prototype:
  * the angular momentum is orbital, ``J = 2 b · p_t(b,q,spins) + S_Ay + S_By``
    (NOT the head-on ``S_A + S_B``); ``p_t`` is the PN closed form of
    ``parametric.quasicircular`` (differentiable jnp twin ``_pt_jax`` here);
  * ``M_ADM`` is read spectrally at the ``A=1`` (infinity) edge of the **φ-averaged
    (m=0)** field — the correct 3-D generalization of ``validation.adm``'s 2-D read.

Add-only / standalone: imports the frozen ``solver_3d`` / ``solver_3d_nk`` /
``parametric_nd_smolyak`` / ``parametric_nd_3d`` / ``quasicircular`` **verbatim**
and defines no new physics.  numpy + jax only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..solver import solver_3d as s3
from ..solver import solver_3d_nk as nk
from ..parametric import parametric_nd_smolyak as sm
from ..parametric.parametric_nd import attach_solve_fn_3d
from ..parametric.parametric_nd_3d import theta_to_slice3d

NAMES = ("b", "q", "S_Ay", "S_By")
FIXED = {"qc": 1.0}


# ==========================================================================
# Model loading (the shipped 4-D QC Smolyak model)
# ==========================================================================
def load_model(path, prob, *, solver: str = "nk"):
    """Load the shipped Smolyak QC model and attach the certified 3-D solve_fn."""
    model = sm.load_smolyak(path)
    attach_solve_fn_3d(model, prob, NAMES, M_tot=1.0, fixed=FIXED, solver=solver)
    return model


# ==========================================================================
# QC observables  F(θ) = [M_ADM, J]   (θ = (b, q, S_Ay, S_By))
# ==========================================================================
def masses(q, M_tot=1.0):
    return M_tot * q / (1.0 + q), M_tot / (1.0 + q)


def p_t_qc(b, q, S_Ay=0.0, S_By=0.0, M_tot=1.0):
    """PN tangential momentum ``p_t`` (numpy; = quasicircular.qc_scalar_momenta[0])."""
    from ..parametric import quasicircular as qc
    m_A, m_B = masses(q, M_tot)
    p_t, _ = qc.qc_scalar_momenta(float(b), m_A, m_B,
                                  float(S_Ay) / m_A ** 2, float(S_By) / m_B ** 2,
                                  radial=False)
    return p_t


def J_qc(theta, M_tot=1.0):
    """Orbital + spin angular momentum ``J = 2 b p_t + S_Ay + S_By`` (closed form)."""
    b, q, S_Ay, S_By = (float(theta[0]), float(theta[1]),
                        float(theta[2]), float(theta[3]))
    return 2.0 * b * p_t_qc(b, q, S_Ay, S_By, M_tot) + S_Ay + S_By


def M_ADM(prob, U, b, M_tot=1.0, q=None):
    """``M_ADM = m_A+m_B - 2b⟨∂_A u⟩_{A=1}`` on the φ-averaged (m=0) field (numpy)."""
    Umat = np.asarray(U).reshape(prob.shape)               # (Na+1, Nb, Nphi)
    Uavg = Umat.mean(axis=2)                                # m=0 meridian
    dUdA_inf = (np.asarray(prob.DA1) @ Uavg)[0, :]          # A[0]=1 (infinity)
    c = -float(b) * float(np.mean(dUdA_inf))
    return M_tot + 2.0 * c


def observe(prob, U, theta, target_names, M_tot=1.0):
    """The physical observable vector at a certified field ``U`` and params ``theta``."""
    b = float(theta[0])
    out = []
    for n in target_names:
        if n == "M_ADM":
            out.append(M_ADM(prob, U, b, M_tot))
        elif n == "J":
            out.append(J_qc(theta, M_tot))
        else:
            raise ValueError(f"unknown target {n!r}")
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------
# Differentiable jnp twins (for the analytic ∂F/∂θ of the gradient method)
# --------------------------------------------------------------------------
def _pt_jax(b, q, S_Ay, S_By, M_tot=1.0, pn_order=3, spin_orbit=True):
    """jnp twin of ``quasicircular.qc_scalar_momenta[0]`` (non-spinning 3PN + SO).

    Valid on the box ``q = m_A/m_B ∈ [1,3]`` (so ``m_A ≥ m_B`` and the larger-hole
    branch of the spin-orbit term is fixed — no data-dependent branch on a traced
    value)."""
    m_A = M_tot * q / (1.0 + q)
    m_B = M_tot / (1.0 + q)
    M = m_A + m_B
    mu = m_A * m_B / M
    nu = mu / M
    x = M / (2.0 * b)
    s = jnp.sqrt(x) + 2.0 * x ** 1.5
    if pn_order >= 2:
        s = s + (1.0 / 16.0) * (42.0 - 43.0 * nu) * x ** 2.5
    if pn_order >= 3:
        c3 = 480.0 + (163.0 * jnp.pi ** 2 - 4556.0) * nu + 104.0 * nu ** 2
        s = s + (1.0 / 128.0) * c3 * x ** 3.5
    p_t = mu * s
    if spin_orbit:
        chi_A = S_Ay / m_A ** 2
        chi_B = S_By / m_B ** 2
        qh = m_B / m_A                                     # m1 = m_A (larger), q≥1
        coeff = (2.0 / (3.0 * (1.0 + qh) ** 2)) * ((4.0 + 3.0 * qh) * chi_A
                                                    + qh * (3.0 + 4.0 * qh) * chi_B)
        p_t = p_t - mu * coeff * x ** 2
    return p_t


def build_F_jax(model, prob, target_names, M_tot=1.0):
    """Differentiable ``F(θ) = [obs...]`` on the surrogate field (jnp; for jacfwd)."""
    DA1 = jnp.asarray(prob.DA1)
    shape = prob.shape

    def M_ADM_jax(theta):
        U = jnp.asarray(model.evaluate_jax(theta)).reshape(shape)
        Uavg = jnp.mean(U, axis=2)
        dUdA_inf = (DA1 @ Uavg)[0, :]
        c = -theta[0] * jnp.mean(dUdA_inf)
        return M_tot + 2.0 * c

    def J_jax(theta):
        b, q, S_Ay, S_By = theta[0], theta[1], theta[2], theta[3]
        return 2.0 * b * _pt_jax(b, q, S_Ay, S_By, M_tot) + S_Ay + S_By

    obs = {"M_ADM": M_ADM_jax, "J": J_jax}

    def F_jax(theta):
        theta = jnp.asarray(theta)
        return jnp.stack([obs[n](theta) for n in target_names])

    return F_jax


# ==========================================================================
# Counters
# ==========================================================================
@dataclass
class Result:
    method: str
    theta: np.ndarray
    target: np.ndarray
    n_certified_solves: int          # the headline cost metric
    ctrl_residual: float             # ‖F_true − target‖∞ on the CERTIFIED field
    certified_residual: float        # worst ‖R‖∞ over all certified solves
    history: list = field(default_factory=list)   # (n_solves, ‖F−target‖∞)
    converged: bool = False
    wall_s: float = 0.0


# ==========================================================================
# Method 1 — black-box Broyden (cold or interp-warm), the "TwoPunctures" baseline
# ==========================================================================
def _certified_solve(model, prob, theta, guess, tol, max_iter):
    """One certified elliptic solve at ``theta`` warm-started from ``guess``."""
    sl = theta_to_slice3d(np.asarray(theta, float), NAMES, 1.0, FIXED)
    U, info = nk.newton_solve_nk(prob, sl, U0=guess, tol=tol, max_iter=max_iter)
    return U, float(info.residual_norm)


def broyden_target(model, prob, target, theta0, target_names, box, *,
                   mode="interp", active=(0, 1), tol_ctrl=1e-8, tol_inner=1e-10,
                   max_steps=40, fd_h=1e-4, max_inner=25, max_step_frac=0.35):
    """Mendes-style Broyden on ``G(θ)=F(θ)−target`` over the ``active`` knobs.

    Each ``F``-evaluation is a *certified* elliptic solve (``mode='cold'`` from
    ``u≡0``; ``mode='interp'`` warm-started from ``model.evaluate(θ)``).  The cost
    metric is the number of certified solves.  The inactive components of ``θ`` are
    held at ``theta0``."""
    t0 = time.perf_counter()
    target = np.asarray(target, float)
    active = list(active)
    theta = np.array(theta0, float)
    lo, hi = box[0], box[1]
    worst_R = 0.0
    n_solves = 0
    hist = []

    def G(th_active):
        nonlocal worst_R, n_solves
        th = theta.copy()
        th[active] = np.clip(th_active, lo[active], hi[active])
        guess = None if mode == "cold" else np.asarray(model.evaluate(th))
        U, rn = _certified_solve(model, prob, th, guess, tol_inner, max_inner)
        worst_R = max(worst_R, rn)
        n_solves += 1
        F = observe(prob, U, th, target_names)
        return F - target, th

    x = np.clip(theta[active], lo[active], hi[active])
    g, th = G(x)
    theta = th
    hist.append((n_solves, float(np.max(np.abs(g)))))
    d = len(active)
    cap = max_step_frac * (hi[active] - lo[active])

    # one-time FD initial Jacobian (d extra certified solves — the black-box tax)
    Jac = np.zeros((target.size, d))
    for k in range(d):
        xp = x.copy(); xp[k] += fd_h
        gp, _ = G(xp)
        Jac[:, k] = (gp - g) / fd_h

    for _ in range(max_steps):
        if np.max(np.abs(g)) <= tol_ctrl:
            break
        try:
            step = np.linalg.solve(Jac, -g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(Jac, -g, rcond=None)[0]
        step = np.clip(step, -cap, cap)
        x_new = np.clip(x + step, lo[active], hi[active])
        g_new, th = G(x_new)
        theta = th
        s = x_new - x
        y = g_new - g
        ss = float(s @ s)
        if ss > 0.0:
            Jac = Jac + np.outer(y - Jac @ s, s) / ss
        x, g = x_new, g_new
        hist.append((n_solves, float(np.max(np.abs(g)))))

    return Result(method=f"broyden_{mode}", theta=theta, target=target,
                  n_certified_solves=n_solves,
                  ctrl_residual=float(np.max(np.abs(g))),
                  certified_residual=worst_R, history=hist,
                  converged=bool(np.max(np.abs(g)) <= tol_ctrl),
                  wall_s=time.perf_counter() - t0)


# ==========================================================================
# Method 2 — gradient-based: Gauss–Newton on the free surrogate + certified last mile
# ==========================================================================
def _cc_node_superset(box, level=5):
    """Per-axis Clenshaw--Curtis node superset over the box (nested → contains every
    Smolyak node of level ≤ ``level``).  Used only to detect exact-node coincidences
    for the Jacobian nudge; ``level=5`` (33 nodes) covers the shipped L≤5 models."""
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    n = 2 ** level
    k = np.arange(n + 1)
    cc = -np.cos(np.pi * k / n)                            # ∈ [-1, 1], ascending
    return [0.5 * (lo[a] + hi[a]) + 0.5 * (hi[a] - lo[a]) * cc
            for a in range(lo.size)]


def _nudge(theta, node_sets, box, trigger=1e-8, shift_frac=2e-3):
    """Shift any component exactly on a CGL/CC node off it (branchless jax is 0/0
    there).  Used ONLY for the Jacobian evaluation — the iterate itself is never
    nudged, so inactive (fixed) components stay exact and the target stays
    reachable.  The nudge only perturbs a *search direction*, so it is benign."""
    theta = np.array(theta, float)
    for k in range(theta.size):
        nodes_k = np.asarray(node_sets[k])
        rng = float(nodes_k.max() - nodes_k.min())
        if float(np.min(np.abs(theta[k] - nodes_k))) < trigger:
            shift = shift_frac * (rng if rng > 0 else 1.0)
            hi = box[1][k]
            theta[k] = theta[k] + shift if theta[k] + shift <= hi else theta[k] - shift
            theta[k] = min(max(theta[k], box[0][k]), box[1][k])
    return theta


def gauss_newton_target(model, prob, target, theta0, target_names, box, *,
                        active=(0, 1), tol_ctrl=1e-8, max_steps=60,
                        lm_init=1e-3, lm_down=0.5, lm_up=4.0, lm_max=1e10,
                        polish_steps=2, polish_tol=1e-10, correction_steps=3):
    """Hit ``F(θ)=target`` by damped Gauss–Newton on the **free** surrogate, then a
    certified last-mile.

    The outer LM loop evaluates ``F`` and ``∂F/∂θ`` (``jax.jacfwd``) on the
    interpolant — **no** elliptic solve.  Then ONE certified polish
    (``evaluate_polished``) makes the field constraint-satisfying, and up to
    ``correction_steps`` surrogate-Jacobian Newton steps — each re-certified — drive
    the *true* target residual (measured on the certified field) to ``tol_ctrl``.
    The cost metric is the number of certified solves (= 1 + #corrections)."""
    t0 = time.perf_counter()
    target = np.asarray(target, float)
    active = list(active)
    F_jax = build_F_jax(model, prob, target_names, M_tot=1.0)
    Jf = jax.jacfwd(F_jax)
    lo, hi = box[0], box[1]
    node_sets = _cc_node_superset(box)

    def clip(th):
        return np.clip(np.asarray(th, float), lo, hi)

    def F_surrogate(th):
        # node-safe VALUE via the numpy interpolant (== F_jax off-node)
        return observe(prob, np.asarray(model.evaluate(th)), th, target_names)

    def jac(th):
        # Jacobian off-node (nudge only the direction eval; iterate stays exact)
        return np.asarray(Jf(jnp.asarray(_nudge(th, node_sets, box))))[:, active]

    theta = clip(theta0)
    G = F_surrogate(theta) - target
    mu = lm_init
    # ---- (a) free surrogate LM Gauss–Newton (zero certified solves) ----
    for _ in range(max_steps):
        if np.max(np.abs(G)) <= 1e-11:
            break
        Jm = jac(theta)
        JtJ = Jm.T @ Jm
        grad = Jm.T @ G
        diagJ = np.diag(np.diag(JtJ)) + 1e-30 * np.eye(len(active))
        g2 = float(G @ G)
        accepted = False
        for _lm in range(40):
            try:
                d_active = -np.linalg.solve(JtJ + mu * diagJ, grad)
            except np.linalg.LinAlgError:
                d_active = -np.linalg.lstsq(Jm, G, rcond=None)[0]
            cand = theta.copy(); cand[active] = theta[active] + d_active
            cand = clip(cand)
            G_cand = F_surrogate(cand) - target
            if float(G_cand @ G_cand) < g2:
                theta, G = cand, G_cand
                mu = max(mu * lm_down, 1e-12); accepted = True
                break
            mu *= lm_up
            if mu > lm_max:
                break
        if not accepted:
            break

    # ---- (b) certified last mile: polish + true-residual Newton corrections ----
    worst_R = 0.0
    n_solves = 0
    hist = []
    U, info = model.evaluate_polished(theta, newton_steps=polish_steps, tol=polish_tol)
    worst_R = max(worst_R, float(info.residual_norm)); n_solves += 1
    F_true = observe(prob, np.asarray(U), theta, target_names)
    G_true = F_true - target
    hist.append((n_solves, float(np.max(np.abs(G_true)))))

    for _ in range(correction_steps):
        if np.max(np.abs(G_true)) <= tol_ctrl:
            break
        Jm = jac(theta)                                      # surrogate Jacobian
        try:
            d_active = -np.linalg.solve(Jm, G_true)
        except np.linalg.LinAlgError:
            d_active = -np.linalg.lstsq(Jm, G_true, rcond=None)[0]
        cand = theta.copy(); cand[active] = theta[active] + d_active
        theta = clip(cand)
        U, info = model.evaluate_polished(theta, newton_steps=polish_steps, tol=polish_tol)
        worst_R = max(worst_R, float(info.residual_norm)); n_solves += 1
        F_true = observe(prob, np.asarray(U), theta, target_names)
        G_true = F_true - target
        hist.append((n_solves, float(np.max(np.abs(G_true)))))

    return Result(method="gradient", theta=theta, target=target,
                  n_certified_solves=n_solves,
                  ctrl_residual=float(np.max(np.abs(G_true))),
                  certified_residual=worst_R, history=hist,
                  converged=bool(np.max(np.abs(G_true)) <= tol_ctrl),
                  wall_s=time.perf_counter() - t0)


# ==========================================================================
# Known-answer targets (draw θ*, forward-map to (M_ADM, J) via a certified solve)
# ==========================================================================
def make_target(model, prob, theta_star, target_names, tol=1e-10, max_iter=25):
    """Forward map: certified-solve at ``θ*`` and return its observable vector."""
    U, _ = _certified_solve(model, prob, theta_star, None, tol, max_iter)
    return observe(prob, np.asarray(U), theta_star, target_names)
