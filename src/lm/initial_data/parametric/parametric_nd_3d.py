"""PARASOL — wire the N-D parametric layer to the 3-D non-axisymmetric solver.

The **3-D lift** of the parametric surrogate: a tensor-product Chebyshev-in-
parameter interpolant over a *non-axisymmetric* Bowen–York slice family, with
**certified per-slice polish**.  ``parametric_nd.py`` (the solver-agnostic
tensor-product CGL-in-parameter layer) is reused **verbatim** — it interpolates
any array field elementwise, so the 3-D nodal field ``U`` of shape
``(Na+1, Nb, Nφ)`` works directly.  This module injects the single callable that
layer needs — ``solve_fn(theta_vec, guess, tol, max_iter) -> (U, info)`` — that
maps a parameter point ``θ`` onto a misaligned-spin / off-axis-momentum
``Slice3D`` and runs the validated 3-D solve.

The forward map is the committed **certified Newton–Krylov** solver
(``solver_3d_nk.newton_solve_nk`` / ``evaluate_polished_nk``), so every
interpolant prediction can be certified to ``‖R‖∞ ≤ 1e-10`` at any θ — the
"cannot be silently wrong" gate, now over the 3-D family.  The cheaper
modified-Newton solver (``solver_3d.newton_solve``) is offered as the production
sweep option: its converged FIELD is bit-identical (the NK report's headline —
NK *reproduces*, never improves, the modified-Newton field), so it is the right
choice for the held-out interpolation-error studies, where only the field
matters.

Conventions (mirrors ``parametric_nd_2c``):
  * **single misaligned spin on puncture A**, head-on z-momenta the default.
    The spin axes are dual: POLAR ``(S_mag, theta_S)`` (tilt in the x–z plane,
    ``spin_vec`` of ``run_3d_sweep``) or CARTESIAN ``(S_x, S_z)``; the open
    question (which converges faster) is answered by measuring both.
  * **off-axis momentum** ``P_x`` adds an anti-symmetric transverse component
    ``P_A=(P_x,0,−P), P_B=(−P_x,0,+P)`` (orbital-J series of ``run_3d_sweep`` C).
  * D1/D2 mass map: ``m_A = M q/(1+q), m_B = M/(1+q)``.
  * D7 per-b cache: the per-m block operators (``ops3.mode_operators``) and the
    meridian geometry depend ONLY on ``b`` (+ the fixed grid/Nφ), so with ``b``
    the outermost snake axis they are built once per b-node and reused for all
    other knobs at that b.  ``assemble_cached_3d`` calls the SAME ops/source
    functions as ``solver_3d.assemble`` → byte-identical ``Assembly3D``.

Standalone: numpy + jax + the sibling modules (solver_3d, solver_3d_nk,
operators_3d, source, source_3d, parametric_nd, parametric_nd_2c).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ..solver import solver_3d as s3
from ..solver import solver_3d_nk as s3nk
from ..solver import operators_3d as ops3
from ..solver import source
from ..solver import source_3d
from ..solver.solver_3d import Problem3D, Slice3D
from . import parametric_nd as pnd
from . import quasicircular as qcmod
from .parametric import cheb_param_nodes

# Pure-math / solver-agnostic study helpers reused verbatim from the 2-centre layer.
from .parametric_nd_2c import (  # noqa: F401  (re-export for the report driver/tests)
    geometric_rate,
    bernstein_rate,
    bernstein_rate_from_zero,
    infer_real_singularity,
    fit_error_model,
    Q_needed,
    cost_table,
    smolyak_points,
)


# canonical axis names and the inactive-axis defaults (mirror parametric_nd_2c)
AXIS_NAMES_3D = ("b", "S_mag", "theta_S", "S_x", "S_z", "P", "P_x", "q",
                 # full generic-spin axes (both punctures, all 3 components) — the
                 # 8-D "vary both spin vectors" extension; source_3d already supports
                 # arbitrary S_A/S_B, this exposes them as independent knobs.
                 "S_Ax", "S_Ay", "S_Az", "S_Bx", "S_By", "S_Bz",
                 # dimensionless-spin axes chi_Xi = S_Xi/m_X^2 (add-only; converted
                 # to the physical spin S in theta_to_slice3d once m_A,m_B are known)
                 "chi_Ax", "chi_Ay", "chi_Az", "chi_Bx", "chi_By", "chi_Bz")
DEFAULTS_3D = {
    "b": 2.0,
    "S_mag": 0.0,       # spin magnitude (polar parametrisation)
    "theta_S": 45.0,    # spin tilt off the collision (z) axis, DEGREES (polar)
    "S_x": 0.0,         # spin x-component (Cartesian parametrisation)
    "S_z": 0.0,         # spin z-component (Cartesian parametrisation)
    "P": 0.5,           # head-on (collision-axis) infall momentum magnitude
    "P_x": 0.0,         # off-axis (transverse) momentum component
    "q": 1.0,           # mass ratio m_A/m_B
    # full generic-spin components (default 0 → the single-spin/planar families are
    # byte-for-byte unchanged unless one of these names is activated/fixed)
    "S_Ax": 0.0, "S_Ay": 0.0, "S_Az": 0.0,
    "S_Bx": 0.0, "S_By": 0.0, "S_Bz": 0.0,
    # dimensionless-spin axes (default 0 → converted to S=0, so single-spin/planar
    # families are byte-for-byte unchanged unless a chi_* name is activated/fixed)
    "chi_Ax": 0.0, "chi_Ay": 0.0, "chi_Az": 0.0,
    "chi_Bx": 0.0, "chi_By": 0.0, "chi_Bz": 0.0,
}

# the generic-spin axis names; if ANY is active/fixed, theta_to_slice3d takes the
# full-vector branch (both punctures, all components) instead of the planar
# single-spin (polar/Cartesian) branch.
FULL_SPIN_NAMES_3D = ("S_Ax", "S_Ay", "S_Az", "S_Bx", "S_By", "S_Bz")

# dimensionless-spin (Kerr-like) axis names; a chi_Xi axis carries S_Xi/m_X^2 and
# is converted to the physical spin S_Xi = chi_Xi m_X^2 in theta_to_slice3d (m_X
# depends on q, so the conversion is q-coupled and cannot be a static rescale).
CHI_SPIN_NAMES_3D = ("chi_Ax", "chi_Ay", "chi_Az", "chi_Bx", "chi_By", "chi_Bz")


# --------------------------------------------------------------------------
# D1/D2 — parameter point -> physical non-axisymmetric Slice3D
# --------------------------------------------------------------------------
def theta_to_slice3d(theta_vec, active_names: Sequence[str], M_tot: float = 1.0,
                     fixed: Optional[Dict[str, float]] = None) -> Slice3D:
    """Map a parameter point to a ``solver_3d.Slice3D`` (single misaligned spin
    on A + optional off-axis momentum, head-on z-momenta).

    ``active_names`` labels the components of ``theta_vec`` (a subset/ordering of
    ``AXIS_NAMES_3D``); inactive knobs take ``DEFAULTS_3D`` (overridable via
    ``fixed``).

    Spin parametrisation (the open-question dual):
      * CARTESIAN if any of ``S_x``/``S_z`` is active or fixed:
        ``S_A = (S_x, 0, S_z)``;
      * POLAR otherwise: ``S_A = (S_mag sinθ_S, 0, S_mag cosθ_S)`` with θ_S in
        degrees (the ``run_3d_sweep.spin_vec`` convention).
    Spin on B is always zero (the Test-E single-spin family).
    """
    vals = dict(DEFAULTS_3D)
    if fixed:
        vals.update(fixed)
    active_names = list(active_names)
    for name, v in zip(active_names, np.atleast_1d(np.asarray(theta_vec, dtype=float))):
        vals[name] = float(v)

    q = vals["q"]
    m_A = M_tot * q / (1.0 + q)
    m_B = M_tot / (1.0 + q)

    # --- dimensionless-spin (chi) axes (add-only) -------------------------------
    # A chi_Xi axis carries the Kerr-like dimensionless spin S_Xi/m_X^2; the
    # Bowen--York source takes the physical spin, so convert here (m_A,m_B known):
    # S_Xi = chi_Xi * m_X^2.  Downstream every branch sees the physical S_*; the
    # existing S_* axes are untouched, so behaviour is byte-for-byte unchanged
    # unless a chi_* name is active or fixed.  A chi_* axis triggers the full-vector
    # spin branch (like the S_* generic-spin names).
    _m2 = {"A": m_A ** 2, "B": m_B ** 2}
    _chi_active = False
    for _cn in CHI_SPIN_NAMES_3D:
        if _cn in active_names or (fixed is not None and _cn in fixed):
            _chi_active = True
            vals["S_" + _cn[4:]] = float(vals[_cn]) * _m2[_cn[4]]
    _spin_full = _chi_active or (
        any(n in active_names for n in FULL_SPIN_NAMES_3D)
        or (fixed is not None and any(n in fixed for n in FULL_SPIN_NAMES_3D)))

    # quasi-circular (QC) branch — opt-in via ``fixed={"qc": 1.0}`` (a flag, not an
    # axis).  Replaces the head-on z-momenta with the deterministic PN quasi-circular
    # momenta ``quasicircular.qc_momenta(b, m_A, m_B, S_A, S_B)`` (tangential along x,
    # small radial along z; orbital L along +y).  The spin vectors are built by the
    # SAME logic as the non-QC branches (full-vector if a generic-spin name is
    # active/fixed, else planar single-spin on A), so QC composes with either spin
    # parametrisation.  When ``qc`` is unset the function is byte-for-byte unchanged.
    qc_on = bool(fixed is not None and fixed.get("qc", 0.0))
    if qc_on:
        if _spin_full:
            S_A_vec = (float(vals["S_Ax"]), float(vals["S_Ay"]), float(vals["S_Az"]))
            S_B_vec = (float(vals["S_Bx"]), float(vals["S_By"]), float(vals["S_Bz"]))
        else:
            cart = (("S_x" in active_names) or ("S_z" in active_names)
                    or (fixed is not None and ("S_x" in fixed or "S_z" in fixed)))
            if cart:
                S_Ax, S_Az = vals["S_x"], vals["S_z"]
            else:
                th = np.deg2rad(vals["theta_S"])
                S_Ax = vals["S_mag"] * np.sin(th)
                S_Az = vals["S_mag"] * np.cos(th)
            S_A_vec = (float(S_Ax), 0.0, float(S_Az))
            S_B_vec = (0.0, 0.0, 0.0)
        P_A_vec, P_B_vec = qcmod.qc_momenta(float(vals["b"]), m_A, m_B,
                                            S_A_vec, S_B_vec)
        return Slice3D(
            b=float(vals["b"]), m_A=float(m_A), m_B=float(m_B),
            P_A_vec=P_A_vec, P_B_vec=P_B_vec,
            S_A_vec=S_A_vec, S_B_vec=S_B_vec)

    # full generic-spin branch — BOTH punctures carry arbitrary 3-vectors (the 8-D
    # "vary both spin directions" family).  Triggered only when a generic-spin name
    # is active/fixed, so all existing single-spin families are unaffected.
    full_spin = _spin_full
    if full_spin:
        P = vals["P"]
        Px = vals["P_x"]
        return Slice3D(
            b=float(vals["b"]), m_A=float(m_A), m_B=float(m_B),
            P_A_vec=(float(Px), 0.0, float(-P)),
            P_B_vec=(float(-Px), 0.0, float(P)),
            S_A_vec=(float(vals["S_Ax"]), float(vals["S_Ay"]), float(vals["S_Az"])),
            S_B_vec=(float(vals["S_Bx"]), float(vals["S_By"]), float(vals["S_Bz"])),
        )

    # spin on A — Cartesian if any Cartesian knob is in play, else polar
    cart = (("S_x" in active_names) or ("S_z" in active_names)
            or (fixed is not None and ("S_x" in fixed or "S_z" in fixed)))
    if cart:
        S_Ax, S_Az = vals["S_x"], vals["S_z"]
    else:
        th = np.deg2rad(vals["theta_S"])
        S_Ax = vals["S_mag"] * np.sin(th)
        S_Az = vals["S_mag"] * np.cos(th)

    P = vals["P"]
    Px = vals["P_x"]
    return Slice3D(
        b=float(vals["b"]), m_A=float(m_A), m_B=float(m_B),
        P_A_vec=(float(Px), 0.0, float(-P)),
        P_B_vec=(float(-Px), 0.0, float(P)),
        S_A_vec=(float(S_Ax), 0.0, float(S_Az)),
        S_B_vec=(0.0, 0.0, 0.0),
    )


# --------------------------------------------------------------------------
# D7 — per-b assembly cache (byte-identical to solver_3d.assemble)
# --------------------------------------------------------------------------
def assemble_cached_3d(prob: Problem3D, sl: Slice3D,
                       cache: Dict[float, tuple]) -> s3.Assembly3D:
    """``solver_3d.assemble`` with the b-dependent geometry + per-m operators
    cached by ``sl.b``.

    Cached (depend only on b + the frozen grid/Nφ): the per-m BC-applied block
    operators / B-factors (``ops3.mode_operators``) and the meridian geometry
    (``rho, z`` and the finite-edge masks).  Rebuilt per slice: ``ψ_BL`` (b +
    masses) and the non-axisymmetric ``Â²`` (b + momenta + spins).  Calls the
    IDENTICAL ``ops3``/``source``/``source_3d`` functions as ``solver_3d.assemble``,
    so the returned ``Assembly3D`` is byte-identical to a fresh ``s3.assemble``
    (verified in the test suite).
    """
    key = sl.b
    geo = cache.get(key)
    if geo is None:
        Lap, rho, z, Af, Bf, DA, DB, inv_rho2 = ops3.axisym_blocks(
            prob.A, prob.B, prob.DA1, prob.DB1, sl.b)
        M0_list, w_list, interior = ops3.mode_operators(
            prob.A, prob.B, prob.DA1, prob.DB1, sl.b, prob.m_vals)
        finite = np.isfinite(rho)
        rho_s = np.where(finite, rho, 1.0)
        z_s = np.where(finite, z, 0.0)
        geo = (M0_list, w_list, interior, rho, z, finite, rho_s, z_s)
        cache[key] = geo
    M0_list, w_list, interior, rho, z, finite, rho_s, z_s = geo

    psi = np.array(source.psi_BL_2c(rho_s, z_s, sl.b, sl.m_A, sl.m_B))
    psi = np.where(finite, psi, 1.0)
    A2 = source_3d.A2_at_nodes_3d(rho, z, prob.phi, sl.b,
                                  sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec)
    return s3.Assembly3D(M0=M0_list, w=w_list, interior=interior, rho=rho, z=z,
                         psi=psi, A2=A2, m_vals=prob.m_vals)


# --------------------------------------------------------------------------
# solve_fn factory — the certified (NK) or cheap (modified-Newton) forward map
# --------------------------------------------------------------------------
def make_solve_fn(prob: Problem3D, active_names: Sequence[str], M_tot: float = 1.0,
                  fixed: Optional[Dict[str, float]] = None, use_cache: bool = True,
                  solver: str = "nk", gmres_rtol: float = 1e-4,
                  retry_tol: Optional[float] = None):
    """Return ``solve_fn(theta, guess, tol, max_iter) -> (U, info)`` (+ its cache).

    ``solver='nk'`` (default) → the **certified** Newton–Krylov solve; the
    returned ``info.residual_norm`` is the equilibrated (certified) constraint
    residual.  ``solver='modified'`` → the cheap modified-Newton solve (same
    converged field, looser monitor) for production sweeps / convergence studies.

    For ``solver='nk'`` we pass ``max_iter+1`` to ``newton_solve_nk`` so the
    residual reported is the one AFTER ``max_iter`` Newton steps (the
    ``evaluate_polished_nk`` convention — its loop measures the residual at the
    start of each iteration, so the +1 certifies the final step).  Hence
    ``ParametricSolutionND.evaluate_polished(θ, newton_steps=2)`` performs exactly
    2 certified NK steps.

    ``retry_tol`` (default ``None`` → OFF, behaviour bit-for-bit unchanged) enables
    a **damped-Newton globalization fallback** for the ``nk`` build path. At
    extreme parameter corners (observed at ``b=7`` + high ``q`` + strong spin)
    NK's exact-Jacobian Newton suffers a *global*-convergence failure: it
    stagnates at a high-residual plateau (~3 iters) from *any* available start
    (warm OR cold), while the damped modified-Newton reaches the basin. When an NK
    solve returns ``residual_norm > retry_tol``, we reach the basin with a cold
    modified-Newton solve and then NK-polish from that field, which certifies to
    the NK floor (~4e-11 in ~2 steps — the exact Newton converges *locally* once
    near the solution); the better iterate is kept. Only fires on stagnation, so
    well-converged builds are untouched. (The modified step is used purely as a
    globalization — like a line search / trust region — not as a separate build
    solver: the stored node still lands at the certified NK floor.)
    """
    cache: Dict[float, tuple] = {}
    active_names = list(active_names)

    def solve_fn(theta, guess, tol, max_iter):
        sl = theta_to_slice3d(theta, active_names, M_tot, fixed)
        asm = assemble_cached_3d(prob, sl, cache) if use_cache else None
        if solver == "nk":
            U, info = s3nk.newton_solve_nk(prob, sl, U0=guess, tol=tol,
                                           max_iter=int(max_iter) + 1, asm=asm,
                                           gmres_rtol=gmres_rtol)
            if retry_tol is not None and info.residual_norm > retry_tol:
                Um, _ = s3.newton_solve(prob, sl, U0=None, tol=tol,
                                        max_iter=max(60, int(max_iter)), asm=asm)
                U2, info2 = s3nk.newton_solve_nk(prob, sl, U0=np.asarray(Um), tol=tol,
                                                 max_iter=int(max_iter) + 1, asm=asm,
                                                 gmres_rtol=gmres_rtol)
                if info2.residual_norm < info.residual_norm:
                    U, info = U2, info2
            return U, info
        return s3.newton_solve(prob, sl, U0=guess, tol=tol,
                               max_iter=int(max_iter), asm=asm)

    return solve_fn, cache


# --------------------------------------------------------------------------
# Wiring: build a ParametricSolverND around the 3-D solver
# --------------------------------------------------------------------------
def from_problem_nd_3d(prob: Problem3D, axes: Sequence[dict], M_tot: float = 1.0,
                       fixed: Optional[Dict[str, float]] = None, use_cache: bool = True,
                       solver: str = "nk", gmres_rtol: float = 1e-4,
                       retry_tol: Optional[float] = None):
    """``ParametricSolverND`` over the active axes ``[{name,min,max,Q}, ...]``.

    Put ``b`` first (the outermost snake axis) to maximise D7 cache reuse.  Use
    ``solver='modified'`` for the (cheaper, field-identical) held-out convergence
    studies and ``solver='nk'`` (default) for the certified-prediction gate.
    """
    active_names = [a["name"] for a in axes]
    solve_fn, _ = make_solve_fn(prob, active_names, M_tot=M_tot, fixed=fixed,
                                use_cache=use_cache, solver=solver, gmres_rtol=gmres_rtol,
                                retry_tol=retry_tol)
    spec = [(a["min"], a["max"], a["Q"]) for a in axes]
    return pnd.ParametricSolverND(solve_fn, spec)


# --------------------------------------------------------------------------
# Held-out parameter samples (generic — never on any CGL node set)
# --------------------------------------------------------------------------
# Per-axis irrational-ish fractions (the P1 set) cycled across axes so each axis
# is probed at a different fraction in each held-out tuple.
_FRACS = np.array([0.137, 0.371, 0.523, 0.689, 0.853, 0.293])


def holdout_points_1axis(p_min, p_max, fracs=None):
    fr = _FRACS[:5] if fracs is None else np.asarray(fracs, float)
    return p_min + fr * (p_max - p_min)


def holdout_points_nd(axes: Sequence[dict], n_points: int = 6):
    """``n_points`` generic off-node θ in the box, each axis at a cycled fraction."""
    pts = []
    for i in range(n_points):
        theta = []
        for k, a in enumerate(axes):
            fr = _FRACS[(i + k) % len(_FRACS)]
            theta.append(a["min"] + fr * (a["max"] - a["min"]))
        pts.append(np.array(theta, dtype=float))
    return pts


def assert_off_node(theta_pts, axes, gap_min=1e-4):
    """Every held-out θ must be genuinely OFF each axis's CGL nodes (else the
    interpolation error collapses and the decay check is vacuous)."""
    for kidx, a in enumerate(axes):
        nodes, _ = cheb_param_nodes(a["min"], a["max"], a["Q"])
        for theta in theta_pts:
            gap = float(np.min(np.abs(theta[kidx] - nodes)))
            assert gap > gap_min, (
                f"axis {a['name']}: held-out within {gap:.1e} of a Q={a['Q']} node")


# --------------------------------------------------------------------------
# Study 1 — per-axis held-out convergence (single active axis via the N-D layer)
# --------------------------------------------------------------------------
def held_out_convergence_1axis(prob, name, p_min, p_max, Qs, M_tot=1.0,
                               fixed=None, fracs=None, use_cache=True,
                               solver="nk", tol=1e-12, max_iter=30):
    """Held-out interp-in-``name`` error vs Q (all other knobs fixed).

    ``ps.evaluate([p])`` vs a direct solve at each held-out ``p`` (same frozen
    grid → the spatial error cancels, leaving genuine interpolation-in-parameter
    error).  Returns ``(rows, hold)`` with ``rows=[(Q, err, sweep_iters), ...]``.

    Uses the cheap modified-Newton field by default (bit-identical to NK), so the
    convergence study is fast; switch ``solver='nk'`` for a certified study.
    """
    hold = holdout_points_1axis(p_min, p_max, fracs)
    active = [name]
    solve_fn, _ = make_solve_fn(prob, active, M_tot=M_tot, fixed=fixed,
                                use_cache=use_cache, solver=solver)
    U_direct = {}
    for p in hold:
        U, _ = solve_fn(np.array([float(p)]), None, tol, max_iter)
        U_direct[float(p)] = np.asarray(U)

    rows = []
    for Q in Qs:
        ps = from_problem_nd_3d(prob, [{"name": name, "min": p_min, "max": p_max, "Q": Q}],
                                M_tot=M_tot, fixed=fixed, use_cache=use_cache,
                                solver=solver).build(tol=tol, max_iter=max_iter)
        e = max(float(np.max(np.abs(ps.evaluate([float(p)]) - U_direct[float(p)])))
                for p in hold)
        rows.append((Q, e, int(np.sum(ps.iters))))
    return rows, hold


# --------------------------------------------------------------------------
# Study 2 — joint multi-D held-out convergence
# --------------------------------------------------------------------------
def held_out_convergence_joint(prob, axes_template, Qs_per_dim, theta_holdout=None,
                               M_tot=1.0, fixed=None, use_cache=True,
                               solver="nk", tol=1e-12, max_iter=30):
    """Joint held-out error at generic θ vs the (per-dim) Q.

    ``axes_template`` = ``[{name,min,max}, ...]``; ``Qs_per_dim`` a list of
    Q-tuples (one per refinement level).  Returns ``(rows, theta_holdout)`` with
    ``rows=[(Qtuple, n_nodes, err, iters), ...]``.
    """
    if theta_holdout is None:
        probe_axes = [dict(a, Q=8) for a in axes_template]   # Q only for off-node check
        theta_holdout = holdout_points_nd(probe_axes)
    active = [a["name"] for a in axes_template]
    solve_fn, _ = make_solve_fn(prob, active, M_tot=M_tot, fixed=fixed,
                                use_cache=use_cache, solver=solver)
    U_direct = []
    for theta in theta_holdout:
        Ud, _ = solve_fn(np.asarray(theta, float), None, tol, max_iter)
        U_direct.append(np.asarray(Ud))

    rows = []
    for Qs in Qs_per_dim:
        axes = [dict(a, Q=int(Qk)) for a, Qk in zip(axes_template, Qs)]
        ps = from_problem_nd_3d(prob, axes, M_tot=M_tot, fixed=fixed,
                                use_cache=use_cache, solver=solver).build(
            tol=tol, max_iter=max_iter)
        e = max(float(np.max(np.abs(ps.evaluate(theta) - U_direct[i])))
                for i, theta in enumerate(theta_holdout))
        rows.append((tuple(int(q) for q in Qs), ps.n_nodes, e, int(np.sum(ps.iters))))
    return rows, theta_holdout


# --------------------------------------------------------------------------
# Study 3 — the analyticity walls
# --------------------------------------------------------------------------
def analyticity_wall_b(prob, b_mins, b_max, Qs, M_tot=1.0, fixed=None,
                       fit_window=None, use_cache=True, solver="nk",
                       tol=1e-12, max_iter=30):
    """Merger wall: geometric rate at several ``b_min`` vs the b=0 Bernstein pred.

    Reproduces P1 (b is the slow, hard wall — nearest singularity at the b=0
    merger).  Returns a list of dicts ``{b_min, Qs, errs, iters, rate, rate_pred}``.
    """
    out = []
    for b_min in b_mins:
        rows, _ = held_out_convergence_1axis(
            prob, "b", b_min, b_max, Qs, M_tot=M_tot, fixed=fixed,
            use_cache=use_cache, solver=solver, tol=tol, max_iter=max_iter)
        Q_arr = [r[0] for r in rows]
        e_arr = [r[1] for r in rows]
        it_arr = [r[2] for r in rows]
        q_lo, q_hi = (None, None) if fit_window is None else fit_window
        out.append(dict(b_min=float(b_min), Qs=Q_arr, errs=e_arr, iters=it_arr,
                        rate=geometric_rate(Q_arr, e_arr, q_lo, q_hi),
                        rate_pred=bernstein_rate_from_zero(b_min, b_max)))
    return out


def analyticity_wall_spin(prob, name, p_max, Qs, b=2.0, M_tot=1.0, fixed=None,
                          fit_window=(4, 12), use_cache=True, solver="nk",
                          tol=1e-12, max_iter=30):
    """Spin wall: rate vs ``p_max`` on ``[0, p_max]`` for a single spin axis.

    ``name`` is ``'theta_S'`` (POLAR tilt, with ``S_mag`` in ``fixed``) or
    ``'S_x'`` (CARTESIAN, with ``S_z`` in ``fixed``) — the open-question dual.
    The spin walls are expected SOFT/FAR (smooth dependence).  Returns a list of
    dicts ``{p_max, Qs, errs, iters, rate, p_star}`` where ``p_star`` is the
    nearest REAL singularity inferred from the rate.
    """
    out = []
    base_fixed = dict(fixed or {})
    base_fixed.setdefault("b", b)
    for pm in p_maxes_or_list(p_max):
        rows, _ = held_out_convergence_1axis(
            prob, name, 0.0, pm, Qs, M_tot=M_tot, fixed=base_fixed,
            use_cache=use_cache, solver=solver, tol=tol, max_iter=max_iter)
        Q_arr = [r[0] for r in rows]
        e_arr = [r[1] for r in rows]
        it_arr = [r[2] for r in rows]
        rate = geometric_rate(Q_arr, e_arr, *fit_window)
        out.append(dict(p_max=float(pm), Qs=Q_arr, errs=e_arr, iters=it_arr,
                        rate=rate,
                        p_star=infer_real_singularity(0.0, pm, rate, side="right")))
    return out


def p_maxes_or_list(p_max):
    """Accept a scalar or a list of upper bounds for the spin-wall sweep."""
    if np.isscalar(p_max):
        return [float(p_max)]
    return [float(x) for x in p_max]
