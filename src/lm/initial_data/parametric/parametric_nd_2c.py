"""LM-initial-data — wire the N-D parametric layer to the ABT two-centre solver (P3).

The multi-dimensional milestone.  ``parametric_nd.py`` (the solver-agnostic
tensor-product Chebyshev-in-parameter layer) is reused; this module injects the
single callable it needs — ``solve_fn(theta_vec, guess, tol, max_iter)`` — that
maps a 4-D parameter point ``θ = (q, b, χ_A, χ_B)`` onto a head-on(+aligned-spin)
``Slice`` and runs the validated ABT Newton solve.

Conventions (reports/P3/analysis.md §0):
  D1  χ_X = S_X/m_X²  (bare puncture mass m_X)  ⇒  S_X = χ_X m_X².
  D2  fixed total mass M:  m_A = M q/(1+q),  m_B = M/(1+q);  P fixed in Problem.
  D7  per-b Laplacian cache: M0 depends only on b (Lap∝1/b², BC rows b-indep), so
      with b the outermost axis it is built once per b-node and reused for all
      (q,χ_A,χ_B) at that b.  ``assemble_cached`` calls the SAME ops/source
      functions as ``solver_abt.assemble`` → byte-identical Assembly.

Also provides the analyticity-wall machinery (b reproduces P1; the χ soft wall is
new), the held-out convergence studies (per-axis + joint), and the cost-scaling
report.

Standalone: numpy + jax + the sibling modules (solver_abt, operators_abt, source,
parametric_nd).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..solver import solver_abt as sa
from ..solver import operators_abt as ops
from ..solver import source
from . import parametric_nd as pnd
from .parametric import cheb_param_nodes


# canonical axis order and the inactive-axis defaults (D3)
AXIS_NAMES = ("q", "b", "chi_A", "chi_B")
DEFAULTS = {"q": 1.0, "b": 4.0, "chi_A": 0.0, "chi_B": 0.0}


# --------------------------------------------------------------------------
# D1/D2 — parameter point -> physical head-on(+spin) Slice
# --------------------------------------------------------------------------
def theta_to_slice(theta_vec, active_names: Sequence[str], M_tot: float = 1.0,
                   fixed: Optional[Dict[str, float]] = None) -> sa.Slice:
    """Map a parameter point to a ``solver_abt.Slice`` (D1/D2).

    ``active_names`` labels the components of ``theta_vec`` (a subset/ordering of
    ``AXIS_NAMES``); inactive knobs take ``DEFAULTS`` (overridable via ``fixed``).
    """
    vals = dict(DEFAULTS)
    if fixed:
        vals.update(fixed)
    for name, v in zip(active_names, np.atleast_1d(np.asarray(theta_vec, dtype=float))):
        vals[name] = float(v)
    q = vals["q"]
    m_A = M_tot * q / (1.0 + q)
    m_B = M_tot / (1.0 + q)
    S_A = vals["chi_A"] * m_A ** 2          # D1: S = χ m_bare²
    S_B = vals["chi_B"] * m_B ** 2
    return sa.Slice(float(vals["b"]), float(m_A), float(m_B), float(S_A), float(S_B))


# --------------------------------------------------------------------------
# D7 — per-b assembly cache (byte-identical to solver_abt.assemble)
# --------------------------------------------------------------------------
def assemble_cached(prob: sa.Problem, sl: sa.Slice, cache: Dict[float, tuple]) -> sa.Assembly:
    """``solver_abt.assemble`` with the b-dependent geometry (M0, grid, masks)
    cached by ``sl.b``.  Calls the identical ``ops``/``source`` functions, so the
    returned ``Assembly`` is byte-identical to a fresh ``sa.assemble`` (verified
    in the test suite)."""
    key = sl.b
    geo = cache.get(key)
    if geo is None:
        Lap, rho, z, Af, Bf, DA, DB = ops.laplacian_matrix(
            prob.A, prob.B, prob.DA1, prob.DB1, sl.b)
        M0, interior = ops.apply_bcs(Lap, prob.A, prob.B, DA)
        finite = np.isfinite(rho)
        rho_s = np.where(finite, rho, 1.0)
        z_s = np.where(finite, z, 0.0)
        geo = (M0, interior, rho, z, finite, rho_s, z_s)
        cache[key] = geo
    M0, interior, rho, z, finite, rho_s, z_s = geo
    psi = np.array(source.psi_BL_2c(rho_s, z_s, sl.b, sl.m_A, sl.m_B))
    A2 = np.array(source.A2_2c_spin(rho_s, z_s, sl.b, prob.P, sl.S_A, sl.S_B))
    psi = np.where(finite, psi, 1.0)
    A2 = np.where(finite, A2, 0.0)
    return sa.Assembly(M0=M0, interior=interior, rho=rho, z=z, psi=psi, A2=A2)


# --------------------------------------------------------------------------
# Wiring: build a ParametricSolverND around the ABT spatial solver
# --------------------------------------------------------------------------
def from_problem_nd(prob: sa.Problem, axes: Sequence[dict], M_tot: float = 1.0,
                    fixed: Optional[Dict[str, float]] = None, use_cache: bool = True):
    """ParametricSolverND over the active axes ``[{name,min,max,Q}, ...]``.

    For bit-for-bit reduction to the 1-D P1 sweeps put ``use_cache=False`` (then
    ``solve_fn`` calls ``sa.newton_solve`` with ``asm=None``, i.e. the exact P1
    code path); the cache is a verified-equivalent accelerator (D7).
    """
    active_names = [a["name"] for a in axes]
    cache: Dict[float, tuple] = {}

    def solve_fn(theta, guess, tol, max_iter):
        sl = theta_to_slice(theta, active_names, M_tot, fixed)
        asm = assemble_cached(prob, sl, cache) if use_cache else None
        return sa.newton_solve(prob, sl, U0=guess, tol=tol, max_iter=max_iter, asm=asm)

    spec = [(a["min"], a["max"], a["Q"]) for a in axes]
    return pnd.ParametricSolverND(solve_fn, spec)


# --------------------------------------------------------------------------
# Held-out parameter samples (generic — never on any CGL node set)
# --------------------------------------------------------------------------
# Per-axis irrational-ish fractions (the P1 set) cycled across axes so each axis
# is probed at a different fraction in each held-out 4-tuple.
_FRACS = np.array([0.137, 0.371, 0.523, 0.689, 0.853, 0.293])


def holdout_points_1axis(p_min, p_max, fracs=None):
    fr = _FRACS[:5] if fracs is None else np.asarray(fracs, float)
    return p_min + fr * (p_max - p_min)


def holdout_points_nd(axes: Sequence[dict], n_points: int = 6):
    """``n_points`` generic off-node θ in the box, each axis at a cycled fraction."""
    d = len(axes)
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
    interpolation error collapses and the decay check is vacuous — the P1
    adversarial-review hardening, lifted to N-D)."""
    for a in axes:
        nodes, _ = cheb_param_nodes(a["min"], a["max"], a["Q"])
        # the axis component index in theta is its position in `axes`
        kidx = list(axes).index(a)
        for theta in theta_pts:
            gap = float(np.min(np.abs(theta[kidx] - nodes)))
            assert gap > gap_min, (
                f"axis {a['name']}: held-out within {gap:.1e} of a Q={a['Q']} node")


# --------------------------------------------------------------------------
# Study 1 — per-axis held-out convergence (single active axis via the N-D layer)
# --------------------------------------------------------------------------
def held_out_convergence_1axis(prob, name, p_min, p_max, Qs, M_tot=1.0,
                               fixed=None, fracs=None, use_cache=True,
                               tol=1e-12, max_iter=20):
    """Held-out interp-in-``name`` error vs Q (all other knobs fixed).

    ``ps.evaluate([p])`` vs a direct ``newton_solve`` at each held-out ``p`` (same
    frozen grid → spatial error cancels).  Returns ``(rows, hold)`` with
    ``rows=[(Q, err, sweep_iters), ...]``.
    """
    hold = holdout_points_1axis(p_min, p_max, fracs)
    active = [name]
    U_direct = {}
    cache_d: Dict[float, tuple] = {}
    for p in hold:
        sl = theta_to_slice([float(p)], active, M_tot, fixed)
        asm = assemble_cached(prob, sl, cache_d) if use_cache else None
        Up, _ = sa.newton_solve(prob, sl, tol=tol, max_iter=25, asm=asm)
        U_direct[float(p)] = np.asarray(Up)

    rows = []
    for Q in Qs:
        ps = from_problem_nd(prob, [{"name": name, "min": p_min, "max": p_max, "Q": Q}],
                             M_tot=M_tot, fixed=fixed, use_cache=use_cache).build(
            tol=tol, max_iter=max_iter)
        e = max(float(np.max(np.abs(ps.evaluate([float(p)]) - U_direct[float(p)])))
                for p in hold)
        rows.append((Q, e, int(np.sum(ps.iters))))
    return rows, hold


# --------------------------------------------------------------------------
# Study 2 — joint multi-D held-out convergence
# --------------------------------------------------------------------------
def held_out_convergence_joint(prob, axes_template, Qs_per_dim, theta_holdout=None,
                               M_tot=1.0, fixed=None, use_cache=True,
                               tol=1e-12, max_iter=20):
    """Joint held-out error at generic 4-D points vs the (isotropic or per-dim) Q.

    ``axes_template`` = ``[{name,min,max}, ...]`` (no Q); ``Qs_per_dim`` is a list
    of Q-tuples (one per refinement level), each giving the Q for every axis.
    Returns ``(rows, theta_holdout)`` with ``rows=[(Qtuple, n_nodes, err, iters), ...]``.
    """
    if theta_holdout is None:
        probe_axes = [dict(a, Q=8) for a in axes_template]   # Q only for off-node check
        theta_holdout = holdout_points_nd(probe_axes)
    active = [a["name"] for a in axes_template]
    cache_d: Dict[float, tuple] = {}
    U_direct = []
    for theta in theta_holdout:
        sl = theta_to_slice(theta, active, M_tot, fixed)
        asm = assemble_cached(prob, sl, cache_d) if use_cache else None
        Ud, _ = sa.newton_solve(prob, sl, tol=tol, max_iter=25, asm=asm)
        U_direct.append(np.asarray(Ud))

    rows = []
    for Qs in Qs_per_dim:
        axes = [dict(a, Q=int(Qk)) for a, Qk in zip(axes_template, Qs)]
        ps = from_problem_nd(prob, axes, M_tot=M_tot, fixed=fixed,
                             use_cache=use_cache).build(tol=tol, max_iter=max_iter)
        e = max(float(np.max(np.abs(ps.evaluate(theta) - U_direct[i])))
                for i, theta in enumerate(theta_holdout))
        rows.append((tuple(int(q) for q in Qs), ps.n_nodes, e, int(np.sum(ps.iters))))
    return rows, theta_holdout


# --------------------------------------------------------------------------
# Geometric rate + Bernstein nearest-singularity machinery
# --------------------------------------------------------------------------
def geometric_rate(Qs, errs, q_lo=None, q_hi=None):
    """Decades of held-out error lost per unit Q (``-slope`` of log10(err) vs Q),
    fit over the clean ``[q_lo, q_hi]`` window (so a high-Q floor does not
    contaminate the slope)."""
    Qs = np.asarray(Qs, float)
    errs = np.asarray(errs, float)
    mask = errs > 0
    if q_lo is not None:
        mask &= Qs >= q_lo
    if q_hi is not None:
        mask &= Qs <= q_hi
    return -float(np.polyfit(Qs[mask], np.log10(errs[mask]), 1)[0])


def bernstein_rate(p_sing, p_min, p_max):
    """Predicted decades/Q for a (possibly complex) nearest singularity ``p_sing``.

    A Chebyshev interpolant on ``[p_min,p_max]`` converges like ``ρ^{-Q}``; ``ρ``
    is the Bernstein parameter of the singularity's image under the affine map to
    ``[-1,1]``: ``ξ = (2 p_sing - (p_max+p_min))/(p_max-p_min)``, ``ρ`` the larger
    of ``|ξ ± sqrt(ξ²-1)|``.  Returns ``log10(ρ)``.
    """
    xi = (2.0 * p_sing - (p_max + p_min)) / (p_max - p_min)
    xi = complex(xi)
    s = np.sqrt(xi * xi - 1.0)
    rho = max(abs(xi + s), abs(xi - s))
    return float(np.log10(rho))


def bernstein_rate_from_zero(b_min, b_max):
    """The a-priori merger prediction: nearest singularity at ``b=0`` (P1)."""
    return bernstein_rate(0.0, b_min, b_max)


def infer_real_singularity(p_min, p_max, rate, side="right"):
    """Invert a measured ``rate`` to the nearest REAL singularity location.

    ``rate = log10(ρ)`` ⇒ ``ρ = 10**rate`` ⇒ ``ξ = (ρ + 1/ρ)/2 ≥ 1``.  The
    singularity lies outside ``[p_min,p_max]``: ``side='right'`` →
    ``p* = midpoint + (half-width)·ξ``; ``side='left'`` → ``midpoint - half·ξ``.
    """
    rho = 10.0 ** rate
    xi = 0.5 * (rho + 1.0 / rho)
    mid = 0.5 * (p_min + p_max)
    half = 0.5 * (p_max - p_min)
    return mid + (xi if side == "right" else -xi) * half


# --------------------------------------------------------------------------
# Study 3 — the analyticity walls
# --------------------------------------------------------------------------
def analyticity_wall_b(prob, b_mins, b_max, Qs, M_tot=1.0, fixed=None,
                       fit_window=None, use_cache=True, tol=1e-12, max_iter=20):
    """Merger wall: geometric rate at several ``b_min`` vs the b=0 Bernstein pred.

    Reproduces P1 (b is the slow, hard wall).  Returns a list of dicts
    ``{b_min, Qs, errs, iters, rate, rate_pred}``.
    """
    out = []
    for b_min in b_mins:
        rows, _ = held_out_convergence_1axis(
            prob, "b", b_min, b_max, Qs, M_tot=M_tot, fixed=fixed,
            use_cache=use_cache, tol=tol, max_iter=max_iter)
        Q_arr = [r[0] for r in rows]
        e_arr = [r[1] for r in rows]
        it_arr = [r[2] for r in rows]
        q_lo, q_hi = (None, None) if fit_window is None else fit_window
        out.append(dict(b_min=float(b_min), Qs=Q_arr, errs=e_arr, iters=it_arr,
                        rate=geometric_rate(Q_arr, e_arr, q_lo, q_hi),
                        rate_pred=bernstein_rate_from_zero(b_min, b_max)))
    return out


def analyticity_wall_chi(prob, chi_maxes, Qs, b=4.0, M_tot=1.0,
                         which="chi_A", chi_other=0.0, fit_window=(4, 12),
                         use_cache=True, tol=1e-12, max_iter=20):
    """Spin wall: rate vs ``χ_max`` on ``[0, χ_max]`` (single active spin axis).

    The other spin is held at ``chi_other`` (default 0 → single-spin, the clean
    1-D family; equal-spin gives the same rate, verified).  Returns a list of
    dicts ``{chi_max, Qs, errs, iters, rate, chi_star}`` where ``chi_star`` is the
    nearest REAL singularity inferred from the rate (D8: a soft, far wall).
    """
    other = "chi_B" if which == "chi_A" else "chi_A"
    out = []
    for cm in chi_maxes:
        fixed = {"b": b, other: chi_other}
        rows, _ = held_out_convergence_1axis(
            prob, which, 0.0, cm, Qs, M_tot=M_tot, fixed=fixed,
            use_cache=use_cache, tol=tol, max_iter=max_iter)
        Q_arr = [r[0] for r in rows]
        e_arr = [r[1] for r in rows]
        it_arr = [r[2] for r in rows]
        rate = geometric_rate(Q_arr, e_arr, *fit_window)
        out.append(dict(chi_max=float(cm), Qs=Q_arr, errs=e_arr, iters=it_arr,
                        rate=rate,
                        chi_star=infer_real_singularity(0.0, cm, rate, side="right")))
    return out


# --------------------------------------------------------------------------
# Study 4 — cost scaling  Q^d  and the sparse-grid crossover (D9)
# --------------------------------------------------------------------------
def fit_error_model(rows, q_lo=None, q_hi=None):
    """Fit ``log10(err) = log10(C) - rate·Q`` over a clean window.

    Returns ``(rate, log10C)`` so ``Q_needed(eps) = (log10C - log10(eps))/rate``.
    """
    Qs = np.asarray([r[0] for r in rows], float)
    es = np.asarray([r[1] for r in rows], float)
    mask = es > 0
    if q_lo is not None:
        mask &= Qs >= q_lo
    if q_hi is not None:
        mask &= Qs <= q_hi
    slope, intercept = np.polyfit(Qs[mask], np.log10(es[mask]), 1)
    return -float(slope), float(intercept)


def Q_needed(eps, rate, log10C):
    """CGL order Q to reach held-out error ``eps`` for an axis with this model."""
    return (log10C - np.log10(eps)) / rate


def cost_table(axis_models: Dict[str, tuple], eps=1e-9):
    """Tensor-grid node counts for d=1..len(axes), anisotropic vs isotropic Q,
    plus the Smolyak/sparse crossover note (D9).

    ``axis_models[name] = (rate, log10C)``.  Returns a list of dicts.
    """
    names = list(axis_models)
    Q_aniso = {n: int(np.ceil(Q_needed(eps, *axis_models[n]))) for n in names}
    Q_iso = max(Q_aniso.values())                            # isotropic must match slowest
    out = []
    for d in range(1, len(names) + 1):
        sub = names[:d]
        n_aniso = int(np.prod([Q_aniso[n] + 1 for n in sub]))
        n_iso = int((Q_iso + 1) ** d)
        out.append(dict(d=d, axes=tuple(sub),
                        Q_aniso={n: Q_aniso[n] for n in sub}, Q_iso=Q_iso,
                        n_aniso=n_aniso, n_iso=n_iso))
    return out, Q_aniso, Q_iso


def smolyak_points(d, level):
    """Approx. Clenshaw–Curtis Smolyak sparse-grid point count at ``(d, level)``.

    Exact small-level count via the standard nested CC rule (m(0)=1, m(i)=2^i+1):
    ``|A(d,ℓ)| = Σ_{ℓ≤|i|≤ℓ+d-1} ∏ (m(i_k)-m(i_k-1))`` over multi-indices i≥1.
    Used only to illustrate the ``O(2^ℓ ℓ^{d-1})`` vs ``(2^ℓ)^d`` crossover.
    """
    from itertools import product as iproduct

    def m(i):
        return 1 if i == 0 else (2 ** i + 1)

    def dm(i):                                               # new points at CC level i
        return 1 if i == 0 else m(i) - m(i - 1)
    total = 0
    # multi-indices i_k >= 0 with sum <= level (standard CC Smolyak, 0-based levels)
    for idx in iproduct(range(level + 1), repeat=d):
        if sum(idx) <= level:
            total += int(np.prod([dm(j) for j in idx]))
    return total
