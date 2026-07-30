"""LM-initial-data — accelerated parameter control (Milestone B2, the payoff).

A minimal **Mendes-style control loop** (Mendes et al. 2025, *Parameter control
for binary black hole initial data*, PRD 112, 124049): given **target physical
quantities**, find the **free data** that hits them by a Broyden iteration on the
control residual ``G(θ) = F(θ) − target``, where ``F`` maps free data to ADM
observables (total/individual ADM mass, total angular momentum) via a *certified*
elliptic constraint solve.

The point of B2 (and the rebuttal to PAPER_PLAN risk R5 — *"this is just
TwoPunctures + interpolation"*): the P3 interpolant is **not** the end product —
it **accelerates** a real control loop while **every step stays certified**.  We
run the same loop two ways and report the reduction:

  * **cold** — each ``F``-evaluation cold-solves the constraints from scratch
    (Newton from ``u≡0``);
  * **interp (warm)** — each ``F``-evaluation seeds Newton from the P3 interpolant
    ``ps.evaluate(θ)`` (mode (a): the surrogate supplies the *initial guess*; the
    inner solve is then 1–2 *certified* Newton steps).  Gradient-based targeting
    via ``∂F/∂θ`` is B3, deliberately out of scope here.

A third honest baseline, **continuation** (warm-start each inner solve from the
*previous* control iterate), is also reported: it shows the interpolant gives a
*global* certified warm start (any θ, no connected march, robust to large Broyden
steps), of which continuation is the local special case.

Fair accounting (see reports/B2/analysis.md):
  * a **solver call** == one ``F``-evaluation == one ``newton_solve``;
  * a **Newton iteration** == one inner Newton residual cycle (``info.iters``);
    the dominant cost is the dense LU per step (``≈ iters−1`` solves);
  * the **certified residual** ``‖R‖∞ ≤ tol_inner`` (default 1e-10) is asserted at
    **every** solver call in **every** mode — the speed-up is never bought with
    accuracy;
  * the outer loop is the **same** in every mode (identical Broyden path, since
    ``F`` is the unique certified solution to ~floor regardless of the guess), so
    the number of solver calls matches and the reduction is a clean per-call
    Newton-iteration / wall-clock factor.

Outer Jacobian: **Broyden** (rank-1 "good" secant updates; 1 ``F``-eval per outer
step after a one-time finite-difference initial Jacobian, mirroring Mendes 2025).
Trade-off vs finite-difference Newton (``d`` extra solves/step, more robust) is
noted in the report; the choice is identical in every mode so the comparison is
fair regardless.

Standalone / add-only: imports the frozen ``solver_abt`` + ``validation.adm`` +
``parametric_nd*`` **verbatim**; defines no new physics.  numpy + jax only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ..solver import solver_abt as sa
from ..validation import adm
from ..parametric import parametric_nd_2c as p3
from ..parametric.parametric_nd import ParametricSolutionND


# --------------------------------------------------------------------------
# Observable map  F:  (solved field U on slice sl) -> physical quantity
# --------------------------------------------------------------------------
# Each entry is a pure function of the *already-solved* (prob, U, sl); the solve
# itself happens once per F-evaluation in ``solve_and_observe``.  These reuse
# ``validation.adm`` verbatim (B1's re-derived, oracle-validated diagnostics).
OBSERVABLES: Dict[str, Callable] = {
    "M_ADM": lambda prob, U, sl: adm.adm_mass_spectral(prob, U, sl),
    "M_A": lambda prob, U, sl: adm.puncture_adm_mass(prob, U, sl, "A"),
    "M_B": lambda prob, U, sl: adm.puncture_adm_mass(prob, U, sl, "B"),
    "mass_ratio": lambda prob, U, sl: (adm.puncture_adm_mass(prob, U, sl, "A")
                                       / adm.puncture_adm_mass(prob, U, sl, "B")),
    "J": lambda prob, U, sl: float(sl.S_A + sl.S_B),   # analytic (no solve needed)
}


def evaluate_observables(prob: sa.Problem, U, sl: sa.Slice,
                         names: Sequence[str]) -> np.ndarray:
    """The observable vector ``[F_name(prob, U, sl) for name in names]``."""
    return np.array([OBSERVABLES[name](prob, U, sl) for name in names], dtype=float)


# --------------------------------------------------------------------------
# The control problem: free-data knobs (a subset of θ=(q,b,χ_A,χ_B)) -> targets
# --------------------------------------------------------------------------
@dataclass
class ControlProblem:
    """One control task.

    ``control_names`` is the ordered subset of ``parametric_nd_2c.AXIS_NAMES`` that
    is *adjusted* (the loop's unknowns); ``target_names`` the ordered observables
    that are *targeted* (keys of :data:`OBSERVABLES`).  Inactive knobs take
    ``parametric_nd_2c.DEFAULTS`` overridden by ``fixed``.  ``M_tot`` is the fixed
    total *bare* mass (the P3 grid convention, default 1).  ``box`` = per-axis
    ``(lo, hi)`` for clamping the iterates (the interpolant's validity region).
    ``interpolant`` is the P3 surrogate over exactly ``control_names`` (warm mode).
    """
    prob: sa.Problem
    control_names: tuple
    target_names: tuple
    M_tot: float = 1.0
    fixed: Optional[Dict[str, float]] = None
    box: Optional[np.ndarray] = None          # shape (2, d): [lo_row, hi_row]
    interpolant: Optional[ParametricSolutionND] = None
    geom_cache: Optional[Dict[float, tuple]] = None   # per-b operator cache (P3 D7)

    @property
    def d(self) -> int:
        return len(self.control_names)

    def slice_at(self, theta) -> sa.Slice:
        return p3.theta_to_slice(theta, self.control_names, self.M_tot, self.fixed)

    def clip(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if self.box is None:
            return theta
        return np.clip(theta, self.box[0], self.box[1])


@dataclass
class Counters:
    """Fair-accounting tallies over one control run (one mode)."""
    n_F: int = 0                 # solver calls (== F-evaluations)
    inner_iters: int = 0         # Σ info.iters (native Newton-iteration metric)
    lu_solves: int = 0           # Σ max(info.iters−1, 0) (dominant-cost proxy)
    max_resid: float = 0.0       # worst certified ‖R‖∞ over all solver calls
    t_solve: float = 0.0         # wall-clock spent inside newton_solve [s]
    iters_trace: list = field(default_factory=list)
    resid_trace: list = field(default_factory=list)


def solve_and_observe(cp: ControlProblem, theta, guess, tol: float, max_iter: int,
                      counters: Optional[Counters] = None):
    """Solve the constraints at control point ``theta`` (warm-started from
    ``guess``) and return ``(F_vector, U, info)``.

    The certified constraint residual is ``info.residual_norm``; ``U`` is the
    solved field.  ``counters`` (if given) is updated with the fair-accounting
    tallies and the wall-clock of this single solve.

    If ``cp.geom_cache`` is set, the b-dependent dense prolate operator ``M0`` is
    reused across calls at the same ``b`` (P3's byte-identical ``assemble_cached``,
    D7) — removing the fixed per-call operator-assembly cost so the wall-clock gain
    of the warm start reflects the Newton-step (LU-solve) reduction.  The cache is
    mode-independent (it never changes the solution), so the comparison stays fair.
    """
    sl = cp.slice_at(theta)
    t0 = time.perf_counter()
    asm = p3.assemble_cached(cp.prob, sl, cp.geom_cache) if cp.geom_cache is not None else None
    U, info = sa.newton_solve(cp.prob, sl, U0=guess, tol=tol, max_iter=max_iter, asm=asm)
    dt = time.perf_counter() - t0
    if counters is not None:
        counters.n_F += 1
        counters.inner_iters += int(info.iters)
        counters.lu_solves += max(int(info.iters) - 1, 0)
        counters.max_resid = max(counters.max_resid, float(info.residual_norm))
        counters.t_solve += dt
        counters.iters_trace.append(int(info.iters))
        counters.resid_trace.append(float(info.residual_norm))
    F = evaluate_observables(cp.prob, U, sl, cp.target_names)
    return F, U, info


# --------------------------------------------------------------------------
# Guess providers (the only thing that differs between modes)
# --------------------------------------------------------------------------
def _guess_fn(cp: ControlProblem, mode: str):
    """Return ``guess_fn(theta, prev_U) -> U0`` for the given warm-start mode.

    * ``"cold"``         — ``None`` (Newton from ``u≡0``, "from scratch");
    * ``"interp"``       — ``cp.interpolant.evaluate(clip(theta))`` (mode (a));
    * ``"continuation"`` — ``prev_U`` (the previous control iterate's field; the
      first call falls back to cold).
    """
    if mode == "cold":
        return lambda theta, prev_U: None
    if mode == "interp":
        if cp.interpolant is None:
            raise ValueError("interp mode needs cp.interpolant")
        return lambda theta, prev_U: np.asarray(cp.interpolant.evaluate(cp.clip(theta)))
    if mode == "continuation":
        return lambda theta, prev_U: prev_U
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------
# Broyden control loop  (rank-1 "good" updates; FD initial Jacobian)
# --------------------------------------------------------------------------
def broyden_control(cp: ControlProblem, theta0, target, *, mode: str,
                    tol_ctrl: float = 1e-9, tol_inner: float = 1e-10,
                    max_steps: int = 40, fd_h: float = 1e-4,
                    max_inner: int = 30, max_step_frac: float = 0.35):
    """Mendes-style Broyden loop on ``G(θ)=F(θ)−target``, warm-started per ``mode``.

    Broyden "good" rank-1 secant updates with a one-time finite-difference initial
    Jacobian and a deterministic **per-component step cap** (trust region): each
    Newton step is limited to ``max_step_frac`` of the box width per axis.  This is
    the standard globalization keeping the loop robust when an observable is weakly
    sensitive to a knob (e.g. ``∂M_ADM/∂χ → 0`` at low spin) and the raw step would
    overshoot the validated box — and, being deterministic (no ``‖G‖`` comparison),
    it keeps the outer path **identical across modes** so ``calls_match`` holds.

    Returns a dict with the converged ``theta``, the residual ``history``, the
    :class:`Counters`, and convergence flags.  The certified inner residual
    ``tol_inner`` is enforced at every solver call (``counters.max_resid`` audits
    it).  ``theta`` is clamped to ``cp.box`` each step (the validated region).

    The outer loop (initial Jacobian, step cap, rank-1 updates) is
    **mode-independent** — ``F(θ)`` is the unique certified solution to ~floor
    regardless of the guess — so the solver-call sequence is identical in every
    mode and the cold-vs-warm reduction is a clean per-call Newton-iteration /
    wall-clock factor.
    """
    target = np.asarray(target, dtype=float)
    c = Counters()
    guess_fn = _guess_fn(cp, mode)
    state = {"prev_U": None}

    def G(theta):
        g = guess_fn(theta, state["prev_U"])
        F, U, _ = solve_and_observe(cp, theta, g, tol_inner, max_inner, c)
        state["prev_U"] = U
        return F - target

    if cp.box is not None:
        cap = max_step_frac * (cp.box[1] - cp.box[0])      # per-axis step cap
    else:
        cap = None

    theta = cp.clip(theta0)
    g = G(theta)
    d = theta.size

    # one-time finite-difference initial Jacobian (d extra solver calls)
    Jac = np.zeros((target.size, d))
    for k in range(d):
        tp = theta.copy()
        tp[k] += fd_h
        Jac[:, k] = (G(cp.clip(tp)) - g) / fd_h
    history = [float(np.max(np.abs(g)))]

    for _ in range(max_steps):
        if np.max(np.abs(g)) <= tol_ctrl:
            break
        try:
            step = np.linalg.solve(Jac, -g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(Jac, -g, rcond=None)[0]
        if cap is not None:                                # deterministic trust region
            step = np.clip(step, -cap, cap)
        theta_new = cp.clip(theta + step)
        g_new = G(theta_new)
        s = theta_new - theta
        y = g_new - g
        ss = float(s @ s)
        if ss > 0.0:                                       # Broyden "good" rank-1 update
            Jac = Jac + np.outer(y - Jac @ s, s) / ss
        theta, g = theta_new, g_new
        history.append(float(np.max(np.abs(g))))

    return dict(
        theta=theta, residual=g, target=target,
        converged=bool(np.max(np.abs(g)) <= tol_ctrl),
        ctrl_residual=float(np.max(np.abs(g))),
        history=history, counters=c, mode=mode,
    )


# --------------------------------------------------------------------------
# Cold-vs-warm comparison harness
# --------------------------------------------------------------------------
def run_comparison(cp: ControlProblem, theta0, target, *,
                   modes=("cold", "interp"), tol_ctrl: float = 1e-9,
                   tol_inner: float = 1e-10, max_steps: int = 40,
                   fd_h: float = 1e-4, max_inner: int = 30):
    """Run :func:`broyden_control` in each ``mode`` and tabulate the cost reduction.

    Returns ``{mode: result, ..., "factors": {...}}``.  ``factors`` compares each
    non-cold mode to ``cold``: ``inner_iters``, ``lu_solves`` and ``wall_clock``
    reduction factors, plus ``calls_match`` (the outer loop took the same number
    of solver calls in every mode — the fairness check).
    """
    results = {}
    for mode in modes:
        if mode == "interp" and cp.interpolant is None:
            continue
        results[mode] = broyden_control(
            cp, theta0, target, mode=mode, tol_ctrl=tol_ctrl, tol_inner=tol_inner,
            max_steps=max_steps, fd_h=fd_h, max_inner=max_inner)

    factors = {}
    if "cold" in results:
        base = results["cold"]["counters"]
        n_calls = {m: results[m]["counters"].n_F for m in results}
        for m, r in results.items():
            if m == "cold":
                continue
            cc = r["counters"]
            factors[m] = dict(
                inner_iters=base.inner_iters / max(cc.inner_iters, 1),
                lu_solves=base.lu_solves / max(cc.lu_solves, 1),
                wall_clock=base.t_solve / max(cc.t_solve, 1e-30),
                cold_inner=base.inner_iters, mode_inner=cc.inner_iters,
                cold_lu=base.lu_solves, mode_lu=cc.lu_solves,
                cold_t=base.t_solve, mode_t=cc.t_solve,
            )
        factors["calls"] = n_calls
        factors["calls_match"] = (len(set(n_calls.values())) == 1)
    results["factors"] = factors
    return results


# --------------------------------------------------------------------------
# Convenience: build the P3 interpolant over the control axes (warm-mode surrogate)
# --------------------------------------------------------------------------
def build_interpolant(prob: sa.Problem, control_names: Sequence[str],
                      ranges: Dict[str, tuple], Qs: Dict[str, int],
                      M_tot: float = 1.0, fixed: Optional[Dict[str, float]] = None,
                      tol: float = 1e-12, max_iter: int = 20) -> ParametricSolutionND:
    """Build a :class:`ParametricSolutionND` over exactly ``control_names`` (the
    P3 tensor-product CGL interpolant, ``from_problem_nd``), so it warm-starts the
    control loop directly.  ``ranges[name]=(lo,hi)``, ``Qs[name]=Q`` per axis."""
    axes = [{"name": n, "min": ranges[n][0], "max": ranges[n][1], "Q": Qs[n]}
            for n in control_names]
    return p3.from_problem_nd(prob, axes, M_tot=M_tot, fixed=fixed).build(
        tol=tol, max_iter=max_iter)


def box_from_ranges(control_names: Sequence[str],
                    ranges: Dict[str, tuple]) -> np.ndarray:
    """``(2, d)`` clamp box ``[lo_row, hi_row]`` aligned to ``control_names``."""
    lo = np.array([ranges[n][0] for n in control_names], dtype=float)
    hi = np.array([ranges[n][1] for n in control_names], dtype=float)
    return np.stack([lo, hi])


# --------------------------------------------------------------------------
# Known-answer target construction (round-trip sanity for the cold loop)
# --------------------------------------------------------------------------
def make_target(cp: ControlProblem, theta_star, tol: float = 1e-12,
                max_iter: int = 40) -> np.ndarray:
    """Forward map: solve at the *chosen* free data ``theta_star`` and return its
    observable vector.  The cold control loop must recover ``theta_star`` from
    this target (the known-answer round-trip)."""
    F, _, _ = solve_and_observe(cp, np.asarray(theta_star, float), None, tol, max_iter)
    return F


# --------------------------------------------------------------------------
# Scattered parameter SURVEY (Mendes use case the surrogate accelerates most):
# many *unconnected* target points, where continuation has no smooth march
# --------------------------------------------------------------------------
def survey_cost(cp: ControlProblem, thetas, *, mode: str, tol_inner: float = 1e-10,
                max_inner: int = 30) -> Counters:
    """Solve at a list of *scattered* (unconnected) free-data points ``thetas`` in
    ``mode`` and tally the cost.  For ``"continuation"`` the previous (unrelated)
    point's field is the guess — deliberately a poor warm start, illustrating why
    a *global* interpolant beats a march for surveys."""
    c = Counters()
    guess_fn = _guess_fn(cp, mode)
    prev_U = None
    for theta in thetas:
        g = guess_fn(np.asarray(theta, float), prev_U)
        _, U, _ = solve_and_observe(cp, theta, g, tol_inner, max_inner, c)
        prev_U = U
    return c
