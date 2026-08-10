"""LM-initial-data — differentiable initial data ``∂ID/∂θ`` (Milestone B3).

The capability the whole paper-track is named for (PAPER_PLAN §1 claim 3): a
**differentiable** certified ID generator.  P3 left the hook in place — the
branchless ``ParametricSolutionND.evaluate_jax(θ)`` is a ``jnp`` interpolant, so
``jax.jacfwd``/``grad`` of it is ``∂U/∂θ`` of the surrogate.  B3 turns that hook
into two concrete deliverables:

  (a) a **gradient-based parameter solve** — hit a target ADM mass + spin by
      **Gauss–Newton on θ** using the *analytic* ``∂F/∂θ`` (``jax.jacfwd`` of the
      observable map built on ``evaluate_jax``).  This is the differentiable
      cousin of B2's Broyden loop: B2 had no gradient, so it formed a
      finite-difference initial Jacobian and made rank-1 secant updates over a
      *certified solve per outer evaluation*; B3 has the **exact** outer Jacobian
      and runs the outer loop entirely on the *free differentiable surrogate*,
      certifying only **once** at the end via ``evaluate_polished`` (‖R‖∞ ≤ 1e-10).

  (b) **sensitivity maps** ``∂ψ/∂χ_A``, ``∂ψ/∂χ_B`` as ``(ρ,z)`` fields — figure
      (viii).  ``ψ = ψ_BL + u`` with ``ψ_BL`` χ-independent *and* the ABT chart
      ``(ρ,z)↔(A,B)`` χ-independent, so ``∂ψ/∂χ = ∂u/∂χ`` **exactly**; the nodal
      ``∂U/∂χ`` from ``jacfwd(evaluate_jax)`` is ABT-interpolated to ``(ρ,z)`` with
      the numpy ``evaluate_field_phys`` (linear in the nodal field — no JAX ABT
      interp needed).  :func:`sensitivity_psi` is exact for the chart-fixed axes
      (χ, and q via the analytic ``∂ψ_BL/∂q``); the ``b`` axis moves the chart, so
      it is rejected (χ is the canonical, chart-independent figure).

Cross-checks (the B3 gate, three independent routes):
  * **autodiff correctness** — ``jacfwd`` of the surrogate vs central finite
    differences of the *same* surrogate (O(h²), FD accuracy);
  * **certified-ID faithfulness** — the surrogate's ``∂U/∂θ`` vs the *true*
    analytic tangent of the certified solve (the implicit-function-theorem
    ``J_solve · dU/dθ = −∂R/∂θ``: ``solver_abt.tangent_b``/``tangent_q`` for b,q,
    and the new :func:`tangent_chi` here for the spin axes).  This confirms the
    surrogate gradient *is* the certified-ID sensitivity, not merely FD of itself.

Add-only / standalone: imports the frozen ``solver_abt`` / ``operators_abt`` /
``source`` / ``validation.adm`` / ``parametric_nd*`` **verbatim** and defines no
new physics; :func:`tangent_chi` is a *new* function here (it does not alter any
existing signature — ``solver_abt`` ships ``tangent_b``/``tangent_q`` but no
``tangent_chi``).  numpy + jax only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..solver import solver_abt as sa
from ..solver import operators_abt as ops
from ..solver import source
from ..parametric import parametric_nd_2c as p3
from ..parametric.parametric_nd import ParametricSolutionND
from . import control as ctl   # reused verbatim: evaluate_observables (B2's ADM map)


# ==========================================================================
# 1.  The analytic spin tangent  dU/dχ_X  (the new IFT tangent; cf. tangent_b/q)
# ==========================================================================
def dA2_spin_dS(rho, z, b, S_self, S_other, which: str):
    """∂Â²/∂S_X of the aligned-spin source (closed form; cf. ``source.A2_spin_extra``).

    From ``Δ(Â²)_spin = 18 S_A² ρ²/r_A⁸ + 18 S_B² ρ²/r_B⁸
                        + 36 S_A S_B ρ²(ρ²+s_A s_B)/(r_A⁵ r_B⁵)`` (``s_X=z∓b``),

        ∂Â²/∂S_A = 36 ρ² [ S_A/r_A⁸ + S_B (ρ²+s_A s_B)/(r_A⁵ r_B⁵) ],
        ∂Â²/∂S_B = 36 ρ² [ S_B/r_B⁸ + S_A (ρ²+s_A s_B)/(r_A⁵ r_B⁵) ].

    The momentum part ``A2_2c`` is S-independent, so this is the whole derivative.
    Verified against ``jax.jacfwd`` of ``source.A2_2c_spin`` in the test suite.
    ``which`` in {"chi_A","chi_B"} selects which spin to differentiate; for the
    A↔B symmetry of the closed form, ``S_self`` is the spin being differentiated
    and ``S_other`` the partner.
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    r_A = np.sqrt(rho ** 2 + (z - b) ** 2)
    r_B = np.sqrt(rho ** 2 + (z + b) ** 2)
    sA = z - b
    sB = z + b
    rho2 = rho ** 2
    cross_geom = rho2 * (rho2 + sA * sB) / (r_A ** 5 * r_B ** 5)
    if which == "chi_A":
        self_geom = rho2 / r_A ** 8
    else:
        self_geom = rho2 / r_B ** 8
    return 36.0 * (S_self * self_geom + S_other * cross_geom)


def tangent_chi(prob: sa.Problem, U: np.ndarray, sl: sa.Slice, which: str,
                M_tot: float, asm: Optional[sa.Assembly] = None) -> np.ndarray:
    """dU/dχ_X at fixed (q, b, χ_other) by implicit differentiation, ``J dU/dχ = -dR/dχ``.

    χ enters the residual **only** through ``Â² = A2_2c_spin`` via ``S_X=χ_X m_X²``;
    ``ψ_BL`` is χ-independent.  So

        ∂R/∂χ_X |_interior = 1/8 (ψ_BL+u)^{-7} · m_X² · ∂Â²/∂S_X,

    (BC rows χ-independent).  The new sibling of ``solver_abt.tangent_b``/``tangent_q``
    (added here, not in the frozen solver, so no existing signature changes).
    ``which`` in {"chi_A","chi_B"}.
    """
    if asm is None:
        asm = sa.assemble(prob, sl)
    u = np.asarray(U, dtype=float).ravel()
    base = asm.psi + u
    finite = np.isfinite(asm.rho)
    rho_s = np.where(finite, asm.rho, 1.0)
    z_s = np.where(finite, asm.z, 0.0)
    if which == "chi_A":
        m_self, S_self, S_other = sl.m_A, sl.S_A, sl.S_B
    elif which == "chi_B":
        m_self, S_self, S_other = sl.m_B, sl.S_B, sl.S_A
    else:
        raise ValueError(f"which must be 'chi_A' or 'chi_B', got {which!r}")
    dA2_dS = dA2_spin_dS(rho_s, z_s, sl.b, S_self, S_other, which)
    dA2_dchi = np.where(finite, dA2_dS * m_self ** 2, 0.0)
    dsrc_dchi = 0.125 * base ** (-7.0) * dA2_dchi
    dR_dchi = np.where(asm.interior, dsrc_dchi, 0.0)
    J = sa.jacobian_mat(asm, u)
    dU = ops.solve_equilibrated(J, -dR_dchi)
    return dU.reshape(prob.shape)


def certified_tangent(prob: sa.Problem, U: np.ndarray, sl: sa.Slice, name: str,
                      M_tot: float, asm: Optional[sa.Assembly] = None) -> np.ndarray:
    """The certified-ID sensitivity ``dU/dθ_name`` (implicit-function tangent of the
    certified solve) for any active axis ``name`` ∈ ``parametric_nd_2c.AXIS_NAMES``.

    Dispatches to ``solver_abt.tangent_b`` / ``tangent_q`` (frozen) or
    :func:`tangent_chi` (new).  This is the ground truth the surrogate gradient is
    compared against (the "match the certified-ID sensitivity" half of the gate)."""
    if name == "b":
        return sa.tangent_b(prob, U, sl, asm=asm)
    if name == "q":
        return sa.tangent_q(prob, U, sl, M_tot, asm=asm)
    if name in ("chi_A", "chi_B"):
        return tangent_chi(prob, U, sl, name, M_tot, asm=asm)
    raise ValueError(f"no certified tangent for axis {name!r}")


# ==========================================================================
# 2.  The JAX-differentiable observable map  F_jax(θ)
# ==========================================================================
def _slice_params_jax(theta, control_names: Sequence[str], M_tot: float,
                      fixed: Optional[Dict[str, float]]):
    """Differentiable ``θ -> {q,b,m_A,m_B,S_A,S_B,M}`` (jnp twin of
    ``parametric_nd_2c.theta_to_slice``; D1/D2 conventions)."""
    vals = dict(p3.DEFAULTS)
    if fixed:
        vals.update(fixed)
    theta = jnp.asarray(theta)
    active = {name: theta[i] for i, name in enumerate(control_names)}

    def get(name):
        return active[name] if name in active else jnp.asarray(float(vals[name]))

    q, b = get("q"), get("b")
    chi_A, chi_B = get("chi_A"), get("chi_B")
    m_A = M_tot * q / (1.0 + q)
    m_B = M_tot / (1.0 + q)
    S_A = chi_A * m_A ** 2
    S_B = chi_B * m_B ** 2
    return dict(q=q, b=b, m_A=m_A, m_B=m_B, S_A=S_A, S_B=S_B, M=m_A + m_B)


def _u_at_puncture_jax(U, prob: sa.Problem, wB, which: str):
    """u at puncture X via barycentric extrapolation of the A=0 row to B=±1.

    The puncture maps to the prolate corner (A=0, B=+1) for A / (A=0, B=−1) for B
    (``operators_abt.inverse_map``); A=0 is the last A-node (inner-axis edge), so
    the 2-D interp of ``solver_abt.evaluate_field_phys`` reduces to a 1-D
    barycentric-in-B of the last row — implemented here in jnp (linear in U; B=±1
    is never a GL node, so the quotient is finite).  Matches
    ``validation.adm.puncture_adm_mass``'s field evaluation."""
    Umat = jnp.asarray(U).reshape(prob.shape)
    B = jnp.asarray(prob.B)
    Bq = 1.0 if which == "A" else -1.0
    t = jnp.asarray(wB) / (Bq - B)
    row = Umat[-1, :]                                  # A=0 (inner-axis edge) row
    return (t @ row) / jnp.sum(t)


def _make_observables_jax(prob: sa.Problem):
    """Return ``{name: f(U, sp) -> jnp scalar}`` mirroring B2's ``OBSERVABLES``
    (``validation.adm``), but as differentiable jnp functions of the field ``U``
    and the slice params ``sp`` (from :func:`_slice_params_jax`)."""
    DA1 = jnp.asarray(prob.DA1)
    wB = sa._bary_weights(prob.B)

    def M_ADM(U, sp):
        # adm.adm_mass_spectral:  M_ADM = M − 2 b ⟨∂_A u⟩_{A=1}   (linear in U)
        Umat = jnp.asarray(U).reshape(prob.shape)
        dUdA_inf = (DA1 @ Umat)[0, :]                  # A[0]=1 (infinity edge)
        c = -sp["b"] * jnp.mean(dUdA_inf)
        return sp["M"] + 2.0 * c

    def J(U, sp):
        return sp["S_A"] + sp["S_B"]                   # analytic (no field)

    def M_A(U, sp):
        u_xA = _u_at_puncture_jax(U, prob, wB, "A")
        return sp["m_A"] * (1.0 + u_xA + sp["m_B"] / (2.0 * (2.0 * sp["b"])))

    def M_B(U, sp):
        u_xB = _u_at_puncture_jax(U, prob, wB, "B")
        return sp["m_B"] * (1.0 + u_xB + sp["m_A"] / (2.0 * (2.0 * sp["b"])))

    return {"M_ADM": M_ADM, "J": J, "M_A": M_A, "M_B": M_B}


def build_F_jax(ps: ParametricSolutionND, control_names: Sequence[str],
                target_names: Sequence[str], prob: sa.Problem, M_tot: float = 1.0,
                fixed: Optional[Dict[str, float]] = None) -> Callable:
    """The differentiable observable map ``F(θ) = [obs(θ) for obs in target_names]``.

    ``θ`` are the control variables (= the interpolant ``ps``'s axes, in order);
    ``U(θ) = ps.evaluate_jax(θ)`` is the branchless surrogate field; the
    observables are :func:`_make_observables_jax` (= B2's ADM diagnostics).  The
    returned closure is pure-jnp, so ``jax.jacfwd(F_jax)`` is the **analytic**
    outer Jacobian ``∂F/∂θ`` (B3's contribution vs B2's FD/secant Jacobian)."""
    obs = _make_observables_jax(prob)
    for n in target_names:
        if n not in obs:
            raise ValueError(f"observable {n!r} not differentiable here; "
                             f"available: {sorted(obs)}")

    def F_jax(theta):
        theta = jnp.asarray(theta)
        U = ps.evaluate_jax(theta)
        sp = _slice_params_jax(theta, control_names, M_tot, fixed)
        return jnp.stack([obs[n](U, sp) for n in target_names])

    return F_jax


def jacobian_F(F_jax: Callable) -> Callable:
    """``θ -> ∂F/∂θ`` via forward-mode AD (the exact outer Jacobian)."""
    return jax.jacfwd(F_jax)


# ==========================================================================
# 3.  Gradient-based parameter solve  (Gauss–Newton on the analytic ∂F/∂θ)
# ==========================================================================
@dataclass
class GNResult:
    theta: np.ndarray            # converged free data
    residual: np.ndarray         # G(θ) = F(θ) − target at convergence
    target: np.ndarray
    converged: bool
    ctrl_residual: float         # ‖G‖∞
    history: list                # ‖G‖∞ per step
    n_F: int                     # surrogate F-evaluations (interpolations; no solve)
    n_jac: int                   # analytic Jacobian (jacfwd) evaluations
    steps: int
    certified_residual: float = np.nan   # ‖R‖∞ of the polished solution at θ
    certified_U: object = field(default=None, repr=False)


def _nudge_off_nodes(theta, ps: ParametricSolutionND, box=None,
                     trigger: float = 1e-8, shift_frac: float = 2e-3):
    """Shift any component **exactly on** a CGL node (within ``trigger``) off it.

    The branchless ``evaluate_jax`` (hence ``jacfwd``) is 0/0 *at* a node; a
    round-number start (e.g. the central node 0.3) or a box-clip onto an endpoint
    node (the box edges *are* the endpoint nodes) would poison the Jacobian.  The
    **trigger** is tiny (1e-8) — it fires only on an *exact* coincidence, never at a
    generic target merely *near* a node (``jacfwd`` is accurate down to a gap of
    ~1e-4, so a target 3e-3 from a node converges fine) — while the **shift**
    (``shift_frac·range``) is comfortable so the nudged point has an accurate
    Jacobian.  Nudging is benign: GN converges to the target regardless of the
    (off-node) start, and the shift is far below the interpolant's parameter
    resolution.  Returns the nudged θ."""
    theta = np.array(theta, dtype=float)
    for k in range(theta.size):
        nodes_k = np.asarray(ps.nodes[k])
        rng = float(nodes_k.max() - nodes_k.min())
        if float(np.min(np.abs(theta[k] - nodes_k))) < trigger:
            shift = shift_frac * (rng if rng > 0 else 1.0)
            lo, hi = (box[0][k], box[1][k]) if box is not None else (-np.inf, np.inf)
            theta[k] = theta[k] + shift if theta[k] + shift <= hi else theta[k] - shift
            theta[k] = min(max(theta[k], lo), hi)
    return theta


def gauss_newton_target(ps: ParametricSolutionND, control_names: Sequence[str],
                        target_names: Sequence[str], prob: sa.Problem,
                        target, theta0, *, M_tot: float = 1.0,
                        fixed: Optional[Dict[str, float]] = None,
                        box: Optional[np.ndarray] = None,
                        tol_ctrl: float = 1e-9, max_steps: int = 60,
                        lm_init: float = 1e-3, lm_down: float = 0.5,
                        lm_up: float = 4.0, lm_max: float = 1e10,
                        polish: bool = True, polish_steps: int = 2,
                        polish_tol: float = 1e-12) -> GNResult:
    """Hit ``F(θ)=target`` by **damped Gauss–Newton** on the analytic ``∂F/∂θ``.

    The outer loop runs on the **differentiable surrogate** (each evaluation is a
    barycentric interpolation — *no* elliptic solve) with the **exact** Jacobian
    ``jacfwd(F_jax)``.  The step is Levenberg–Marquardt (damped Gauss–Newton /
    Gauss–Newton-with-descent): ``δθ = -(JᵀJ + μ·diag(JᵀJ))⁻¹ Jᵀ G`` with the
    damping ``μ`` adapted by accept/reject (↓ on a ‖G‖-decreasing step → undamped
    Newton near the solution → quadratic; ↑ on a rejected one → a short
    gradient-descent step).  LM robustly globalizes the **ill-conditioned**
    ``(J,M_ADM)`` Jacobian (the M_ADM row is ~5× weaker than the analytic J row),
    where plain GN with a box step-cap stalls on the boundary.  ``θ`` is box-clipped
    and nudged off CGL nodes each step (the branchless ``evaluate_jax`` is 0/0 *at*
    a node, and the box edges *are* the endpoint nodes).

    Contrast with B2's Broyden loop: B2 had no gradient, so it built a
    finite-difference initial Jacobian (``d`` extra *certified solver calls*) and
    made rank-1 secant updates over a *certified elliptic solve per outer
    F-evaluation*; here ``∂F/∂θ`` is **exact** (autodiff, no FD sweep) and the outer
    F-evaluations are *interpolations* (no solve).  The certified elliptic solve is
    run **once** at the end (``ps.evaluate_polished``, 1–2 Newton steps of the
    *real* solver), so the returned ID satisfies ‖R‖∞ ≤ tol independently of the
    surrogate's interpolation error (R7).

    The **value** of ``F`` is read through the node-safe numpy interpolant
    (``ps.evaluate`` + B2's ``control.evaluate_observables`` — byte-identical to
    ``F_jax`` off-node, but with the exact-node guard so a round-number start does
    not divide by zero); the **Jacobian** is ``jacfwd(F_jax)``.  Returns
    :class:`GNResult` (``n_F`` = surrogate F-evals incl. LM trials, ``n_jac`` =
    analytic Jacobian evals = accepted steps)."""
    target = np.asarray(target, dtype=float)
    F_jax = build_F_jax(ps, control_names, target_names, prob, M_tot, fixed)
    Jf = jacobian_F(F_jax)

    def clip(th):
        th = np.asarray(th, dtype=float)
        if box is not None:
            th = np.clip(th, box[0], box[1])
        return _nudge_off_nodes(th, ps, box)

    def F_val(th):
        # node-safe surrogate observable vector (== F_jax off-node)
        U = ps.evaluate(th)
        sl = p3.theta_to_slice(th, control_names, M_tot, fixed)
        return ctl.evaluate_observables(prob, U, sl, target_names)

    theta = clip(theta0)
    n_F = n_jac = 0
    G = F_val(theta) - target
    n_F += 1
    history = [float(np.max(np.abs(G)))]
    mu = lm_init
    steps = 0
    for steps in range(1, max_steps + 1):
        if np.max(np.abs(G)) <= tol_ctrl:
            steps -= 1
            break
        Jm = np.asarray(Jf(jnp.asarray(theta)))
        n_jac += 1
        JtJ = Jm.T @ Jm
        grad = Jm.T @ G
        diagJ = np.diag(np.diag(JtJ)) + 1e-30 * np.eye(JtJ.shape[0])
        g2 = float(G @ G)
        accepted = False
        for _lm in range(40):                       # adapt damping (LM trials)
            try:
                delta = -np.linalg.solve(JtJ + mu * diagJ, grad)
            except np.linalg.LinAlgError:
                delta = -np.linalg.lstsq(Jm, G, rcond=None)[0]
            cand = clip(theta + delta)
            G_cand = F_val(cand) - target
            n_F += 1
            if float(G_cand @ G_cand) < g2:
                theta, G = cand, G_cand
                mu = max(mu * lm_down, 1e-12)        # success → less damping
                accepted = True
                break
            mu *= lm_up                              # reject → more damping
            if mu > lm_max:
                break
        history.append(float(np.max(np.abs(G))))
        if not accepted:                             # damping saturated → stall
            break

    res = GNResult(theta=theta, residual=G, target=target,
                   converged=bool(np.max(np.abs(G)) <= tol_ctrl),
                   ctrl_residual=float(np.max(np.abs(G))), history=history,
                   n_F=n_F, n_jac=n_jac, steps=steps)
    if polish:
        U, info = ps.evaluate_polished(theta, newton_steps=polish_steps, tol=polish_tol)
        res.certified_residual = float(info.residual_norm)
        res.certified_U = np.asarray(U)
    return res


# ==========================================================================
# 4.  Sensitivity fields  ∂ψ/∂θ  on a (ρ,z) grid  (figure (viii))
# ==========================================================================
def nodal_dU_dtheta(ps: ParametricSolutionND, theta) -> np.ndarray:
    """``∂U/∂θ`` of the surrogate at ``θ`` — shape ``(*field_shape, d)`` (nodal).

    ``jacfwd(evaluate_jax)``; the last axis indexes the control parameters."""
    theta = jnp.asarray(theta, dtype=float)
    return np.asarray(jax.jacfwd(ps.evaluate_jax)(theta))


def dpsi_BL_dq(rho, z, b, M_tot, q):
    """Analytic ∂ψ_BL/∂q at fixed total mass M (m_A=Mq/(1+q), m_B=M/(1+q)).

    ∂m_A/∂q = M/(1+q)², ∂m_B/∂q = −M/(1+q)²; ψ_BL=1+m_A/2r_A+m_B/2r_B ⇒
    ∂ψ_BL/∂q = (∂m_A/∂q)/(2 r_A) + (∂m_B/∂q)/(2 r_B).  (The same background piece
    ``solver_abt.tangent_q`` differentiates.)"""
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    dmA = M_tot / (1.0 + q) ** 2
    dmB = -M_tot / (1.0 + q) ** 2
    return (dmA * np.asarray(source.dpsiBL_dmA(rho, z, b))
            + dmB * np.asarray(source.dpsiBL_dmB(rho, z, b)))


def sensitivity_psi(ps: ParametricSolutionND, control_names: Sequence[str],
                    theta, which_param: str, rho, z, prob: sa.Problem,
                    M_tot: float = 1.0, fixed: Optional[Dict[str, float]] = None):
    """``∂ψ/∂(which_param)`` on physical points ``(rho, z)`` at ``θ`` (figure (viii)).

    ``ψ = ψ_BL + u``.  The nodal ``∂u/∂θ`` (from :func:`nodal_dU_dtheta`) is
    ABT-interpolated to ``(rho, z)`` with the frozen
    ``solver_abt.evaluate_field_phys`` (linear in the nodal field — no JAX ABT
    interp needed).  This is exact **only for the chart-fixed axes** — the ABT map
    ``(ρ,z)↔(A,B)`` depends on ``b`` alone:

      * ``chi_A`` / ``chi_B`` — ``ψ_BL`` is χ-independent *and* the chart is
        χ-independent ⇒ ``∂ψ/∂χ = ∂u/∂χ`` **exactly** (the canonical, clean figure);
      * ``q`` — the chart is q-independent; add the analytic ``∂ψ_BL/∂q``
        (:func:`dpsi_BL_dq`) to the interpolated ``∂u/∂q`` ⇒ exact;
      * ``b`` — the chart *moves* with ``b``, so the fixed-physical-point derivative
        carries an extra chart term ``u_A ∂A/∂b + u_B ∂B/∂b`` not captured here;
        ``b`` is therefore **rejected** (this is exactly why the prompt makes χ the
        canonical sensitivity figure).  A jnp physical-point interp would lift it.
    """
    if which_param == "b":
        raise ValueError(
            "∂ψ/∂b at a fixed physical point carries an ABT-chart-motion term "
            "(the (ρ,z)↔(A,B) map depends on b); use a χ axis (the canonical, "
            "chart-independent sensitivity figure) or q.")
    k = list(control_names).index(which_param)
    dU = nodal_dU_dtheta(ps, theta)[..., k]            # nodal ∂u/∂(param)
    sl = p3.theta_to_slice(theta, control_names, M_tot, fixed)
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    dpsi = np.asarray(sa.evaluate_field_phys(prob, dU, rho, z, sl.b))   # ∂u/∂param
    if which_param == "q":
        dpsi = dpsi + dpsi_BL_dq(rho, z, sl.b, M_tot, sl.q)             # background piece
    return dpsi
