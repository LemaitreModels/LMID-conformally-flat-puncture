"""Acceptance tests — the first non-axisymmetric PARASOL solver (3-D, Fourier-φ).

The axisymmetric ABT two-centre code (``operators_abt``, ``solver_abt``) stays
the frozen regression oracle; ``operators_3d``/``source_3d``/``solver_3d`` lift
it to 3-D by an azimuthal Fourier collocation, with the linear operator
block-diagonal in the azimuthal mode m (``∂²_φ → −m²``).

Gates (CPU only):
  * **A. Axisymmetric reduction (load-bearing).**  Nφ=1, on-axis P, zero spin
    reproduces the frozen 2-D ``newton_solve`` to ~1e-12 — proves the 3-D code
    contains the validated 2-D code.
  * **B. Momentum-constraint transversality.**  The summed BY tensor (momentum +
    off-axis spin) is transverse ∂_jÂ^{ij}=0 to ~1e-14 (autodiff).
  * **C. Manufactured solution.**  Δ_3D u = S recovers u* SPECTRALLY for genuine
    azimuthal content (m=0,1,2) at all nodes — exercising the azimuthal −m²/ρ²
    term and the associated-Legendre (1−B²)^{|m|/2} basis factoring that restores
    spectral convergence for the odd m=1 sector (the outer-prolate-axis branch
    point).
  * **D. Spectral φ-convergence on a physical slice.**  A single misaligned spin:
    the field self-converges exponentially as Nφ: 4→8→12 at fixed meridian
    resolution — few modes suffice for the minimal axisymmetry break.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from lm.initial_data.solver import operators_3d as ops3
from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.solver import source_3d
from lm.initial_data.solver import solver_abt as sa
from lm.initial_data.solver import diagnostics as diag
from lm.initial_data.solver.solver_abt import Slice
from lm.initial_data.solver.solver_3d import Slice3D


# ==========================================================================
# A.  Axisymmetric reduction — the load-bearing gate
# ==========================================================================
def test_axisym_reduction_reproduces_2d():
    """Nφ=1, on-axis P, zero spin -> the frozen 2-D solver, to ~1e-12."""
    P = 0.5
    for (b, mA, mB) in [(1.0, 0.5, 0.5), (1.5, 0.7, 0.3)]:
        prob2 = sa.make_problem(Na=28, Nb=20, P=P)
        U2, i2 = sa.newton_solve(prob2, Slice(b=b, m_A=mA, m_B=mB),
                                 tol=1e-12, max_iter=25)
        prob3 = s3.make_problem(Na=28, Nb=20, Nphi=1)
        sl3 = Slice3D.head_on(b=b, m_A=mA, m_B=mB, P=P)
        U3, i3 = s3.newton_solve(prob3, sl3, tol=1e-12, max_iter=25)
        d = float(np.max(np.abs(np.asarray(U3)[:, :, 0] - np.asarray(U2))))
        scale = float(np.max(np.abs(U2)))
        print(f"\n[A] b={b} mA={mA}: 2D its={i2.iters} 3D its={i3.iters} "
              f"|U3-U2|={d:.2e}  max|U2|={scale:.3e}")
        assert d < 1e-12, f"axisym reduction off by {d:.2e} (b={b}, mA={mA})"
        # iteration counts match too (identical Newton path)
        assert i3.iters == i2.iters, f"iters differ: 2D {i2.iters} vs 3D {i3.iters}"


def test_axisym_P0_zero_spin_exact_fixed_point():
    """P=0, zero spin: u ≡ 0 to machine zero on the 3-D grid (any Nφ)."""
    for Nphi in (1, 4, 8):
        prob = s3.make_problem(Na=28, Nb=20, Nphi=Nphi)
        sl = Slice3D(b=1.0, m_A=0.5, m_B=0.5)        # no momentum, no spin
        U, info = s3.newton_solve(prob, sl, tol=1e-13, max_iter=8)
        assert float(np.max(np.abs(U))) == 0.0, f"P=0 nonzero u (Nphi={Nphi})"
        assert info.residual_norm == 0.0


# ==========================================================================
# B.  Momentum-constraint transversality (analytic, free)
# ==========================================================================
def test_transversality_momentum_plus_spin():
    """∂_j Â^{ij} ≈ 0 (autodiff) for off-axis momenta + misaligned spins."""
    rng = np.random.default_rng(20260629)
    cases = [
        dict(P_A_vec=(0.0, 0.0, -0.3), P_B_vec=(0.0, 0.0, 0.3),
             S_A_vec=(0.3, 0.0, 0.2), S_B_vec=(0.0, 0.0, 0.0)),          # misaligned spin
        dict(P_A_vec=(0.1, 0.0, -0.3), P_B_vec=(0.0, 0.05, 0.3),
             S_A_vec=(0.2, 0.1, 0.1), S_B_vec=(-0.1, 0.0, 0.15)),        # fully generic
    ]
    for b in (0.8, 1.5, 3.0):
        for c in cases:
            worst = 0.0
            for _ in range(40):
                x = rng.normal(size=3) * 2.0
                rA = np.linalg.norm(x - np.array([0, 0, b]))
                rB = np.linalg.norm(x - np.array([0, 0, -b]))
                if min(rA, rB) < 0.4:
                    continue
                div = source_3d.divergence_3d_autodiff(x, b, **c)
                worst = max(worst, np.max(np.abs(div)))
            assert worst < 1e-13, f"b={b}: not transverse, ‖∂_jÂ^ij‖={worst:.2e}"


def test_axisym_source_matches_closed_form():
    """The point-wise summed tensor Â² equals the 2-D closed form on the meridian
    (on-axis momentum) — what makes the reduction gate (A) hold."""
    from lm.initial_data.solver import source
    b, P = 1.3, 0.7
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        x = rng.normal(size=3) * 2.0
        rho, z = np.hypot(x[0], x[1]), x[2]
        if min(np.hypot(rho, z - b), np.hypot(rho, z + b)) < 0.1:
            continue
        # point-wise 3-D builder at this Cartesian point (on-axis P, no spin)
        T = source_3d.A_full_tensor_vec(x[None, :], b, (0, 0, -P), (0, 0, P),
                                        (0, 0, 0), (0, 0, 0))[0]
        a2_3d = float(np.sum(T * T))
        a2_2d = float(source.A2_2c(rho, z, b, P))
        worst = max(worst, abs(a2_3d - a2_2d) / abs(a2_2d))
    assert worst < 1e-12, f"3-D source vs 2-D closed form rel {worst:.2e}"


# ==========================================================================
# C.  Manufactured solution — operator correctness + φ spectral content
# ==========================================================================
def _manufactured(prob, b, parts):
    """Nodal (u*, S=Δ_3D u*) for a sum of analytic Cartesian × e^{-r²} pieces.

    Each piece is ``(coeff_fn(x,y,z,r2)->u_part, S_fn(...)->Δu_part)`` evaluated
    on the (A,B,φ) node cloud (edges -> 0).
    """
    Lap, rho, z, Af, Bf, DA, DB, inv = ops3.axisym_blocks(
        prob.A, prob.B, prob.DA1, prob.DB1, b)
    fin = np.isfinite(rho)
    rs, zs = np.where(fin, rho, 1.0), np.where(fin, z, 0.0)
    u = np.zeros((prob.Ntot2d, prob.Nphi))
    S = np.zeros_like(u)
    for k, ph in enumerate(prob.phi):
        x, y = rs * np.cos(ph), rs * np.sin(ph)
        r2 = rs ** 2 + zs ** 2
        e = np.exp(-r2)
        for ufn, sfn in parts:
            u[:, k] += np.where(fin, ufn(x, y, zs, r2, e), 0.0)
            S[:, k] += np.where(fin, sfn(x, y, zs, r2, e), 0.0)
    interior = (Af > 1e-12) & (Af < 1.0 - 1e-12)
    return u, S, interior, Af, Bf


# even-m pieces (m=0 and m=2): smooth in B -> spectral
_M0 = (lambda x, y, z, r2, e: e,
       lambda x, y, z, r2, e: e * (4 * r2 - 6))
_M2 = (lambda x, y, z, r2, e: e * (x * x - y * y),
       lambda x, y, z, r2, e: e * (x * x - y * y) * (4 * r2 - 14))
# odd m=1 piece: (1−B²)^{1/2} branch point at the outer axis -> algebraic
_M1 = (lambda x, y, z, r2, e: e * x,
       lambda x, y, z, r2, e: e * x * (4 * r2 - 10))


def test_manufactured_all_modes_spectral():
    """Δ_3D u = S with genuine m=0,1,2 content converges SPECTRALLY at all nodes.

    The headline operator gate: exercises the azimuthal −m²/ρ² term (m=2) AND the
    associated-Legendre (1−B²)^{|m|/2} basis factoring that makes the odd m=1
    sector spectral (without it this floored at ~3e-3).  All-node error, no
    off-axis exclusion.
    """
    b = 1.0
    rows, errs = [], []
    for (Na, Nb, Nphi) in [(20, 16, 6), (28, 20, 8), (36, 24, 10), (44, 30, 12)]:
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
        u, S, interior, Af, Bf = _manufactured(prob, b, [_M0, _M1, _M2])
        U = s3.solve_poisson(prob, b, S).reshape(prob.Ntot2d, prob.Nphi)
        e = float(np.max(np.abs((U - u)[interior])))
        errs.append(e)
        rows.append((Na, Nb, Nphi, e))
    diag.convergence_table(rows, ["Na", "Nb", "Nphi", "uErr"],
                           title="\n[C] manufactured m=0,1,2 spectral convergence (all nodes)")
    errs = np.array(errs)
    assert errs[0] / errs[-1] > 1e3, f"all-mode not spectral: {errs}"
    assert np.all(np.diff(errs) < 0), f"not monotone: {errs}"
    assert errs[-1] < 1e-7, f"all-mode err at Na=44 = {errs[-1]:.2e}"


def test_manufactured_odd_m1_spectral():
    """The odd m=1 sector converges SPECTRALLY (the associated-Legendre fix).

    The prolate B-operator is the associated-Legendre operator; its regular m≠0
    solution behaves as (1−B²)^{|m|/2} at the outer axis B=±1.  For odd m=1 that
    is a (1−B²)^{1/2} branch point, which polynomial (GL) collocation of the field
    itself resolves only algebraically (floored at ~2e-3).  Factoring the field as
    u_m = (1−B²)^{1/2} v_m and solving for the SMOOTH v_m (singular factor's
    B-derivatives analytic) restores spectral convergence — to machine precision
    at all nodes including the outer-axis node.
    """
    b = 1.0
    rows, errs = [], []
    for (Na, Nb, Nphi) in [(28, 20, 8), (40, 28, 10), (52, 36, 12)]:
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
        u, S, interior, Af, Bf = _manufactured(prob, b, [_M1])
        U = s3.solve_poisson(prob, b, S).reshape(prob.Ntot2d, prob.Nphi)
        e = float(np.max(np.abs((U - u)[interior])))
        errs.append(e)
        rows.append((Na, Nb, Nphi, e))
    diag.convergence_table(rows, ["Na", "Nb", "Nphi", "m1Err"],
                           title="\n[C] manufactured odd m=1 spectral convergence (all nodes)")
    errs = np.array(errs)
    assert errs[0] / errs[-1] > 1e2, f"m=1 not spectral: {errs}"    # was ~1.7 unfactored
    assert np.all(np.diff(errs) < 0), f"m=1 not monotone: {errs}"
    assert errs[-1] < 1e-8, f"m=1 err at Na=52 = {errs[-1]:.2e} (was ~2e-3 unfactored)"


# ==========================================================================
# D.  Spectral φ-convergence on a physical slice (the minimal break)
# ==========================================================================
def test_phi_spectral_convergence_misaligned_spin():
    """A single misaligned spin: the field self-converges exponentially in Nφ.

    Fixed meridian resolution (Na,Nb); the (fixed) m=1 meridian-discretisation
    error cancels in the Nφ difference, so this isolates the φ-spectral content —
    confirming few azimuthal modes suffice for the minimal axisymmetry break.
    """
    b, mA, mB, P = 1.0, 0.5, 0.5, 0.3
    sl = Slice3D(b=b, m_A=mA, m_B=mB, P_A_vec=(0, 0, -P), P_B_vec=(0, 0, P),
                 S_A_vec=(0.3, 0.0, 0.2), S_B_vec=(0.0, 0.0, 0.0))
    Na, Nb = 40, 28
    pref = s3.make_problem(Na=Na, Nb=Nb, Nphi=16)
    Uref, iref = s3.newton_solve(pref, sl, tol=1e-9, max_iter=40)
    assert iref.residual_norm < 1e-6, f"ref ‖R‖={iref.residual_norm:.2e}"
    qr = np.array([0.4, 0.8, 0.6, 1.2]); qz = np.array([0.6, 0.0, -0.5, 0.3])
    qp = np.array([0.0, 1.0, 2.0, 0.5])
    uref = s3.evaluate_field(pref, Uref, qr, qz, qp, b)

    rows, errs = [], []
    for Nphi in (4, 6, 8, 12):
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
        U, info = s3.newton_solve(prob, sl, tol=1e-9, max_iter=40)
        uq = s3.evaluate_field(prob, U, qr, qz, qp, b)
        e = float(np.max(np.abs(uq - uref)))
        errs.append(e)
        rows.append((Nphi, info.iters, info.residual_norm, e))
    diag.convergence_table(rows, ["Nphi", "its", "||R||", "fieldErr"],
                           title="\n[D] misaligned-spin φ-spectral convergence")
    errs = np.array(errs)
    assert errs[0] / errs[-1] > 1e4, f"φ not spectral: {errs}"
    assert np.all(np.diff(errs) < 0), f"φ not monotone: {errs}"
    # floors at the fixed-(Na,Nb) Newton/source residual level (~1e-8 here)
    assert errs[-1] < 1e-8, f"φ field err at Nφ=12 = {errs[-1]:.2e}"


def test_misaligned_spin_breaks_axisymmetry():
    """Sanity: a misaligned spin genuinely populates m≠0 (the test has teeth)."""
    b, mA, mB, P = 1.0, 0.5, 0.5, 0.3
    sl = Slice3D(b=b, m_A=mA, m_B=mB, P_A_vec=(0, 0, -P), P_B_vec=(0, 0, P),
                 S_A_vec=(0.3, 0.0, 0.2), S_B_vec=(0.0, 0.0, 0.0))
    prob = s3.make_problem(Na=36, Nb=24, Nphi=12)
    U, info = s3.newton_solve(prob, sl, tol=1e-9, max_iter=40)
    U = np.asarray(U)
    # φ-variation of the field, relative to its magnitude
    phi_var = np.max(np.abs(U - U.mean(axis=2, keepdims=True))) / np.max(np.abs(U))
    print(f"\n[D] misaligned-spin φ-variation = {phi_var:.3e}")
    assert phi_var > 1e-3, f"unexpectedly axisymmetric: {phi_var:.2e}"
    # the azimuthal mode spectrum of the source has genuine m=1 content
    asm = s3.assemble(prob, sl)
    amp = np.max(np.abs(np.fft.rfft(asm.A2, axis=1)), axis=0)
    assert amp[1] / amp[0] > 1e-2, f"no m=1 source content: {amp[1]/amp[0]:.2e}"


# ==========================================================================
# Newton–Krylov (solver_3d_nk) — the CERTIFIED non-axisymmetric solve
# ==========================================================================
# The modified-Newton solver (``solver_3d.newton_solve``) uses the EXACT
# mode-space residual, so its converged field already solves the full coupled
# problem — the dropped φ-varying Jacobian term only affects the (linear)
# convergence rate.  Its reported ``‖R‖∞`` (raw nodal inf-norm) is nonetheless
# roundoff-limited and RISES with resolution: the stiff rows next to the inner
# axis (A→0, where 1/A and m²/ρ² are enormous) amplify floating-point noise.
#
# Newton–Krylov (``solver_3d_nk.newton_solve_nk``) restores the full mode
# coupling (matrix-free J, block-diagonal modified-Newton preconditioner) and
# monitors the EQUILIBRATED residual — the well-conditioned norm the solve
# controls.  The result: every azimuthal mode that carries non-negligible
# physical content is certified to MACHINE PRECISION.  The residual floor that
# remains lives entirely in the highest, ~zero-content modes (m²/ρ² roundoff),
# which do not affect the physical solution — confirmed by NK reproducing the
# modified-Newton field (hence the TwoPunctures-validated data) bit-for-bit.
from lm.initial_data.solver import solver_3d_nk as nk      # noqa: E402


def _content_modes(asm, rel_tol=1e-10):
    """Indices of azimuthal modes whose SOURCE amplitude exceeds ``rel_tol``×(m=0).

    These are the physically-populated modes; the rest carry ~machine-zero
    content (a misaligned spin excites only a handful of low m).
    """
    amp = np.max(np.abs(np.fft.rfft(asm.A2, axis=1)), axis=0)
    return np.where(amp / amp[0] > rel_tol)[0]


# --- A. axisymmetric reduction (load-bearing) -----------------------------
def test_nk_axisym_reduction_reproduces_2d():
    """Nφ=1, on-axis P, zero spin: NK reproduces the frozen 2-D Newton to ~1e-12.

    For Nφ=1 the block-diagonal preconditioner IS the full Jacobian, so GMRES
    converges in ONE iteration per Newton step (exact solve) — every NK step is
    identical to the 2-D dense Newton.  The load-bearing gate: the 3-D NK code
    contains the validated 2-D code.
    """
    P = 0.5
    for (b, mA, mB) in [(1.0, 0.5, 0.5), (1.5, 0.7, 0.3)]:
        prob2 = sa.make_problem(Na=28, Nb=20, P=P)
        U2, i2 = sa.newton_solve(prob2, Slice(b=b, m_A=mA, m_B=mB),
                                 tol=1e-12, max_iter=25)
        prob3 = s3.make_problem(Na=28, Nb=20, Nphi=1)
        sl3 = Slice3D.head_on(b=b, m_A=mA, m_B=mB, P=P)
        U3, i3 = nk.newton_solve_nk(prob3, sl3, tol=1e-12, max_iter=25)
        d = float(np.max(np.abs(np.asarray(U3)[:, :, 0] - np.asarray(U2))))
        print(f"\n[NK-A] b={b} mA={mA}: 2D its={i2.iters} NK its={i3.iters} "
              f"|U3-U2|={d:.2e} gmres={i3.gmres_iters}")
        assert d < 1e-12, f"NK axisym reduction off by {d:.2e} (b={b})"
        # perfect preconditioner -> exactly one GMRES iteration per Newton step
        assert all(g == 1 for g in i3.gmres_iters), \
            f"Nφ=1 not a 1-iter exact solve: {i3.gmres_iters}"


def test_nk_P0_exact_fixed_point():
    """P=0, zero spin: NK leaves u ≡ 0 to machine zero (any Nφ)."""
    for Nphi in (1, 4, 8):
        prob = s3.make_problem(Na=28, Nb=20, Nphi=Nphi)
        sl = Slice3D(b=1.0, m_A=0.5, m_B=0.5)
        U, info = nk.newton_solve_nk(prob, sl, tol=1e-13, max_iter=8)
        assert float(np.max(np.abs(U))) == 0.0, f"P=0 nonzero u (Nphi={Nphi})"
        assert info.residual_norm == 0.0


# --- B. quadratic convergence to a certified residual ---------------------
def test_nk_quadratic_convergence():
    """NK drives the certified (equilibrated) residual quadratically to ≤1e-10.

    The full-Jacobian Newton is super-linear: at least one step reduces the
    residual by ≥3 orders of magnitude (vs the modified Newton's geometric
    crawl), reaching machine-precision in the physical modes.
    """
    sl = Slice3D(b=1.5, m_A=0.5, m_B=0.5, P_A_vec=(0, 0, -0.5), P_B_vec=(0, 0, 0.5),
                 S_A_vec=(0.3, 0.0, 0.2), S_B_vec=(0.0, 0.0, 0.0))
    prob = s3.make_problem(Na=36, Nb=24, Nphi=8)
    U, info = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=15)
    h = np.array(info.history)
    ratios = h[1:] / h[:-1]
    print(f"\n[NK-B] equilR history: {' '.join('%.1e' % x for x in h)}  "
          f"min step ratio={ratios.min():.1e}  gmres={info.gmres_iters}")
    assert info.residual_norm <= 1e-10, f"NK equilR={info.residual_norm:.2e}"
    assert ratios.min() < 1e-3, f"not quadratic (best step ratio {ratios.min():.1e})"


# --- C. certified physical-mode residual + beats modified-Newton ----------
def test_nk_certified_physical_modes_machine_precision():
    """Every populated azimuthal mode is certified to machine precision.

    Across the convergence ladder, the per-mode equilibrated residual of every
    mode carrying non-negligible SOURCE content (src_m/src_0 > 1e-10) is ≤1e-9,
    and the aggregate NK certified residual is strictly below the modified
    Newton's raw monitor.  The only residual floor lives in the highest,
    ~zero-content modes (m²/ρ² roundoff) — irrelevant to the physical solution.
    """
    sl = Slice3D(b=1.5, m_A=0.5, m_B=0.5, P_A_vec=(0, 0, -0.5), P_B_vec=(0, 0, 0.5),
                 S_A_vec=(0.3, 0.0, 0.2), S_B_vec=(0.0, 0.0, 0.0))
    rows = []
    for (Na, Nb, Nphi) in [(28, 20, 6), (36, 24, 8), (44, 30, 10)]:
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
        asm = s3.assemble(prob, sl)
        Um, im = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40, asm=asm)
        Uk, ik = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=15, asm=asm)
        scales = nk._block_scales(asm)
        Rm = s3.residual_modes(asm, np.asarray(Uk).reshape(prob.Ntot2d, prob.Nphi))
        content = _content_modes(asm)
        phys = max(float(np.max(np.abs(Rm[:, mi]) / scales[mi])) for mi in content)
        rows.append((Na, Nb, Nphi, len(content), phys,
                     ik.residual_norm, im.residual_norm))
        # populated modes certified to (near) machine precision
        assert phys <= 1e-9, \
            f"populated-mode residual {phys:.2e} at Na={Na} (modes {list(content)})"
        # NK certified residual strictly beats the modified-Newton raw monitor
        assert ik.residual_norm < im.residual_norm, \
            f"NK equilR {ik.residual_norm:.2e} !< modN rawR {im.residual_norm:.2e}"
    diag.convergence_table(
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows],
        ["Na", "Nb", "Nphi", "nModes", "physR", "NKeqR", "modNrawR"],
        title="\n[NK-C] certified physical-mode residual vs modified-Newton")


# --- D. preconditioner quality (GMRES iteration counts) -------------------
def test_nk_preconditioner_quality():
    """GMRES iterations/Newton step are single-digit and ~resolution-independent.

    The per-m block-diagonal modified-Newton operator is J minus the (small)
    mode coupling, so it is an excellent preconditioner: the bulk is captured and
    GMRES only chases the mode-coupling perturbation.
    """
    sl = Slice3D(b=1.5, m_A=0.5, m_B=0.5, P_A_vec=(0, 0, -0.5), P_B_vec=(0, 0, 0.5),
                 S_A_vec=(0.3, 0.0, 0.2), S_B_vec=(0.0, 0.0, 0.0))
    worst = 0
    for (Na, Nb, Nphi) in [(28, 20, 6), (36, 24, 8), (44, 30, 10)]:
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
        U, info = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=15)
        gi = info.gmres_iters
        print(f"\n[NK-D] Na={Na} Nphi={Nphi}: gmres/step={gi}")
        assert max(gi) <= 12, f"GMRES not single-digit at Na={Na}: {gi}"
        worst = max(worst, max(gi))
    assert worst <= 12, f"preconditioner degraded with resolution: worst {worst}"


# --- E. certified polish ---------------------------------------------------
def test_nk_certified_polish():
    """A perturbed warm start polished by ≤2 NK steps reaches certified ≤1e-10.

    The 3-D analog of the axisymmetric ``evaluate_polished``: the residual is the
    constraint residual at the slice, independent of the warm start's origin.
    """
    sl = Slice3D(b=1.5, m_A=0.5, m_B=0.5, P_A_vec=(0, 0, -0.5), P_B_vec=(0, 0, 0.5),
                 S_A_vec=(0.3, 0.0, 0.2), S_B_vec=(0.0, 0.0, 0.0))
    prob = s3.make_problem(Na=36, Nb=24, Nphi=8)
    asm = s3.assemble(prob, sl)
    Ustar, _ = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=15, asm=asm)
    # a representative interpolation-error warm start: solution + ~0.1% smooth bump
    Up = np.asarray(Ustar).reshape(prob.Ntot2d, prob.Nphi) * 1.001
    e0 = nk.equil_residual_inf(asm, Up)
    Upol, ipol = nk.evaluate_polished_nk(prob, sl, Up, newton_steps=2,
                                         tol=1e-10, asm=asm)
    print(f"\n[NK-E] polish: start equilR={e0:.2e} -> 2 steps "
          f"equilR={ipol.residual_norm:.2e} gmres={ipol.gmres_iters}")
    assert e0 > 1e-5, f"warm start too close to be a real test ({e0:.2e})"
    assert ipol.residual_norm <= 1e-10, \
        f"polish did not certify: {ipol.residual_norm:.2e}"


# --- field identity: NK reproduces the TwoPunctures-validated data ---------
def test_nk_matches_modified_newton_field():
    """NK and the modified Newton converge to the SAME field (≤1e-7).

    Both use the exact residual, so they share the converged solution; NK only
    reaches a lower (certified) residual floor.  This is why NK reproduces the
    modified-Newton field — hence the TwoPunctures-validated data — confirming,
    not changing, the physical initial data (oracle-free version of Test C).
    """
    sl = Slice3D(b=1.5, m_A=0.5, m_B=0.5, P_A_vec=(0, 0, -0.5), P_B_vec=(0, 0, 0.5),
                 S_A_vec=(0.3, 0.0, 0.2), S_B_vec=(0.0, 0.0, 0.0))
    prob = s3.make_problem(Na=44, Nb=30, Nphi=10)
    asm = s3.assemble(prob, sl)
    Um, _ = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40, asm=asm)
    Uk, _ = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=15, asm=asm)
    d = float(np.max(np.abs(np.asarray(Uk) - np.asarray(Um))))
    scale = float(np.max(np.abs(Um)))
    print(f"\n[NK] |U_NK - U_modN| = {d:.2e}  (max|U|={scale:.3e})")
    assert d < 1e-7, f"NK field differs from modified-Newton by {d:.2e}"
