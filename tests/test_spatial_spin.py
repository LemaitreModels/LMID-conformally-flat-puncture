"""P2 acceptance (solver) — aligned-spin head-on Newton solve on the ABT grid.

Gates:
  * S=0 reproduces the P1 no-spin solve BIT-FOR-BIT (assembly + solution);
  * P=0,S≠0 and P≠0,S≠0 both Newton-converge to the residual floor;
  * spatial spectral convergence is preserved with spin;
  * PARITY (R1): equal-mass + equal-magnitude spin stays z-even (modes at
    roundoff); a SINGLE spinning puncture (|S_A|≠|S_B|) genuinely populates
    odd modes (z-asymmetry O(1%+)); anti-aligned EQUAL-magnitude spin is z-even.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from lemaitre.initial_data.solver import solver_abt as sa
from lemaitre.initial_data.solver import diagnostics as diag
from lemaitre.initial_data.solver.solver_abt import Slice


# --------------------------------------------------------------------------
# P2-X.0 — S=0 reproduces P1 bit-for-bit (assembly + solved field)
# --------------------------------------------------------------------------
def test_S0_reduces_to_P1_bit_for_bit():
    prob = sa.make_problem(Na=32, Nb=22, P=0.5)
    sl_nospin = Slice(b=1.0, m_A=0.5, m_B=0.5)                 # P1-style ctor
    sl_zero = Slice(b=1.0, m_A=0.5, m_B=0.5, S_A=0.0, S_B=0.0)  # explicit zero spin
    # the assembled source A2 is byte-identical
    asm0 = sa.assemble(prob, sl_nospin)
    asmz = sa.assemble(prob, sl_zero)
    assert np.array_equal(asm0.A2, asmz.A2), "A2 differs at S=0"
    # and the solved field is byte-identical
    U0, _ = sa.newton_solve(prob, sl_nospin, tol=1e-10, max_iter=20)
    Uz, _ = sa.newton_solve(prob, sl_zero, tol=1e-10, max_iter=20)
    assert np.array_equal(np.asarray(U0), np.asarray(Uz)), "U differs at S=0"


# --------------------------------------------------------------------------
# P2-X.1 — Newton converges:  P=0,S≠0  and  P≠0,S≠0
# --------------------------------------------------------------------------
def test_newton_converges_spin_only():
    """P=0 but S≠0: the spin source alone drives a nontrivial converged u.

    Uses the ABT solver's ``tol=1e-9`` convention (the steeper spin source,
    Â_S²~r^{-6} vs the momentum r^{-4}, carries more high-frequency content so
    the nonlinear-source residual floor sits a little above the single-centre
    1e-10 — exactly the documented "floor grows with source content").
    """
    prob = sa.make_problem(Na=36, Nb=24, P=0.0)
    sl = Slice(b=1.0, m_A=0.5, m_B=0.5, S_A=0.4, S_B=0.4)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=20)
    assert info.converged, f"spin-only Newton ‖R‖={info.residual_norm:.2e}"
    assert info.residual_norm <= 1e-9
    # genuine quadratic step (the source ∝ S² drives a nontrivial correction)
    drops = [info.history[i] / info.history[i + 1] for i in range(len(info.history) - 1)]
    assert max(drops) > 1e2, f"no quadratic step (max drop {max(drops):.1e})"
    assert float(np.max(np.abs(U))) > 1e-3, "spin source gave trivial u"


def test_newton_converges_momentum_and_spin():
    """P≠0 and S≠0 (the full aligned head-on+spin slice) converges to the floor."""
    prob = sa.make_problem(Na=36, Nb=24, P=0.5)
    sl = Slice(b=1.0, m_A=0.5, m_B=0.5, S_A=0.3, S_B=0.3)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=20)
    assert info.converged, f"mom+spin Newton ‖R‖={info.residual_norm:.2e}"
    assert info.residual_norm <= 1e-9
    assert info.iters <= 12
    drops = [info.history[i] / info.history[i + 1] for i in range(len(info.history) - 1)]
    assert max(drops) > 1e2, f"no quadratic step (max drop {max(drops):.1e})"
    assert float(np.max(np.abs(U))) > 1e-3


# --------------------------------------------------------------------------
# P2-X.2 — spatial spectral convergence preserved with spin (slow)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_spin_spatial_spectral_convergence():
    """Equal-mass + equal-spin (z-even) head-on+spin field self-converges
    EXPONENTIALLY — spin does not spoil spectral convergence.

    The steeper spin source (Â_S²~r^{-6} ⇒ effective source ~r^1 at the
    punctures, vs the momentum r^3) carries more near-puncture structure, so the
    floor is reached at a slightly higher resolution than momentum-only
    (PAPER_PLAN R7); it crosses 1e-9 by Na≈56 and the certified Newton polish
    (B-thread) recovers ≤1e-10 regardless.
    """
    b, m_A, m_B, P, S = 1.0, 0.5, 0.5, 0.5, 0.3
    slc = Slice(b=b, m_A=m_A, m_B=m_B, S_A=S, S_B=S)
    pref = sa.make_problem(Na=72, Nb=48, P=P)
    Uref, iref = sa.newton_solve(pref, slc, tol=1e-9, max_iter=30)
    assert iref.residual_norm < 1e-7, f"reference ‖R‖={iref.residual_norm:.2e}"
    qpts = [(0.3, 0.7), (0.6, 0.0), (0.3, -0.7), (1.2, 0.4), (3.0, 0.0)]
    rq = np.array([p[0] for p in qpts]); zq = np.array([p[1] for p in qpts])
    uref = sa.evaluate_field_phys(pref, Uref, rq, zq, b)

    rows, errs = [], []
    for (Na, Nb) in [(24, 16), (32, 22), (40, 28), (56, 38)]:
        prob = sa.make_problem(Na=Na, Nb=Nb, P=P)
        U, info = sa.newton_solve(prob, slc, tol=1e-9, max_iter=30)
        uq = sa.evaluate_field_phys(prob, U, rq, zq, b)
        e = float(np.max(np.abs(uq - uref)))
        errs.append(e)
        rows.append((Na, Nb, info.iters, info.residual_norm, e))
    diag.convergence_table(rows, ["Na", "Nb", "its", "||R||", "fieldErr"],
                           title="\n[P2] two-centre spin spatial spectral convergence")
    errs = np.array(errs)
    assert errs[0] / errs[-1] > 1e3, f"not exponential: {errs}"
    assert np.all(np.diff(errs) < 0), f"not monotone: {errs}"
    assert errs[-1] < 2e-9, f"spin field err at Na=56 = {errs[-1]:.2e}"


# --------------------------------------------------------------------------
# P2-X.3 — PARITY (R1): the load-bearing equatorial-evenness analysis
# --------------------------------------------------------------------------
def _z_asymmetry(U):
    """max|U − U[:, ::-1]| / max|U|; GL B-nodes are symmetric ⇒ U[:,::-1] is the
    z-reflected field, so this measures the z→−z asymmetry."""
    U = np.asarray(U)
    return float(np.max(np.abs(U - U[:, ::-1])) / np.max(np.abs(U)))


def test_equal_spin_is_z_even():
    """Equal mass + EQUAL-magnitude aligned spin ⇒ z-even (modes at roundoff)."""
    prob = sa.make_problem(Na=40, Nb=28, P=0.5)
    sl = Slice(b=1.0, m_A=0.5, m_B=0.5, S_A=0.4, S_B=0.4)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=25)
    assert info.residual_norm < 1e-8
    asym = _z_asymmetry(U)
    print(f"\n[P2] equal-spin z-asymmetry = {asym:.3e}")
    assert asym < 1e-9, f"equal-spin not z-even: {asym:.2e}"


def test_antialigned_equal_spin_is_z_even():
    """Anti-aligned but EQUAL-magnitude spin (S_A=+S, S_B=−S) is STILL z-even:
    Â² depends on spin only through S_A², S_B², S_A S_B, all even under the
    reflection+swap when |S_A|=|S_B| (the subtle parity result)."""
    prob = sa.make_problem(Na=40, Nb=28, P=0.5)
    sl = Slice(b=1.0, m_A=0.5, m_B=0.5, S_A=0.4, S_B=-0.4)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=25)
    assert info.residual_norm < 1e-8
    asym = _z_asymmetry(U)
    print(f"\n[P2] anti-aligned equal-|spin| z-asymmetry = {asym:.3e}")
    assert asym < 1e-9, f"anti-aligned equal-|spin| not z-even: {asym:.2e}"


def test_single_spinning_puncture_breaks_parity():
    """A SINGLE spinning puncture (|S_A|≠|S_B|) genuinely populates odd modes:
    the z-asymmetry is O(1%)+, NOT roundoff — the teeth for the parity claim.

    This is the case the prompt's R1 targets: the self-spin term
    18 S_A² ρ²/r_A^8 (localized at +b) has no −b mirror, so the source is not
    z-even and the full Gauss–Legendre angular basis must (and does) carry the
    odd modes.  Equal-mass momentum is z-even, so the asymmetry here is purely
    the spin's doing.
    """
    prob = sa.make_problem(Na=40, Nb=28, P=0.5)
    sl = Slice(b=1.0, m_A=0.5, m_B=0.5, S_A=0.4, S_B=0.0)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=25)
    assert info.residual_norm < 1e-8
    asym = _z_asymmetry(U)
    print(f"\n[P2] single-spin (S_A=0.4,S_B=0) z-asymmetry = {asym:.3e}")
    assert asym > 1e-2, f"single-spin unexpectedly z-even: {asym:.2e} (odd modes not populated?)"


def test_odd_modes_genuinely_populated():
    """Project the single-spin solution onto odd Legendre modes in B and confirm
    the odd content is >> roundoff (not numerical noise)."""
    from lemaitre.initial_data.solver import spectral
    prob = sa.make_problem(Na=40, Nb=28, P=0.5)
    sl = Slice(b=1.0, m_A=0.5, m_B=0.5, S_A=0.4, S_B=0.0)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=25)
    assert info.residual_norm < 1e-8
    U = np.asarray(U)
    B = np.asarray(prob.B)
    # GL weights for the B nodes (modal projection u_ell = (2ell+1)/2 ∫ u P_ell)
    _, w = np.polynomial.legendre.leggauss(B.size)
    # use the A-row of largest |u| (most resolved structure)
    irow = int(np.argmax(np.max(np.abs(U), axis=1)))
    urow = U[irow, :]
    odd, even = 0.0, 0.0
    for ell in range(0, 10):
        Pell = spectral.legendre_P_eval(ell, B)
        coef = (2 * ell + 1) / 2.0 * np.sum(w * urow * Pell)
        if ell % 2 == 1:
            odd = max(odd, abs(coef))
        else:
            even = max(even, abs(coef))
    print(f"\n[P2] single-spin modal: max|odd-ℓ|={odd:.3e}, max|even-ℓ|={even:.3e}")
    assert odd > 1e-6 * even, f"odd modes are at roundoff: odd/even={odd/even:.2e}"
    assert odd > 1e-9, f"odd modal amplitude {odd:.2e} is roundoff-level"
