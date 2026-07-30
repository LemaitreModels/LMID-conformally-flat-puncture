"""P2 acceptance (source) — the aligned-spin Bowen–York Â² of the four-tensor sum.

Verifies the crux spin derivation independently of the solver:
  * the full Â² closed form ``A2_2c_spin`` vs a raw 3×3 tensor contraction (≤1e-12);
  * S=0 reduces to the P1 ``A2_2c`` BIT-FOR-BIT;
  * the momentum–spin cross terms are IDENTICALLY ZERO for aligned spin
    (Â²_full = Â²_mom + Â²_spin, no cross) — the load-bearing parity fact;
  * transversality ∂_j Â^{ij}=0 of the full summed tensor to ≤1e-14 (autodiff);
  * sympy: the closed form equals the raw symbolic contraction EXACTLY;
  * the effective source ψ_BL^{-7}·(spin part) → 0 at each puncture.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from lemaitre.initial_data.solver import source


# --------------------------------------------------------------------------
# P2-S.1 — full Â² closed form vs raw two-tensor (4-term) contraction
# --------------------------------------------------------------------------
def test_A2_spin_vs_raw_contraction():
    """A2_2c_spin(rho,z;b,P,S_A,S_B) vs direct Â² of the summed 3×3 tensor."""
    rng = np.random.default_rng(20260626)
    worst = 0.0
    for b, P in [(1.7, 0.83), (1.0, 0.5), (3.0, 0.0)]:
        for _ in range(400):
            x = rng.normal(size=3) * 2.0
            rA = np.linalg.norm(x - np.array([0, 0, b]))
            rB = np.linalg.norm(x - np.array([0, 0, -b]))
            if min(rA, rB) < 0.1:
                continue
            S_A, S_B = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2)
            rho, z = np.hypot(x[0], x[1]), x[2]
            formula = float(source.A2_2c_spin(rho, z, b, P, S_A, S_B))
            raw = source.A2_raw_2c_spin_at_point(x, b, P, S_A, S_B)
            rel = abs(formula - raw) / abs(raw)
            worst = max(worst, rel)
    assert worst < 1e-12, f"A2_2c_spin vs raw contraction rel error {worst:.2e}"


# --------------------------------------------------------------------------
# P2-S.2 — S=0 reduces to P1 A2_2c BIT-FOR-BIT
# --------------------------------------------------------------------------
def test_A2_spin_reduces_to_P1_bit_for_bit():
    """A2_2c_spin(...,0,0) is byte-identical to the momentum-only A2_2c."""
    rng = np.random.default_rng(5)
    for b, P in [(1.3, 0.7), (2.5, 0.5), (1.0, 0.9)]:
        rho = np.abs(rng.normal(size=200)) + 0.05
        z = rng.normal(size=200) * 2.0
        base = np.asarray(source.A2_2c(rho, z, b, P))
        spin0 = np.asarray(source.A2_2c_spin(rho, z, b, P, 0.0, 0.0))
        assert np.array_equal(base, spin0), "A2_2c_spin(S=0) != A2_2c bit-for-bit"
    # the spin addition is exactly 0.0 at S=0
    extra = np.asarray(source.A2_spin_extra(rho, z, 1.0, 0.0, 0.0))
    assert np.all(extra == 0.0)


# --------------------------------------------------------------------------
# P2-S.3 — momentum–spin cross terms are IDENTICALLY ZERO (the parity fact)
# --------------------------------------------------------------------------
def test_no_momentum_spin_cross():
    """Â²_full = Â²_momentum + Â²_spin (no P–S cross) for aligned spin.

    Directly tests the load-bearing claim: the raw full contraction equals the
    raw momentum-only plus the raw spin-only contraction — i.e. the cross
    2 Â_P:Â_S contributes nothing.  (Teeth: a genuine P×S cross would make this
    fail at O(P·S/r^5).)
    """
    rng = np.random.default_rng(99)
    b, P = 1.6, 0.9
    worst = 0.0
    for _ in range(600):
        x = rng.normal(size=3) * 2.0
        rA = np.linalg.norm(x - np.array([0, 0, b]))
        rB = np.linalg.norm(x - np.array([0, 0, -b]))
        if min(rA, rB) < 0.15:
            continue
        S_A, S_B = rng.uniform(-1, 1), rng.uniform(-1, 1)
        full = source.A2_raw_2c_spin_at_point(x, b, P, S_A, S_B)
        mom = source.A2_raw_2c_spin_at_point(x, b, P, 0.0, 0.0)   # = A2_raw mom
        spin = source.A2_raw_2c_spin_at_point(x, b, 0.0, S_A, S_B)  # mom-free
        rel = abs(full - (mom + spin)) / abs(full)
        worst = max(worst, rel)
    assert worst < 1e-12, f"P–S cross is NOT zero (rel {worst:.2e})"


def test_per_puncture_PS_cross_zero():
    """At a single puncture, the raw P×S self-cross (P×S)·n = 0 (P∥S∥z)."""
    rng = np.random.default_rng(3)
    P, S = 0.8, 0.7
    A = np.array([0.0, 0.0, 1.0])
    worst = 0.0
    for _ in range(400):
        x = rng.normal(size=3) * 2.0
        if np.linalg.norm(x - A) < 0.2:
            continue
        Tp = source._A_single_tensor(x, A, [0.0, 0.0, -P])
        Ts = source._A_single_spin_tensor(x, A, [0.0, 0.0, S])
        worst = max(worst, abs(float(np.sum(Tp * Ts))))
    assert worst < 1e-12, f"per-puncture P:S contraction nonzero ({worst:.2e})"


# --------------------------------------------------------------------------
# P2-S.4 — transversality of the full summed tensor (autodiff, exact)
# --------------------------------------------------------------------------
def test_transversality_autodiff():
    """∂_j Â^{ij}=0 to ≤1e-14 for the full (momentum + spin) summed BY tensor."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for b, P in [(1.5, 0.9), (1.0, 0.0)]:
        for _ in range(60):
            x = rng.normal(size=3) * 2.0
            rA = np.linalg.norm(x - np.array([0, 0, b]))
            rB = np.linalg.norm(x - np.array([0, 0, -b]))
            if min(rA, rB) < 0.4:
                continue
            S_A, S_B = rng.uniform(-1, 1), rng.uniform(-1, 1)
            div = source.divergence_2c_spin_autodiff(x, b, P, S_A, S_B)
            # normalize by a local Â scale ~ |S|/r^4 + P/r^3 (the steeper, spin)
            scale = 3.0 * (max(abs(S_A), abs(S_B)) / min(rA, rB) ** 4
                           + P / min(rA, rB) ** 3)
            worst = max(worst, np.max(np.abs(div)) / scale)
    assert worst < 1e-14, f"BY sum (mom+spin) not transverse, rel {worst:.2e}"


# --------------------------------------------------------------------------
# P2-S.5 — sympy: the closed form equals the raw symbolic contraction EXACTLY
# --------------------------------------------------------------------------
def test_sympy_exact_spin_closed_form():
    """sympy simplify( raw Â²_spin − closed-form Δ(Â²)_spin ) = 0 (exact), AND
    the PRODUCTION ``source.A2_spin_extra`` equals the raw symbolic contraction
    numerically (so the symbolic proof pins the shipped code, not just an inline
    copy of the formula)."""
    import sympy as sp
    rho, z, b, SA, SB = sp.symbols("rho z b S_A S_B", real=True)

    def Aspin_sym(x0z, S):
        d = sp.Matrix([rho, 0, z - x0z])
        r = sp.sqrt(d.dot(d))
        n = d / r
        v = sp.Matrix([0, 0, S]).cross(n)
        return (3 / r ** 3) * (v * n.T + n * v.T)

    T = Aspin_sym(b, SA) + Aspin_sym(-b, SB)
    A2_raw = sum(T[i, j] ** 2 for i in range(3) for j in range(3))
    rA = sp.sqrt(rho ** 2 + (z - b) ** 2)
    rB = sp.sqrt(rho ** 2 + (z + b) ** 2)
    closed = (18 * SA ** 2 * rho ** 2 / rA ** 8
              + 18 * SB ** 2 * rho ** 2 / rB ** 8
              + 36 * SA * SB * rho ** 2 * (rho ** 2 + (z - b) * (z + b))
              / (rA ** 5 * rB ** 5))
    assert sp.simplify(A2_raw - closed) == 0
    # pin the PRODUCTION code to the (independent) raw symbolic contraction
    f_raw = sp.lambdify((rho, z, b, SA, SB), A2_raw, "numpy")
    rng = np.random.default_rng(123)
    for _ in range(50):
        rr, zz = float(np.abs(rng.normal()) + 0.1), float(rng.normal() * 2)
        bb, sa_, sb_ = 1.3, 0.6, -0.4
        prod = float(source.A2_spin_extra(rr, zz, bb, sa_, sb_))
        ref = float(f_raw(rr, zz, bb, sa_, sb_))
        assert abs(prod - ref) <= 1e-12 * (abs(ref) + 1e-30), \
            f"A2_spin_extra != raw symbolic ({prod} vs {ref})"


# --------------------------------------------------------------------------
# P2-S.6 — per-puncture regularity of the spin source (ψ_BL^{-7} kills it)
# --------------------------------------------------------------------------
def test_spin_source_vanishes_at_each_puncture():
    """ψ^{-7}·(self-spin) ~ r^1 and ψ^{-7}·(spin-cross) ~ r^5 at A and at B.

    Â_S² ~ r^{-6} ⇒ ×ψ_BL^{-7}(~r^7) → r^1; the two-centre spin–spin cross is
    EXTRA-suppressed by the azimuthal alignment factor (γ−μ_Aμ_B)=ρ²/(r_A r_B),
    so it falls as r^{-2} near a puncture ⇒ ×ψ^{-7} → r^5 (the same exponent as
    the momentum cross).  Both → 0 ⇒ the effective source is regular and
    spectral convergence survives.

    (The r^1 / r^5 scalings are the RADIAL limit r→0 along a fixed ray; the
    explicit ρ² in Δ(Â²)_spin contributes the angular sin²θ factor — at fixed
    off-axis ρ the source does not vanish, only at each puncture POINT.)
    """
    b, P, mA, mB = 1.5, 0.7, 0.6, 0.4
    S_A, S_B = 0.5, 0.4
    radii = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    for which, mu in (("A", 0.3), ("B", -0.2)):
        zc = b if which == "A" else -b
        s_self, s_cross = [], []
        for r in radii:
            z = zc + r * mu
            rho = r * np.sqrt(1.0 - mu ** 2)
            psi = float(source.psi_BL_2c(rho, z, b, mA, mB))
            mom, selfA, selfB, cross = source.A2_2c_spin_parts(rho, z, b, P, S_A, S_B)
            part_self = float(selfA if which == "A" else selfB)
            s_self.append(psi ** (-7.0) * part_self)
            s_cross.append(psi ** (-7.0) * abs(float(cross)))
        s_self, s_cross = np.array(s_self), np.array(s_cross)
        slope_self = (np.log(s_self[-1]) - np.log(s_self[-2])) / \
                     (np.log(radii[-1]) - np.log(radii[-2]))
        slope_cross = (np.log(s_cross[-1]) - np.log(s_cross[-2])) / \
                      (np.log(radii[-1]) - np.log(radii[-2]))
        assert s_self[-1] < s_self[0], f"{which}: spin self not decreasing"
        assert s_cross[-1] < s_cross[0], f"{which}: spin cross not decreasing"
        assert abs(slope_self - 1.0) < 0.1, \
            f"{which}: spin self slope {slope_self:.3f} (expected 1)"
        assert abs(slope_cross - 5.0) < 0.2, \
            f"{which}: spin cross slope {slope_cross:.3f} (expected 5)"


def test_spin_source_vanishes_on_axis():
    """The aligned-spin source ∝ ρ² ⇒ it vanishes on the z-axis (ρ=0)."""
    z = np.linspace(-2.0, 2.0, 11)
    extra = np.asarray(source.A2_spin_extra(0.0 * z, z, 1.3, 0.6, 0.4))
    assert np.max(np.abs(extra)) < 1e-300 or np.allclose(extra, 0.0, atol=1e-30)
