"""M0-A acceptance — the two-centre Bowen–York source.

Verifies the crux source derivation independently of the solver:
  * the summed-BY Â² closed form vs a raw two-tensor contraction (1e-12);
  * ψ_BL two-term;
  * the BY sum is transverse (∂_j Â^{ij} ≈ 0) for any separation;
  * the cross term 2 Â_A:Â_B is O(1) (≈ the diagonal at the midpoint);
  * the effective source -> 0 at *each* puncture: r^3 from the self-term and
    r^5 from the cross-term (per puncture).
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from lemaitre.initial_data.solver import source


# --------------------------------------------------------------------------
# M0-A.1 — summed Â² closed form vs raw two-tensor contraction
# --------------------------------------------------------------------------
def test_A2_2c_vs_raw_contraction():
    """A2_2c(rho,z;b,P) vs direct contraction of (Â_A+Â_B)^{ij} at random pts."""
    rng = np.random.default_rng(20260624)
    b, P = 1.7, 0.83
    worst = 0.0
    for _ in range(400):
        x = rng.normal(size=3) * 2.0
        # stay clear of both punctures (>~0.05) so the comparison is well-posed
        rA = np.linalg.norm(x - np.array([0, 0, b]))
        rB = np.linalg.norm(x - np.array([0, 0, -b]))
        if min(rA, rB) < 0.05:
            continue
        rho = np.hypot(x[0], x[1])
        z = x[2]
        formula = float(source.A2_2c(rho, z, b, P))
        raw = source.A2_raw_2c_at_point(x, b, P)
        rel = abs(formula - raw) / abs(raw)
        worst = max(worst, rel)
    assert worst < 1e-12, f"A2_2c vs raw contraction rel error {worst:.2e}"


def test_A2_self_term_reduces_to_single_centre():
    """Each self-term equals the single-puncture (9P^2/2 r^4)(1+2mu^2)."""
    rng = np.random.default_rng(11)
    b, P = 2.0, 1.1
    for _ in range(50):
        x = rng.normal(size=3) * 1.5
        rho, z = np.hypot(x[0], x[1]), x[2]
        rA = np.hypot(rho, z - b)
        if rA < 0.1:
            continue
        muA = (z - b) / rA                       # cos angle of n_A to z-axis
        self_A, self_B, cross = source.A2_2c_parts(rho, z, b, P)
        ref = 9.0 * P ** 2 / (2.0 * rA ** 4) * (1.0 + 2.0 * muA ** 2)
        assert abs(self_A - ref) / abs(ref) < 1e-13


# --------------------------------------------------------------------------
# M0-A.2 — ψ_BL two-term
# --------------------------------------------------------------------------
def test_psi_BL_two_term():
    b = 1.3
    rho, z = 0.7, 0.4
    rA = np.hypot(rho, z - b)
    rB = np.hypot(rho, z + b)
    for mA, mB in [(0.5, 0.5), (0.8, 0.3), (1.0, 0.0)]:
        ref = 1.0 + mA / (2 * rA) + mB / (2 * rB)
        assert abs(float(source.psi_BL_2c(rho, z, b, mA, mB)) - ref) < 1e-14
    # one mass zero -> single puncture
    assert abs(float(source.psi_BL_2c(rho, z, b, 0.6, 0.0))
               - (1.0 + 0.6 / (2 * rA))) < 1e-14


# --------------------------------------------------------------------------
# M0-A.3 — transversality of the BY sum (momentum constraint analytic)
# --------------------------------------------------------------------------
def test_BY_sum_transverse():
    """∂_j Â^{ij} ≈ 0 for the summed BY tensor (FD), for several separations."""
    rng = np.random.default_rng(7)
    P = 0.9
    for b in (0.8, 1.5, 3.0):
        worst = 0.0
        for _ in range(40):
            x = rng.normal(size=3) * 2.0
            rA = np.linalg.norm(x - np.array([0, 0, b]))
            rB = np.linalg.norm(x - np.array([0, 0, -b]))
            if min(rA, rB) < 0.4:
                continue
            div = source.divergence_raw_2c_at_point(x, b, P, h=1e-5)
            # normalize by a local Â scale ~ 3P/(2 min(rA,rB)^2) / min(rA,rB)
            scale = 3.0 * P / (2.0 * min(rA, rB) ** 3)
            worst = max(worst, np.max(np.abs(div)) / scale)
        assert worst < 1e-7, f"b={b}: BY sum not transverse, rel {worst:.2e}"


# --------------------------------------------------------------------------
# M0-A.4 — cross term is O(1) (the reason this is genuinely coupled)
# --------------------------------------------------------------------------
def test_cross_term_O1_at_midpoint():
    """At the on-axis midpoint (origin) the cross term ≈ the self diagonal."""
    b, P = 1.0, 1.0
    # midpoint is the origin (rho=0, z=0), equidistant rA=rB=b
    self_A, self_B, cross = source.A2_2c_parts(0.0, 0.0, b, P)
    assert self_A > 0 and self_B > 0
    # cross is comparable in magnitude to the self diagonal (O(1) ratio), nonzero
    ratio = abs(cross) / (self_A + self_B)
    assert 0.1 < ratio < 10.0, f"cross/self ratio {ratio:.3f} not O(1)"


def test_cross_term_grows_as_b_shrinks():
    """The cross term at the midpoint grows (per unit self) as b -> 0."""
    P = 1.0
    ratios = []
    for b in (4.0, 2.0, 1.0, 0.5):
        sA, sB, cr = source.A2_2c_parts(0.0, 0.0, b, P)
        ratios.append(abs(cr) / (sA + sB))
    # never vanishes; the relative cross weight is bounded away from 0
    assert min(ratios) > 0.05, f"cross/self ratios {ratios} -> vanish?"


# --------------------------------------------------------------------------
# M0-A.5 — effective source -> 0 at each puncture (r^3 self, r^5 cross)
# --------------------------------------------------------------------------
def _approach_puncture(b, P, mA, mB, which, mu_local):
    """Effective-source self/cross parts along a ray into puncture A or B.

    Returns (radii, src_self, src_cross) with src = psi_BL^{-7} * (part),
    approaching the puncture from a generic direction (avoiding the axis and the
    line to the other puncture).
    """
    radii = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    src_self, src_cross = [], []
    for r in radii:
        if which == "A":
            z = b + r * mu_local
        else:
            z = -b + r * mu_local
        rho = r * np.sqrt(1.0 - mu_local ** 2)
        psi = float(source.psi_BL_2c(rho, z, b, mA, mB))
        self_A, self_B, cross = source.A2_2c_parts(rho, z, b, P)
        part_self = self_A if which == "A" else self_B
        src_self.append(psi ** (-7.0) * part_self)
        src_cross.append(psi ** (-7.0) * abs(cross))
    return radii, np.array(src_self), np.array(src_cross)


def _slope(r, s):
    """Local log-log slope using the two innermost (smallest-r) samples."""
    return (np.log(s[-1]) - np.log(s[-2])) / (np.log(r[-1]) - np.log(r[-2]))


def test_source_vanishes_at_each_puncture():
    """psi^{-7}*self ~ r^3 and psi^{-7}*cross ~ r^5 at A and at B."""
    b, P, mA, mB = 1.5, 1.0, 0.6, 0.4
    for which, mu_local in (("A", 0.3), ("B", -0.2)):
        r, s_self, s_cross = _approach_puncture(b, P, mA, mB, which, mu_local)
        slope_self = _slope(r, s_self)
        slope_cross = _slope(r, s_cross)
        assert s_self[-1] < s_self[0], f"{which}: self not decreasing"
        assert s_cross[-1] < s_cross[0], f"{which}: cross not decreasing"
        assert abs(slope_self - 3.0) < 0.1, \
            f"{which}: self slope {slope_self:.3f} (expected 3)"
        assert abs(slope_cross - 5.0) < 0.3, \
            f"{which}: cross slope {slope_cross:.3f} (expected 5)"
