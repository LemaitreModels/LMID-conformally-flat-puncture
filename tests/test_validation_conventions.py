"""B1 Step 1 — the PARASOL <-> TwoPunctures convention map (risk R2).

Pure data-transform checks: the map is the load-bearing object whose error would
masquerade as a physics disagreement, so it is unit-tested in isolation.
"""

import numpy as np

from lm.initial_data.validation import conventions as cv


def test_parasol_to_tp_basic_map():
    tp = cv.parasol_to_tp(b=3.0, m_A=0.5, m_B=0.5, P=0.5)
    # par_b is the half-separation (= PARASOL b); +b puncture carries m_A
    assert tp.par_b == 3.0
    assert tp.par_m_plus == 0.5 and tp.par_m_minus == 0.5
    # head-on infall, x-axis image of (0,0,-/+P): momenta point toward the origin
    assert tp.par_P_plus == (-0.5, 0.0, 0.0)
    assert tp.par_P_minus == (0.5, 0.0, 0.0)
    assert tp.par_S_plus == (0.0, 0.0, 0.0) and tp.par_S_minus == (0.0, 0.0, 0.0)
    assert tp.give_bare_mass is True


def test_unequal_mass_assignment():
    tp = cv.parasol_to_tp(b=2.0, m_A=0.7, m_B=0.3, P=0.1)
    # the +b puncture (PARASOL A) must receive m_A, never m_B
    assert tp.par_m_plus == 0.7 and tp.par_m_minus == 0.3
    assert tp.par_b == 2.0


def test_momentum_sign_antiparallel():
    """Head-on: the two punctures carry exactly opposite momenta (total = 0)."""
    tp = cv.parasol_to_tp(b=4.0, m_A=0.5, m_B=0.5, P=0.37)
    Pp = np.array(tp.par_P_plus)
    Pm = np.array(tp.par_P_minus)
    assert np.allclose(Pp + Pm, 0.0)
    assert np.isclose(np.linalg.norm(Pp), 0.37)


def test_point_map_roundtrip():
    """PARASOL (rho, z) -> TP Cartesian -> (rho, z) is the identity."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        rho = abs(rng.uniform(0, 5))
        z = rng.uniform(-5, 5)
        x_tp, y_tp, z_tp = cv.parasol_point_to_tp(rho, z)
        # axial coord z -> x_TP ; cylindrical radius rho -> sqrt(y^2+z^2)_TP
        assert np.isclose(x_tp, z)
        assert np.isclose(np.hypot(y_tp, z_tp), rho)
        rho2, z2 = cv.tp_point_axial_radius(x_tp, y_tp, z_tp)
        assert np.isclose(rho2, rho) and np.isclose(z2, z)


def test_cactus_par_render():
    tp = cv.parasol_to_tp(b=3.0, m_A=0.6, m_B=0.4, P=0.2)
    s = tp.to_cactus_par()
    assert "par_b = 3.0" in s
    assert "par_m_plus = 0.6" in s and "par_m_minus = 0.4" in s
    assert "par_P_plus[0] = -0.2" in s and "par_P_minus[0] = 0.2" in s
    assert "give_bare_mass = yes" in s
