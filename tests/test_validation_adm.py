"""B1 Step 3 — ADM / quasi-local diagnostics, re-derived and unit-tested.

Anchors that do NOT need the elliptic solve (pure source / convention checks)
plus solved-field observables validated against the frozen ``solver_abt``
reference values and internal consistency.
"""

import numpy as np
import pytest

from lm.initial_data.solver import solver_abt as sa, source
from lm.initial_data.validation import adm


B, MA, MB, P = 3.0, 0.5, 0.5, 0.5


# --------------------------------------------------------------------------
# Bowen–York tensor + the K_ij <-> Â_ij conformal-weight relation
# --------------------------------------------------------------------------
def test_BY_tensor_matches_source_and_contraction():
    rng = np.random.default_rng(0)
    rho = np.abs(rng.uniform(0.2, 2.5, 6))
    z = rng.uniform(-2.5, 2.5, 6)
    Avec = adm.A_tensor_2c(rho, z, B, P)                 # (6,3,3)
    for k in range(rho.size):
        A2_vec = float(np.sum(Avec[k] * Avec[k]))
        A2_raw = source.A2_raw_2c_at_point([rho[k], 0.0, z[k]], B, P)
        A2_cls = float(source.A2_2c(rho[k], z[k], B, P))
        assert abs(A2_vec - A2_raw) < 1e-10 * max(1.0, abs(A2_raw))
        assert abs(A2_vec - A2_cls) < 1e-10 * max(1.0, abs(A2_cls))


def test_K_from_A_traceless_and_contraction():
    """K_ij = psi^{-2} Â_ij is trace-free w.r.t. gamma=psi^4 delta, and
    K_ij K^ij = psi^{-12} Â² (the conformal-weight relation, re-derived)."""
    rng = np.random.default_rng(2)
    rho = np.abs(rng.uniform(0.3, 2.0, 5))
    z = rng.uniform(-2.0, 2.0, 5)
    psi = 1.0 + np.abs(rng.uniform(0.1, 1.5, 5))
    Avec = adm.A_tensor_2c(rho, z, B, P)
    Klow = adm.physical_K_lower(psi, Avec)
    ginv = np.zeros_like(Klow)
    ginv[:, 0, 0] = ginv[:, 1, 1] = ginv[:, 2, 2] = psi ** (-4.0)
    tr = np.einsum("kab,kab->k", ginv, Klow)              # g^{ab} K_ab = K
    assert np.max(np.abs(tr)) < 1e-12
    Kup = np.einsum("kia,kjb,kab->kij", ginv, ginv, Klow)
    KK = np.einsum("kab,kab->k", Klow, Kup)
    A2 = np.einsum("kab,kab->k", Avec, Avec)
    assert np.max(np.abs(KK - adm.KK_physical(psi, A2))) < 1e-10


# --------------------------------------------------------------------------
# ADM linear momentum: per-puncture Gauss law (= ±P) and total (= 0)
# --------------------------------------------------------------------------
def test_by_momentum_gauss_recovers_P():
    PA = adm.by_momentum_gauss(B, P, "A")
    PB = adm.by_momentum_gauss(B, P, "B")
    assert abs(PA - (-P)) < 1e-9       # A at +b carries (0,0,-P)
    assert abs(PB - (+P)) < 1e-9


def test_total_linear_momentum_zero():
    assert abs(adm.adm_linear_momentum_total(B, P)) < 1e-6


def test_gauss_momentum_radius_independent():
    """The Gauss integral is independent of the enclosing-sphere radius."""
    vals = [adm.by_momentum_gauss(B, P, "A", radius=r) for r in (0.3, 0.6, 0.9)]
    assert np.max(np.abs(np.diff(vals))) < 1e-9


# --------------------------------------------------------------------------
# ADM mass + individual puncture masses on the solved binary
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def solved():
    prob = sa.make_problem(Na=44, Nb=30, P=P)
    sl = sa.Slice(B, MA, MB)
    U, info = sa.newton_solve(prob, sl, tol=1e-11, max_iter=25)
    assert info.residual_norm < 1e-9
    return prob, U, sl


def test_adm_mass_matches_reference(solved):
    prob, U, sl = solved
    M_spec = adm.adm_mass_spectral(prob, U, sl)
    # frozen solver_abt reference: M_ADM(b=3, equal mass, P=0.5) = 1.291;
    # the high-precision TwoPunctures value is 1.290742336687.
    assert abs(M_spec - 1.290742336687) < 5e-7, M_spec


def test_adm_mass_extractors_agree(solved):
    """Three independent extractions (spectral-boundary, monopole tail, surface
    integral) agree; the spectral one is the high-precision primary."""
    prob, U, sl = solved
    M_spec = adm.adm_mass_spectral(prob, U, sl)
    M_mono = adm.adm_mass_monopole(prob, U, sl)
    M_surf = adm.adm_mass_surface(prob, U, sl)
    assert abs(M_spec - M_mono) < 1e-3, (M_spec, M_mono)
    assert abs(M_spec - M_surf) < 1e-3, (M_spec, M_surf)


def test_adm_mass_spectral_converges():
    """The spectral ADM-mass extraction converges with LM-initial-data resolution to the
    high-precision TwoPunctures value (1.290742336687)."""
    Eref = 1.290742336687
    sl = sa.Slice(B, MA, MB)
    errs = []
    for (Na, Nb) in [(44, 30), (52, 36), (64, 44)]:
        prob = sa.make_problem(Na=Na, Nb=Nb, P=P)
        U, _ = sa.newton_solve(prob, sl, tol=1e-12, max_iter=30)
        errs.append(abs(adm.adm_mass_spectral(prob, U, sl) - Eref))
    assert errs[-1] < 1e-9, errs           # ~1e-11 at 64x44
    assert errs[-1] < errs[0]              # spectral decrease


def test_individual_puncture_masses_equal(solved):
    prob, U, sl = solved
    MA = adm.puncture_adm_mass(prob, U, sl, "A")
    MB = adm.puncture_adm_mass(prob, U, sl, "B")
    assert abs(MA - MB) < 1e-9            # equal mass -> equal individual masses
    assert 0.5 < MA < 0.65               # m_A=0.5 rescaled up by binding
    # TwoPunctures reference at this config: mp_adm = 0.56377...
    assert abs(MA - 0.56377) < 5e-3, MA
