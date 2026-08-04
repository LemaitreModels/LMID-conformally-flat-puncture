"""Acceptance — the parameter-targeting observables on the PRODUCTION χ box.

``applications/qc_targeting`` was ported from the superseded narrow box with
DIMENSIONFUL Bowen--York spins ``(S_Ay, S_By)`` to the production box with the
DIMENSIONLESS ``(χ_Ay, χ_By)``.  Under that change the angular momentum stops
being ``J = 2 b p_t + S_Ay + S_By`` and becomes

    J = 2 b p_t(b, q, χ) + χ_Ay m_A(q)² + χ_By m_B(q)²,

so the spin term acquires a ``q`` dependence — the mass→spin chain
``∂S_X/∂q = χ_X ∂(m_X²)/∂q`` of HISTORY_AND_FINDINGS §2.3, which is precisely what
an earlier certified tangent dropped (exact at χ=0, ~900% wrong by |χ|≈0.6).  The
gradient targeting method differentiates ``J`` with ``jax.jacfwd``, so a stale
hand-written spin term would corrupt the search direction and be invisible at χ=0.

Gates:
  * the jnp twin ``_pt_jax`` reproduces ``quasicircular.qc_scalar_momenta[0]`` and
    the jnp ``J`` reproduces ``J_qc`` to round-off, at χ≠0;
  * ``∂J/∂q`` from ``jax.jacfwd`` matches the central FD of the closed form — and a
    tangent that DROPS the mass→spin chain is materially wrong at χ≠0 while being
    exact at χ=0 (the §2.3 failure mode, with teeth);
  * ``∂J/∂χ_X = m_X²`` plus the spin-orbit term of ``p_t``.

Standalone (numpy/jax); reuses the committed ``quasicircular`` / ``production_box``
and ``applications.qc_targeting`` verbatim.  No elliptic solves: every gate is on
the closed-form observable, so the file runs in seconds.
"""

import numpy as np
import pytest

from lm.initial_data.parametric import quasicircular as qcmod
from lm.initial_data.applications import qc_targeting as T
from lm.initial_data.pipeline import production_box as pb

M_TOT = 1.0


def _J_jnp(theta):
    """The jnp ``J`` of ``build_F_jax`` (same closed form, model-free)."""
    b, q, cA, cB = theta[0], theta[1], theta[2], theta[3]
    m_A = M_TOT * q / (1.0 + q)
    m_B = M_TOT / (1.0 + q)
    return (2.0 * b * T._pt_jax(b, q, cA, cB, M_TOT)
            + cA * m_A ** 2 + cB * m_B ** 2)


def _draw(rng):
    return np.array([rng.uniform(pb.B_MIN, pb.B_MAX),
                     rng.uniform(pb.Q_MIN, pb.Q_MAX),
                     rng.uniform(-pb.CHI_MAX, pb.CHI_MAX),
                     rng.uniform(-pb.CHI_MAX, pb.CHI_MAX)])


# ==========================================================================
# U0 — the module is wired to the production χ box
# ==========================================================================
def test_axes_are_the_production_chi_box():
    assert T.NAMES == tuple(a["name"] for a in pb.aligned_box())
    assert T.NAMES == ("b", "q", "chi_Ay", "chi_By")
    assert dict(T.FIXED) == dict(pb.FIXED_QC)


# ==========================================================================
# U1 — p_t: the jnp twin, and χ passed through UNCONVERTED
# ==========================================================================
def test_pt_jax_matches_numpy_closed_form():
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(12):
        th = _draw(rng)
        ref = T.p_t_qc(th[0], th[1], th[2], th[3], M_TOT)
        got = float(T._pt_jax(jnp.asarray(th[0]), jnp.asarray(th[1]),
                              jnp.asarray(th[2]), jnp.asarray(th[3]), M_TOT))
        worst = max(worst, abs(got - ref) / abs(ref))
    assert worst < 1e-13, worst


def test_pt_qc_passes_chi_straight_through():
    """``p_t_qc`` feeds χ to ``qc_scalar_momenta`` UNCONVERTED (it already takes χ).

    Dividing by m² again — the pre-port behaviour, harmless-looking — rescales the
    spin-orbit term by 1/m⁴.
    """
    b, q, cA, cB = 4.0, 2.0, 0.7, -0.3
    m_A, m_B = T.masses(q, M_TOT)
    ref, _ = qcmod.qc_scalar_momenta(b, m_A, m_B, cA, cB, radial=False)
    assert abs(T.p_t_qc(b, q, cA, cB, M_TOT) - ref) <= 1e-15 * abs(ref)
    # and the conversion is not a no-op: the mis-converted value differs materially
    bad, _ = qcmod.qc_scalar_momenta(b, m_A, m_B, cA / m_A ** 2, cB / m_B ** 2,
                                     radial=False)
    assert abs(bad - ref) > 1e-6 * abs(ref)


# ==========================================================================
# U2 — J: the two implementations agree, and the spin term is the PHYSICAL spin
# ==========================================================================
def test_J_jax_matches_numpy():
    """The jnp J supplying ∂F/∂θ agrees with the numpy J that MEASURES the target.

    These sit on opposite sides of the control loop — numpy defines the target and
    the achieved residual, jnp supplies the search direction — so a mismatch
    converges the loop to the wrong configuration.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(12):
        th = _draw(rng)
        ref = T.J_qc(th, M_TOT)
        got = float(_J_jnp(jnp.asarray(th)))
        worst = max(worst, abs(got - ref) / abs(ref))
    assert worst < 1e-13, worst


def test_J_spin_term_is_physical_spin():
    """At fixed χ the spin term is χ m(q)², so it MOVES with q at fixed b, χ."""
    b, cA, cB = 5.0, 0.6, 0.6
    for q in (1.0, 3.0):
        th = np.array([b, q, cA, cB])
        m_A, m_B = T.masses(q, M_TOT)
        orbital = 2.0 * b * T.p_t_qc(b, q, cA, cB, M_TOT)
        assert abs(T.J_qc(th, M_TOT) - orbital
                   - (cA * m_A ** 2 + cB * m_B ** 2)) < 1e-14
    # equal mass m_A²+m_B² = 1/2; at q=3, 9/16+1/16 = 5/8 — a 25% larger spin term
    s1 = T.J_qc(np.array([b, 1.0, cA, cB]), M_TOT) - 2.0 * b * T.p_t_qc(b, 1.0, cA, cB)
    s3 = T.J_qc(np.array([b, 3.0, cA, cB]), M_TOT) - 2.0 * b * T.p_t_qc(b, 3.0, cA, cB)
    assert abs(s3 / s1 - (5.0 / 8.0) / (1.0 / 2.0)) < 1e-12


# ==========================================================================
# U3 — ∂J/∂q vs central FD; the §2.3 dropped-chain failure has TEETH
# ==========================================================================
def test_dJ_dq_matches_fd_and_dropped_chain_is_wrong():
    import jax
    import jax.numpy as jnp

    dJ = jax.jacfwd(_J_jnp)

    def fd_dq(th, h=1e-6):
        tp, tm = th.copy(), th.copy()
        tp[1] += h
        tm[1] -= h
        return (T.J_qc(tp, M_TOT) - T.J_qc(tm, M_TOT)) / (2.0 * h)

    def dropped_chain_dq(th, h=1e-6):
        """∂J/∂q with the spin term FROZEN at the base point (the §2.3 bug)."""
        m_A, m_B = T.masses(th[1], M_TOT)
        spin = th[2] * m_A ** 2 + th[3] * m_B ** 2

        def J_frozen(q):
            return 2.0 * th[0] * T.p_t_qc(th[0], q, th[2], th[3], M_TOT) + spin

        return (J_frozen(th[1] + h) - J_frozen(th[1] - h)) / (2.0 * h)

    b, q = 5.0, 2.0
    # (a) at χ=0 the chain term vanishes and BOTH agree — why the bug hid
    th0 = np.array([b, q, 0.0, 0.0])
    exact0 = float(dJ(jnp.asarray(th0))[1])
    assert abs(exact0 - fd_dq(th0)) <= 1e-6 * max(1.0, abs(exact0))
    assert abs(dropped_chain_dq(th0) - exact0) <= 1e-6 * max(1.0, abs(exact0))

    # (b) at production spin the analytic tangent tracks FD, the dropped-chain one does not
    worst_fd, mildest_bug = 0.0, np.inf
    for chi in (0.3, 0.6, pb.CHI_MAX):
        th = np.array([b, q, chi, chi])
        exact = float(dJ(jnp.asarray(th))[1])
        worst_fd = max(worst_fd, abs(exact - fd_dq(th)) / abs(exact))
        mildest_bug = min(mildest_bug, abs(dropped_chain_dq(th) - exact) / abs(exact))
    assert worst_fd < 1e-6, worst_fd
    assert mildest_bug > 0.1, mildest_bug          # >10% wrong even at the mildest χ


def test_dJ_dchi_is_mass_squared_plus_spin_orbit():
    """∂J/∂χ_X = m_X² + 2b ∂p_t/∂χ_X, the second term the (small) SO correction."""
    import jax
    import jax.numpy as jnp

    dJ = jax.jacfwd(_J_jnp)
    b, q = 6.0, 1.8
    th = np.array([b, q, 0.4, -0.2])
    m_A, m_B = T.masses(q, M_TOT)
    g = np.asarray(dJ(jnp.asarray(th)))

    h = 1e-7
    for k, m2 in ((2, m_A ** 2), (3, m_B ** 2)):
        tp, tm = th.copy(), th.copy()
        tp[k] += h
        tm[k] -= h
        so = 2.0 * b * (T.p_t_qc(tp[0], tp[1], tp[2], tp[3], M_TOT)
                        - T.p_t_qc(tm[0], tm[1], tm[2], tm[3], M_TOT)) / (2.0 * h)
        assert abs(g[k] - (m2 + so)) <= 1e-6 * abs(g[k])
        assert abs(so) < 0.2 * m2                  # the mass² term dominates
