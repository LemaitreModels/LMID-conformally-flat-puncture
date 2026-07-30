"""Fast wiring gates for the quasi-circular (QC) branch of
``parametric_nd_3d.theta_to_slice3d`` (Phase P1) — no solver, pure mapping.

Guards two things: (a) EVERY existing head-on / single-spin / spin8 family is
byte-for-byte unchanged when QC is off (the QC branch only fires on the opt-in
``fixed={"qc": 1.0}`` flag), and (b) with QC on the slice's momenta equal
``quasicircular.qc_momenta(...)`` built from the same spins, giving orbital
angular momentum L along +y (J = (0, 2b·p_t + S_Ay + S_By, 0)).
"""
import numpy as np

from lemaitre.initial_data.parametric import parametric_nd_3d as p3
from lemaitre.initial_data.parametric import quasicircular as qc
from lemaitre.initial_data.parametric import parametric_nd_smolyak as sm
from lemaitre.initial_data.solver import diagnostics_3d as d3


# ==========================================================================
# (a) QC OFF ⇒ every existing family is byte-for-byte unchanged
# ==========================================================================
def test_head_on_families_unchanged_without_qc():
    # planar polar single-spin
    sl = p3.theta_to_slice3d([2.3, 0.3, 40.0, 1.7], ["b", "S_mag", "theta_S", "q"])
    th = np.deg2rad(40.0)
    assert sl.S_A_vec == (0.3 * np.sin(th), 0.0, 0.3 * np.cos(th))
    assert sl.S_B_vec == (0.0, 0.0, 0.0)
    assert sl.P_A_vec == (0.0, 0.0, -0.5) and sl.P_B_vec == (0.0, 0.0, 0.5)
    # planar Cartesian single-spin
    sl2 = p3.theta_to_slice3d([0.2, 0.1], ["S_x", "S_z"], fixed={"b": 2.0})
    assert sl2.S_A_vec == (0.2, 0.0, 0.1) and sl2.S_B_vec == (0.0, 0.0, 0.0)
    assert sl2.P_A_vec == (0.0, 0.0, -0.5) and sl2.P_B_vec == (0.0, 0.0, 0.5)
    # full 8-D spin family (head-on momenta)
    names = ["b", "q", "S_Ax", "S_Ay", "S_Az", "S_Bx", "S_By", "S_Bz"]
    sl3 = p3.theta_to_slice3d([2.0, 1.5, 0.3, 0.25, 0.2, -0.2, 0.3, -0.15], names)
    assert sl3.P_A_vec == (0.0, 0.0, -0.5) and sl3.P_B_vec == (0.0, 0.0, 0.5)
    # off-axis momentum P_x still head-on-style anti-symmetric
    sl4 = p3.theta_to_slice3d([0.3], ["P_x"], fixed={"b": 2.0})
    assert sl4.P_A_vec == (0.3, 0.0, -0.5) and sl4.P_B_vec == (-0.3, 0.0, 0.5)


# ==========================================================================
# (b) QC ON ⇒ momenta = qc_momenta from the same spins, L along +y
# ==========================================================================
def test_qc_nonspinning_matches_qc_momenta():
    b, q, M = 3.7, 1.0, 1.0
    m_A, m_B = M * q / (1 + q), M / (1 + q)
    sl = p3.theta_to_slice3d([b, q], ["b", "q"], fixed={"qc": 1.0})
    P_A, P_B = qc.qc_momenta(b, m_A, m_B, (0, 0, 0), (0, 0, 0))
    assert np.allclose(sl.P_A_vec, P_A) and np.allclose(sl.P_B_vec, P_B)
    assert sl.S_A_vec == (0.0, 0.0, 0.0) and sl.S_B_vec == (0.0, 0.0, 0.0)
    # tangential along x, radial along z, zero net linear momentum, L along +y
    assert sl.P_A_vec[1] == 0.0 and np.allclose(np.array(P_A) + np.array(P_B), 0.0)
    J = d3.adm_J_closed_form(b, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec)
    p_t, _ = qc.qc_scalar_momenta(b, m_A, m_B)
    assert abs(J[0]) < 1e-14 and abs(J[2]) < 1e-14
    assert abs(J[1] - 2.0 * b * p_t) < 1e-13 and J[1] > 0.0


def test_qc_unequal_mass_matches_qc_momenta():
    b, q, M = 4.2, 2.5, 1.0
    m_A, m_B = M * q / (1 + q), M / (1 + q)
    sl = p3.theta_to_slice3d([b, q], ["b", "q"], fixed={"qc": 1.0})
    P_A, P_B = qc.qc_momenta(b, m_A, m_B)
    assert np.allclose(sl.P_A_vec, P_A) and np.allclose(sl.P_B_vec, P_B)
    assert abs(sl.m_A - m_A) < 1e-14 and abs(sl.m_B - m_B) < 1e-14


def test_qc_aligned_spin_feeds_so_and_stays_along_y():
    # full-spin QC: aligned spin = S_y; it feeds the SO correction to p_t and adds
    # S_Ay+S_By to J_y, keeping J along y.
    names = ["b", "q", "S_Ay", "S_By"]
    b, q = 6.0, 1.0
    m_A, m_B = 0.5, 0.5
    S_Ay, S_By = 0.3, 0.1
    sl = p3.theta_to_slice3d([b, q, S_Ay, S_By], names, fixed={"qc": 1.0})
    S_A_vec = (0.0, S_Ay, 0.0)
    S_B_vec = (0.0, S_By, 0.0)
    P_A, P_B = qc.qc_momenta(b, m_A, m_B, S_A_vec, S_B_vec)
    assert np.allclose(sl.P_A_vec, P_A) and np.allclose(sl.P_B_vec, P_B)
    assert sl.S_A_vec == S_A_vec and sl.S_B_vec == S_B_vec
    # SO correction actually moved p_t away from the non-spinning value
    p0 = qc.pt_nonspinning(b, m_A, m_B)
    assert abs(sl.P_A_vec[0] - p0) > 1e-6
    # J along +y, = 2b·p_t + S_Ay + S_By
    p_t, _ = qc.qc_scalar_momenta(b, m_A, m_B, S_Ay / m_A ** 2, S_By / m_B ** 2)
    J = d3.adm_J_closed_form(b, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec)
    assert abs(J[0]) < 1e-13 and abs(J[2]) < 1e-13
    assert abs(J[1] - (2.0 * b * p_t + S_Ay + S_By)) < 1e-12


def test_qc_in_plane_spin_tilts_J_off_y():
    # in-plane spin (S_z) does NOT feed the aligned SO correction; it tilts J off y.
    names = ["b", "q", "S_Az"]
    b, q = 6.0, 1.0
    m_A, m_B = 0.5, 0.5
    sl = p3.theta_to_slice3d([b, q, 0.25], names, fixed={"qc": 1.0})
    sl0 = p3.theta_to_slice3d([b, q], ["b", "q"], fixed={"qc": 1.0})
    assert sl.P_A_vec[0] == sl0.P_A_vec[0]        # S_z leaves p_t untouched
    J = d3.adm_J_closed_form(b, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec)
    assert abs(J[2] - 0.25) < 1e-13               # J_z = S_z


def test_qc_planar_polar_spin_is_nonspinning_qc():
    # QC composed with the planar (polar) spin family: S_y=0 always ⇒ non-spinning QC
    b = 3.0
    sl = p3.theta_to_slice3d([b, 0.3, 40.0], ["b", "S_mag", "theta_S"],
                             fixed={"qc": 1.0})
    th = np.deg2rad(40.0)
    assert sl.S_A_vec == (0.3 * np.sin(th), 0.0, 0.3 * np.cos(th))
    P_A, P_B = qc.qc_momenta(b, 0.5, 0.5, sl.S_A_vec, (0, 0, 0))
    assert np.allclose(sl.P_A_vec, P_A)           # S_y=0 ⇒ no SO shift


def test_qc_box_plugs_into_smolyak_builder():
    # the QC 4-D aligned-spin box builds a valid Smolyak grid (synthetic solve_fn)
    box = [{"name": n, "min": lo, "max": hi} for n, lo, hi in [
        ("b", 2.0, 6.0), ("q", 1.0, 3.0),
        ("S_Ay", -0.4, 0.4), ("S_By", -0.4, 0.4)]]
    axes = [(a["min"], a["max"]) for a in box]

    class _Info:
        iters, residual_norm = 2, 1e-15

    def solve_fn(theta, guess, tol, max_iter):
        return np.array(float(np.sum(np.sin(theta)))), _Info()

    s = sm.SmolyakSolverND(solve_fn, axes).build_isotropic(2)
    assert np.isfinite(float(s.evaluate([3.0, 1.5, 0.1, -0.2])))


def test_d4_qc_box_matches_spin8_slice_and_builds():
    # Pin the production d4_qc box (P2 workhorse): it must be exactly the
    # in-plane-spin=0 slice of the spin8 box so P3 (8-D precessing QC) reuses the
    # P2 corpus node-for-node.  Guards: same b,q ranges; aligned axes S_Ay/S_By over
    # the same [-0.4,0.4] and SYMMETRIC about 0 (nested level-0 midpoint = exactly 0);
    # FIXED flag = {"qc": 1.0}.
    import lemaitre.initial_data.pipeline.build_surrogate as bs

    box = bs.BOXES["d4_qc"]
    names = [a["name"] for a in box]
    assert names == ["b", "q", "S_Ay", "S_By"]          # b first → D7 per-b cache
    assert bs.FIXED["d4_qc"] == {"qc": 1.0}

    spin8 = {a["name"]: a for a in bs.BOXES["spin8"]}
    for a in box:
        s8 = spin8[a["name"]]
        assert (a["min"], a["max"]) == (s8["min"], s8["max"]), a["name"]
    for name in ("S_Ay", "S_By"):
        a = spin8[name]
        assert a["min"] == -a["max"]                     # symmetric about 0

    # builds a valid Smolyak grid through the real box (synthetic solve_fn), and the
    # nodes map to QC momenta (tangential x, zero net linear momentum)
    axes = [(a["min"], a["max"]) for a in box]

    class _Info:
        iters, residual_norm = 2, 1e-15

    def solve_fn(theta, guess, tol, max_iter):
        sl = p3.theta_to_slice3d(theta, names, fixed=bs.FIXED["d4_qc"])
        assert sl.P_A_vec[1] == 0.0                      # tangential along x
        assert np.allclose(np.array(sl.P_A_vec) + np.array(sl.P_B_vec), 0.0)
        return np.array(float(np.sum(np.sin(theta)))), _Info()

    s = sm.SmolyakSolverND(solve_fn, axes).build_isotropic(1)
    assert np.isfinite(float(s.evaluate([2.5, 1.5, 0.1, -0.2])))


def test_spin8_qc_box_matches_d4_qc_subslice_and_builds():
    # Pin the production spin8_qc box (P3 8-D precessing QC): same axes/ranges as the
    # head-on spin8 box, FIXED={"qc":1.0}, in-plane axes symmetric about 0.  Its
    # in-plane=0 sub-slice must produce the SAME Slice3D as d4_qc at matching
    # (b,q,S_Ay,S_By) — so P3 reuses the P2 (d4_qc) corpus node-for-node.
    import lemaitre.initial_data.pipeline.build_surrogate as bs

    box = bs.BOXES["spin8_qc"]
    names = [a["name"] for a in box]
    assert names == ["b", "q", "S_Ax", "S_Ay", "S_Az", "S_Bx", "S_By", "S_Bz"]
    assert bs.FIXED["spin8_qc"] == {"qc": 1.0}

    # same axes/ranges as the head-on spin8 box (only the momenta differ, via qc flag)
    spin8 = {a["name"]: a for a in bs.BOXES["spin8"]}
    for a in box:
        s8 = spin8[a["name"]]
        assert (a["min"], a["max"]) == (s8["min"], s8["max"]), a["name"]
    # in-plane axes symmetric about 0 → nested level-0 midpoint exactly 0
    for name in ("S_Ax", "S_Az", "S_Bx", "S_Bz"):
        a = spin8[name]
        assert a["min"] == -a["max"], name

    # in-plane=0 sub-slice coincides with d4_qc node-for-node
    d4_names = ["b", "q", "S_Ay", "S_By"]
    b, q, S_Ay, S_By = 2.7, 1.8, 0.15, -0.25
    sl_d4 = p3.theta_to_slice3d([b, q, S_Ay, S_By], d4_names, fixed={"qc": 1.0})
    theta8 = [b, q, 0.0, S_Ay, 0.0, 0.0, S_By, 0.0]  # S_Ax=S_Az=S_Bx=S_Bz=0
    sl_8 = p3.theta_to_slice3d(theta8, names, fixed={"qc": 1.0})
    assert sl_8.b == sl_d4.b
    assert sl_8.m_A == sl_d4.m_A and sl_8.m_B == sl_d4.m_B
    assert np.allclose(sl_8.S_A_vec, sl_d4.S_A_vec)
    assert np.allclose(sl_8.S_B_vec, sl_d4.S_B_vec)
    assert np.allclose(sl_8.P_A_vec, sl_d4.P_A_vec)
    assert np.allclose(sl_8.P_B_vec, sl_d4.P_B_vec)
    # tangential along x, zero net linear momentum (generic precessing point too)
    sl_gen = p3.theta_to_slice3d([2.0, 1.5, 0.3, 0.25, 0.2, -0.2, 0.3, -0.15],
                                 names, fixed={"qc": 1.0})
    assert np.allclose(np.array(sl_gen.P_A_vec) + np.array(sl_gen.P_B_vec), 0.0)

    # builds a valid Smolyak grid through the real 8-D box (synthetic solve_fn)
    axes = [(a["min"], a["max"]) for a in box]

    class _Info:
        iters, residual_norm = 2, 1e-15

    def solve_fn(theta, guess, tol, max_iter):
        return np.array(float(np.sum(np.sin(theta)))), _Info()

    s = sm.SmolyakSolverND(solve_fn, axes).build_isotropic(1)
    assert np.isfinite(float(s.evaluate([2.0, 1.5, 0.1, -0.1, 0.2, 0.0, 0.3, -0.2])))
