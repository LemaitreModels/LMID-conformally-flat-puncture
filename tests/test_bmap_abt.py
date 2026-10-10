"""Acceptance tests — the angular node map (``operators_abt.B_MAPS``).

The operator is written in ``B = cos η`` throughout; the map only decides where the
B nodes sit and in which coordinate ``s`` the discrete field is a polynomial.

  * ``"cos"`` (default): ``s = B``, the original grid.  Every existing number is
    pinned to it, so the first group of tests asserts that it is BIT-FOR-BIT what
    it was — nodes, differentiation matrix, evaluator, solve-store keys, saved
    artifacts.
  * ``"abt"``: ``B = 2s/(1+s²)``, the angular coordinate of Ansorg, Brügmann &
    Tichy, PRD 70, 064011 (2004), eq. (61).  It is linear in η at a puncture where
    ``"cos"`` is quadratic, so it resolves the punctures far better.  The second
    group checks the construction (map, chain rule, evaluator, the m=0 oracle, the
    separable route, the Nφ=1 reduction) and the third checks the point of it: a
    field closer to a converged reference near the punctures at the same grid.
"""

import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.solver import operators_3d as ops3
from lemaitre.initial_data.conformally_flat_puncture.solver import operators_abt as ops
from lemaitre.initial_data.conformally_flat_puncture.solver import separable
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d_nk as nk
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_abt as sa
from lemaitre.initial_data.conformally_flat_puncture.solver import source
from lemaitre.initial_data.conformally_flat_puncture.solver.solver_3d import Slice3D
from lemaitre.initial_data.conformally_flat_puncture.solver.solver_abt import Slice
from lemaitre.initial_data.conformally_flat_puncture.parametric import parametric_nd as pnd
from lemaitre.initial_data.conformally_flat_puncture.parametric import parametric_nd_3d as p3
from lemaitre.initial_data.conformally_flat_puncture.parametric import solve_store as ss
from lemaitre.initial_data.conformally_flat_puncture.validation import export_grteclyn as ex

GRID = dict(Na=20, Nb=14, Nphi=6)


def _slice(b=2.5):
    """Non-axisymmetric: off-axis momenta + misaligned spins (test_solver_3d_fast's)."""
    return Slice3D(b=b, m_A=0.6, m_B=0.4,
                   P_A_vec=(0.0, 0.11, -0.02), P_B_vec=(0.0, -0.11, 0.02),
                   S_A_vec=(0.05, 0.0, 0.03), S_B_vec=(0.0, 0.04, -0.01))


# ==========================================================================
# 1.  "cos" is the original grid, bit-for-bit
# ==========================================================================
def test_cos_is_the_default_and_the_original_grid():
    assert ops.DEFAULT_BMAP == "cos"
    A0, DA0 = ops.cheb01(GRID["Na"])
    B0, DB0 = ops.gl_nodes_diffmat(GRID["Nb"])
    for got in (ops.build_grid(GRID["Na"], GRID["Nb"]),
                ops.build_grid(GRID["Na"], GRID["Nb"], "cos"),
                ops.build_grid(GRID["Na"], GRID["Nb"], None)):
        for x, y in zip(got, (A0, B0, DA0, DB0)):
            assert np.array_equal(np.asarray(x), np.asarray(y))
    prob = s3.make_problem(**GRID)
    assert prob.bmap == "cos" and np.array_equal(prob.Bs, prob.B)
    assert sa.make_problem(Na=20, Nb=14).bmap == "cos"


def test_unknown_map_is_refused():
    with pytest.raises(ValueError, match="unknown angular node map"):
        ops.build_grid(10, 8, "chebyshev")
    with pytest.raises(ValueError, match="unknown angular node map"):
        s3.make_problem(10, 8, 4, bmap="tan")


def test_solve_store_keys_unchanged_for_cos_and_distinct_for_abt():
    """A store key written before maps existed must still be found."""
    sl = _slice()
    old = ss.slice_key(sl, (20, 14, 6), "t")
    assert ss.slice_key(sl, (20, 14, 6), "t", "cos") == old
    assert ss.slice_key(sl, (20, 14, 6), "t", "abt") != old
    assert "bmap" not in ss.slice_payload(sl, (20, 14, 6), "t")


def _tiny_model(bmap):
    """A 1-axis dense model whose attached solver is stamped with ``bmap``."""
    def solve_fn(*a, **k):                                  # pragma: no cover
        raise AssertionError("not called")
    solve_fn.bmap = bmap
    nodes = [np.array([2.0, 1.0])]
    return pnd.ParametricSolutionND(
        axes=[(1.0, 2.0, 1)], nodes=nodes, weights=[np.array([0.5, -0.5])],
        U_nodes=np.zeros((2, 3, 2, 1)), iters=np.zeros(2, dtype=int),
        residuals=np.zeros(2), _solve_fn=solve_fn)


def test_saved_model_records_a_mapped_grid_and_leaves_cos_artifacts_alone(tmp_path):
    p_cos = _tiny_model("cos").save(str(tmp_path / "cos.npz"), meta={"Na": 2})
    p_abt = _tiny_model("abt").save(str(tmp_path / "abt.npz"), meta={"Na": 2})
    m_cos = pnd._unpack_meta(np.load(p_cos))
    m_abt = pnd._unpack_meta(np.load(p_abt))
    assert "bmap" not in m_cos, "a 'cos' artifact must stay byte-identical to before"
    assert m_abt["bmap"] == "abt"
    with pytest.raises(ValueError, match="contradict|says bmap"):
        _tiny_model("abt").save(str(tmp_path / "x.npz"), meta={"bmap": "cos"})


def test_attaching_a_solver_on_the_other_grid_is_refused(tmp_path):
    p_cos = _tiny_model("cos").save(str(tmp_path / "cos.npz"))
    p_abt = _tiny_model("abt").save(str(tmp_path / "abt.npz"))
    m_cos, m_abt = pnd.load_parametric(p_cos), pnd.load_parametric(p_abt)
    prob_cos = s3.make_problem(10, 8, 4)
    prob_abt = s3.make_problem(10, 8, 4, bmap="abt")
    pnd.attach_solve_fn_3d(m_cos, prob_cos, ["b"])          # same grid: fine
    pnd.attach_solve_fn_3d(m_abt, prob_abt, ["b"])
    with pytest.raises(ValueError, match="angular grid"):
        pnd.attach_solve_fn_3d(m_cos, prob_abt, ["b"])
    with pytest.raises(ValueError, match="angular grid"):
        pnd.attach_solve_fn_3d(m_abt, prob_cos, ["b"])


def test_export_refuses_a_mapped_grid():
    prob = s3.make_problem(**GRID, bmap="abt")
    with pytest.raises(NotImplementedError, match="angular node map"):
        ex.from_solution(prob, _slice(), np.zeros(prob.shape))


# ==========================================================================
# 2.  The "abt" construction
# ==========================================================================
def test_abt_map_and_its_inverse():
    s = np.concatenate([np.linspace(-1.0, 1.0, 2001), [0.0, 1e-300, -1e-12]])
    B = ops.b_of_s(s, "abt")
    # with 1 -/+ B supplied exactly, the inverse is good to ~eps everywhere ...
    om, op = (1 - s) ** 2 / (1 + s * s), (1 + s) ** 2 / (1 + s * s)
    assert np.max(np.abs(ops.s_of_b(B, "abt", om, op) - s)) < 5e-16      # 2 ulp at |s| ~ 1
    # ... and from B alone it is limited by ds/dB ~ 1/(1 - s^2) near the ends
    err = np.abs(ops.s_of_b(B, "abt") - s)
    assert np.all(err <= 1e-15 + 4e-16 / np.maximum(1 - s * s, 1e-300))
    assert ops.s_of_b(np.array([1.0, -1.0, 0.0]), "abt").tolist() == [1.0, -1.0, 0.0]
    # the nodes are kept exactly as laid
    assert np.array_equal(s3.make_problem(10, 12, 4, bmap="abt").Bs,
                          ops.gl_nodes_diffmat(12)[0])
    # eta is linear in (1 - s) at the puncture: 1 - B = (1-s)^2/(1+s^2)
    # (moderate e only: 1 - B itself cancels in float64 as e -> 0, the very effect
    #  s_of_b's one_minus_B argument exists for)
    e = np.array([0.3, 1e-1, 1e-2, 1e-3])
    assert np.allclose(1.0 - ops.b_of_s(1.0 - e, "abt"), e ** 2 / (1.0 + (1.0 - e) ** 2),
                       rtol=1e-9, atol=0)


def test_abt_DB1_is_the_chain_rule_derivative():
    """DB1 differentiates, with respect to B, every polynomial in s of degree < Nb."""
    Nb = GRID["Nb"]
    _, B, _, DB1 = ops.build_grid(GRID["Na"], Nb, "abt")
    s = ops.s_of_b(B, "abt")
    rng = np.random.default_rng(1)
    c = rng.standard_normal(Nb)
    f = np.polynomial.legendre.legval(s, c)
    dfds = np.polynomial.legendre.legval(s, np.polynomial.legendre.legder(c))
    want = dfds / ops.dB_ds(s, "abt")
    err = np.max(np.abs(DB1 @ f - want)) / np.max(np.abs(want))
    print(f"\n[abt DB1] rel err {err:.2e}")
    assert err < 1e-10


def test_m0_block_is_the_2d_operator_on_the_abt_grid():
    """The oracle relation of test_solver_3d_fast holds on the mapped grid too."""
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"], "abt")
    Lap, _, _, _, _, DA, _ = ops.laplacian_matrix(A, B, DA1, DB1, 2.5)
    M_2d, int_2d = ops.apply_bcs(Lap, A, B, DA)
    M0_list, w_list, interior = ops3.mode_operators(A, B, DA1, DB1, 2.5, np.array([0]))
    assert np.array_equal(interior, int_2d)
    assert np.array_equal(M0_list[0], M_2d)


def test_operator_cache_serves_each_map_its_own_blocks():
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"], "abt")
    m_vals = ops3.fourier_modes(GRID["Nphi"])
    got = ops3.mode_operators_cached(A, B, DA1, DB1, 2.5, m_vals)[0]
    ref = ops3.mode_operators(A, B, DA1, DB1, 2.5, m_vals)[0]
    for x, y in zip(got, ref):
        assert np.array_equal(x, y)
    Ac, Bc, DA1c, DB1c = ops.build_grid(GRID["Na"], GRID["Nb"], "cos")
    got_c = ops3.mode_operators_cached(Ac, Bc, DA1c, DB1c, 2.5, m_vals)[0]
    assert not np.array_equal(got_c[1], got[1]), "the two maps shared a cache entry"


def test_evaluate_field_on_abt_is_exact_on_its_discrete_space():
    """Nodal values recovered, and an off-grid query exact for û_m polynomial in (A, s)."""
    Na, Nb, Nphi, b = GRID["Na"], GRID["Nb"], GRID["Nphi"], 2.5
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi, bmap="abt")
    rng = np.random.default_rng(7)
    cA = rng.standard_normal((Nphi, 6)) * 1e-2
    cS = rng.standard_normal((Nphi, 7))

    def field(A, s, phi):
        out = 0.0
        for k in range(Nphi // 2 + 1):
            pa = np.polynomial.chebyshev.chebval(2 * A - 1, cA[k])
            ps = np.polynomial.legendre.legval(s, cS[k])
            out = out + pa * ps * (np.cos(k * phi) + (0.3 * np.sin(k * phi)
                                                       if 0 < k < Nphi / 2 else 0.0))
        return out

    AA, SS, PP = np.meshgrid(prob.A, prob.Bs, prob.phi, indexing="ij")
    U = field(AA, SS, PP)
    A_q = rng.uniform(0.05, 0.95, 50)
    s_q = np.concatenate([rng.uniform(-0.99, 0.99, 40),
                          [0.999, -0.999, 0.99999, -0.99999, 1 - 1e-9, -(1 - 1e-9),
                           0.0, 0.5, -0.5, 0.9]])
    phi_q = rng.uniform(0, 2 * np.pi, 50)
    # the physical point from s directly (sin η = (1-s²)/(1+s²)), so a query a hair
    # off the axis is not rounded onto it by going through B
    rho_q = b * 2 * A_q / (1 - A_q ** 2) * (1 - s_q ** 2) / (1 + s_q ** 2)
    z_q = b * (1 + A_q ** 2) / (1 - A_q ** 2) * ops.b_of_s(s_q, "abt")
    got = s3.evaluate_field(prob, U, rho_q, z_q, phi_q, b)
    err = np.max(np.abs(got - field(A_q, s_q, phi_q))) / np.max(np.abs(U))
    print(f"\n[abt eval] expansion err {err:.2e}")
    assert err < 1e-11
    idx = [(3, 5, 2), (1, 0, 0), (Na - 1, Nb - 1, Nphi - 1)]
    rho_n, z_n = ops.abt_map(np.array([prob.A[i] for i, _, _ in idx]),
                             np.array([prob.B[j] for _, j, _ in idx]), b)
    got_n = s3.evaluate_field(prob, U, rho_n, z_n, np.array([prob.phi[k] for *_, k in idx]), b)
    assert np.max(np.abs(got_n - np.array([U[i, j, k] for i, j, k in idx]))) < 1e-12


def test_separable_route_matches_dense_on_abt():
    prob = s3.make_problem(**GRID, bmap="abt")
    sl = _slice()
    U_s, i_s = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30, separable=True)
    U_d, i_d = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30, separable=False)
    d = float(np.max(np.abs(np.asarray(U_s) - np.asarray(U_d))))
    print(f"\n[abt sep vs dense] |dU|={d:.2e}  resid {i_s.residual_norm:.1e}/"
          f"{i_d.residual_norm:.1e}  its {i_s.iters}/{i_d.iters}")
    assert i_s.converged and i_d.converged
    assert d < 1e-12


def test_separable_assembly_refuses_a_hand_built_grid():
    """A Problem3D whose nodes are not its declared map's would be preconditioned
    on the wrong grid — refused rather than solved."""
    import dataclasses
    prob = s3.make_problem(**GRID)
    _, B, _, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"], "abt")
    bad = dataclasses.replace(prob, B=B, DB1=DB1)          # abt nodes, says "cos"
    with pytest.raises(ValueError, match="canonical"):
        s3.assemble(bad, _slice(), separable=True)


def test_axisym_reduction_reproduces_2d_on_abt():
    """Nφ=1 NK on the abt grid IS the 2-D Newton on the same grid."""
    P = 0.5
    for (b, mA, mB) in [(1.0, 0.5, 0.5), (1.5, 0.7, 0.3)]:
        prob2 = sa.make_problem(Na=28, Nb=20, P=P, bmap="abt")
        U2, _ = sa.newton_solve(prob2, Slice(b=b, m_A=mA, m_B=mB), tol=1e-12, max_iter=25)
        prob3 = s3.make_problem(Na=28, Nb=20, Nphi=1, bmap="abt")
        U3, i3 = nk.newton_solve_nk(prob3, Slice3D.head_on(b=b, m_A=mA, m_B=mB, P=P),
                                    tol=1e-12, max_iter=25)
        d = float(np.max(np.abs(np.asarray(U3)[:, :, 0] - np.asarray(U2))))
        print(f"\n[abt axisym] b={b}: |U3-U2|={d:.2e}")
        assert d < 1e-12
        # and the two evaluators agree off-grid on the mapped grid
        rho, z = np.array([0.7, 1.9, 0.2]), np.array([0.3, -1.1, 0.4])
        e3 = s3.evaluate_field(prob3, U3, rho, z, np.zeros(3), b)
        e2 = np.asarray(sa.evaluate_field_phys(prob2, U2, rho, z, b))
        assert np.max(np.abs(e3 - e2)) < 1e-12


# ==========================================================================
# 3.  The point: the punctures are resolved
# ==========================================================================
def _psi(prob, U, sl, rho, z, phi):
    return (np.asarray(source.psi_BL_2c(rho, z, sl.b, sl.m_A, sl.m_B))
            + s3.evaluate_field(prob, U, rho, z, phi, sl.b))


def _near_probes(sl, n=20, seed=0):
    """Points 0.02–0.5 m from each puncture, random directions."""
    rng = np.random.default_rng(seed)
    out = []
    for zc, m in ((+sl.b, sl.m_A), (-sl.b, sl.m_B)):
        r = np.exp(rng.uniform(np.log(0.02 * m), np.log(0.5 * m), n))
        ct, ph = rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n)
        st = np.sqrt(1 - ct ** 2)
        x, y, zz = r * st * np.cos(ph), r * st * np.sin(ph), zc + r * ct
        out.append((np.hypot(x, y), zz, np.arctan2(y, x)))
    return tuple(np.concatenate([out[0][i], out[1][i]]) for i in range(3))


def test_abt_resolves_the_punctures_at_the_production_grid():
    """At 44×32×8 and b = 8 the near-puncture error drops by ≥ 50× under "abt".

    Both grids are judged against the same converged reference (abt 72×72×8, itself
    checked against a cos 72×128×8 one), so the comparison cannot favour either map.
    Measured 2026-10-10 at b = 10: 7.7e-4 → 5.0e-6 (cos → abt) within 0.5 m of a hole.
    """
    sl = p3.theta_to_slice3d([8.0, 1.5, 0.3, -0.2], ["b", "q", "chi_Ay", "chi_By"], 1.0,
                             {"qc": 1.0})
    rho, z, phi = _near_probes(sl)

    def run(Na, Nb, bmap):
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=8, bmap=bmap)
        U, info = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=40)
        assert info.converged
        return _psi(prob, U, sl, rho, z, phi)

    ref = run(72, 72, "abt")
    ref_cos = run(72, 128, "cos")
    agree = float(np.max(np.abs(ref - ref_cos)))
    e_cos = float(np.max(np.abs(run(44, 32, "cos") - ref)))
    e_abt = float(np.max(np.abs(run(44, 32, "abt") - ref)))
    print(f"\n[abt vs cos @44x32x8, b=8] near-puncture max|dpsi|: cos {e_cos:.2e}, "
          f"abt {e_abt:.2e} ({e_cos / e_abt:.0f}x); references agree to {agree:.1e}")
    assert agree < 1e-7, "the two references disagree: the comparison is not valid"
    assert e_abt * 50.0 < e_cos
    assert e_abt < 1e-5


def test_inverse_map_s_is_accurate_next_to_the_axis():
    """``s`` from the physical point keeps ~eps where ``s_of_b(B)`` would not."""
    b = 3.0
    A = np.array([0.1, 0.4, 0.7, 0.9, 0.3, 0.6, 0.2, 0.8])
    s = np.array([1 - 1e-9, -(1 - 1e-9), 1 - 1e-6, -(1 - 1e-12), 0.0, 0.3, -0.7, 1.0])
    rho = b * 2 * A / (1 - A ** 2) * (1 - s ** 2) / (1 + s ** 2)
    z = b * (1 + A ** 2) / (1 - A ** 2) * ops.b_of_s(s, "abt")
    A_q, s_q = ops.inverse_map_s(rho, z, b, "abt")
    print(f"\n[abt inverse] max|ds|={np.max(np.abs(s_q - s)):.1e}, "
          f"max|dA|={np.max(np.abs(A_q - A)):.1e}")
    assert np.max(np.abs(s_q - s)) < 1e-14
    assert np.max(np.abs(A_q - A)) < 1e-12
    for x, y in zip(ops.inverse_map_s(rho, z, b, "cos"), ops.inverse_map(rho, z, b)):
        assert np.array_equal(x, y)


def test_puncture_readouts_on_abt():
    """The B=±1 corner readouts (qc_effpot, sensitivity) interpolate in s."""
    from lemaitre.initial_data.conformally_flat_puncture.applications import qc_effpot as E
    from lemaitre.initial_data.conformally_flat_puncture.applications import sensitivity as sens
    import jax.numpy as jnp

    def f(A, s):                       # a polynomial in s, so exactly representable
        return 0.01 * (1 + A) * (0.3 + 0.2 * s - 0.5 * s ** 3 + 0.1 * s ** 4)

    prob = s3.make_problem(16, 12, 4, bmap="abt")
    AA, SS = np.meshgrid(prob.A, prob.Bs, indexing="ij")
    U = np.repeat(f(AA, SS)[:, :, None], 4, axis=2)
    for which, sq in (("A", 1.0), ("B", -1.0)):
        got = E._u_at_puncture(prob, U, which)
        assert abs(got - f(0.0, sq)) < 1e-14, (which, got, f(0.0, sq))
    prob2 = sa.make_problem(16, 12, bmap="abt")
    U2 = f(*np.meshgrid(prob2.A, prob2.Bs, indexing="ij"))
    wB = sa._bary_weights(prob2.Bs)
    for which, sq in (("A", 1.0), ("B", -1.0)):
        got = float(sens._u_at_puncture_jax(jnp.asarray(U2), prob2, wB, which))
        assert abs(got - f(0.0, sq)) < 1e-14
