"""Acceptance tests — the caller-supplied angular map (``operators_abt.AngularMap``).

This package defines no instance: the hook lets a sibling hand in its own angle (the
conformally curved leaf's separation-dependent map) and have every construction path
treat it as a mapped grid.  The tests use two test-local maps:

  * ``BLEND`` — ``B = (1 − β)s + β·2s/(1 + s²)`` at ``β = 0.6``: ``B′(±1) = 0.4 ≠ 0``, so
    ``√(1 − B²)`` keeps its axis branch point (``axis_rational = False``).  No factor cap,
    and the evaluator divides ``w_m`` out as on ``"cos"``.
  * ``ABT_COPY`` — the ``"abt"`` formula under another key, ``axis_rational = True``: the
    cap applies, as on ``"abt"``.

That ``"cos"`` and ``"abt"`` are unchanged bit for bit is the job of the existing gates
(``test_bmap_abt.py``, ``test_scheme_v2.py``), which this file does not repeat.
"""

import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.solver import operators_3d as ops3
from lemaitre.initial_data.conformally_flat_puncture.solver import operators_abt as ops
from lemaitre.initial_data.conformally_flat_puncture.solver import separable
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d_nk as nk
from lemaitre.initial_data.conformally_flat_puncture.solver.solver_3d import Slice3D
from lemaitre.initial_data.conformally_flat_puncture.validation import export_grteclyn as ex

GRID = dict(Na=20, Nb=14, Nphi=6)


class _Blend(ops.AngularMap):
    axis_rational = False

    def __init__(self, beta):
        self.beta = float(beta)
        self.key = f"test-blend:{self.beta!r}"

    def b_of_s(self, s):
        return (1 - self.beta) * s + self.beta * 2 * s / (1 + s * s)

    def dB_ds(self, s):
        return (1 - self.beta) + self.beta * 2 * (1 - s * s) / (1 + s * s) ** 2

    def d2B_ds2(self, s):
        return self.beta * 4 * s * (s * s - 3) / (1 + s * s) ** 3

    def s_of_b(self, B, one_minus_B, one_plus_B):
        # solve for the distance σ to the nearer end from 1 ∓ B, which the hook passes in
        # without cancellation: 1 − B(1 − σ) = (1 − β)σ + βσ²/(1 + (1 − σ)²), increasing
        B = np.asarray(B, dtype=float)
        pos = B >= 0
        t = np.where(pos, one_minus_B, one_plus_B)
        sig = t / (1 - self.beta)
        for _ in range(80):
            f = (1 - self.beta) * sig + self.beta * sig ** 2 / (1 + (1 - sig) ** 2) - t
            sig = np.clip(sig - f / self.dB_ds(1 - sig), 0.0, 1.0)
        return np.where(pos, 1 - sig, sig - 1)


class _AbtCopy(ops.AngularMap):
    key = "test-abt-copy"
    axis_rational = True

    def b_of_s(self, s):
        return ops.b_of_s(s, "abt")

    def dB_ds(self, s):
        return ops.dB_ds(s, "abt")

    def d2B_ds2(self, s):
        return ops.d2B_ds2(s, "abt")

    def s_of_b(self, B, one_minus_B, one_plus_B):
        return ops.s_of_b(B, "abt", one_minus_B, one_plus_B)


BLEND = _Blend(0.6)
ABT_COPY = _AbtCopy()


def _slice(b=2.5):
    return Slice3D(b=b, m_A=0.6, m_B=0.4,
                   P_A_vec=(0.0, 0.11, -0.02), P_B_vec=(0.0, -0.11, 0.02),
                   S_A_vec=(0.05, 0.0, 0.03), S_B_vec=(0.0, 0.04, -0.01))


def test_identity_is_the_key_and_a_bad_key_is_refused():
    assert _Blend(0.6) == BLEND and hash(_Blend(0.6)) == hash(BLEND)
    assert _Blend(0.5) != BLEND and BLEND != "abt" and BLEND != "cos"

    class _NoKey(_Blend):
        key = ""

    bad = _NoKey.__new__(_NoKey)
    bad.beta, bad.key = 0.6, ""
    with pytest.raises(ValueError, match="key"):
        ops.check_bmap(bad)
    bad.key = "abt"
    with pytest.raises(ValueError, match="key"):
        ops.check_bmap(bad)
    with pytest.raises(ValueError, match="bmap"):
        ops.resolve_scheme("paper-v1", BLEND, None)          # a scheme name and a custom angle


def test_grid_is_gauss_legendre_in_s_and_DB1_the_chain_rule():
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"], BLEND, "lobatto")
    s, Ds = ops.gl_nodes_diffmat(GRID["Nb"])
    assert np.array_equal(B, BLEND.b_of_s(s))
    assert np.array_equal(DB1, (1.0 / BLEND.dB_ds(s))[:, None] * np.asarray(Ds))
    assert np.array_equal(A, ops.cheb01(GRID["Na"])[0])
    assert ops.scheme_of(BLEND, "lobatto") is None              # a mixed grid, never a scheme


def test_second_derivative_is_exact_and_every_operator_uses_it():
    Na, Nb = GRID["Na"], GRID["Nb"]
    A, B, DA1, DB1 = ops.build_grid(Na, Nb, BLEND, "lobatto")
    DBB1 = ops.d2_dB2_1d(Nb, BLEND)
    assert np.array_equal(ops.angular_d2(B, DB1), DBB1)
    assert ops.angular_d2(B * (1 + 1e-15), DB1) is None          # exact recognition only
    s = ops.s_nodes(Nb, BLEND)
    p = s ** 11 - 3 * s ** 4 + s                                  # degree < Nb
    ps, pss = 11 * s ** 10 - 12 * s ** 3 + 1, 110 * s ** 9 - 36 * s ** 2
    ref = pss / BLEND.dB_ds(s) ** 2 - BLEND.d2B_ds2(s) * ps / BLEND.dB_ds(s) ** 3
    err_exact = np.max(np.abs(DBB1 @ p - ref)) / np.max(np.abs(ref))
    err_nested = np.max(np.abs(DB1 @ (DB1 @ p) - ref)) / np.max(np.abs(ref))
    print(f"\n[blend d2] exact {err_exact:.1e}, nested square {err_nested:.1e}")
    assert err_exact < 1e-11 and err_nested > 1e-6
    _, G_B, _ = separable.kron_factors(A, B, DA1, DB1, 0)
    assert np.array_equal(G_B, (1.0 - B ** 2)[:, None] * DBB1 + (-2.0 * B)[:, None] * DB1)
    pc = ops3._mode_independent_pieces(A, B, DA1, DB1, 3.0)
    assert np.array_equal(pc["DBB"], np.kron(np.eye(Na + 1), DBB1))


def test_factor_cap_follows_axis_rational_not_mapped():
    Nb = GRID["Nb"]
    _, Bb, _, _ = ops.build_grid(1, Nb, BLEND, "lobatto")
    _, Ba, _, _ = ops.build_grid(1, Nb, ABT_COPY, "lobatto")
    assert not ops3._is_mapped_angular(Bb)
    assert ops3._is_mapped_angular(Ba)
    m = ops3.ABT_FACTOR_M_MAX + 1
    assert ops3.angular_factor_order(Bb, m) == m
    assert ops3.angular_factor_order(Ba, m) == ops3.ABT_FACTOR_M_MAX
    assert ops.axis_rational(ABT_COPY) and not ops.axis_rational(BLEND)
    assert ops.axis_rational("abt") and not ops.axis_rational("cos")


def test_operator_cache_serves_the_map_its_own_blocks():
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"], BLEND, "lobatto")
    assert ops3._canonical_grid(A, B, DA1, DB1) == (BLEND, "lobatto")
    m_vals = ops3.fourier_modes(GRID["Nphi"])
    got = ops3.mode_operators_cached(A, B, DA1, DB1, 2.5, m_vals)[0]
    ref = ops3.mode_operators(A, B, DA1, DB1, 2.5, m_vals)[0]
    for x, y in zip(got, ref):
        assert np.array_equal(x, y)
    Ac, Bc, DA1c, DB1c = ops.build_grid(GRID["Na"], GRID["Nb"], "cos", radial="lobatto")
    got_c = ops3.mode_operators_cached(Ac, Bc, DA1c, DB1c, 2.5, m_vals)[0]
    assert not np.array_equal(got_c[1], got[1])


def test_inverse_is_accurate_next_to_the_axis():
    b = 2.5
    s = np.array([1 - 1e-12, 1 - 1e-9, 0.999, 0.3, -0.3, -0.999, -(1 - 1e-9)])
    A = np.full(s.size, 0.3)
    B = BLEND.b_of_s(s)
    # the physical point from s, with 1 − B² from the exact factorisation
    om = (1 - s) * ((1 - BLEND.beta) + BLEND.beta * (1 - s) / (1 + s * s))
    op = (1 + s) * ((1 - BLEND.beta) + BLEND.beta * (1 + s) / (1 + s * s))
    rho = b * 2 * A / (1 - A ** 2) * np.sqrt(om * op)
    z = b * (1 + A ** 2) / (1 - A ** 2) * B
    _, s_back = ops.inverse_map_s(rho, z, b, BLEND)
    rel = np.abs(s_back - s) / np.minimum(1 - np.abs(s), 1.0)
    print(f"\n[blend inverse] max |ds|/(1-|s|) = {np.max(rel):.1e}")
    assert np.max(rel) < 1e-6


def test_evaluate_field_is_exact_on_its_discrete_space_odd_m_included():
    """û_m = w_m(B)·P_m(A, s) with the FULL exponent: nodal values back, and an off-grid
    query exact — the branch for a non-axis-rational map divides w_m out as on "cos"."""
    Na, Nb, Nphi, b = GRID["Na"], GRID["Nb"], GRID["Nphi"], 2.5
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi, bmap=BLEND, radial="lobatto")
    rng = np.random.default_rng(3)
    cA = rng.standard_normal((Nphi, 6)) * 1e-2
    cS = rng.standard_normal((Nphi, 7))

    def field(A, s, phi):
        B = BLEND.b_of_s(s)
        out = 0.0
        for k in range(Nphi // 2 + 1):
            w = np.clip(1 - B ** 2, 0, None) ** (k / 2)
            pa = np.polynomial.chebyshev.chebval(2 * A - 1, cA[k])
            ps = np.polynomial.legendre.legval(s, cS[k])
            out = out + w * pa * ps * (np.cos(k * phi) + (0.3 * np.sin(k * phi)
                                                          if 0 < k < Nphi / 2 else 0.0))
        return out

    AA, SS, PP = np.meshgrid(prob.A, prob.Bs, prob.phi, indexing="ij")
    U = field(AA, SS, PP)
    A_q = rng.uniform(0.05, 0.95, 50)
    s_q = np.concatenate([rng.uniform(-0.99, 0.99, 44), [0.9999, -0.9999, 0.0, 0.5, -0.5, 0.9]])
    phi_q = rng.uniform(0, 2 * np.pi, 50)
    B_q = BLEND.b_of_s(s_q)
    rho_q, z_q = ops.abt_map(A_q, B_q, b)
    got = s3.evaluate_field(prob, U, rho_q, z_q, phi_q, b)
    err = np.max(np.abs(got - field(A_q, s_q, phi_q))) / np.max(np.abs(U))
    print(f"\n[blend eval] expansion err {err:.2e}")
    assert err < 1e-10
    idx = [(3, 5, 2), (1, 0, 0), (Na - 1, Nb - 1, Nphi - 1)]
    rho_n, z_n = ops.abt_map(np.array([prob.A[i] for i, _, _ in idx]),
                             np.array([prob.B[j] for _, j, _ in idx]), b)
    got_n = s3.evaluate_field(prob, U, rho_n, z_n, np.array([prob.phi[k] for *_, k in idx]), b)
    assert np.max(np.abs(got_n - np.array([U[i, j, k] for i, j, k in idx]))) < 1e-12


def test_separable_route_matches_dense_on_the_map():
    prob = s3.make_problem(**GRID, bmap=BLEND, radial="lobatto")
    sl = _slice()
    U_s, i_s = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30, separable=True)
    U_d, i_d = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30, separable=False)
    d = float(np.max(np.abs(np.asarray(U_s) - np.asarray(U_d))))
    print(f"\n[blend sep vs dense] |dU|={d:.2e}  its {i_s.iters}/{i_d.iters}")
    assert i_s.converged and i_d.converged
    assert d < 1e-12


def test_export_refuses_the_map():
    prob = s3.make_problem(**GRID, bmap=BLEND, radial="lobatto")
    with pytest.raises(NotImplementedError, match="mixed grid"):
        ex.from_solution(prob, _slice(), np.zeros(prob.shape))


def test_a_map_that_rounds_to_cos_never_changes_paper_v1():
    """A registered map whose nodes round to "cos"'s must not be recognised on a paper-v1
    grid: same-process results may not depend on which maps were used before (without the
    guard, ``_Blend(1e-20)`` turned paper-v1's ``angular_d2`` from None to a matrix)."""
    A, B, DA1, DB1 = ops.build_grid(36, 24, scheme="paper-v1")
    Lap0 = ops.laplacian_matrix(A, B, DA1, DB1, 3.0)[0]
    tiny = _Blend(1e-20)
    At, Bt, DA1t, DB1t = ops.build_grid(36, 24, tiny, "lobatto")
    assert np.array_equal(Bt, B) and np.array_equal(DB1t, DB1)     # the premise: identical arrays
    assert ops.angular_d2(B, DB1) is None
    assert np.array_equal(ops.laplacian_matrix(A, B, DA1, DB1, 3.0)[0], Lap0)
    assert not ops3._is_mapped_angular(B)


def test_persistence_refuses_a_caller_map(tmp_path):
    from lemaitre.initial_data.conformally_flat_puncture.parametric import solve_store as ss
    with pytest.raises(NotImplementedError, match="angular map"):
        ss.SolveStore(str(tmp_path), (20, 14, 6), code_tag="t", bmap=BLEND, radial="lobatto")
