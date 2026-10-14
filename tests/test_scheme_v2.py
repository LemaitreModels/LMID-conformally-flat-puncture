"""Acceptance tests — the two discretisation schemes (``operators_abt.SCHEMES``).

  * ``"paper-v1"`` = ``("cos", "lobatto")``: the grid of the first version of the paper —
    Gauss–Legendre in ``B = cos η``, Chebyshev–Gauss–Lobatto in ``A`` with a Dirichlet
    row at ``A = 1`` and a Neumann/Dirichlet row at ``A = 0``.  Every shipped number is
    pinned to it, so the first group asserts it is BIT-FOR-BIT the operator the leaf
    built before the radial axis existed, against a verbatim copy of that code kept here.
  * ``"v2"`` = ``("abt", "gauss")``: the angle and the radial representation of Ansorg,
    Brügmann & Tichy, PRD 70, 064011 (2004) — ``B = 2s/(1+s²)`` with Gauss–Legendre
    nodes in ``s`` (eq. (61)), and ``u = (A − 1) V`` on Chebyshev–Gauss ``A`` nodes with
    no boundary row (eqs. (5), (8)).  The second group checks the construction (nodes,
    the operator on its own discrete space, the m=0 oracle, the separable route, the
    Nφ=1 reduction, the evaluator and every edge readout); the third checks the point
    of it at the production grid: the ADM mass, and ψ not worse anywhere.
  * The radial choice is part of the grid's identity wherever the angular map is: the
    fourth group checks solve-store keys, saved models, attaching and the export.
  * Milestone 2 made ``"v2"`` the default.  The fifth group gates that, the refusal of
    one axis named alone, the paper-v1 constant ``pipeline.production_model.SCHEME``,
    and a source scan that no producer, figure, table or oracle script builds a grid
    without naming one.  Everything above that means the paper's grid names it.
"""

import dataclasses

import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.solver import diagnostics_3d as d3
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
from lemaitre.initial_data.conformally_flat_puncture.validation import adm
from lemaitre.initial_data.conformally_flat_puncture.validation import constraints
from lemaitre.initial_data.conformally_flat_puncture.validation import export_grteclyn as ex
from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm

GRID = dict(Na=20, Nb=14, Nphi=6)


def _slice(b=2.5):
    """Non-axisymmetric: off-axis momenta + misaligned spins (test_solver_3d_fast's)."""
    return Slice3D(b=b, m_A=0.6, m_B=0.4,
                   P_A_vec=(0.0, 0.11, -0.02), P_B_vec=(0.0, -0.11, 0.02),
                   S_A_vec=(0.05, 0.0, 0.03), S_B_vec=(0.0, 0.04, -0.01))


# ==========================================================================
# 1.  "paper-v1" is the original operator, bit-for-bit
# ==========================================================================
# Verbatim copies of the Lobatto-grid assembly as it stood before the radial axis was
# added (leaf commit 3bc10d3).  They are the oracle: a change to how the shipped code
# builds a "lobatto" operator — even a regrouping that moves the last ulp — fails here.
def _frozen_laplacian(A, B, DA1, DB1, b):
    Na1, Nb1 = A.size, B.size
    IA, IB = np.eye(Na1), np.eye(Nb1)
    DA = np.kron(np.asarray(DA1), IB)
    DB = np.kron(IA, np.asarray(DB1))
    DAA = DA @ DA
    DBB1 = ops.angular_d2(B, DB1)
    DBB = DB @ DB if DBB1 is None else np.kron(IA, DBB1)
    AA, BB = np.meshgrid(A, B, indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    alpha, pcoef, gamma, qcoef = ops._coeffs(Af, Bf, b)
    Lap = (alpha[:, None] * DAA + pcoef[:, None] * DA
           + gamma[:, None] * DBB + qcoef[:, None] * DB)
    return Lap, DA


def _frozen_bcs(Lap, Na1, Nb1, DA, m):
    M = np.array(Lap, dtype=float)
    interior = np.ones(Na1 * Nb1, dtype=bool)
    for j in range(Nb1):
        M[j, :] = 0.0
        M[j, j] = 1.0
        interior[j] = False
        r = (Na1 - 1) * Nb1 + j
        if m == 0:
            M[r, :] = DA[r, :]
        else:
            M[r, :] = 0.0
            M[r, r] = 1.0
        interior[r] = False
    return M, interior


def _frozen_block(A, B, DA1, DB1, b, m):
    Na1, Nb1 = A.size, B.size
    IA, IB = np.eye(Na1), np.eye(Nb1)
    DA = np.kron(np.asarray(DA1), IB)
    DB = np.kron(IA, np.asarray(DB1))
    DAA = DA @ DA
    DBB1 = ops.angular_d2(B, DB1)
    DBB = DB @ DB if DBB1 is None else np.kron(IA, DBB1)
    AA, BB = np.meshgrid(A, B, indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    rho, _ = ops.abt_map(Af, Bf, b)
    alpha, pcoef, gamma, qcoef = ops._coeffs(Af, Bf, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_rho2 = 1.0 / np.asarray(rho) ** 2
    inv_rho2 = np.where(np.isfinite(inv_rho2) & (np.asarray(rho) > 1e-14), inv_rho2, 0.0)
    cent = -(int(m) ** 2) * inv_rho2
    w, wp, wpp = ops3.bc_factor(Bf, m)
    Mv = (alpha * w)[:, None] * DAA + (pcoef * w)[:, None] * DA
    Mv += gamma[:, None] * (2.0 * wp[:, None] * DB + w[:, None] * DBB)
    Mv += qcoef[:, None] * (w[:, None] * DB)
    Mv[np.diag_indices_from(Mv)] += gamma * wpp + qcoef * wp + cent * w
    M, interior = _frozen_bcs(Mv, Na1, Nb1, DA, m)
    return M, w, interior


def _frozen_kron_factors(A, B, DA1, DB1, m):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    DA1, DB1 = np.asarray(DA1, dtype=float), np.asarray(DB1, dtype=float)
    edge = (A <= 1e-14) | (A >= 1.0 - 1e-14)
    Asafe = np.where(edge, 0.5, A)
    oa2 = 1.0 - Asafe ** 2
    L_A = (oa2 ** 2 / 4.0)[:, None] * (DA1 @ DA1) + (oa2 ** 2 / (4.0 * Asafe))[:, None] * DA1
    W, Wp, Wpp = ops3.bc_factor(B, m)
    ob2 = 1.0 - B ** 2
    DBB1 = ops.angular_d2(B, DB1)
    if DBB1 is None:
        DBB1 = DB1 @ DB1
    C_B = (ob2[:, None] * (np.diag(Wpp) + 2.0 * Wp[:, None] * DB1 + W[:, None] * DBB1)
           + (-2.0 * B)[:, None] * (np.diag(Wp) + W[:, None] * DB1))
    if m:
        L_A = L_A - (m ** 2) * np.diag(oa2 ** 2 / (4.0 * Asafe ** 2))
        C_B = C_B - (m ** 2) * np.diag(W / ob2)
    return L_A, C_B, W


# paper-v1 by name (the grid of the first submitted paper; production_model.SCHEME
# until 2026-10-14) and the 2026-10-10 abt+lobatto grid.
@pytest.mark.parametrize("kw", [{"scheme": "paper-v1"},
                                {"bmap": "abt", "radial": "lobatto"}],
                         ids=["paper-v1", "abt-lobatto"])
@pytest.mark.parametrize("Na,Nb", [(12, 10), (20, 14), (44, 32)])
def test_lobatto_operators_are_bit_identical_to_the_frozen_assembly(kw, Na, Nb):
    """paper-v1 (and the 2026-10-10 abt+lobatto grid) build exactly the old operator."""
    prob = s3.make_problem(Na, Nb, 4, **kw)
    assert prob.radial == "lobatto"
    A, B, DA1, DB1 = prob.A, prob.B, prob.DA1, prob.DB1
    A0, DA0 = ops.cheb01(Na)
    assert np.array_equal(A, A0) and np.array_equal(DA1, DA0)
    Lap_f, DA_f = _frozen_laplacian(A, B, DA1, DB1, 3.0)
    Lap, _, _, _, _, DA, _ = ops.laplacian_matrix(A, B, DA1, DB1, 3.0)
    assert np.array_equal(Lap, Lap_f) and np.array_equal(DA, DA_f)
    M2, i2 = ops.apply_bcs(Lap, A, B, DA)
    M2f, i2f = _frozen_bcs(Lap_f, A.size, B.size, DA_f, 0)
    assert np.array_equal(M2, M2f) and np.array_equal(i2, i2f)
    M0, w, interior = ops3.mode_operators(A, B, DA1, DB1, 3.0, [0, 1, 2])
    for mi, m in enumerate((0, 1, 2)):
        Mf, wf, intf = _frozen_block(A, B, DA1, DB1, 3.0, m)
        assert np.array_equal(M0[mi], Mf), m
        assert np.array_equal(w[mi], wf) and np.array_equal(interior, intf)
        for got, want in zip(separable.kron_factors(A, B, DA1, DB1, m),
                             _frozen_kron_factors(A, B, DA1, DB1, m)):
            assert np.array_equal(got, want), m


def test_schemes_and_their_resolution():
    assert ops.SCHEMES == {"paper-v1": ("cos", "lobatto"), "v2": ("abt", "gauss")}
    # milestone 2: v2 is the default (milestone 1 asserted "paper-v1" here)
    assert ops.DEFAULT_SCHEME == "v2"
    assert (ops.DEFAULT_BMAP, ops.DEFAULT_RADIAL) == ops.SCHEMES[ops.DEFAULT_SCHEME]
    assert ops.resolve_scheme() == ("abt", "gauss")
    assert ops.resolve_scheme("v2") == ("abt", "gauss")
    assert ops.resolve_scheme("v2", bmap="abt", radial="gauss") == ("abt", "gauss")
    assert ops.resolve_scheme("paper-v1", bmap="cos") == ("cos", "lobatto")   # agreeing axis
    # the 2026-10-10 opt-in is now spelled with both axes (bmap="abt" alone is refused,
    # test_a_lone_axis_is_refused_at_every_entry_point)
    assert ops.resolve_scheme(bmap="abt", radial="lobatto") == ("abt", "lobatto")
    assert ops.resolve_scheme(bmap="cos", radial="gauss") == ("cos", "gauss")  # mixed
    with pytest.raises(ValueError, match="scheme 'v2' has bmap"):
        ops.resolve_scheme("v2", bmap="cos")
    with pytest.raises(ValueError, match="scheme 'paper-v1' has radial"):
        ops.resolve_scheme("paper-v1", radial="gauss")
    with pytest.raises(ValueError, match="unknown scheme"):
        s3.make_problem(10, 8, 4, scheme="v3")
    with pytest.raises(ValueError, match="unknown radial map"):
        ops.build_grid(10, 8, "cos", "radau")
    p = s3.make_problem(10, 8, 4, scheme="v2")
    assert (p.bmap, p.radial, p.scheme) == ("abt", "gauss", "v2")
    assert s3.make_problem(10, 8, 4).scheme == "v2"
    assert s3.make_problem(10, 8, 4, scheme="paper-v1").scheme == "paper-v1"
    assert s3.make_problem(10, 8, 4, bmap="abt", radial="lobatto").scheme is None
    p2 = sa.make_problem(10, 8, scheme="v2")
    assert (p2.bmap, p2.radial, p2.scheme) == ("abt", "gauss", "v2")


# ==========================================================================
# 2.  The "v2" construction
# ==========================================================================
def test_gauss_radial_nodes():
    """ABT eq. (8): A_i = sin²[π(i+½)/(2K)], descending, neither edge a node."""
    for Na in (8, 20, 44):
        A, DA1 = ops.build_grid(Na, 6, "abt", "gauss")[0::2]
        K = Na + 1
        want = np.sort(np.sin(np.pi / (2 * K) * (np.arange(K) + 0.5)) ** 2)[::-1]
        assert np.array_equal(A, want)
        assert 0.0 < A.min() and A.max() < 1.0
        assert ops.radial_of(A, DA1) == "gauss" and ops.radial_of(ops.cheb01(Na)[0]) == "lobatto"
        # d/dA exact on polynomials of degree <= Na
        p = np.polynomial.Polynomial(np.random.default_rng(Na).standard_normal(K))
        err = np.max(np.abs(DA1 @ p(A) - p.deriv()(A))) / np.max(np.abs(p.deriv()(A)))
        assert err < 1e-11, err
    with pytest.raises(ValueError, match="hand-built radial"):
        ops.radial_of(A, DA1 * (1 + 1e-15))


def _poly_pair(Na, Nb, seed):
    rng = np.random.default_rng(seed)
    return (np.polynomial.Polynomial(rng.standard_normal(Na + 1)),
            np.polynomial.Polynomial(rng.standard_normal(Nb)))


def test_v2_operator_is_exact_on_its_discrete_space():
    """Lap @ V = Δ[(A−1) p(A) q(s)] for any p of degree ≤ Na and q of degree ≤ Nb−1.

    The analytic side differentiates u = (A−1)·p·q with numpy polynomials (∂_B by the
    chain rule through B(s)) and contracts with the prolate coefficients; the discrete
    side is the operator the solver uses.  This is the claim that the product rule of
    ``operators_abt.radial_derivs`` is right: replacing ``DAu1`` by the plain ``DA1``,
    or dropping the factor on the B-terms, misses by O(1).
    """
    Na, Nb, b = 14, 12, 2.0
    prob = sa.make_problem(Na, Nb, scheme="v2")
    A, s = prob.A, prob.Bs
    p, q = _poly_pair(Na, Nb, 3)
    Lap, _, _, Af, Bf, _, _ = ops.laplacian_matrix(prob.A, prob.B, prob.DA1, prob.DB1, b)
    got = Lap @ (p(A)[:, None] * q(s)[None, :]).ravel()
    ua = np.polynomial.Polynomial([-1.0, 1.0]) * p
    Bp, Bpp = ops.dB_ds(s, "abt"), ops.d2B_ds2(s, "abt")
    qB = q.deriv(1)(s) / Bp
    qBB = (q.deriv(2)(s) - Bpp * qB) / Bp ** 2
    alpha, pcoef, gamma, qcoef = ops._coeffs(Af, Bf, b)
    want = (alpha * np.outer(ua.deriv(2)(A), q(s)).ravel()
            + pcoef * np.outer(ua.deriv(1)(A), q(s)).ravel()
            + gamma * np.outer(ua(A), qBB).ravel() + qcoef * np.outer(ua(A), qB).ravel())
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    print(f"\n[v2 operator on (A-1)·p·q] rel err {rel:.1e}")
    assert rel < 1e-10


def test_v2_blocks_m0_oracle_and_factor():
    """m=0 block == the 2-D operator bit-for-bit; w = (A−1)·(1−B²)^{m/2}; no BC rows."""
    prob = s3.make_problem(**GRID, scheme="v2")
    M0, w, interior = ops3.mode_operators(prob.A, prob.B, prob.DA1, prob.DB1, 2.5, prob.m_vals)
    Lap, _, _, _, Bf, DA, _ = ops.laplacian_matrix(prob.A, prob.B, prob.DA1, prob.DB1, 2.5)
    M2, int2 = ops.apply_bcs(Lap, prob.A, prob.B, DA)
    assert np.array_equal(M0[0], M2) and interior.all() and int2.all()
    fAf = np.repeat(prob.A - 1.0, prob.Nb)
    for mi, m in enumerate(prob.m_vals):
        assert np.array_equal(w[mi], fAf * ops3.bc_factor(Bf, int(m))[0])
    asm = sa.assemble(sa.make_problem(GRID["Na"], GRID["Nb"], scheme="v2"), Slice(2.5, 0.6, 0.4))
    assert np.array_equal(asm.M0, M2) and np.array_equal(asm.w, fAf)


def test_v2_separable_matches_dense():
    """Equation (2) of ``separable``: apply, solve, both row scales, and a full solve."""
    prob = s3.make_problem(**GRID, scheme="v2")
    sl = _slice()
    asm = s3.assemble(prob, sl, separable=False)
    sep = separable.get_separable(prob.Na, prob.Nb, prob.Nphi, "abt", "gauss")
    assert sep.interior.all() and all(np.array_equal(a, b) for a, b in zip(sep.w_nodes, asm.w))
    rng = np.random.default_rng(0)
    for mi in range(prob.m_vals.size):
        V = rng.standard_normal((prob.Na + 1, prob.Nb))
        Y = asm.M0[mi] @ V.ravel()
        assert np.max(np.abs(sep.apply(mi, V, sl.b).ravel() - Y)) < 1e-11 * np.max(np.abs(Y))
        assert np.max(np.abs(sep.solve(mi, Y.reshape(V.shape), sl.b) - V)) < 1e-10
    scu = ops3.u_row_scales(asm.M0, asm.w)
    for got, want in ((sep.row_scales(sl.b), asm.scales), (sep.row_scales_u(sl.b), scu)):
        assert max(np.max(np.abs(g / x - 1.0)) for g, x in zip(got, want)) < 1e-10
    Ud, idn = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30, separable=False)
    Us, isp = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30, separable=True)
    assert idn.converged and isp.converged
    d = float(np.max(np.abs(Us - Ud)))
    print(f"\n[v2 separable vs dense] |dU| = {d:.1e}, gmres/step {isp.gmres_iters}")
    assert d < 1e-13


def test_v2_axisym_reduction_reproduces_2d():
    """Nφ=1 NK on the v2 grid IS the 2-D Newton on the same grid — operator, field and
    monitor.

    On "gauss" the 2-D Newton monitors the equilibrated residual in the u norm (the
    raw ``max|R|`` floors at ~1e-8 here: the innermost row sits at A = 7e-4 with row
    scale ~3e9), so it converges at 1e-12 and its number is the one the 3-D solver
    reports for the same field at Nφ=1.
    """
    P = 0.5
    for (b, mA, mB) in [(1.0, 0.5, 0.5), (1.5, 0.7, 0.3)]:
        prob2 = sa.make_problem(Na=28, Nb=20, P=P, scheme="v2")
        sl2 = Slice(b=b, m_A=mA, m_B=mB)
        U2, i2 = sa.newton_solve(prob2, sl2, tol=1e-12, max_iter=25)
        assert i2.converged and i2.norm == "u" and i2.residual_norm < 1e-14, i2
        prob3 = s3.make_problem(Na=28, Nb=20, Nphi=1, scheme="v2")
        sl3 = Slice3D.head_on(b=b, m_A=mA, m_B=mB, P=P)
        U3, i3 = nk.newton_solve_nk(prob3, sl3, tol=1e-12, max_iter=25)
        assert i3.converged
        # the 2-D monitor is the 3-D definition: same number for the same field —
        # checked on an UNconverged field (0.9·U), where the residual is O(1e-3) and
        # the two must agree to many digits (at convergence both are roundoff)
        Ub = 0.9 * np.asarray(U2)
        asm3 = s3.assemble(prob3, sl3, separable=False)
        r3 = nk.equil_residual_inf(asm3, Ub.reshape(-1, 1), norm="u")
        asm2 = sa.assemble(prob2, sl2)
        su = ops3.u_row_scales([asm2.M0], [asm2.w])[0]
        r2 = float(np.max(np.abs(sa.residual_vec(asm2, Ub.ravel())) / su))
        assert r2 > 1e-6 and abs(r2 - r3) <= 1e-12 * r2, (r2, r3)
        d = float(np.max(np.abs(np.asarray(U3)[:, :, 0] - np.asarray(U2))))
        print(f"\n[v2 axisym] b={b}: |U3-U2|={d:.2e}")
        assert d < 1e-12
        rho, z = np.array([0.7, 1.9, 0.2, 30.0]), np.array([0.3, -1.1, 0.4, 5.0])
        e3 = s3.evaluate_field(prob3, U3, rho, z, np.zeros(4), b)
        e2 = np.asarray(sa.evaluate_field_phys(prob2, U2, rho, z, b))
        e2v = constraints.evaluate_u_vec(prob2, U2, rho, z, b)
        assert np.max(np.abs(e3 - e2)) < 1e-12 and np.max(np.abs(e2v - e2)) < 1e-14


def test_v2_2d_path_certifies_under_its_own_gate():
    """The 2-D route's gate follows its norm: CERT_TOL (raw) on the paper's grid,
    CERT_TOL_U (equilibrated u) on "gauss" — for a direct solve and for a 2-D model,
    whose polish then certifies strictly at the gauss gate."""
    from lemaitre.initial_data.conformally_flat_puncture.parametric import parametric_nd_2c as p2c
    from lemaitre.initial_data.conformally_flat_puncture.parametric.certification import (
        CERT_TOL, CERT_TOL_U, gate_for, gate_for_problem)

    axes = [dict(name="b", min=1.0, max=1.6, Q=3)]
    for kw, gate, norm in ((dict(scheme="paper-v1"), CERT_TOL, "raw"),
                           (dict(scheme="v2"), CERT_TOL_U, "u")):
        prob = sa.make_problem(20, 14, P=0.3, **kw)
        assert gate_for_problem(prob) == gate
        solver = p2c.from_problem_nd(prob, axes, fixed={"q": 1.5})
        assert gate_for(solver.solve_fn) == gate
        model = solver.build(tol=1e-12, max_iter=25)
        U, info = model.evaluate_polished(np.array([1.37]), newton_steps=2, strict=True)
        assert info.converged and info.residual_norm <= gate, (kw, info.residual_norm)
        _, direct = sa.newton_solve(prob, p2c.theta_to_slice([1.37], ["b"], 1.0, {"q": 1.5}),
                                    tol=gate, max_iter=25)
        assert direct.converged and direct.norm == norm


def _discrete_field(prob, seed):
    """A field in the v2 discrete space: per mode (A−1)·p_m(A)·q_m(s), and its exact
    value at arbitrary points."""
    rng = np.random.default_rng(seed)
    terms = []
    for m in prob.m_vals:
        p, q = _poly_pair(prob.Na, prob.Nb, 100 * seed + int(m))
        c = complex(rng.standard_normal(), 0.0 if (m == 0 or 2 * m == prob.Nphi)
                    else rng.standard_normal())
        terms.append((int(m), p, q, c))

    def f(A, s, phi):
        out = 0.0
        for m, p, q, c in terms:
            wt = 1.0 if (m == 0 or 2 * m == prob.Nphi) else 2.0
            out = out + wt * np.real(c * np.exp(1j * m * phi)) * (A - 1.0) * p(A) * q(s)
        return out

    AA, SS, PP = np.meshgrid(prob.A, prob.Bs, prob.phi, indexing="ij")
    return f(AA, SS, PP), f


def test_v2_evaluate_field_is_exact_on_its_discrete_space():
    prob = s3.make_problem(**GRID, scheme="v2")
    U, f = _discrete_field(prob, 1)
    b = 2.5
    rng = np.random.default_rng(5)
    A = np.concatenate([rng.uniform(0.0, 0.999, 30), [0.0, 0.0, 0.9999]])
    s = np.concatenate([rng.uniform(-1, 1, 30), [1.0, -1.0, 0.3]])
    phi = rng.uniform(0, 2 * np.pi, A.size)
    rho = b * 2 * A / (1 - A ** 2) * (1 - s ** 2) / (1 + s ** 2)
    z = b * (1 + A ** 2) / (1 - A ** 2) * ops.b_of_s(s, "abt")
    got = s3.evaluate_field(prob, U, rho, z, phi, b)
    err = float(np.max(np.abs(got - f(A, s, phi))))
    print(f"\n[v2 evaluate_field on its discrete space] max err {err:.1e}")
    assert err < 1e-11
    # u = (A − 1)·V vanishes at A = 1 by construction — exactly, whatever V is
    assert s3.evaluate_field(prob, U, np.array([0.0]), np.array([1e200]), np.zeros(1), b)[0] == 0.0


def test_v2_edge_readouts_are_exact_on_its_discrete_space():
    """ADM mass (slope at A=1) and the puncture values (A=0, B=±1), with no node there.

    On u = (A−1) p(A) q(s): ∂_A u|_{A=1} = p(1) q(s) and u(A=0) = −p(0) q(s), so every
    reader must return exactly that — the 3-D and 2-D ADM masses, qc_targeting's, the
    puncture masses (numpy and jax), and qc_effpot's binding energy (numpy and jax).
    """
    import jax.numpy as jnp
    from lemaitre.initial_data.conformally_flat_puncture.applications import qc_effpot as E
    from lemaitre.initial_data.conformally_flat_puncture.applications import qc_targeting as qt
    from lemaitre.initial_data.conformally_flat_puncture.applications import sensitivity as sens

    Na, Nb, b = 16, 12, 3.0
    p, q = _poly_pair(Na, Nb, 7)
    p = p * 0.01
    prob = s3.make_problem(Na, Nb, 4, scheme="v2")
    U = np.repeat(((prob.A - 1.0) * p(prob.A))[:, None, None] * q(prob.Bs)[None, :, None], 4, axis=2)
    M_want = 1.0 - 2.0 * b * p(1.0) * np.mean(q(prob.Bs))
    sl = Slice3D(b=b, m_A=0.5, m_B=0.5)
    assert abs(d3.adm_mass_spectral_3d(prob, U, sl) - M_want) < 1e-13
    assert abs(qt.M_ADM(prob, U, b) - M_want) < 1e-13
    uA, uB = -p(0.0) * q(1.0), -p(0.0) * q(-1.0)
    assert abs(E._u_at_puncture(prob, U, "A") - uA) < 1e-13
    assert abs(E._u_at_puncture(prob, U, "B") - uB) < 1e-13
    Eb = E.binding_energy(prob, U, b)

    class _Model:
        def evaluate_jax(self, theta):
            return jnp.asarray(U)

    assert abs(float(E.build_Eb_jax(_Model(), prob)(jnp.array([b, 0.0]))) - Eb) < 1e-13

    prob2 = sa.make_problem(Na, Nb, scheme="v2")
    U2 = (prob2.A - 1.0)[:, None] * p(prob2.A)[:, None] * q(prob2.Bs)[None, :]
    sl2 = Slice(b=b, m_A=0.5, m_B=0.5)
    assert abs(adm.adm_mass_spectral(prob2, U2, sl2) - M_want) < 1e-13
    assert abs(adm.puncture_adm_mass(prob2, U2, sl2, "A") - 0.5 * (1 + uA + 0.5 / (4 * b))) < 1e-13
    wB = sa._bary_weights(prob2.Bs)
    for which, want in (("A", uA), ("B", uB)):
        assert abs(float(sens._u_at_puncture_jax(jnp.asarray(U2), prob2, wB, which)) - want) < 1e-13
    obs = sens._make_observables_jax(prob2)
    assert abs(float(obs["M_ADM"](jnp.asarray(U2), {"b": b, "M": 1.0})) - M_want) < 1e-13


def test_mixed_grids_stay_selectable():
    """cos + gauss and abt + lobatto build and solve; their separable route matches dense."""
    sl = _slice()
    for kw in (dict(bmap="cos", radial="gauss"), dict(bmap="abt", radial="lobatto")):
        prob = s3.make_problem(16, 12, 4, **kw)
        Ud, i_d = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30, separable=False)
        Us, i_s = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30, separable=True)
        assert i_d.converged and i_s.converged
        assert float(np.max(np.abs(Us - Ud))) < 1e-12, kw


# ==========================================================================
# 3.  The point: the ADM mass, and ψ not worse anywhere
# ==========================================================================
def _probes(sl, seed=0):
    """near (0.02–0.5 m of a hole), mid (0.3–3 b), far (5–200 b)."""
    rng = np.random.default_rng(seed)

    def sph(n, rmin, rmax, zc):
        r = np.exp(rng.uniform(np.log(rmin), np.log(rmax), n))
        ct, ph = rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n)
        st = np.sqrt(1 - ct ** 2)
        x, y, zz = r * st * np.cos(ph), r * st * np.sin(ph), zc + r * ct
        return np.hypot(x, y), zz, np.arctan2(y, x)

    nA, nB = sph(20, 0.02 * sl.m_A, 0.5 * sl.m_A, sl.b), sph(20, 0.02 * sl.m_B, 0.5 * sl.m_B, -sl.b)
    return {"near": tuple(np.concatenate([nA[i], nB[i]]) for i in range(3)),
            "mid": sph(60, 0.3 * sl.b, 3.0 * sl.b, 0.0), "far": sph(40, 5.0 * sl.b, 200.0 * sl.b, 0.0)}


@pytest.fixture(scope="module")
def production_qc10():
    """b = 10, q = 1.5 quasi-circular, spins (0.3, −0.2): four arms at 44×32×8 and two
    references (v2 64×96×8; abt+lobatto 80×96×8 — a different radial form, so their
    agreement checks the reference)."""
    sl = p3.theta_to_slice3d([10.0, 1.5, 0.3, -0.2], ["b", "q", "chi_Ay", "chi_By"], 1.0,
                             {"qc": 1.0})
    P = _probes(sl)

    def run(Na, Nb, **kw):
        prob = s3.make_problem(Na, Nb, 8, **kw)
        U, info = nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=40)
        assert info.converged
        u0 = np.asarray(U).mean(axis=2)
        rows = ops.radial_edge_rows(prob.A)
        c = -sl.b * ((prob.DA1 @ u0)[0, :] if rows is None else rows[0] @ u0)
        psi = {k: np.asarray(source.psi_BL_2c(r, z, sl.b, sl.m_A, sl.m_B))
               + s3.evaluate_field(prob, U, r, z, ph, sl.b) for k, (r, z, ph) in P.items()}
        return dict(M=d3.adm_mass_spectral_3d(prob, U, sl), c=c, psi=psi)

    # the references name their axes, so they do not move if a scheme's definition does
    return dict(ref=run(64, 96, bmap="abt", radial="gauss"),
                ref_lob=run(80, 96, bmap="abt", radial="lobatto"),
                v1=run(44, 32, scheme="paper-v1"), abt_lob=run(44, 32, bmap="abt", radial="lobatto"),
                v2=run(44, 32, scheme="v2"))


def test_v2_reads_the_adm_mass_far_better_at_the_production_grid(production_qc10):
    """|ΔM_ADM| at 44×32×8, b = 10: paper-v1 ≥ 100× v2.

    Measured 2026-10-11: paper-v1 6.7e-5, abt+lobatto 1.0e-6, v2 2.2e-9 (the radial
    form alone, abt+lobatto → v2, is ~500×).  The two references agree to ≤ 1e-10.
    """
    R = production_qc10
    agree = abs(R["ref"]["M"] - R["ref_lob"]["M"])
    dM = {k: abs(R[k]["M"] - R["ref"]["M"]) for k in ("v1", "abt_lob", "v2")}
    print(f"\n[M_ADM @44x32x8, b=10] |dM|: paper-v1 {dM['v1']:.1e}, abt+lobatto "
          f"{dM['abt_lob']:.1e}, v2 {dM['v2']:.1e}; references agree to {agree:.1e}")
    assert agree < 1e-10, "the references disagree: the comparison is not valid"
    assert dM["v2"] * 100.0 < dM["v1"]
    assert dM["v2"] * 30.0 < dM["abt_lob"]
    assert dM["v2"] < 1e-8


def test_v2_slope_at_infinity_is_isotropic(production_qc10):
    """c(B) = −b ∂_A u|_{A=1} is B-independent for the exact solution (it is the ADM
    mass): a reference-free measure of the radial representation.  v2's spread over
    the B nodes is far below abt+lobatto's on the same angular grid.  It is NOT a
    measure of the mass itself: cos+gauss has a spread as small as v2's and the mass
    error of paper-v1, because there the angular discretisation shifts the mean."""
    R = production_qc10
    sp = {k: float(np.ptp(R[k]["c"])) for k in ("ref", "abt_lob", "v2")}
    print(f"\n[c(B) spread @44x32x8, b=10] abt+lobatto {sp['abt_lob']:.1e}, v2 {sp['v2']:.1e} "
          f"(ref {sp['ref']:.1e})")
    assert sp["v2"] * 30.0 < sp["abt_lob"]           # measured 8.4e-9 vs 6.5e-6 (770×)
    assert sp["v2"] < 3e-8


def test_v2_psi_is_not_worse_than_abt_lobatto_anywhere(production_qc10):
    """near / mid / far ψ against the reference: v2 within 1.5× of abt+lobatto in each
    region (they share the angular grid, so near and mid are B-limited and tie; far is
    where the radial form shows)."""
    R = production_qc10
    for region in ("near", "mid", "far"):
        e = {k: float(np.max(np.abs(R[k]["psi"][region] - R["ref"]["psi"][region])))
             for k in ("abt_lob", "v2", "ref_lob")}
        print(f"\n[psi {region} @44x32x8, b=10] abt+lobatto {e['abt_lob']:.1e}, v2 {e['v2']:.1e}; "
              f"references agree to {e['ref_lob']:.1e}")
        assert e["ref_lob"] < 0.1 * e["v2"] or e["ref_lob"] < 1e-12
        assert e["v2"] <= 1.5 * e["abt_lob"], region


# ==========================================================================
# 4.  The radial map is part of the grid's identity
# ==========================================================================
def test_solve_store_keys():
    """paper-v1 and abt keys unchanged; a gauss key differs from every one of them."""
    sl = _slice()
    old_cos = ss.slice_key(sl, (20, 14, 6), "t")
    old_abt = ss.slice_key(sl, (20, 14, 6), "t", "abt")
    assert ss.slice_key(sl, (20, 14, 6), "t", "cos", "lobatto") == old_cos
    assert ss.slice_key(sl, (20, 14, 6), "t", "abt", "lobatto") == old_abt
    v2 = ss.slice_key(sl, (20, 14, 6), "t", "abt", "gauss")
    cg = ss.slice_key(sl, (20, 14, 6), "t", "cos", "gauss")
    assert len({old_cos, old_abt, v2, cg}) == 4
    assert "radial" not in ss.slice_payload(sl, (20, 14, 6), "t", "abt")
    assert ss.slice_payload(sl, (20, 14, 6), "t", "abt", "gauss")["radial"] == "gauss"


def test_solve_store_refuses_a_problem_on_the_other_radial_map(tmp_path):
    store = ss.SolveStore(str(tmp_path), (10, 8, 4), code_tag="t", bmap="abt")
    assert store.radial == "lobatto"
    with pytest.raises(ValueError, match="radial grid"):
        ss._as_store(store, s3.make_problem(10, 8, 4, scheme="v2"))
    built = ss._as_store(str(tmp_path / "v2"), s3.make_problem(10, 8, 4, scheme="v2"))
    assert (built.bmap, built.radial) == ("abt", "gauss")


def _tiny_model(bmap, radial=None):
    def solve_fn(*a, **k):                                  # pragma: no cover
        raise AssertionError("not called")
    solve_fn.bmap = bmap
    if radial is not None:
        solve_fn.radial = radial
    return pnd.ParametricSolutionND(
        axes=[(1.0, 2.0, 1)], nodes=[np.array([2.0, 1.0])], weights=[np.array([0.5, -0.5])],
        U_nodes=np.zeros((2, 3, 2, 1)), iters=np.zeros(2, dtype=int),
        residuals=np.zeros(2), _solve_fn=solve_fn)


def test_saved_models_record_the_radial_map_and_attaching_checks_it(tmp_path):
    p_v1 = _tiny_model("cos", "lobatto").save(str(tmp_path / "v1.npz"), meta={"Na": 2})
    p_old = _tiny_model("cos").save(str(tmp_path / "old.npz"), meta={"Na": 2})
    p_v2 = _tiny_model("abt", "gauss").save(str(tmp_path / "v2.npz"), meta={"Na": 2})
    m_v1, m_v2 = pnd._unpack_meta(np.load(p_v1)), pnd._unpack_meta(np.load(p_v2))
    assert "radial" not in m_v1 and "bmap" not in m_v1, "a paper-v1 artifact must not change"
    assert pnd._unpack_meta(np.load(p_old)) == m_v1
    assert (m_v2["bmap"], m_v2["radial"]) == ("abt", "gauss")
    with pytest.raises(ValueError, match="says radial"):
        _tiny_model("abt", "gauss").save(str(tmp_path / "x.npz"), meta={"radial": "lobatto"})
    mod_v1, mod_v2 = pnd.load_parametric(p_v1), pnd.load_parametric(p_v2)
    prob_v1 = s3.make_problem(10, 8, 4, scheme="paper-v1")
    prob_v2 = s3.make_problem(10, 8, 4, scheme="v2")
    pnd.attach_solve_fn_3d(mod_v1, prob_v1, ["b"])
    pnd.attach_solve_fn_3d(mod_v2, prob_v2, ["b"])
    assert mod_v2._solve_fn.radial == "gauss" and mod_v1._solve_fn.radial == "lobatto"
    with pytest.raises(ValueError, match="grid"):
        pnd.attach_solve_fn_3d(pnd.load_parametric(p_v1), prob_v2, ["b"])
    with pytest.raises(ValueError, match="grid"):
        pnd.attach_solve_fn_3d(pnd.load_parametric(p_v2), prob_v1, ["b"])
    # same angular map, other radial map: refused by the radial check itself
    with pytest.raises(ValueError, match="radial grid"):
        pnd.attach_solve_fn_3d(pnd.load_parametric(p_v2),
                               s3.make_problem(10, 8, 4, bmap="abt", radial="lobatto"), ["b"])


def test_export_has_a_format_per_scheme_and_refuses_mixed_grids():
    # was test_export_refuses_everything_but_paper_v1, before format 4 carried v2
    for scheme, fmt in (("paper-v1", 2), ("v2", 4)):
        prob = s3.make_problem(**GRID, scheme=scheme)
        assert ex.from_solution(prob, _slice(), np.zeros(prob.shape)).fmt == fmt
    for kw in (dict(bmap="cos", radial="gauss"), dict(bmap="abt", radial="lobatto")):
        prob = s3.make_problem(**GRID, **kw)
        with pytest.raises(NotImplementedError, match="mixed grid"):
            ex.from_solution(prob, _slice(), np.zeros(prob.shape))


def test_separable_assembly_refuses_a_hand_built_radial_grid():
    prob = s3.make_problem(**GRID, scheme="v2")
    A, DA1 = ops.cheb01(GRID["Na"])
    bad = dataclasses.replace(prob, A=A, DA1=DA1)          # lobatto nodes, says gauss
    with pytest.raises(ValueError, match="canonical"):
        s3.assemble(bad, _slice(), separable=True)


# ==========================================================================
# 5.  Milestone 2: "v2" is the default; the paper's grid is named, never defaulted
# ==========================================================================
_ENTRY_POINTS = {
    "build_grid": lambda **kw: ops.build_grid(10, 8, **kw),
    "build_grid_3d": lambda **kw: ops3.build_grid_3d(10, 8, 4, **kw),
    "get_separable": lambda **kw: separable.get_separable(10, 8, 4, **kw),
    "SeparableModes": lambda **kw: separable.SeparableModes(10, 8, 4, **kw),
    "solver_3d.make_problem": lambda **kw: s3.make_problem(10, 8, 4, **kw),
    "solver_abt.make_problem": lambda **kw: sa.make_problem(10, 8, **kw),
}


def _grid_arrays(obj):
    if isinstance(obj, tuple):
        return [np.asarray(x) for x in obj]
    return [np.asarray(getattr(obj, k)) for k in ("A", "B", "DA1", "DB1")]


@pytest.mark.parametrize("name", list(_ENTRY_POINTS))
def test_the_default_is_v2_at_every_entry_point(name):
    """Nothing named builds exactly what ``scheme="v2"`` builds — and not paper-v1."""
    f = _ENTRY_POINTS[name]
    got, v2, v1 = f(), f(scheme="v2"), f(scheme="paper-v1")
    assert all(np.array_equal(x, y) for x, y in zip(_grid_arrays(got), _grid_arrays(v2)))
    assert not np.array_equal(_grid_arrays(got)[0], _grid_arrays(v1)[0])     # A nodes
    assert not np.array_equal(_grid_arrays(got)[1], _grid_arrays(v1)[1])     # B nodes
    if hasattr(got, "radial"):
        assert (got.bmap, got.radial) == ("abt", "gauss")
    if name == "get_separable":                     # one cache entry per resolved pair
        assert got is v2 is separable.get_separable(10, 8, 4, bmap="abt", radial="gauss")


def test_the_default_is_read_when_the_grid_is_built(monkeypatch):
    """``resolve_scheme`` reads ``DEFAULT_SCHEME`` at call time, so a caller that names
    no grid follows it — which is why everything meaning the paper's grid names it."""
    monkeypatch.setattr(ops, "DEFAULT_SCHEME", "paper-v1")
    assert s3.make_problem(10, 8, 4).scheme == "paper-v1"
    monkeypatch.setattr(ops, "DEFAULT_SCHEME", "v2")
    assert s3.make_problem(10, 8, 4).scheme == "v2"


@pytest.mark.parametrize("name", list(_ENTRY_POINTS) + ["resolve_scheme"])
@pytest.mark.parametrize("lone", [dict(bmap="cos"), dict(bmap="abt"),
                                  dict(radial="lobatto"), dict(radial="gauss")],
                         ids=["bmap=cos", "bmap=abt", "radial=lobatto", "radial=gauss"])
def test_a_lone_axis_is_refused_at_every_entry_point(name, lone):
    """Decided 2026-10-11: name ``scheme=`` or both axes.  Without it,
    ``bmap="cos"`` alone would silently build cos + gauss under the v2 default."""
    f = ops.resolve_scheme if name == "resolve_scheme" else _ENTRY_POINTS[name]
    with pytest.raises(ValueError, match="alone is ambiguous"):
        f(**lone)


def test_a_single_axis_has_no_default_of_its_own():
    """``check_bmap``/``check_radial`` validate a name; ``None`` is refused there, so no
    code path can default one half of a grid behind ``resolve_scheme``'s back."""
    with pytest.raises(ValueError, match="no default on its own"):
        ops.check_bmap(None)
    with pytest.raises(ValueError, match="no default on its own"):
        ops.check_radial(None)
    assert ops.check_bmap("abt") == "abt" and ops.check_radial("gauss") == "gauss"


def test_the_paper_constant_is_v2():
    """One constant names the grid every shipped model, figure, table and the GRTeclyn
    export is on; the export's own format constant agrees; the module stays stdlib.

    Was ``test_the_paper_constant_is_paper_v1`` until the paper moved to v2 (2026-10-14);
    the constant now equals the solver default, but the call sites still name it."""
    import ast
    assert pm.SCHEME == "v2" and pm.SCHEME in ops.SCHEMES
    assert pm.SCHEME == ops.DEFAULT_SCHEME
    assert ex.FORMAT_OF_SCHEME == {"paper-v1": 2, "v2": 4}      # each grid its format
    tree = ast.parse(open(pm.__file__).read())
    mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any("solver" in m or m.split(".")[0] in ("jax", "numpy") for m in mods), mods


# (name -> positional index of bmap, radial, scheme), as in the signatures
_GRID_CALLS = {"make_problem": (3, 4, 5), "build_grid": (2, 3, 4), "build_grid_3d": (3, 4, 5),
               "get_separable": (3, 4, 5), "SeparableModes": (3, 4, 5)}


def _calls_relying_on_the_default(path):
    """Every grid-building call in ``path`` that names neither ``scheme=`` nor both axes
    (a ``**kwargs`` or ``*args`` call cannot be verified and is reported too)."""
    import ast
    bad, n = [], 0
    for node in ast.walk(ast.parse(open(path).read())):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name not in _GRID_CALLS:
            continue
        n += 1
        bi, ri, si = _GRID_CALLS[name]
        kws = {k.arg for k in node.keywords}
        npos = len(node.args)
        starred = None in kws or any(isinstance(a, ast.Starred) for a in node.args)
        named = ("scheme" in kws or npos > si
                 or (("bmap" in kws or npos > bi) and ("radial" in kws or npos > ri)))
        if starred and path.endswith(("operators_3d.py", "separable.py")) and name in (
                "build_grid", "SeparableModes"):
            named = True          # the two memo keys: (Na, Nb[, Nphi], bmap, radial) tuples
        if not named:
            bad.append(f"{path}:{node.lineno} {name}(...)")
    return bad, n


def test_no_producer_figure_table_or_oracle_relies_on_the_default():
    """Source scan: in the package (``pipeline/`` included), ``paper/`` and ``oracle/``,
    every ``make_problem``/``build_grid``/``build_grid_3d``/``get_separable``/
    ``SeparableModes`` call names its grid.  A producer that relied on the default
    would silently move from the paper's grid to v2 — and a figure, table or the export
    would follow — so this is checked on the source, not by running them."""
    import os
    import lemaitre.initial_data.conformally_flat_puncture as pkg
    pkg_dir = os.path.dirname(pkg.__file__)
    leaf = os.path.abspath(os.path.join(pkg_dir, *[".."] * 4))
    roots = [pkg_dir, os.path.join(leaf, "paper"), os.path.join(leaf, "oracle")]
    assert all(os.path.isdir(r) for r in roots), roots
    bad, n, files = [], 0, 0
    for root in roots:
        for d, _, fs in os.walk(root):
            for fn in fs:
                if fn.endswith(".py"):
                    b, k = _calls_relying_on_the_default(os.path.join(d, fn))
                    bad += b
                    n += k
                    files += 1
    print(f"\n[source scan] {files} files, {n} grid-building calls, {len(bad)} unnamed")
    assert n >= 40, f"the scan saw only {n} calls: it is not looking where the producers are"
    assert not bad, "grid built on the default (name scheme= or both axes):\n" + "\n".join(bad)
