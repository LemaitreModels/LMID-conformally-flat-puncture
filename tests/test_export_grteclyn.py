"""Step 2 of docs/GRTECLYN_CONSTRAINTS_PLAN.md — the exporter and its reference
evaluator.

The exported file is the interface to an external evolution code, so the format
has to be pinned *here*, independently of any C++, before anything consumes it.
Two things are gated:

  * the phi representation is an exact rewrite, not an approximation — the
    cos/sin series reproduces ``solver_3d._fourier_interp`` to machine
    precision, including at the Nyquist wavenumber;
  * the numpy-only reference evaluator reproduces the solver's own
    ``evaluate_field`` (+ ``psi_BL``, + the closed-form Bowen-York ``Ahat``) to
    machine precision from the *written file alone*.

The second is what makes the reference table a valid target for the C++ port:
if the port matches the table, it matches the solver.

Most gates use a *manufactured* random ``U`` rather than a converged solve — the
identity being tested is the interpolation/serialisation chain, which has nothing
to do with whether ``U`` solves anything, and this keeps the test fast.  One
solver-backed test covers the ``from_solution`` wiring.
"""

import os

import numpy as np
import pytest

from lm.initial_data.solver import operators_abt as ops
from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.solver import source, source_3d
from lm.initial_data.validation import export_grteclyn as eg


B_ANCHOR = 3.0
M_A = M_B = 0.5


def _manufactured(Na=20, Nb=14, Nphi=8, seed=3):
    """A small problem plus a smooth random nodal ``U`` (not a solution)."""
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    rng = np.random.default_rng(seed)
    U = rng.normal(size=prob.shape) * 1e-2
    sl = s3.Slice3D(b=B_ANCHOR, m_A=M_A, m_B=M_B,
                    P_A_vec=(0.1, -0.2, -0.5), P_B_vec=(-0.1, 0.2, 0.5),
                    S_A_vec=(0.0, 0.05, 0.02), S_B_vec=(0.03, 0.0, -0.04))
    return prob, sl, U


def _sample_points(b=B_ANCHOR, n=40, half_width=9.0, seed=11):
    """Random Cartesian points with the punctures avoided."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-half_width, half_width, size=3)
        rA = np.linalg.norm(p - np.array([0.0, 0.0, b]))
        rB = np.linalg.norm(p - np.array([0.0, 0.0, -b]))
        if min(rA, rB) > 0.5:
            pts.append(p)
    return np.array(pts)


# ==========================================================================
# The phi representation is exact
# ==========================================================================
@pytest.mark.parametrize("nphi", [1, 2, 4, 8, 9, 16])
def test_phi_series_reproduces_fourier_interp(nphi):
    """cos/sin series == solver_3d._fourier_interp, to machine precision."""
    rng = np.random.default_rng(nphi)
    vals = rng.normal(size=nphi)
    cos_c, sin_c = eg.phi_modes(vals)
    cos_m, sin_m = eg.phi_mode_layout(nphi)

    # arbitrary phi, plus the collocation nodes themselves
    phis = np.concatenate([rng.uniform(0.0, 2.0 * np.pi, size=17),
                           2.0 * np.pi * np.arange(nphi) / nphi])
    for phi in phis:
        got = eg.phi_eval(cos_c, sin_c, cos_m, sin_m, phi)
        want = s3._fourier_interp(vals, float(phi))
        assert abs(float(got) - want) < 1e-13, (nphi, phi)

    # and it interpolates: at the nodes it returns the samples
    for k in range(nphi):
        phi_k = 2.0 * np.pi * k / nphi
        got = float(eg.phi_eval(cos_c, sin_c, cos_m, sin_m, phi_k))
        assert abs(got - vals[k]) < 1e-13


def test_phi_mode_layout_nyquist():
    """Even Nphi carries a Nyquist cosine but no Nyquist sine.

    The sine at wavenumber Nphi/2 vanishes at every collocation node, so it is
    not determined by the samples; shipping one would be a free parameter that
    the consumer and the solver could disagree about.
    """
    cos_m, sin_m = eg.phi_mode_layout(8)
    assert list(cos_m) == [0, 1, 2, 3, 4]
    assert list(sin_m) == [1, 2, 3]
    cos_m, sin_m = eg.phi_mode_layout(9)
    assert list(cos_m) == [0, 1, 2, 3, 4]
    assert list(sin_m) == [1, 2, 3, 4]
    # total real degrees of freedom == number of samples
    for nphi in (1, 2, 4, 8, 9, 16):
        c, s = eg.phi_mode_layout(nphi)
        assert c.size + s.size == nphi, nphi


# ==========================================================================
# The file format round-trips
# ==========================================================================
def test_write_read_roundtrip(tmp_path):
    prob, sl, U = _manufactured()
    exp = eg.from_solution(prob, sl, U, residual=1.6e-10, raw_residual=3e-9,
                           note="unit test")
    path = str(tmp_path / "slice.lmid")
    exp.write(path)
    got = eg.Export.read(path)

    for k in ("b", "m_A", "m_B"):
        assert getattr(got, k) == pytest.approx(getattr(exp, k), rel=0, abs=0)
    for k in ("P_A", "P_B", "S_A", "S_B"):
        assert np.allclose(getattr(got, k), getattr(exp, k), rtol=0, atol=0)
    assert (got.Na, got.Nb, got.Nphi) == (exp.Na, exp.Nb, exp.Nphi)
    for k in ("A", "B", "cos_m", "sin_m", "C", "S"):
        assert np.array_equal(getattr(got, k), getattr(exp, k)), k
    # provenance survives and records the certified residual
    assert any("newton_residual_equilibrated_inf" in p for p in got.provenance)


def test_read_rejects_wrong_format(tmp_path):
    prob, sl, U = _manufactured()
    path = str(tmp_path / "bad.lmid")
    eg.from_solution(prob, sl, U).write(path)
    txt = open(path).read().replace("format 1", "format 99")
    open(path, "w").write(txt)
    with pytest.raises(ValueError):
        eg.Export.read(path)


# ==========================================================================
# The reference evaluator reproduces the solver, from the file alone
# ==========================================================================
def test_reference_evaluator_matches_solver(tmp_path):
    """eval_u (numpy, from file) == solver_3d.evaluate_field, machine precision."""
    prob, sl, U = _manufactured()
    path = str(tmp_path / "slice.lmid")
    eg.from_solution(prob, sl, U).write(path)
    exp = eg.Export.read(path)

    P = _sample_points(b=sl.b)
    rho = np.hypot(P[:, 0], P[:, 1])
    phi = np.arctan2(P[:, 1], P[:, 0])
    want = s3.evaluate_field(prob, U, rho, P[:, 2], phi, sl.b)
    got = eg.eval_u(exp, P[:, 0], P[:, 1], P[:, 2])
    assert np.max(np.abs(got - want)) < 1e-12 * max(1.0,
                                                    np.max(np.abs(want)))


def test_inverse_map_matches_solver():
    """The transcribed inverse ABT map == operators_abt.inverse_map."""
    P = _sample_points()
    rho = np.hypot(P[:, 0], P[:, 1])
    A1, B1 = eg.inverse_abt(rho, P[:, 2], B_ANCHOR)
    A2, B2 = ops.inverse_map(rho, P[:, 2], B_ANCHOR)
    assert np.max(np.abs(A1 - A2)) < 1e-15
    assert np.max(np.abs(B1 - B2)) < 1e-15


def test_psi_BL_and_Ahat_match_solver(tmp_path):
    """The closed forms carried by the evaluator == the solver's own."""
    prob, sl, U = _manufactured()
    path = str(tmp_path / "slice.lmid")
    eg.from_solution(prob, sl, U).write(path)
    exp = eg.Export.read(path)

    P = _sample_points(b=sl.b)
    rho = np.hypot(P[:, 0], P[:, 1])

    want_psi = np.asarray(source.psi_BL_2c(rho, P[:, 2], sl.b, sl.m_A, sl.m_B))
    got_psi = eg.eval_psi_BL(exp, P[:, 0], P[:, 1], P[:, 2])
    assert np.max(np.abs(got_psi - want_psi)) < 1e-14

    want_A = source_3d.A_full_tensor_vec(P, sl.b, sl.P_A_vec, sl.P_B_vec,
                                        sl.S_A_vec, sl.S_B_vec)
    got_A = eg.eval_Ahat(exp, P[:, 0], P[:, 1], P[:, 2])
    scale = max(1.0, float(np.max(np.abs(want_A))))
    assert np.max(np.abs(got_A - want_A)) < 1e-12 * scale
    # Ahat is symmetric and trace-free (the Bowen-York property the evolution
    # code's make_trace_free would otherwise silently repair)
    assert np.max(np.abs(got_A - np.swapaxes(got_A, 1, 2))) < 1e-14
    assert np.max(np.abs(np.trace(got_A, axis1=1, axis2=2))) < 1e-12 * scale


def test_full_psi_is_BL_plus_u(tmp_path):
    prob, sl, U = _manufactured()
    path = str(tmp_path / "slice.lmid")
    eg.from_solution(prob, sl, U).write(path)
    exp = eg.Export.read(path)
    P = _sample_points(b=sl.b)
    lhs = eg.eval_psi(exp, P[:, 0], P[:, 1], P[:, 2])
    rhs = (eg.eval_psi_BL(exp, P[:, 0], P[:, 1], P[:, 2])
           + eg.eval_u(exp, P[:, 0], P[:, 1], P[:, 2]))
    assert np.max(np.abs(lhs - rhs)) == 0.0


def test_evaluator_exact_at_nodes(tmp_path):
    """At a grid node the barycentric rule must return the nodal value exactly.

    This is the case a naive ``w/(xq-x)`` implementation divides by zero on, and
    the one a C++ port is most likely to get wrong.
    """
    prob, sl, U = _manufactured()
    path = str(tmp_path / "slice.lmid")
    eg.from_solution(prob, sl, U).write(path)
    exp = eg.Export.read(path)

    # pick interior nodes (A=1 is spatial infinity, A=0 the axis edge)
    for i in (3, 7, 12):
        for j in (2, 6, 10):
            rho, z = ops.abt_map(exp.A[i], exp.B[j], exp.b)
            for p in (0, 2, 5):
                phi = 2.0 * np.pi * p / exp.Nphi
                x, y = rho * np.cos(phi), rho * np.sin(phi)
                got = float(eg.eval_u(exp, x, y, z)[0])
                assert abs(got - U[i, j, p]) < 1e-11, (i, j, p, got,
                                                       U[i, j, p])


def test_reference_table(tmp_path):
    """The C++-port target table is well formed and reproducible."""
    prob, sl, U = _manufactured()
    exp = eg.from_solution(prob, sl, U)
    path = str(tmp_path / "ref.dat")
    eg.dump_reference_table(exp, path, n=16, half_width=6.0)
    tab = np.loadtxt(path)
    assert tab.shape == (16, 10)
    # column 3 is psi > 1 everywhere outside the punctures
    assert np.all(tab[:, 3] > 1.0)
    # re-evaluating the table's own points reproduces it
    psi = eg.eval_psi(exp, tab[:, 0], tab[:, 1], tab[:, 2])
    assert np.max(np.abs(psi - tab[:, 3])) < 1e-14


# ==========================================================================
# Standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lm.initial_data.validation.export_grteclyn as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm",
                      "import context", "torch"):
        assert forbidden not in src, forbidden
    # the evaluator half must not need jax: the solver import is deliberately
    # deferred into solve_and_export so that reading a file is numpy-only
    head = src.split("def solve_and_export")[0]
    assert "jax" not in head


# ==========================================================================
# Solver-backed wiring (slow)
# ==========================================================================
@pytest.mark.slow
def test_solve_and_export_wiring(tmp_path):
    """A real converged solve exports and re-evaluates consistently."""
    path = str(tmp_path / "anchor.lmid")
    exp = eg.solve_and_export(B_ANCHOR, M_A, M_B,
                              P_A=(0.0, 0.0, -0.5), P_B=(0.0, 0.0, 0.5),
                              Na=28, Nb=20, Nphi=8, path=path,
                              note="test_solve_and_export_wiring")
    assert os.path.exists(path)
    got = eg.Export.read(path)
    P = _sample_points(b=exp.b, n=24)
    a = eg.eval_psi(exp, P[:, 0], P[:, 1], P[:, 2])
    c = eg.eval_psi(got, P[:, 0], P[:, 1], P[:, 2])
    assert np.max(np.abs(a - c)) < 1e-14
    # the axisymmetric head-on slice has no sine content and no |m|>0 cosine
    # content beyond roundoff: a genuine physics check on the phi transform
    assert np.max(np.abs(exp.S)) < 1e-9
    assert np.max(np.abs(exp.C[:, :, 1:])) < 1e-9
