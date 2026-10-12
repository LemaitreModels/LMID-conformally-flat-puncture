"""Step 2 of the GRTeclyn constraint check — the exporter and its reference
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

Two formats, one per discretisation (``eg.FORMAT_OF_SCHEME``): format 2 for
``"paper-v1"`` and format 4 for ``"v2"`` (the ABT angle ``s`` and TwoPunctures'
``u = (A - 1) V``).  The chain gates run on both; a mixed grid has no format.

Most gates use a *manufactured* random ``U`` rather than a converged solve — the
identity being tested is the interpolation/serialisation chain, which has nothing
to do with whether ``U`` solves anything, and this keeps the test fast.  One
solver-backed test covers the ``from_solution`` wiring.
"""

import os

import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.solver import operators_abt as ops
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3
from lemaitre.initial_data.conformally_flat_puncture.solver import source, source_3d
from lemaitre.initial_data.conformally_flat_puncture.validation import export_grteclyn as eg


B_ANCHOR = 3.0
M_A = M_B = 0.5


SCHEMES = ("paper-v1", "v2")


def _manufactured(Na=20, Nb=14, Nphi=8, seed=3, scheme="paper-v1"):
    """A small problem plus a smooth random nodal ``U`` (not a solution)."""
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi, scheme=scheme)
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
@pytest.mark.parametrize("scheme", SCHEMES)
def test_write_read_roundtrip(tmp_path, scheme):
    prob, sl, U = _manufactured(scheme=scheme)
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


@pytest.mark.parametrize("scheme", SCHEMES)
def test_read_rejects_wrong_format(tmp_path, scheme):
    prob, sl, U = _manufactured(scheme=scheme)
    path = str(tmp_path / "bad.lmid")
    eg.from_solution(prob, sl, U).write(path)
    fmt = eg.FORMAT_OF_SCHEME[scheme]
    txt = open(path).read()
    assert f"\nformat {fmt}\n" in txt
    txt = txt.replace(f"\nformat {fmt}\n", "\nformat 99\n")
    open(path, "w").write(txt)
    with pytest.raises(ValueError):
        eg.Export.read(path)


# ==========================================================================
# The reference evaluator reproduces the solver, from the file alone
# ==========================================================================
@pytest.mark.parametrize("scheme", SCHEMES)
def test_reference_evaluator_matches_solver(tmp_path, scheme):
    """eval_u (numpy, from file) == solver_3d.evaluate_field, machine precision."""
    prob, sl, U = _manufactured(scheme=scheme)
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


@pytest.mark.parametrize("scheme", SCHEMES)
def test_full_psi_is_BL_plus_u(tmp_path, scheme):
    prob, sl, U = _manufactured(scheme=scheme)
    path = str(tmp_path / "slice.lmid")
    eg.from_solution(prob, sl, U).write(path)
    exp = eg.Export.read(path)
    P = _sample_points(b=sl.b)
    lhs = eg.eval_psi(exp, P[:, 0], P[:, 1], P[:, 2])
    rhs = (eg.eval_psi_BL(exp, P[:, 0], P[:, 1], P[:, 2])
           + eg.eval_u(exp, P[:, 0], P[:, 1], P[:, 2]))
    assert np.max(np.abs(lhs - rhs)) == 0.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_evaluator_exact_at_nodes(tmp_path, scheme):
    """At a grid node the barycentric rule must return the nodal value exactly.

    This is the case a naive ``w/(xq-x)`` implementation divides by zero on, and
    the one a C++ port is most likely to get wrong.
    """
    prob, sl, U = _manufactured(scheme=scheme)
    path = str(tmp_path / "slice.lmid")
    eg.from_solution(prob, sl, U).write(path)
    exp = eg.Export.read(path)

    # pick interior nodes (A=1 is spatial infinity, A=0 the axis edge); exp.B holds
    # the angular nodes in the format's coordinate (s in format 4)
    Bphys = exp.B if exp.fmt == 2 else ops.b_of_s(exp.B, "abt")
    for i in (3, 7, 12):
        for j in (2, 6, 10):
            rho, z = ops.abt_map(exp.A[i], Bphys[j], exp.b)
            for p in (0, 2, 5):
                phi = 2.0 * np.pi * p / exp.Nphi
                x, y = rho * np.cos(phi), rho * np.sin(phi)
                got = float(eg.eval_u(exp, x, y, z)[0])
                assert abs(got - U[i, j, p]) < 1e-11, (i, j, p, got,
                                                       U[i, j, p])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_reference_table(tmp_path, scheme):
    """The C++-port target table is well formed and reproducible."""
    prob, sl, U = _manufactured(scheme=scheme)
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
# Format 4: the "v2" grid
# ==========================================================================
_FORMAT2_KEYS = ["format", "b", "m_A", "m_B", "P_A", "P_B", "S_A", "S_B", "Na", "Nb",
                 "Nphi", "ncos", "nsin", "cos_m", "sin_m", "A", "B", "C", "S"]


def _keys(path):
    return [ln.split()[0] for ln in open(path) if ln.strip() and not ln.startswith("#")]


def test_the_grid_decides_the_format(tmp_path):
    """paper-v1 writes format 2 with exactly its old keys; v2 writes format 4 with the
    angular nodes under ``s`` and the two grid records; a mixed grid is refused."""
    assert eg.FORMAT_OF_SCHEME == {"paper-v1": 2, "v2": 4} and eg.FORMAT_VERSION == 4
    for scheme, fmt in eg.FORMAT_OF_SCHEME.items():
        prob, sl, U = _manufactured(scheme=scheme)
        exp = eg.from_solution(prob, sl, U)
        assert exp.fmt == fmt
        path = exp.write(str(tmp_path / f"{scheme}.lmid"))
        keys = _keys(path)
        if fmt == 2:
            assert keys == _FORMAT2_KEYS, "a paper-v1 file must keep the format-2 layout"
        else:
            want = _FORMAT2_KEYS[:13] + ["angular_map", "radial_map"] + _FORMAT2_KEYS[13:]
            assert keys == [k if k != "B" else "s" for k in want]
            assert np.array_equal(exp.B, prob.Bs)                  # s, as laid
    for kw in (dict(bmap="cos", radial="gauss"), dict(bmap="abt", radial="lobatto")):
        prob = s3.make_problem(12, 10, 4, **kw)
        with pytest.raises(NotImplementedError, match="mixed grid"):
            eg.from_solution(prob, _manufactured()[1], np.zeros(prob.shape))


def test_format4_reader_refuses_a_file_without_its_grid_records(tmp_path):
    prob, sl, U = _manufactured(scheme="v2")
    path = eg.from_solution(prob, sl, U).write(str(tmp_path / "v2.lmid"))
    txt = open(path).read()
    open(path, "w").write(txt.replace("radial_map gauss\n", ""))
    with pytest.raises(ValueError, match="radial_map gauss"):
        eg.Export.read(path)


def test_inverse_abt_s_matches_solver():
    """The transcribed (A, s) map == operators_abt.inverse_map_s, including a hair off
    the outer axis, where s is ill-conditioned in B."""
    P = _sample_points()
    rho = np.concatenate([np.hypot(P[:, 0], P[:, 1]), [1e-12, 1e-9, 1e-6, 0.0]])
    z = np.concatenate([P[:, 2], [7.0, -11.0, 3.0 + 1e-4, 50.0]])
    A1, s1, omb2 = eg.inverse_abt_s(rho, z, B_ANCHOR)
    A2, s2 = ops.inverse_map_s(rho, z, B_ANCHOR, "abt")
    assert np.max(np.abs(A1 - A2)) < 1e-15 and np.max(np.abs(s1 - s2)) < 1e-15
    B = ops.b_of_s(s1, "abt")
    assert np.max(np.abs(omb2 - (1.0 - B) * (1.0 + B))) < 1e-14


def test_format4_evaluator_near_the_axis_the_punctures_and_infinity(tmp_path):
    """The places a v2 consumer can go wrong: s next to the outer axis, the puncture
    corner (A = 0 lies outside the Gauss nodes: an extrapolation), and spatial
    infinity, where u = (A - 1) V vanishes by construction."""
    prob, sl, U = _manufactured(scheme="v2")
    exp = eg.Export.read(eg.from_solution(prob, sl, U).write(str(tmp_path / "v2.lmid")))
    b = sl.b
    P = np.array([[1e-9, 0.0, 7.0], [0.0, 1e-7, -9.0], [1e-6, 1e-6, b + 1e-3],
                  [0.0, 0.01, b - 0.01], [0.0, 0.0, b], [0.0, 0.0, -b],
                  [0.0, 1e-8, 0.5 * b], [300.0, 0.0, 0.0]])
    rho, phi = np.hypot(P[:, 0], P[:, 1]), np.arctan2(P[:, 1], P[:, 0])
    want = s3.evaluate_field(prob, U, rho, P[:, 2], phi, b)
    got = eg.eval_u(exp, P[:, 0], P[:, 1], P[:, 2])
    assert np.max(np.abs(got - want)) < 1e-12 * max(1.0, np.max(np.abs(want)))
    assert eg.eval_u(exp, 0.0, 1e150, 0.0)[0] == 0.0


# ==========================================================================
# Standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lemaitre.initial_data.conformally_flat_puncture.validation.export_grteclyn as mod
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
