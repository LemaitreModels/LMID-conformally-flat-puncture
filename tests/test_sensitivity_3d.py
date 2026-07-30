"""H5a acceptance — the ``solver_3d`` certified-ID parameter tangent ``dU/dθ_k``
(GRADIENT_ENHANCED_PLAN §4 H5a; the gating de-risk of the sparse gradient track).

Gates (the Q0-analog GO/NO-GO):
  * ``certified_tangent_3d`` matches the **central FD of the certified 3-D solve**
    ``d/dθ_k[newton_solve→U]`` to the FD-oracle floor, on ≥3 held-out slices per
    active axis (``b``, ``q``, ``P``, and the spin components ``S_x``/``S_z``);
  * the tangent's **own IFT linear residual** ``‖J·dU + ∂R/∂θ‖/‖∂R/∂θ‖`` is machine
    (proving the residual floor above is the FD oracle, not the method);
  * the modified-Newton node and the NK-built (``solver_3d_nk``) node give the SAME
    tangent (the fields are bit-identical) — NK nodes are not required for the
    tangent (the R2/R6 check);
  * the per-mode back-solve reuses ONE shared assembly across all axes;
  * axisymmetric-reducible cross-check: ``b``/``q`` reproduce ``solver_abt.tangent_b``
    /``tangent_q`` to machine (the reducible limit, block-diagonal == full Jacobian);
  * the cheap block-diagonal (``jac="modified"``) tangent is bit-for-bit the full-J
    tangent for an aligned slice, and quantifiably off (the dropped mode-coupling)
    for a misaligned one — documenting that ``jac="nk"`` is the accurate route.

Standalone (numpy/scipy); reuses the frozen ``solver_3d``/``solver_3d_nk``/
``operators_3d``/``source``/``source_3d`` and the axisymmetric ``solver_abt``
verbatim; the new tangent lives in ``applications.sensitivity_3d``.
"""

import numpy as np
import pytest

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.solver import solver_3d_nk as s3nk
from lm.initial_data.solver import solver_abt as sa
from lm.initial_data.solver import operators_3d as ops3
from lm.initial_data.solver import source_3d
from lm.initial_data.applications import sensitivity_3d as s3d


M_TOT = 1.0
NA, NB, NPHI = 18, 14, 6


# --------------------------------------------------------------------------
# Fixtures — build the frozen grids once
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prob():
    return s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)


@pytest.fixture(scope="module")
def prob2():
    """Matched 2-D meridian grid (same Na,Nb) — the axisymmetric-reduction oracle."""
    return sa.make_problem(Na=NA, Nb=NB, P=0.5)


def _slice(b=2.4, q=1.7, P=0.5, S_A=(0.0, 0.0, 0.0), M_tot=M_TOT):
    m_A = M_tot * q / (1.0 + q)
    m_B = M_tot / (1.0 + q)
    return s3.Slice3D(b=b, m_A=m_A, m_B=m_B,
                      P_A_vec=(0.0, 0.0, -P), P_B_vec=(0.0, 0.0, P),
                      S_A_vec=tuple(map(float, S_A)), S_B_vec=(0.0, 0.0, 0.0))


# ≥3 genuinely non-axisymmetric held-out slices (misaligned spin on A)
MISALIGNED = [
    _slice(b=2.4, q=1.7, P=0.5, S_A=(0.20, 0.0, 0.10)),
    _slice(b=3.1, q=1.0, P=0.3, S_A=(0.15, 0.0, -0.08)),
    _slice(b=2.0, q=2.3, P=0.6, S_A=(0.25, 0.0, 0.05)),
]


# ==========================================================================
# H5a-U1 — axisymmetric-reducible cross-check vs the frozen solver_abt tangent
# ==========================================================================
def test_reduces_to_solver_abt_bq(prob, prob2):
    """On-axis-P, no-spin: the 3-D solve/tangent reduce to the 2-D solver_abt.

    ``tangent[b]``/``tangent[q]`` (BOTH jac routes — the source is φ-independent so
    the block-diagonal is the full Jacobian) reproduce ``solver_abt.tangent_b``/
    ``tangent_q`` to machine on the shared meridian grid."""
    b, q, P = 2.4, 1.7, 0.5
    sl = _slice(b=b, q=q, P=P)
    sl2 = sa.Slice(b=b, m_A=sl.m_A, m_B=sl.m_B)
    U3, info3 = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    U2, info2 = sa.newton_solve(prob2, sl2, tol=1e-12, max_iter=40)
    assert info3.residual_norm < 1e-9 and info2.residual_norm < 1e-9
    assert np.max(np.abs(U3[..., 0] - U2)) < 1e-13   # 3-D solve == 2-D bit-for-bit

    asm = s3.assemble(prob, sl)
    for name, ref in (("b", sa.tangent_b(prob2, U2, sl2)),
                      ("q", sa.tangent_q(prob2, U2, sl2, M_TOT))):
        ref = np.asarray(ref)
        for jac in ("nk", "modified"):
            t = np.asarray(s3d.certified_tangent_3d(prob, U3, sl, name, M_TOT,
                                                    asm=asm, jac=jac))
            rel = np.max(np.abs(t[..., 0] - ref)) / max(np.max(np.abs(ref)), 1e-30)
            assert rel < 1e-11, (name, jac, rel)
            # φ-independent source ⇒ no azimuthal variation in the tangent
            assert np.max(np.abs(t - t[..., :1])) < 1e-12


# ==========================================================================
# H5a-T1 — nk tangent vs central FD of the certified solve (the primary gate)
# ==========================================================================
def test_nk_tangent_vs_fd_nonaxisymmetric(prob):
    """certified_tangent_3d(jac='nk') vs central FD of newton_solve, ≥3 slices per
    axis, for b/q/P and the non-axisymmetric spin axes S_x/S_z — to the FD floor."""
    axes = ("b", "q", "P", "S_x", "S_z")
    worst = {a: 0.0 for a in axes}
    for sl in MISALIGNED:
        U, info = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
        assert info.residual_norm < 1e-9
        asm = s3.assemble(prob, sl)
        for name in axes:
            fd = s3d.fd_tangent_3d(prob, sl, name, M_TOT, h=1e-4, solver="modified")
            t = np.asarray(s3d.certified_tangent_3d(prob, U, sl, name, M_TOT,
                                                    asm=asm, jac="nk"))
            rel = np.max(np.abs(t - fd)) / max(np.max(np.abs(fd)), 1e-30)
            worst[name] = max(worst[name], rel)
            assert rel < 1e-6, (name, sl.b, rel)   # O(h²) FD-oracle floor
    # the axes are genuinely exercised (non-trivial tangents throughout)
    assert all(v > 0 for v in worst.values())


def test_nk_tangent_vs_fd_axisymmetric(prob):
    """Reducible head-on: tangent[P] jac='nk' matches central FD to the FD floor,
    and jac='modified' == jac='nk' bit-for-bit (φ-independent source)."""
    sl = _slice(b=2.4, q=1.7, P=0.5)
    U, _ = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    asm = s3.assemble(prob, sl)
    fd = s3d.fd_tangent_3d(prob, sl, "P", M_TOT, h=1e-4, solver="modified")
    t_nk = np.asarray(s3d.certified_tangent_3d(prob, U, sl, "P", M_TOT, asm=asm, jac="nk"))
    t_mod = np.asarray(s3d.certified_tangent_3d(prob, U, sl, "P", M_TOT, asm=asm, jac="modified"))
    assert np.max(np.abs(t_nk - fd)) / np.max(np.abs(fd)) < 1e-7
    assert np.max(np.abs(t_nk - t_mod)) < 1e-11        # block-diag IS the full J here


# ==========================================================================
# H5a-U2 — the tangent solves the IFT equation to machine (FD is the floor, not us)
# ==========================================================================
def test_nk_tangent_ift_residual_machine(prob):
    """‖J·dU + ∂R/∂θ‖_∞ / ‖∂R/∂θ‖_∞ is machine — the full-J tangent is an exact
    (to roundoff) solve of the implicit-function equation, so the ~1e-8 FD mismatch
    above is the O(h²) FD oracle, not the tangent."""
    sl = MISALIGNED[0]
    U, _ = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    asm = s3.assemble(prob, sl)
    Uarr = np.asarray(U).reshape(prob.Ntot2d, prob.Nphi)
    _, base = s3._nl_source(asm, Uarr)
    D_nl = -0.875 * base ** (-8.0) * asm.A2

    def Jmatvec(dU):
        dU = np.asarray(dU).reshape(prob.Ntot2d, prob.Nphi)
        dUhat = np.fft.rfft(dU, axis=1)
        DdU = np.fft.rfft(D_nl * dU, axis=1)
        out = np.empty((prob.Ntot2d, asm.m_vals.size), dtype=complex)
        for mi in range(asm.m_vals.size):
            out[:, mi] = (asm.M0[mi] @ (dUhat[:, mi] / asm.w[mi])
                          + np.where(asm.interior, DdU[:, mi], 0.0))
        return np.fft.irfft(out, n=prob.Nphi, axis=1)

    for name in ("b", "q", "P", "S_x", "S_z"):
        dR = s3d.dR_dtheta_node(prob, asm, Uarr, sl, name, M_TOT)
        dU = np.asarray(s3d.certified_tangent_3d(prob, U, sl, name, M_TOT,
                                                 asm=asm, jac="nk"))
        lin = Jmatvec(dU) + dR
        rel = np.max(np.abs(lin)) / max(np.max(np.abs(dR)), 1e-30)
        assert rel < 1e-9, (name, rel)


# ==========================================================================
# H5a-U3 — node floor: modified-Newton vs NK-built node give the same tangent
# ==========================================================================
def test_node_floor_modified_vs_nk(prob):
    """The modified-Newton node and the NK node produce the same tangent (their
    converged fields are bit-identical), so NK-built nodes are NOT required for the
    tangent at these settings (the Q0/R2 dependency check)."""
    sl = MISALIGNED[0]
    Umod, imod = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    Unk, ink = s3nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30)
    assert ink.residual_norm < imod.residual_norm    # NK certifies tighter
    assert np.max(np.abs(np.asarray(Umod) - np.asarray(Unk))) < 1e-11
    asm = s3.assemble(prob, sl)
    for name in ("b", "q", "P", "S_x", "S_z"):
        t_mod = np.asarray(s3d.certified_tangent_3d(prob, Umod, sl, name, M_TOT, asm=asm))
        t_nk = np.asarray(s3d.certified_tangent_3d(prob, Unk, sl, name, M_TOT, asm=asm))
        gap = np.max(np.abs(t_mod - t_nk)) / max(np.max(np.abs(t_nk)), 1e-30)
        assert gap < 1e-8, (name, gap)


# ==========================================================================
# H5a-U4 — the modified-Newton (block-diagonal) vs NK (full-J) tangent-solve gap
# ==========================================================================
def test_modified_vs_nk_solve_gap(prob):
    """For a genuinely non-axisymmetric slice the cheap block-diagonal per-mode
    tangent drops the φ-varying source Jacobian → a materially non-zero gap vs the
    full-J tangent; the full-J route (few GMRES iters) is the accurate one."""
    sl = MISALIGNED[0]
    U, _ = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    asm = s3.assemble(prob, sl)
    for name in ("S_x", "S_z", "b"):
        t_nk = np.asarray(s3d.certified_tangent_3d(prob, U, sl, name, M_TOT, asm=asm, jac="nk"))
        t_mod = np.asarray(s3d.certified_tangent_3d(prob, U, sl, name, M_TOT, asm=asm, jac="modified"))
        gap = np.max(np.abs(t_nk - t_mod)) / np.max(np.abs(t_nk))
        assert 1e-3 < gap < 2e-1, (name, gap)   # dropped mode-coupling, not machine


# ==========================================================================
# H5a-U5 — the per-mode back-solve reuses ONE shared assembly across axes
# ==========================================================================
def test_shared_assembly_across_axes(prob, monkeypatch):
    """certified_tangent_3d(asm=...) never rebuilds the assembly; one
    s3.assemble is amortised across every axis (the certified_tangent asm= pattern)."""
    sl = MISALIGNED[0]
    U, _ = s3.newton_solve(prob, sl, tol=1e-12, max_iter=40)
    asm = s3.assemble(prob, sl)                       # the ONE assembly

    calls = [0]
    orig = s3.assemble
    monkeypatch.setattr(s3, "assemble",
                        lambda *a, **k: (calls.__setitem__(0, calls[0] + 1), orig(*a, **k))[1])
    for name in ("b", "q", "P", "S_x", "S_z"):
        s3d.certified_tangent_3d(prob, U, sl, name, M_TOT, asm=asm, jac="modified")
    assert calls[0] == 0                              # no rebuilds when asm passed
    # ... but it DOES build one when asm is omitted
    s3d.certified_tangent_3d(prob, U, sl, "b", M_TOT, jac="modified")
    assert calls[0] == 1


# ==========================================================================
# H5a-U6 — the genuinely-new analytic ∂Â²/∂θ vs FD of the BY source (oracle)
# ==========================================================================
def test_dA2_dtheta_vs_fd_source(prob):
    """The analytic ``∂Â²/∂θ`` (the linear-tensor chain rule / the b scale law)
    against central FD of ``source_3d.A2_at_nodes_3d`` (FD as validation oracle
    only).  P/S_x/S_z are fixed-node; b recomputes the moving (A,B)→(ρ,z) nodes."""
    sl = MISALIGNED[0]
    asm = s3.assemble(prob, sl)
    rho, z, phi = asm.rho, asm.z, prob.phi
    h = 1e-5

    def A2(s):
        return source_3d.A2_at_nodes_3d(s.rho if hasattr(s, "rho") else rho, z, phi,
                                        s.b, s.P_A_vec, s.P_B_vec, s.S_A_vec, s.S_B_vec)

    # fixed-node axes: ρ,z frozen, only the vectors move
    for name in ("P", "S_x", "S_z"):
        sp = s3d.perturb_slice_3d(sl, name, +h, M_TOT)
        sm = s3d.perturb_slice_3d(sl, name, -h, M_TOT)
        fd = (source_3d.A2_at_nodes_3d(rho, z, phi, sp.b, sp.P_A_vec, sp.P_B_vec,
                                       sp.S_A_vec, sp.S_B_vec)
              - source_3d.A2_at_nodes_3d(rho, z, phi, sm.b, sm.P_A_vec, sm.P_B_vec,
                                         sm.S_A_vec, sm.S_B_vec)) / (2 * h)
        an = s3d._dA2_dtheta(asm, phi, sl, name)
        fin = np.isfinite(rho)
        rel = np.max(np.abs((an - fd)[fin])) / max(np.max(np.abs(fd[fin])), 1e-30)
        assert rel < 1e-5, (name, rel)

    # b: the nodes move with b (ρ,z = abt_map(A,B,b)); FD must recompute them
    from lm.initial_data.solver import operators_abt as ops
    _, _, _, Af, Bf, _, _, _ = ops3.axisym_blocks(prob.A, prob.B, prob.DA1, prob.DB1, sl.b)

    def A2_b(bv):
        rr, zz = ops.abt_map(Af, Bf, bv)
        return source_3d.A2_at_nodes_3d(rr, zz, phi, bv, sl.P_A_vec, sl.P_B_vec,
                                        sl.S_A_vec, sl.S_B_vec)

    fd_b = (A2_b(sl.b + h) - A2_b(sl.b - h)) / (2 * h)
    an_b = s3d._dA2_dtheta(asm, phi, sl, "b")
    fin = np.isfinite(rho)
    rel_b = np.max(np.abs((an_b - fd_b)[fin])) / max(np.max(np.abs(fd_b[fin])), 1e-30)
    assert rel_b < 1e-5, rel_b


def test_dvec_dtheta_conventions():
    """The direct-axis physical-vector derivatives are the expected unit vectors."""
    sl = _slice(b=2.4, q=1.0, P=0.5, S_A=(0.2, 0.0, 0.1))
    dPA, dPB, dSA, dSB = s3d._dvec_dtheta(sl, "P")
    assert np.allclose(dPA, [0, 0, -1]) and np.allclose(dPB, [0, 0, 1])
    dPA, dPB, _, _ = s3d._dvec_dtheta(sl, "P_x")
    assert np.allclose(dPA, [1, 0, 0]) and np.allclose(dPB, [-1, 0, 0])
    _, _, dSA, _ = s3d._dvec_dtheta(sl, "S_x")
    assert np.allclose(dSA, [1, 0, 0])
    _, _, dSA, dSB = s3d._dvec_dtheta(sl, "S_Bz")
    assert np.allclose(dSA, [0, 0, 0]) and np.allclose(dSB, [0, 0, 1])
    # polar: dS/dS_mag is the unit spin direction; dS/dθ_S ⟂ it
    _, _, dSmag, _ = s3d._dvec_dtheta(sl, "S_mag")
    assert abs(np.linalg.norm(dSmag) - 1.0) < 1e-12
    _, _, dth, _ = s3d._dvec_dtheta(sl, "theta_S")
    assert abs(np.dot(dSmag, dth)) < 1e-12          # orthogonal (magnitude vs tilt)


# ==========================================================================
# H5a-T2 — add-only / standalone discipline
# ==========================================================================
def test_standalone_imports():
    import lm.initial_data.applications.sensitivity_3d as mod
    src = open(mod.__file__).read()
    for forbidden in ("import nrpy", "src.bbhfm", "from bbhfm", "import context", "torch"):
        assert forbidden not in src, forbidden
