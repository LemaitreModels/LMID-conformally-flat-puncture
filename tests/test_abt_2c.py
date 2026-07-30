"""M2-A acceptance — the ABT / prolate-spheroidal single-patch two-centre solver.

Gates:
  * geometry + inverse-map round-trip;
  * prolate Laplacian matches the closed form  Delta e^{-r^2} = (4r^2-6)e^{-r^2}
    (spectral) and a manufactured Poisson solve converges spectrally;
  * P=0 is an EXACT fixed point (u≡0 to machine zero) on the ABT grid;
  * the two-centre field self-converges EXPONENTIALLY to <=1e-9 (mortar-free,
    both punctures resolved);
  * equal-mass solution is EVEN in B to spectral precision (the B-as-mirror
    proof — the clean analog of half-vs-full; contrast the single A-centred
    grid's 39% asymmetry).
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from lm.initial_data.solver import operators_abt as ops
from lm.initial_data.solver import solver_abt as sa
from lm.initial_data.solver import diagnostics as diag
from lm.initial_data.solver.solver_abt import Slice


# --------------------------------------------------------------------------
# M2-A.0 — geometry + inverse map
# --------------------------------------------------------------------------
def test_geometry_and_inverse():
    b = 1.3
    # punctures at corners
    rA, zA = ops.abt_map(1e-9, 1.0 - 1e-13, b)
    assert abs(zA - b) < 1e-6 and rA < 1e-7
    rB, zB = ops.abt_map(1e-9, -1.0 + 1e-13, b)
    assert abs(zB + b) < 1e-6 and rB < 1e-7
    # infinity at A=1
    rinf, zinf = ops.abt_map(1.0 - 1e-7, 0.3, b)
    assert rinf > 1e5
    # inverse round-trip at generic interior points
    A, B, _, _ = ops.build_grid(20, 14)
    AA, BB = np.meshgrid(A, B, indexing="ij")
    interior = (AA > 0.05) & (AA < 0.95)
    rho, z = ops.abt_map(AA[interior], BB[interior], b)
    A2, B2 = ops.inverse_map(rho, z, b)
    assert np.max(np.abs(A2 - AA[interior])) < 1e-10
    assert np.max(np.abs(B2 - BB[interior])) < 1e-10


# --------------------------------------------------------------------------
# M2-A.1 — prolate Laplacian vs closed form (spectral)
# --------------------------------------------------------------------------
def test_laplacian_closed_form():
    b = 1.0
    errs = []
    for (Na, Nb) in [(16, 12), (24, 16), (32, 20), (40, 28), (48, 32)]:
        A, B, DA1, DB1 = ops.build_grid(Na, Nb)
        Lap, rho, z, Af, Bf, DA, DB = ops.laplacian_matrix(A, B, DA1, DB1, b)
        r2 = rho ** 2 + z ** 2
        r2s = np.where(np.isfinite(r2), r2, 0.0)
        g = np.exp(-r2s)
        g[~np.isfinite(r2)] = 0.0
        true = (4.0 * r2s - 6.0) * np.exp(-r2s)
        interior = (Af > 1e-12) & (Af < 1.0 - 1e-12)
        errs.append(float(np.max(np.abs((Lap @ g)[interior] - true[interior]))))
    errs = np.array(errs)
    # geometric convergence to the exact closed form (the sharp Gaussian is
    # moderately hard in prolate coords; the puncture field decays only as 1/r
    # and converges far faster — see test_spatial_spectral_convergence).
    assert errs[0] / errs[-1] > 1e3, f"Laplacian not converging: {errs}"
    assert np.all(np.diff(errs) < 0), f"not monotone: {errs}"
    assert errs[-1] < 1e-3, f"Laplacian err at Na=48 = {errs[-1]:.2e}"


def test_manufactured_poisson():
    """Delta u = (4r^2-6)e^{-r^2}, u=e^{-r^2}: spectral convergence."""
    b = 1.0
    rows, errs = [], []
    for (Na, Nb) in [(16, 12), (24, 16), (32, 20), (40, 28)]:
        A, B, DA1, DB1 = ops.build_grid(Na, Nb)
        Lap, rho, z, Af, Bf, DA, DB = ops.laplacian_matrix(A, B, DA1, DB1, b)
        M0, interior = ops.apply_bcs(Lap, A, B, DA)
        r2 = rho ** 2 + z ** 2
        r2s = np.where(np.isfinite(r2), r2, 1e30)
        S = np.where(interior, (4.0 * r2s - 6.0) * np.exp(-r2s), 0.0)
        S = np.where(np.isfinite(S), S, 0.0)
        u = ops.solve_equilibrated(M0, S)
        u_exact = np.where(np.isfinite(r2), np.exp(-r2s), 0.0)
        e = float(np.max(np.abs((u - u_exact)[interior])))
        errs.append(e)
        rows.append((Na, Nb, e))
    diag.convergence_table(rows, ["Na", "Nb", "uErr"],
                           title="\n[M2-A] manufactured Poisson convergence")
    errs = np.array(errs)
    assert errs[0] / errs[-1] > 1e4, f"Poisson not spectral: {errs}"
    assert errs[-1] < 1e-7, f"Poisson err at Na=40 = {errs[-1]:.2e}"


def test_manufactured_poisson_one_over_r():
    """Delta u = -3(1+r^2)^{-5/2}, u=(1+r^2)^{-1/2}: a 1/r-decaying solution
    (the physical decay class of the puncture monopole) converges spectrally.

    Sharp localized functions (e.g. a Gaussian) are 'moderately hard' on this
    compactified, focus-clustering prolate grid; the physically-relevant
    1/r-decaying class converges cleanly to MACHINE PRECISION — this is the
    operator's real-use validation (added in response to the M2-A adversarial
    review).  With row equilibration (``ops.solve_equilibrated``) the dense
    prolate operator's condition number is ~1e4, so a single absolute solve
    reaches ~1e-13.
    """
    b = 1.0
    rows, errs = [], []
    for (Na, Nb) in [(16, 12), (24, 16), (32, 20), (40, 28), (48, 32)]:
        A, B, DA1, DB1 = ops.build_grid(Na, Nb)
        Lap, rho, z, Af, Bf, DA, DB = ops.laplacian_matrix(A, B, DA1, DB1, b)
        M0, interior = ops.apply_bcs(Lap, A, B, DA)
        r2 = rho ** 2 + z ** 2
        r2s = np.where(np.isfinite(r2), r2, 1e30)
        S = np.where(interior, -3.0 * (1.0 + r2s) ** (-2.5), 0.0)
        u = ops.solve_equilibrated(M0, S)
        u_exact = np.where(np.isfinite(r2), (1.0 + r2s) ** (-0.5), 0.0)
        e = float(np.max(np.abs((u - u_exact)[interior])))
        errs.append(e)
        rows.append((Na, Nb, e))
    diag.convergence_table(rows, ["Na", "Nb", "uErr"],
                           title="\n[M2-A] manufactured Poisson (1/r-decaying solution, equilibrated)")
    errs = np.array(errs)
    assert errs[0] / errs[3] > 1e4, f"1/r Poisson not spectral: {errs}"
    assert np.min(errs) < 1e-11, f"1/r Poisson floor = {np.min(errs):.2e} (expect ~1e-13)"


# --------------------------------------------------------------------------
# M2-A.2 — PRIMARY GATE: P=0 is an EXACT fixed point on the ABT grid
# --------------------------------------------------------------------------
def test_P0_exact_fixed_point_abt():
    prob = sa.make_problem(Na=32, Nb=22, P=0.0)
    for sl in (Slice(b=1.0, m_A=0.5, m_B=0.5), Slice(b=1.5, m_A=0.7, m_B=0.3)):
        U, info = sa.newton_solve(prob, sl, tol=1e-13, max_iter=8)
        assert float(np.max(np.abs(U))) == 0.0, f"P=0 nonzero u, b={sl.b}"
        assert info.residual_norm == 0.0


# --------------------------------------------------------------------------
# M2-A.3 — two-centre Newton + EXPONENTIAL spatial self-convergence (<=1e-9)
# --------------------------------------------------------------------------
def test_spatial_spectral_convergence():
    b, m_A, m_B, P = 1.0, 0.5, 0.5, 0.5
    # high-resolution reference
    pref = sa.make_problem(Na=52, Nb=36, P=P)
    slc = Slice(b=b, m_A=m_A, m_B=m_B)
    Uref, iref = sa.newton_solve(pref, slc, tol=1e-9, max_iter=25)
    assert iref.residual_norm < 1e-8, f"reference ||R||={iref.residual_norm:.2e}"
    # query points near A, midpoint, near B, off-axis, far
    qpts = [(0.3, 0.7), (0.6, 0.0), (0.3, -0.7), (1.2, 0.4), (3.0, 0.0)]
    rq = np.array([p[0] for p in qpts]); zq = np.array([p[1] for p in qpts])
    uref = sa.evaluate_field_phys(pref, Uref, rq, zq, b)

    rows, errs = [], []
    for (Na, Nb) in [(20, 16), (28, 20), (36, 24), (44, 30)]:
        prob = sa.make_problem(Na=Na, Nb=Nb, P=P)
        U, info = sa.newton_solve(prob, slc, tol=1e-9, max_iter=25)
        uq = sa.evaluate_field_phys(prob, U, rq, zq, b)
        e = float(np.max(np.abs(uq - uref)))
        errs.append(e)
        rows.append((Na, Nb, info.iters, info.residual_norm, e))
    diag.convergence_table(rows, ["Na", "Nb", "its", "||R||", "fieldErr"],
                           title="\n[M2-A] two-centre spatial spectral convergence")
    errs = np.array(errs)
    assert errs[0] / errs[-1] > 1e2, f"not exponential: {errs}"
    assert np.all(np.diff(errs) < 0), f"not monotone: {errs}"
    assert errs[-1] < 1e-9, f"field err at Na=44 = {errs[-1]:.2e} (target <=1e-9)"


# --------------------------------------------------------------------------
# M2-A.4 — equal-mass B-as-mirror proof: u is EVEN in B to spectral precision
# --------------------------------------------------------------------------
def test_equal_mass_B_even():
    """For equal mass the (exactly z-even) solution is even in B: U[i,j]=U[i,-j].

    GL B-nodes are symmetric about 0, the prolate operator commutes with B->-B,
    and the equal-mass source is even in B, so the discrete solution is even to
    near machine precision — the clean B-as-mirror proof (vs the single
    A-centred grid's 39% z-asymmetry).
    """
    prob = sa.make_problem(Na=40, Nb=28, P=0.5)
    sl = Slice(b=1.0, m_A=0.5, m_B=0.5)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=25)
    assert info.residual_norm < 1e-8
    U = np.asarray(U)
    # GL nodes are symmetric: B[j] = -B[Nb-1-j]
    asym = np.max(np.abs(U - U[:, ::-1])) / np.max(np.abs(U))
    print(f"\n[M2-A] equal-mass B-asymmetry (ABT grid) = {asym:.3e}  "
          f"(vs single A-centred 39%)")
    assert asym < 1e-9, f"equal-mass not B-even: {asym:.2e}"


def test_unequal_mass_not_B_even():
    """Sanity: unequal mass is NOT B-even (the test above has teeth)."""
    prob = sa.make_problem(Na=40, Nb=28, P=0.5)
    sl = Slice(b=1.0, m_A=0.7, m_B=0.3)
    U, info = sa.newton_solve(prob, sl, tol=1e-9, max_iter=25)
    U = np.asarray(U)
    asym = np.max(np.abs(U - U[:, ::-1])) / np.max(np.abs(U))
    assert asym > 1e-2, f"unequal-mass unexpectedly B-even: {asym:.2e}"


# --------------------------------------------------------------------------
# M2-A.5 — parameter tangents consistent with finite differences
# --------------------------------------------------------------------------
def test_tangent_b_consistent():
    prob = sa.make_problem(Na=28, Nb=20, P=0.5)
    b0 = 1.2
    sl = Slice(b=b0, m_A=0.5, m_B=0.5)
    U, _ = sa.newton_solve(prob, sl, tol=1e-10, max_iter=25)
    dU = np.asarray(sa.tangent_b(prob, U, sl))
    h = 1e-5
    Up, _ = sa.newton_solve(prob, Slice(b0 + h, 0.5, 0.5), U0=U, tol=1e-11, max_iter=15)
    Um, _ = sa.newton_solve(prob, Slice(b0 - h, 0.5, 0.5), U0=U, tol=1e-11, max_iter=15)
    dU_fd = (np.asarray(Up) - np.asarray(Um)) / (2 * h)
    rel = np.max(np.abs(dU - dU_fd)) / np.max(np.abs(dU_fd))
    assert rel < 1e-5, f"tangent_b vs FD rel error {rel:.2e}"


def test_tangent_q_consistent():
    prob = sa.make_problem(Na=28, Nb=20, P=0.5)
    b0, M = 1.2, 1.0
    q0 = 1.4
    mA = M * q0 / (1 + q0); mB = M / (1 + q0)
    sl = Slice(b=b0, m_A=mA, m_B=mB)
    U, _ = sa.newton_solve(prob, sl, tol=1e-10, max_iter=25)
    dU = np.asarray(sa.tangent_q(prob, U, sl, M))
    h = 1e-5
    def slq(q):
        return Slice(b0, M * q / (1 + q), M / (1 + q))
    Up, _ = sa.newton_solve(prob, slq(q0 + h), U0=U, tol=1e-11, max_iter=15)
    Um, _ = sa.newton_solve(prob, slq(q0 - h), U0=U, tol=1e-11, max_iter=15)
    dU_fd = (np.asarray(Up) - np.asarray(Um)) / (2 * h)
    rel = np.max(np.abs(dU - dU_fd)) / np.max(np.abs(dU_fd))
    assert rel < 1e-5, f"tangent_q vs FD rel error {rel:.2e}"
