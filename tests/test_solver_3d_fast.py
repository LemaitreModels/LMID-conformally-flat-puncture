"""Acceptance tests — the fast linear algebra behind the 3-D Newton–Krylov solve.

These pin the four accelerations that were introduced without changing the
mathematics of the solve.  Each is an *exact-arithmetic* claim, so each test
compares the fast route against a straightforward slow route written out in
full here — the tests carry their own oracle rather than a stored number.

  * **Assembly.**  The m-independent pieces of the per-mode block — including the
    two dense squarings — are built once for the whole mode set instead of once
    per mode.  Pinned against a verbatim copy of the per-mode assembly it
    replaced, and, at m=0, against the frozen axisymmetric operator bit-for-bit.
  * **Real/imaginary split.**  The blocks are real and the mode vectors complex;
    ``M @ v`` promotes the *matrix*.  ``M @ [Re v, Im v]`` is the same arithmetic
    on half the flops; a mode that is exactly real (m=0, and Nyquist for even Nφ)
    takes the one-column real path and stays bit-identical to the 2-D solver.
  * **Operator reuse.**  The row-equilibrated per-mode block does not depend on
    the separation ``b`` (every interior coefficient carries exactly one 1/b²;
    the BC rows are b-free), so one operator serves the whole model.
  * **The separable operator.**  In prolate coordinates ``ξ²−η² = (ξ²−1)+(1−η²)``
    makes the centrifugal term exactly separable, so each per-m block is a
    generalized Kronecker sum and fast diagonalization inverts it exactly.
"""

import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.solver import operators_3d as ops3
from lemaitre.initial_data.conformally_flat_puncture.solver import operators_abt as ops
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3
from lemaitre.initial_data.conformally_flat_puncture.solver import separable
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d_nk as nk
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_abt as sa
from lemaitre.initial_data.conformally_flat_puncture.solver.solver_3d import Slice3D
from lemaitre.initial_data.conformally_flat_puncture.solver.solver_abt import Slice
from lemaitre.initial_data.conformally_flat_puncture.applications import sensitivity_3d as s3d
from lemaitre.initial_data.conformally_flat_puncture.parametric.certification import CERT_TOL


# A grid small enough to run in seconds, large enough that Na != Nb != Nφ so an
# index transposition cannot pass by coincidence.
GRID = dict(Na=20, Nb=14, Nphi=6)


def _slice(b=2.5):
    """A genuinely non-axisymmetric slice: off-axis momenta + misaligned spins."""
    return Slice3D(b=b, m_A=0.6, m_B=0.4,
                   P_A_vec=(0.0, 0.11, -0.02), P_B_vec=(0.0, -0.11, 0.02),
                   S_A_vec=(0.05, 0.0, 0.03), S_B_vec=(0.0, 0.04, -0.01))


# ==========================================================================
# Stage 1.  Assembly — the Kronecker identity for the second derivatives
# ==========================================================================
def _reference_block_operator_m_v(A, B, DA1, DB1, b, m):
    """The pre-acceleration assembly, verbatim: dense ``DA@DA``, dense ``diag()``.

    This is the oracle for :func:`ops3.block_operator_m_v`.  It forms the 2-D
    derivative matrices and squares THEM — the ``(Na·Nb)³`` matmul the shipped
    assembly replaced by an ``Na³`` one — and adds each node-diagonal term as a
    full dense ``np.diag``.
    """
    Na1, Nb1 = A.size, B.size
    IA, IB = np.eye(Na1), np.eye(Nb1)
    DA = np.kron(np.asarray(DA1), IB)
    DB = np.kron(IA, np.asarray(DB1))
    DAA = DA @ DA
    DBB = DB @ DB
    AA, BB = np.meshgrid(A, B, indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    rho, _ = ops.abt_map(Af, Bf, b)
    alpha, pcoef, gamma, qcoef = ops._coeffs(Af, Bf, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_rho2 = 1.0 / rho ** 2
    inv_rho2 = np.where(np.isfinite(inv_rho2) & (rho > 1e-14), inv_rho2, 0.0)
    cent = -(int(m) ** 2) * inv_rho2
    w, wp, wpp = ops3.bc_factor(Bf, m)
    Mv = (alpha * w)[:, None] * DAA + (pcoef * w)[:, None] * DA
    Mv = Mv + gamma[:, None] * (np.diag(wpp) + 2.0 * wp[:, None] * DB
                                + w[:, None] * DBB)
    Mv = Mv + qcoef[:, None] * (np.diag(wp) + w[:, None] * DB)
    Mv = Mv + np.diag(cent * w)
    return Mv, w


def _reference_mode_operators(A, B, DA1, DB1, b, m_vals):
    DA = np.kron(np.asarray(DA1), np.eye(B.size))
    M_bc, w_list, interior = [], [], None
    for m in m_vals:
        Mv, w = _reference_block_operator_m_v(A, B, DA1, DB1, b, int(m))
        Mm, interior = ops3.apply_bcs_m(Mv, A, B, DA, int(m))
        M_bc.append(Mm)
        w_list.append(w)
    return M_bc, w_list, interior


def test_kron_mixed_product_identity():
    """``kron(D,I) @ kron(D,I) == kron(D@D, I)`` — to roundoff, not bit-for-bit.

    Exact in exact arithmetic (the mixed-product rule with an identity factor);
    in float64 the two orderings sum a different number of exactly-zero terms, so
    they agree to ~1e-16 relative and no better.  That gap is why the assembly
    does NOT use the identity even though it would be ~40× cheaper: the m=0 block
    has to equal the frozen axisymmetric operator bit-for-bit (see
    :func:`test_m0_block_is_the_frozen_axisymmetric_operator`).  The identity is
    pinned here because ``solver/separable.py`` is built on the same structure.

    The tolerance is on the RELATIVE error: the entries of a spectral
    second-derivative matrix scale like N⁴, so an absolute one is meaningless.
    """
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    IA, IB = np.eye(A.size), np.eye(B.size)
    for D1, big, ident, left in ((DA1, np.kron(DA1, IB), IB, True),
                                 (DB1, np.kron(IA, DB1), IA, False)):
        ref = big @ big
        fast = (np.kron(np.asarray(D1) @ np.asarray(D1), ident) if left
                else np.kron(ident, np.asarray(D1) @ np.asarray(D1)))
        rel = np.max(np.abs(ref - fast)) / np.max(np.abs(ref))
        print(f"\n[kron] {'kron(D,I)' if left else 'kron(I,D)'} squared: rel={rel:.2e}")
        assert rel < 1e-14, f"mixed-product identity off by {rel:.2e}"


def test_mode_operators_match_dense_assembly():
    """The shipped per-m blocks equal the per-mode assembly to ≤1e-14 relative.

    The load-bearing stage-1 gate: sharing the m-independent work is pure
    bookkeeping, so every assembled block — BC rows included — must be the same
    matrix.  The only arithmetic that moves is the three node-diagonal terms,
    which are now added onto the diagonal instead of through three dense
    ``np.diag`` temporaries of ``(Na·Nb)²`` each; that is a different rounding at
    the ulp level for m≠0 and exactly zero for m=0 (where all three vanish).

    Checked entrywise-relative and in the ROW-EQUILIBRATED norm, which is the one
    the solve actually sees (``operators_abt.solve_equilibrated`` divides each row
    by its own max modulus, and the raw rows span ~13 decades).
    """
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    m_vals = ops3.fourier_modes(GRID["Nphi"])
    for b in (2.5, 7.0):
        ref, w_ref, int_ref = _reference_mode_operators(A, B, DA1, DB1, b, m_vals)
        got, w_got, int_got = ops3.mode_operators(A, B, DA1, DB1, b, m_vals)
        assert np.array_equal(int_ref, int_got), "interior mask changed"
        for mi, m in enumerate(m_vals):
            assert np.array_equal(w_ref[mi], w_got[mi]), f"B-factor changed (m={m})"
            rel = np.max(np.abs(ref[mi] - got[mi])) / np.max(np.abs(ref[mi]))
            s = np.max(np.abs(ref[mi]), axis=1)
            s = np.where(s > 0.0, s, 1.0)
            rel_eq = np.max(np.abs(ref[mi] - got[mi]) / s[:, None])
            print(f"\n[asm] b={b} m={m}: rel={rel:.2e} equil={rel_eq:.2e}")
            assert rel < 1e-14, f"block m={m} moved by {rel:.2e} (b={b})"
            assert rel_eq < 1e-13, f"equilibrated block m={m} moved by {rel_eq:.2e}"
            if m == 0:
                assert np.array_equal(ref[mi], got[mi]), \
                    "the m=0 block must not move at all — it is the 2-D operator"


def test_m0_block_is_the_frozen_axisymmetric_operator():
    """At m=0 the 3-D block IS ``operators_abt``'s Laplacian, bit-for-bit.

    This is what makes the Nφ=1 reduction reproduce the frozen 2-D Newton solve
    to ~1e-16 rather than merely to the ~1e-13 a roundoff-equivalent operator
    would give, and it is the reason the assembly may not use the mixed-product
    identity of :func:`test_kron_mixed_product_identity` on its own.  Assert
    equality, not closeness: "close" here would mean the two solvers had silently
    stopped being the same code.
    """
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    b = 2.5
    Lap, _, _, _, _, DA, _ = ops.laplacian_matrix(A, B, DA1, DB1, b)
    M_2d, int_2d = ops.apply_bcs(Lap, A, B, DA)
    M0_list, w_list, interior = ops3.mode_operators(A, B, DA1, DB1, b,
                                                    np.array([0]))
    assert np.array_equal(w_list[0], np.ones_like(w_list[0])), "m=0 B-factor is not 1"
    assert np.array_equal(interior, int_2d), "m=0 interior mask differs from 2-D"
    assert np.array_equal(M0_list[0], M_2d), \
        "the m=0 3-D block is no longer the frozen 2-D operator bit-for-bit"


def test_meridian_geometry_matches_the_operator_route():
    """``meridian_geometry`` returns exactly what the old operator route did.

    It replaced a call that assembled the dense axisymmetric Laplacian in order
    to hand back four coordinate arrays.  The coordinates themselves must be
    bit-identical: they feed ψ_BL and Â², so any drift here is a physics change.
    """
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    b = 2.5
    _, rho_ref, z_ref, Af_ref, Bf_ref, _, _ = ops.laplacian_matrix(A, B, DA1, DB1, b)
    rho, z, Af, Bf, inv_rho2 = ops3.meridian_geometry(A, B, b)
    for name, ref, got in (("rho", rho_ref, rho), ("z", z_ref, z),
                           ("Af", Af_ref, Af), ("Bf", Bf_ref, Bf)):
        assert np.array_equal(np.asarray(ref), np.asarray(got)), f"{name} drifted"
    # inv_rho2 is 1/ρ² where finite, 0 on the two BC edges
    fin = np.isfinite(rho) & (rho > 1e-14)
    assert np.allclose(inv_rho2[fin], 1.0 / np.asarray(rho)[fin] ** 2, rtol=0, atol=0)
    assert np.all(inv_rho2[~fin] == 0.0)


def test_assembly_unchanged_end_to_end():
    """A full certified solve is unmoved by the assembly change.

    The blocks agree to roundoff, so the Newton and GMRES iteration counts must
    be IDENTICAL and the certified residual must land in the same place.  Run
    against an assembly built the old way, injected into the same ``Assembly3D``.
    """
    prob = s3.make_problem(**GRID)
    sl = _slice()
    asm_new = s3.assemble(prob, sl)
    asm_old = s3.assemble(prob, sl)
    M0, w, interior = _reference_mode_operators(prob.A, prob.B, prob.DA1, prob.DB1,
                                                sl.b, prob.m_vals)
    asm_old.M0, asm_old.w, asm_old.interior = M0, w, interior

    U_new, i_new = nk.newton_solve_nk(prob, sl, tol=1e-11, max_iter=12, asm=asm_new)
    U_old, i_old = nk.newton_solve_nk(prob, sl, tol=1e-11, max_iter=12, asm=asm_old)
    d = np.max(np.abs(np.asarray(U_new) - np.asarray(U_old))) / np.max(np.abs(U_old))
    print(f"\n[asm-e2e] steps {i_old.iters}->{i_new.iters}  "
          f"gmres {i_old.gmres_iters}->{i_new.gmres_iters}  "
          f"equilR {i_old.residual_norm:.3e}->{i_new.residual_norm:.3e}  |dU|/|U|={d:.2e}")
    assert i_new.iters == i_old.iters
    assert i_new.gmres_iters == i_old.gmres_iters
    assert i_new.converged and i_old.converged
    assert d < 1e-12, f"converged field moved by {d:.2e}"


# ==========================================================================
# Stage 2.  A real block applied to a complex mode vector
# ==========================================================================
def test_apply_mode_block_equals_the_promoted_product():
    """``apply_mode_block(M, v)`` is ``M @ v``, for a real M and complex v.

    ``M(x+iy) = Mx + iMy`` identically for real ``M``, so the split is the same
    arithmetic — but not the same *code*: the promoted product runs a complex
    GEMM, which accumulates in a different order and blocks differently, so the
    two agree to a few ulps rather than bit-for-bit.  Measured ~1e-16 relative at
    n=37 and ~1e-15 at n=640; the gate is a few ulps of the largest entry, which
    is the strongest statement that is actually true.

    A REAL right-hand side does stay bit-identical, and that matters: it is the
    path the Nφ=1 axisymmetric reduction takes.
    """
    rng = np.random.default_rng(0)
    for n in (37, 60, 640):
        M = rng.standard_normal((n, n))
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        got = ops3.apply_mode_block(M, v)
        ref = M @ v
        assert got.dtype == np.complex128
        ulps = np.max(np.abs(got - ref)) / np.spacing(np.max(np.abs(ref)))
        print(f"\n[matvec] n={n}: {ulps:.1f} ulp")
        assert ulps < 64.0, f"real/imag split moved the matvec by {ulps:.1f} ulp"
        # the real path is bit-for-bit untouched
        vr = rng.standard_normal(n)
        assert np.array_equal(ops3.apply_mode_block(M, vr), M @ vr)


def test_solve_mode_block_matches_solve_equilibrated():
    """``solve_mode_block`` reproduces ``operators_abt.solve_equilibrated``.

    Not bit-for-bit in the complex case — a real ``gesv`` on two columns is a
    different LAPACK path from a complex ``gesv`` on one — but to the accuracy
    the equilibrated block supports, and BIT-for-bit for a real right-hand side
    (that is the path the Nφ=1 axisymmetric reduction takes).
    """
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    M0, _, _ = ops3.mode_operators(A, B, DA1, DB1, 2.5,
                                   ops3.fourier_modes(GRID["Nphi"]))
    rng = np.random.default_rng(1)
    for mi, M in enumerate(M0):
        n = M.shape[0]
        yr = rng.standard_normal(n)
        assert np.array_equal(ops3.solve_mode_block(M, yr),
                              ops.solve_equilibrated(M, yr)), "real path moved"
        y = yr + 1j * rng.standard_normal(n)
        got = ops3.solve_mode_block(M, y)
        ref = ops.solve_equilibrated(M, y)
        rel = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        # the block is deliberately ill-conditioned before equilibration; judge
        # the two solves by the residual they leave, in the equilibrated norm
        s = np.max(np.abs(M), axis=1)
        s = np.where(s > 0.0, s, 1.0)
        res = np.max(np.abs((M @ got - y) / s)) / np.max(np.abs(y / s))
        print(f"\n[solve] mi={mi} cond={np.linalg.cond(M):.2e} "
              f"|x-x_ref|/|x|={rel:.2e} equil-residual={res:.2e}")
        assert res < 1e-12, f"split solve leaves residual {res:.2e}"


def test_real_split_leaves_the_solve_unchanged():
    """A full certified solve is unmoved by the real/imaginary split.

    Same Newton steps, same GMRES counts, same field.  Compared against a
    monkeypatched build in which both helpers fall back to the promoting form.
    """
    prob = s3.make_problem(**GRID)
    sl = _slice()
    U_fast, i_fast = nk.newton_solve_nk(prob, sl, tol=1e-11, max_iter=12)

    apply_orig, solve_orig, lu_orig = (ops3.apply_mode_block, ops3.solve_mode_block,
                                       nk._lu_solve_equilibrated)
    try:
        ops3.apply_mode_block = lambda M, v: np.asarray(M) @ np.asarray(v)
        ops3.solve_mode_block = ops.solve_equilibrated

        def _lu_promoting(fac, rhs):
            lu, piv, scale = fac
            import scipy.linalg as sla
            return sla.lu_solve((lu, piv), np.asarray(rhs) / scale)

        nk._lu_solve_equilibrated = _lu_promoting
        U_slow, i_slow = nk.newton_solve_nk(prob, sl, tol=1e-11, max_iter=12)
    finally:
        ops3.apply_mode_block, ops3.solve_mode_block = apply_orig, solve_orig
        nk._lu_solve_equilibrated = lu_orig

    d = np.max(np.abs(np.asarray(U_fast) - np.asarray(U_slow))) / np.max(np.abs(U_slow))
    print(f"\n[split-e2e] steps {i_slow.iters}->{i_fast.iters}  "
          f"gmres {i_slow.gmres_iters}->{i_fast.gmres_iters}  "
          f"equilR {i_slow.residual_norm:.3e}->{i_fast.residual_norm:.3e}  "
          f"|dU|/|U|={d:.2e}")
    assert i_fast.iters == i_slow.iters
    assert i_fast.gmres_iters == i_slow.gmres_iters
    assert d < 1e-12, f"converged field moved by {d:.2e}"


# ==========================================================================
# Stage 3.  One operator per grid, not one per slice
# ==========================================================================
def test_operator_cache_returns_the_same_blocks():
    """The cache is a cache: same grid and separation ⇒ the very same arrays.

    Identity, not equality — the point is that nothing was rebuilt.  A different
    separation is a genuine miss, because the raw (un-equilibrated) block does
    depend on b.
    """
    ops3.clear_block_cache()
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    m_vals = ops3.fourier_modes(GRID["Nphi"])
    first = ops3.mode_operators_cached(A, B, DA1, DB1, 2.5, m_vals)
    again = ops3.mode_operators_cached(A, B, DA1, DB1, 2.5, m_vals)
    assert all(x is y for x, y in zip(first[0], again[0])), "blocks were rebuilt"
    assert first[3] is again[3], "row scales were rebuilt"
    other = ops3.mode_operators_cached(A, B, DA1, DB1, 7.0, m_vals)
    assert all(x is not y for x, y in zip(first[0], other[0])), \
        "a different separation must not hit"


def test_cached_blocks_equal_a_fresh_build_and_are_read_only():
    """Cached blocks are bit-identical to a fresh build, and immutable.

    They are shared between every assembly at that separation, so a caller that
    mutated one in place would corrupt every other solve.  The Newton steps copy
    before adding their nonlinear diagonal; the write flag makes any future
    caller that forgets fail loudly instead of silently.
    """
    ops3.clear_block_cache()
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    m_vals = ops3.fourier_modes(GRID["Nphi"])
    ref, w_ref, int_ref = ops3.mode_operators(A, B, DA1, DB1, 2.5, m_vals)
    got, w_got, int_got, scales = ops3.mode_operators_cached(A, B, DA1, DB1, 2.5, m_vals)
    for mi in range(len(ref)):
        assert np.array_equal(ref[mi], got[mi]), f"cached block m={m_vals[mi]} differs"
        assert not got[mi].flags.writeable, "cached block is writable"
        s = np.max(np.abs(ref[mi]), axis=1)
        assert np.array_equal(scales[mi], np.where(s > 0.0, s, 1.0)), "scales differ"
    assert np.array_equal(int_ref, int_got)


def test_cache_ignores_a_non_canonical_grid():
    """A caller with a hand-built grid gets its own operator, never a cache hit.

    The key is ``(Na, Nb, Nφ, b)``, which only identifies the operator if the 1-D
    grid is the canonical one for that size.  Feed it a perturbed grid and the
    result must follow the grid, not the key.
    """
    ops3.clear_block_cache()
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    m_vals = ops3.fourier_modes(GRID["Nphi"])
    canon = ops3.mode_operators_cached(A, B, DA1, DB1, 2.5, m_vals)[0]
    DB1_odd = np.asarray(DB1) * 1.000001
    odd = ops3.mode_operators_cached(A, B, DA1, DB1_odd, 2.5, m_vals)[0]
    assert not np.allclose(canon[0], odd[0]), "a non-canonical grid was served from cache"
    again = ops3.mode_operators_cached(A, B, DA1, DB1, 2.5, m_vals)[0]
    assert np.array_equal(canon[0], again[0]), "the canonical entry was evicted or poisoned"


def test_repeated_assembly_at_one_separation_is_reused():
    """A second slice at the same b reuses the operator and only rebuilds the source."""
    ops3.clear_block_cache()
    prob = s3.make_problem(**GRID)
    a1 = s3.assemble(prob, _slice(b=2.5))
    a2 = s3.assemble(prob, Slice3D(b=2.5, m_A=0.5, m_B=0.5,
                                   S_A_vec=(0.02, -0.03, 0.01)))
    assert all(x is y for x, y in zip(a1.M0, a2.M0)), "operator rebuilt for a new slice"
    assert a1.scales is a2.scales
    assert not np.array_equal(a1.A2, a2.A2), "the source should NOT be shared"
    # and the certified monitor reads the scales off the assembly
    assert nk._block_scales(a1) is a1.scales


# ==========================================================================
# Stage 4.  The separable operator and the fast-diagonalization inverse
# ==========================================================================
def test_separable_action_matches_the_dense_block():
    """GATE 1 — the matrix-free action IS the dense block, every m, ≤1e-11 relative.

    This is the claim the whole module rests on: the per-m block is an exact
    generalized Kronecker sum, so applying it through the 1-D factors is not an
    approximation.  Checked on the BC rows too, which the Kronecker form does not
    cover and ``apply`` writes out in closed form.
    """
    S = separable.get_separable(**GRID)
    rng = np.random.default_rng(3)
    for b in (2.5, 7.0):
        M0, _, _, _ = ops3.mode_operators_cached(S.A, S.B, S.DA1, S.DB1, b, S.m_vals)
        for mi, m in enumerate(S.m_vals):
            V = rng.standard_normal((S.Na1, S.Nb1))
            ref = (np.asarray(M0[mi]) @ V.ravel()).reshape(S.Na1, S.Nb1)
            rel = np.max(np.abs(ref - S.apply(mi, V, b))) / np.max(np.abs(ref))
            print(f"\n[sep-apply] b={b} m={m}: rel={rel:.2e}")
            assert rel < 1e-11, f"separable action off by {rel:.2e} (m={m}, b={b})"


def test_separable_solve_inverts_the_block():
    """GATE 2 — fast diagonalization inverts the block, judged in the right norm.

    The raw block has condition number ~1e13–1e15 (the reason the solver
    equilibrates at all), so no float64 route round-trips to 1e-16 and comparing
    solution VECTORS would measure the conditioning, not the method.  What is
    asserted is the residual ``M x − y`` in the EQUILIBRATED norm — the norm the
    Newton solve actually controls — against the dense
    ``operators_abt.solve_equilibrated`` measured the same way.
    """
    S = separable.get_separable(**GRID)
    rng = np.random.default_rng(4)
    for b in (2.5, 7.0):
        M0, _, _, scales = ops3.mode_operators_cached(S.A, S.B, S.DA1, S.DB1, b, S.m_vals)
        for mi, m in enumerate(S.m_vals):
            M = np.asarray(M0[mi])
            s = scales[mi]
            y = rng.standard_normal((S.Na1, S.Nb1))
            scale = np.max(np.abs(y.ravel() / s))

            def equil_res(x):
                return np.max(np.abs((M @ np.asarray(x).ravel() - y.ravel()) / s)) / scale

            r_fdm = equil_res(S.solve(mi, y, b))
            r_dense = equil_res(ops.solve_equilibrated(M, y.ravel()))
            print(f"\n[sep-solve] b={b} m={m}: fdm={r_fdm:.2e} dense={r_dense:.2e} "
                  f"cond(M0)={np.linalg.cond(M):.1e} "
                  f"cond(S_A,S_B)={S.fdm[mi].cond[0]:.1e},{S.fdm[mi].cond[1]:.1e}")
            assert r_fdm < 1e-9, f"FDM leaves residual {r_fdm:.2e} (m={m}, b={b})"


def test_separable_row_scales_match_block_scales():
    """GATE 3 — the row scales computed from the 1-D factors are the block's.

    ``solver_3d_nk.equil_residual_inf`` divides by these, so they are part of the
    certified number.  Getting them in O(Na·Nb) instead of by scanning an
    (Na·Nb)² block is what lets the separable route never form one.
    """
    S = separable.get_separable(**GRID)
    for b in (2.5, 7.0):
        _, _, _, scales = ops3.mode_operators_cached(S.A, S.B, S.DA1, S.DB1, b, S.m_vals)
        for mi, m in enumerate(S.m_vals):
            rel = np.max(np.abs(S.row_scales(b)[mi] - scales[mi])) / np.max(scales[mi])
            print(f"\n[sep-scales] b={b} m={m}: rel={rel:.2e}")
            assert rel < 1e-11, f"row scales off by {rel:.2e} (m={m}, b={b})"


def test_u_row_scales_attain_the_inv_wmin_closed_form():
    """The ``u``/``v`` norm ratio is ``1/w_min``, from the GRID alone — and attained.

    ``scales_u[j] = max_k |M0_m[j,k]|/w_k`` against ``scales[j] = max_k |M0_m[j,k]|``.
    Since ``w_k >= w_min > 0`` at every node the ratio is bounded by ``1/w_min`` row
    by row; the bound is ATTAINED because ``w = (1−B²)^{|m|/2}`` is smallest at the
    outermost B node while the m-block's largest entries are the centrifugal
    ``m²·Den/(4A²)`` rows, largest toward that same axis.  ``bc_factor`` takes no
    ``b``, no ``Na`` and no physical parameter, so

        1/w_min = (1 − max_j B_j²)^{−m_max/2}

    is closed form in ``(Nb, Nφ)``.  **That is what prices the norm change without a
    re-solve**: every residual ever recorded in the ``v`` norm moves by exactly this
    factor when requoted in ``u``, so no historical number needs re-measuring to be
    converted — which is the reversibility the two-norm design exists for.
    """
    S = separable.get_separable(**GRID)
    closed = ops3.inv_wmin(S.B, GRID["Nphi"])
    for b in (2.5, 7.0):
        M0, w, _, scales = ops3.mode_operators_cached(S.A, S.B, S.DA1, S.DB1, b, S.m_vals)
        su = ops3.u_row_scales(M0, w)
        got = max(float(np.max(su[mi] / scales[mi])) for mi in range(S.m_vals.size))
        print(f"\n[u-norm] b={b}: max(scales_u/scales)={got:.6e} "
              f"1/w_min(closed)={closed:.6e} ratio={got / closed:.9f}")
        assert got <= closed * (1 + 1e-12), (
            f"the bound scales_u/scales <= 1/w_min is VIOLATED: {got:.6e} > {closed:.6e}")
        assert abs(got / closed - 1.0) < 1e-6, (
            f"the bound is no longer ATTAINED ({got / closed:.9f}); the closed form "
            "then over-prices the norm change and cannot convert a recorded residual")


def test_m0_block_is_identical_in_both_norms():
    """``m = 0`` has ``w ≡ 1``, so the norm change cannot move the axisymmetric rung.

    Load-bearing for the whole two-norm design: every bit-for-bit reduction gate in
    this leaf and in the curved sibling is measured at ``Nφ = 1`` or on the ``m = 0``
    block, so none of them moves when the monitor's norm is selected differently.
    Asserted as exact array equality, not a tolerance.
    """
    S = separable.get_separable(**GRID)
    for b in (2.5, 7.0):
        M0, w, _, scales = ops3.mode_operators_cached(S.A, S.B, S.DA1, S.DB1, b, S.m_vals)
        su = ops3.u_row_scales(M0, w)
        print(f"\n[u-norm m0] b={b}: scales_u[0] == scales[0]: "
              f"{np.array_equal(su[0], scales[0])}")
        assert np.array_equal(su[0], scales[0]), "the m=0 block MOVED under the u norm"


def test_separable_row_scales_u():
    """GATE — the ``u`` row scales from the 1-D factors are the dense block's.

    The **only** thing tying ``separable.row_scales_u``'s O(Na·Nb) derivation to
    ``operators_3d.u_row_scales``'s definition.  It is a genuine derivation and not
    a rescaling of :meth:`row_scales`: the ``u`` norm divides by column while a row
    scale maxes over columns, so the B-block's off-diagonal term needs its own
    ``max_l |C_B[j,l]|/w_l`` and is not ``offB[j]/w_j``.  Getting it wrong is
    invisible on the default norm — nothing divides by these until the norm is
    selected — so this gate is what makes it visible.
    """
    S = separable.get_separable(**GRID)
    for b in (2.5, 7.0):
        M0, w, _, _ = ops3.mode_operators_cached(S.A, S.B, S.DA1, S.DB1, b, S.m_vals)
        su_dense = ops3.u_row_scales(M0, w)
        su_sep = S.row_scales_u(b)
        for mi, m in enumerate(S.m_vals):
            rel = np.max(np.abs(su_sep[mi] - su_dense[mi])) / np.max(su_dense[mi])
            print(f"\n[sep-scales-u] b={b} m={m}: rel={rel:.2e}")
            assert rel < 1e-11, f"u row scales off by {rel:.2e} (m={m}, b={b})"


def test_equil_residual_inf_honours_the_norm_argument():
    """Both norms are reachable on BOTH linear-operator routes, and differ as derived.

    A separable assembly has ``M0 = None``, so the ``u`` norm there can only come
    from the Kronecker factors — this is the gate that the separable route carries
    the norm at all.  At ``Nφ = 1`` the two must be bit-identical.
    """
    prob = s3.make_problem(**GRID)
    sl = _slice()
    closed = ops3.inv_wmin(prob.B, GRID["Nphi"])
    for sep in (False, True):
        asm = s3.assemble(prob, sl, separable=sep)
        U, _ = nk.newton_solve_nk(prob, sl, asm=asm, tol=1e-10, max_iter=10)
        Uf = np.asarray(U).reshape(asm.interior.size, GRID["Nphi"])
        rv = nk.equil_residual_inf(asm, Uf, norm="v")
        ru = nk.equil_residual_inf(asm, Uf, norm="u")
        print(f"\n[norm-arg] separable={sep}: v={rv:.4e} u={ru:.4e} "
              f"v/u={rv / ru:.4e} (1/w_min={closed:.4e})")
        assert 1.0 <= rv / ru <= closed * (1 + 1e-9), (
            f"v/u = {rv / ru:.4e} outside [1, 1/w_min={closed:.4e}]")
    with pytest.raises(ValueError):
        nk.equil_residual_inf(asm, Uf, norm="w")

    prob1 = s3.make_problem(Na=GRID["Na"], Nb=GRID["Nb"], Nphi=1)
    sl1 = Slice3D(b=4.0, m_A=0.5, m_B=0.5, P_A_vec=(0.0, 0.0, -0.2),
                  P_B_vec=(0.0, 0.0, 0.2))
    asm1 = s3.assemble(prob1, sl1, separable=False)
    U1, _ = nk.newton_solve_nk(prob1, sl1, asm=asm1, tol=1e-10, max_iter=10)
    U1f = np.asarray(U1).reshape(asm1.interior.size, 1)
    a = nk.equil_residual_inf(asm1, U1f, norm="v")
    c = nk.equil_residual_inf(asm1, U1f, norm="u")
    print(f"\n[norm-arg] Nphi=1: v={a:.17e} u={c:.17e} identical={a == c}")
    assert a == c, "at Nphi=1 (w == 1) the two norms must be bit-identical"


def test_default_monitor_norm_is_still_v():
    """The default norm is ``"v"``, and flipping it is NOT a one-line change.

    Frederik's 2026-08-24 ruling adopts the ``u`` norm in both leaves, but the flip
    is not separable from re-deriving the threshold it is read against: under ``u``
    every recorded residual reads ``1/w_min`` smaller — ``3.3e+04`` at the production
    grid — for reasons that have nothing to do with the solve being better.  So a
    ``u`` residual checked against the ``v``-calibrated ``CERT_TOL = 1e-10`` would
    read as a four-order improvement that is purely a change of units, which is the
    *mixing* failure mode the ruling's reversibility constraint exists to prevent.

    This gate is deliberately a tripwire rather than a claim about which norm is
    right: when the threshold is re-derived, change this test **in the same commit**
    that flips the default and re-points the gates.  A default flipped on its own
    silently reinterprets every published number in ``paper/``.
    """
    print(f"\n[norm-default] EQUIL_NORM_DEFAULT={ops3.EQUIL_NORM_DEFAULT!r} "
          f"CERT_TOL={CERT_TOL:.1e}")
    assert ops3.EQUIL_NORM_DEFAULT == "v", (
        "the monitor's default norm changed; the threshold it is read against must "
        "change in the same commit — see the docstring")
    assert nk.EQUIL_NORM_DEFAULT == ops3.EQUIL_NORM_DEFAULT, (
        "solver_3d_nk's re-export drifted from operators_3d's definition")


def test_separable_factors_do_not_depend_on_b():
    """The 1-D factors carry no separation — the setup is once per grid, ever.

    ``M0_m(b) = D(b)·M0_m(1)`` with D diagonal, so b enters ``apply``/``solve``
    only through the scalar row factor ``pref1/b²``.
    """
    S = separable.get_separable(**GRID)
    for b in (2.5, 7.0, 11.0):
        rel = (np.max(np.abs(separable.pref_rows(S.A, S.B, b) * b ** 2 - S.pref1))
               / np.max(S.pref1))
        print(f"\n[sep-b] b={b}: |pref(b)·b² − pref(1)|/|pref(1)| = {rel:.2e}")
        assert rel < 1e-15


def test_separable_polish_certifies_and_matches_the_dense_one():
    """GATE 4 — a warm-started polish certifies under the EXISTING monitor.

    Same ``equil_residual_inf``, same ``CERT_TOL`` imported from the one place it
    lives; only the linear algebra inside the Newton step differs.  The field must
    match an independently converged iterate, and the iteration count must not
    blow up — the separable preconditioner drops the nonlinear diagonal, which is
    measured to cost one or two GMRES iterations per step, not a Newton step.
    """
    prob = s3.make_problem(**GRID)
    sl = _slice()
    U_star, i_star = nk.newton_solve_nk(prob, sl, tol=1e-13, max_iter=15)
    U_star = np.asarray(U_star).reshape(prob.Ntot2d, prob.Nphi)
    sl_near = Slice3D(b=sl.b * 1.08, m_A=sl.m_A, m_B=sl.m_B, P_A_vec=sl.P_A_vec,
                      P_B_vec=sl.P_B_vec, S_A_vec=sl.S_A_vec, S_B_vec=sl.S_B_vec)
    U0, _ = nk.newton_solve_nk(prob, sl_near, tol=1e-13, max_iter=15)
    U0 = np.asarray(U0).reshape(prob.Ntot2d, prob.Nphi)

    out = {}
    for label, kw in (("dense", {}), ("separable", dict(separable=True))):
        U, info = nk.evaluate_polished_nk(prob, sl, U0, newton_steps=4,
                                          tol=CERT_TOL, **kw)
        out[label] = (np.asarray(U).reshape(prob.Ntot2d, prob.Nphi), info)
        print(f"\n[sep-polish] {label}: steps={info.iters} gmres={info.gmres_iters} "
              f"equilR={info.residual_norm:.3e} certified={info.converged}")
        assert info.converged, f"{label} polish did not certify ({info.residual_norm:.2e})"
    Ud, Us = out["dense"][0], out["separable"][0]
    d_star = np.max(np.abs(Us - U_star)) / np.max(np.abs(U_star))
    d_dense = np.max(np.abs(Us - Ud)) / np.max(np.abs(Ud))
    print(f"[sep-polish] |U_sep-U_converged|/|U|={d_star:.2e}  "
          f"|U_sep-U_dense|/|U|={d_dense:.2e}")
    assert d_star < 1e-9, f"separable polish landed {d_star:.2e} from the solution"
    assert d_dense < 1e-9
    # the cheap preconditioner may cost GMRES iterations; it must not cost Newton steps
    assert out["separable"][1].iters <= out["dense"][1].iters + 1


def test_separable_axisym_reduction_reproduces_2d():
    """Nφ=1: the separable route still reproduces the frozen 2-D Newton.

    The companion to ``test_solver_3d.py::test_nk_axisym_reduction_reproduces_2d``,
    and the reason that test was left alone.  That gate asserts TWO things — the
    same answer, and one GMRES iteration per Newton step — and the second is a
    statement about the *exact* preconditioner: at Nφ=1 the block-diagonal
    operator IS the full Jacobian, so an exact solve takes one iteration.  The
    separable preconditioner deliberately drops the nonlinear diagonal, so it is
    NOT the exact Jacobian at any Nφ and the count rises.  The gate is therefore
    split rather than relaxed: the exact path keeps both assertions, and this test
    makes the same-answer claim for the separable path.
    """
    P = 0.5
    for (b, mA, mB) in [(1.0, 0.5, 0.5), (1.5, 0.7, 0.3)]:
        prob2 = sa.make_problem(Na=28, Nb=20, P=P)
        U2, i2 = sa.newton_solve(prob2, Slice(b=b, m_A=mA, m_B=mB),
                                 tol=1e-12, max_iter=25)
        prob3 = s3.make_problem(Na=28, Nb=20, Nphi=1)
        sl3 = Slice3D.head_on(b=b, m_A=mA, m_B=mB, P=P)
        U3, i3 = nk.newton_solve_nk(prob3, sl3, tol=1e-12, max_iter=25, separable=True)
        d = float(np.max(np.abs(np.asarray(U3)[:, :, 0] - np.asarray(U2))))
        print(f"\n[sep-A] b={b}: 2D its={i2.iters} sep its={i3.iters} "
              f"|U3-U2|={d:.2e} gmres={i3.gmres_iters} equilR={i3.residual_norm:.2e}")
        assert d < 1e-12, f"separable axisym reduction off by {d:.2e} (b={b})"
        assert all(g >= 1 for g in i3.gmres_iters)


def test_separable_assembly_builds_no_dense_block():
    """A separable assembly holds 1-D factors and no ``(Na·Nb)²`` matrix at all.

    That is the storage claim (0.1 MiB against 79 MiB at the production grid) and
    the reason the route is not a wall at higher meridian resolution.  The
    modified-Newton step needs the dense blocks and must say so clearly rather
    than fail on a ``None``.
    """
    prob = s3.make_problem(**GRID)
    asm = s3.assemble(prob, _slice(), separable=True)
    assert asm.M0 is None
    assert asm.sep is not None and asm.b == _slice().b
    dense = s3.assemble(prob, _slice())
    assert np.array_equal(asm.interior, dense.interior)
    for mi in range(prob.m_vals.size):
        assert np.array_equal(asm.w[mi], dense.w[mi]), "B-factor differs"
    with pytest.raises(ValueError, match="dense per-m blocks"):
        s3.newton_step(asm, np.zeros((prob.Ntot2d, prob.Nphi)))


def test_separable_tangent_matches_the_dense_tangent():
    """Differentiability survives: the same dU/dθ from either linear-algebra route.

    The tangent solve reuses the Newton Jacobian and preconditioner, so it has to
    follow the operator representation.  Two things make this safe, and both are
    asserted here rather than argued: the separable setup depends on the grid
    alone — not on b, not on θ, not on the iterate — so no gradient is ever taken
    through an eigendecomposition, and the action itself is plain matmuls.
    """
    prob = s3.make_problem(Na=20, Nb=14, Nphi=6)
    sl = Slice3D(b=2.5, m_A=0.6, m_B=0.4, P_A_vec=(0.05, 0.11, -0.02),
                 P_B_vec=(-0.05, -0.11, 0.02), S_A_vec=(0.05, 0.0, 0.03),
                 S_B_vec=(0.0, 0.04, -0.01))
    U, info = nk.newton_solve_nk(prob, sl, tol=1e-13, max_iter=15)
    U = np.asarray(U).reshape(prob.Ntot2d, prob.Nphi)
    for name in ("b", "S_Ax"):
        tans = []
        for separable in (False, True):
            asm = s3.assemble(prob, sl, separable=separable)
            t = s3d.certified_tangent_3d(prob, U, sl, name, M_tot=1.0, asm=asm,
                                         jac="nk")
            tans.append(np.asarray(t).reshape(prob.Ntot2d, prob.Nphi))
        rel = np.max(np.abs(tans[1] - tans[0])) / np.max(np.abs(tans[0]))
        print(f"\n[sep-tangent] d/d{name}: |sep-dense|/|dense| = {rel:.2e}")
        assert rel < 1e-8, f"separable tangent d/d{name} off by {rel:.2e}"


# ==========================================================================
# The cross-leaf API surface — what another package depends on
# ==========================================================================
def test_axisym_blocks_still_exists_for_the_curved_leaf():
    """``axisym_blocks`` is public API with a consumer OUTSIDE this package.

    Nothing in this package calls it any more — :func:`ops3.meridian_geometry`
    returns the coordinates without assembling a dense operator to discard — but
    ``lemaitre.initial_data.curved_puncture.operators.make_chart`` unpacks this
    exact 8-tuple, and the family's dependency direction is
    ``curved_puncture -> conformally_flat_puncture`` (see ``Lemaitre/CLAUDE.md``).

    This test exists because deleting the function as "unused" made the entire
    curved leaf unimportable, and NOTHING in this leaf's 611-test suite noticed:
    a leaf's suite tests its own package, so it cannot see its own consumers.
    The contract is pinned here, in the leaf that OWNS the symbol, because that
    is where a change to it originates.

    Asserted: the arity and order of the returned tuple, and that the pieces
    agree with the routes that replaced it — a stub returning eight ``None``s
    would satisfy a mere ``hasattr``.
    """
    A, B, DA1, DB1 = ops.build_grid(GRID["Na"], GRID["Nb"])
    b = 2.5
    out = ops3.axisym_blocks(A, B, DA1, DB1, b)
    assert len(out) == 8, f"axisym_blocks arity changed: {len(out)}"
    Lap, rho, z, Af, Bf, DA, DB, inv_rho2 = out

    Lap_ref, rho_ref, z_ref, Af_ref, Bf_ref, DA_ref, DB_ref = ops.laplacian_matrix(
        A, B, DA1, DB1, b)
    for name, ref, got in (("Lap", Lap_ref, Lap), ("rho", rho_ref, rho),
                           ("z", z_ref, z), ("Af", Af_ref, Af),
                           ("Bf", Bf_ref, Bf), ("DA", DA_ref, DA),
                           ("DB", DB_ref, DB)):
        assert np.array_equal(np.asarray(ref), np.asarray(got)), f"{name} drifted"
    # the 8th slot is the one make_chart actually reads
    assert np.array_equal(inv_rho2, ops3.meridian_geometry(A, B, b)[4])


def test_curved_leaf_can_build_its_chart():
    """The curved leaf imports and builds a chart against this solver.

    The end-to-end version of the test above: it is the call that broke, so it is
    the call that is checked.  Skipped rather than failed when the curved leaf is
    not installed — it is an optional sibling, and this leaf must never import it
    (``tests/test_self_containment.py`` enforces the one-way dependency, so this
    import lives in a test, never in ``src/``).
    """
    co = pytest.importorskip(
        "lemaitre.initial_data.curved_puncture.operators",
        reason="LMID-curved-puncture not installed in this environment")
    chart = co.make_chart(Na=12, Nb=10, Nphi=4, b=2.0)
    assert chart is not None


# ==========================================================================
# Stage 5.  Off-grid evaluation IS the truncated expansion
# ==========================================================================
# The discrete approximation space is
#
#     u = Σ_m Σ_{n,l} c^m_{nl} T_n(2A−1) P_l(B) (1−B²)^{|m|/2} e^{imφ},
#
# so the physical mode u_m carries the associated-Legendre factor and is NOT a
# polynomial in B — for odd m not even smooth at B=±1.  ``evaluate_field``
# therefore interpolates u_m/w_m and restores w_m at the query point.  These
# gates pin that it reproduces the expansion EXACTLY, which is the claim the
# paper makes for it; the pre-factoring evaluator interpolated the physical
# nodal values directly and satisfied it only at m=0.
def _basis_field(A, B, phi, coeffs):
    """The expansion above, evaluated in closed form at arbitrary ``(A,B,φ)``.

    ``coeffs`` is ``(cos_c, sin_c)``: ``cos_c[k][n][l]`` multiplies
    ``T_n(2A−1) P_l(B) (1−B²)^{k/2} cos(kφ)``, ``sin_c`` likewise with ``sin``.
    Real by construction, and exactly the real form of the complex expansion.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    phi = np.asarray(phi, dtype=float)
    cos_c, sin_c = coeffs
    out = np.zeros(np.broadcast(A, B, phi).shape)
    for trig, table in ((np.cos, cos_c), (np.sin, sin_c)):
        for k, c_k in table.items():
            w = (1.0 - B ** 2) ** (abs(k) / 2.0) if k else np.ones_like(B)
            radial = np.zeros_like(out)
            for n in range(c_k.shape[0]):
                Tn = np.cos(n * np.arccos(np.clip(2.0 * A - 1.0, -1.0, 1.0)))
                for l in range(c_k.shape[1]):
                    Pl = np.polynomial.legendre.legval(
                        B, np.eye(c_k.shape[1])[l])
                    radial = radial + c_k[n, l] * Tn * Pl
            out = out + w * radial * trig(k * phi)
    return out


def _random_basis_coeffs(Na, Nb, Nphi, seed=17):
    """Random coefficients spanning the whole retained space, cos and sin."""
    rng = np.random.default_rng(seed)
    K = Nphi // 2                       # highest cosine wavenumber (Nyquist)
    Kp = (Nphi - 1) // 2                # highest sine wavenumber (no Nyquist sine)
    cos_c = {k: rng.normal(size=(Na + 1, Nb)) * 0.1 for k in range(K + 1)}
    sin_c = {k: rng.normal(size=(Na + 1, Nb)) * 0.1 for k in range(1, Kp + 1)}
    return cos_c, sin_c


def test_w_factor_matches_the_operator_assembly():
    """``solver_3d._w_factor`` is ``operators_3d.bc_factor``'s factor.

    The evaluator needs the factor at query points, where ``bc_factor``'s
    B-derivatives diverge for odd m, so it carries its own expression.  Two
    expressions for one quantity is exactly how a factor drifts, so pin them.
    """
    _, B, _, _ = ops.build_grid(GRID["Na"], GRID["Nb"])
    for m in range(0, 5):
        assert np.array_equal(s3._w_factor(B, m), ops3.bc_factor(B, m)[0]), \
            f"_w_factor disagrees with bc_factor at m={m}"


@pytest.mark.parametrize("Nphi", [1, 4, 6, 8])
def test_evaluate_field_reproduces_the_truncated_expansion(Nphi):
    """Off-grid evaluation == Eq. (expansion) in closed form, to machine precision.

    Decisive for odd m: with the physical nodal values interpolated directly,
    the (1−B²)^{1/2} branch point of m=1 is resolved only algebraically and this
    gate fails by orders of magnitude near the outer axis.
    """
    Na, Nb, b = GRID["Na"], GRID["Nb"], 2.5
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    coeffs = _random_basis_coeffs(Na, Nb, Nphi)

    AA, BB, PP = np.meshgrid(prob.A, prob.B, prob.phi, indexing="ij")
    U = _basis_field(AA, BB, PP, coeffs)

    # query points, including some pushed hard against the outer axis B→±1
    rng = np.random.default_rng(5)
    A_q = rng.uniform(0.08, 0.92, size=60)
    B_q = np.concatenate([rng.uniform(-0.9, 0.9, size=40),
                          np.array([0.97, -0.97, 0.995, -0.995,
                                    0.999, -0.999, 0.9999, -0.9999,
                                    0.99999, -0.99999, 1.0 - 1e-9,
                                    -(1.0 - 1e-9), 0.5, -0.5,
                                    0.999999, -0.999999, 0.9, -0.9,
                                    0.98, -0.98])])
    phi_q = rng.uniform(0.0, 2.0 * np.pi, size=60)
    rho_q, z_q = ops.abt_map(A_q, B_q, b)

    got = s3.evaluate_field(prob, U, rho_q, z_q, phi_q, b)
    want = _basis_field(A_q, B_q, phi_q, coeffs)
    err = np.max(np.abs(got - want)) / max(1.0, np.max(np.abs(want)))
    print(f"[5] Nphi={Nphi} expansion reproduction err = {err:.3e}")
    assert err < 1e-11, f"evaluate_field is not the expansion: {err:.3e}"


def test_evaluate_field_recovers_the_nodal_values():
    """At a collocation point the evaluator returns the stored value.

    The cardinal property, and the one an exact-node branch is easiest to break:
    the factored route divides by w_m and multiplies it back, so a node hit must
    still round-trip.
    """
    Na, Nb, Nphi, b = GRID["Na"], GRID["Nb"], GRID["Nphi"], 2.5
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    rng = np.random.default_rng(9)
    U = rng.normal(size=prob.shape) * 1e-2

    idx = [(3, 5, 2), (1, 0, 0), (Na - 1, Nb - 1, Nphi - 1), (7, 9, 4)]
    A_q = np.array([prob.A[i] for i, _, _ in idx])
    B_q = np.array([prob.B[j] for _, j, _ in idx])
    phi_q = np.array([prob.phi[k] for _, _, k in idx])
    rho_q, z_q = ops.abt_map(A_q, B_q, b)

    got = s3.evaluate_field(prob, U, rho_q, z_q, phi_q, b)
    want = np.array([U[i, j, k] for i, j, k in idx])
    assert np.max(np.abs(got - want)) < 1e-12, f"node values not recovered: {got - want}"


def test_evaluate_field_axisymmetric_matches_the_2d_solver():
    """At Nφ=1 the evaluator is the frozen axisymmetric one (w≡1, m=0 only)."""
    Na, Nb, b = GRID["Na"], GRID["Nb"], 2.5
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=1)
    rng = np.random.default_rng(4)
    U = rng.normal(size=prob.shape) * 1e-2

    rho_q = np.array([0.7, 1.9, 3.4, 0.2])
    z_q = np.array([0.3, -1.1, 2.0, -0.4])
    got = s3.evaluate_field(prob, U, rho_q, z_q, np.zeros(4), b)

    prob2 = sa.make_problem(Na=Na, Nb=Nb)
    want = np.asarray(sa.evaluate_field_phys(prob2, U[:, :, 0], rho_q, z_q, b))
    assert np.max(np.abs(got - want)) < 1e-13, f"Nφ=1 drifted from 2-D: {got - want}"
