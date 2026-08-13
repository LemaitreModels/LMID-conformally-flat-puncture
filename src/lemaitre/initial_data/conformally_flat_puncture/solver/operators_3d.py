"""LM-initial-data-3D — Fourier-in-φ extension of the ABT / prolate-spheroidal patch.

The first non-axisymmetric LM-initial-data operator.  The axisymmetric two-centre code
(``operators_abt.py``, ``solver_abt.py``) is the **regression oracle**: its
numbers are what the 3-D reduction is checked against, so it is left alone even
though the repo's add-only policy is retired.  This module therefore owns its own
assembly outright — it does not route the 3-D operator through
``operators_abt.laplacian_matrix`` — and lifts the same prolate-spheroidal patch
to 3-D by appending an azimuthal Fourier axis.

The full 3-D flat Laplacian in prolate-spheroidal coordinates (φ the azimuthal
rotation angle, the meridian (ρ,z) part orthogonal and axisymmetric) is

    Δ u = Δ_axisym u + (1/ρ²) ∂²_φ u,

with the meridian operator exactly the dense prolate operator of
``operators_abt.laplacian_matrix``.  In a Fourier basis along φ (``Nφ``
equispaced collocation points, real FFT) the azimuthal term is **diagonal in
the azimuthal mode m**: ``∂²_φ → −m²``.  Hence the linear operator is
**block-diagonal — one 2-D (A,B) block per m**:

    L_m = Lap_axisym − m² diag(1/ρ²).

Each 2-D block is factored once (``operators_abt.solve_equilibrated``); the full
3-D dense matrix (size (Na·Nb·Nφ)²) is **never formed**.

This module owns the **dense** form of those blocks, and is the only place they
are built.  ``separable.py`` owns the matrix-free form — the same operator as 1-D
Kronecker factors, with an exact inverse — and neither needs the other.
:func:`mode_operators_cached` is the entry point the solver uses: it hands out
shared, read-only blocks memoized per ``(grid, b)``, because the blocks depend on
neither the masses, the momenta, the spins nor the iterate.

The 2-D derivative matrices are Kronecker products of the 1-D ones, so the
mixed-product rule ``(A⊗B)(C⊗D) = (AC)⊗(BD)`` with ``B=D=I`` gives::

    kron(D, I) @ kron(D, I) = kron(D @ D, I),   kron(I, D) @ kron(I, D) = kron(I, D @ D)

The left-hand side is a dense ``(Na·Nb)³`` matmul, the right an ``Na³`` (resp.
``Nb³``) one — at the production grid 6 Gflop against 0.0002 Gflop.  The
assembly here does **not** use it, on purpose: the identity holds exactly in
exact arithmetic but only to ~1e-16 relative in float64, and the m=0 block must
match the frozen ``operators_abt`` operator bit-for-bit (see
``_mode_independent_pieces``).  What the assembly does instead is compute each
squaring ONCE for the whole mode set rather than once per m.  The identity is
still pinned by ``tests/test_solver_3d_fast.py`` because the separable operator
of ``solver/separable.py`` is built on the same structure.

``ρ = b·2A/(1−A²)·√(1−B²)`` so ``1/ρ²`` is singular on the axis (A=0, the inner
segment |z|≤b; and B=±1, the outer axis).  The GL B-nodes already avoid B=±1;
the A=0 edge is a BC row.  On the inner axis only the m=0 mode is regular, so we
impose ``u_m = 0`` (Dirichlet) at A=0 for m≠0 — consistent with the existing A=0
Neumann row used for m=0.

The prolate B-operator is the **associated-Legendre operator**, whose regular
m-mode solution carries the factor ``(1−B²)^{|m|/2}`` at the outer axis B=±1.
For ODD m=1 that is a ``(1−B²)^{1/2}`` branch point — polynomial collocation of
the field itself converges only algebraically there.  We therefore solve for the
**smooth** factored unknown ``v_m`` with ``u_m = (1−B²)^{|m|/2} v_m``
(``block_operator_m_v``/``mode_operators``): the singular factor's B-derivatives
are analytic and only ``v_m`` is differentiated numerically, restoring spectral
convergence for every m.  For m=0 the factor is 1 and ``v_m≡u_m`` (the original
operator bit-for-bit).

Standalone: numpy + jax + the frozen sibling ``operators_abt``.
"""

from __future__ import annotations

import collections

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from . import operators_abt as ops


# --------------------------------------------------------------------------
# φ collocation grid and real-FFT azimuthal mode set
# --------------------------------------------------------------------------
def phi_grid(Nphi: int) -> np.ndarray:
    """``Nφ`` equispaced collocation points φ_k = 2π k/Nφ on [0, 2π)."""
    return 2.0 * np.pi * np.arange(Nphi) / Nphi


def fourier_modes(Nphi: int) -> np.ndarray:
    """Azimuthal mode indices produced by ``numpy.fft.rfft`` of length ``Nφ``.

    ``m = 0, 1, ..., Nφ//2`` (the real-FFT half-spectrum); ``∂²_φ → −m²``.
    """
    return np.arange(Nphi // 2 + 1)


def build_grid_3d(Na: int, Nb: int, Nphi: int):
    """Return ``(A, B, DA1, DB1, phi)``: the frozen ABT (A,B) grid + φ nodes.

    ``A, B, DA1, DB1`` come verbatim from ``operators_abt.build_grid`` so the
    meridian discretisation is identical to the 2-D code.
    """
    A, B, DA1, DB1 = ops.build_grid(Na, Nb)
    return A, B, DA1, DB1, phi_grid(Nphi)


# --------------------------------------------------------------------------
# The per-m 2-D block operator
# --------------------------------------------------------------------------
def meridian_geometry(A, B, b):
    """Node coordinates shared by every azimuthal mode — no operator built.

    Returns ``(rho, z, Af, Bf, inv_rho2)`` on the flattened (Na+1, Nb) node set.
    ``inv_rho2 = 1/ρ²`` is finite on the interior and set to 0 on the A=1 (ρ=∞)
    and A=0 (ρ=0) edges (those rows are BC rows, so the dummy value is never
    used).

    This is deliberately *only* the geometry.  It is what every caller of the
    older :func:`axisym_blocks` actually wanted: ``solver_3d.assemble``,
    ``parametric_nd_3d._assemble_cached``, two tests here and ``make_chart`` in
    the curved-puncture leaf all discarded that function's dense Laplacian and
    2-D derivative matrices and kept the coordinates — so a dense ``(Na·Nb)²``
    operator was being assembled per parameter point to produce ``rho`` and
    ``z``.  The per-mode blocks come from :func:`mode_operators`, which builds its
    own derivative matrices; nothing consumed the axisymmetric one.
    """
    AA, BB = np.meshgrid(np.asarray(A), np.asarray(B), indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    rho, z = ops.abt_map(Af, Bf, b)
    rho = np.asarray(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_rho2 = 1.0 / rho ** 2
    good = np.isfinite(inv_rho2) & (rho > 1e-14)
    inv_rho2 = np.where(good, inv_rho2, 0.0)
    return rho, z, Af, Bf, inv_rho2


def axisym_blocks(A, B, DA1, DB1, b):
    """Meridian pieces shared by every azimuthal mode — ``(Lap, rho, z, Af, Bf,
    DA, DB, inv_rho2)``, with ``Lap`` the dense axisymmetric prolate Laplacian.

    **Kept for a consumer outside this package.**  Nothing here calls it any more
    — :func:`meridian_geometry` returns the coordinates without assembling a
    dense ``(Na·Nb)²`` operator to throw away, which is where a large part of the
    old per-parameter-point assembly cost went.  But
    ``lemaitre.initial_data.curved_puncture.operators.make_chart`` unpacks this
    exact 8-tuple, and the family's dependency direction is
    ``curved_puncture -> conformally_flat_puncture``, so this is public API with a
    live external caller, not dead code.  Removing it made the whole curved leaf
    unimportable; ``tests/test_solver_3d_fast.py`` now pins the signature so that
    cannot happen silently again.

    Prefer :func:`meridian_geometry` in new code, and take ``Lap`` from
    ``operators_abt.laplacian_matrix`` if the dense operator is what you want.
    """
    Lap, rho, z, Af, Bf, DA, DB = ops.laplacian_matrix(A, B, DA1, DB1, b)
    _, _, _, _, inv_rho2 = meridian_geometry(A, B, b)
    return Lap, rho, z, Af, Bf, DA, DB, inv_rho2


def block_operator_m(Lap, inv_rho2, m: int) -> np.ndarray:
    """The raw (no-BC) 2-D operator for azimuthal mode m: ``Lap − m² diag(1/ρ²)``.

    The UNfactored form, on ``u_m`` rather than ``v_m``; the solver uses the
    factored :func:`block_operator_m_v` instead (spectral odd-m convergence).
    Kept as the plain statement of what the mode block is; feed it ``Lap`` from
    ``operators_abt.laplacian_matrix`` and ``inv_rho2`` from
    :func:`meridian_geometry`.
    """
    if m == 0:
        return np.array(Lap, dtype=float)
    L = np.array(Lap, dtype=float)
    L[np.diag_indices_from(L)] -= (m ** 2) * inv_rho2
    return L


# --------------------------------------------------------------------------
# Associated-Legendre basis factoring — spectral odd-m convergence
# --------------------------------------------------------------------------
# The prolate B-operator (1−B²)∂²_B − 2B∂_B − m²/(1−B²) is the **associated-
# Legendre operator** (the centrifugal coefficient m²·Den/(4A²) → m² as B→±1,
# since Den → 4A² there).  Its regular m-mode solution behaves as (1−B²)^{|m|/2}
# at the outer prolate axis B=±1.  For ODD m=1 that is a (1−B²)^{1/2} branch
# point, which polynomial (GL) collocation resolves only ALGEBRAICALLY.
#
# Fix: substitute  u_m(A,B) = (1−B²)^{|m|/2} · v_m(A,B), with v_m SMOOTH in B, and
# build the operator on v_m — differentiating only the smooth v numerically; the
# singular factor's B-derivatives are analytic.  The (1−B²)^{|m|/2−1} centrifugal
# singularity then cancels against the γ w'' term (exactly as B→±1, where
# Den/(4A²)→1), leaving a regular operator whose v converges SPECTRALLY.
#
# For m=0 the factor is w≡1 and v_m≡u_m, so the m=0 block is the original Lap
# bit-for-bit (and the axisymmetric reduction is unchanged).
def bc_factor(Bf, m: int):
    """The B-factor ``w=(1−B²)^{|m|/2}`` and its analytic B-derivatives ``w', w''``.

    ``Bf`` is the flattened B-node array.  For m=0 returns ``(1, 0, 0)``.
    """
    Bf = np.asarray(Bf, dtype=float)
    if m == 0:
        return np.ones_like(Bf), np.zeros_like(Bf), np.zeros_like(Bf)
    s = abs(int(m)) / 2.0
    oa = 1.0 - Bf ** 2
    w = oa ** s
    wp = s * oa ** (s - 1.0) * (-2.0 * Bf)
    wpp = (s * (s - 1.0) * oa ** (s - 2.0) * (4.0 * Bf ** 2)
           + s * oa ** (s - 1.0) * (-2.0))
    return w, wp, wpp


def _mode_independent_pieces(A, B, DA1, DB1, b):
    """Everything a per-m block is assembled from that does NOT depend on m.

    The four 2-D derivative matrices, the flattened node coordinates and the four
    prolate Laplacian coefficients.  Split out because :func:`mode_operators`
    used to rebuild all of it once per azimuthal mode — including the two dense
    ``(Na·Nb)³`` squarings, which are the same matrix every time.

    **The squarings are deliberately still the dense 2-D matmuls**, not the
    cheaper mixed-product form of the module docstring.  The identity is exact in
    exact arithmetic but not in float64 (the two orderings sum a different number
    of exactly-zero terms), and the m=0 block has to be the frozen axisymmetric
    ``operators_abt.laplacian_matrix`` operator BIT-FOR-BIT — that is what makes
    the Nφ=1 reduction reproduce the 2-D solver to ~1e-16 rather than ~1e-13, and
    ``tests/test_solver_3d.py::test_axisym_reduction_reproduces_2d`` measures it.
    Adopting the identity here alone moved that gate by 300× and desynchronised
    the two Newton paths at the residual floor.  It can only be taken in BOTH
    modules at once — see the note in ``mode_operators`` — and the payoff is
    small once the operator is built per model rather than per parameter point.
    """
    Na1, Nb1 = A.size, B.size
    IA, IB = np.eye(Na1), np.eye(Nb1)
    DA1 = np.asarray(DA1)
    DB1 = np.asarray(DB1)
    DA = np.kron(DA1, IB)
    DB = np.kron(IA, DB1)
    DAA = DA @ DA
    DBB = DB @ DB
    AA, BB = np.meshgrid(A, B, indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    rho, _ = ops.abt_map(Af, Bf, b)
    alpha, pcoef, gamma, qcoef = ops._coeffs(Af, Bf, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_rho2 = 1.0 / np.asarray(rho) ** 2
    inv_rho2 = np.where(np.isfinite(inv_rho2) & (np.asarray(rho) > 1e-14),
                        inv_rho2, 0.0)
    return dict(DA=DA, DB=DB, DAA=DAA, DBB=DBB, Af=Af, Bf=Bf,
                alpha=alpha, pcoef=pcoef, gamma=gamma, qcoef=qcoef,
                inv_rho2=inv_rho2)


def _block_from_pieces(p, m: int):
    """``(Mv, w)`` for one mode from the shared :func:`_mode_independent_pieces`."""
    alpha, pcoef = p["alpha"], p["pcoef"]
    gamma, qcoef = p["gamma"], p["qcoef"]
    DA, DB, DAA, DBB = p["DA"], p["DB"], p["DAA"], p["DBB"]
    cent = -(int(m) ** 2) * p["inv_rho2"]                   # centrifugal coeff
    w, wp, wpp = bc_factor(p["Bf"], m)
    # A-derivatives: w is constant in A, so it factors through (row scaling)
    Mv = (alpha * w)[:, None] * DAA + (pcoef * w)[:, None] * DA
    # B-derivatives: expand ∂_B(w v), ∂²_B(w v) analytically in w and numerically in v
    Mv += gamma[:, None] * (2.0 * wp[:, None] * DB + w[:, None] * DBB)
    Mv += qcoef[:, None] * (w[:, None] * DB)
    # the three node-diagonal contributions — gamma·w'' and qcoef·w' from the
    # expansion above, and the centrifugal cent·w·v — added on the diagonal
    # rather than through three dense diag() temporaries of (Na·Nb)² each
    diag = gamma * wpp + qcoef * wp + cent * w
    Mv[np.diag_indices_from(Mv)] += diag
    return Mv, w


def block_operator_m_v(A, B, DA1, DB1, b, m: int):
    """The factored 2-D operator for azimuthal mode m, acting on ``v_m`` (no BC).

    Returns ``(Mv, w)`` where ``Mv @ v_m == L_m[(1−B²)^{|m|/2} v_m] = L_m u_m`` and
    ``w`` is the B-factor at the flattened nodes.  For m=0 ``Mv`` is the original
    prolate Laplacian (``w≡1``).

    Assembles the shared pieces itself, so calling it in a loop over m repeats
    that work; :func:`mode_operators` shares them across the mode set instead.
    """
    return _block_from_pieces(_mode_independent_pieces(A, B, DA1, DB1, b), m)


def mode_operators(A, B, DA1, DB1, b, m_vals):
    """Per-mode BC-applied operators and B-factors for the whole mode set.

    Returns ``(M_bc_list, w_list, interior)``: for each m the factored operator
    with BC rows replaced (``apply_bcs_m``) and the node-array B-factor ``w``.
    ``interior`` is the (shared) PDE-row mask.  Unknown per mode is ``v_m`` with
    ``u_m = w · v_m``; for m=0 ``w≡1`` so it is the original u-solve.

    The m-independent pieces (:func:`_mode_independent_pieces`, including the 2-D
    ∂/∂A used for the m=0 Neumann row) are built ONCE for the whole mode set,
    rather than once per mode as before — the two dense ``(Na·Nb)³`` squarings
    dominate the assembly and are the same matrix for every m.

    Still on the table, and NOT taken here: replacing those squarings by
    ``kron(D@D, I)``.  That is another ~1.6× on this function, but it is only
    admissible if ``operators_abt.laplacian_matrix`` takes it in the same commit,
    because the two must produce the same matrix bit-for-bit at m=0 (see
    :func:`_mode_independent_pieces`).  ``operators_abt`` is the axisymmetric
    regression oracle, so that is a maintainer decision, not a refactor.
    """
    p = _mode_independent_pieces(A, B, DA1, DB1, b)
    M_bc_list, w_list, interior = [], [], None
    for m in m_vals:
        Mv, w = _block_from_pieces(p, int(m))
        Mm, interior = apply_bcs_m(Mv, A, B, p["DA"], int(m))
        M_bc_list.append(Mm)
        w_list.append(w)
    return M_bc_list, w_list, interior


# --------------------------------------------------------------------------
# The per-grid operator cache — build the linear blocks once, not per slice
# --------------------------------------------------------------------------
# The per-m blocks depend on the grid and on the SEPARATION b, and on nothing
# else: not the masses, momenta, spins or the iterate.  A slice assembly was
# rebuilding them every time, which is the single largest cost in
# ``solver_3d.assemble``.
#
# They are very nearly independent of b as well.  Every interior coefficient
# carries exactly one factor 1/b² (through ``_coeffs``' ``pref``, and the
# centrifugal −m²/ρ² likewise) while the BC rows are b-free, so
# ``M0_m(b) = D(b)·M0_m(1)`` for a diagonal ``D`` — a left scaling that row
# equilibration cancels.  That is what makes the *preconditioner* reusable
# across separations (see ``solver/separable.py``).  It is NOT used to
# manufacture ``M0_m(b)`` from ``M0_m(1)`` here: ``pref = oa2²/(b² Den)`` and
# ``(oa2²/Den)/b²`` are the same number in exact arithmetic but not in float64,
# and the m=0 block has to stay bit-identical to the frozen axisymmetric
# operator.  So the cache is keyed on b too, and the saving comes from reuse
# rather than from rescaling.
_BLOCK_CACHE = collections.OrderedDict()
_BLOCK_CACHE_MAX = 2      # each entry is Nm dense (Na·Nb)² blocks — 79 MiB at (44,32,8)


def _canonical_grid(A, B, DA1, DB1):
    """True if this is the grid ``operators_abt.build_grid`` builds for its size.

    The cache is keyed on ``(Na, Nb, Nφ, b)``, so it may only serve a caller whose
    1-D grid is the canonical one for that size.  A caller experimenting with a
    hand-built grid gets a fresh, uncached build rather than someone else's nodes.
    """
    Ac, Bc, DA1c, DB1c = ops.build_grid(A.size - 1, B.size)
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in
               ((A, Ac), (B, Bc), (DA1, DA1c), (DB1, DB1c)))


def _readonly(arrs):
    for a in arrs:
        a.setflags(write=False)
    return arrs


def mode_operators_cached(A, B, DA1, DB1, b, m_vals):
    """:func:`mode_operators`, memoized on ``(Na, Nb, Nφ, b)``, plus the row scales.

    Returns ``(M0_list, w_list, interior, scales)`` where ``scales[mi]`` is the
    row-equilibration scale ``max|M0_m row|`` that
    ``operators_abt.solve_equilibrated`` and
    ``solver_3d_nk.equil_residual_inf`` both use — computed here so the certified
    monitor does not re-scan 79 MiB of blocks on every Newton iteration.

    **The returned arrays are shared and read-only.**  A caller that needs to add
    a nonlinear diagonal must copy first (``np.array(M0[mi])``), which is what the
    Newton steps already do.  The cache holds at most
    ``_BLOCK_CACHE_MAX`` separations; a sweep over many b values evicts in FIFO
    order rather than growing without bound.
    """
    m_vals = np.asarray(m_vals)
    key = (A.size - 1, B.size, int(m_vals.size), float(b), tuple(int(m) for m in m_vals))
    hit = _BLOCK_CACHE.get(key)
    if hit is not None and _canonical_grid(A, B, DA1, DB1):
        _BLOCK_CACHE.move_to_end(key)
        return hit
    M0_list, w_list, interior = mode_operators(A, B, DA1, DB1, b, m_vals)
    scales = []
    for M0 in M0_list:
        s = np.max(np.abs(M0), axis=1)
        scales.append(np.where(s > 0.0, s, 1.0))
    out = (_readonly(M0_list), _readonly(w_list), interior, _readonly(scales))
    if _canonical_grid(A, B, DA1, DB1):
        _BLOCK_CACHE[key] = out
        while len(_BLOCK_CACHE) > _BLOCK_CACHE_MAX:
            _BLOCK_CACHE.popitem(last=False)
    return out


def clear_block_cache():
    """Drop every cached per-grid operator (frees ~79 MiB per held separation)."""
    _BLOCK_CACHE.clear()


# --------------------------------------------------------------------------
# Applying a REAL block to a COMPLEX mode vector without promoting the block
# --------------------------------------------------------------------------
# The per-m blocks are float64; the mode vectors ``û_m = rfft_φ(u)`` are complex.
# Handing a complex vector to a real matrix makes NumPy/SciPy promote the
# *matrix* — a fresh complex copy of an (Na·Nb)² array on every apply, and twice
# the arithmetic, for a product whose imaginary parts are all exactly zero.
# Because the block is real, ``M(x+iy) = Mx + i My`` exactly, so the two real
# products can be taken together as a single real GEMM on a 2-column stack.
def _split_ri(v):
    """``(x + i y) -> (…, 2)`` real array of columns ``[x, y]``."""
    return np.stack((v.real, v.imag), axis=-1)


def _join_ri(x):
    """Inverse of :func:`_split_ri`."""
    return x[..., 0] + 1j * x[..., 1]


def _real_part_if_real(v):
    """``v.real`` if ``v`` is real or a complex array with an all-zero imaginary
    part, else ``None``.

    Two of the modes ``numpy.fft.rfft`` returns from a real signal are exactly
    real — ``m=0`` always, and the Nyquist mode ``m=Nφ/2`` whenever ``Nφ`` is
    even — and at ``Nφ=1`` the m=0 mode is the *whole* problem.  Routing those
    through a two-column real solve is both wasteful and, more importantly, a
    different LAPACK path from the one-column solve the axisymmetric code takes:
    ``dtrsm`` with two right-hand sides is not ``dtrsv``, so the results differ
    in the last ulps and the Nφ=1 reduction stops being bit-for-bit identical to
    the frozen 2-D Newton.  Detecting the zero imaginary part keeps that gate
    exact — and halves the work on those modes.
    """
    v = np.asarray(v)
    if not np.iscomplexobj(v):
        return v
    if v.imag.any():
        return None
    return v.real


def apply_mode_block(M, v):
    """``M @ v`` for a REAL block ``M`` and a real-or-complex mode vector ``v``.

    The same arithmetic as ``M @ v``, but not the same code: a genuinely complex
    ``v`` is handled as one real GEMM against ``[Re v, Im v]`` instead of a
    promoted complex GEMM.  That halves the flops and avoids materialising a
    complex copy of ``M``; it agrees with the promoted product to a few ulps, not
    bit-for-bit, because the two BLAS kernels accumulate in different orders.  A
    real ``v`` — including a complex array whose imaginary part is exactly zero,
    see :func:`_real_part_if_real` — goes straight through as a real product and
    is bit-identical.
    """
    M = np.asarray(M)
    v = np.asarray(v)
    if np.iscomplexobj(M):
        return M @ v
    vr = _real_part_if_real(v)
    if vr is not None:
        out = M @ vr
        return out.astype(complex) if np.iscomplexobj(v) else out
    return _join_ri(M @ _split_ri(v))


def solve_mode_block(M, rhs):
    """Row-equilibrated solve of a REAL block against a real-or-complex ``rhs``.

    The same rescaling as ``operators_abt.solve_equilibrated`` — each row divided
    by its own max modulus, an exact rescaling of the equations that leaves the
    solution unchanged and collapses the raw ~1e13–1e15 condition number — but
    the complex right-hand side is solved as two real columns, so the block is
    factored once, in real arithmetic, rather than promoted.
    """
    M = np.asarray(M, dtype=float)
    rhs = np.asarray(rhs)
    scale = np.max(np.abs(M), axis=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    Meq = M / scale[:, None]
    y = rhs / scale
    yr = _real_part_if_real(y)
    if yr is not None:
        out = np.linalg.solve(Meq, yr)
        return out.astype(complex) if np.iscomplexobj(rhs) else out
    return _join_ri(np.linalg.solve(Meq, _split_ri(y)))


def apply_bcs_m(Lap_m, A, B, DA, m: int):
    """Row-replace BC edges for mode m; return ``(M, interior_mask)``.

    A=1 (i=0): Dirichlet u_m=0 (identity row) — all m.
    A=0 (i=Na): m=0 Neumann d/dA u=0 (the DA row, as in the 2-D code);
                m≠0 Dirichlet u_m=0 (axis regularity).
    interior_mask is True on PDE rows (identical for every m).
    """
    if m == 0:
        return ops.apply_bcs(Lap_m, A, B, DA)
    Na1, Nb1 = A.size, B.size
    M = np.array(Lap_m, dtype=float)
    interior = np.ones(Na1 * Nb1, dtype=bool)
    for j in range(Nb1):
        r_inf = 0 * Nb1 + j               # A=1 (infinity): Dirichlet
        M[r_inf, :] = 0.0
        M[r_inf, r_inf] = 1.0
        interior[r_inf] = False
        r_ax = (Na1 - 1) * Nb1 + j        # A=0 (inner axis): Dirichlet for m≠0
        M[r_ax, :] = 0.0
        M[r_ax, r_ax] = 1.0
        interior[r_ax] = False
    return M, interior
