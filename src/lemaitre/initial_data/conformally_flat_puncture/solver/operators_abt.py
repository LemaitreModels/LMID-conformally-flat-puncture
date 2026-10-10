"""LM-initial-data-2C — single-patch ABT / prolate-spheroidal grid + Laplacian (M2-A).

The Ansorg-Brügmann-Tichy axisymmetric coordinates, re-derived from scratch and
recognised to be **prolate-spheroidal coordinates with a compactified radial
coordinate**.  With punctures (foci) at z=±b,

    rho = b * 2A/(1-A^2) * sqrt(1-B^2),     z = b * (1+A^2)B/(1-A^2),
    A in [0,1], B in [-1,1],

equivalently the prolate-spheroidal map ``z = b xi eta``, ``rho = b
sqrt((xi^2-1)(1-eta^2))`` with ``xi = (1+A^2)/(1-A^2) >= 1`` (radial, A=tanh(psi/2)
compactified) and ``eta = B`` (angle).  Coordinate edges:

    A=1 edge        : spatial infinity (a full edge)     -> Dirichlet u=0
    A=0 edge        : inner axis |z|<=b (z=bB)           -> Neumann d/dA u=0
    B=+-1 edges     : outer axis |z|>=b (poles)          -> interior GL B nodes avoid them
    corners (A=0,B=+-1) : the two punctures (analytic in (A^2, 1-B); see report)

Because prolate-spheroidal coordinates are **orthogonal**, the axisymmetric
Laplacian has **no cross term**:

    Delta u = (1-A^2)^2/(b^2 Den) [ (1-A^2)^2/4 u_AA + (1-A^2)^2/(4A) u_A
                                    + (1-B^2) u_BB - 2B u_B ],
    Den = (1+A^2)^2 - B^2 (1-A^2)^2.

This is a single dense 2-D collocation operator (no mortar), spectrally
convergent for both punctures simultaneously.  Standalone: numpy + jax only.

Where the B nodes sit is a separate choice from the operator, which is written in
``B`` throughout: :data:`B_MAPS` offers the original Gauss–Legendre-in-``B`` grid
(``"cos"``, the default) and Ansorg–Brügmann–Tichy's angular coordinate
(``"abt"``), which resolves the punctures far better — see the note there.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from . import spectral   # reused verbatim: spectral.cheb for the Lobatto A-matrix


# --------------------------------------------------------------------------
# 1-D node sets and differentiation matrices
# --------------------------------------------------------------------------
def cheb01(N):
    """Chebyshev-Gauss-Lobatto nodes on [0,1] and the d/dA matrix.

    Built from ``spectral.cheb`` on [-1,1] (descending: x[0]=1), mapped by
    A=(x+1)/2 so A[0]=1 (infinity edge), A[N]=0 (inner-axis edge).  d/dA = 2 d/dx.
    """
    x, D = spectral.cheb(N)
    A = 0.5 * (np.asarray(x) + 1.0)
    DA = 2.0 * np.asarray(D)
    return A, DA


def gl_nodes_diffmat(N):
    """Gauss-Legendre nodes (interior, symmetric about 0) and a barycentric
    nodal differentiation matrix on [-1,1].  ``N`` nodes (degree N-1)."""
    B, _ = np.polynomial.legendre.leggauss(N)
    w = np.ones(N)
    for j in range(N):
        d = B[j] - B
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (B[i] - B[j])
        D[i, i] = -np.sum(D[i, :])
    return B, D


# --------------------------------------------------------------------------
# the ABT / prolate map and its closed-form inverse
# --------------------------------------------------------------------------
def abt_map(A, B, b):
    """(A,B) -> (rho, z).  A=1 maps to infinity (rho,z -> inf)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    den = 1.0 - A ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = b * 2.0 * A / den * np.sqrt(np.clip(1.0 - B ** 2, 0.0, None))
        z = b * (1.0 + A ** 2) * B / den
    return rho, z


def inverse_map(rho, z, b):
    """(rho, z) -> (A, B) via prolate xi,eta and A = sqrt((xi-1)/(xi+1)).

    xi = (r1 + r2)/(2b), eta = (z>0 ? +1 : -1)*... computed from r1,r2:
    r1 = dist to (0,0,+b), r2 = dist to (0,0,-b); z = b xi eta => eta = (r1 - r2)/(-2b)
    (since r1 - r2 = -2 b eta for the standard prolate relation).  Returns (A, B).
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    r1 = np.hypot(rho, z - b)        # to +b
    r2 = np.hypot(rho, z + b)        # to -b
    xi = (r1 + r2) / (2.0 * b)
    eta = (r2 - r1) / (2.0 * b)      # = B ; z = b xi eta is +ve for z>0
    A = np.sqrt(np.clip((xi - 1.0) / (xi + 1.0), 0.0, 1.0))
    return A, np.clip(eta, -1.0, 1.0)


# --------------------------------------------------------------------------
# the prolate-spheroidal axisymmetric Laplacian (dense 2-D collocation matrix)
# --------------------------------------------------------------------------
def build_grid(Na, Nb, bmap=None):
    """Return ``(A, B, DA1, DB1)``: Lobatto A on [0,1], interior B nodes on [-1,1].

    ``A`` has Na+1 nodes (A[0]=1 infinity, A[Na]=0 inner axis); ``B`` has Nb
    interior nodes, placed by the angular node map ``bmap`` (see
    :data:`B_MAPS`; ``None`` means :data:`DEFAULT_BMAP`).  ``DA1, DB1`` are the
    1-D differentiation matrices, ``DB1`` always being ``d/dB`` — so every
    operator coefficient, written in ``B``, is unchanged by the choice of map.

    ``bmap="cos"`` is the original grid, bit-for-bit: Gauss–Legendre nodes in
    ``B = cos η``.  ``bmap="abt"`` puts the Gauss–Legendre nodes in the
    interpolation coordinate ``s`` of :func:`b_of_s` instead.
    """
    bmap = check_bmap(bmap)
    A, DA1 = cheb01(Na)
    if bmap == "cos":
        B, DB1 = gl_nodes_diffmat(Nb)
        return A, B, DA1, DB1
    s, Ds = gl_nodes_diffmat(Nb)
    B = b_of_s(s, bmap)
    DB1 = (1.0 / dB_ds(s, bmap))[:, None] * np.asarray(Ds)    # chain rule, row-scaled
    return A, B, DA1, DB1


# --------------------------------------------------------------------------
# The angular node map: where the B nodes sit, and in which coordinate the
# discrete field is a polynomial
# --------------------------------------------------------------------------
#: The angular node maps.  ``B`` is always ``cos η`` of the prolate angle η (the
#: operator is written in it); the map decides the coordinate ``s`` in which the
#: unknown is represented as a polynomial and the Gauss–Legendre nodes are laid:
#:
#: * ``"cos"`` — ``s = B``.  The original grid.  At a puncture (η → 0) it has
#:   ``1 − B ≈ η²/2``: the puncture's neighbourhood, whose analytic structure has
#:   scale ``η² ~ m/b`` (the nearest complex singularity of the solution sits
#:   where ``ψ_BL + u`` vanishes, a distance ~m from the hole), is squeezed into an
#:   interval of width ~m/b at the end of ``[-1, 1]``, so Chebyshev/Legendre
#:   convergence in ``B`` goes like ``exp(−N_B √(m/b))`` and degrades with the
#:   separation.
#: * ``"abt"`` — ``B = 2s/(1+s²)``, i.e. ``η = π/2 − 2 arctan s``: the angular
#:   coordinate of Ansorg, Brügmann & Tichy, Phys. Rev. D 70, 064011 (2004),
#:   eq. (61) (their ``B`` is our ``s`` up to orientation).  Now ``1 − B =
#:   (1 − s)²/(1 + s²)``, so η is LINEAR in ``1 − s`` at a puncture and the same
#:   singularity sits ~√(m/b) from the end of the interval: convergence like
#:   ``exp(−N_B (m/b)^{1/4})``.  ``sin η = (1 − s²)/(1 + s²)`` is rational in
#:   ``s``, so the azimuthal modes ``u_m ∝ sin^m η`` are analytic in ``s`` without
#:   dividing anything out.
#:
#: The radial coordinate ``A`` is the same in both (ξ = 2 artanh A is ABT's too).
B_MAPS = ("cos", "abt")

#: The map a grid gets when none is named.  ``"cos"`` keeps every existing grid,
#: stored model, cache key and export bit-for-bit what it was; the conformally
#: curved sibling builds its chart through :func:`build_grid` with the default.
DEFAULT_BMAP = "cos"


def check_bmap(bmap):
    """Resolve ``None`` to :data:`DEFAULT_BMAP` and reject an unknown map loudly."""
    bmap = DEFAULT_BMAP if bmap is None else str(bmap)
    if bmap not in B_MAPS:
        raise ValueError(f"unknown angular node map {bmap!r}; expected one of {B_MAPS}")
    return bmap


def b_of_s(s, bmap):
    """``B`` (= cos η) at interpolation coordinate ``s``."""
    s = np.asarray(s, dtype=float)
    if check_bmap(bmap) == "cos":
        return s
    return 2.0 * s / (1.0 + s * s)


def dB_ds(s, bmap):
    """``dB/ds`` — zero at ``s = ±1`` for ``"abt"``, never at an interior node."""
    s = np.asarray(s, dtype=float)
    if check_bmap(bmap) == "cos":
        return np.ones_like(s)
    return 2.0 * (1.0 - s * s) / (1.0 + s * s) ** 2


def d2B_ds2(s, bmap):
    """``d²B/ds²`` — what the exact chain rule for ``d²/dB²`` needs besides ``dB/ds``."""
    s = np.asarray(s, dtype=float)
    if check_bmap(bmap) == "cos":
        return np.zeros_like(s)
    return 4.0 * s * (s * s - 3.0) / (1.0 + s * s) ** 3


def d2_dB2_1d(Nb, bmap):
    """The 1-D ``d²/dB²`` matrix of a mapped grid, by the EXACT chain rule; ``None``
    for ``"cos"``.

    On a mapped grid ``DB1 = diag(1/B′(s)) D_s``, and squaring it,
    ``DB1 @ DB1 = diag(1/B′) D_s diag(1/B′) D_s``, is *not* the second derivative of
    the interpolant: it samples ``(1/B′) D_s u`` at the nodes and differentiates
    that product's interpolating polynomial again.  ``1/B′ = (1+s²)²/(2(1−s²))``
    has poles at ``s = ±1``, just outside the Gauss nodes, so the re-interpolation
    is inaccurate wherever ``D_s u`` does not vanish at the axis — which for a
    discrete field it never exactly does.  The exact form,

        d²u/dB² = (1/B′)² D_s² u − (B″/B′³) D_s u,

    differentiates the polynomial in ``s`` exactly (true collocation, as
    TwoPunctures does with its spectral expansion).  Measured: on the same grid it
    takes our mid-field ψ error from 3–50× behind TwoPunctures to level with or
    ahead of it, and on the Legendre operator applied to ``1/(ξ − B)`` it is
    100–1000× more accurate than the nested square.

    ``"cos"`` returns ``None``: there ``B′ = 1``, the two forms are the same matrix,
    and the callers keep their original (2-D) squaring so every ``"cos"`` operator
    stays bit-for-bit what it was.
    """
    bmap = check_bmap(bmap)
    if bmap == "cos":
        return None
    s, Ds = gl_nodes_diffmat(Nb)
    Ds = np.asarray(Ds)
    Bp, Bpp = dB_ds(s, bmap), d2B_ds2(s, bmap)
    return (1.0 / Bp ** 2)[:, None] * (Ds @ Ds) - (Bpp / Bp ** 3)[:, None] * Ds


# The canonical mapped grids per (Nb, bmap), so a constructor handed bare arrays can
# recognise one (O(Nb²) to compare, built once per size).
_ANGULAR_D2_MEMO: dict = {}


def angular_d2(B, DB1):
    """The exact 1-D ``d²/dB²`` for this angular grid, or ``None``.

    The operator constructors (:func:`laplacian_matrix`,
    ``operators_3d._mode_independent_pieces``, ``separable.kron_factors``) receive
    bare ``(B, DB1)`` arrays, not a map name.  This returns
    :func:`d2_dB2_1d` when ``(B, DB1)`` is exactly the canonical
    :func:`build_grid` angular grid of a mapped ``bmap``, and ``None`` otherwise —
    for every ``"cos"`` grid (whose callers then square ``DB`` exactly as before)
    and for a hand-built grid (which keeps the nested square it always had: a
    caller laying its own nodes owns its own second derivative).  All three
    constructors go through here, so the dense 2-D, dense 3-D and separable
    operators agree on which form they use.
    """
    B = np.asarray(B)
    DB1 = np.asarray(DB1)
    Nb = B.size
    for bmap in B_MAPS:
        if bmap == "cos":
            continue
        key = (Nb, bmap)
        hit = _ANGULAR_D2_MEMO.get(key)
        if hit is None:
            _, Bc, _, DB1c = build_grid(1, Nb, bmap)
            hit = (Bc, DB1c, d2_dB2_1d(Nb, bmap))
            for a in hit:
                a.setflags(write=False)
            _ANGULAR_D2_MEMO[key] = hit
        Bc, DB1c, DBB1 = hit
        if np.array_equal(B, Bc) and np.array_equal(DB1, DB1c):
            return DBB1
    return None


def s_of_b(B, bmap, one_minus_B=None, one_plus_B=None):
    """The interpolation coordinate ``s`` at ``B`` — the inverse of :func:`b_of_s`.

    Barycentric interpolation of a field on a ``bmap`` grid must be done in ``s``,
    not in ``B``: the discrete field is a polynomial in ``s``.  For ``"cos"`` this
    returns ``B`` itself (same values), so the original interpolation is unchanged.
    For ``"abt"`` the form ``B / (1 + √((1 − B)(1 + B)))`` is used rather than
    ``(1 − √(1 − B²)) / B``: equal, but finite at ``B = 0``.  It maps ``±1`` to
    ``±1``, so a puncture readout at ``B = ±1`` is a readout at ``s = ±1`` under
    either map.

    **Near the axis ``s`` is ill-conditioned in ``B``** (``ds/dB → ∞`` at ``B = ±1``:
    ``1 − s ≈ √(2(1 − B))``), so a ``B`` rounded to float64 fixes ``s`` only to
    ~``eps/(1 − |s|)``.  Pass ``one_minus_B``/``one_plus_B`` computed without
    cancellation (as :func:`inverse_map_s` does from the physical point) to keep
    ``s`` accurate to ~eps everywhere.
    """
    B = np.asarray(B, dtype=float)
    if check_bmap(bmap) == "cos":
        return B
    om = 1.0 - B if one_minus_B is None else np.asarray(one_minus_B, dtype=float)
    op = 1.0 + B if one_plus_B is None else np.asarray(one_plus_B, dtype=float)
    return B / (1.0 + np.sqrt(np.clip(om * op, 0.0, None)))


def s_nodes(Nb, bmap):
    """The angular nodes in the interpolation coordinate, exactly as laid.

    ``"cos"``: the B nodes.  ``"abt"``: the Gauss–Legendre nodes in ``s`` that
    :func:`build_grid` mapped to B — returned as laid, not recovered from B (which
    near the ends would cost the conditioning described in :func:`s_of_b`).
    """
    check_bmap(bmap)
    s, _ = gl_nodes_diffmat(Nb)
    return np.asarray(s, dtype=float)


def inverse_map_s(rho, z, b, bmap):
    """``(rho, z) -> (A, s)``: :func:`inverse_map`, with the angle in the map's
    interpolation coordinate and computed without cancellation near the axis.

    For ``"cos"`` it IS :func:`inverse_map`.  Otherwise ``B = 2z/(r₁+r₂)`` (equal to
    ``(r₂ − r₁)/2b`` because ``r₂² − r₁² = 4bz``) and ``1 ∓ B`` are formed from
    ``r − c`` terms rationalised where they would cancel, so a point a hair off
    the outer axis still gets an ``s`` accurate to ~eps.
    """
    if check_bmap(bmap) == "cos":
        return inverse_map(rho, z, b)
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    A, _ = inverse_map(rho, z, b)
    c1, c2 = z - b, z + b                       # r1 = |(rho, c1)|, r2 = |(rho, c2)|
    r1, r2 = np.hypot(rho, c1), np.hypot(rho, c2)
    rho2 = rho * rho

    def r_minus_c(r, c):                        # r - c, stable for c > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(c > 0.0, rho2 / (r + c), r - c)

    def r_plus_c(r, c):                         # r + c, stable for c < 0
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(c < 0.0, rho2 / (r - c), r + c)

    S = r1 + r2
    B = np.clip(2.0 * z / S, -1.0, 1.0)
    om = (r_minus_c(r1, c1) + r_minus_c(r2, c2)) / S        # 1 - B
    op = (r_plus_c(r1, c1) + r_plus_c(r2, c2)) / S          # 1 + B
    return A, s_of_b(B, bmap, om, op)


def _coeffs(Af, Bf, b):
    """Prolate Laplacian coefficients (alpha,p,gamma,q) at flattened nodes,
    with edge rows (A=0, A=1) given finite dummies (they become BC rows)."""
    edge = (Af <= 1e-14) | (Af >= 1.0 - 1e-14)
    Asafe = np.where(edge, 0.5, Af)
    oa2 = 1.0 - Asafe ** 2
    Den = (1.0 + Asafe ** 2) ** 2 - Bf ** 2 * oa2 ** 2
    pref = oa2 ** 2 / (b ** 2 * Den)
    alpha = pref * oa2 ** 2 / 4.0
    pcoef = pref * oa2 ** 2 / (4.0 * Asafe)
    gamma = pref * (1.0 - Bf ** 2)
    qcoef = pref * (-2.0 * Bf)
    for c in (alpha, pcoef, gamma, qcoef):
        c[edge] = 0.0
    return alpha, pcoef, gamma, qcoef


def laplacian_matrix(A, B, DA1, DB1, b):
    """Dense prolate-spheroidal Laplacian (no cross term) + node coordinates.

    Returns ``(Lap, rho, z, Af, Bf, DA, DB)`` where Lap is the
    ((Na+1)*Nb)x((Na+1)*Nb) operator on the flattened field U[i*Nb + j], and
    DA, DB are the 2-D first-derivative matrices (used for BC rows).
    """
    Na1, Nb1 = A.size, B.size
    IA, IB = np.eye(Na1), np.eye(Nb1)
    DA = np.kron(np.asarray(DA1), IB)
    DB = np.kron(IA, np.asarray(DB1))
    DAA = DA @ DA
    # On a mapped grid the exact chain rule, not DB @ DB (see d2_dB2_1d); on "cos"
    # (DBB1 is None) the original squaring, bit-for-bit.
    DBB1 = angular_d2(B, DB1)
    DBB = DB @ DB if DBB1 is None else np.kron(IA, DBB1)
    AA, BB = np.meshgrid(A, B, indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    rho, z = abt_map(Af, Bf, b)
    alpha, pcoef, gamma, qcoef = _coeffs(Af, Bf, b)
    Lap = (alpha[:, None] * DAA + pcoef[:, None] * DA
           + gamma[:, None] * DBB + qcoef[:, None] * DB)
    return Lap, rho, z, Af, Bf, DA, DB


def solve_equilibrated(M, rhs):
    """Solve M x = rhs with row equilibration (scale each row by 1/max|row|).

    The prolate operator's rows span a huge dynamic range (the (1-A^2)^4 factor
    is tiny near the A=1 infinity edge, the 1/A factor large near A=0), giving a
    raw condition number ~1e13-1e15.  Row equilibration is an exact rescaling of
    the equations (the unknown is unchanged) that collapses cond to ~1e4 and
    takes the solve to machine precision.
    """
    M = np.asarray(M)
    scale = np.max(np.abs(M), axis=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    return np.linalg.solve(M / scale[:, None], np.asarray(rhs) / scale)


def apply_bcs(Lap, A, B, DA):
    """Row-replace BC edges; return ``(M, interior_mask)``.

    A=1 (i=0): Dirichlet u=0 (identity row).
    A=0 (i=Na): Neumann d/dA u=0 (the DA row).
    interior_mask is True on PDE rows (RHS/source applied there).
    """
    Na1, Nb1 = A.size, B.size
    M = np.array(Lap, dtype=float)
    interior = np.ones(Na1 * Nb1, dtype=bool)
    for j in range(Nb1):
        r_inf = 0 * Nb1 + j               # A=1 (infinity)
        M[r_inf, :] = 0.0
        M[r_inf, r_inf] = 1.0
        interior[r_inf] = False
        r_ax = (Na1 - 1) * Nb1 + j        # A=0 (inner axis)
        M[r_ax, :] = DA[r_ax, :]
        interior[r_ax] = False
    return M, interior
