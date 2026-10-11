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

How the field is represented in ``A`` is a second, independent choice,
:data:`RADIAL_MAPS`: the original Chebyshev–Gauss–Lobatto grid with boundary rows
at both edges (``"lobatto"``, the default), or Ansorg–Brügmann–Tichy's own
``u = (A − 1) V`` on Chebyshev–Gauss nodes with no boundary row at all
(``"gauss"``), which reads the ADM mass far more accurately — see the note there.
The two named combinations a caller is expected to pick between are the
:data:`SCHEMES`.
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


def cheb_gauss01(K):
    """Chebyshev–Gauss nodes on (0,1) and a barycentric d/dA matrix: the ``"gauss"``
    radial grid.

    ``A_i = sin²[π(i + ½)/(2K)]``, ``i = 0..K−1`` — Ansorg, Brügmann & Tichy,
    Phys. Rev. D 70, 064011 (2004), eq. (8) — returned DESCENDING (``A[0]`` nearest
    infinity), the order :func:`cheb01` uses, so array index 0 is the outermost
    shell under either radial map.  No node sits at ``A = 0`` or ``A = 1``.
    """
    i = np.arange(K)
    A = np.sin(np.pi / (2 * K) * (i + 0.5)) ** 2
    A = np.sort(A)[::-1].copy()
    w = np.ones(K)
    for j in range(K):
        d = A[j] - A
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (A[i] - A[j])
        D[i, i] = -np.sum(D[i, :])
    return A, D


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
def build_grid(Na, Nb, bmap=None, radial=None):
    """Return ``(A, B, DA1, DB1)``: Na+1 A nodes on [0,1], Nb interior B nodes on [-1,1].

    ``A`` is laid by the radial map ``radial`` (see :data:`RADIAL_MAPS`; ``None``
    means :data:`DEFAULT_RADIAL`): ``"lobatto"`` has A[0]=1 (infinity) and
    A[Na]=0 (inner axis), ``"gauss"`` has both edges strictly outside the node
    set.  ``B`` is placed by the angular node map ``bmap`` (see :data:`B_MAPS`;
    ``None`` means :data:`DEFAULT_BMAP`).  ``DA1, DB1`` are the 1-D
    differentiation matrices, ``DA1`` always being ``d/dA`` and ``DB1`` always
    ``d/dB`` — so every operator coefficient, written in ``(A, B)``, is unchanged
    by either choice.

    ``bmap="cos"``, ``radial="lobatto"`` is the original grid, bit-for-bit:
    Chebyshev–Gauss–Lobatto in ``A``, Gauss–Legendre nodes in ``B = cos η``.
    ``bmap="abt"`` puts the Gauss–Legendre nodes in the interpolation coordinate
    ``s`` of :func:`b_of_s` instead; ``radial="gauss"`` is :func:`cheb_gauss01`.
    """
    bmap = check_bmap(bmap)
    A, DA1 = radial_nodes(Na, radial)
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
#: The radial coordinate ``A`` is the same in both (ξ = 2 artanh A is ABT's too);
#: how the field is represented in it is :data:`RADIAL_MAPS`.
B_MAPS = ("cos", "abt")


# --------------------------------------------------------------------------
# The radial representation: which A nodes, and which edges carry a row
# --------------------------------------------------------------------------
#: The radial maps.  Both use the coordinate ``A`` of :func:`abt_map`; they differ
#: in how the two edges are treated and in what the unknown is:
#:
#: * ``"lobatto"`` — the original grid.  Chebyshev–Gauss–Lobatto nodes
#:   (:func:`cheb01`), so there is a node on each edge, and each carries a boundary
#:   row instead of the PDE: Dirichlet ``u = 0`` at ``A = 1`` (spatial infinity), and
#:   at ``A = 0`` (the inner axis) Neumann ``∂_A u = 0`` for ``m = 0``, ``u_m = 0``
#:   otherwise.  The unknown is ``u``.
#: * ``"gauss"`` — the representation of Ansorg, Brügmann & Tichy, Phys. Rev. D 70,
#:   064011 (2004), eqs. (5) and (8): ``u = (A − 1) V`` on Chebyshev–Gauss nodes
#:   (:func:`cheb_gauss01`), neither edge a node, every row a PDE row.  ``u = 0`` at
#:   infinity holds by construction, and regularity on the inner axis is selected by
#:   the PDE's own ``1/A``, ``1/A²`` coefficients at the interior nodes.  The unknown
#:   is ``V``; the factor ``A − 1`` folds into the same nodal factor ``w`` as the
#:   angular ``(1 − B²)^{|m|/2}`` (``operators_3d.mode_operators``), so ``u = w·v``
#:   holds under both maps.
#:
#: Why it matters: the Dirichlet row degrades the slope ``∂_A u|_{A=1}``.  For the
#: exact solution that slope is the same at every ``B`` and IS the ADM mass,
#: ``M_ADM = m_A + m_B − 2b ⟨∂_A u⟩_{A=1}`` (``validation.adm.adm_mass_spectral``);
#: near ``A = 1`` the angular operator is not damped by the ``(1 − A)²`` the
#: ``A``-terms carry, so its leading order says the slope is constant in ``B`` — a
#: condition the ``"gauss"`` unknown ``V`` sees directly (``∂_A u|_{A=1} = V(1, B)``)
#: and the ``"lobatto"`` rows see only times the vanishing ``(1 − A)``.  At the
#: production grid 44×32×8 and ``b = 10`` the ADM mass moves from 1.0e-6 off a
#: converged reference (``"abt"`` angle, ``"lobatto"``) to 2.2e-9 (``"abt"``,
#: ``"gauss"``); the near- and mid-field ψ are B-limited and unchanged.  The
#: ``"gauss"`` form's one weak spot is ``u`` at the punctures (read by
#: extrapolating to ``A = 0``) when ``A`` is starved relative to ``B``,
#: ``Na ≲ Nb + 8``.
RADIAL_MAPS = ("lobatto", "gauss")

#: The two discretisations a caller chooses between, as ``(bmap, radial)``:
#:
#: * ``"paper-v1"`` — ``("cos", "lobatto")``, exactly what the first version of the
#:   paper used: every shipped model, figure, table and the GRTeclyn export.
#: * ``"v2"`` — ``("abt", "gauss")``, the angular and radial representations of
#:   Ansorg, Brügmann & Tichy (with Gauss–Legendre rather than Chebyshev nodes in the
#:   angle, and the exact chain-rule ``d²/dB²``).  Better near the punctures (the
#:   angle) and in the ADM mass (the radius), and level elsewhere.
#:
#: The axes stay separately selectable (``bmap=``, ``radial=``), so the mixed
#: combinations can be built too; ``scheme=`` names the two that are meant.
SCHEMES = {"paper-v1": ("cos", "lobatto"), "v2": ("abt", "gauss")}

#: The scheme a grid gets when none is named.  ``"paper-v1"`` keeps every existing
#: grid, stored model, cache key and export bit-for-bit what it was; the conformally
#: curved sibling builds its chart through :func:`build_grid` with the default.
DEFAULT_SCHEME = "paper-v1"

#: The map and the radial representation a grid gets when none is named — the two
#: halves of :data:`DEFAULT_SCHEME`, so there is one place to change.
DEFAULT_BMAP, DEFAULT_RADIAL = SCHEMES[DEFAULT_SCHEME]


def check_radial(radial):
    """Resolve ``None`` to :data:`DEFAULT_RADIAL` and reject an unknown map loudly."""
    radial = DEFAULT_RADIAL if radial is None else str(radial)
    if radial not in RADIAL_MAPS:
        raise ValueError(f"unknown radial map {radial!r}; expected one of {RADIAL_MAPS}")
    return radial


def resolve_scheme(scheme=None, bmap=None, radial=None):
    """``(bmap, radial)`` from a named scheme and/or the two axes.

    With ``scheme`` given, its pair is returned, and an axis passed alongside it
    must agree with it (a contradiction is refused, never silently resolved).
    Without it, each axis resolves on its own — ``None`` to its half of
    :data:`DEFAULT_SCHEME`.
    """
    if scheme is not None:
        if str(scheme) not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; expected one of {tuple(SCHEMES)}")
        sb, sr = SCHEMES[str(scheme)]
        if bmap is not None and check_bmap(bmap) != sb:
            raise ValueError(f"scheme {scheme!r} has bmap={sb!r}, not {bmap!r}")
        if radial is not None and check_radial(radial) != sr:
            raise ValueError(f"scheme {scheme!r} has radial={sr!r}, not {radial!r}")
        return sb, sr
    return check_bmap(bmap), check_radial(radial)


def scheme_of(bmap, radial):
    """The name of the scheme ``(bmap, radial)`` is, or ``None`` for a mixed grid."""
    for name, pair in SCHEMES.items():
        if pair == (bmap, radial):
            return name
    return None


def radial_nodes(Na, radial=None):
    """``(A, DA1)``: the Na+1 radial nodes, descending, and ``d/dA`` on them."""
    if check_radial(radial) == "lobatto":
        return cheb01(Na)
    return cheb_gauss01(Na + 1)


# The canonical "gauss" radial grid per node count, so a constructor handed bare
# arrays can recognise one (O(Na) to compare, built once per size).
_RADIAL_MEMO: dict = {}


def radial_of(A, DA1=None):
    """``"gauss"`` if ``A`` is exactly the canonical :func:`cheb_gauss01` grid, else
    ``"lobatto"``.

    The operator constructors (:func:`laplacian_matrix`, :func:`apply_bcs`,
    ``operators_3d``, ``separable``) receive bare arrays, not a map name — the same
    situation :func:`angular_d2` handles for the angle.  A grid with a node at
    ``A = 1`` is never ``"gauss"``; anything that is not the canonical Gauss grid,
    including a hand-built one, keeps the original boundary-row treatment it always
    had.  ``DA1``, when passed, must then be the canonical matrix too: a Gauss grid
    with someone else's ``d/dA`` is refused rather than half-recognised.
    """
    A = np.asarray(A)
    if A.size == 0 or A[0] == 1.0:
        return "lobatto"
    hit = _RADIAL_MEMO.get(A.size)
    if hit is None:
        hit = cheb_gauss01(A.size)
        for a in hit:
            a.setflags(write=False)
        _RADIAL_MEMO[A.size] = hit
    if not np.array_equal(A, hit[0]):
        return "lobatto"
    if DA1 is not None and not np.array_equal(np.asarray(DA1), hit[1]):
        raise ValueError("A is the canonical 'gauss' radial grid but DA1 is not its d/dA; "
                         "a hand-built radial operator is not supported on it")
    return "gauss"


def radial_factor(A):
    """The radial factor ``f(A) = A − 1`` of ``u = f·V`` on a ``"gauss"`` grid (an
    array over the A nodes), or ``None`` on any other grid."""
    if radial_of(A) != "gauss":
        return None
    return np.asarray(A, dtype=float) - 1.0


def radial_derivs(A, DA1):
    """``(f, DAu1, DAAu1)`` on a ``"gauss"`` grid: the 1-D ``∂_A`` and ``∂²_A`` of
    ``u = f V`` acting on ``V``.

    With ``f = A − 1`` (``f′ = 1``, ``f″ = 0``): ``∂_A u = V + f ∂_A V`` and
    ``∂²_A u = 2 ∂_A V + f ∂²_A V``, so ``DAu1 = I + diag(f) DA1`` and
    ``DAAu1 = 2 DA1 + diag(f) DA1²``.  The one place these are built, so the 2-D
    operator, the 3-D blocks and the separable factors use the same matrices.
    """
    A = np.asarray(A, dtype=float)
    DA1 = np.asarray(DA1, dtype=float)
    fA = A - 1.0
    DAu1 = np.diag(np.ones(A.size)) + fA[:, None] * DA1
    DAAu1 = 2.0 * DA1 + fA[:, None] * (DA1 @ DA1)
    return fA, DAu1, DAAu1


def bary_row(x, nodes):
    """The barycentric cardinal row at ``x`` (a scalar): ``row @ vals`` is the
    degree-``n−1`` interpolant of ``vals`` at ``x``.  ``x`` may lie outside the node
    set — the ``"gauss"`` grid reads ``V`` at ``A = 1`` and ``A = 0`` this way."""
    nodes = np.asarray(nodes, dtype=float)
    n = nodes.size
    w = np.ones(n)
    for j in range(n):
        d = nodes[j] - nodes
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    d = float(x) - nodes
    hit = np.isclose(d, 0.0, atol=1e-13)
    if hit.any():
        return hit.astype(float) / hit.sum()
    t = w / d
    return t / t.sum()


def radial_edge_rows(A):
    """``(slope_row, axis_row)`` on a ``"gauss"`` grid, else ``None``.

    The two linear readouts at the edges that a ``"lobatto"`` grid takes off its
    edge nodes (``(DA1 @ U)[0]`` and ``U[-1]``), for a grid that has no node there.
    Both act on the PHYSICAL field along ``A`` (``row @ U[:, j]``):

    * ``slope_row @ U = ∂_A u|_{A=1}``.  With ``u = (A − 1) V``,
      ``∂_A u = V + (A − 1) ∂_A V``, which is ``V(1)`` at ``A = 1``: the slope at
      infinity is the interpolant of ``V`` read there — the ADM-mass coefficient
      ``c = −b ∂_A u|_{A=1}`` (``validation.adm.adm_mass_spectral``).
    * ``axis_row @ U = u|_{A=0}`` ``= (0 − 1) V(0)``, read by extrapolating ``V`` a
      short way past the innermost Gauss node; the puncture values (``A = 0``,
      ``B = ±1``) are built from this row.

    Each is ``bary_row(edge, A) / (A − 1)`` (times ``−1`` on the axis), so it is a
    plain vector: numpy and jax callers apply it the same way.
    """
    f = radial_factor(A)
    if f is None:
        return None
    return bary_row(1.0, A) / f, -bary_row(0.0, A) / f


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

    On a ``"gauss"`` radial grid (:data:`RADIAL_MAPS`) the unknown is ``V`` with
    ``u = (A − 1) V``: ``Lap @ vec(V) = Δu``, ``DA`` is ``∂_A u`` as a matrix on
    ``V`` (:func:`radial_derivs`), and the ``B``-terms carry the factor ``A − 1``
    the ``A``-derivatives have already absorbed.  No row needs replacing there.
    """
    Na1, Nb1 = A.size, B.size
    IA, IB = np.eye(Na1), np.eye(Nb1)
    gauss = radial_of(A, DA1) == "gauss"
    if gauss:
        fA, DAu1, DAAu1 = radial_derivs(A, DA1)
        DA = np.kron(DAu1, IB)
        DAA = np.kron(DAAu1, IB)
    else:
        DA = np.kron(np.asarray(DA1), IB)
    DB = np.kron(IA, np.asarray(DB1))
    if not gauss:
        DAA = DA @ DA
    # On a mapped grid the exact chain rule, not DB @ DB (see d2_dB2_1d); on "cos"
    # (DBB1 is None) the original squaring, bit-for-bit.
    DBB1 = angular_d2(B, DB1)
    DBB = DB @ DB if DBB1 is None else np.kron(IA, DBB1)
    AA, BB = np.meshgrid(A, B, indexing="ij")
    Af, Bf = AA.ravel(), BB.ravel()
    rho, z = abt_map(Af, Bf, b)
    alpha, pcoef, gamma, qcoef = _coeffs(Af, Bf, b)
    if gauss:
        fAf = np.repeat(fA, Nb1)
        Lap = (alpha[:, None] * DAA + pcoef[:, None] * DA
               + (fAf * gamma)[:, None] * DBB + (fAf * qcoef)[:, None] * DB)
        return Lap, rho, z, Af, Bf, DA, DB
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

    A ``"gauss"`` radial grid (:data:`RADIAL_MAPS`) has no edge node, so nothing
    is replaced: ``M`` is a copy of ``Lap`` and every row is a PDE row.
    """
    Na1, Nb1 = A.size, B.size
    if radial_of(A) == "gauss":
        return np.array(Lap, dtype=float), np.ones(Na1 * Nb1, dtype=bool)
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
