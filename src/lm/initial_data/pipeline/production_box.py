"""LM-initial-data — the single canonical definition of the PRODUCTION PARAMETER BOX.

Every producer and model builder in this package must take its box from here.
Before this module the box was redeclared independently in eight producers, and
the ``b`` range had already drifted between them ([1.5,4] / [2,4] / [2,7]); a
from-scratch rebuild against hand-edited per-file constants is how a corpus ends
up built on two different boxes, which is invisible until the paper's numbers
disagree with each other.

The production box
-----------------
    b     in [B_MIN, B_MAX] M      (puncture separations D = 2b)
    q     in [Q_MIN, Q_MAX]
    chi_* in [-CHI_MAX, CHI_MAX]   every spin component, dimensionless chi = S/m^2

Each edge is set by a measured constraint, not by convention; see the constants
below.  The box keys registered by ``build_surrogate_chi`` carry a ``_prod`` suffix
(``d4_qc_chi_prod``, ``spin8_qc_chi_prod``); it marks them as the production boxes
and deliberately encodes no range, so retargeting an edge here does not strand a
name.  They were previously ``_b27``, after the then-current b in [2,7].

``CHI_MAX`` is the *free-data* spin parameter, not the horizon spin.  It is set to
0.9 to sit inside the Bowen--York horizon-spin ceiling chi_hor ~ 0.93, above which
the horizon spin saturates no matter how large the free-data parameter: past that
point extra sampling range buys physically degenerate configurations.  Restricting
it does not reduce the offline solve count (that is set by the Smolyak level and
the dimension, not the box width); it buys a modestly better bare interpolant and
a box that matches the moving-puncture regime the family is used in.

Changing ``CHI_MAX`` invalidates the solve store for every node with a nonzero
spin component: :func:`~lm.initial_data.parametric.solve_store.slice_key` keys on
the physical slice, and the Chebyshev nodes ``CHI_MAX * cos(k*pi/n)`` all move.
Only the all-spins-zero nodes (essentially the (b, q) subgrid) survive.  Treat any
change here as a full rebuild, not a top-up.
"""
from __future__ import annotations

# --- spins -----------------------------------------------------------------
CHI_MAX = 0.9
"""Half-width of every dimensionless-spin axis (free-data chi = S/m^2)."""

CHI_REP = 0.5
"""Representative interior spin at which held axes are pinned in one-axis sweeps."""

ALIGNED_SPIN_AXES = ("chi_Ay", "chi_By")
"""Aligned (orbital-angular-momentum) spin components of the 4-D model."""

SPIN8_AXES = ("chi_Ax", "chi_Ay", "chi_Az", "chi_Bx", "chi_By", "chi_Bz")
"""All six spin components of the 8-D general-spin model."""

# --- separation and mass ratio ---------------------------------------------
B_MIN, B_MAX = 3.0, 10.0
"""Production separation range; puncture separations D = 2b in [6, 20] M.

Both edges are set by measurement.  The LOWER edge is a *physics* limit, not a
solver one: the 3PN quasi-circularity series of Eq. (eq:pt), which is what makes the
family orbiting rather than a free momentum scan, degrades toward coincidence --
at D = 4 M its 1PN term is 50% of leading (2PN and 3PN each ~25% of their
predecessor), and D = 4 M is at or inside the quasi-circular ISCO for equal mass,
so such configurations are not quasi-circular orbits at all.  At D = 6 M the 1PN
term is down to 33%.  The UPPER edge reaches the separations production
moving-puncture runs actually start from (D >~ 15-20 M).

The b axis is cheap to SHIFT but expensive to WIDEN: its wall is pinned at
b* ~ 0 (see ``WALL_B_MIN_SWEEP``), so the Bernstein rate depends on the interval's
ratio rather than its width -- [3,10] converges at ~0.53 decades/node, marginally
FASTER than the old [2,7] (~0.52), whereas [2,14] would drop to ~0.35.
"""

B_MIN_NARROW, B_MAX_NARROW = 2.0, 4.0
"""Legacy narrow-separation boxes (``d4_qc_chi``/``spin8_qc_chi``), NOT production.

Kept at their historical edges so the earlier study stays reproducible; the
production models use the full box above.
"""

B_REP = 3.5
"""Representative interior separation at which the b axis is held in one-axis sweeps.

Replaces the historical b = 2.5, which sat interior to the old [2,7] but is OUTSIDE
the box above.  Kept near the lower end (7% into the range, as 2.5 was 10% into
[2,7]) so the per-axis spin/q rates are still quoted in the harder close-separation
regime rather than flattered by a mid-box point.
"""

Q_MIN, Q_MAX = 1.0, 3.0
"""Production mass-ratio range, q = m_A/m_B >= 1.

The q rate on [1,3] is ~0.60 decades/node, which inverts to a nearest inferred
singularity at q* ~ 4.1.  That much is confirmed by measurement (0.617, q* = 4.19).

An earlier revision of this docstring extrapolated from those two numbers that
widening would be expensive -- rate ~0.38 at q_max = 3.6 and ~0.17 at q_max = 4 --
by assuming q* stays PINNED at ~4.1 as the box grows, so that the interval marches
into the singularity.  **``WALL_Q_MAX`` has now been run and that assumption is
wrong: q* MARCHES.**  Measured on the production corpus (chi-rebuild, grid 44x32x8;
``reports/3D_parametric/qc_chi_prod/walls_d4_qc_chi.json``, key ``Q_wall_q``):

    q_max   rate (dec/node)   inferred q*   q*/q_max
    3.0         0.617             4.19        1.40
    3.5         0.578             4.78        1.37
    4.0         0.522             5.22        1.31

So the inferred singularity RECEDES roughly in proportion to the interval
(q*/q_max ~ 1.3-1.4, near-constant) instead of sitting still: the box never
approaches it.  Widening to q_max = 4 costs ~15% of the rate (0.617 -> 0.522), not
the ~72% the pinned-wall extrapolation predicted.

Consequences: the q wall is a COMPLEX PAIR, not a hard limit; q = 3 is NOT near a
genuine edge; and q_max could be raised to cover the asymmetric mass ratios at
modest cost.  The edge is left at 3.0 here because raising it is a scope decision
(it invalidates the solve store for every node with q > 3), not because the
interpolation would fail.
"""

FIXED_QC = {"qc": 1.0}
"""Fixed non-swept parameters selecting quasi-circular momenta."""

# --- analyticity-wall sweep ranges -----------------------------------------
WALL_CHI_INSIDE_FRACS = (0.5, 0.75, 1.0)
"""Inside-box spin-wall ranges, as fractions of ``CHI_MAX`` (auto-track it)."""

WALL_CHI_OUTSIDE = (1.6, 2.4, 3.2)
"""Outside-box (super-extremal) spin-wall ranges.

Deliberately past the box: the wall-*character* test (is the inferred singularity
pinned, hence real, or marching, hence a complex pair?) only has discriminating
power once the sampling interval approaches the singularity distance.  For a fixed
complex pair the inferred chi* is nearly constant inside the box (~2% over
WALL_CHI_INSIDE) and marches strongly outside it (~40%), so these ranges cannot be
replaced by inside-box ones.  They are a diagnostic, not a claim about where the
model is valid.
"""


WALL_B_MIN_SWEEP = (B_MIN, 2.0, 1.5, 1.0)
"""Merger-wall sweep: b_min pushed from the box edge down toward coincidence.

The wall is a HARD (real) one pinned at b* ~ 0, so unlike the spin sweep these
ranges must descend BELOW the box for the rate to respond.
"""

WALL_Q_MAX = (3.0, 3.5, 4.0)
"""Mass-ratio-wall sweep: q_max raised from the box edge toward the inferred q* ~ 4.1.

Stops just short of the inferred wall on purpose.  If the wall is real, sweeping a
range that CONTAINS it breaks the inversion (which assumes the singularity lies
outside the interval); if the three ranges come out marching instead of pinned the
wall is a complex pair.

RESULT (see ``Q_MIN, Q_MAX`` above for the table): the three ranges came out
**marching** -- q* = 4.19, 4.78, 5.22 -- so the wall is a complex pair.  The probe
at q_max >~ 4.5 that this sweep was designed to motivate is therefore the sensible
follow-up if a wider q box is ever wanted; nothing measured so far argues against
one.
"""


def wall_chi_inside() -> tuple[float, ...]:
    """Inside-box spin-wall sweep ranges, in chi."""
    return tuple(f * CHI_MAX for f in WALL_CHI_INSIDE_FRACS)


def _axis(name: str, lo: float, hi: float) -> dict:
    return {"name": name, "min": lo, "max": hi}


def spin_axes(names) -> list[dict]:
    """``[-CHI_MAX, CHI_MAX]`` axis dicts for the named spin components."""
    return [_axis(n, -CHI_MAX, CHI_MAX) for n in names]


def aligned_box(b_min: float = B_MIN, b_max: float = B_MAX) -> list[dict]:
    """The 4-D aligned-spin box ``(b, q, chi_Ay, chi_By)``."""
    return [_axis("b", b_min, b_max), _axis("q", Q_MIN, Q_MAX)] + spin_axes(ALIGNED_SPIN_AXES)


def spin8_box(b_min: float = B_MIN, b_max: float = B_MAX) -> list[dict]:
    """The 8-D general-spin box ``(b, q, chi_A, chi_B)``."""
    return [_axis("b", b_min, b_max), _axis("q", Q_MIN, Q_MAX)] + spin_axes(SPIN8_AXES)
