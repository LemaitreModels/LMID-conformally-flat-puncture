"""PARASOL <-> TwoPunctures convention map  (Milestone B1, Step 1 — risk R2).

A spurious disagreement on **conventions** (not physics) is the predicted B1
failure mode, so the map between PARASOL free data and TwoPunctures parameters
is pinned *before* any number is compared, and documented against TwoPunctures'
own source.  Every field below is justified; nothing is taken from memory
without a cross-check against the code we actually build and run
(:mod:`parasol.validation.twopunctures`).

PARASOL free data  (``solver.solver_abt.Slice`` + a problem momentum ``P``)
--------------------------------------------------------------------------
Two punctures on the **z-axis** (``source.py``):

    puncture A:  z = +b,  bare mass m_A,  P_A = (0, 0, -P)   (infall: toward 0)
    puncture B:  z = -b,  bare mass m_B,  P_B = (0, 0, +P)   (infall: toward 0)

    psi = 1 + m_A/(2 r_A) + m_B/(2 r_B) + u,   r_X = |x - x_X|,
    Delta u = -1/8 (psi)^{-7} Â_ij Â^ij,       Â^ij = Â_A^ij + Â_B^ij,
    Â_X^ij = (3/2 r_X^2)[ P_X^i n_X^j + P_X^j n_X^i - (delta^ij - n_X^i n_X^j) P_X·n_X ].

TwoPunctures parameters  (Ansorg–Brügmann–Tichy 2004; Einstein Toolkit thorn)
-----------------------------------------------------------------------------
Verified facts (ET TwoPunctures ``param.ccl`` / documentation; re-checked
against the built code in ``twopunctures.py``):

  * **Axis.** By default the punctures lie on the **x-axis** at ``(±par_b, 0, 0)``.
    (``swap_xz=yes`` would place them on the z-axis.)  We keep the *default
    x-axis* and relabel axes for the comparison — the data are axisymmetric
    about the collision axis, so psi is a function of (axial coord, cylindrical
    radius) only, and  psi_PARASOL(rho, z) ≡ psi_TP(R_cyl=rho, x_axial=z).
    This avoids the swap_xz component-permutation subtlety entirely.

  * **Separation.** ``par_b`` is the **half-separation**: punctures at x=±par_b,
    full coordinate separation 2·par_b.  Same convention as PARASOL's ``b``  ⇒
    **par_b = b**.

  * **Masses.** ``par_m_plus`` / ``par_m_minus`` are the **bare puncture mass
    parameters** (the 1/r coefficients in psi), the SAME object as PARASOL's
    ``m_A`` / ``m_B``, provided ``give_bare_mass = yes`` (the default — when it
    is ``no`` the thorn instead iterates bare masses to hit *target ADM* masses,
    which we do NOT use).  The "+" puncture (x=+par_b) carries ``par_m_plus``.
    ⇒ **par_m_plus = m_A** (the +b puncture),  **par_m_minus = m_B**.

  * **Bowen–York momentum.** ``par_P_plus`` / ``par_P_minus`` are the linear
    momentum 3-vectors entering the *same* Bowen–York Â^ij formula PARASOL uses
    (identical 3/(2r^2) normalization and (P_i n_j + P_j n_i - (f_ij-n_i n_j)P·n)
    structure — confirmed term-by-term in ``twopunctures.py`` against the built
    source).  In the x-axis frame, head-on infall means each puncture's momentum
    points toward the origin:

        par_P_plus  = (-P, 0, 0)   (the +x puncture moves in -x)
        par_P_minus = (+P, 0, 0)   (the -x puncture moves in +x)

    This is the x-axis image of PARASOL's z-axis (0,0,∓P).  **Sign caveat
    (load-bearing, and harmless for the headline):** Â^ij is *linear* in P, so
    Â_ij Â^ij — hence psi, u and the ADM mass — are **even in P**.  The overall
    infall-vs-explosion sign therefore does NOT affect the psi / M_ADM agreement;
    it only flips the sign of the (per-puncture) ADM *linear momentum*, which is
    compared separately and whose sign we verify directly against TwoPunctures'
    reported P_ADM.

  * **Spin.** Head-on, non-spinning: ``par_S_plus = par_S_minus = (0,0,0)``
    (spin is Milestone P2, not B1).

  * **Conformal factor.** TwoPunctures solves for the *same* regular correction
    u in  psi = 1 + m_+/(2 r_+) + m_-/(2 r_-) + u  — identical decomposition to
    PARASOL.  This is why psi-on-shared-points is the clean headline comparison.

The map is intentionally a pure data transform with no solver coupling, so it is
trivially unit-testable (round-trips, P-evenness, axis relabeling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TPParams:
    """TwoPunctures parameters for one head-on (+aligned-spin-ready) slice.

    Field names mirror the Einstein Toolkit ``TwoPunctures`` ``param.ccl`` so the
    oracle wrapper can emit them verbatim.  Punctures on the x-axis at ±par_b;
    "plus" puncture at +par_b.
    """
    par_b: float                                   # half-separation (x=±par_b)
    par_m_plus: float                              # bare mass at +par_b  (= m_A)
    par_m_minus: float                             # bare mass at -par_b  (= m_B)
    par_P_plus: Tuple[float, float, float]         # BY momentum at +par_b
    par_P_minus: Tuple[float, float, float]        # BY momentum at -par_b
    par_S_plus: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    par_S_minus: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    give_bare_mass: bool = True                    # par_m_* ARE bare masses
    center_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_cactus_par(self, prefix: str = "TwoPunctures") -> str:
        """Render as Einstein-Toolkit-style ``key = value`` parameter lines."""
        def vec(name, v):
            return "\n".join(f"{prefix}::{name}[{i}] = {v[i]!r}" for i in range(3))
        lines = [
            f"{prefix}::par_b = {self.par_b!r}",
            f"{prefix}::par_m_plus = {self.par_m_plus!r}",
            f"{prefix}::par_m_minus = {self.par_m_minus!r}",
            vec("par_P_plus", self.par_P_plus),
            vec("par_P_minus", self.par_P_minus),
            vec("par_s_plus", self.par_S_plus),
            vec("par_s_minus", self.par_S_minus),
            f'{prefix}::give_bare_mass = {"yes" if self.give_bare_mass else "no"}',
            vec("center_offset", self.center_offset),
        ]
        return "\n".join(lines)


def parasol_to_tp(b: float, m_A: float, m_B: float, P: float,
                  S_A: float = 0.0, S_B: float = 0.0) -> TPParams:
    """Map PARASOL free data ``(b, m_A, m_B, P, S_A, S_B)`` -> TwoPunctures params.

    See the module docstring for the full justification of every assignment.

    Aligned spin (Milestone P2).  PARASOL carries spins ``S_X ẑ`` along the
    z (collision) axis; the same axis relabel that takes the momentum
    ``(0,0,∓P) -> (∓P,0,0)`` is a PROPER rotation ``ẑ -> x̂`` (det +1), under
    which the spin **pseudovector** transforms identically to the polar momentum
    (a proper rotation does not introduce the parity sign), so
    ``S_X ẑ -> (S_X, 0, 0)`` in TwoPunctures' x-axis frame.  This sign is
    confirmed end-to-end against TwoPunctures' reported angular momentum
    ``J = par_S_plus + par_S_minus = (S_A + S_B, 0, 0)`` (head-on on-axis ⇒ no
    orbital contribution).  ``S_A=S_B=0`` reproduces the B1 head-on map exactly.
    """
    return TPParams(
        par_b=float(b),
        par_m_plus=float(m_A),                     # +b puncture  <-> A
        par_m_minus=float(m_B),                    # -b puncture  <-> B
        par_P_plus=(-float(P), 0.0, 0.0),          # infall, x-axis image of (0,0,-P)
        par_P_minus=(float(P), 0.0, 0.0),          # infall, x-axis image of (0,0,+P)
        par_S_plus=(float(S_A), 0.0, 0.0),         # spin along collision axis (= z->x)
        par_S_minus=(float(S_B), 0.0, 0.0),
        give_bare_mass=True,
    )


def parasol_point_to_tp(rho: float, z: float) -> Tuple[float, float, float]:
    """Map a PARASOL meridian point ``(rho, z)`` (axisymmetric about z) to a
    TwoPunctures Cartesian point on its x-axis collision frame.

    PARASOL's axial coordinate z maps to TwoPunctures' x; the cylindrical radius
    rho maps to a TP transverse offset (we place it along +y, z_TP=0).  Because
    both fields are axisymmetric about their collision axis, this preserves the
    physical (axial, cylindrical-radius) location, so psi is directly comparable.
    """
    return (float(z), float(rho), 0.0)            # (x_TP, y_TP, z_TP)


def tp_point_axial_radius(x_tp: float, y_tp: float, z_tp: float) -> Tuple[float, float]:
    """Inverse of :func:`parasol_point_to_tp`: TP Cartesian -> PARASOL (rho, z).

    Axial coordinate = x_TP -> z_PARASOL; cylindrical radius = sqrt(y^2+z^2)_TP
    -> rho_PARASOL.
    """
    return (float((y_tp ** 2 + z_tp ** 2) ** 0.5), float(x_tp))


# --------------------------------------------------------------------------
# Non-axisymmetric (Test E) extensions — the PARASOL<->TP frame is a single
# PROPER rotation taking PARASOL's z (collision) axis to TP's x (collision)
# axis.  The axisymmetric meridian map ``parasol_point_to_tp`` is the φ=0
# specialisation; the full vector/point maps below are its 3-D lift.
# --------------------------------------------------------------------------
#
# The rotation R is fixed by the established axisymmetric map: a PARASOL
# meridian point (ρ, z) at φ=0 is the Cartesian (ρ, 0, z), and it goes to TP
# (z, ρ, 0).  Demanding R be a proper rotation (det +1) that does this gives the
# cyclic permutation
#
#     R: ê_x^P -> ê_y^TP,   ê_y^P -> ê_z^TP,   ê_z^P -> ê_x^TP,
#
# i.e. for any vector  v_TP = (v_z^P, v_x^P, v_y^P).  This takes PARASOL's
# z-axis momentum (0,0,∓P) -> (∓P,0,0) and a z-aligned spin (0,0,S) -> (S,0,0),
# reproducing :func:`parasol_to_tp` exactly (consistency unit-tested).  Being a
# proper rotation, the spin PSEUDO-vector transforms identically to the polar
# momentum (no parity sign).


def parasol_vec_to_tp(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Rotate a PARASOL Cartesian 3-vector into the TwoPunctures native frame.

    The proper rotation  z^P -> x^TP  (cyclic):  ``v_TP = (v_z, v_x, v_y)``.
    Applies identically to momenta (polar) and spins (axial), since a proper
    rotation carries the pseudovector with no parity sign.
    """
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return (vz, vx, vy)


def tp_vec_to_parasol(w: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Inverse of :func:`parasol_vec_to_tp`:  ``v_PARASOL = (w_y, w_z, w_x)``."""
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return (wy, wz, wx)


def parasol_point_to_tp_3d(rho: float, z: float, phi: float) -> Tuple[float, float, float]:
    """Map a full PARASOL point ``(ρ, z, φ)`` to a TwoPunctures Cartesian point.

    PARASOL Cartesian is ``(ρ cosφ, ρ sinφ, z)``; applying :func:`parasol_vec_to_tp`
    (the cyclic z^P->x^TP rotation) gives  ``(z, ρ cosφ, ρ sinφ)`` in the TP
    native (x-axis collision) frame.  Reduces to :func:`parasol_point_to_tp` at
    φ=0.
    """
    import math
    rho, z, phi = float(rho), float(z), float(phi)
    return (z, rho * math.cos(phi), rho * math.sin(phi))
