"""External oracle wrapper: the standalone TwoPunctures solver  (B1, Step 4).

This is the **only** place the LM-initial-data package reaches outside itself.  It does
NOT import the oracle's build-time deps (nrpy etc.); it merely shells out (via
``subprocess``) to a pre-built standalone binary that solves the
Ansorg–Brügmann–Tichy puncture equation (PRD 70, 064011) — the Einstein-Toolkit
TwoPunctures code, ported to C by Z. Etienne (NRPy) and compiled against GSL.

Build recipe (outside the package, so the package stays jax/numpy/matplotlib):
    bash ~/.cache/bbhfm/parasol_tp_oracle/build.sh        # -> tp_solve binary

The binary reads ``b mA mB P nA nB nphi`` from argv and Cartesian query points
``x y z`` (TP native x-axis frame) from stdin; it writes a ``SUMMARY`` line
(ADM masses, J) and one ``POINT`` line per query (u and psi).  If the binary is
absent, :func:`available` returns False so the oracle-dependent tests skip
cleanly while the oracle-independent B1 deliverables still run.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import numpy as np

from . import conventions


# Default build location (see build.sh); override with LM_TP_BIN.
_DEFAULT_BIN = os.path.expanduser("~/.cache/bbhfm/parasol_tp_oracle/tp_solve")


def binary_path() -> str:
    """Path to the TwoPunctures binary (env ``LM_TP_BIN`` or the default)."""
    return os.environ.get("LM_TP_BIN", _DEFAULT_BIN)


def available() -> bool:
    """True iff the compiled TwoPunctures binary is present and executable."""
    p = binary_path()
    return os.path.isfile(p) and os.access(p, os.X_OK)


@dataclass
class TPResult:
    """Parsed output of one TwoPunctures solve."""
    b: float
    mp: float            # bare mass at +b (= m_A)
    mm: float            # bare mass at -b (= m_B)
    E: float             # total ADM mass
    mp_adm: float        # individual ADM mass of the +b puncture
    mm_adm: float        # individual ADM mass of the -b puncture
    J: np.ndarray        # (3,) ADM angular momentum
    points_tp: np.ndarray  # (N,3) query points in TP native frame
    u: np.ndarray        # (N,) regular correction at the query points
    psi: np.ndarray      # (N,) full conformal factor at the query points


def solve_tp(b, m_A, m_B, P, points_tp, nA=48, nB=48, nphi=4,
             newton_tol=1e-12, newton_maxit=12, S_A=0.0, S_B=0.0,
             timeout=600) -> TPResult:
    """Run TwoPunctures at the head-on slice and evaluate u/psi at ``points_tp``.

    ``points_tp`` is an (N,3) array of Cartesian points in TwoPunctures' native
    x-axis frame (m+ at (+b,0,0)).  Bare masses m_A (-> par_m_plus) and m_B
    (-> par_m_minus) per :mod:`lm.initial_data.validation.conventions`.

    ``S_A, S_B`` are the aligned (∥z) LM-initial-data spins (Milestone P2); they are
    passed to the binary as the collision-axis (x) spin components
    ``par_S_plus[0]=S_A``, ``par_S_minus[0]=S_B`` (see ``conventions``).  They
    default to 0, reproducing the B1 head-on solve exactly.

    ``nphi=4`` is the default: head-on (on-axis-momentum) **and aligned-spin**
    data is axisymmetric about the collision axis, so the phi-Fourier carries
    only the m=0 mode and nphi=4 reproduces nphi=12/20 (verified) at ~5x lower
    cost.
    """
    if not available():
        raise RuntimeError(
            f"TwoPunctures binary not found at {binary_path()!r}. "
            "Build it with ~/.cache/bbhfm/parasol_tp_oracle/build.sh "
            "or set LM_TP_BIN.")
    pts = np.atleast_2d(np.asarray(points_tp, dtype=float))
    stdin = "".join(f"{x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in pts)
    cmd = [binary_path(), repr(float(b)), repr(float(m_A)), repr(float(m_B)),
           repr(float(P)), str(int(nA)), str(int(nB)), str(int(nphi)),
           repr(float(newton_tol)), str(int(newton_maxit)),
           repr(float(S_A)), repr(float(S_B))]
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"TwoPunctures failed (rc={proc.returncode}):\n"
                           f"{proc.stderr[-2000:]}")
    summary = None
    us, psis, pts_out = [], [], []
    for ln in proc.stdout.splitlines():
        if ln.startswith("SUMMARY"):
            kv = dict(tok.split("=") for tok in ln.split()[1:])
            summary = {k: float(v) for k, v in kv.items()}
        elif ln.startswith("POINT"):
            _, x, y, z, u, psi = ln.split()
            pts_out.append([float(x), float(y), float(z)])
            us.append(float(u))
            psis.append(float(psi))
    if summary is None:
        raise RuntimeError(f"no SUMMARY line in TwoPunctures output:\n{proc.stdout[:2000]}")
    return TPResult(
        b=summary["b"], mp=summary["mp"], mm=summary["mm"], E=summary["E"],
        mp_adm=summary["mp_adm"], mm_adm=summary["mm_adm"],
        J=np.array([summary["J1"], summary["J2"], summary["J3"]]),
        points_tp=np.array(pts_out) if pts_out else np.empty((0, 3)),
        u=np.array(us), psi=np.array(psis))


def solve_lm_initial_data_points(b, m_A, m_B, P, rho, z, S_A=0.0, S_B=0.0, **kw) -> TPResult:
    """Run TwoPunctures and evaluate psi at LM-initial-data meridian points ``(rho, z)``.

    Maps each LM-initial-data (rho, z) to the TwoPunctures native frame via
    :func:`conventions.lm_initial_data_point_to_tp` (axial z -> x_TP, radius rho -> y_TP).
    ``S_A, S_B`` are the aligned LM-initial-data spins (P2; default 0 = head-on).
    The returned ``psi``/``u`` arrays are aligned to the input (rho, z) order.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    pts_tp = np.array([conventions.lm_initial_data_point_to_tp(float(r), float(zz))
                       for r, zz in zip(rho, z)])
    return solve_tp(b, m_A, m_B, P, pts_tp, S_A=S_A, S_B=S_B, **kw)


# --------------------------------------------------------------------------
# Non-axisymmetric (Test E) — full per-puncture VECTOR momenta / spins.
# Uses the extended binary's argc>=24 override path; the 12 vector components
# are rotated LM-initial-data -> TP frame here (conventions.lm_initial_data_vec_to_tp) so the C
# side receives TP-native vectors.  The scalar solve_tp above is unchanged.
# --------------------------------------------------------------------------
def solve_tp_3d(b, m_A, m_B, P_A_vec, P_B_vec, S_A_vec, S_B_vec, points_tp,
                nA=48, nB=48, nphi=8, newton_tol=1e-12, newton_maxit=12,
                timeout=1200) -> TPResult:
    """Run TwoPunctures with arbitrary per-puncture momentum/spin VECTORS.

    ``P_A_vec, P_B_vec, S_A_vec, S_B_vec`` are LM-initial-data-frame Cartesian 3-vectors
    (punctures on the z-axis at ±b); they are rotated into the TP native x-axis
    frame via :func:`conventions.lm_initial_data_vec_to_tp`.  ``points_tp`` is (N,3) in
    the TP native frame.  Requires the extended binary (argc>=24 override path).
    """
    if not available():
        raise RuntimeError(
            f"TwoPunctures binary not found at {binary_path()!r}. "
            "Build it with ~/.cache/bbhfm/parasol_tp_oracle/build.sh "
            "or set LM_TP_BIN.")
    Pp = conventions.lm_initial_data_vec_to_tp(P_A_vec)
    Pm = conventions.lm_initial_data_vec_to_tp(P_B_vec)
    Sp = conventions.lm_initial_data_vec_to_tp(S_A_vec)
    Sm = conventions.lm_initial_data_vec_to_tp(S_B_vec)
    pts = np.atleast_2d(np.asarray(points_tp, dtype=float))
    stdin = "".join(f"{x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in pts)
    # argv 1..11 keep the scalar shape (P/SA/SB are ignored once the >=24
    # override fires); 12..23 carry the rotated TP-frame vectors.
    vecs = list(Pp) + list(Pm) + list(Sp) + list(Sm)
    cmd = ([binary_path(), repr(float(b)), repr(float(m_A)), repr(float(m_B)),
            repr(0.0), str(int(nA)), str(int(nB)), str(int(nphi)),
            repr(float(newton_tol)), str(int(newton_maxit)), repr(0.0), repr(0.0)]
           + [repr(float(c)) for c in vecs])
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"TwoPunctures failed (rc={proc.returncode}):\n"
                           f"{proc.stderr[-2000:]}")
    summary = None
    us, psis, pts_out = [], [], []
    for ln in proc.stdout.splitlines():
        if ln.startswith("SUMMARY"):
            kv = dict(tok.split("=") for tok in ln.split()[1:])
            summary = {k: float(v) for k, v in kv.items()}
        elif ln.startswith("POINT"):
            _, x, y, z, u, psi = ln.split()
            pts_out.append([float(x), float(y), float(z)])
            us.append(float(u))
            psis.append(float(psi))
    if summary is None:
        raise RuntimeError(f"no SUMMARY line in TwoPunctures output:\n{proc.stdout[:2000]}")
    return TPResult(
        b=summary["b"], mp=summary["mp"], mm=summary["mm"], E=summary["E"],
        mp_adm=summary["mp_adm"], mm_adm=summary["mm_adm"],
        J=np.array([summary["J1"], summary["J2"], summary["J3"]]),
        points_tp=np.array(pts_out) if pts_out else np.empty((0, 3)),
        u=np.array(us), psi=np.array(psis))


def solve_lm_initial_data_points_3d(b, m_A, m_B, P_A_vec, P_B_vec, S_A_vec, S_B_vec,
                            rho, z, phi, **kw) -> TPResult:
    """Run TwoPunctures (vector data) and evaluate psi at LM-initial-data points (ρ,z,φ).

    Maps each LM-initial-data ``(ρ, z, φ)`` to the TP native frame via
    :func:`conventions.lm_initial_data_point_to_tp_3d` (the cyclic z^P->x^TP rotation),
    and passes the per-puncture momentum/spin vectors through
    :func:`solve_tp_3d`.  Returned ``psi``/``u`` are aligned to the input order;
    ``J`` is in the TP native frame (rotate with ``conventions.tp_vec_to_lm_initial_data``
    to compare in the LM-initial-data frame).
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    pts_tp = np.array([conventions.lm_initial_data_point_to_tp_3d(float(r), float(zz), float(pp))
                       for r, zz, pp in zip(rho, z, phi)])
    return solve_tp_3d(b, m_A, m_B, P_A_vec, P_B_vec, S_A_vec, S_B_vec, pts_tp, **kw)
