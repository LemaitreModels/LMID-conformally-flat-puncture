"""LM-initial-data — production sweep of the first non-axisymmetric (Fourier-φ) data.

Exercises the validated 3-D two-centre solver (``solver_3d``) and the 3-D ADM
diagnostics (``diagnostics_3d``) over a small physical grid, and cross-checks a
handful of anchor slices against the TwoPunctures oracle (Test E machinery).

Four blocks, all written to ``reports/3D/sweep_results.json``:

  A. **Misaligned-spin grid** — over (b, |S|, tilt angle θ_S of the spin off the
     collision axis).  Per slice: ‖R‖∞ (the floor), Newton iters, M_ADM, the J
     vector (closed form + York surface integral), and the azimuthal φ-mode
     amplitude spectrum of the solved field.  The headline tables: how the floor
     and the φ-spectrum behave across the grid, and J-tilt vs spin-tilt.

  B. **φ / meridian convergence study** — one representative misaligned slice,
     resolution ladder; ‖R‖∞ and (oracle present) ψ-vs-TP.  The credibility
     anchor figure.

  C. **Off-axis-momentum (orbital-J) series** — anti-symmetric transverse
     momentum Px so the orbital term x×P feeds a 1/R tail in J; checks the
     Richardson-in-1/R surface extrapolation recovers Σ x_X×P_X.

  D. **Oracle anchors** — a few TP solves (ψ spectral agreement, M_ADM, J vector
     in the PARASOL frame).  Skipped cleanly if the binary is absent.

The TwoPunctures oracle is OPTIONAL here: block B's psi cross-check and block D
(the anchor solves) skip cleanly when the binary is absent, and the remaining
blocks -- which are what ``fig08_3d_validation`` distils -- still run.  Contrast
``run_3d_validation_sweep``, whose ``main`` returns immediately without the
oracle and which writes a DIFFERENT artifact (``3D_parametric/validation_results.json``).

Run:  python -m lm.initial_data.pipeline.run_3d_sweep
The companion plotter is ``plot_3d_sweep.py`` (reads the JSON, writes figures).
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from lm.initial_data.solver import solver_3d as s3, solver_3d_nk as s3nk, source, diagnostics_3d as d3
from lm.initial_data.solver.solver_3d import Slice3D
from lm.initial_data.validation import twopunctures as tp, conventions as cv

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REPDIR = os.path.join(REPORTS, "3D")
os.makedirs(REPDIR, exist_ok=True)

M_A = M_B = 0.5            # equal mass, M=1
P_HEADON = 0.5             # on-axis infall momentum for the spin grid


def spin_vec(mag, tilt_deg):
    """Spin of magnitude ``mag`` tilted ``tilt_deg`` off the collision (z) axis,
    in the x-z plane.  tilt=0 -> aligned (axisymmetric); tilt=90 -> transverse."""
    th = np.deg2rad(tilt_deg)
    return (mag * np.sin(th), 0.0, mag * np.cos(th))


def phi_mode_spectrum(prob, U):
    """Max-over-(A,B) amplitude of each azimuthal mode m of the solved field.

    Returns ``(m_vals, amps)`` with ``amps[k] = max_{i,j}|rfft_φ(U)[i,j,k]|/Nφ``
    — the physical-space azimuthal content of u.  m=0 is the axisymmetric part.
    """
    U3 = np.asarray(U).reshape(prob.shape)            # (Na+1, Nb, Nφ)
    Uhat = np.fft.rfft(U3, axis=2) / prob.Nphi
    amps = np.max(np.abs(Uhat), axis=(0, 1))          # (Nφ//2+1,)
    return np.arange(amps.size), amps


def J_tilt_angle(J):
    """Tilt of a J vector off the collision (z) axis, in degrees (in the x-z
    plane); robust to J≈0."""
    Jt = np.hypot(J[0], J[1])
    return float(np.rad2deg(np.arctan2(Jt, J[2])))


def solve_slice(prob, sl, tol=1e-11, max_iter=60):
    t0 = time.time()
    # Option B: certified Newton-Krylov (exact Jacobian). info.residual_norm is the
    # EQUILIBRATED certified residual — it sits at the ~1e-11 floor and, unlike the
    # modified-Newton raw monitor, does NOT rise with resolution.
    U, info = s3nk.newton_solve_nk(prob, sl, tol=tol, max_iter=max_iter, gmres_rtol=1e-8)
    return U, info, time.time() - t0


# ==========================================================================
# Block A — misaligned-spin grid
# ==========================================================================
def block_A():
    b_vals = [1.5, 2.0, 3.0]
    S_mags = [0.1, 0.2, 0.3]
    tilts = [0.0, 30.0, 45.0, 60.0, 90.0]
    Na, Nb, Nphi = 48, 34, 8
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    rows = []
    print(f"\n=== Block A: misaligned-spin grid (Na={Na} Nb={Nb} Nphi={Nphi}) ===")
    print(f"{'b':>4} {'|S|':>5} {'tilt':>5} {'iters':>5} {'||R||':>10} "
          f"{'M_ADM':>9} {'Jtilt':>6} {'a_m1/a_m0':>10} {'t/s':>5}")
    for b in b_vals:
        for mag in S_mags:
            for tilt in tilts:
                SA = spin_vec(mag, tilt)
                sl = Slice3D(b=b, m_A=M_A, m_B=M_B,
                             P_A_vec=(0, 0, -P_HEADON), P_B_vec=(0, 0, P_HEADON),
                             S_A_vec=SA, S_B_vec=(0, 0, 0))
                U, info, dt = solve_slice(prob, sl)
                M = d3.adm_mass_spectral_3d(prob, U, sl)
                Jc = d3.adm_J_closed_form(b, sl.P_A_vec, sl.P_B_vec, SA, (0, 0, 0))
                Js = d3.adm_J_surface(b, sl.P_A_vec, sl.P_B_vec, SA, (0, 0, 0))
                mvals, amps = phi_mode_spectrum(prob, U)
                a0 = amps[0] if amps[0] > 0 else 1.0
                ratio = float(amps[1] / a0) if amps.size > 1 else 0.0
                rows.append(dict(
                    b=b, S_mag=mag, tilt_deg=tilt, iters=info.iters,
                    resid=float(info.residual_norm), M_ADM=M,
                    J_closed=Jc.tolist(), J_surface=Js.tolist(),
                    J_tilt_deg=J_tilt_angle(Jc),
                    phi_amps=amps.tolist(), m_vals=mvals.tolist(),
                    dt=dt))
                print(f"{b:>4.1f} {mag:>5.2f} {tilt:>5.0f} {info.iters:>5d} "
                      f"{info.residual_norm:>10.2e} {M:>9.5f} "
                      f"{J_tilt_angle(Jc):>6.1f} {ratio:>10.2e} {dt:>5.1f}")
    return dict(Na=Na, Nb=Nb, Nphi=Nphi, b_vals=b_vals, S_mags=S_mags,
                tilts=tilts, rows=rows)


# ==========================================================================
# Block B — φ / meridian convergence study (credibility anchor)
# ==========================================================================
def block_B(run_oracle=True):
    b, mag, tilt = 1.5, 0.3, 56.30993           # the Test-E slice S=(0.3,0,0.2)
    SA = spin_vec(mag, tilt)
    sl = Slice3D(b=b, m_A=M_A, m_B=M_B,
                 P_A_vec=(0, 0, -P_HEADON), P_B_vec=(0, 0, P_HEADON),
                 S_A_vec=SA, S_B_vec=(0, 0, 0))
    # shared query points with genuine φ-content (Test-E set)
    QR = np.array([0.4, 0.8, 0.6, 1.2, 2.0]) * b
    QZ = np.array([0.6, 0.0, -0.5, 0.3, 0.4]) * b
    QP = np.array([0.0, 1.0, 2.0, 0.5, 2.5])
    psi_tp = None
    if run_oracle and tp.available():
        print("\n[B] TP oracle solve (nA=64 nB=64 nphi=12) ...", flush=True)
        t0 = time.time()
        res = tp.solve_parasol_points_3d(b, M_A, M_B, sl.P_A_vec, sl.P_B_vec,
                                         SA, (0, 0, 0), QR, QZ, QP,
                                         nA=64, nB=64, nphi=12, timeout=1800)
        psi_tp = res.psi
        print(f"[B]   TP done in {time.time()-t0:.0f}s  E={res.E:.8f}")
    ladder = [(40, 28, 6), (48, 34, 8), (56, 40, 10), (64, 46, 12)]
    print("\n=== Block B: convergence ladder (slice S=(0.3,0,0.2), b=1.5) ===")
    print(f"{'Na':>4} {'Nb':>4} {'Nphi':>5} {'iters':>5} {'||R||':>10} "
          f"{'dpsi_vs_TP':>11} {'t/s':>5}")
    rows = []
    for (Na, Nb, Nphi) in ladder:
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
        U, info, dt = solve_slice(prob, sl)
        dpsi = None
        if psi_tp is not None:
            u = np.asarray(s3.evaluate_field(prob, U, QR, QZ, QP, b))
            psi = np.asarray(source.psi_BL_2c(QR, QZ, b, M_A, M_B)) + u
            dpsi = float(np.max(np.abs(psi - psi_tp)))
        rows.append(dict(Na=Na, Nb=Nb, Nphi=Nphi, iters=info.iters,
                         resid=float(info.residual_norm), dpsi_vs_TP=dpsi, dt=dt))
        ds = f"{dpsi:.3e}" if dpsi is not None else "   (no TP)"
        print(f"{Na:>4} {Nb:>4} {Nphi:>5} {info.iters:>5d} "
              f"{info.residual_norm:>10.2e} {ds:>11} {dt:>5.1f}")
    return dict(slice=dict(b=b, S=list(SA)), ladder=rows,
                psi_tp=(psi_tp.tolist() if psi_tp is not None else None))


# ==========================================================================
# Block C — off-axis-momentum (orbital-J) series
# ==========================================================================
def block_C():
    b, Pz = 1.5, 0.5
    Px_vals = [0.0, 0.05, 0.10, 0.15, 0.20]
    Na, Nb, Nphi = 48, 34, 8
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    rows = []
    print("\n=== Block C: off-axis momentum (orbital J along y) ===")
    print(f"{'Px':>5} {'iters':>5} {'||R||':>10} {'M_ADM':>9} "
          f"{'Jy_closed':>10} {'Jy_extrap':>10} {'Jy_surfR':>10}")
    for Px in Px_vals:
        PA = (Px, 0.0, -Pz)
        PB = (-Px, 0.0, Pz)
        sl = Slice3D(b=b, m_A=M_A, m_B=M_B, P_A_vec=PA, P_B_vec=PB,
                     S_A_vec=(0, 0, 0), S_B_vec=(0, 0, 0))
        U, info, dt = solve_slice(prob, sl)
        M = d3.adm_mass_spectral_3d(prob, U, sl)
        Jc = d3.adm_J_closed_form(b, PA, PB, (0, 0, 0), (0, 0, 0))
        Jext = d3.adm_J_surface_extrap(b, PA, PB, (0, 0, 0), (0, 0, 0))
        JsR = d3.adm_J_surface(b, PA, PB, (0, 0, 0), (0, 0, 0))   # single (un-extrap) R
        rows.append(dict(Px=Px, iters=info.iters, resid=float(info.residual_norm),
                         M_ADM=M, J_closed=Jc.tolist(),
                         J_extrap=Jext.tolist(), J_surfR=JsR.tolist(), dt=dt))
        print(f"{Px:>5.2f} {info.iters:>5d} {info.residual_norm:>10.2e} "
              f"{M:>9.5f} {Jc[1]:>10.4f} {Jext[1]:>10.4f} {JsR[1]:>10.4f}")
    return dict(b=b, Pz=Pz, Px_vals=Px_vals, rows=rows)


# ==========================================================================
# Block D — oracle anchors (ψ / M_ADM / J vs TP)
# ==========================================================================
def block_D():
    if not tp.available():
        print("\n[D] TP oracle binary absent — skipping anchors.")
        return dict(available=False, anchors=[])
    # (label, S_A_vec, P_A, P_B)
    anchors = [
        ("aligned_spin",   (0.0, 0.0, 0.2), (0, 0, -P_HEADON), (0, 0, P_HEADON)),
        ("tilt56_S0.3",    (0.3, 0.0, 0.2), (0, 0, -P_HEADON), (0, 0, P_HEADON)),
        ("transverse_S0.3",(0.3, 0.0, 0.0), (0, 0, -P_HEADON), (0, 0, P_HEADON)),
        ("offaxis_P",      (0.0, 0.0, 0.0), (0.1, 0, -P_HEADON), (-0.1, 0, P_HEADON)),
    ]
    b = 1.5
    QR = np.array([0.4, 0.8, 0.6, 1.2, 2.0]) * b
    QZ = np.array([0.6, 0.0, -0.5, 0.3, 0.4]) * b
    QP = np.array([0.0, 1.0, 2.0, 0.5, 2.5])
    prob = s3.make_problem(Na=56, Nb=40, Nphi=10)
    out = []
    print("\n=== Block D: oracle anchors (nA=56 nB=44 nphi=8 TP) ===")
    print(f"{'label':>16} {'dpsi':>10} {'dM/M':>10} {'dJ_max':>10}")
    for (label, SA, PA, PB) in anchors:
        sl = Slice3D(b=b, m_A=M_A, m_B=M_B, P_A_vec=PA, P_B_vec=PB,
                     S_A_vec=SA, S_B_vec=(0, 0, 0))
        U, info, dt = solve_slice(prob, sl)
        res = tp.solve_parasol_points_3d(b, M_A, M_B, PA, PB, SA, (0, 0, 0),
                                         QR, QZ, QP, nA=56, nB=44, nphi=8,
                                         timeout=1800)
        u = np.asarray(s3.evaluate_field(prob, U, QR, QZ, QP, b))
        psi = np.asarray(source.psi_BL_2c(QR, QZ, b, M_A, M_B)) + u
        dpsi = float(np.max(np.abs(psi - res.psi)))
        M = d3.adm_mass_spectral_3d(prob, U, sl)
        dM = abs(M - res.E) / res.E
        J_tp_par = np.array(cv.tp_vec_to_parasol(res.J))
        J_par = d3.adm_J_surface_extrap(b, PA, PB, SA, (0, 0, 0))
        dJ = float(np.max(np.abs(J_par - J_tp_par)))
        out.append(dict(label=label, S_A=list(SA), P_A=list(PA),
                        resid=float(info.residual_norm), dpsi=dpsi,
                        M_ADM=M, E_tp=res.E, dM_rel=dM,
                        J_parasol=J_par.tolist(),
                        J_tp_parasol=J_tp_par.tolist(), dJ_max=dJ))
        print(f"{label:>16} {dpsi:>10.2e} {dM:>10.2e} {dJ:>10.2e}")
    return dict(available=True, anchors=out)


def main():
    t_start = time.time()
    results = {}
    results["A_spin_grid"] = block_A()
    results["B_convergence"] = block_B(run_oracle=True)
    results["C_orbital"] = block_C()
    results["D_anchors"] = block_D()
    results["meta"] = dict(M_A=M_A, M_B=M_B, P_headon=P_HEADON,
                           wall_s=time.time() - t_start)
    out = os.path.join(REPDIR, "sweep_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}  (total {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
