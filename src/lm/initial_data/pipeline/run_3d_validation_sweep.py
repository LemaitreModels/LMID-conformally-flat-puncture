"""PARASOL 3D — TwoPunctures cross-validation of the certified parametric surrogate.

The independent-oracle credibility run for the 3-D lift.  Builds ONE certified
Newton–Krylov interpolant over the misaligned-spin family and shows that its
``evaluate_polished`` predictions agree with the TwoPunctures oracle ACROSS the
parameter family (not just at one point), plus the meridian-resolution ψ-vs-TP
ladder and the amortised "fast generator" speedup.

Blocks (→ ``reports/3D_parametric/validation_results.json`` + figures):
  V1. interpolant-vs-TP across the family — at several held-out θ=(b,θ_S), the
      certified interpolant prediction (evaluate_polished) is reconstructed to ψ
      and compared to a TwoPunctures solve at the same θ.  Records the certified
      ‖R‖ AND |dψ| vs TP at each θ.
  V2. ψ-vs-TP meridian-resolution ladder at one representative held-out θ — a
      direct certified NK solve at growing (Na,Nb,Nφ); |dψ| vs TP drops
      spectrally toward the meridian floor.
  T.  amortised cost — interpolant build time (N nodes) vs a single full NK
      solve vs an evaluate()+2-step certified polish; the break-even node count.

Skips cleanly if the TP binary is absent.  Run:

    caffeinate -ims ~/micromamba/envs/BBHFM/bin/python sandbox/parasol/run_3d_validation_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lemaitre.initial_data.solver import solver_3d as s3, source, solver_3d_nk as s3nk
from lemaitre.initial_data.parametric import parametric_nd_3d as p3
from lemaitre.initial_data.validation import twopunctures as tp

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "3D_parametric")
os.makedirs(REPDIR, exist_ok=True)

S_MAG = 0.3                       # fixed spin magnitude for the family
FIXED = {"S_mag": S_MAG}
ACTIVE = ["b", "theta_S"]
# the parameter box (small b: genuinely non-trivial field; TP is happiest here)
AXES = [{"name": "b", "min": 1.3, "max": 2.0, "Q": 6},
        {"name": "theta_S", "min": 30.0, "max": 70.0, "Q": 6}]
TP_TIMEOUT = 2400


def _t(m):
    print(m, flush=True)


def _query_points(b):
    QR = np.array([0.4, 0.8, 0.6, 1.2, 2.0]) * b
    QZ = np.array([0.6, 0.0, -0.5, 0.3, 0.4]) * b
    QP = np.array([0.0, 1.0, 2.0, 0.5, 2.5])
    return QR, QZ, QP


def _psi_from_U(prob, U, sl, QR, QZ, QP):
    u = np.asarray(s3.evaluate_field(prob, U, QR, QZ, QP, sl.b))
    return np.asarray(source.psi_BL_2c(QR, QZ, sl.b, sl.m_A, sl.m_B)) + u


# ==========================================================================
# V1 — certified interpolant vs TwoPunctures across the family
# ==========================================================================
def block_V1():
    Na, Nb, Nphi = 56, 40, 10
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    t0 = time.time()
    ps = p3.from_problem_nd_3d(prob, AXES, fixed=FIXED, solver="nk").build()
    build_s = time.time() - t0
    _t(f"\n=== V1: certified interpolant (Na={Na} Nb={Nb} Nφ={Nphi}, "
       f"Q=6x6={ps.n_nodes} nodes) built in {build_s:.0f}s; "
       f"max node ‖R‖={ps.residuals.max():.2e} ===")
    hold = p3.holdout_points_nd(AXES, n_points=6)
    rows = []
    for th in hold:
        sl = p3.theta_to_slice3d(th, ACTIVE, fixed=FIXED)
        U, info = ps.evaluate_polished(th, newton_steps=2)
        QR, QZ, QP = _query_points(sl.b)
        res = tp.solve_parasol_points_3d(
            sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec,
            QR, QZ, QP, nA=64, nB=64, nphi=12, timeout=TP_TIMEOUT)
        psi = _psi_from_U(prob, U, sl, QR, QZ, QP)
        dpsi = float(np.max(np.abs(psi - res.psi)))
        rows.append(dict(theta=[float(x) for x in th],
                         certified_resid=float(info.residual_norm),
                         dpsi_vs_TP=dpsi, E_tp=float(res.E)))
        _t(f"   θ=(b={th[0]:.3f}, tilt={th[1]:.2f}°)  certified‖R‖="
           f"{info.residual_norm:.2e}  |dψ-TP|={dpsi:.3e}  E_tp={res.E:.6f}")
    return dict(Na=Na, Nb=Nb, Nphi=Nphi, n_nodes=ps.n_nodes,
                build_s=build_s, max_node_resid=float(ps.residuals.max()),
                holdout=rows)


# ==========================================================================
# V2 — ψ-vs-TP meridian-resolution ladder at one held-out θ
# ==========================================================================
def block_V2():
    th = p3.holdout_points_nd(AXES, n_points=1)[0]
    sl = p3.theta_to_slice3d(th, ACTIVE, fixed=FIXED)
    QR, QZ, QP = _query_points(sl.b)
    _t(f"\n=== V2: ψ-vs-TP resolution ladder at θ=(b={th[0]:.3f}, tilt={th[1]:.2f}°) ===")
    # one fine TP reference
    res = tp.solve_parasol_points_3d(
        sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec,
        QR, QZ, QP, nA=72, nB=72, nphi=12, timeout=TP_TIMEOUT)
    ladder = [(40, 28, 8), (48, 34, 8), (56, 40, 10), (64, 46, 12)]
    rows = []
    for (Na, Nb, Nphi) in ladder:
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
        U, info = s3nk.newton_solve_nk(prob, sl, tol=1e-10, max_iter=20)
        psi = _psi_from_U(prob, U, sl, QR, QZ, QP)
        dpsi = float(np.max(np.abs(psi - res.psi)))
        rows.append(dict(Na=Na, Nb=Nb, Nphi=Nphi,
                         certified_resid=float(info.residual_norm), dpsi_vs_TP=dpsi))
        _t(f"   Na={Na} Nb={Nb} Nφ={Nphi}  certified‖R‖={info.residual_norm:.2e}  "
           f"|dψ-TP|={dpsi:.3e}")
    return dict(theta=[float(x) for x in th], E_tp=float(res.E), ladder=rows)


# ==========================================================================
# T — amortised cost (interpolant build vs solve vs evaluate+polish)
# ==========================================================================
def block_T():
    Na, Nb, Nphi = 56, 40, 10
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    th = p3.holdout_points_nd(AXES, n_points=1)[0]
    sl = p3.theta_to_slice3d(th, ACTIVE, fixed=FIXED)
    # cold full NK solve
    t0 = time.time(); s3nk.newton_solve_nk(prob, sl, tol=1e-10, max_iter=20); t_solve = time.time() - t0
    # interpolant build
    t0 = time.time(); ps = p3.from_problem_nd_3d(prob, AXES, fixed=FIXED, solver="nk").build(); t_build = time.time() - t0
    # evaluate + 2-step certified polish
    t0 = time.time(); ps.evaluate_polished(th, newton_steps=2); t_eval = time.time() - t0
    # pure interpolant evaluate (no polish)
    t0 = time.time(); [ps.evaluate(p3.holdout_points_nd(AXES, n_points=1)[0]) for _ in range(20)]; t_pure = (time.time() - t0) / 20
    breakeven = t_build / max(t_solve - t_eval, 1e-9)
    _t(f"\n=== T: amortised cost (Na={Na} Nb={Nb} Nφ={Nphi}, {ps.n_nodes}-node interpolant) ===")
    _t(f"   full NK solve:        {t_solve:.2f}s")
    _t(f"   interpolant build:    {t_build:.1f}s ({ps.n_nodes} nodes)")
    _t(f"   evaluate+2-step polish:{t_eval:.2f}s  (certified)")
    _t(f"   pure evaluate():      {t_pure*1e3:.2f} ms  ({t_solve/max(t_pure,1e-12):.0f}x faster than a solve)")
    _t(f"   build break-even:     {breakeven:.0f} certified queries")
    return dict(Na=Na, Nb=Nb, Nphi=Nphi, n_nodes=ps.n_nodes,
                t_solve=t_solve, t_build=t_build, t_eval_polish=t_eval,
                t_pure_eval=t_pure, breakeven_queries=breakeven)


def make_figures(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        _t(f"[fig] matplotlib unavailable ({e})"); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    V1 = results["V1_interp_vs_tp"]["holdout"]
    idx = np.arange(len(V1))
    ax1.semilogy(idx, [r["dpsi_vs_TP"] for r in V1], "o-", label="|dψ| interp vs TP")
    ax1.semilogy(idx, [r["certified_resid"] for r in V1], "s--", label="certified ‖R‖")
    ax1.set_xlabel("held-out θ index (across the family)")
    ax1.set_ylabel("error")
    ax1.set_title("V1: certified interpolant vs TwoPunctures across the family")
    ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=8)
    V2 = results["V2_resolution_ladder"]["ladder"]
    ax2.semilogy([r["Na"] for r in V2], [r["dpsi_vs_TP"] for r in V2], "o-",
                 label="|dψ| vs TP")
    ax2.set_xlabel("meridian resolution Na")
    ax2.set_ylabel("|dψ| vs TwoPunctures")
    ax2.set_title("V2: ψ-vs-TP meridian-resolution ladder")
    ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPDIR, "fig_tp_validation.png"), dpi=140)
    plt.close(fig)
    _t(f"[fig] wrote fig_tp_validation.png to {REPDIR}")


def main():
    if not tp.available():
        _t("TwoPunctures binary absent — nothing to do."); return
    t_start = time.time()
    results = {}
    results["V1_interp_vs_tp"] = block_V1()
    results["V2_resolution_ladder"] = block_V2()
    results["T_amortised"] = block_T()
    results["meta"] = dict(S_mag=S_MAG, axes=AXES, wall_s=time.time() - t_start)
    out = os.path.join(REPDIR, "validation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _t(f"\nWrote {out}")
    make_figures(results)
    _t(f"\nTOTAL {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
