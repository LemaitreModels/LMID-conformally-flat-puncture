"""PARASOL — figures for the first non-axisymmetric (3-D) data sweep.

Reads ``reports/3D/sweep_results.json`` (written by ``run_3d_sweep.py``) and
writes two publication-quality figures:

  fig09_3d_validation.png  (two panels)
    (a) Convergence ladder — ‖R‖∞ and ψ-vs-TwoPunctures vs meridian resolution
        for a genuinely non-axisymmetric slice (the credibility anchor).
    (b) Azimuthal φ-mode amplitude spectrum of the solved field — exponential
        decay in m, confirming that few Fourier modes resolve a minimal break of
        axisymmetry.

  fig10_3d_angular_momentum.png  (two panels)
    (a) ADM-J tilt off the collision axis tracks the spin tilt (J=Σ S_X for
        pure spin), with the TwoPunctures anchors overlaid.
    (b) Orbital J_y vs off-axis momentum P_x — the 1/R-extrapolated York surface
        integral recovers the closed form Σ x_X×P_X.

Also writes ``reports/3D/floor_table.txt`` (the ‖R‖∞ floor across the grid).
Run:  ~/micromamba/envs/BBHFM/bin/python sandbox/parasol/plot_3d_sweep.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "3D")
FIGDIR = os.path.join(HERE, "figures")
MANFIG = os.path.join(HERE, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

with open(os.path.join(REPDIR, "sweep_results.json")) as f:
    R = json.load(f)


def _save(fig, name):
    for d in (FIGDIR, MANFIG):
        if os.path.isdir(d):
            fig.savefig(os.path.join(d, name), dpi=150, bbox_inches="tight")
    print("wrote", name)


# ==========================================================================
# fig09 — convergence anchor + φ-mode spectrum
# ==========================================================================
def fig_validation():
    B = R["B_convergence"]["ladder"]
    Na = np.array([r["Na"] for r in B])
    resid = np.array([r["resid"] for r in B])
    dpsi = np.array([r["dpsi_vs_TP"] if r["dpsi_vs_TP"] is not None else np.nan
                     for r in B])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    ax1.semilogy(Na, resid, "o-", label=r"$\|R\|_\infty$ (residual)")
    if np.any(np.isfinite(dpsi)):
        ax1.semilogy(Na, dpsi, "s--",
                     label=r"$|\psi-\psi_{\rm TP}|_\infty$")
    ax1.set_xlabel(r"meridian resolution $N_A$  (with $N_B,N_\phi$ scaled)")
    ax1.set_ylabel("error")
    ax1.set_title("(a) non-axisymmetric convergence vs TwoPunctures")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # φ-spectrum: pick the hardest spin slice in the grid (largest |S|, tilt 90)
    rows = R["A_spin_grid"]["rows"]
    sel = [r for r in rows if r["b"] == 1.5 and r["S_mag"] == 0.3]
    for r in sorted(sel, key=lambda r: r["tilt_deg"]):
        amps = np.array(r["phi_amps"])
        m = np.array(r["m_vals"])
        amps_n = np.where(amps > 0, amps, np.nan)
        ax2.semilogy(m, amps_n, "o-", label=fr"$\theta_S={r['tilt_deg']:.0f}^\circ$")
    ax2.set_xlabel(r"azimuthal mode $m$")
    ax2.set_ylabel(r"$\max_{A,B}|\hat u_m|$")
    ax2.set_title(r"(b) $\phi$-mode spectrum ($|S|=0.3$, $b=1.5$)")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig09_3d_validation.png")
    plt.close(fig)


# ==========================================================================
# fig10 — J tilt vs spin tilt + orbital J recovery
# ==========================================================================
def fig_angular_momentum():
    rows = R["A_spin_grid"]["rows"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    # (a) J tilt vs spin tilt, at b=1.5 for each |S|
    for mag in R["A_spin_grid"]["S_mags"]:
        sel = sorted([r for r in rows if r["b"] == 1.5 and r["S_mag"] == mag],
                     key=lambda r: r["tilt_deg"])
        ts = [r["tilt_deg"] for r in sel]
        jt = [r["J_tilt_deg"] for r in sel]
        ax1.plot(ts, jt, "o-", label=fr"$|S|={mag}$")
    ax1.plot([0, 90], [0, 90], "k:", lw=1, label=r"$\theta_J=\theta_S$")
    # TP anchors
    D = R.get("D_anchors", {})
    if D.get("available"):
        for a in D["anchors"]:
            S = np.array(a["S_A"])
            if np.hypot(S[0], S[1]) > 0 or S[2] != 0:
                tS = np.rad2deg(np.arctan2(np.hypot(S[0], S[1]), S[2])) if np.any(S) else 0
                Jtp = np.array(a["J_tp_parasol"])
                tJ = np.rad2deg(np.arctan2(np.hypot(Jtp[0], Jtp[1]), Jtp[2]))
                ax1.plot(tS, tJ, "k*", ms=12, zorder=5)
        ax1.plot([], [], "k*", ms=10, label="TwoPunctures")
    ax1.set_xlabel(r"spin tilt $\theta_S$ (deg)")
    ax1.set_ylabel(r"ADM-$J$ tilt $\theta_J$ (deg)")
    ax1.set_title("(a) angular momentum tracks the spin")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (b) orbital J_y vs P_x
    C = R["C_orbital"]
    Px = np.array([r["Px"] for r in C["rows"]])
    Jy_c = np.array([r["J_closed"][1] for r in C["rows"]])
    Jy_e = np.array([r["J_extrap"][1] for r in C["rows"]])
    Jy_R = np.array([r["J_surfR"][1] for r in C["rows"]])
    ax2.plot(Px, Jy_c, "k-", lw=2, label=r"closed form $\sum x_X\times P_X$")
    ax2.plot(Px, Jy_e, "o", ms=7, label=r"surface, $R\to\infty$ extrap.")
    ax2.plot(Px, Jy_R, "x", ms=7, label=r"surface, single $R$")
    ax2.set_xlabel(r"off-axis momentum $P_x$")
    ax2.set_ylabel(r"orbital $J_y$")
    ax2.set_title(fr"(b) orbital $J$ recovery ($b={C['b']}$)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig10_3d_angular_momentum.png")
    plt.close(fig)


# ==========================================================================
# floor table
# ==========================================================================
def floor_table():
    rows = R["A_spin_grid"]["rows"]
    lines = ["# ||R||_inf floor across the misaligned-spin grid",
             f"# {'b':>4} {'|S|':>5} {'tilt':>5} {'iters':>5} {'||R||':>11} {'M_ADM':>9}"]
    for r in rows:
        lines.append(f"  {r['b']:>4.1f} {r['S_mag']:>5.2f} {r['tilt_deg']:>5.0f} "
                     f"{r['iters']:>5d} {r['resid']:>11.3e} {r['M_ADM']:>9.5f}")
    resids = np.array([r["resid"] for r in rows])
    lines.append(f"\n# floor stats: min={resids.min():.2e} median="
                 f"{np.median(resids):.2e} max={resids.max():.2e}")
    txt = "\n".join(lines)
    with open(os.path.join(REPDIR, "floor_table.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    fig_validation()
    fig_angular_momentum()
    floor_table()
