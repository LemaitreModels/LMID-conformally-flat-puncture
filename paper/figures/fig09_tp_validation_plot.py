#!/usr/bin/env python
"""Generates fig09_tp_validation.pdf (paper Fig. 9).

Consolidated TwoPunctures validation, three panels:
 (a) ADM-J tilt off the collision axis tracks the spin tilt (theta_J = theta_S),
     with TwoPunctures anchors overlaid -- the genuinely 3-D observable;
 (b) quasi-circular max|psi_PARASOL - psi_TP| vs PARASOL grid;
 (c) quasi-circular |M_ADM - E_TP|/E_TP vs PARASOL grid.

Reads ONLY figdata/fig09_tp_validation.json (build it with
fig09_tp_validation_data.py). No reports/, no jax.

Run:  python fig09_tp_validation_plot.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _figdata import load
from _figstyle import figdims


def main():
    d = load("fig09_tp_validation")
    C = d["C_psi_adm"]
    ns = [f"{r['Na']}$\\times${r['Nb']}" for r in C]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figdims(1, 3))

    # (a) ADM-J tilt tracks spin tilt, with TwoPunctures anchors
    # reserve C0-C2 (tab:blue/orange/green) for the paper's models; start at C3
    for i, pa in enumerate(d["panelA"]):
        ax1.plot(pa["ts"], pa["jt"], "o-", color=f"C{3 + i}", label=fr"$|S|={pa['S_mag']}$")
    ax1.plot([0, 90], [0, 90], "k:", lw=1, label=r"$\theta_J=\theta_S$")
    if d["anchors_available"]:
        for tS, tJ in d["anchors"]:
            ax1.plot(tS, tJ, "k*", ms=12, zorder=5)
        ax1.plot([], [], "k*", ms=10, label="TwoPunctures")
    ax1.set_xlabel(r"spin tilt $\theta_S$ (deg)")
    ax1.set_ylabel(r"ADM-$J$ tilt $\theta_J$ (deg)")
    ax1.set_title(r"(a) ADM-$J$ tracks the spin")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (b) quasi-circular psi vs TwoPunctures
    ax2.semilogy(range(len(C)), [r["max_dpsi"] for r in C], "o-", color="C3")
    ax2.set_xticks(range(len(C)))
    ax2.set_xticklabels(ns, rotation=30)
    ax2.set_xlabel(r"PARASOL grid $N_A\times N_B$ ($N_\phi=8$)")
    ax2.set_ylabel(r"$\max|\psi_{\rm PARASOL}-\psi_{\rm TP}|$")
    ax2.set_title(r"(b) quasi-circular field vs TP")
    ax2.grid(True, which="both", alpha=0.3)

    # (c) quasi-circular ADM mass vs TwoPunctures
    ax3.semilogy(range(len(C)), [r["M_ADM_rel_diff"] for r in C], "o-", color="C4")
    ax3.set_xticks(range(len(C)))
    ax3.set_xticklabels(ns, rotation=30)
    ax3.set_xlabel(r"PARASOL grid $N_A\times N_B$")
    ax3.set_ylabel(r"$|M_{\rm ADM}-E_{\rm TP}|/E_{\rm TP}$")
    ax3.set_title(r"(c) quasi-circular ADM mass vs TP")
    ax3.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig09_tp_validation")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig09_tp_validation.pdf")


if __name__ == "__main__":
    main()
