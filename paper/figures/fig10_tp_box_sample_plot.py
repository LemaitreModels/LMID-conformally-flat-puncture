#!/usr/bin/env python
"""Generates fig10_tp_box_sample.pdf (paper Fig. 10).

TwoPunctures agreement as a DISTRIBUTION over the production parameter box, a
SINGLE-COLUMN 2x1 figure:
 (a) TOP     empirical CDF of the two agreement metrics over the interior sample --
             max|psi - psi_TP| (pointwise) and |M_ADM - E_TP|/E_TP (integral) -- with
             the box-edge stress configurations marked, and the oracle's own
             self-convergence floor drawn as the resolution limit of the comparison;
 (b) BOTTOM  the same pointwise metric against the mass ratio q, coloured by max|chi|,
             which is what attributes the tail (and shows the q != 1 coverage that the
             fixed-configuration ladders of Figs. 8-9 do not have).

This figure answers a different question from Figs. 8 and 9 and does not replace them:
those are convergence ladders at a FIXED configuration (the x axis is resolution), which
is what shows the difference to be resolution-limited; this one shows whether the
agreement at such a configuration is representative of the box.

Reads ONLY figdata/fig10_tp_box_sample.json (build it with fig10_tp_box_sample_data.py).
No reports/, no jax.

Run:  python fig10_tp_box_sample_plot.py
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
from _figstyle import figdims, PANEL_H_STACK


def _cdf(v):
    """Empirical CDF as a staircase: sorted values against fraction <= value."""
    v = np.sort(np.asarray(v, dtype=float))
    return v, np.arange(1, v.size + 1) / v.size


def main():
    d = load("fig10_tp_box_sample")
    I, E = d["interior"], d["edge"]
    floor = d.get("oracle_floor")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figdims(2, 1, panel_h=PANEL_H_STACK))

    # ---- (a) the distribution over the box --------------------------------------
    # C3/C4 match Fig. 9's pointwise/integral colours (C0-C2 are reserved for models)
    for key, col, lab in ((    "max_dpsi", "C3", r"$\max|\psi-\psi_{\rm TP}|$"),
                          ("M_ADM_rel_diff", "C4", r"$|M_{\rm ADM}-E_{\rm TP}|/E_{\rm TP}$")):
        v, F = _cdf([r[key] for r in I])
        ax1.step(v, F, where="post", color=col, lw=1.6, label=lab)
        if E:                       # edge stress set: rug at the foot of the panel
            ax1.plot([r[key] for r in E], np.full(len(E), 0.02), "|",
                     color=col, ms=7, alpha=0.85)
    if floor:
        ax1.axvline(floor, ls=":", color="0.45", lw=1.2)
        ax1.text(floor * 1.35, 0.55, "oracle floor", rotation=90, fontsize=7,
                 color="0.35", va="center")
    ax1.set_xscale("log")
    ax1.set_xlabel("agreement with TwoPunctures")
    ax1.set_ylabel("fraction of sample")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title(r"(a) distribution over the production box", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=7.5, loc="lower right")

    # ---- (b) attribute the tail -------------------------------------------------
    q = np.array([r["q"] for r in I])
    y = np.array([r["max_dpsi"] for r in I])
    chi = np.array([r["chi_absmax"] for r in I])
    sc = ax2.scatter(q, y, c=chi, s=14, cmap="viridis", vmin=0.0,
                     vmax=max(1e-12, float(chi.max())), edgecolors="none")
    if E:
        ax2.scatter([r["q"] for r in E], [r["max_dpsi"] for r in E],
                    facecolors="none", edgecolors="C3", s=26, lw=0.9,
                    label="box edge")
        ax2.legend(fontsize=7.5, loc="lower right")
    cb = fig.colorbar(sc, ax=ax2, pad=0.02)
    cb.set_label(r"$\max|\chi|$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"mass ratio $q$")
    ax2.set_ylabel(r"$\max|\psi-\psi_{\rm TP}|$")
    ax2.set_title(r"(b) attribution of the tail", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig10_tp_box_sample")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig10_tp_box_sample.pdf")


if __name__ == "__main__":
    main()
