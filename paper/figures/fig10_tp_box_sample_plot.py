#!/usr/bin/env python
"""Generates fig10_tp_box_sample.pdf (paper Fig. 10).

TwoPunctures agreement as a DISTRIBUTION over the production parameter box -- a 2x2
figure at \\textwidth (a `figure*`, as Fig. 8 is), on the flatter stacked panel:
 (a) empirical CDF of the pointwise agreement max|psi - psi_TP| over the interior sample,
     one curve per solve grid, with the box-edge stress configurations as a rug and the
     oracle's own self-convergence floor as the resolution limit of the comparison;
 (b) the same for the integral agreement |M_ADM - E_TP|/E_TP;
 (c), (d) attribution: the pointwise agreement against the mass ratio q and against the
     separation b, coloured by the spin magnitude.  BOTH axes are needed -- measured, at
     fixed spin the disagreement grows ~100x from q=1 to q=3, and a further ~10x from
     b=3 to b=10 -- and together they carry the q != 1 and production-spin coverage that
     the fixed-configuration ladders of Figs. 8-9 do not have.

This figure answers a DIFFERENT question from Figs. 8 and 9 and does not replace them:
those are convergence ladders at a fixed configuration (the x axis is resolution), which
is what shows the difference to be resolution-limited rather than solver-limited.  Here
the per-grid curves in (a) and (b) generalize that: the whole distribution shifting down
under refinement is the box-wide version of the same statement.

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

# Grid styling, coarsest -> finest.  The FIRST grid is the production one -- the grid
# whose certificate is meaningful and whose numbers the text quotes -- so it carries the
# panel colour; refinements are lighter and dashed, and still read as "the distribution
# moves left".
GRID_STYLE = [dict(color=None, ls="-", lw=1.7), dict(color="0.5", ls="--", lw=1.3),
              dict(color="0.7", ls="-.", lw=1.2)]


def _cdf(v):
    """Empirical CDF as a staircase: sorted values against fraction <= value."""
    v = np.sort(np.asarray(v, dtype=float))
    return v, np.arange(1, v.size + 1) / v.size


def _gridlabel(g):
    return rf"${g[0]}\times{g[1]}$, $N_\phi={g[2]}$"


def _panel_cdf(ax, I, E, key, grids, colour, floor=None):
    for i, g in enumerate(grids):
        st = dict(GRID_STYLE[min(i, len(GRID_STYLE) - 1)])
        st["color"] = st["color"] or colour
        v, F = _cdf([r["per_grid"][i][key] for r in I])
        ax.step(v, F, where="post", label=_gridlabel(g), **st)
        if E and i == 0:                   # edge stress set at the production grid
            ax.plot([r["per_grid"][i][key] for r in E], np.full(len(E), 0.025), "|",
                    color=colour, ms=7, alpha=0.9)
    if floor:
        ax.axvline(floor, ls=":", color="0.4", lw=1.1)
        ax.text(floor * 1.4, 0.5, "oracle floor", rotation=90, fontsize=7,
                color="0.35", va="center")
    ax.set_xscale("log")
    ax.set_ylabel("fraction of sample")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")


def main():
    d = load("fig10_tp_box_sample")
    I, E = d["interior"], d["edge"]
    grids = d["meta"]["grids"]
    floor = d.get("oracle_floor")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=figdims(2, 2, panel_h=PANEL_H_STACK))

    # (a)+(b) the two agreement metrics; C3/C4 match Fig. 9's pointwise/integral colours
    _panel_cdf(ax1, I, E, "max_dpsi", grids, "C3", floor)
    ax1.set_xlabel(r"$\|\psi-\psi_{\rm TP}\|_\infty$")
    ax1.set_title(r"(a) pointwise field over the box", fontsize=10)

    _panel_cdf(ax2, I, E, "M_ADM_rel_diff", grids, "C4")
    ax2.set_xlabel(r"$|M_{\rm ADM}-E_{\rm TP}|/E_{\rm TP}$")
    ax2.set_title(r"(b) integral (ADM mass) over the box", fontsize=10)

    # (c),(d) attribution at the PRODUCTION grid (index 0): the grid whose certificate is
    # meaningful, and where the spread the panels explain is largest.  Colour by the
    # PHYSICAL spin magnitude, not the per-component box coordinate: the box is a
    # hyper-rectangle, so its corners reach |chi| ~ sqrt(3)*0.9, and it is the magnitude
    # the near-puncture field sharpness follows.
    y = np.array([r["per_grid"][0]["max_dpsi"] for r in I])
    chi = np.array([r["chi_mag_max"] for r in I])
    yE = [r["per_grid"][0]["max_dpsi"] for r in E]
    vmax = max(1e-12, float(chi.max()))
    for ax, key, xlab, title in (
            (ax3, "q", r"mass ratio $q$", r"(c) attribution vs $q$"),
            (ax4, "b", r"half-separation $b/M$", r"(d) attribution vs $b$")):
        sc = ax.scatter([r[key] for r in I], y, c=chi, s=15, cmap="viridis",
                        vmin=0.0, vmax=vmax, edgecolors="none")
        if E:
            ax.scatter([r[key] for r in E], yE, facecolors="none", edgecolors="C3",
                       s=28, lw=0.9, label="box edge")
            ax.legend(fontsize=7.5, loc="lower right")
        if floor:           # points at or below it are oracle-limited, not solver-limited
            ax.axhline(floor, ls=":", color="0.4", lw=1.1)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$\max_X|\boldsymbol{\chi}^{X}|$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$\|\psi-\psi_{\rm TP}\|_\infty$")
        ax.set_title(rf"{title} ({_gridlabel(grids[0])})", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig10_tp_box_sample")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig10_tp_box_sample.pdf")


if __name__ == "__main__":
    main()
