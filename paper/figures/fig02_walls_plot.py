#!/usr/bin/env python
"""Generates fig02_walls.pdf (paper Fig. 2 — the two analyticity walls).

Merges what used to be two separate figures (the separation/merger wall and the spin
wall) into one 1x2 figure of held-out convergence curves:
  LEFT  separation wall: held-out error vs separation nodes Q_b, per fit range;
  RIGHT spin wall:       held-out error vs spin nodes Q_S,       per fit range.
Each curve shows the data (markers) AND its geometric fit line eps ~ A*10^(-rho*Q);
the legend reports, per fit range, the rate rho (decades/node) and the inferred
nearest real singularity theta* (b* pinned near merger; chi* marching outward).
The former right-hand theta*-vs-range panels are folded into these legends.

Reads ONLY figdata/fig02_walls.json (build it with fig02_walls_data.py).
No reports/, no jax.

Run:  python fig02_walls_plot.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _figdata import load
from _figstyle import figdims


def _panel(ax, curves, key, label_fmt, xlabel, title):
    """Draw one wall: data markers + geometric fit line per fit range."""
    for i, w in enumerate(curves):
        Qs = np.asarray(w["Qs"], float)
        # reserve C0-C2 (tab:blue/orange/green) for the paper's models; start at C3
        (line,) = ax.semilogy(Qs, w["errs"], "o", ms=5, color=f"C{3 + i}",
                              label=label_fmt(w[key], w["rate"], w["theta_star"]))
        col = line.get_color()
        # geometric fit line over the fitted window (first n_fit points)
        Qw = np.array([Qs[0], Qs[w["n_fit"] - 1]])
        ax.plot(Qw, 10.0 ** (w["fit_slope"] * Qw + w["fit_intercept"]),
                "--", lw=1.2, color=col, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("held-out interpolation error")
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_minor_locator(NullLocator())  # no y sub-ticks
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8)


def main():
    d = load("fig02_walls")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figdims(1, 2))

    _panel(ax1, d["B_wall_b"], "b_min",
           lambda r, rho, th: rf"$b_{{\min}}={r}$;   fit:  $\rho={rho:.2f}$,  $b_\ast={th:.2f}$",
           r"separation nodes  $Q_b$",
           "Separation (merger) wall — hard, real")
    _panel(ax2, d["C_wall_spin"], "chi_max",
           lambda r, rho, th: rf"$\chi_{{\max}}={r:.1f}$;   fit:  $\rho={rho:.2f}$,  $\chi_\ast={th:.1f}$",
           r"spin nodes  $Q_S$",
           "Spin wall — soft, complex")

    fig.tight_layout()
    stem = os.path.join(HERE, "fig02_walls")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig02_walls.pdf")


if __name__ == "__main__":
    main()
