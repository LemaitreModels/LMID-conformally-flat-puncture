#!/usr/bin/env python
"""Generates fig02_walls_row.pdf: paper Fig. 2 as ONE ROW of three panels with a shared y-axis.

This is the version the paper uses since 2026-11-08 (a `figure*` at \\textwidth).  The stacked
single-column original is kept as fig02_walls_plot.py / fig02_walls.pdf, so the paper can revert to
it by switching the \\includegraphics and the figure environment back.  What makes the row work where
the note below found it did not: the canvas is 9 in wide (the Figs. 3-5 width, so it scales like
them), the legend entries drop the "fit:" label, and every legend sits lower left, kept off the data
by a shared y-axis that runs down to 1e-14, two decades below the float64 floor the curves reach
(Frederik, 2026-11-08).  The rest of this docstring describes the original.

Original description -- generates fig02_walls.pdf (paper Fig. 2 — the three analyticity walls).

A SINGLE-COLUMN 3x1 figure of held-out convergence curves, stacked in order of how
close the inferred wall sits to the sampled box:
  TOP     separation wall: held-out error vs separation nodes Q_b, per fit range;
  MIDDLE  mass-ratio wall: held-out error vs mass-ratio nodes Q_q, per fit range;
  BOTTOM  spin wall:       held-out error vs spin nodes Q_chi,     per fit range.
Each curve shows the data (markers) AND its geometric fit line eps ~ A*10^(-zeta*Q);
the legend reports, per fit range, the rate zeta (decades/node) and the inferred
nearest real singularity theta* (b* pinned near merger; chi* marching outward).
The former theta*-vs-range panels are folded into these legends.

Paper side: this is a `figure` at width=\\columnwidth, NOT a `figure*`.

WHY SINGLE COLUMN.  figdims scales the canvas by ncol while LaTeX scales the result
to its target width, so the RENDERED text of a wide figure shrinks like 1/ncol.  A
1x3 at \\textwidth (510 pt) renders an 8 pt legend at 4.2 pt -- below the legibility
check -- and compensating the font size instead makes the legend wider than a panel,
which clips it.  Stacked at \\columnwidth (246 pt) the scale is 0.756, against 0.784
for the old 1x2 at \\textwidth, so the NATURAL sizes already render as before (legend
6.05 pt vs 6.27 pt) and each panel keeps its full author width, so the full legends
and the long y-label both fit.  No compensation, no concession.

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
from matplotlib.ticker import NullLocator, MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _figdata import load
from _figstyle import PANEL_W

# DELIBERATE EXCEPTION to the paper-wide uniform panel aspect of _figstyle (3:2).
# Three stacked panels at the uniform PANEL_H=3.0 are 73% of the column height, and
# measurement (a RevTeX prd harness reading "Float too large for page by") puts the
# figure PLUS this caption 2.3 pt over the column -- 22.8 pt over once the caption
# gains its mass-ratio sentence.  At 2.4 in the figure is 59% of the column and
# places with room to spare.  Hence figdims is NOT used for the height here; the
# panels are 4.5 x 2.4 (aspect 1.88:1) rather than 4.5 x 3.0.
PANEL_H_SHORT = 2.4

# The mass-ratio wall's CHARACTER, now measured (block Q_wall_q): q* MARCHES with the
# sampled range -- 4.22, 4.80, 5.11 as q_max goes 3.0, 3.5, 4.0 -- rather than sitting
# at a fixed q, so the limiting singularity is not a real branch point pinned inside
# the box.  That is the SPIN panel's behaviour (chi* 2.11, 2.98, 3.79), not the
# separation panel's (b* pinned near merger).  The panel titles name the axis only;
# each wall's soft/hard CHARACTER is stated in the caption of Fig. 2.
Q_TITLE = "Mass-ratio wall"


def _panel(ax, curves, key, label_fmt, xlabel, title):
    """Draw one wall: data markers + geometric fit line per fit range."""
    for i, w in enumerate(curves):
        Qs = np.asarray(w["Qs"], float)
        # reserve C0-C1 (tab:blue/orange) for the paper's two models; start at C2
        (line,) = ax.semilogy(Qs, w["errs"], "o", ms=5, color=f"C{2 + i}",
                              label=label_fmt(w[key], w["rate"], w["theta_star"]))
        col = line.get_color()
        # geometric fit line over the fitted window (first n_fit points)
        Qw = np.array([Qs[0], Qs[w["n_fit"] - 1]])
        ax.plot(Qw, 10.0 ** (w["fit_slope"] * Qw + w["fit_intercept"]),
                "--", lw=1.2, color=col, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_minor_locator(NullLocator())  # no y sub-ticks
    ax.grid(True, which="major", alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(4 if max(curves[0]["Qs"]) > 12 else 2))
    ax.legend(fontsize=8, loc="lower left", handletextpad=0.2, borderpad=0.3, labelspacing=0.25)


def main():
    d = load("fig02_walls")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.0, 2.9), sharey=True)

    _panel(ax1, d["B_wall_b"], "b_min",
           lambda r, rate, th: rf"$b_{{\min}}={r}$:  $\zeta={rate:.2f}$,  $b_\ast={th:.2f}$",
           r"separation nodes  $Q_b$",
           "Separation wall")
    _panel(ax2, d["Q_wall_q"], "q_max",
           lambda r, rate, th: rf"$q_{{\max}}={r:.1f}$:  $\zeta={rate:.2f}$,  $q_\ast={th:.2f}$",
           r"mass-ratio nodes  $Q_q$",
           Q_TITLE)
    _panel(ax3, d["C_wall_spin"], "chi_max",
           lambda r, rate, th: rf"$\chi_{{\max}}={r:.1f}$:  $\zeta={rate:.2f}$,  $\chi_\ast={th:.1f}$",
           r"spin nodes  $Q_\chi$",
           "Spin wall")

    ax1.set_ylabel("held-out error")
    # all legends sit lower left; the extra decades below the float64 floor keep them off the data
    ax1.set_ylim(bottom=1e-14)
    fig.tight_layout(w_pad=0.6)
    stem = os.path.join(HERE, "fig02_walls_row")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig02_walls_row.pdf")


if __name__ == "__main__":
    main()
