#!/usr/bin/env python
"""Generates fig01_peraxis_hermite.pdf (paper Fig. 1).

Per-axis held-out interpolation error, value-only vs gradient-enhanced (Hermite), as a
2x4 grid of panels (shared y-range) over the eight quasi-circular axes.  Each curve
is a DISTRIBUTION over the paper's random base points: the median (marker)
carries a fitted geometric rate (dec/Q, shown in each panel's upper-right legend),
and the min-max whiskers span the best-to-worst held-out error across those base
points (cf. Fig. 3).  The two series are named once, in a second legend in the
lower left of the bottom row's first panel, so the per-panel legends stay compact.

Colours match Fig. 3/5: value-only = C0 (tab:blue), gradient-enhanced = C1 (tab:orange).

Reads ONLY figdata/fig01_peraxis_hermite.json (build it with
fig01_peraxis_hermite_data.py). No reports/, no jax.

Run:  python fig01_peraxis_hermite_plot.py
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

LBL = {"b": r"$b$", "q": r"$q$",
       "chi_Ax": r"$\chi^{A}_{x}$", "chi_Ay": r"$\chi^{A}_{y}$", "chi_Az": r"$\chi^{A}_{z}$",
       "chi_Bx": r"$\chi^{B}_{x}$", "chi_By": r"$\chi^{B}_{y}$", "chi_Bz": r"$\chi^{B}_{z}$"}
YLIM = (1e-12, 1e-2)

# (figdata key, colour, marker, label stem) — both drawn as solid lines.
# A per-axis sweep is one-dimensional, so the enhanced curve carries the value and the
# single first tangent along that axis: there is no mixed cross tangent to match here
# (unlike the joint models of Figs. 3-5, which are the full-bilinear gradient-enhanced
# family).  The caption states that difference; the label stays the concept name so that
# one model has one name in every figure.
SERIES = (("value",   "C0", "o", "value-only"),
          ("hermite", "C1", "s", "gradient-enhanced"))


def _arr(vals):
    return np.array([np.nan if v is None else float(v) for v in vals], float)


def _series(ax, Qs, st, color, marker, label):
    """Median (marker) with min-max whiskers spanning best..worst (cf. Fig. 3)."""
    med, best, worst = _arr(st["median"]), _arr(st["best"]), _arr(st["worst"])
    ax.errorbar(Qs, med, yerr=[med - best, worst - med],
                fmt=marker + "-", color=color, ms=5, lw=1.7, elinewidth=1.0,
                capsize=4, capthick=1.0, zorder=3, label=label)


def main():
    A = load("fig01_peraxis_hermite")["A_per_axis"]
    names = [n for n in A]

    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=figdims(nrow, ncol),
                            squeeze=False, sharex=True, sharey=True)
    k_names = ncol * (nrow - 1)  # panel that carries the series-name legend
    for k, name in enumerate(names):
        ax = axs[k // ncol][k % ncol]
        d = A[name]
        Qs = d["Qs"]
        ax.set_yscale("log")
        for key, color, mk, lab in SERIES:
            _series(ax, Qs, d[key], color, mk,
                    rf"$r={d['rate_' + key]:.2f}$ dec/$Q$")
        ax.set_title(LBL.get(name, name), fontsize=18)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(*YLIM)
        ax.set_xticks(Qs)
        ax.tick_params(labelsize=14)
        handles, _ = ax.get_legend_handles_labels()  # errorbar containers (with whiskers)
        rates = ax.legend(fontsize=12, loc="upper right")
        # the two series are named once, in the first panel of the bottom row, so that
        # the per-panel legends carry only the fitted rates.  That panel's lower-left
        # corner is the one that is free of data in every axis (the b panel's is not).
        # Same fontsize and the same errorbar handles as the rate legends; the handle
        # box and padding are trimmed so that this wider box still ends left of the
        # Q=8 whisker.
        if k == k_names:
            ax.add_artist(rates)
            ax.legend(handles, [lab for *_, lab in SERIES],
                      fontsize=12, loc="lower left", handlelength=1.1,
                      borderpad=0.25, labelspacing=0.3, handletextpad=0.35,
                      borderaxespad=0.3)
        # shared x: label only the bottom-most populated panel in each column
        if k + ncol >= len(names):
            ax.set_xlabel("parameter nodes  $Q$", fontsize=17)
    for k in range(len(names), nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    # shared y: label only the left column
    for r in range(nrow):
        axs[r][0].set_ylabel("held-out error", fontsize=17)
    fig.tight_layout()
    stem = os.path.join(HERE, "fig01_peraxis_hermite")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig01_peraxis_hermite.pdf")


if __name__ == "__main__":
    main()
