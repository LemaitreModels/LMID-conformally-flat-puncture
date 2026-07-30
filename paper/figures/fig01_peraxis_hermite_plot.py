#!/usr/bin/env python
"""Generates fig01_peraxis_hermite.pdf (paper Fig. 1).

Per-axis held-out interpolation error, value vs value+gradient (Hermite), as a
2x4 grid of panels (shared y-range) over the eight quasi-circular axes.  Each curve
is a DISTRIBUTION over the paper's random base points: the median (marker)
carries a fitted geometric rate (dec/Q, shown in the legend), and the min-max
whiskers span the best-to-worst held-out error across those base points (cf. Fig. 3).

Colours match Fig. 3/5: value = C0 (tab:blue), gradient-enhanced = C1 (tab:orange).

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

# (figdata key, colour, marker, label stem) — both drawn as solid lines
SERIES = (("value",   "C0", "o", "value"),
          ("hermite", "C1", "s", "value+gradient"))


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
    for k, name in enumerate(names):
        ax = axs[k // ncol][k % ncol]
        d = A[name]
        Qs = d["Qs"]
        ax.set_yscale("log")
        for key, color, mk, lab in SERIES:
            _series(ax, Qs, d[key], color, mk,
                    rf"{lab}  ($r={d['rate_' + key]:.2f}$ dec/$Q$)")
        ax.set_title(LBL.get(name, name), fontsize=18)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(*YLIM)
        ax.set_xticks(Qs)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12, loc="upper right")
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
