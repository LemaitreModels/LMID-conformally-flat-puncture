#!/usr/bin/env python
"""Generates fig07_eccentricity.pdf (paper Fig. 7).

Cook effective-potential eccentricity control: the field-dependent binding
energy E_b(b) at each fixed J (the circular-orbit sequence), with the classical
certified scan (squares), the smooth surrogate curve (line), and the
differentiable gradient minimum (star).

Reads ONLY figdata/fig07_eccentricity.json (build it with fig07_eccentricity_data.py,
which does the one-time jax/model precompute of the smooth curves). No reports/, no
jax, no model here.

Run:  python fig07_eccentricity_plot.py
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
    d = load("fig07_eccentricity")
    Jlist = [float(J) for J in d["Jlist"]]
    n_scan, n_grad = int(d["n_scan"]), int(d["n_grad"])
    bg = np.asarray(d["bg"], float)

    # reserve tab:blue/orange for the paper's two models; start after the first 2
    PALETTE = ["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    cols = {J: PALETTE[i % len(PALETTE)] for i, J in enumerate(Jlist)}

    fig, ax = plt.subplots(figsize=figdims(1, 1))
    lo, hi = np.inf, -np.inf
    curves = []
    for J in Jlist:
        c = cols[J]
        pj = d["per_J"][f"{J:.2f}"]
        Vg = 1e3 * np.asarray(pj["Vg"], float)
        lo, hi = min(lo, Vg.min()), max(hi, Vg.max())
        curves += ax.plot(bg, Vg, "-", color=c, lw=1.9, label=f"$J={J:.2f}$")
        ax.plot(np.asarray(pj["scan_b"], float), 1e3 * np.asarray(pj["scan_Eb"], float),
                "s", color=c, ms=3.8, alpha=0.85, zorder=3)
        ax.plot([pj["b_circ"]], [1e3 * float(pj["Vc"])], "*", color=c, ms=16, zorder=6)
    star = ax.plot([], [], "k*", ms=11, label=f"gradient minimum ({n_grad} solves)")
    square = ax.plot([], [], "ks", ms=6, label=f"classical scan ({n_scan} solves each)")
    ax.set_xlabel(r"separation $b$  [$M$]")
    ax.set_ylabel(r"binding energy $E_b \times 10^{3}$")
    # fontsize 10 matches Fig. 6, the figure with identical geometry (figdims(1,1) at
    # \columnwidth); the matplotlib default 'large' (12 pt) rendered visibly bigger.
    ax.set_title("Accelerated eccentricity reduction", fontsize=10)
    # Legend on the right: two columns, the J branches on the left and the two marker
    # keys (scan, then gradient) on the right, matplotlib filling column-major.  The
    # upper-right corner is the only region free of curves -- every E_b(b) branch lies
    # below the J=1.10 rise -- but it is shallower than the legend, hence the explicit
    # top headroom below.
    # The window is shifted up relative to the data: the J=1.00 star sits at the very
    # bottom of the range and needs clearance for its marker, while the top only has to
    # clear the legend.
    span = hi - lo
    ax.set_ylim(lo - 0.13 * span, hi + 0.26 * span)
    ax.legend(handles=curves + square + star,
              fontsize=8, frameon=False, ncol=2, loc="upper right",
              handletextpad=0.5, labelspacing=0.35, columnspacing=1.2,
              borderaxespad=0.4)
    fig.tight_layout()
    stem = os.path.join(HERE, "fig07_eccentricity")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig07_eccentricity.pdf")


if __name__ == "__main__":
    main()
