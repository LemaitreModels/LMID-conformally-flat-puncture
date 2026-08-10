#!/usr/bin/env python
"""Generates fig06_targeting.pdf (paper Fig. 6).

Certified parameter targeting: target residual vs cumulative certified elliptic
solves over a set of random known-answer targets, as a median line with min--max
whiskers across targets (the same distribution style as Figs. 1 and 5), for the
black-box (cold) control loop vs the differentiable parametric model.

Reads ONLY figdata/fig06_targeting.json (build it with fig06_targeting_data.py,
which reduces the per-target run log to the whisker curves). No reports/, no jax.

Run:  python fig06_targeting_plot.py
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

# grey black-box control loop vs tab:green differentiable parametric model (C0/C1
# are reserved for the paper's two models, so the applications figures start at C2)
C = {"cold": "0.6", "gradient": "tab:green"}
LB = {"cold": "black-box model", "gradient": "differentiable parametric model"}
PLOT = ("cold", "gradient")
OFFS = {"cold": -0.08, "gradient": +0.08}   # nudge apart so whiskers don't overlap
TOL = 1e-8


def main():
    d = load("fig06_targeting")

    fig, ax = plt.subplots(1, 1, figsize=figdims(1, 1))
    for m in PLOT:
        mm = d["methods"][m]
        x = np.asarray(mm["x"], float)
        med = np.asarray(mm["med"], float)
        lo = np.asarray(mm["lo"], float)
        hi = np.asarray(mm["hi"], float)
        ax.errorbar(x + OFFS[m], med, yerr=[med - lo, hi - med], fmt="o-",
                    color=C[m], ms=5, lw=1.7, capsize=3.5, elinewidth=1.1,
                    capthick=1.1, zorder=5, label=LB[m])
    ax.set_yscale("log")
    ax.axhline(TOL, color="grey", ls=":", lw=1)
    ax.set_xlabel("certified elliptic solves")
    ax.set_ylabel(r"target residual $\|F-F_\star\|_\infty$")
    ax.set_title("Convergence to target", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    # Framed legend with the matplotlib-default semi-transparent white background (as in
    # Fig. 3).  Upper right, not the historical lower right: on the fixed-budget run the
    # black box keeps iterating past its own stopping criterion and its median falls to
    # the M_ADM read's noise floor, so the lower-right corner now carries data.  The
    # upper right is empty once both curves have descended.
    # The errorbars carry zorder=5, which ties the legend's default, so the markers
    # draw over the frame; lift the legend above them.
    ax.legend(fontsize=8, loc="upper right").set_zorder(10)
    fig.tight_layout()
    stem = os.path.join(HERE, "fig06_targeting")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig06_targeting.pdf")


if __name__ == "__main__":
    main()
