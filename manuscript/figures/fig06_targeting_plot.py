#!/usr/bin/env python
"""Generates fig06_targeting.pdf (manuscript Fig. 6).

Certified parameter targeting: target residual vs cumulative certified elliptic
solves over a set of random known-answer targets, as a median line with min--max
whiskers across targets (the same distribution style as Figs. 1 and 5), for the
black-box (cold) control loop vs the differentiable surrogate.

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

# grey black-box control loop vs default-tab:blue differentiable surrogate
C = {"cold": "0.6", "gradient": "tab:blue"}
LB = {"cold": "black-box", "gradient": "differentiable surrogate"}
PLOT = ("cold", "gradient")
OFFS = {"cold": -0.08, "gradient": +0.08}   # nudge apart so whiskers don't overlap
TOL = 1e-8


def main():
    d = load("fig06_targeting")
    n = int(d["n"])

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
    ax.text(ax.get_xlim()[1], TOL, " tol", color="grey", va="bottom", ha="right",
            fontsize=8)
    ax.set_xlabel("certified elliptic solves")
    ax.set_ylabel(r"target residual $\|F-F_\star\|_\infty$")
    ax.set_title(f"convergence over {n} random targets", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    stem = os.path.join(HERE, "fig06_targeting")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig06_targeting.pdf")


if __name__ == "__main__":
    main()
