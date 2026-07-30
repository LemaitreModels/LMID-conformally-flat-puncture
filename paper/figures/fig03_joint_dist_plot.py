#!/usr/bin/env python
"""Generates fig03_joint_dist.pdf (paper Fig. 3).

Joint held-out error distribution (best / median / worst over 1000 random off-node
points) vs solver node count, across Smolyak levels l=1..5. Two shared-y panels:
  left   -- the 4D model: BARE sparse interpolant vs value + gradient + cross
            (full-bilinear Hermite-Smolyak) enhanced on the two spin axes
  right  -- the 8D general-spin model, same two curves (mirrors the left panel)

Colours match Fig. 5: value = C0, gradient-enhanced = C1.

Reads ONLY figdata/fig03_joint_dist.json (build it with fig03_joint_dist_data.py).
No reports/, no jax.

Run:  python fig03_joint_dist_plot.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _figdata import load
from _figstyle import figdims

BARE_C, CROSS_C = "C0", "C1"          # value / gradient-enhanced (matches Fig. 5)


def _arr(s):
    return {k: np.asarray(s[k], float) for k in ("nodes", "best", "med", "worst")}


def _series(ax, s, color, marker, label):
    # median with min–max whiskers (whiskers span best..worst of the distribution)
    ax.errorbar(s["nodes"], s["med"], yerr=[s["med"] - s["best"], s["worst"] - s["med"]],
                fmt=marker + "-", color=color, ms=5, lw=1.7, elinewidth=1.0,
                capsize=4, capthick=1.0, zorder=3, label=label)


def _levels(ax, bare, cross, levels):
    # one Smolyak-level label per node count, above the higher of the two whiskers
    top = np.maximum(bare["worst"], cross["worst"])
    for x, w, lev in zip(bare["nodes"], top, levels):
        dx = 0.5 if lev == 2 else 0         # nudge l=2 clear of the l=1 point
        ha = "left" if lev == 2 else "center"
        ax.annotate(rf"$\ell={lev}$", (x, w), fontsize=8, color="0.3",
                    ha=ha, va="bottom", xytext=(dx, 4),
                    textcoords="offset points", zorder=5)


def _panel(ax, side, title):
    bare, cross = _arr(side["bare"]), _arr(side["cross"])
    _series(ax, bare, BARE_C, "o", "bare interpolant")
    _series(ax, cross, CROSS_C, "s", r"value + gradient + cross ($\chi_{Ay},\chi_{By}$)")
    _levels(ax, bare, cross, side["bare"]["levels"])
    ax.set_xlabel("solver node count")
    ax.set_title(title, fontsize=10)
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.3)
    return bare, cross


def main():
    d = load("fig03_joint_dist")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=figdims(1, 2), sharey=True)
    axL.set_yscale("log")

    bL, cL = _panel(axL, d["left"], "four-dimensional model")
    bR, cR = _panel(axR, d["right"], "eight-dimensional model")
    axL.set_ylabel(f"joint held-out error over {d['left']['bare']['n_points']} points")
    axL.legend(fontsize=9, loc="lower left")

    # common y-limits spanning both panels, with headroom for the level labels
    lo = min(bL["best"].min(), cL["best"].min(), bR["best"].min(), cR["best"].min())
    hi = max(bL["worst"].max(), cL["worst"].max(), bR["worst"].max(), cR["worst"].max())
    axL.set_ylim(lo * 0.5, hi * 4.0)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig03_joint_dist")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig03_joint_dist.pdf")


if __name__ == "__main__":
    main()
