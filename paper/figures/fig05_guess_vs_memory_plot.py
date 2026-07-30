#!/usr/bin/env python
"""Generates fig05_guess_vs_memory.pdf (paper Fig. 5).

Reduced-basis (POD) compression versus stored model memory, as the POD truncation rank r is swept.
A 2x2 grid with a SHARED memory x-axis per column:
  TOP row    bare-guess constraint residual ||R||_inf vs memory
  BOTTOM row bare-guess field error ||u-u_true||_2/||u_true||_2 vs memory
  LEFT col 4D,  RIGHT col 8D;  each panel: value (C0) + value+gradient [cross] (C1).

Median line with min--max whiskers; each point labelled by r; the full-rank (un-compressed) model
is the star. The field row (bottom) saturates at its floor while the residual (top) keeps falling,
so compression discards exactly the high-derivative content the residual amplifies and the field
norm cannot see.

Reads ONLY figdata/fig05_guess_vs_memory.json (build it with fig05_guess_vs_memory_data.py).
No reports/, no jax.

Run:  python fig05_guess_vs_memory_plot.py
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


def sweep_xy(cur):
    mem = np.array([c["mem_bytes"] for c in cur]) / 1e6
    med = np.array([c["median"] for c in cur])
    lo = np.array([c["min"] for c in cur])
    hi = np.array([c["max"] for c in cur])
    return mem, med, lo, hi, [c["r"] for c in cur]


def plot_family(ax, spec):
    cur = spec["cur"][:-1] if spec["drop_last"] else spec["cur"]
    color, ann = spec["color"], spec["ann"]
    mem, med, lo, hi, rs = sweep_xy(cur)
    ax.errorbar(mem, med, yerr=[med - lo, hi - med], fmt=spec["marker"] + "-", color=color,
                ms=5, lw=1.7, elinewidth=1.0, capsize=3, capthick=1.0, zorder=3,
                label=spec["label"])

    def _r_label(x, hi_i, lo_i, r):
        if ann == "above":
            ax.annotate(f"{r}", (x, hi_i), fontsize=6.4, color=color, ha="center",
                        va="bottom", xytext=(0, 3), textcoords="offset points", zorder=5)
        else:
            ax.annotate(f"{r}", (x, lo_i), fontsize=6.4, color=color, ha="center",
                        va="top", xytext=(0, -3), textcoords="offset points", zorder=5)
    for x, hi_i, lo_i, r in zip(mem, hi, lo, rs):
        _r_label(x, hi_i, lo_i, r)
    bare, bm = spec["bare"], spec["bare_mem"]
    ax.plot([mem[-1], bm], [med[-1], bare["median"]], color=color, ls="-", lw=1.7, zorder=2)
    ax.errorbar([bm], [bare["median"]],
                yerr=[[bare["median"] - bare["min"]], [bare["max"] - bare["median"]]],
                fmt="*", color=color, ms=15, elinewidth=1.0, capsize=4, capthick=1.0,
                zorder=4, label="_nolegend_")
    _r_label(bm, bare["max"], bare["min"], spec["bare_r"])


def _note(ax, text, loc):
    xy = {"bl": (0.03, 0.03, "left", "bottom"), "br": (0.97, 0.03, "right", "bottom"),
          "tr": (0.97, 0.97, "right", "top"), "tl": (0.03, 0.97, "left", "top")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, fontsize=7.5,
            color="0.35", ha=xy[2], va=xy[3])


def main():
    P = load("fig05_guess_vs_memory")["panels"]
    fig, axes = plt.subplots(2, 2, figsize=figdims(2, 2), sharex="col", sharey="row")
    for key, ax in (("TL", axes[0, 0]), ("TR", axes[0, 1]),
                    ("BL", axes[1, 0]), ("BR", axes[1, 1])):
        pan = P[key]
        for spec in pan["curves"]:
            plot_family(ax, spec)
        if pan.get("title"):
            ax.set_title(pan["title"], fontsize=10)
        if pan.get("note"):
            _note(ax, pan["note"], pan["note_loc"])

    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.3)
        ax.minorticks_off()
        ax.legend(fontsize=9, loc="lower left", framealpha=0.92)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlabel("stored model memory (MB)")
    axes[0, 0].set_ylabel(r"bare-guess constraint residual $\|R\|_\infty$")
    axes[1, 0].set_ylabel(r"bare-guess field error $\|u-u_{\rm true}\|_2/\|u_{\rm true}\|_2$")
    fig.tight_layout()
    stem = os.path.join(HERE, "fig05_guess_vs_memory")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig05_guess_vs_memory.pdf")


if __name__ == "__main__":
    main()
