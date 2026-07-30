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


# Panel titles are presentation, so they live here rather than in the figdata json:
# a notation change then needs only a re-plot, not a solver recompute.
TITLES = {
    "TL": r"4D quasi-circular model:  $\theta=(b,\ q,\ \chi^{A}_{y},\ \chi^{B}_{y})$",
    "TR": r"8D quasi-circular model:  $\theta=(b,\ q,\ \boldsymbol{\chi}^{A},\ \boldsymbol{\chi}^{B})$",
}

RANK_GID = "rank-label"


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
        xy, va, dy = ((x, hi_i), "bottom", 3) if ann == "above" else ((x, lo_i), "top", -3)
        a = ax.annotate(f"{r}", xy, fontsize=6.4, color=color, ha="center", va=va,
                        xytext=(0, dy), textcoords="offset points", zorder=5)
        a.set_gid(RANK_GID)
    for x, hi_i, lo_i, r in zip(mem, hi, lo, rs):
        _r_label(x, hi_i, lo_i, r)
    bare, bm = spec["bare"], spec["bare_mem"]
    ax.plot([mem[-1], bm], [med[-1], bare["median"]], color=color, ls="-", lw=1.7, zorder=2)
    ax.errorbar([bm], [bare["median"]],
                yerr=[[bare["median"] - bare["min"]], [bare["max"] - bare["median"]]],
                fmt="*", color=color, ms=15, elinewidth=1.0, capsize=4, capthick=1.0,
                zorder=4, label="_nolegend_")
    _r_label(bm, bare["max"], bare["min"], spec["bare_r"])


def declutter(fig, axes, pad=4.0, max_iter=8):
    """Nudge overlapping rank labels apart horizontally.

    The last swept rank sits close to the full-rank star in log-memory (0.2 decade in
    the 8D column), so their labels collide.  Both are centred on their own marker, so
    the pair is pushed apart horizontally: the right-hand one alone when the panel has
    room for it (which leaves the left one centred on its point, clear of its
    neighbour's whisker), otherwise a symmetric split.  Measured on the drawn figure,
    so it self-corrects if the rank ladder changes.
    """
    to_pt = 72.0 / fig.dpi
    fig.canvas.draw()
    for ax in axes:
        anns = [t for t in ax.texts if t.get_gid() == RANK_GID]
        for _ in range(max_iter):
            rnd = fig.canvas.get_renderer()
            frame = ax.get_window_extent(rnd)
            boxes = [a.get_window_extent(rnd) for a in anns]
            moved = False
            for i in range(len(anns)):
                for j in range(i + 1, len(anns)):
                    bi, bj = boxes[i], boxes[j]
                    if not bi.overlaps(bj):
                        continue
                    ov = min(bi.x1, bj.x1) - max(bi.x0, bj.x0) + pad
                    if ov <= 0:      # they overlap only vertically; leave them be
                        continue
                    lo, hi = (i, j) if bi.x0 <= bj.x0 else (j, i)
                    shifts = (((hi, ov),) if boxes[hi].x1 + ov <= frame.x1 - pad
                              else ((lo, -0.5 * ov), (hi, 0.5 * ov)))
                    for k, dpx in shifts:
                        dx, dy = anns[k].get_position()
                        anns[k].set_position((dx + dpx * to_pt, dy))
                    moved = True
            if not moved:
                break
            fig.canvas.draw()


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
        if TITLES.get(key):
            ax.set_title(TITLES[key], fontsize=10)
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
    declutter(fig, axes.ravel())
    stem = os.path.join(HERE, "fig05_guess_vs_memory")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig05_guess_vs_memory.pdf")


if __name__ == "__main__":
    main()
