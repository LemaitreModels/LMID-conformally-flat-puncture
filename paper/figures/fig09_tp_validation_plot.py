#!/usr/bin/env python
"""Generates fig09_tp_validation.pdf (paper Fig. 9).

Quasi-circular TwoPunctures validation, a SINGLE-COLUMN 2x1 figure whose panels
share the one x axis (the LM-initial-data grid sequence):
 (a) TOP     ||psi - psi_TP||_inf       -- the pointwise field comparison;
 (b) BOTTOM  |M_ADM - E_TP|/E_TP        -- the integral (ADM mass) comparison.

Paper side: this is a `figure` at width=\\columnwidth, NOT a `figure*`.

A former panel (a), the ADM-J tilt against the spin tilt, was DROPPED: measured,
theta_J equals theta_S to ~1e-14 deg for every |S| and every TwoPunctures anchor,
so the panel was three mutually coincident curves on the line y=x.  That identity
and its residual (<=1.4e-14) are stated quantitatively in the appendix text
instead.  The `panelA`/`anchors` blocks are still produced by the data script and
are simply unused here.

Reads ONLY figdata/fig09_tp_validation.json (build it with
fig09_tp_validation_data.py). No reports/, no jax.

Run:  python fig09_tp_validation_plot.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _figdata import load
from _figstyle import figdims, PANEL_H_STACK


class _LogMinorSkip(LogFormatterSciNotation):
    """LogFormatterSciNotation with a blacklist of tick values left unlabelled.

    Subclassed rather than wrapped in a FuncFormatter so that `set_locs` still
    reaches the base class, which is what selects *which* minor ticks get a
    label at all (wrapping labels every minor tick and the axis becomes a jumble).
    """

    def __init__(self, skip=(), **kw):
        super().__init__(**kw)
        self._skip = tuple(skip)

    def __call__(self, x, pos=None):
        if any(abs(x / s - 1.0) < 1e-6 for s in self._skip):
            return ""
        return super().__call__(x, pos)


def main():
    d = load("fig09_tp_validation")
    C = d["C_psi_adm"]
    ns = [f"{r['Na']}$\\times${r['Nb']}" for r in C]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figdims(2, 1, panel_h=PANEL_H_STACK),
                                   sharex=True)

    # (a) quasi-circular psi vs TwoPunctures
    # reserve C0-C1 (tab:blue/orange) for the paper's two models; start at C2
    ax1.semilogy(range(len(C)), [r["max_dpsi"] for r in C], "o-", color="C2")
    ax1.set_ylabel(r"$\|\psi-\psi_{\rm TP}\|_\infty$")
    ax1.set_title(r"Quasi-circular field vs TP", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    # this panel spans barely one decade, so the lone labelled decade leaves the
    # scale unreadable: label the minor ticks too (panel (b) spans four and needs none).
    # 3e-9 is dropped: its label crowds the 2e-9 and 4e-9 ones on either side.
    ax1.yaxis.set_minor_formatter(
        _LogMinorSkip(skip=(3e-9,), minor_thresholds=(2.0, 0.6)))
    # 8 pt here renders at ~6.3 pt after LaTeX's 0.78 downscale, matching the
    # smallest text already accepted elsewhere in the paper (fig02's 6.05 pt legend)
    ax1.tick_params(axis="y", which="minor", labelsize=8)

    # (b) quasi-circular ADM mass vs TwoPunctures -- shares the grid axis with (a)
    ax2.semilogy(range(len(C)), [r["M_ADM_rel_diff"] for r in C], "o-", color="C2")
    ax2.set_ylabel(r"$|M_{\rm ADM}-E_{\rm TP}|/E_{\rm TP}$")
    ax2.set_title(r"Quasi-circular ADM mass vs TP", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    # one shared x axis: ticks set on the shared axis, labelled only on the bottom
    ax2.set_xticks(range(len(C)))
    ax2.set_xticklabels(ns, rotation=30)
    ax2.set_xlabel(r"grid $N_A\times N_B$ ($N_\phi=8$)")
    ax1.tick_params(labelbottom=False)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig09_tp_validation")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig09_tp_validation.pdf")


if __name__ == "__main__":
    main()
