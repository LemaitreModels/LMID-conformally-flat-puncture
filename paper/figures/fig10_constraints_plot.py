#!/usr/bin/env python
"""Generates fig10_constraints.pdf (paper Fig. 10).

Constraint violation of the initial data on a uniform Cartesian evolution grid: the bulk
L2-RMS of the Hamiltonian and momentum constraints, measured by a generic second-order
finite-difference monitor, against the Cartesian grid spacing h.  Both fall at the monitor's
second order, and the TwoPunctures-sourced points (black, open) sit on the same curves — the
measured violation is the monitor's truncation error, not a property of either initial-data
solution.

COLOURS follow the paper convention (see README): two non-model series, so C2 (Hamiltonian)
and C3 (momentum); black open markers for the external oracle; dotted grey for the h^2
reference slopes, which are guides rather than measurements.

Reads ONLY figdata/fig10_constraints.json (build it with fig10_constraints_data.py, the one
data script that recomputes rather than distilling a reports/ artifact).  No solver, no
oracle, no jax here.

Run:  python fig10_constraints_plot.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _figdata import load
from _figstyle import figdims

# key suffix -> (colour, marker, legend label)
SERIES = (
    ("H", "C2", "o", r"Hamiltonian $\|\mathcal{H}\|_2$"),
    ("M", "C3", "s", r"momentum $\|\mathcal{M}\|_2$"),
)


def main():
    d = load("fig10_constraints")
    c, meta = d["curves"], d["meta"]
    h = np.asarray(c["h"], float)

    fig, ax = plt.subplots(figsize=figdims(1, 1))
    handles = []
    for key, col, mk, lab in SERIES:
        e = np.asarray(c[f"{key}_lm"], float)
        p = float(c[f"order_{key}_lm"])
        handles += ax.loglog(h, e, "-", marker=mk, color=col, ms=5, lw=1.7,
                             label=rf"{lab}  ($h^{{{p:.2f}}}$)")
        # Second-order guide anchored at the FINEST rung: both series converge slightly
        # faster than h^2, so anchoring at the coarse end swings the guide away from the
        # data at small h and through the legend; anchoring at the fine end keeps it
        # tucked against the curve across the whole ladder.
        ax.loglog(h, e[-1] * (h / h[-1]) ** 2, ls=":", color="0.55", lw=1.1, zorder=1)
        if meta["has_tp"]:
            ax.loglog(h, np.asarray(c[f"{key}_tp"], float), ls="none", marker=mk,
                      mfc="none", mec="k", ms=9, mew=1.0, zorder=5)

    ref = ax.plot([], [], ls=":", color="0.55", lw=1.1, label=r"$\propto h^{2}$")
    oracle = ([ax.plot([], [], ls="none", marker="o", mfc="none", mec="k", ms=8,
                       label="TwoPunctures-sourced")[0]] if meta["has_tp"] else [])

    ax.set_xlabel(r"Cartesian grid spacing $h$  [$M$]")
    ax.set_ylabel(r"constraint violation (bulk $L_2$)")
    ax.set_title("Constraints on an evolution grid", fontsize=10)
    # The ladder spans well under a decade in h, where matplotlib's LogLocator labels the
    # sub-decade minors (2x10^-1, 3x10^-1, ...).  Tick the rungs instead: they are the
    # abscissa, and a plain decimal reads at a glance.  Past five rungs the labels collide
    # at the fine end (h is geometric, so they crowd there), so label every other one.
    step = 1 if len(h) <= 5 else 2
    ax.set_xticks(h[::step])
    ax.set_xticklabels([f"{v:.3g}" for v in h[::step]])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())      # no y sub-ticks (matches Figs. 2, 3, 8, 9)
    ax.grid(True, which="major", alpha=0.3)
    # upper left: the only corner free of curves (both series fall to the left)
    ax.legend(handles=handles + oracle + ref, fontsize=8, frameon=False,
              loc="upper left", handletextpad=0.6, labelspacing=0.35, borderaxespad=0.4)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig10_constraints")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig10_constraints.pdf")


if __name__ == "__main__":
    main()
