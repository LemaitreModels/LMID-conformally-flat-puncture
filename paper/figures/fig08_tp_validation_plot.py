#!/usr/bin/env python
"""Generates fig08_tp_validation.pdf (the TwoPunctures resolution ladder).

A 3x1 figure at \\columnwidth, the three panels SHARING the resolution abscissa, from top
to bottom:

 * the certified residual ||R||_inf (rising, against the certification gate),
 * the pointwise field agreement ||psi - psi_TP||_inf (falling),
 * the integral agreement |M_ADM - E_TP|/E_TP (falling).

The first two run in opposite directions, and the shared abscissa is what makes that
comparison: the residual's rise is roundoff amplification in unpopulated high-m azimuthal
modes, not a loss of convergence.

PANELS ARE NOT LETTERED.  Each carries a descriptive title and a symbol y label, which
identifies it without an (a)/(b)/(c) tag; the caption and the text name the quantity rather
than a letter, so nothing in the paper depends on the panel order.

Every point is a median over configurations drawn from the production box, with min--max
whiskers, so no panel rests on one arbitrary parameter point; the ladder (x axis) is what
still shows the disagreement to be limited by spatial resolution rather than by the
nonlinear solve.

WHY 3x1 SINGLE COLUMN.  This began as a 1x3 ``figure*`` in which (a) and (b) were crammed
into ONE panel ~2.3 in wide, spanning nine decades and needing a legend and two reference
lines to be readable.  One quantity per panel on a shared abscissa gives each series its own
2-4 decade y range, buys ~46% panel width, prints the five rotated grid labels once instead
of three times, needs no legend at all (the title and y label name the series), and places
as an ordinary single-column float instead of a top-of-page-only ``figure*``.  The azimuthal
spectrum, which has a different abscissa and a different question, is its own figure
(fig09_tp_spectrum).

COLOURS/LINES follow the paper convention (see README): one series per panel, so all three
are the first non-model colour C2, solid.

Reads ONLY figdata/fig08_tp_validation.json (build it with fig08_tp_validation_data.py).
No reports/, no jax.

Run:  python fig08_tp_validation_plot.py
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
from _figstyle import figdims, PANEL_H_STACK

SERIES_C = "C2"        # non-model diagnostic: the cycle starts at C2 (see README)

# panel key -> (title, y label).  The order is the panel order: the certification first, then
# the two comparisons against the oracle, so the two FALLING agreement metrics sit together
# below the RISING residual.
PANELS = (
    ("residual", "Certified residual", r"$\|R\|_\infty$"),
    ("psi",      "Field vs TP",        r"$\|\psi-\psi_{\rm TP}\|_\infty$"),
    ("M_ADM",    "ADM mass vs TP",     r"$|M_{\rm ADM}-E_{\rm TP}|/E_{\rm TP}$"),
)


def _gridlabels(ladder):
    """Meridional labels only; Nphi is fixed along the ladder and goes in the axis label."""
    return [rf"${g[0]}\times{g[1]}$" for g in [r["grid"] for r in ladder]]


def _whisker(ax, x, rows, key):
    """Median marker with min--max whiskers (the paper's distribution idiom, cf. Figs. 1, 3-6)."""
    lo = np.array([r[key]["min"] for r in rows])
    md = np.array([r[key]["median"] for r in rows])
    hi = np.array([r[key]["max"] for r in rows])
    ax.errorbar(x, md, yerr=[md - lo, hi - md], fmt="-o", color=SERIES_C,
                ms=5, lw=1.7, capsize=3.5, elinewidth=1.1, capthick=1.1)


def main():
    d = load("fig08_tp_validation")
    L, meta = d["ladder"], d["meta"]
    x = np.arange(len(L))

    fig, axes = plt.subplots(len(PANELS), 1, figsize=figdims(len(PANELS), 1,
                                                             panel_h=PANEL_H_STACK),
                             sharex=True)

    for ax, (key, title, ylab) in zip(axes, PANELS):
        _whisker(ax, x, L, key)
        ax.set_yscale("log")
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_minor_locator(NullLocator())  # no y sub-ticks (matches Figs. 2, 3)
        ax.grid(True, which="major", alpha=0.3)

    # the certification gate belongs to the residual panel only; label it at the LEFT and
    # tucked just under the line (va="top" puts the text block's top at y), where the residual
    # is smallest, so it clears the rising whiskers
    ax_res = axes[0]
    ax_res.axhline(meta["cert_tol"], ls=":", color="0.35", lw=1.1)
    ax_res.text(0.02, meta["cert_tol"] / 1.08, "certification gate", fontsize=8.5,
                color="0.3", va="top", transform=ax_res.get_yaxis_transform())

    # shared abscissa: the rotated grid labels are drawn once, on the bottom panel
    nphi = sorted({r["grid"][2] for r in L})
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(_gridlabels(L), fontsize=8, rotation=20)
    axes[-1].set_xlabel(rf"grid $N_A\times N_B$ ($N_\phi={nphi[0]}$)" if len(nphi) == 1
                        else r"grid $N_A\times N_B\times N_\phi$")

    fig.tight_layout()
    stem = os.path.join(HERE, "fig08_tp_validation")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig08_tp_validation.pdf")


if __name__ == "__main__":
    main()
