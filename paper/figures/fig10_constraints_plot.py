#!/usr/bin/env python
"""Generates fig10_constraints.pdf (paper Fig. 10).

Constraint violation of the initial data on an evolution grid, measured by
GRTeclyn's own fourth-order constraint operator working from the evolution
variables.  One panel: the App. A anchor.  Hamiltonian and momentum norms fall
at fourth order, and the TwoPunctures-sourced points (black, open) sit on the
same curves — through the identical initial-data class, interpolation, stencils
and grid, so the two initial-data solutions are indistinguishable to an
independent code over most of the ladder.

A second panel comparing against GRTeclyn's analytic Bowen-York data at P = 0.1
was dropped as confusing: it measured a different configuration from the anchor
(that data requires |P| < 0.3 m) and its momentum curves coincided by
construction.  The P = 0.1 series stay in the figdata (the producer is
unchanged) and are still printed below, so the comparison remains checkable
without occupying a panel.

COLOURS follow the paper convention (see README): C2 for the Hamiltonian, C3 for
the momentum, black open markers for the external oracle, dotted grey for the
h^4 reference slopes (guides, not measurements).

Reads ONLY figdata/fig10_constraints.json.  No solver, no oracle, no jax here.

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

H_COL, M_COL = "C2", "C3"


def _ticks(ax, h):
    # The ladder spans well under a decade in h, where matplotlib's LogLocator
    # labels sub-decade minors.  Tick the rungs instead — they are the abscissa.
    step = 1 if len(h) <= 5 else 2
    ax.set_xticks(h[::step])
    ax.set_xticklabels([f"{v:.3g}" for v in h[::step]])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", alpha=0.3)


def _guide(ax, h, e, p=4):
    """h^p guide anchored at the finest rung, so it stays tucked against the
    data instead of swinging away at small h."""
    ax.loglog(h, e[-1] * (h / h[-1]) ** p, ls=":", color="0.55", lw=1.1,
              zorder=1)


def main():
    d = load("fig10_constraints")
    c, meta = d["curves"], d["meta"]
    amr, variants = d.get("amr") or [], d.get("variants") or {}
    h = np.asarray(c["h"], float)

    fig, axa = plt.subplots(1, 1, figsize=figdims(1, 1))

    # ---------------- the anchor + the parity overlay ----------------------
    handles = []
    for key, col, mk, lab in (("Ham", H_COL, "o", r"$\|\mathcal{H}\|_2$"),
                              ("Mom", M_COL, "s", r"$\|\mathcal{M}\|_2$")):
        e = np.asarray(c[f"L2_{key}_lm"], float)
        handles += axa.loglog(h, e, "-", marker=mk, color=col, ms=5, lw=1.7,
                              label=lab)
        _guide(axa, h, e)
        if meta["has_tp"]:
            axa.loglog(h, np.asarray(c[f"L2_{key}_tp"], float), ls="none",
                       marker=mk, mfc="none", mec="k", ms=9, mew=1.0, zorder=5)
    extra = [axa.plot([], [], ls=":", color="0.55", lw=1.1,
                      label=r"$\propto h^{4}$")[0]]
    if meta["has_tp"]:
        extra.append(axa.plot([], [], ls="none", marker="o", mfc="none",
                              mec="k", ms=8, label="TwoPunctures")[0])
    axa.set_title(r"Constraint violation in GRTeclyn", fontsize=10)
    axa.set_ylabel(r"constraint violation")
    axa.set_xlabel(r"grid spacing $h$  [$M$]")
    _ticks(axa, h)
    axa.legend(handles=handles + extra, fontsize=8, frameon=False,
               loc="lower right", handletextpad=0.6, labelspacing=0.35)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig10_constraints")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)

    # Numbers the caption quotes, printed so they are checkable against the PDF.
    print("wrote fig10_constraints.pdf")
    print(f"  orders  H {['%.2f' % p for p in c['order_H_lm']]}"
          f"  M {['%.2f' % p for p in c['order_M_lm']]}")
    if meta["has_tp"]:
        rel = np.abs(np.asarray(c["L2_Ham_lm"], float)
                     - np.asarray(c["L2_Ham_tp"], float)) / np.asarray(
                         c["L2_Ham_lm"], float)
        print(f"  |lm-tp|/lm per rung: {['%.1e' % v for v in rel]}")
    # Not plotted (the dropped P=0.1 panel), kept as a checkable diagnostic.
    if meta.get("has_by"):
        ratio = (np.asarray(c["L2_Ham_by_p010"], float)
                 / np.asarray(c["L2_Ham_lm_p010"], float))
        print(f"  [P=0.1, unplotted] analytic/solved H: "
              f"{['%.1f' % v for v in ratio]}")
        print(f"  [P=0.1, unplotted] analytic orders:   "
              f"{['%.2f' % p for p in c['order_H_by_p010']]}")
    for rec in amr:
        print(f"  AMR {rec['tag']}: L2(H) {rec['L2_Ham']:.3e}, "
              f"{rec['n_levels']} levels, finest dx {rec['dx_finest']:.4g}")
    for k, v in variants.items():
        print(f"  variant {k}: L2(H) {['%.3e' % x for x in v['L2_Ham']]}")


if __name__ == "__main__":
    main()
