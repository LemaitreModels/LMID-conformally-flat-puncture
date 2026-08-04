#!/usr/bin/env python
"""Generates fig08_3d_validation.pdf (paper Fig. 8).

First non-axisymmetric (3-D) sweep, ONE single-column panel: the convergence
ladder -- ||R||_inf and psi-vs-TwoPunctures vs meridian resolution for a
genuinely non-axisymmetric slice.

The figure used to carry a second panel (the azimuthal phi-mode amplitude
spectrum).  That panel was a diagnostic of OUR OWN discretization -- no
TwoPunctures data appeared in it -- so it made no code-to-code claim and was
dropped to keep this appendix figure purely a validation against the oracle.
Its content now lives as prose in the appendix; the ``spectrum`` block is kept
in figdata as the traceable source of those quoted numbers.

Reads ONLY figdata/fig08_3d_validation.json (build it with
fig08_3d_validation_data.py). No reports/, no jax.

Run:  python fig08_3d_validation_plot.py
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


def main():
    d = load("fig08_3d_validation")
    L = d["ladder"]
    Na = np.array(L["Na"])
    resid = np.array(L["resid"])
    dpsi = np.array([v if v is not None else np.nan for v in L["dpsi"]])

    fig, ax1 = plt.subplots(1, 1, figsize=figdims(1, 1))

    # reserve C0-C1 (tab:blue/orange) for the two models; diagnostics start at C2.
    # C2 is the TwoPunctures comparison and C3 our own certified residual, so the
    # field error here carries the same colour as the field panel of fig09.
    ax1.semilogy(Na, resid, "o-", color="C3", label=r"$\|R\|_\infty$ (residual)")
    if np.any(np.isfinite(dpsi)):
        # double-bar norm, matching the residual entry above (and the caption)
        ax1.semilogy(Na, dpsi, "s-", color="C2",
                     label=r"$\|\psi-\psi_{\rm TP}\|_\infty$")
    # name the grid outright, as fig09 does: the ladder refines N_A, N_B and N_phi
    # together, so "meridian resolution N_A (with N_B,N_phi scaled)" made the reader
    # guess the two numbers that are not shown.  x stays numeric (the Na are evenly
    # spaced) and only the tick labels carry the full grid.
    ax1.set_xticks(Na)
    ax1.set_xticklabels([rf"{a}$\times${b}$\times${p}"
                         for a, b, p in zip(Na, L["Nb"], L["Nphi"])], rotation=30)
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_xlabel(r"grid $N_A\times N_B\times N_\phi$")
    ax1.set_ylabel("residual / error")
    ax1.set_title("Non-axisymmetric field vs TP", fontsize=10)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    # this panel spans ~7 decades, so the log subticks (and their gridlines) are
    # dense clutter; the labelled decades alone carry the scale
    ax1.yaxis.set_minor_locator(NullLocator())

    fig.tight_layout()
    stem = os.path.join(HERE, "fig08_3d_validation")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig08_3d_validation.pdf")


if __name__ == "__main__":
    main()
