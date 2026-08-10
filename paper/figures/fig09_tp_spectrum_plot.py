#!/usr/bin/env python
"""Generates fig09_tp_spectrum.pdf (the azimuthal spectrum of the quasi-circular data).

A 1x1 figure at \\columnwidth (same geometry as Figs. 6 and 7): the azimuthal Fourier
spectrum |u_m|/|u_0| at the best-resolved rung, as a median with min--max whiskers over the
same sampled configurations as fig08.  It shows how many phi modes the quasi-circular data
actually need, i.e. what sets the azimuthal truncation N_phi.

WHY ITS OWN FIGURE.  This was the third panel of a 1x3 ``figure*`` whose other two panels
walk the meridional resolution ladder.  It shares no abscissa with them (mode index, not
grid) and answers a different question -- a design statement about N_phi rather than a
validation statement against the oracle -- so the wide float only shared a caption, not a
coordinate.  On its own it gets the full column width at the paper's standard single-panel
geometry, and places freely next to the paragraph that discusses it.

m=0 is the normalisation and carries no information; plotting it would only compress the
decades that matter.  The one check the quasi-circular family cannot supply -- that a head-on
slice with spin along the collision axis keeps every m>=1 mode at roundoff -- is a text
number (``meta.axisym_m_ge1_max``), not a curve.

WHY THE AXIS STOPS AT m=4.  It is not a truncation: ``operators_3d.fourier_modes`` gives
m = 0..Nphi//2, so the production Nphi=8 grid carries m<=4 and every mode it represents is
plotted.  That last point is the grid's Nyquist mode and is not comparable like-for-like
with the modes below it: on Nphi equispaced nodes sin(4 phi) vanishes identically, so that
bin holds only the cos(4 phi) component (one real degree of freedom against two for
m=1..3).  Consistently it is the one point that does not continue the decay.  That is a
caption statement, not a plotted feature -- the panel is left unannotated.

COLOURS follow the paper convention (see README): C0--C1 are reserved for the standard
surrogate models, so this non-model diagnostic starts the cycle at C2.

Reads ONLY figdata/fig09_tp_spectrum.json (build it with fig09_tp_spectrum_data.py).
No reports/, no jax.

Run:  python fig09_tp_spectrum_plot.py
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

SPEC_C = "C2"          # non-model diagnostic: the cycle starts at C2 (see README)


def main():
    d = load("fig09_tp_spectrum")
    S = [r for r in d["spectrum"] if r["m"] >= 1]

    mm = np.array([r["m"] for r in S])
    lo = np.maximum(np.array([r["min"] for r in S]), 1e-18)
    md = np.array([r["median"] for r in S])
    hi = np.array([r["max"] for r in S])

    fig, ax = plt.subplots(1, 1, figsize=figdims(1, 1))
    ax.errorbar(mm, md, yerr=[md - lo, hi - md], fmt="-o", color=SPEC_C,
                ms=5, lw=1.7, capsize=3.5, elinewidth=1.1, capthick=1.1)

    ax.set_yscale("log")
    ax.set_xticks(mm)
    ax.set_xlabel(r"azimuthal mode $m$")
    ax.set_ylabel(r"$|\hat u_m|/|\hat u_0|$")
    ax.set_title("Azimuthal spectrum at the finest resolution", fontsize=10)
    ax.yaxis.set_minor_locator(NullLocator())  # no y sub-ticks (matches Figs. 2, 3, 8)
    ax.grid(True, which="major", alpha=0.3)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig09_tp_spectrum")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig09_tp_spectrum.pdf")


if __name__ == "__main__":
    main()
