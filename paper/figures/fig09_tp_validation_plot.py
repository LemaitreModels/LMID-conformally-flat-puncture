#!/usr/bin/env python
"""Generates fig09_tp_validation.pdf (the paper's TwoPunctures validation figure).

A 1x3 figure at \\textwidth (a `figure*`).  Every curve is a min/median/max BAND over
configurations drawn from the production box, so no panel rests on one arbitrary parameter
point -- the ladder structure (x axis = resolution) is what still shows the disagreement to
be limited by spatial resolution rather than by the nonlinear solve:

 (a) the pointwise field agreement ||psi - psi_TP||_inf (falling) TOGETHER WITH the
     certified residual ||R||_inf (rising).  They share one panel deliberately: the opposite
     trends are the argument that the residual's rise is roundoff amplification in
     unpopulated high-m azimuthal modes, not a loss of convergence.
 (b) the integral agreement |M_ADM - E_TP|/E_TP.
 (c) the azimuthal Fourier spectrum |u_m|/|u_0| at the best-resolved rung, which shows how
     many phi modes the quasi-circular data actually need.

This figure REPLACES both the former fig08 (a separate non-axisymmetric validation at
b=1.5, head-on) and the former single-configuration fig09.  The QC data are already
non-axisymmetric -- the tangential momentum puts ~2% of the field in m=2 and generic spins
add ~2% at m=1 -- so panel (c) covers what fig08's spectrum panel did, inside the family the
models are actually built on.  The one check QC cannot supply, that a head-on slice with spin
along the collision axis keeps every m>=1 mode at roundoff, is a text number
(``meta.axisym_m_ge1_max``), not a panel.

Reads ONLY figdata/fig09_tp_validation.json (build it with fig09_tp_validation_data.py).
No reports/, no jax.

Run:  python fig09_tp_validation_plot.py
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


def _gridlabels(ladder):
    """Meridional labels only; Nphi is fixed along the ladder and goes in the axis label."""
    return [rf"${g[0]}\times{g[1]}$" for g in [r["grid"] for r in ladder]]


def _band(ax, x, rows, key, colour, label, ls="-", marker="o"):
    """median line + min-max envelope over the sampled configurations."""
    lo = np.array([r[key]["min"] for r in rows])
    md = np.array([r[key]["median"] for r in rows])
    hi = np.array([r[key]["max"] for r in rows])
    ax.fill_between(x, lo, hi, color=colour, alpha=0.20, lw=0)
    ax.plot(x, md, ls, marker=marker, ms=3.5, color=colour, lw=1.6, label=label)


def main():
    d = load("fig09_tp_validation")
    L, S, meta = d["ladder"], d["spectrum"], d["meta"]
    x = np.arange(len(L))
    labs = _gridlabels(L)
    floor = meta.get("oracle_floor")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figdims(1, 3))

    # ---- (a) field agreement (falling) vs certified residual (rising) -------------
    # C3/C4 keep the pointwise/integral colours the single-configuration figure used
    _band(ax1, x, L, "psi", "C3", r"$\|\psi-\psi_{\rm TP}\|_\infty$")
    _band(ax1, x, L, "residual", "C0", r"$\|R\|_\infty$ (certified)", ls="--", marker="s")
    ax1.axhline(meta["cert_tol"], ls=":", color="0.35", lw=1.1)
    ax1.text(0.02, meta["cert_tol"] * 1.6, "certification gate", fontsize=7, color="0.3",
             transform=ax1.get_yaxis_transform())
    if floor:
        ax1.axhline(floor, ls=":", color="0.6", lw=1.0)
    ax1.set_yscale("log")
    ax1.set_ylabel("agreement / residual")
    ax1.set_title(r"(a) field vs TP, and the residual", fontsize=10)
    ax1.legend(fontsize=7.5, loc="center left")

    # ---- (b) the integral (ADM mass) agreement ------------------------------------
    _band(ax2, x, L, "M_ADM", "C4", r"$|M_{\rm ADM}-E_{\rm TP}|/E_{\rm TP}$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$|M_{\rm ADM}-E_{\rm TP}|/E_{\rm TP}$")
    ax2.set_title(r"(b) ADM mass vs TP", fontsize=10)

    nphi = sorted({r["grid"][2] for r in L})
    xlab = (rf"grid $N_A\times N_B$ ($N_\phi={nphi[0]}$)" if len(nphi) == 1
            else r"grid $N_A\times N_B\times N_\phi$")
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=7.5, rotation=20)
        ax.set_xlabel(xlab)
        ax.grid(True, which="both", alpha=0.3)

    # ---- (c) how many azimuthal modes the QC data need ----------------------------
    # m=0 is the normalisation and carries no information; plotting it only compresses
    # the decades that matter
    Sm = [r for r in S if r["m"] >= 1]
    mm = np.array([r["m"] for r in Sm])
    lo = np.maximum(np.array([r["min"] for r in Sm]), 1e-18)
    md = np.array([r["median"] for r in Sm])
    hi = np.array([r["max"] for r in Sm])
    ax3.fill_between(mm, lo, hi, color="C2", alpha=0.20, lw=0)
    ax3.plot(mm, md, "-o", ms=3.5, color="C2", lw=1.6, label="median")
    ax3.plot(mm, hi, ":", color="C2", lw=1.0, label="max")
    ax3.set_yscale("log")
    ax3.set_xticks(mm)
    ax3.set_xlabel(r"azimuthal mode $m$")
    ax3.set_ylabel(r"$|\hat u_m|/|\hat u_0|$")
    ax3.set_title(r"(c) azimuthal spectrum", fontsize=10)
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    stem = os.path.join(HERE, "fig09_tp_validation")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig09_tp_validation.pdf")


if __name__ == "__main__":
    main()
