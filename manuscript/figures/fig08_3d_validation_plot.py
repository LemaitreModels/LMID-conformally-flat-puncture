#!/usr/bin/env python
"""Generates fig08_3d_validation.pdf (manuscript Fig. 8).

First non-axisymmetric (3-D) sweep, two panels:
 (a) convergence ladder -- ||R||_inf and psi-vs-TwoPunctures vs meridian
     resolution for a genuinely non-axisymmetric slice;
 (b) azimuthal phi-mode amplitude spectrum of the solved field (exponential
     decay in m).

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figdims(1, 2))

    # reserve C0-C2 (tab:blue/orange/green) for the models; diagnostics start at C3
    ax1.semilogy(Na, resid, "o-", color="C3", label=r"$\|R\|_\infty$ (residual)")
    if np.any(np.isfinite(dpsi)):
        ax1.semilogy(Na, dpsi, "s--", color="C4", label=r"$|\psi-\psi_{\rm TP}|_\infty$")
    ax1.set_xlabel(r"meridian resolution $N_A$  (with $N_B,N_\phi$ scaled)")
    ax1.set_ylabel("error")
    ax1.set_title("(a) non-axisymmetric convergence vs TwoPunctures")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    for i, r in enumerate(d["spectrum"]):
        amps = np.array(r["phi_amps"])
        m = np.array(r["m_vals"])
        amps_n = np.where(amps > 0, amps, np.nan)
        ax2.semilogy(m, amps_n, "o-", color=f"C{3 + i}",
                     label=fr"$\theta_S={r['tilt_deg']:.0f}^\circ$")
    ax2.set_xlabel(r"azimuthal mode $m$")
    ax2.set_ylabel(r"$\max_{A,B}|\hat u_m|$")
    ax2.set_title(r"(b) $\phi$-mode spectrum ($|S|=0.3$, $b=1.5$)")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    stem = os.path.join(HERE, "fig08_3d_validation")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig08_3d_validation.pdf")


if __name__ == "__main__":
    main()
