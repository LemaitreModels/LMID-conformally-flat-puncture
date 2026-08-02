#!/usr/bin/env python
"""Generates fig04_polish_staircase.pdf (paper Fig. 4).

A 2x2 grid with a SHARED Newton-step x-axis per column. Columns are 4D | 8D; each panel draws three
curves as a median line with 1000-point min--max whiskers on every Newton polish step:
(1) cold start (no surrogate), (2) value-only surrogate, and (3) the shipped gradient-enhanced
(value + gradient + cross) POD warm start.

  TOP row    constraint residual ||R||_inf per step (the certified quantity).
  BOTTOM row field error ||u-u_true||_2/||u_true||_2 per step.

The bottom row makes the decoupling concrete: the POD warm start is already field-accurate at the
guess (~5e-5) while its residual is O(1e-2); both then fall to the numerical floor. Field error is
measured against the converged (certified) iterate.

Reads ONLY figdata/fig04_polish_staircase.json (build it with fig04_polish_staircase_data.py).
No reports/, no jax.

Run:  python fig04_polish_staircase_plot.py
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

# Column titles are presentation, so they live here rather than in the figdata json:
# a notation change then needs only a re-plot, not a solver recompute.
TITLES = {4: r"4D quasi-circular model: $\theta=(b,q,\chi^{A}_{y},\chi^{B}_{y})$",
          8: r"8D quasi-circular model: $\theta=(b,q,\boldsymbol{\chi}^{A},\boldsymbol{\chi}^{B})$"}

THRESH = 1e-10
FIELD_FLOOR = 1e-13          # below this the iterate == converged solution (log floor)
# convention (Figs. 1, 3, 4, 5): value = C0 (tab:blue),
# value + gradient + cross = C1 (tab:orange); cold baseline = grey
COLD_C, VALUE_C, POD_C = "0.6", "C0", "C1"


def _arr(s):
    """(steps, med, lo, hi) from a distilled staircase dict {K, med, lo, hi}."""
    K = int(s["K"])
    return (np.arange(K + 1, dtype=float),
            np.array(s["med"]), np.array(s["lo"]), np.array(s["hi"]))


def _trim(steps, med, extra=1):
    fc = next((i for i, m in enumerate(med) if m <= THRESH), len(med) - 1)
    return min(len(steps), fc + 1 + extra)


def _trim_field(med, extra=1):
    fc = next((i for i, m in enumerate(med) if m <= FIELD_FLOOR), len(med) - 1)
    return min(len(med), fc + 1 + extra)


def _plot_residual(ax, col):
    """Top-row constraint-residual staircase; returns the trimmed x-extent."""
    xmax = 0
    x, med, lo, hi = _arr(col["res_cold"])
    n = _trim(x, med, extra=1)
    ax.errorbar(x[:n] - 0.12, med[:n], yerr=[med[:n] - lo[:n], hi[:n] - med[:n]],
                fmt="-s", color=COLD_C, ms=5, lw=1.7, capsize=3.5,
                elinewidth=1.1, capthick=1.1, zorder=3,
                label="cold start (no surrogate)")
    xmax = max(xmax, x[n - 1])
    if "res_value" in col:                        # optional value-only series
        x, med, lo, hi = _arr(col["res_value"])
        n = _trim(x, med, extra=1)
        rv = col["res_value"].get("r")             # set only for the POD value-only curve
        vlabel = (fr"value POD ($r{{=}}{rv}$)" if rv else "value surrogate")
        ax.errorbar(x[:n], med[:n], yerr=[med[:n] - lo[:n], hi[:n] - med[:n]],
                    fmt="-^", color=VALUE_C, ms=5, lw=1.7, capsize=3.5,
                    elinewidth=1.1, capthick=1.1, zorder=4,
                    label=vlabel)
        xmax = max(xmax, x[n - 1])
    x, med, lo, hi = _arr(col["res_pod"])
    n = _trim(x, med, extra=1)
    r = col["res_pod"].get("r")
    ax.errorbar(x[:n] + 0.12, med[:n], yerr=[med[:n] - lo[:n], hi[:n] - med[:n]],
                fmt="-o", color=POD_C, ms=5, lw=1.7, capsize=3.5,
                elinewidth=1.1, capthick=1.1, zorder=5,
                label=fr"value + gradient + cross POD ($r{{=}}{r}$)")
    xmax = max(xmax, x[n - 1])
    ax.axhline(THRESH, ls=":", color="k", alpha=0.7, lw=1.1)
    ax.set_yscale("log")
    ax.set_title(TITLES[col["dim"]], fontsize=9.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.8, loc="upper right", framealpha=0.93)
    return xmax


def _plot_field(ax, col):
    """Bottom-row field-error staircase (cold + POD); same column colours as the residual panel."""
    for fam_key, color, fmt, off, label, z in (
            ("fld_cold", COLD_C, "-s", -0.12, "cold start (no surrogate)", 3),
            ("fld_value", VALUE_C, "-^", 0.0, "value", 4),
            ("fld_pod", POD_C, "-o", 0.12, "value + gradient + cross", 5)):
        if fam_key not in col:                    # optional value-only series
            continue
        x, med, lo, hi = _arr(col[fam_key])
        n = _trim_field(med, extra=1)
        x = x[:n]
        med = np.maximum(med[:n], FIELD_FLOOR)
        lo = np.maximum(lo[:n], FIELD_FLOOR)
        hi = np.maximum(hi[:n], FIELD_FLOOR)
        ax.errorbar(x + off, med, yerr=[med - lo, hi - med],
                    fmt=fmt, color=color, ms=5, lw=1.7, capsize=3.5,
                    elinewidth=1.1, capthick=1.1, zorder=z, label=label)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


def main():
    d = load("fig04_polish_staircase")
    cols = d["cols"]
    fig, axes = plt.subplots(2, 2, figsize=figdims(2, 2), sharex="col", sharey="row")
    xmax = 0
    for j, col in enumerate(cols):
        xmax = max(xmax, _plot_residual(axes[0, j], col))
    for j, col in enumerate(cols):
        _plot_field(axes[1, j], col)

    xmax = int(round(xmax))
    for ax in (axes[1, 0], axes[1, 1]):           # sharex='col' -> propagates to top row
        ax.set_xlim(-0.5, xmax + 0.5)
        ax.set_xticks(range(xmax + 1))
        ax.set_xticklabels(["guess"] + [str(k) for k in range(1, xmax + 1)])
        ax.set_xlabel("Newton polish steps")
    axes[0, 0].set_ylabel(r"constraint residual $\|R\|_\infty$")
    axes[1, 0].set_ylabel(r"field error $\|u-u_{\rm true}\|_2/\|u_{\rm true}\|_2$")
    axes[0, 0].text(0.03, THRESH * 1.6, r"certification threshold $10^{-10}$",
                    transform=axes[0, 0].get_yaxis_transform(), fontsize=8.2,
                    color="k", alpha=0.8)
    fig.tight_layout()
    stem = os.path.join(HERE, "fig04_polish_staircase")
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote fig04_polish_staircase.pdf")


if __name__ == "__main__":
    main()
