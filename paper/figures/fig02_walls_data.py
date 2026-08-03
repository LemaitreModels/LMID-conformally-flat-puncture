#!/usr/bin/env python
"""Data for fig02_walls: distill the plotted arrays to figdata/.

Source (raw): reports/3D_parametric/qc_chi_prod/walls_d4_qc_chi.json (key "walls_dense",
whose registry entry documents why this superseded the old qc/walls_d4_qc_dense.json).
This ONE figure carries all three analyticity walls (formerly two separate figures):
the separation/merger wall (block B_wall_b), the mass-ratio wall (block Q_wall_q),
and the spin wall (block C_wall_spin).  Panel order is separation | q | spin, i.e.
increasing distance-to-wall in units of the sampled range.

For each fit range we keep the held-out convergence curve (Qs, errs), the geometric
rate, the inferred nearest real singularity theta*, and the fit-window size n_fit.
We also precompute the geometric fit LINE (slope, intercept of log10(err) vs Q over
the first n_fit points) so the plotter can draw data + fit with no refitting; the
fit slope reproduces the reported rate exactly (asserted here).

Run:  python fig02_walls_data.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump

# Separation fit ranges dropped from the figure for legibility (still in the raw
# source); this keeps the panel to three curves, matching its two neighbours.
#
# 1.2 was in the superseded narrow-box sweep.  1.5 is a PRESENTATION drop from the
# production sweep production_box.WALL_B_MIN_SWEEP = (3.0, 2.0, 1.5, 1.0), made by
# the author so the separation panel carries three curves like the mass-ratio and
# spin panels rather than four.  It costs the figure nothing: the panel's claim is
# that the rate degrades smoothly toward coincidence while b* stays pinned at the
# merger, and 3.0/2.0/1.0 (rho = 0.52/0.40/0.27, b* = 0.22/0.18/0.12) carry both --
# 3.0 is the production box edge, hence the in-box rate, and 1.0 is the smallest
# range, the one the text's momentum-divergence undershoot refers to.  No sentence
# in the paper quotes the 1.5 curve.
DROP_B_MIN = (1.2, 1.5)


def _fit(Qs, errs, n_fit):
    """Slope/intercept of log10(err) vs Q over the first n_fit points (the clean
    geometric window before the machine-precision floor); slope == -rate."""
    Qf = np.asarray(Qs[:n_fit], float)
    lf = np.log10(np.asarray(errs[:n_fit], float))
    slope, intercept = np.polyfit(Qf, lf, 1)
    return float(slope), float(intercept)


# The fitted-window size.  The producer records it as "n_fit_points", but the
# production sweep's Q_wall_q block does not carry the key (and its B/C blocks
# predate it), so derive it when absent by the producer's OWN rule --
# run_qc_walls_sweep_chi._rate_n: the window is the points inside
# (FIT_FLOOR, FIT_CEIL), i.e. above the machine-precision floor and below the
# useless-error ceiling, which that function asserts is a leading run.  This is a
# recovery of the recorded quantity, not a re-choice of the window: the `assert`
# at each call site refits the recovered window and requires it to reproduce the
# stored `rate` to 1e-9, so a wrong n_fit fails the build loudly.
FIT_FLOOR, FIT_CEIL = 1e-9, 1.0


def _n_fit(w):
    """``n_fit_points`` if the source records it, else recovered (see above)."""
    if "n_fit_points" in w:
        return w["n_fit_points"]
    e = np.asarray(w["errs"], float)
    m = (e > FIT_FLOOR) & (e < FIT_CEIL) & np.isfinite(e)
    if m.sum() < 2:
        m = (e > 0) & np.isfinite(e)
    n = int(m.sum())
    assert m[:n].all(), f"fit window is not a leading run: {m.astype(int)}"
    return n


def build():
    r = load_source("walls_dense")

    B = []
    for w in r["B_wall_b"]:
        if any(abs(w["b_min"] - b) < 1e-9 for b in DROP_B_MIN):
            continue
        n_fit = _n_fit(w)
        slope, intercept = _fit(w["Qs"], w["errs"], n_fit)
        assert abs(-slope - w["rate"]) < 1e-9, (w["b_min"], -slope, w["rate"])
        B.append(dict(Qs=w["Qs"], errs=w["errs"], b_min=w["b_min"], rate=w["rate"],
                      theta_star=w["inferred_sing"], n_fit=n_fit,
                      fit_slope=slope, fit_intercept=intercept))

    # mass-ratio wall: same shape as the spin block, with q_max/q_star in place of
    # chi_max/chi_star.  The plotter reads the inferred singularity as "theta_star",
    # so q_star is mapped onto that key here.
    Q = []
    for w in r["Q_wall_q"]:
        n_fit = _n_fit(w)
        slope, intercept = _fit(w["Qs"], w["errs"], n_fit)
        assert abs(-slope - w["rate"]) < 1e-9, (w["q_max"], -slope, w["rate"])
        Q.append(dict(Qs=w["Qs"], errs=w["errs"], q_max=w["q_max"], rate=w["rate"],
                      theta_star=w["q_star"], n_fit=n_fit,
                      fit_slope=slope, fit_intercept=intercept))

    C = []
    for w in r["C_wall_spin"]:
        n_fit = _n_fit(w)
        slope, intercept = _fit(w["Qs"], w["errs"], n_fit)
        assert abs(-slope - w["rate"]) < 1e-9, (w["chi_max"], -slope, w["rate"])
        C.append(dict(Qs=w["Qs"], errs=w["errs"], chi_max=w["chi_max"], rate=w["rate"],
                      theta_star=w["chi_star"], n_fit=n_fit,
                      fit_slope=slope, fit_intercept=intercept))

    p = dump("fig02_walls", dict(B_wall_b=B, Q_wall_q=Q, C_wall_spin=C))
    print(f"wrote {os.path.relpath(p)}  ({len(B)} separation ranges, "
          f"{len(Q)} mass-ratio ranges, {len(C)} spin ranges)")


if __name__ == "__main__":
    build()
