#!/usr/bin/env python
"""Data for fig02_walls: distill the plotted arrays to figdata/.

Source (raw): reports/3D_parametric/qc/walls_d4_qc_dense.json  (key "walls_dense").
This ONE figure now carries both analyticity walls (formerly two separate figures):
the separation/merger wall (block B_wall_b) and the spin wall (block C_wall_spin).

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

# separation fit ranges dropped from the figure for legibility (still in the raw source);
# keeps the panel to three curves, matching the spin panel.
DROP_B_MIN = (1.2,)


def _fit(Qs, errs, n_fit):
    """Slope/intercept of log10(err) vs Q over the first n_fit points (the clean
    geometric window before the machine-precision floor); slope == -rate."""
    Qf = np.asarray(Qs[:n_fit], float)
    lf = np.log10(np.asarray(errs[:n_fit], float))
    slope, intercept = np.polyfit(Qf, lf, 1)
    return float(slope), float(intercept)


def build():
    r = load_source("walls_dense")

    B = []
    for w in r["B_wall_b"]:
        if any(abs(w["b_min"] - b) < 1e-9 for b in DROP_B_MIN):
            continue
        slope, intercept = _fit(w["Qs"], w["errs"], w["n_fit_points"])
        assert abs(-slope - w["rate"]) < 1e-9, (w["b_min"], -slope, w["rate"])
        B.append(dict(Qs=w["Qs"], errs=w["errs"], b_min=w["b_min"], rate=w["rate"],
                      theta_star=w["inferred_sing"], n_fit=w["n_fit_points"],
                      fit_slope=slope, fit_intercept=intercept))

    C = []
    for w in r["C_wall_spin"]:
        slope, intercept = _fit(w["Qs"], w["errs"], w["n_fit_points"])
        assert abs(-slope - w["rate"]) < 1e-9, (w["chi_max"], -slope, w["rate"])
        C.append(dict(Qs=w["Qs"], errs=w["errs"], chi_max=w["chi_max"], rate=w["rate"],
                      theta_star=w["chi_star"], n_fit=w["n_fit_points"],
                      fit_slope=slope, fit_intercept=intercept))

    p = dump("fig02_walls", dict(B_wall_b=B, C_wall_spin=C))
    print(f"wrote {os.path.relpath(p)}  ({len(B)} separation ranges, {len(C)} spin ranges)")


if __name__ == "__main__":
    build()
