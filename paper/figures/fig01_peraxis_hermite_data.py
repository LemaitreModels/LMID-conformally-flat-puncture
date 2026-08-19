#!/usr/bin/env python
"""Data for fig01_peraxis_hermite: distill the DISTRIBUTIONAL per-axis curves.

Source (raw): reports/3D_parametric/qc_chi/peraxis_dist_chi.json (key
"peraxis_dist_chi"), produced by ``run_qc_peraxis_dist_chi.py --assemble``.  Unlike
the earlier single-base-point study, each (axis, Q) held-out error is measured over
``n_samples`` random base points (the seven non-swept axes drawn uniformly in their
boxes), so every curve is a DISTRIBUTION.

Keeps only what the 2x4 per-axis grid draws: per axis the Q ladder, the value and
value+gradient (Hermite) order-statistics {median, best, worst, p05, p95}, and a
median-fitted geometric rate (dec/Q) via the same ``_rate`` convention as the
original run (log-linear fit over the [1e-9, 1] window).  The heavy ``raw`` samples
and ``base_points`` meta are dropped so the committed figdata stays small.

Run:  python fig01_peraxis_hermite_data.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump
from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm

STATS = ("median", "best", "worst", "p05", "p95")

# The four axes of the shipped 4-D model (production_box "d4_qc_chi_prod"); the
# 8-D model is every axis this sweep measures.  Named here because the cost block
# below must say WHICH model each factor belongs to.
AXES_4D = ("b", "q", "chi_Ay", "chi_By")


def _rate(Qs, errs, floor=1e-9, ceil=1.0):
    """Geometric decay rate (decades/node) — identical to run_qc_peraxis_chi6._rate:
    log-linear fit over the [floor, ceil] window (drops the Hermite high-Q plateau).

    Returns ``(rate, intercept)``.  The intercept is what lets the cost block below
    invert the fit for a target accuracy; without it the rate alone fixes only the
    slope, and any node count derived from it would smuggle in an unstated offset."""
    Qs = np.asarray(Qs, float)
    errs = np.asarray([np.nan if v is None else v for v in errs], float)
    m = (errs > floor) & (errs < ceil) & np.isfinite(errs)
    if m.sum() < 2:
        m = (errs > 0) & np.isfinite(errs)
    slope, intercept = np.polyfit(Qs[m], np.log10(errs[m]), 1)
    return -float(slope), float(intercept)


def build():
    src = load_source("peraxis_dist_chi")
    A = src["A_per_axis"]
    m = src.get("meta", {})
    out = {"A_per_axis": {}, "meta": {
        "n_samples": m.get("n_samples"), "n_holdout": m.get("n_holdout"),
        "Q_ladder": m.get("Q_ladder"), "code_tag": m.get("code_tag"),
        "grid": [m.get("Na"), m.get("Nb"), m.get("Nphi")]}}
    for name in A:
        d = A[name]
        Qs = d["Qs"]
        rate_v, icept_v = _rate(Qs, d["value"]["median"])
        rate_h, icept_h = _rate(Qs, d["hermite"]["median"])
        out["A_per_axis"][name] = {
            "Qs": Qs,
            "value": {k: d["value"][k] for k in STATS},
            "hermite": {k: d["hermite"][k] for k in STATS},
            "rate_value": rate_v, "fit_intercept_value": icept_v,
            "rate_hermite": rate_h, "fit_intercept_hermite": icept_h,
        }
    # ---- the tensor-vs-sparse cost claim, derived rather than asserted ----
    # Sec. "Cost, memory, and certified residuals" quotes a tensor-grid blow-up
    # factor.  Nothing recorded it, so it could not be checked against the rates
    # this very figure measures.  Derived here, for BOTH shipped models, so the
    # paper can cite a traceable number and say which model it means -- the factor
    # differs by five orders of magnitude between them.
    #
    # Method, and its assumptions, stated because the extrapolation is the whole
    # content: each axis is taken to converge at its OWN fitted 1-D value-model
    # rate (the log-linear fit above, median held-out error, same [1e-9, 1] window
    # as `_rate`), and a full tensor grid is required to reach the target on every
    # axis, so Q_k = (intercept_k - log10(eps)) / rate_k and the node count is
    # prod_k (Q_k + 1).  That is a 1-D rate extrapolated to d dimensions; it
    # ignores any cross-axis interaction, which is exactly what the sparse grid
    # exploits, so it is a LOWER bound on the tensor cost.
    LEVEL = 5
    nodes_per_axis = 2 ** LEVEL + 1          # 33: the Clenshaw-Curtis rule at l=5
    cost = {"level": LEVEL, "nodes_per_axis": nodes_per_axis,
            "fit_window": [1e-9, 1.0], "statistic": "median held-out, value model",
            "assumption": "per-axis 1-D rate, independent axes; lower bound",
            "models": {}}
    for dim, axes in ((4, AXES_4D), (8, sorted(A))):
        sparse = pm.N_NODES[dim]
        tensor = float(nodes_per_axis) ** dim
        entry = {"axes": list(axes), "n_sparse_level5": sparse,
                 "n_tensor_at_level5": tensor,
                 "ratio_tensor_over_sparse": tensor / sparse,
                 "by_target_accuracy": {}}
        for eps in (1e-6, 1e-9):
            Qs = {ax: (out["A_per_axis"][ax]["fit_intercept_value"]
                       - math.log10(eps)) / out["A_per_axis"][ax]["rate_value"]
                  for ax in axes}
            n = 1.0
            for q in Qs.values():
                n *= (q + 1.0)
            entry["by_target_accuracy"][f"{eps:.0e}"] = {
                "Q_per_axis": Qs, "n_tensor": n,
                "ratio_over_sparse": n / sparse,
                "ratio_over_tensor_at_level5": n / tensor}
        cost["models"][f"{dim}D"] = entry
    out["cost"] = cost

    p = dump("fig01_peraxis_hermite", out)
    print(f"wrote {os.path.relpath(p)}  ({len(out['A_per_axis'])} axes, "
          f"n_samples={out['meta']['n_samples']})")


if __name__ == "__main__":
    build()
