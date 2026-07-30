#!/usr/bin/env python
"""Data for fig02_peraxis_hermite: distill the DISTRIBUTIONAL per-axis curves.

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

Run:  python fig02_peraxis_hermite_data.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump

STATS = ("median", "best", "worst", "p05", "p95")


def _rate(Qs, errs, floor=1e-9, ceil=1.0):
    """Geometric decay rate (decades/node) — identical to run_qc_peraxis_chi6._rate:
    log-linear fit over the [floor, ceil] window (drops the Hermite high-Q plateau)."""
    Qs = np.asarray(Qs, float)
    errs = np.asarray([np.nan if v is None else v for v in errs], float)
    m = (errs > floor) & (errs < ceil) & np.isfinite(errs)
    if m.sum() < 2:
        m = (errs > 0) & np.isfinite(errs)
    return -float(np.polyfit(Qs[m], np.log10(errs[m]), 1)[0])


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
        out["A_per_axis"][name] = {
            "Qs": Qs,
            "value": {k: d["value"][k] for k in STATS},
            "hermite": {k: d["hermite"][k] for k in STATS},
            "rate_value": _rate(Qs, d["value"]["median"]),
            "rate_hermite": _rate(Qs, d["hermite"]["median"]),
        }
    p = dump("fig02_peraxis_hermite", out)
    print(f"wrote {os.path.relpath(p)}  ({len(out['A_per_axis'])} axes, "
          f"n_samples={out['meta']['n_samples']})")


if __name__ == "__main__":
    build()
