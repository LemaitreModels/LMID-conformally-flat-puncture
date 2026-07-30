#!/usr/bin/env python
"""Data for fig06_targeting: distill the min/median/max whisker curves to figdata/.

Source (raw): reports/P3/qc_targeting_100.json  (key "qc_targeting"; the shipped N=100 run).
The plotter used to recompute the across-target whiskers from the full per-target run log;
this script does that reduction ONCE and stores only the (x, med, lo, hi) curve per method,
so the plotter just draws errorbars.

Run:  python fig06_targeting_data.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import source, dump

PLOT = ("cold", "gradient")


def _whiskers(runs, m, n):
    """Across-target min/median/max residual at each cumulative-solve count, kept where a
    majority of targets are still iterating (verbatim from the old targeting plotter)."""
    by_x = {}
    for r in runs:
        for ns, res in r[m]["history"]:
            by_x.setdefault(int(ns), []).append(max(float(res), 1e-16))
    xs = [x for x in sorted(by_x) if len(by_x[x]) >= n // 2]
    med = np.array([np.median(by_x[x]) for x in xs])
    lo = np.array([np.min(by_x[x]) for x in xs])
    hi = np.array([np.max(by_x[x]) for x in xs])
    return np.array(xs, float), med, lo, hi


def build():
    with open(source("qc_targeting")) as f:
        d = json.load(f)
    runs = d["runs"]
    n = len(runs)
    out = {"n": n, "methods": {}}
    for m in PLOT:
        x, med, lo, hi = _whiskers(runs, m, n)
        out["methods"][m] = dict(x=x, med=med, lo=lo, hi=hi)
    p = dump("fig06_targeting", out)
    print(f"wrote {os.path.relpath(p)}  (n={n} targets)")


if __name__ == "__main__":
    build()
