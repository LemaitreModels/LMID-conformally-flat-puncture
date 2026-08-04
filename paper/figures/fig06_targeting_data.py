#!/usr/bin/env python
"""Data for fig06_targeting: distill the min/median/max whisker curves to figdata/.

Source (raw): reports/P3/qc_targeting_chi_prod_100.json (key "qc_targeting"; the N=100
run on the PRODUCTION 4-D chi model).  The plotter used to recompute the across-target
whiskers from the full per-target run log; this script does that reduction ONCE and
stores only the (x, med, lo, hi) curve per method, so the plotter just draws errorbars.

Alongside the curves it stores a ``meta`` block — the box, axis names, Smolyak level,
node count and model file the run was measured on.  The caption states the box, and the
predecessor of this figure was measured on a superseded narrow model without that being
visible anywhere in the figdata; ``meta`` makes the provenance travel with the numbers
and is a declared plotter key, so a pre-production figdata fails loudly.

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


def _meta(summary):
    """Provenance of the run these curves come from (see the module docstring)."""
    keys = ("box", "axis_names", "level", "n_solver_nodes", "grid", "model",
            "model_git_commit", "spin_parameterization", "targets", "active",
            "theta0", "seed")
    meta = {k: summary[k] for k in keys if k in summary}
    # the headline cost numbers, so the caption can be checked against the figdata
    meta["solves_median"] = {m: summary["solves"][m]["median"] for m in summary["solves"]}
    meta["solves_max"] = {m: summary["solves"][m]["max"] for m in summary["solves"]}
    meta["worst_certified_residual"] = summary.get("worst_certified_residual")
    meta["all_converged"] = summary.get("all_converged")
    return meta


def build():
    with open(source("qc_targeting")) as f:
        d = json.load(f)
    runs = d["runs"]
    n = len(runs)
    out = {"n": n, "methods": {}, "meta": _meta(d["summary"])}
    for m in PLOT:
        x, med, lo, hi = _whiskers(runs, m, n)
        out["methods"][m] = dict(x=x, med=med, lo=lo, hi=hi)
    p = dump("fig06_targeting", out)
    box = out["meta"].get("box")
    print(f"wrote {os.path.relpath(p)}  (n={n} targets, "
          f"model={out['meta'].get('model')}, box={box})")


if __name__ == "__main__":
    build()
