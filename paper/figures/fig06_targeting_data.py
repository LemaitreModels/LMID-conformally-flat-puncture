#!/usr/bin/env python
"""Data for fig06_targeting: distill the min/median/max whisker curves to figdata/.

Source (raw): reports/P3/qc_targeting_chi_prod_fixed_100.json (key "qc_targeting"; the
N=100 FIXED-BUDGET run on the PRODUCTION 4-D chi model — gradient 4 certified solves,
black box 14, every target carried to the same count).  The plotter used to recompute
the across-target whiskers from the full per-target run log; this script does that
reduction ONCE and stores only the (x, med, lo, hi) curve per method, so the plotter
just draws errorbars.

The cost metric is NOT the number of solves performed (that is just the budget) but
``solves_to_tol``, the first solve count reaching the control tolerance.  Both control
loops test the tolerance at the TOP of their iteration, so the fixed-budget history is
the early-exit history plus a tail: the shared prefix is bit-identical and the crossing
is unchanged.  That was verified on the two runs (same seed) over all 100 targets and
3 methods, max|delta| = 0.0, which is what lets this figure plot a fixed budget while
the text still quotes the early-exit cost.

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
TOL = 1e-8          # the control tolerance the run stops at (qc_targeting tol_ctrl)


def _whiskers(runs, m, n, min_support=1.0, hold_gaps=True, stop_at_tol=TOL):
    """Across-target min/median/max residual at each cumulative-solve count.

    Three deliberate choices, each fixing a way the predecessor of this figure
    misread its own data:

    ``min_support`` — the fraction of targets a solve count must carry to be plotted.
    The predecessor used 0.5 on an early-exit run, which truncated both curves near
    their median: the last plotted point was not "every target converged" but "half
    have stopped", so its max whisker sat ABOVE the tolerance line while the steps
    that carried the stragglers below it were dropped.  It also biased the surviving
    sample toward the hard targets.  On a FIXED-BUDGET run every target reaches the
    same count, so 1.0 keeps every point and each is the full sample.

    ``hold_gaps`` — the black-box loop spends ``d`` solves on a finite-difference
    Jacobian without changing the iterate, so those solve counts carry no residual
    entry.  Interpolating across the gap draws a descending line and reads as
    improvement that did not happen; holding the previous value draws the flat
    segment that actually occurred, and makes the Jacobian tax visible.

    ``stop_at_tol`` — draw each curve up to and including the first solve count at
    which the MAX whisker reaches tolerance, i.e. where the last target converged.
    Beyond that the iteration is running past its own stopping criterion and drives
    the residual to the ``M_ADM`` read's noise floor (~1e-15), which plunges the
    median and stretches the axis without saying anything about cost.  Pass ``None``
    to draw the full budget.
    """
    by_x = {}
    for r in runs:
        for ns, res in r[m]["history"]:
            by_x.setdefault(int(ns), []).append(max(float(res), 1e-16))
    need = min_support * n
    xs = [x for x in sorted(by_x) if len(by_x[x]) >= need]
    dropped = [x for x in sorted(by_x) if len(by_x[x]) < need]
    if dropped:
        print(f"    [{m}] NOTE: dropped solve counts {dropped} "
              f"(support < {min_support:.0%} of {n} targets)")
    if hold_gaps and xs:
        filled, last = {}, None
        for x in range(min(xs), max(xs) + 1):
            if x in by_x:
                last = by_x[x]
            filled[x] = last
        by_x, xs = filled, list(range(min(xs), max(xs) + 1))
    med = np.array([np.median(by_x[x]) for x in xs])
    lo = np.array([np.min(by_x[x]) for x in xs])
    hi = np.array([np.max(by_x[x]) for x in xs])
    if stop_at_tol is not None and len(hi):
        k = next((i for i, h in enumerate(hi) if h <= stop_at_tol), len(hi) - 1)
        print(f"    [{m}] drawn to x={xs[k]} (max whisker {hi[k]:.2e} "
              f"<= tol {stop_at_tol:.0e}); budget ran to x={xs[-1]}")
        xs, med, lo, hi = xs[:k + 1], med[:k + 1], lo[:k + 1], hi[:k + 1]
    return np.array(xs, float), med, lo, hi


def _meta(summary):
    """Provenance of the run these curves come from (see the module docstring)."""
    keys = ("box", "axis_names", "level", "n_solver_nodes", "grid", "model",
            "model_git_commit", "spin_parameterization", "targets", "active",
            "theta0", "seed")
    meta = {k: summary[k] for k in keys if k in summary}
    meta["budgets"] = summary.get("budgets") or {}
    # The COST numbers the caption quotes are solves-to-tolerance.  Under a fixed
    # budget ``solves`` is just the budget, so quoting it would silently turn the
    # cost claim into a restatement of the run configuration; ``solves_to_tol`` is
    # the crossing and equals ``solves`` in early-exit mode.
    cost = summary.get("solves_to_tol") or summary["solves"]
    for stat in ("min", "median", "max"):
        meta[f"cost_{stat}"] = {m: cost[m][stat] for m in cost if cost[m]}
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
    # the extent each curve is drawn to (= where its last target converged) and the
    # max whisker there, so the caption's "for every target" is checkable
    out["meta"]["drawn_to"] = {m: float(out["methods"][m]["x"][-1]) for m in PLOT}
    out["meta"]["max_whisker_at_end"] = {m: float(out["methods"][m]["hi"][-1])
                                         for m in PLOT}
    out["meta"]["tol"] = TOL
    p = dump("fig06_targeting", out)
    box = out["meta"].get("box")
    print(f"wrote {os.path.relpath(p)}  (n={n} targets, "
          f"model={out['meta'].get('model')}, box={box})")
    print(f"  drawn to {out['meta']['drawn_to']}, "
          f"max whisker there {out['meta']['max_whisker_at_end']}, tol {TOL:.0e}")


if __name__ == "__main__":
    build()
