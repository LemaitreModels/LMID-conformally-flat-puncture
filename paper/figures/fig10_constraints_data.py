#!/usr/bin/env python
"""Builds figdata/fig10_constraints.json (paper Fig. 10).

Constraint violation of the initial data on an evolution grid, measured by
**GRTeclyn** — an independent numerical-relativity code — rather than by the
in-house finite-difference monitor this figure used previously.  See
``docs/GRTECLYN_CONSTRAINTS_PLAN.md``.

WHY THE MEASUREMENT MOVED.  The previous version interpolated the spectral data
onto uniform Cartesian grids and evaluated the constraints with a bespoke
second-order monitor in this repo (``validation/constraints.py``, which stays as
an internal check).  The measurement is now made by GRTeclyn's own
``Constraints`` class: fourth-order stencils, working from the *evolution*
variables ``(chi, h_ij, A_ij, K)``, so it also exercises the conversion into
BSSN/CCZ4 form that an evolution actually performs.  Fourth order reaches the
initial data's own error at a spacing a second-order monitor cannot: the same
absolute error that needs ``h ~ 2e-4 M`` at second order arrives near
``h ~ 1e-1 M`` at fourth.

WHAT THIS SCRIPT DOES.  It distils GRTeclyn's ``constraint_norms.json`` output
(one file per rung, collected per series into ``<tag>/ladder.json`` by
``runs/lm_constraints/run_ladder.sh``).  No solver, no oracle, no jax here — the
heavy tier is the GRTeclyn runs, a cluster job
(``runs/lm_constraints/submit_step4.slurm``).  Point ``--runs`` at that tree or
set ``$LM_GRTECLYN_RUNS``.

THE SERIES, AND WHAT EACH IS FOR.

  lm_single   exactness gate.  One puncture: psi = 1 + m_A/2r_A is harmonic and
              K_ij = 0, so the continuum constraints vanish IDENTICALLY and the
              measurement is pure truncation error.  L2(M) is exactly zero at
              every rung, which is a check on the consumer, not a convergence
              result.
  lm_anchor   the App. A anchor (equal bare masses, b = 3, P = 0.5).
  tp_anchor   TwoPunctures' conformal factor through the IDENTICAL class,
              interpolation, stencils and grid — the parity gate.  The only
              difference between this series and the last is the initial data.
  lm_p010 /   the one configuration both initial-data sets can be handed to the
  by_p010     same code.  GRTeclyn's analytic Bowen-York initial data enforces
              |P| < 0.3 m, so it cannot represent the anchor at all; at P = 0.1
              it can, and its Hamiltonian violation stops converging while ours
              does not.  This is the panel that separates a solved solution from
              an O(P^2)-accurate one.
  lm_qc       a genuinely non-axisymmetric quasi-circular slice with spins.

Run:  python fig10_constraints_data.py --runs /path/to/runs/lm_constraints
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _figdata import dump

# tag -> (figdata key, label)
SERIES = (
    ("lm_single", "single", r"single puncture (exact solution)"),
    ("lm_anchor", "lm", r"this work, $P=0.5$"),
    ("tp_anchor", "tp", r"TwoPunctures, same code path"),
    ("lm_p010", "lm_p010", r"this work, $P=0.1$"),
    ("by_p010", "by_p010", r"GRTeclyn analytic Bowen--York, $P=0.1$"),
    ("lm_qc", "qc", r"quasi-circular with spins"),
)
# Resolution variants, used to attribute an observed floor to the exported
# representation rather than to the solve.  Optional.
VARIANTS = (
    ("lm_qc_nphi16", "qc_nphi16", r"quasi-circular, $N_\phi=16$"),
    ("lm_anchor_hi", "lm_hi", r"this work, $52{\times}36\to72{\times}48$"),
)
AMR_TAGS = ("lm_anchor_amr4", "lm_anchor_amr6")


def _runs_root(cli):
    root = cli or os.environ.get("LM_GRTECLYN_RUNS")
    if not root:
        raise SystemExit(
            "Point --runs at the GRTeclyn run tree (or set $LM_GRTECLYN_RUNS).\n"
            "Produce it with runs/lm_constraints/submit_step4.slurm on the "
            "cluster; see docs/DATA.md.")
    if not os.path.isdir(root):
        raise SystemExit(f"--runs {root!r} is not a directory")
    return root


def _local_orders(h, e):
    """Pairwise log-log slopes.

    Reported as measured local orders rather than one least-squares slope: a
    series that leaves the asymptotic regime (or hits the initial data's own
    error) is then visible instead of averaged away, and for this figure that
    departure IS the result.
    """
    out = []
    for (h0, e0), (h1, e1) in zip(zip(h, e), zip(h[1:], e[1:])):
        if e0 and e1 and e0 > 0 and e1 > 0:
            out.append(math.log(e0 / e1) / math.log(h0 / h1))
        else:
            out.append(float("nan"))
    return out


def _read_ladder(root, tag):
    path = os.path.join(root, tag, "ladder.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rungs = json.load(f)["rungs"]
    rungs.sort(key=lambda r: -r["h"])            # coarse -> fine
    return rungs


def _add(curves, key, label, rungs, h_ref, tag):
    h = [r["h"] for r in rungs]
    if h_ref is not None and h != h_ref:
        # Series on different ladders cannot share one abscissa; say so rather
        # than silently plotting against the wrong h.
        raise SystemExit(
            f"{tag} ladder {h} differs from the reference ladder {h_ref}")
    for norm in ("L2_Ham", "L2_Mom", "Linf_Ham", "Linf_Mom"):
        curves[f"{norm}_{key}"] = [r[norm] for r in rungs]
    curves[f"order_H_{key}"] = _local_orders(h, curves[f"L2_Ham_{key}"])
    curves[f"order_M_{key}"] = _local_orders(h, curves[f"L2_Mom_{key}"])
    curves[f"label_{key}"] = label
    print(f"  {tag:16s} L2(H) {curves[f'L2_Ham_{key}'][0]:.3e} -> "
          f"{curves[f'L2_Ham_{key}'][-1]:.3e}   local orders "
          + " ".join(f"{p:.2f}" for p in curves[f"order_H_{key}"]))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    args = ap.parse_args()
    root = _runs_root(args.runs)

    curves, have, h_ref = {}, {}, None
    for tag, key, label in SERIES:
        rungs = _read_ladder(root, tag)
        if rungs is None:
            print(f"  (missing: {tag})")
            have[key] = False
            continue
        have[key] = True
        h = _add(curves, key, label, rungs, h_ref, tag)
        if h_ref is None:
            h_ref, curves["h"] = h, h
            curves["N"] = [r["N"] for r in rungs]
    if not any(have.values()):
        raise SystemExit(f"no ladders found under {root}")

    # Resolution variants may sit on a shorter ladder (only the rungs where the
    # floor showed), so they get their own abscissa.
    variants = {}
    for tag, key, label in VARIANTS:
        rungs = _read_ladder(root, tag)
        if rungs is None:
            continue
        variants[key] = dict(
            label=label, h=[r["h"] for r in rungs], N=[r["N"] for r in rungs],
            L2_Ham=[r["L2_Ham"] for r in rungs],
            L2_Mom=[r["L2_Mom"] for r in rungs])
        print(f"  {tag:16s} (variant) L2(H) "
              + " ".join(f"{v:.3e}" for v in variants[key]["L2_Ham"]))

    # The AMR measurement is a single number per hierarchy depth, deliberately
    # NOT part of the convergence statement: an AMR L2 mixes refinement levels.
    # The per-level breakdown travels with it so the caption can quote the
    # finest spacing honestly.
    amr = []
    for tag in AMR_TAGS:
        rungs = _read_ladder(root, tag)
        if not rungs:
            continue
        r = rungs[0]
        levels = r.get("levels", [])
        rec = dict(tag=tag, N=r["N"], h_coarse=r["h"], L2_Ham=r["L2_Ham"],
                   L2_Mom=r["L2_Mom"], n_cells=r["n_cells"], levels=levels,
                   n_levels=len(levels),
                   dx_finest=min((lv["dx"] for lv in levels), default=None))
        amr.append(rec)
        print(f"  {tag:16s} L2(H) {rec['L2_Ham']:.3e} over "
              f"{rec['n_levels']} levels, finest dx = {rec['dx_finest']}")

    # Provenance.  This figure's caption states the box, the ladder, the
    # exclusion radius and the spectral grid, so they belong in the figdata and
    # not only in prose.
    probe_tag = next(t for t, k, _ in SERIES if have.get(k))
    probe = _read_ladder(root, probe_tag)[0]
    meta = dict(
        measured_by="GRTeclyn Constraints (4th-order stencils, CCZ4 variables)",
        norm=probe.get("norm"),
        box_half_width=9.0, L_full=18.0,
        r_excl=probe.get("r_excl"), border_cells=probe.get("border_cells"),
        b=3.0, m_A=0.5, m_B=0.5, P_anchor=0.5, P_small=0.1,
        spectral_grid=[52, 36], spectral_nphi=8,
        boost_guard="GRTeclyn analytic ID requires |P| < 0.3 m",
        has_tp=have.get("tp", False), has_by=have.get("by_p010", False),
        has_amr=bool(amr), has_variants=bool(variants),
        series={k: lab for _, k, lab in SERIES if have.get(k)},
        runs_root=os.path.abspath(root),
    )
    dump("fig10_constraints",
         dict(curves=curves, meta=meta, amr=amr, variants=variants))
    print("wrote figdata/fig10_constraints.json")


if __name__ == "__main__":
    main()
