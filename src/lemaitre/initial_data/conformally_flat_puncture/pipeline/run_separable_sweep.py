"""LM-initial-data — is the separable preconditioner safe across the production box?

Why this exists
---------------
``solver/separable.py`` replaces the per-mode Newton preconditioner: instead of
LU-factoring the linear block plus its nonlinear diagonal once per Newton step, it
inverts the *linear* block exactly by fast diagonalization of the block's 1-D
Kronecker factors, built once per grid.  Dropping the nonlinear diagonal is
precisely what makes the setup independent of the separation, of the physical
parameters and of the iterate — and it is also the one thing that could cost
Krylov iterations, or, worse, an extra Newton step at a hard corner of the box.

The exactness claims are unit-tested (``tests/test_solver_3d_fast.py``) and hold
pointwise, which is not the same question.  This producer asks the *statistical*
one that a default flip actually rests on: over the real 8-D production box, does
the cheaper preconditioner ever change the answer, fail to certify, or cost a
Newton step?

Every point is solved TWICE from the SAME warm start — once with each linear
algebra — through the shipped forward map ``parametric_nd_3d.make_solve_fn``, not
a reimplementation, so what is measured is what production runs.  The gate is
``parametric.certification.CERT_TOL``, imported rather than restated.

What "pass" means
-----------------
* every point certifies on BOTH routes (a point that fails on both is a solver
  limitation, not a preconditioner one — it is reported as such, not hidden);
* the separable route costs **zero** extra Newton steps (GMRES iterations may and
  do rise by ~1 per step; that is the trade being made);
* the two converged fields agree to ``FIELD_TOL`` relative — far below the
  interpolation error the surrogate is built to, so the choice of preconditioner
  is invisible downstream.

Run::

    python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.run_separable_sweep \\
        --n 24 --out sweep.json

Exit status is 0 only on a clean pass, so it can gate a cluster job.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from ..parametric import parametric_nd_3d as p3
from ..parametric.certification import CERT_TOL
from ..solver import operators_3d as ops3
from ..solver import solver_3d as s3
from . import production_box as pb

# The quasi-circular branch of ``theta_to_slice3d``: a flag, not an axis.  Without
# it the momenta are a free scan rather than the 3PN quasi-circular series, and the
# sweep would not be sampling the family the model is built over.
QC = {"qc": 1.0}

# Relative field agreement demanded between the two routes.  Set ~3 decades below
# the certified residual's own floor rather than at machine precision: the two
# routes take different Krylov paths to the same solution, so they land at
# different points inside the certified ball, and demanding bit-agreement would be
# demanding something untrue.
FIELD_TOL = 1e-9

# The warm start for each point: the same configuration at a slightly wider
# separation.  A real interpolant guess is closer than this, so this is the
# pessimistic end of what a polish is asked to fix.
WARM_B_FACTOR = 1.08


def sweep(n=24, grid=None, seed=20260813, newton_steps=4, verbose=True):
    """Solve ``n`` random production-box points by both routes; return the rows."""
    Na, Nb, Nphi = grid or pb.PROD_GRID
    box = pb.spin8_box()
    names = [a["name"] for a in box]
    lo = np.array([a["min"] for a in box])
    hi = np.array([a["max"] for a in box])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    if verbose:
        print(f"host   : {platform.node()}  {platform.machine()}")
        print(f"grid   : ({Na}, {Nb}, {Nphi})   CERT_TOL = {CERT_TOL:.0e}   "
              f"FIELD_TOL = {FIELD_TOL:.0e}")
        print(f"box    : b in [{lo[0]:g}, {hi[0]:g}], q in [{lo[1]:g}, {hi[1]:g}], "
              f"chi_* in [{lo[2]:g}, {hi[2]:g}]  (3PN quasi-circular momenta)")
        print(f"points : {n} (seed {seed}); warm start = same point at "
              f"{WARM_B_FACTOR}x b\n")

    rng = np.random.default_rng(seed)
    thetas = lo + (hi - lo) * rng.random((n, len(box)))

    fns = {}
    for label, sep in (("dense", False), ("separable", True)):
        fns[label], _ = p3.make_solve_fn(prob, names, fixed=QC, solver="nk",
                                         separable=sep)

    if verbose:
        print(f"{'#':>3s} {'b':>5s} {'q':>5s} {'|chiA|':>6s} {'|chiB|':>6s}  "
              f"{'dense  steps gmres      equilR':>34s}  "
              f"{'separable  steps gmres      equilR':>34s}  "
              f"{'dU/|U|':>9s} {'x':>5s}")
    rows = []
    for i, th in enumerate(thetas):
        th_near = th.copy()
        th_near[0] = min(th[0] * WARM_B_FACTOR, hi[0])
        U0, _ = fns["dense"](th_near, None, CERT_TOL, 8)
        U0 = np.asarray(U0)

        out = {}
        for label in ("dense", "separable"):
            # a cold operator cache each time, so the timing is the honest
            # per-parameter-point cost and neither route is credited with the
            # other's warm cache
            ops3.clear_block_cache()
            t0 = time.perf_counter()
            U, info = fns[label](th, U0, CERT_TOL, newton_steps)
            out[label] = (np.asarray(U), info, time.perf_counter() - t0)

        Ud, id_, td = out["dense"]
        Us, is_, ts = out["separable"]
        d = float(np.max(np.abs(Us - Ud)) / max(np.max(np.abs(Ud)), 1e-300))
        row = dict(
            theta=[float(x) for x in th],
            chiA=float(np.linalg.norm(th[2:5])), chiB=float(np.linalg.norm(th[5:8])),
            field_rel_diff=d)
        for label, (_, info, t) in out.items():
            row[label] = dict(steps=int(info.iters), gmres=list(info.gmres_iters),
                              equilR=float(info.residual_norm),
                              certified=bool(info.residual_norm <= CERT_TOL),
                              seconds=float(t))
        rows.append(row)
        if verbose:
            print(f"{i:3d} {th[0]:5.2f} {th[1]:5.2f} {row['chiA']:6.3f} "
                  f"{row['chiB']:6.3f}  "
                  f"{id_.iters:5d} {str(id_.gmres_iters):>12s} "
                  f"{id_.residual_norm:11.3e}  "
                  f"{is_.iters:5d} {str(is_.gmres_iters):>12s} "
                  f"{is_.residual_norm:11.3e}  {d:9.2e} "
                  f"{td / max(ts, 1e-9):5.1f}")
    return rows


def report(rows):
    """Print the summary and return ``(ok, summary_dict)``."""
    def col(label, f):
        return [f(r[label]) for r in rows]

    n = len(rows)
    cert = {k: sum(r[k]["certified"] for r in rows) for k in ("dense", "separable")}
    steps = {k: col(k, lambda x: x["steps"]) for k in ("dense", "separable")}
    gm = {k: [g for r in rows for g in r[k]["gmres"] if g] for k in ("dense", "separable")}
    secs = {k: col(k, lambda x: x["seconds"]) for k in ("dense", "separable")}
    worst_R = {k: max(col(k, lambda x: x["equilR"])) for k in ("dense", "separable")}
    dd = [r["field_rel_diff"] for r in rows]
    # count only points where the separable route is WORSE; a point that fails on
    # both routes is a solver limitation and is reported separately
    extra = sum(1 for r in rows if r["separable"]["steps"] > r["dense"]["steps"])
    both_failed = [i for i, r in enumerate(rows)
                   if not r["dense"]["certified"] and not r["separable"]["certified"]]
    sep_only = [i for i, r in enumerate(rows)
                if r["dense"]["certified"] and not r["separable"]["certified"]]

    def spread(values):
        return f"{min(values)}/{np.median(values):.1f}/{max(values)}"

    print(f"\n{'':26s}{'dense':>16s} {'separable':>16s}")
    print(f"{'certified / points':26s}{cert['dense']:>11d}/{n:<4d} "
          f"{cert['separable']:>11d}/{n:<4d}")
    for name, d in (("Newton steps min/med/max", steps),
                    ("GMRES/step   min/med/max", gm)):
        print(f"{name:26s}{spread(d['dense']):>16s} {spread(d['separable']):>16s}")
    print(f"{'worst certified residual':26s}{worst_R['dense']:>16.3e} "
          f"{worst_R['separable']:>16.3e}")
    print(f"{'median seconds / point':26s}{np.median(secs['dense']):>15.3f}s "
          f"{np.median(secs['separable']):>15.3f}s   "
          f"({np.median(secs['dense']) / max(np.median(secs['separable']), 1e-9):.1f}x)")

    print(f"\nextra Newton steps the separable preconditioner costs : {extra}/{n}")
    print(f"worst |U_sep - U_dense| / max|U| over the box         : {max(dd):.2e}")
    if both_failed:
        print(f"points that failed to certify on BOTH routes (a solver limit, "
              f"not a preconditioner one): {both_failed}")
    if sep_only:
        print(f"!! points that certify DENSE but not SEPARABLE: {sep_only}")

    ok = (not sep_only and extra == 0 and max(dd) < FIELD_TOL
          and cert["separable"] == cert["dense"])
    print(f"\nVERDICT: {'PASS' if ok else 'REVIEW'} — the separable route must "
          f"certify wherever the dense one does, cost no Newton step, and agree "
          f"to {FIELD_TOL:.0e}")
    summary = dict(n=n, certified=cert, extra_newton_steps=extra,
                   worst_field_rel_diff=max(dd), worst_residual=worst_R,
                   median_seconds={k: float(np.median(v)) for k, v in secs.items()},
                   both_failed=both_failed, separable_only_failures=sep_only)
    return ok, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--Na", type=int, default=pb.PROD_GRID[0])
    ap.add_argument("--Nb", type=int, default=pb.PROD_GRID[1])
    ap.add_argument("--Nphi", type=int, default=pb.PROD_GRID[2])
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--newton-steps", type=int, default=4)
    ap.add_argument("--out", default=None, help="write the full per-point record here")
    args = ap.parse_args()

    rows = sweep(n=args.n, grid=(args.Na, args.Nb, args.Nphi), seed=args.seed,
                 newton_steps=args.newton_steps)
    ok, summary = report(rows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(meta=dict(grid=[args.Na, args.Nb, args.Nphi],
                                     seed=args.seed, cert_tol=CERT_TOL,
                                     field_tol=FIELD_TOL, host=platform.node()),
                           summary=summary, rows=rows, verdict=bool(ok)), f, indent=2)
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
