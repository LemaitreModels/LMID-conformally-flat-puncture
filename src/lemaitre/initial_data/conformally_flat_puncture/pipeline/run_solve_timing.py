"""LM-initial-data — wall-clock cost of one certified solve, separable vs dense.

Why this exists
---------------
The paper certifies every datum it returns by a Newton--Krylov solve to
``CERT_TOL_U``.  That design is affordable only because the solve is cheap, and
it is cheap because of ``solver/separable.py``: each azimuthal-mode block of the
preconditioner is inverted by fast diagonalization of 1-D Kronecker factors built
once per grid, instead of a dense LU of the block rebuilt at every separation.
This producer measures both halves of that sentence on the production grid:

* the cold certified solve with the shipped (separable) route, and
* the same solve with the dense per-mode route (``separable=False``), so the
  ratio between them is a measurement and not an estimate.

Each solve starts from ``u = 0`` (no model guess), runs ``newton_solve_nk`` to
``CERT_TOL_U`` and is timed alone.  The first call of each route is made at a
point outside the sample and is excluded: it pays the JIT compile and, for the
separable route, the one-off construction of the 1-D factors.  The sample is the
first ``n`` points of fig04's off-node sample (``run_polish_cold.
random_offnode_points(box, n, seed=0)``).  Both routes must certify every point,
and their fields must agree to ``FIELD_TOL``; otherwise the exit status is 1.

A timing means something only on a quiet, exclusive node with one thread: set
``OMP_NUM_THREADS=1`` (and the BLAS equivalents) and
``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`` before Python starts.  The
output records the host, CPU and thread settings next to the numbers.

Run::

    python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.run_solve_timing \\
        --dim 4 --n 20 --out solve_timing_4d.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ..parametric import parametric_nd_3d as p3
from ..parametric.certification import CERT_TOL_U
from ..solver import operators_3d as ops3
from ..solver import solver_3d as s3
from ..solver import solver_3d_nk as s3nk
from . import production_box as pb
from . import production_model as pm
from .run_polish_cold import random_offnode_points

# Relative agreement demanded between the two routes' certified fields (the
# value ``run_separable_sweep`` uses, for the same reason: both land inside the
# certified ball by different Krylov paths, so bit agreement is not expected).
FIELD_TOL = 1e-9
WARM_SEED = 999     # the excluded first call, at a point outside the sample
MAX_ITER = 30       # Newton step budget; the loop stops at the gate


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            return next(l.split(":", 1)[1].strip() for l in f
                        if l.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def _solve(prob, sl, separable):
    t = time.perf_counter()
    U, info = s3nk.newton_solve_nk(prob, sl, tol=CERT_TOL_U, max_iter=MAX_ITER,
                                   separable=separable)
    wall = time.perf_counter() - t
    return np.asarray(U), dict(wall=wall, iters=int(info.iters),
                               gmres=int(np.sum(info.gmres_iters)),
                               resid=float(info.residual_norm),
                               certified=bool(info.converged
                                              and info.residual_norm <= CERT_TOL_U))


def _stats(a):
    a = np.asarray(a, float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(n=int(a.size), min=float(a.min()), q1=float(q1), median=float(med),
                q3=float(q3), max=float(a.max()))


def run(dim=4, n=20, routes=("separable", "dense"), verbose=True):
    box = pb.aligned_box() if dim == 4 else pb.spin8_box()
    names = [a["name"] for a in box]
    bx = [(a["min"], a["max"]) for a in box]
    prob = s3.make_problem(*pb.PROD_GRID, scheme=pm.SCHEME)
    fixed = dict(pb.FIXED_QC)
    slice_of = lambda th: p3.theta_to_slice3d(th, names, 1.0, fixed)

    pts = random_offnode_points(bx, n, 0)
    th0 = random_offnode_points(bx, 1, WARM_SEED)[0]
    out = dict(dim=dim, grid=list(pb.PROD_GRID), scheme=pm.SCHEME, tol=CERT_TOL_U,
               axes=names, n=n, first_call={}, rows=[], summary={})
    fields = {}
    for route in routes:
        sep = route == "separable"
        ops3.clear_block_cache()
        _, out["first_call"][route] = _solve(prob, slice_of(th0), sep)
        fields[route] = []
        for i, th in enumerate(pts):
            if not sep:
                ops3.clear_block_cache()    # every new b rebuilds the dense blocks
            U, row = _solve(prob, slice_of(th), sep)
            fields[route].append(U)
            if len(out["rows"]) <= i:
                out["rows"].append(dict(i=i, theta=th.tolist()))
            out["rows"][i][route] = row
            if verbose:
                print(f"  {route:9s} [{i + 1:3d}/{n}] {1e3 * row['wall']:9.1f} ms  "
                      f"{row['iters']} Newton  {row['gmres']:3d} GMRES  "
                      f"R = {row['resid']:.1e}", flush=True)
        walls = [r[route]["wall"] for r in out["rows"]]
        out["summary"][route] = dict(
            wall=_stats(walls),
            iters=_stats([r[route]["iters"] for r in out["rows"]]),
            all_certified=all(r[route]["certified"] for r in out["rows"]))
    if len(routes) == 2:
        diff = [float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
                for a, b in zip(fields["separable"], fields["dense"])]
        ratio = [r["dense"]["wall"] / r["separable"]["wall"] for r in out["rows"]]
        out["summary"]["field_diff_max"] = max(diff)
        out["summary"]["dense_over_separable"] = _stats(ratio)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dim", type=int, choices=(4, 8), default=4)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--routes", default="separable,dense")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a.dim, a.n, tuple(a.routes.split(",")))
    res["env"] = dict(
        host=platform.node(), cpu=_cpu_model(), python=platform.python_version(),
        jax=jax.__version__, slurm_job=os.environ.get("SLURM_JOB_ID"),
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                "OPENBLAS_NUM_THREADS", "XLA_FLAGS")},
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    s = res["summary"]
    for route in [r for r in ("separable", "dense") if r in s]:
        w = s[route]["wall"]
        print(f"{route:9s}: median {1e3 * w['median']:.1f} ms "
              f"(IQR {1e3 * w['q1']:.1f}-{1e3 * w['q3']:.1f}), "
              f"all certified: {s[route]['all_certified']}")
    ok = all(s[r]["all_certified"] for r in ("separable", "dense") if r in s)
    if "dense_over_separable" in s:
        print(f"dense/separable: median {s['dense_over_separable']['median']:.0f}x; "
              f"field diff max {s['field_diff_max']:.1e}")
        ok = ok and s["field_diff_max"] <= FIELD_TOL
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
