"""LM-initial-data — per-Newton-step FIELD ERROR + residual for the VALUE-ONLY *POD* warm start.

Companion to ``run_polish_fielderr_value.py`` (which warm-starts from the FULL, un-
compressed value-only Smolyak surrogate).  This script warm-starts from the value-only
surrogate *POD-compressed to a chosen rank* ``r`` — the apples-to-apples partner of the
value+gradient POD staircase (``run_polish_podrank.py``): the two share the IDENTICAL
shipped spatial basis ``Phi[:, :r]`` and the SAME rank, differing only in whether the
per-node coefficient interpolation uses the certified tangents (value+gradient) or not
(value-only).  Used to add a "value-only POD" curve to fig04 alongside the
value+gradient POD curve at the same rank (r=75 in 4D, r=250 in 8D).

Construction (verified): the value-only POD is the shipped gradient-enhanced POD model
truncated to rank ``r`` (a cheap ``Phi``/coeff slice — NO re-solve, NO corpus) but with
``enhanced_axes=()`` so the coeff-space interpolant is plain value-only Smolyak
(no Hermite tangent terms).  Its guess is ``mean + Phi[:, :r] @ c_value(theta)``.

It reuses the committed field-error machinery VERBATIM — ``run_polish_fielderr.run_family``
(the instrumented NK loop ``polish_history`` that stores the field at every step and
records BOTH the field error and the equilibrated-residual staircase) — over the IDENTICAL
1000 seed-0 off-node points (``run_polish_cold.random_offnode_points``), so the value-POD
staircase shares the cold / value+gradient-POD step axis and whiskers.  A single run yields
both fig04 rows: ``residual_rows`` (top) and ``field_rows`` (bottom).

Writes ``reports/P3/polish_fielderr_value_pod_chi<dim>d_r<r>_<n>.json`` with a single
``value_pod`` family (``field_rows`` + ``residual_rows``, ``run_polish_fielderr`` schema).

Add-only.  Imports committed ``lm.initial_data`` modules read-only; edits nothing.

Run:
  python -m lm.initial_data.pipeline.run_polish_fielderr_value_pod --dim 4 --rank 75             # full 1000 pt
  python -m lm.initial_data.pipeline.run_polish_fielderr_value_pod --dim 8 --rank 250
  python -m lm.initial_data.pipeline.run_polish_fielderr_value_pod --dim 4 --rank 75 --n-points 3 # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lm.initial_data.solver import solver_3d as s3
# reuse the EXACT seed-shared off-node sampling and instrumented polish of the cold/pod families
from lm.initial_data.pipeline.run_polish_cold import random_offnode_points, read_meta
from lm.initial_data.pipeline.run_polish_fielderr import run_family
# low-level POD loader primitives (committed, read-only) + the shipped Hermite POD paths
from lm.initial_data.parametric.parametric_nd_smolyak import _node_key
from lm.initial_data.parametric.hermite_smolyak import HermiteSmolyakSolverND
from lm.initial_data.parametric.hermite_smolyak_pod import PODHermiteSmolyak
from lm.initial_data.pipeline.run_guess_vs_memory import (
    MODELS, _load_npz, _unpack_meta, _check_meta,
)

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REPDIR = os.path.join(REPORTS, "P3")


def load_pod_truncated_value(path, r_new) -> PODHermiteSmolyak:
    """The shipped gradient-enhanced POD model truncated to rank ``r_new`` but with
    ``enhanced_axes=()`` — a value-only Smolyak interpolation of the value POD
    coefficients (drops the Hermite tangent terms).  Identical to
    ``run_guess_vs_memory.load_pod_truncated`` EXCEPT for the empty enhanced set, so it
    keeps the IDENTICAL shared basis ``Phi[:, :r_new]``, mean, nodes and rank; the only
    difference from the value+gradient POD is the tangent-free coefficient interpolant.
    NO re-solve, NO corpus."""
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "pod_hermite_smolyak")
    r_full = int(data["r"])
    r_new = int(min(r_new, r_full))
    Phi = np.asarray(data["Phi"], dtype=float)[:, :r_new]
    mean = np.asarray(data["mean"], dtype=float)
    field_shape = tuple(int(x) for x in np.asarray(data["field_shape"], dtype=np.int64))
    node_thetas = np.asarray(data["node_thetas"], dtype=float)
    node_U = np.asarray(data["node_U"], dtype=float)[:, :r_new]
    node_dU = np.asarray(data["node_dU"], dtype=float)[:, :, :r_new]
    node_iters = np.asarray(data["node_iters"])
    node_resids = np.asarray(data["node_resids"], dtype=float)
    index_set = [tuple(int(x) for x in row) for row in np.asarray(data["index_set"])]
    axes = [(float(a[0]), float(a[1])) for a in np.asarray(data["axes"], dtype=float)]
    pool = {}
    for i in range(node_thetas.shape[0]):
        pool[_node_key(node_thetas[i])] = (
            np.asarray(node_U[i], dtype=float), np.asarray(node_dU[i], dtype=float),
            int(node_iters[i]), float(node_resids[i]))
    solver = HermiteSmolyakSolverND(solve_fn=None, axes=axes, tangent_fn=None,
                                    enhanced_axes=())            # <-- value-only interpolant
    pod = PODHermiteSmolyak(solver._finalize(index_set, pool), Phi, mean, field_shape)
    pod.meta = meta
    return pod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=(4, 8), required=True)
    ap.add_argument("--rank", type=int, required=True,
                    help="POD truncation rank (fig04 uses 75 in 4D, 250 in 8D)")
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4)    # match the value+gradient-POD step axis
    args = ap.parse_args()
    os.makedirs(REPDIR, exist_ok=True)
    t0 = time.time()

    path = MODELS[args.dim]["pod"]
    meta = read_meta(path)
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    print(f"[fielderr-value-pod] {args.dim}-D qc  d={len(box)}  grid={Na}x{Nb}x{Nphi}  "
          f"box={dict(zip(names, box))}  fixed={fixed}", flush=True)
    print(f"[fielderr-value-pod] loading + truncating shipped POD "
          f"({os.path.getsize(path)/1e6:.0f} MB) to r={args.rank}, enhanced=() ...",
          flush=True)
    model = load_pod_truncated_value(path, args.rank)
    r_eff = int(model.r)
    r_ship = int(np.load(path, allow_pickle=True, mmap_mode="r")["r"])
    print(f"[fielderr-value-pod] r_shipped={r_ship} -> r={r_eff}  enhanced={model.enhanced}",
          flush=True)

    print(f"[fielderr-value-pod] drawing {args.n_points} off-node points (seed={args.seed}) ...",
          flush=True)
    pts = random_offnode_points(box, args.n_points, args.seed)

    value_pod = run_family("value_pod", prob, names, fixed, pts, args.steps,
                           guess_fn=lambda th: np.asarray(model.evaluate(th)))

    out = {
        "config": {"tag": f"chi{args.dim}d_value_pod_r{r_eff}", "dim": len(box),
                   "n_points": args.n_points, "seed": args.seed,
                   "Na": Na, "Nb": Nb, "Nphi": Nphi,
                   "box": [{"name": n, "min": lo, "max": hi}
                           for n, (lo, hi) in zip(names, box)],
                   "fixed": fixed, "metric": "field_error_relL2",
                   "u_ref": "best (converged) NK iterate",
                   "r": r_eff, "r_shipped": r_ship, "enhanced": [],
                   "model_file": os.path.basename(path),
                   "max_steps": args.steps},
        "value_pod": value_pod,
    }
    outp = os.path.join(REPDIR,
                        f"polish_fielderr_value_pod_chi{args.dim}d_r{r_eff}_{args.n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)

    keys = ["guess"] + [f"after{k}" for k in range(1, value_pod["max_steps"] + 1)]
    print(f"\n=== [chi{args.dim}d] value-only POD (r={r_eff}) per-step over "
          f"{args.n_points} off-node points ===")
    print("-- median FIELD error --   "
          + "  ".join(f"{k}:{value_pod['field_rows'][k]['median']:.2e}" for k in keys))
    print("-- median RESIDUAL   --   "
          + "  ".join(f"{k}:{value_pod['residual_rows'][k]['median']:.2e}" for k in keys))
    print(f"[fielderr-value-pod] DONE in {(time.time()-t0)/60:.1f} min -> {outp}", flush=True)


if __name__ == "__main__":
    main()
