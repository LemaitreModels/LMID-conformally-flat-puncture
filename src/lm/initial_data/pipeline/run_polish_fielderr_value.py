"""LM-initial-data — per-Newton-step FIELD ERROR for the VALUE-ONLY warm start (4-D + 8-D).

Companion to ``run_polish_fielderr.py`` (which records the ``cold`` and value+gradient
``pod`` families).  fig04's bottom row compares the field error per Newton step of the
warm starts against the cold start; the committed field-error files carry ``cold`` and
``pod`` (value+gradient) only.  This script adds the missing third family:

  * ``value`` — warm start from the value-ONLY Smolyak surrogate (no gradient
    enhancement), the same model whose residual staircase is
    ``polish_table_qc_chi_prod_1000.json`` (4-D) / ``polish_table_chi8d_value_1000.json``
    (8-D).

It re-uses the committed field-error machinery VERBATIM — ``polish_history`` (the
instrumented NK loop that stores the field at every step) and ``run_family`` from
``run_polish_fielderr`` — over the IDENTICAL 1000 seed-0 off-node points
(``random_offnode_points``), so the value staircase shares the cold/pod step axis and
whiskers.  The only new ingredient is the value-only guess ``model.evaluate(theta)``.

The recorded per-step residual staircase (``residual_rows``) reproduces the committed
value residual tables — a built-in cross-check that the model and the seed-0 sampling
match the fig04 top row.

Add-only.  Imports committed ``lm.initial_data`` modules read-only; edits nothing.

Run:
  python -m lm.initial_data.pipeline.run_polish_fielderr_value --dim 4              # full 1000 pt
  python -m lm.initial_data.pipeline.run_polish_fielderr_value --dim 8
  python -m lm.initial_data.pipeline.run_polish_fielderr_value --dim 4 --n-points 5 # smoke test
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
from lm.initial_data.parametric.parametric_nd_3d import theta_to_slice3d
from lm.initial_data.parametric.parametric_nd_smolyak import load_smolyak
# reuse the EXACT seed-shared off-node sampling and instrumented polish of the cold/pod families
from lm.initial_data.pipeline.run_polish_cold import random_offnode_points, read_meta
from lm.initial_data.pipeline.run_polish_fielderr import run_family

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REPDIR = os.path.join(REPORTS, "P3")
MODELS = os.path.join(REPORTS, "3D_parametric", "models_chi")
# the value-ONLY Smolyak surrogates (no gradient enhancement), per dimension
VALUE_MODEL = {
    4: os.path.join(MODELS, "surrogate_smolyak_d4_qc_chi_prod_L5.npz"),
    8: os.path.join(MODELS, "surrogate_smolyak_spin8_qc_chi_prod_L5.npz"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=(4, 8), required=True)
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-steps", type=int, default=4)   # match the pod warm-start step axis
    args = ap.parse_args()
    os.makedirs(REPDIR, exist_ok=True)

    model_path = VALUE_MODEL[args.dim]
    meta = read_meta(model_path)
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    print(f"[fielderr-value] {args.dim}-D qc  d={len(box)}  grid={Na}x{Nb}x{Nphi}  "
          f"box={dict(zip(names, box))}  fixed={fixed}", flush=True)
    print(f"[fielderr-value] drawing {args.n_points} off-node points (seed={args.seed}) ...",
          flush=True)
    pts = random_offnode_points(box, args.n_points, args.seed)

    print(f"[fielderr-value] loading value-only Smolyak surrogate "
          f"{os.path.basename(model_path)} ({os.path.getsize(model_path)/1e6:.0f} MB) ...",
          flush=True)
    model = load_smolyak(model_path)

    t0 = time.time()
    value = run_family("value", prob, names, fixed, pts, args.value_steps,
                       guess_fn=lambda th: np.asarray(model.evaluate(th)))

    out = {
        "config": {"tag": f"chi{args.dim}d", "dim": len(box), "n_points": args.n_points,
                   "seed": args.seed, "Na": Na, "Nb": Nb, "Nphi": Nphi,
                   "box": [{"name": n, "min": lo, "max": hi}
                           for n, (lo, hi) in zip(names, box)],
                   "fixed": fixed, "metric": "field_error_relL2",
                   "u_ref": "best (converged) NK iterate",
                   "value_guess": os.path.basename(model_path)},
        "value": value,
    }
    outp = os.path.join(REPDIR, f"polish_fielderr_value_chi{args.dim}d_{args.n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)

    keys = ["guess"] + [f"after{k}" for k in range(1, value["max_steps"] + 1)]
    print(f"\n=== [chi{args.dim}d] value-only per-step FIELD ERROR over "
          f"{args.n_points} off-node points ===")
    print("-- value (median field error) --  "
          + "  ".join(f"{k}:{value['field_rows'][k]['median']:.2e}" for k in keys))
    print("-- value (median residual, cross-check vs value table) --  "
          + "  ".join(f"{k}:{value['residual_rows'][k]['median']:.2e}" for k in keys))
    print(f"[fielderr-value] DONE in {(time.time()-t0)/60:.1f} min -> {outp}", flush=True)


if __name__ == "__main__":
    main()
