"""PARASOL — per-Newton-step FIELD ERROR during certified refinement (4-D).

Companion to ``run_polish_cold.py`` / ``run_polish_table.py``, which record the
constraint residual ``||R||_inf`` at the guess (step 0) and after 1..K
Newton--Krylov steps.  Those runs do NOT store the field iterates, so the field
error ``||u^(k) - u_true||_2 / ||u_true||_2`` per step is unavailable from disk.

This script re-runs the polish for the 4-D quasi-circular model over the IDENTICAL
1000 seed-0 off-node points, recording the FIELD at every Newton step, and reports
the relative-L2 field error per step for the two families the fig01 staircase shows:

  * ``cold`` — cold NK from the zero field (U0=None), the surrogate-free start.
  * ``pod``  — warm start from the shipped r=75 value+gradient (cross) POD guess.

It faithfully replicates ``solver_3d_nk.newton_solve_nk``'s Newton loop (same
equilibrated-residual convergence + stagnation break) using the committed
primitives ``newton_step_nk`` / ``equil_residual_inf`` / ``_block_scales`` — the
only difference being that it also stores ``U`` at each step so the field error
vs the converged (best) iterate can be formed.  The residual it records must (and
does) reproduce ``polish_cold_chi4d_1000.json`` / ``polish_table_chi4d_pod_r75_
cross_1000.json`` per-step, so the field-error staircase shares their step axis.

Field error convention matches ``run_cross_fielderror_chi.field_err`` and the
``guess_vs_memory_4d_field_1000.json`` sweep: plain relative Frobenius L2 against
the certified solve (here the best/converged iterate ``u_ref``).

Writes ``reports/P3/polish_fielderr_chi4d_<n>.json`` with, per family, the
per-step field-error stats (min/median/mean/p95/max) AND the raw per-point arrays
(so the figure builder can draw honest min--max whiskers), plus the reproduced
per-step residual stats for the shared-step cross-check.

Add-only.  Imports committed ``parasol`` modules read-only; edits nothing.

Run:
  python sandbox/parasol/run_polish_fielderr.py                       # full 1000 pt
  python sandbox/parasol/run_polish_fielderr.py --n-points 5          # smoke test
  python sandbox/parasol/run_polish_fielderr.py --cold-steps 8 --pod-steps 4
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


from lemaitre.initial_data.solver import solver_3d as s3
from lemaitre.initial_data.solver import solver_3d_nk as s3nk
from lemaitre.initial_data.parametric.parametric_nd_3d import theta_to_slice3d
from lemaitre.initial_data.parametric.hermite_smolyak_pod_cross import load_pod_hermite_smolyak_cross
# reuse the EXACT seed-shared off-node sampling of the residual staircases
from lemaitre.initial_data.pipeline.run_polish_cold import random_offnode_points, read_meta

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "P3")
MODELS = os.path.join(HERE, "reports", "P2", "models_chi")
# box / grid / axes / fixed source (metadata only) — the same 4-D model run_polish_cold uses
BASE_4D = os.path.join(MODELS, "pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By.npz")
# the shipped r=75 value+gradient (full-bilinear CROSS) POD warm-start guess
POD_R75_CROSS = os.path.join(MODELS, "pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r75.npz")


def polish_history(prob, sl, U0, max_steps, tol=1e-12):
    """Instrumented NK polish: faithfully replicate ``newton_solve_nk``'s loop
    (equilibrated-residual convergence + stagnation break) while ALSO storing the
    field ``U`` at every step.  Returns (residuals[0..max_steps],
    field_error[0..max_steps]); field error is relative Frobenius L2 vs the best
    (converged) iterate ``u_ref``.  Both lists are padded with their last value
    once the solve has converged/stagnated (matching the residual-staircase
    padding in run_polish_cold / run_polish_table)."""
    asm = s3.assemble(prob, sl)
    scales = s3nk._block_scales(asm)
    if U0 is None:
        U = np.zeros((prob.Ntot2d, prob.Nphi))
    else:
        U = np.asarray(U0, dtype=float).reshape(prob.Ntot2d, prob.Nphi)

    fields, hist = [], []
    best_U, best_rn = U, np.inf
    for it in range(1, max_steps + 2):          # store U_0..U_{max_steps}
        rn = s3nk.equil_residual_inf(asm, U, scales)
        hist.append(float(rn))
        fields.append(U.copy())
        if rn < best_rn:
            best_U, best_rn = U.copy(), rn
        if rn < tol:
            break
        if it >= 3 and rn > 0.5 * hist[-2]:     # stagnation near the floor
            break
        if it <= max_steps:                     # no wasted step past the last stored iterate
            U, _ = s3nk.newton_step_nk(asm, U, gmres_rtol=1e-4)

    while len(hist) < max_steps + 1:            # pad to the full step axis
        hist.append(hist[-1])
        fields.append(fields[-1])

    uref_norm = float(np.linalg.norm(best_U))
    ferr = [float(np.linalg.norm(f - best_U) / max(uref_norm, 1e-300)) for f in fields]
    return hist, ferr


def _stats(a):
    a = np.asarray(a, float)
    return dict(min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), p95=float(np.percentile(a, 95)),
                max=float(a.max()))


def run_family(name, prob, names, fixed, pts, max_steps, guess_fn):
    """guess_fn(theta) -> U0 (None for cold).  Returns the family result dict."""
    n = len(pts)
    F = [[] for _ in range(max_steps + 1)]      # field error per step
    R = [[] for _ in range(max_steps + 1)]      # residual per step (cross-check)
    t0 = time.time()
    for i, theta in enumerate(pts):
        sl = theta_to_slice3d(np.asarray(theta, float), names, 1.0, fixed)
        U0 = guess_fn(theta)
        hist, ferr = polish_history(prob, sl, U0, max_steps)
        for k in range(max_steps + 1):
            R[k].append(hist[k])
            F[k].append(ferr[k])
        if (i + 1) % 50 == 0 or i == n - 1:
            el = time.time() - t0
            print(f"   [{name}] {i+1}/{n}  ({el:.0f}s, {el/(i+1):.2f}s/pt, "
                  f"ETA {el/(i+1)*(n-i-1):.0f}s)  "
                  f"med field-err@guess={np.median(F[0]):.2e} "
                  f"@last={np.median(F[max_steps]):.2e}", flush=True)

    keys = ["guess"] + [f"after{k}" for k in range(1, max_steps + 1)]
    F = [np.array(x) for x in F]
    R = [np.array(x) for x in R]
    return {
        "max_steps": max_steps,
        "field_rows": {keys[k]: _stats(F[k]) for k in range(max_steps + 1)},
        "field_error": {keys[k]: [float(x) for x in F[k]] for k in range(max_steps + 1)},
        "residual_rows": {keys[k]: _stats(R[k]) for k in range(max_steps + 1)},
        "wall_clock_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold-steps", type=int, default=8)   # match polish_cold_chi4d
    ap.add_argument("--pod-steps", type=int, default=4)    # match polish_table_chi4d_pod_r75_cross
    args = ap.parse_args()
    os.makedirs(REPDIR, exist_ok=True)

    meta = read_meta(BASE_4D)
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    print(f"[fielderr] 4-D qc  d={len(box)}  grid={Na}x{Nb}x{Nphi}  "
          f"box={dict(zip(names, box))}  fixed={fixed}", flush=True)
    print(f"[fielderr] drawing {args.n_points} off-node points (seed={args.seed}) ...",
          flush=True)
    pts = random_offnode_points(box, args.n_points, args.seed)

    print(f"[fielderr] loading r=75 cross POD guess "
          f"({os.path.getsize(POD_R75_CROSS)/1e6:.0f} MB) ...", flush=True)
    pod = load_pod_hermite_smolyak_cross(POD_R75_CROSS)
    # sanity: same axis order/box as the sampling
    assert list(pod.axis_names) == names if hasattr(pod, "axis_names") else True

    t0 = time.time()
    cold = run_family("cold", prob, names, fixed, pts, args.cold_steps,
                      guess_fn=lambda th: None)
    pod_res = run_family("pod", prob, names, fixed, pts, args.pod_steps,
                         guess_fn=lambda th: np.asarray(pod.evaluate(th)))

    out = {
        "config": {"tag": "chi4d", "dim": len(box), "n_points": args.n_points,
                   "seed": args.seed, "Na": Na, "Nb": Nb, "Nphi": Nphi,
                   "box": [{"name": n, "min": lo, "max": hi}
                           for n, (lo, hi) in zip(names, box)],
                   "fixed": fixed, "metric": "field_error_relL2",
                   "u_ref": "best (converged) NK iterate",
                   "pod_guess": os.path.basename(POD_R75_CROSS)},
        "cold": cold,
        "pod": pod_res,
    }
    outp = os.path.join(REPDIR, f"polish_fielderr_chi4d_{args.n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"\n=== [chi4d] per-step FIELD ERROR over {args.n_points} off-node points ===")
    for name, fam in (("cold", cold), ("pod", pod_res)):
        keys = ["guess"] + [f"after{k}" for k in range(1, fam["max_steps"] + 1)]
        print(f"-- {name} (median field error) --  "
              + "  ".join(f"{k}:{fam['field_rows'][k]['median']:.2e}" for k in keys))
    print(f"[fielderr] DONE in {(time.time()-t0)/60:.1f} min -> {outp}", flush=True)


if __name__ == "__main__":
    main()
