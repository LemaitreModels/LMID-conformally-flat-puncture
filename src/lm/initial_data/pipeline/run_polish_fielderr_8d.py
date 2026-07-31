"""LM-initial-data — per-Newton-step FIELD ERROR during certified refinement (8-D).

The 8-D spin quasi-circular sibling of ``run_polish_fielderr.py`` (4-D): it
re-runs the certified NK polish over the IDENTICAL 1000 seed-0 off-node points of
the 8-D staircase, recording the FIELD at every Newton step, and reports the
relative-L2 field error ``||u^(k) - u_true||_2 / ||u_true||_2`` per step for the
two families the ``fig04`` staircase shows:

  * ``cold`` — cold NK from the zero field (U0=None), the surrogate-free start.
  * ``pod``  — warm start from the r=250 y-pair CROSS value+gradient POD guess.

It is byte-faithful to the 4-D driver except for the 8-D model wiring:

  * METADATA (box / grid / axes / fixed) is read from the shipped PLAIN 8-D model
    ``pod_hermite_smolyak_spin8qc_L5_enh-chi_Ax-...-chi_Bz.npz`` — the SAME model
    ``run_polish_cold.py --dim 8`` reads its metadata (and off-node points) from,
    so the 8-D field-error points match the residual staircase exactly.
  * The POD warm start is the 8-D y-pair CROSS POD (value + gradient in
    ``chi_Ay``/``chi_By`` + the mixed 2nd partial ``∂²U/∂χ_Ay∂χ_By``) TRUNCATED to
    rank 250 — the ``run_cross_pod_r250_8d.py`` prereq-build artifact
    ``pod_hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross_r250.npz``, loaded via
    the CROSS loader ``load_pod_hermite_smolyak_cross`` (the 8-D analog of the 4-D
    driver's ``pod_hermite_smolyak_d4qc..._cross_r75.npz`` warm start).

Step axes match the committed 8-D producers: ``cold`` = 8 steps
(``run_polish_cold.py --dim 8`` default ``--steps 8``); ``pod`` = 4 steps
(``run_polish_podrank.py`` hard-codes ``MAXSTEPS = 4``).  The residual recorded at
each step reproduces ``polish_cold_chi8d_1000.json`` /
``polish_table_chi8d_pod_r250_1000.json`` per-step, so the field-error staircase
shares their step axis.

Field error convention matches ``run_polish_fielderr.py`` (plain relative
Frobenius L2 against the best/converged iterate ``u_ref`` — NO separate certified
solve).

Writes ``reports/P3/polish_fielderr_chi8d_<n>.json`` (so the 1000-pt run writes
``polish_fielderr_chi8d_1000.json`` — the registry ``polish_fielderr_8d`` path)
with, per family, the per-step field-error stats (min/median/mean/p95/max) AND the
raw per-point arrays (for honest min--max whiskers), plus the reproduced per-step
residual stats for the shared-step cross-check.

Add-only.  Imports committed ``lm.initial_data`` modules read-only; edits nothing.

Run:
  python -m lm.initial_data.pipeline.run_polish_fielderr_8d                    # full 1000 pt
  python -m lm.initial_data.pipeline.run_polish_fielderr_8d --n-points 5       # smoke test
  python -m lm.initial_data.pipeline.run_polish_fielderr_8d --cold-steps 8 --pod-steps 4
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
from lm.initial_data.solver import solver_3d_nk as s3nk
from lm.initial_data.parametric.parametric_nd_3d import theta_to_slice3d
# the y-pair CROSS Hermite-Smolyak POD loader (value + grad in chi_Ay/chi_By +
# the mixed 2nd partial) — the 8-D gradient-enhanced warm start
from lm.initial_data.parametric.hermite_smolyak_pod_cross import load_pod_hermite_smolyak_cross
# metadata / box / grid / axes / fixed source (the SAME plain 8-D model
# run_polish_cold --dim 8 reads its metadata + off-node points from)
from lm.initial_data.pipeline.run_guess_vs_memory import MODELS
# reuse the EXACT seed-shared off-node sampling of the residual staircases
from lm.initial_data.pipeline.run_polish_cold import random_offnode_points, read_meta

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REPDIR = os.path.join(REPORTS, "P3")
MODELS_DIR = os.path.join(REPORTS, "P2", "models_chi")
# metadata source — the plain 8-D model (box/grid/axes/fixed + the shared points)
BASE_8D = MODELS[8]["pod"]
# the r=250 y-pair CROSS POD warm-start guess (built by run_cross_pod_r250_8d.py)
POD_R250_CROSS = os.path.join(
    MODELS_DIR, "pod_hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross_r250.npz")


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
    ap.add_argument("--cold-steps", type=int, default=8)   # match polish_cold_chi8d (--steps 8)
    ap.add_argument("--pod-steps", type=int, default=4)    # match polish_table_chi8d_pod_r250 (MAXSTEPS=4)
    args = ap.parse_args()
    os.makedirs(REPDIR, exist_ok=True)

    meta = read_meta(BASE_8D)
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    print(f"[fielderr] 8-D spin qc  d={len(box)}  grid={Na}x{Nb}x{Nphi}  "
          f"box={dict(zip(names, box))}  fixed={fixed}", flush=True)
    print(f"[fielderr] drawing {args.n_points} off-node points (seed={args.seed}) ...",
          flush=True)
    pts = random_offnode_points(box, args.n_points, args.seed)

    print(f"[fielderr] loading r=250 y-pair CROSS POD "
          f"({os.path.getsize(POD_R250_CROSS)/1e6:.0f} MB) ...", flush=True)
    pod = load_pod_hermite_smolyak_cross(POD_R250_CROSS)
    rank = int(pod.r)
    # sanity: same dimensionality as the sampling box
    assert pod.d == len(box), (pod.d, len(box))
    print(f"[fielderr] CROSS POD ready: d={pod.d}  r={rank}", flush=True)

    t0 = time.time()
    cold = run_family("cold", prob, names, fixed, pts, args.cold_steps,
                      guess_fn=lambda th: None)
    pod_res = run_family("pod", prob, names, fixed, pts, args.pod_steps,
                         guess_fn=lambda th: np.asarray(pod.evaluate(th)))

    out = {
        "config": {"tag": "chi8d", "dim": len(box), "n_points": args.n_points,
                   "seed": args.seed, "Na": Na, "Nb": Nb, "Nphi": Nphi,
                   "box": [{"name": n, "min": lo, "max": hi}
                           for n, (lo, hi) in zip(names, box)],
                   "fixed": fixed, "metric": "field_error_relL2",
                   "u_ref": "best (converged) NK iterate",
                   "pod_guess": os.path.basename(POD_R250_CROSS),
                   "pod_kind": "hermite_smolyak_cross (r=250, y-pair)",
                   "pod_rank": rank},
        "cold": cold,
        "pod": pod_res,
    }
    outp = os.path.join(REPDIR, f"polish_fielderr_chi8d_{args.n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"\n=== [chi8d] per-step FIELD ERROR over {args.n_points} off-node points ===")
    for name, fam in (("cold", cold), ("pod", pod_res)):
        keys = ["guess"] + [f"after{k}" for k in range(1, fam["max_steps"] + 1)]
        print(f"-- {name} (median field error) --  "
              + "  ".join(f"{k}:{fam['field_rows'][k]['median']:.2e}" for k in keys))
    print(f"[fielderr] DONE in {(time.time()-t0)/60:.1f} min -> {outp}", flush=True)


if __name__ == "__main__":
    main()
