"""LM-initial-data — value+gradient POD certified-refinement staircase at a CHOSEN rank r.

Companion to ``run_polish_table.py``: runs the identical 1000-point certified-
refinement sweep, but on the shipped gradient-enhanced (Hermite) POD model
*truncated* to a smaller rank ``r`` (a cheap slice of ``Phi`` / coeffs — NO
re-solve, NO corpus), so the warm-start guess is the leading-``r`` POD
reconstruction.  Used to pick the "plateau/knee" rank at which the median bare-
guess constraint residual is not visibly changed from the shipped model, then to
produce that curve for the two-panel ``fig_polish_staircase``.

The off-node points are drawn by the committed ``run_polish_table``
``random_offnode_points`` (seed-shared, same box / grid / gap rule), so this
staircase lands on the IDENTICAL 1000 points as the cold-start (``run_polish_cold``)
and the shipped-r POD tables.  Truncation reuses the committed
``run_guess_vs_memory.load_pod_truncated`` (bit-for-bit the POD loader, sliced).

Writes ``reports/P3/polish_table_chi<dim>d_pod_r<r>_<n>.json`` with the
``run_polish_table`` summary schema PLUS the raw per-step residual arrays
(``residuals``) so the figure can draw honest min--max whiskers.

Add-only.  Imports committed ``lm.initial_data`` + sibling drivers read-only; edits nothing.

Run:
  python src/lm/initial_data/pipeline/run_polish_podrank.py --dim 4 --rank 75
  python src/lm/initial_data/pipeline/run_polish_podrank.py --dim 8 --rank 250
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
from lm.initial_data.parametric.parametric_nd import attach_solve_fn_3d

# committed sibling drivers (read-only reuse — identical points + truncation)
from lm.initial_data.pipeline.run_polish_table import (
    random_offnode_points, step_residuals, _stats, REPDIR)
from lm.initial_data.pipeline.run_guess_vs_memory import MODELS, load_pod_truncated


def run(dim, rank, n_points=1000, seed=0):
    os.makedirs(REPDIR, exist_ok=True)
    t0 = time.time()
    path = MODELS[dim]["pod"]
    z = np.load(path, allow_pickle=True, mmap_mode="r")
    meta = json.loads(z["meta_json"].item())
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    r_ship = int(z["r"])
    rank = int(min(rank, r_ship))
    tag = f"chi{dim}d_pod_r{rank}"
    print(f"[{tag}] d={len(box)} grid={Na}x{Nb}x{Nphi} r_ship={r_ship} -> r={rank} "
          f"enhanced={meta.get('enhanced')}", flush=True)
    print(f"[{tag}] box={dict(zip(names, box))} fixed={fixed}", flush=True)

    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    print(f"[{tag}] loading + truncating POD ({os.path.getsize(path)/1e6:.0f} MB) ...",
          flush=True)
    model = load_pod_truncated(path, rank)
    attach_solve_fn_3d(model, prob, names, M_tot=1.0, fixed=fixed, solver="nk")

    print(f"[{tag}] drawing {n_points} uniform off-node points (seed={seed}) ...",
          flush=True)
    hold = random_offnode_points(box, n_points, seed)

    MAXSTEPS = 4
    R = [[] for _ in range(MAXSTEPS + 1)]
    STC, CERT = [], []
    for i, theta in enumerate(hold):
        res_list, stc, cert = step_residuals(model, theta, max_steps=MAXSTEPS)
        for k in range(MAXSTEPS + 1):
            R[k].append(res_list[k])
        CERT.append(cert)
        STC.append(stc if stc is not None else 99)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"   {i+1}/{n_points}  ({el:.0f}s, {el/(i+1):.2f}s/pt, "
                  f"ETA {el/(i+1)*(n_points-i-1):.0f}s)", flush=True)

    R = [np.array(r) for r in R]
    STC = np.array(STC)
    keys = ["guess"] + [f"after{k}" for k in range(1, MAXSTEPS + 1)]
    rows = {keys[k]: _stats(R[k]) for k in range(MAXSTEPS + 1)}
    residuals = {keys[k]: [float(x) for x in R[k]] for k in range(MAXSTEPS + 1)}
    frac_cert = {keys[k]: float(np.mean(R[k] <= 1e-10)) for k in range(MAXSTEPS + 1)}
    certified = STC[STC < 99]

    res = {"config": {"tag": tag, "model": "pod_hermite_smolyak", "level": meta.get("level"),
                      "dim": len(box), "box": [{"name": n, "min": lo, "max": hi}
                                               for n, (lo, hi) in zip(names, box)],
                      "fixed": fixed, "Na": Na, "Nb": Nb, "Nphi": Nphi,
                      "enhanced": list(meta.get("enhanced", []) or []),
                      "r": rank, "r_shipped": r_ship,
                      "model_file": os.path.basename(path),
                      "n_solver_nodes": int(model.n_solver_nodes),
                      "n_points": n_points, "seed": seed, "gap_min": 1e-4,
                      "max_steps": MAXSTEPS},
           "rows": rows,
           "residuals": residuals,
           "frac_certified_le_1e-10": frac_cert,
           "worst_per_step": {keys[k]: float(R[k].max()) for k in range(MAXSTEPS + 1)},
           "steps_to_certify_hist": {int(k): int(np.sum(STC == k))
                                     for k in sorted(set(STC.tolist()))},
           "median_steps_to_certify": (float(np.median(certified))
                                       if certified.size else None),
           "max_steps_to_certify": (int(certified.max()) if certified.size else None),
           "n_never_certified": int(np.sum(STC == 99)),
           "wall_clock_s": time.time() - t0}
    out = os.path.join(REPDIR, f"polish_table_{tag}_{n_points}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)

    print(f"\n=== [{tag}] value+grad POD (r={rank}) certified refinement, "
          f"{n_points} off-node points ===")
    hdr = (f"{'':<22}{'min':>11}{'median':>11}{'mean':>11}{'p95':>11}{'max':>11}"
           f"{'%<=1e-10':>10}")
    print(hdr)
    for k in keys:
        s = rows[k]
        lbl = "guess (no polish)" if k == "guess" else \
              f"after {k[5:]} Newton step{'s' if k != 'after1' else ''}"
        print(f"{lbl:<22}" + "".join(f"{s[c]:>11.2e}"
              for c in ("min", "median", "mean", "p95", "max"))
              + f"{100*frac_cert[k]:>9.1f}%")
    print(f"\nsteps-to-certify hist: {res['steps_to_certify_hist']}"
          f"  (median {res['median_steps_to_certify']}, "
          f"max {res['max_steps_to_certify']}, never {res['n_never_certified']})")
    print(f"[{tag}] DONE in {res['wall_clock_s']:.0f}s -> {out}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, choices=[4, 8], required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.dim, args.rank, n_points=args.n_points, seed=args.seed)


if __name__ == "__main__":
    main()
