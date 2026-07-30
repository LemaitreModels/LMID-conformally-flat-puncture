"""PARASOL — COLD-START certified-refinement (Newton-polish) staircase.

Companion to ``run_polish_table.py``: instead of warm-starting the Newton polish
from the barycentric surrogate guess, this runs the genuine elliptic solver
**cold** — i.e. from the solver's own zero-field initial iterate (``U0=None``, the
same cold start the manuscript's ``tab:timing`` cold Newton--Krylov solve uses,
``run_qc_timing.py``: ``solve_nk(theta, None, 1e-10, 30)``).  No surrogate, no
interpolant.  It records the equilibrated constraint residual ``||R||_inf`` of the
zero-field guess (step 0) and after 1..K Newton--Krylov steps, over the IDENTICAL
1000 seed-0 off-node points as ``run_polish_table.py`` (same box / grid / gap
rule), so the cold staircase is directly comparable to the warm (POD) staircase.

Because the cold solve never touches the surrogate, its per-point cost depends
only on the spatial grid (44x32x8 for every shipped chi model), not the parameter
dimension — so the 8-D cold run does NOT load the 483 MB POD model; it reads only
the box / axes / grid / fixed params from the model's ``.npz`` metadata.

Writes ``reports/P3/polish_cold_<tag>_<n>.json`` with the same summary schema as
``run_polish_table.py`` (per-step min/median/mean/p95/max, steps-to-certify hist,
%certified) PLUS the raw per-step residual arrays (``residuals`` per stage) so the
figure builder can draw honest min--max whiskers.

Add-only.  Imports committed ``parasol`` modules read-only; edits nothing.

Run:
  python sandbox/parasol/run_polish_cold.py                 # 4-D then 8-D
  python sandbox/parasol/run_polish_cold.py --dim 4
  python sandbox/parasol/run_polish_cold.py --n-points 1000 --steps 8 --seed 0
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
from lm.initial_data.parametric.parametric import cheb_param_nodes
from lm.initial_data.parametric.parametric_nd_3d import make_solve_fn

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "P3")
GAP_MIN = 1e-4

# The same two shipped models keyed by dimension (only their .npz metadata is read —
# box, axes, grid, fixed — NOT the stored fields, so no big load for the cold solve).
MODELS = {
    4: os.path.join(HERE, "reports/P2/models_chi/"
                    "pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By.npz"),
    8: os.path.join(HERE, "reports/P2/models_chi/pod_hermite_smolyak_"
                    "spin8qc_L5_enh-chi_Ax-chi_Ay-chi_Az-chi_Bx-chi_By-chi_Bz.npz"),
}


def read_meta(path):
    z = np.load(path, allow_pickle=True, mmap_mode="r")
    return json.loads(z["meta_json"].item())


def random_offnode_points(box, n_points, seed):
    """IDENTICAL sampling to run_polish_table.random_offnode_points (seed-shared)."""
    rng = np.random.default_rng(seed)
    node_sets = [cheb_param_nodes(lo, hi, 16)[0] for lo, hi in box]
    d = len(box)
    pts = np.empty((n_points, d))
    for i in range(n_points):
        while True:
            theta = np.array([rng.uniform(lo, hi) for lo, hi in box])
            if all(np.min(np.abs(theta[k] - node_sets[k])) > GAP_MIN for k in range(d)):
                break
        pts[i] = theta
    return pts


def cold_history(solve_fn, theta, steps):
    """Cold NK from the zero field; return the equilibrated ||R||_inf at steps
    0..``steps`` (0 = the bare zero-field guess), padded with the final (floor)
    value once the solve has converged/stagnated, plus (certified residual, first
    step k with ||R||<=1e-10 or None)."""
    # tol 1e-12 (like run_polish_table) so the loop descends to the NK floor, not
    # 1e-10; max_iter=steps -> newton_solve_nk runs steps+1 iters -> history[0..steps].
    _U, info = solve_fn(np.asarray(theta, dtype=float), None, 1e-12, steps)
    h = list(info.history)
    res = [float(h[k] if k < len(h) else h[-1]) for k in range(steps + 1)]
    stc = next((k for k, r in enumerate(h) if r <= 1e-10), None)
    return res, stc, float(info.residual_norm)


def _stats(a):
    a = np.asarray(a, float)
    return dict(min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), p95=float(np.percentile(a, 95)),
                max=float(a.max()))


def run_dim(dim, n_points, steps, seed):
    model_path = MODELS[dim]
    meta = read_meta(model_path)
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    tag = f"chi{dim}d"
    print(f"[{tag}] d={len(box)} grid={Na}x{Nb}x{Nphi} box={dict(zip(names, box))} "
          f"fixed={fixed}  (cold NK from zeros; NO surrogate)", flush=True)

    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    # plain cold NK solve_fn (retry_tol=None) — matches run_qc_timing's cold solve
    solve_fn, _ = make_solve_fn(prob, names, M_tot=1.0, fixed=fixed,
                                use_cache=True, solver="nk")

    print(f"[{tag}] drawing {n_points} uniform off-node points (seed={seed}) ...",
          flush=True)
    hold = random_offnode_points(box, n_points, seed)

    R = [[] for _ in range(steps + 1)]
    STC, CERT = [], []
    t0 = time.time()
    for i, theta in enumerate(hold):
        res_list, stc, cert = cold_history(solve_fn, theta, steps)
        for k in range(steps + 1):
            R[k].append(res_list[k])
        CERT.append(cert)
        STC.append(stc if stc is not None else 99)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"   {i+1}/{n_points}  ({el:.0f}s, {el/(i+1):.2f}s/pt, "
                  f"ETA {el/(i+1)*(n_points-i-1):.0f}s)", flush=True)

    R = [np.array(r) for r in R]
    STC = np.array(STC)
    keys = ["guess"] + [f"after{k}" for k in range(1, steps + 1)]
    rows = {keys[k]: _stats(R[k]) for k in range(steps + 1)}
    residuals = {keys[k]: [float(x) for x in R[k]] for k in range(steps + 1)}
    frac_cert = {keys[k]: float(np.mean(R[k] <= 1e-10)) for k in range(steps + 1)}
    certified = STC[STC < 99]

    res = {"config": {"tag": tag, "model": "cold_nk", "guess": "zero_field",
                      "level": meta.get("level"), "dim": len(box),
                      "box": [{"name": n, "min": lo, "max": hi}
                              for n, (lo, hi) in zip(names, box)],
                      "fixed": fixed, "Na": Na, "Nb": Nb, "Nphi": Nphi,
                      "model_file": os.path.basename(model_path),
                      "n_points": n_points, "seed": seed, "gap_min": GAP_MIN,
                      "max_steps": steps},
           "rows": rows,
           "residuals": residuals,
           "frac_certified_le_1e-10": frac_cert,
           "worst_per_step": {keys[k]: float(R[k].max()) for k in range(steps + 1)},
           "steps_to_certify_hist": {int(k): int(np.sum(STC == k))
                                     for k in sorted(set(STC.tolist()))},
           "median_steps_to_certify": (float(np.median(certified))
                                       if certified.size else None),
           "max_steps_to_certify": (int(certified.max()) if certified.size else None),
           "n_never_certified": int(np.sum(STC == 99)),
           "wall_clock_s": time.time() - t0}
    out = os.path.join(REPDIR, f"polish_cold_{tag}_{n_points}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)

    print(f"\n=== [{tag}] COLD certified refinement over {n_points} off-node points ===")
    hdr = (f"{'':<22}{'min':>11}{'median':>11}{'mean':>11}{'p95':>11}{'max':>11}"
           f"{'%<=1e-10':>10}")
    print(hdr)
    for k in keys:
        s = rows[k]
        lbl = "guess (cold, no polish)" if k == "guess" else \
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
    ap.add_argument("--dim", choices=["4", "8", "both"], default="both")
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(REPDIR, exist_ok=True)
    dims = [4, 8] if args.dim == "both" else [int(args.dim)]
    for dim in dims:                       # sequential — avoid CPU contention (JAX)
        run_dim(dim, args.n_points, args.steps, args.seed)


if __name__ == "__main__":
    main()
