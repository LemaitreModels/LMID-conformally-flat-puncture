"""LM-initial-data — GENERAL certified-refinement (Newton-polish) sweep driver.

One driver for every (dimension, guess-type) certified-refinement experiment that
feeds the manuscript ``tab:polish`` presentation.  Given a model .npz, it reads the
box / axes / grid / fixed params from that model's OWN metadata (no hardcoding),
draws ``n_points`` uniform off-node points in the box, and records the certified
constraint residual ||R||_inf of the model's barycentric warm-start guess and after
1..4 Newton--Krylov steps (from the NK residual history of one certified solve per
point).  The guess model is one of:

  * value-only Smolyak            (kind ``smolyak``)            -> load_smolyak
  * gradient-enhanced Hermite      (kind ``hermite_smolyak``)    -> load_hermite_smolyak
  * POD-compressed Hermite         (kind ``pod_hermite_smolyak``)-> load_pod_hermite_smolyak

The loader is inferred from ``meta["kind"]``.  All configs at the same box+seed draw
the IDENTICAL point set, so the runs are directly comparable.

Writes ``reports/P3/polish_table_<tag>_<n>.json`` with a schema shared by
``run_polish_table_qc.py`` / ``run_polish_table_qc_chi.py`` so the summary/figure
generator (``make_polish_summary.py``) can read all of them uniformly.

Run:
  python -m lm.initial_data.pipeline.run_polish_table --model <path.npz> --tag <name>
         [--n-points 1000] [--seed 0]
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
from lm.initial_data.parametric.parametric_nd import attach_solve_fn_3d

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "P3")
GAP_MIN = 1e-4


def read_meta(path):
    z = np.load(path, allow_pickle=True, mmap_mode="r")
    return json.loads(z["meta_json"].item())


def load_model(path, meta, prob, names, fixed, M_tot=1.0):
    kind = meta["kind"]
    if kind == "smolyak":
        from lm.initial_data.parametric.parametric_nd_smolyak import load_smolyak
        model = load_smolyak(path)
    elif kind == "hermite_smolyak":
        from lm.initial_data.parametric.hermite_smolyak import load_hermite_smolyak
        model = load_hermite_smolyak(path)
    elif kind == "pod_hermite_smolyak":
        from lm.initial_data.parametric.hermite_smolyak_pod import load_pod_hermite_smolyak
        model = load_pod_hermite_smolyak(path)
    elif kind == "hermite_smolyak_cross":
        from lm.initial_data.parametric.hermite_smolyak_cross import load_hermite_smolyak_cross
        model = load_hermite_smolyak_cross(path)
    elif kind == "pod_hermite_smolyak_cross":
        from lm.initial_data.parametric.hermite_smolyak_pod_cross import load_pod_hermite_smolyak_cross
        model = load_pod_hermite_smolyak_cross(path)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    attach_solve_fn_3d(model, prob, names, M_tot=M_tot, fixed=fixed, solver="nk")
    return model


def random_offnode_points(box, n_points, seed):
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


def step_residuals(model, theta, max_steps=4):
    _U, info = model.evaluate_polished(theta, newton_steps=max_steps, tol=1e-12)
    h = list(info.history)
    res = [float(h[k] if k < len(h) else h[-1]) for k in range(max_steps + 1)]
    stc = next((k for k, r in enumerate(h) if r <= 1e-10), None)
    return res, stc, float(info.residual_norm)


def _stats(a):
    a = np.asarray(a, float)
    return dict(min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), p95=float(np.percentile(a, 95)),
                max=float(a.max()))


def main(model_path, tag, n_points=1000, seed=0):
    os.makedirs(REPDIR, exist_ok=True)
    t0 = time.time()
    meta = read_meta(model_path)
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    kind = meta["kind"]
    enhanced = list(meta.get("enhanced", []) or [])
    print(f"[{tag}] kind={kind} d={len(box)} level={meta.get('level')} "
          f"nodes={meta.get('n_solver_nodes')} r={meta.get('r')} "
          f"enhanced={enhanced}", flush=True)
    print(f"[{tag}] box={dict(zip(names, box))} fixed={fixed} "
          f"grid Na={Na},Nb={Nb},Nphi={Nphi}", flush=True)

    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    print(f"[{tag}] loading model ({os.path.getsize(model_path)/1e6:.0f} MB) ...",
          flush=True)
    model = load_model(model_path, meta, prob, names, fixed)

    print(f"[{tag}] drawing {n_points} uniform off-node points (seed={seed}) ...",
          flush=True)
    hold = random_offnode_points(box, n_points, seed)

    MAXSTEPS = 4
    R = [[] for _ in range(MAXSTEPS + 1)]
    STC, CERT = [], []
    t_eval = time.time()
    for i, theta in enumerate(hold):
        res_list, stc, cert = step_residuals(model, theta, max_steps=MAXSTEPS)
        for k in range(MAXSTEPS + 1):
            R[k].append(res_list[k])
        CERT.append(cert)
        STC.append(stc if stc is not None else 99)
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{n_points}  (elapsed {time.time()-t_eval:.0f}s)",
                  flush=True)

    R = [np.array(r) for r in R]
    STC = np.array(STC)
    keys = ["guess"] + [f"after{k}" for k in range(1, MAXSTEPS + 1)]
    rows = {keys[k]: _stats(R[k]) for k in range(MAXSTEPS + 1)}
    frac_cert = {k: float(np.mean(R[k] <= 1e-10)) for k in range(MAXSTEPS + 1)}

    res = {"config": {"tag": tag, "model": kind, "level": meta.get("level"),
                      "dim": len(box), "box": [{"name": n, "min": lo, "max": hi}
                                               for n, (lo, hi) in zip(names, box)],
                      "fixed": fixed, "Na": Na, "Nb": Nb, "Nphi": Nphi,
                      "enhanced": enhanced, "r": meta.get("r"),
                      "model_file": os.path.basename(model_path),
                      "n_solver_nodes": int(model.n_solver_nodes),
                      "n_points": n_points, "seed": seed, "gap_min": GAP_MIN,
                      "max_steps": MAXSTEPS},
           "rows": rows,
           "frac_certified_le_1e-10": {keys[k]: frac_cert[k]
                                       for k in range(MAXSTEPS + 1)},
           "worst_per_step": {keys[k]: float(R[k].max())
                              for k in range(MAXSTEPS + 1)},
           "steps_to_certify_hist": {int(k): int(np.sum(STC == k))
                                     for k in sorted(set(STC.tolist()))},
           "median_steps_to_certify": float(np.median(STC[STC < 99])),
           "max_steps_to_certify": int(STC[STC < 99].max()),
           "wall_clock_s": time.time() - t0}
    out = os.path.join(REPDIR, f"polish_table_{tag}_{n_points}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)

    print(f"\n=== [{tag}] certified refinement over {n_points} off-node points ===")
    hdr = (f"{'':<22}{'min':>11}{'median':>11}{'mean':>11}{'p95':>11}{'max':>11}"
           f"{'%<=1e-10':>10}")
    print(hdr)
    label = {"guess": "guess (no polish)"}
    label.update({f"after{k}": f"after {k} Newton step{'s' if k > 1 else ''}"
                  for k in range(1, MAXSTEPS + 1)})
    for k in keys:
        s = rows[k]
        print(f"{label[k]:<22}" + "".join(f"{s[c]:>11.2e}"
              for c in ("min", "median", "mean", "p95", "max"))
              + f"{100*frac_cert[keys.index(k)]:>9.1f}%")
    print(f"\nsteps-to-certify hist: {res['steps_to_certify_hist']}"
          f"  (median {res['median_steps_to_certify']:.0f}, "
          f"max {res['max_steps_to_certify']})")
    print(f"[{tag}] DONE in {res['wall_clock_s']:.0f}s -> {out}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the guess model .npz")
    ap.add_argument("--tag", required=True, help="short config tag for the output file")
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.model, args.tag, n_points=args.n_points, seed=args.seed)
