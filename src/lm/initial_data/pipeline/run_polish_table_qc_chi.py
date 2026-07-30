"""LM-initial-data — certified-refinement (Newton-polish) table for the 4-D QUASI-CIRCULAR
model, in the dimensionless-spin (chi) parameterization, over MANY
uniformly-random off-node points.

χ / production-box / ℓ=5 redo of the paper ``tab:polish`` experiment (the ``\\todo``
on that table). Same methodology as ``run_polish_table_qc.py`` (which is the OLD
ℓ=4 S-parameterized model), only the box, the model, and the Smolyak level change:

    b       in [B_MIN, B_MAX] M            (coordinate separation, production range)
    q       in [Q_MIN, Q_MAX]              (mass ratio)
    chi_Ay  in [-CHI_MAX, CHI_MAX]         (dimensionless aligned spin of hole A, χ=S/m²)
    chi_By  in [-CHI_MAX, CHI_MAX]         (dimensionless aligned spin of hole B)

with all edges taken from ``production_box``,

at fixed quasi-circular tangential momentum (``qc=1.0``), on the production 3-D
spatial grid (Na=44, Nb=32, Nphi=8), against the shipped 4-D χ Smolyak model
(isotropic level L=5, 1105 solves; manifest row S3):

    reports/3D_parametric/models_chi/surrogate_smolyak_d4_qc_chi_prod_L5.npz

For each query theta the barycentric (combination-technique) prediction is the
warm start, and we record the certified (equilibrated) constraint residual
||R||_inf of the bare guess and after 1..4 Newton--Krylov steps, read from the NK
residual history of a SINGLE certified solve per point.

Writes ``reports/P3/polish_table_qc_chi_prod_<n>.json`` and prints a LaTeX-ready
summary.

Run:  ~/micromamba/envs/BBHFM/bin/python -m lm.initial_data.pipeline.run_polish_table_qc_chi
      [--n-points 1000] [--seed 0] [--level 5]
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
from lm.initial_data.parametric import parametric_nd_smolyak as sm
from lm.initial_data.parametric.parametric import cheb_param_nodes
from lm.initial_data.pipeline import production_box as pb

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "P3")
MODELDIR = os.path.join(HERE, "reports", "3D_parametric", "models_chi")

# --- the paper's 4-D quasi-circular box, chi parameterization ----
BOX = pb.aligned_box()
NAMES = [a["name"] for a in BOX]
FIXED = dict(pb.FIXED_QC)
NA, NB, NPHI, M_TOT = 44, 32, 8, 1.0
GAP_MIN = 1e-4  # off-node guard
MODEL_NAME = "surrogate_smolyak_d4_qc_chi_prod_L5.npz"


def build_or_load(prob, level):
    os.makedirs(MODELDIR, exist_ok=True)
    path = os.path.join(MODELDIR, MODEL_NAME)
    if os.path.exists(path):
        print(f"[qc-chi] loading existing model {path}", flush=True)
        model = sm.load_smolyak(path)
        from lm.initial_data.parametric.parametric_nd import attach_solve_fn_3d
        attach_solve_fn_3d(model, prob, NAMES, M_tot=M_TOT, fixed=FIXED, solver="nk")
        return model
    # The shipped chi prod L5 model (manifest S3) is expected on disk; only build
    # as a fallback.
    print(f"[qc-chi] building Smolyak L={level} model (this is the ~hour-long step) ...",
          flush=True)
    tb = time.time()
    solver = sm.from_problem_smolyak_3d(prob, BOX, M_tot=M_TOT, fixed=FIXED,
                                        solver="nk")
    model = solver.build_isotropic(level, tol=1e-12, max_iter=25, verbose=True)
    print(f"[qc-chi]   built {model.n_solver_nodes} nodes in {time.time()-tb:.0f}s",
          flush=True)
    model.save(path, meta={"box": BOX, "fixed": FIXED,
                           "grid": [NA, NB, NPHI], "level": level})
    print(f"[qc-chi]   saved -> {path}", flush=True)
    return model


def random_offnode_points(n_points, seed):
    """``n_points`` i.i.d. uniform points in the box, rejected-&-resampled off a
    dense per-axis CGL node superset (a formality; uniform draws are generically
    off any finite node set)."""
    rng = np.random.default_rng(seed)
    node_sets = [cheb_param_nodes(a["min"], a["max"], 16)[0] for a in BOX]
    pts = np.empty((n_points, len(BOX)))
    for i in range(n_points):
        while True:
            theta = np.array([rng.uniform(a["min"], a["max"]) for a in BOX])
            if all(np.min(np.abs(theta[k] - node_sets[k])) > GAP_MIN
                   for k in range(len(BOX))):
                break
        pts[i] = theta
    return pts


def step_residuals(model, theta, max_steps=4):
    """Certified ||R||_inf after 0..max_steps NK steps, from the history of one
    solve.  Pads with the last (certified) value if NK converged early.  Returns
    ``(res_list, steps_to_certify, certified_residual)``."""
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


def main(n_points=1000, seed=0, level=5):
    os.makedirs(REPDIR, exist_ok=True)
    t0 = time.time()
    boxstr = ", ".join(f"{a['name']}:[{a['min']:g},{a['max']:g}]" for a in BOX)
    print(f"[qc-chi] model: Smolyak L={level} over box {{{boxstr}}} "
          f"(fixed {FIXED})  grid Na={NA},Nb={NB},Nphi={NPHI}", flush=True)
    prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)
    model = build_or_load(prob, level)

    print(f"[qc-chi] drawing {n_points} uniform off-node points (seed={seed}) ...",
          flush=True)
    hold = random_offnode_points(n_points, seed)

    MAXSTEPS = 4
    R = [[] for _ in range(MAXSTEPS + 1)]     # residuals after 0,1,2,3,4 steps
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
    CERT = np.array(CERT); STC = np.array(STC)
    keys = ["guess"] + [f"after{k}" for k in range(1, MAXSTEPS + 1)]
    rows = {keys[k]: _stats(R[k]) for k in range(MAXSTEPS + 1)}
    frac_cert = {k: float(np.mean(R[k] <= 1e-10)) for k in range(MAXSTEPS + 1)}

    res = {"config": {"model": "smolyak_isotropic_chi", "level": level, "box": BOX,
                      "fixed": FIXED, "Na": NA, "Nb": NB, "Nphi": NPHI,
                      "model_file": MODEL_NAME,
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
           "wall_clock_s": time.time() - t0}
    out = os.path.join(REPDIR, f"polish_table_qc_chi_prod_{n_points}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)

    # ---- report -----------------------------------------------------------
    print(f"\n=== QC-chi certified refinement over {n_points} uniform off-node points ===")
    print(f"model: Smolyak L={level} ({model.n_solver_nodes} solves)  "
          f"chi b∈[{pb.B_MIN:g},{pb.B_MAX:g}]")
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
    print(f"\nsteps-to-certify (<=1e-10) histogram: {res['steps_to_certify_hist']}"
          f"  (median {res['median_steps_to_certify']:.0f})")

    def tex(x):
        m, e = f"{x:.1e}".split("e")
        return f"${m}\\mathrm{{e}}{{-}}{abs(int(e))}$"
    print("\n--- LaTeX (min | median | max) ---")
    for k in keys:
        s = rows[k]
        print(f"    {label[k]:<20} & {tex(s['min'])}  & {tex(s['median'])}  "
              f"& {tex(s['max'])} \\\\")
    print(f"\n[qc-chi] DONE in {res['wall_clock_s']:.0f}s -> {out}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=5)
    args = ap.parse_args()
    main(n_points=args.n_points, seed=args.seed, level=args.level)
