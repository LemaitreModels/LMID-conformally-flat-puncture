"""PARASOL — the certified-and-differentiable PARAMETER-TARGETING demonstrator
(paper §VI), on the shipped 4-D quasi-circular model.

Hit a physical target ``(M_ADM, J)`` by adjusting ``(b, q)`` (spins fixed), three
ways, over ``N`` random known-answer targets, and tabulate the honest cost metric
— the **number of certified elliptic solves** to reach the target:

  * ``cold``     — black-box Broyden, each F-eval a cold certified solve;
  * ``broyden``  — black-box Broyden, each F-eval warm-started from the surrogate
                   (standard NR practice: cuts per-solve cost, NOT solve count);
  * ``gradient`` — Gauss–Newton on the *free* differentiable surrogate (analytic
                   ∂F/∂θ, no solve per step) + a certified last-mile.

Every emitted configuration is certified to ``‖R‖∞ ≤ 1e-10``.  Writes
``reports/P3/qc_targeting_<N>.json`` and the figure
``figures/fig_qc_targeting.png``.

Run: ~/micromamba/envs/BBHFM/bin/python sandbox/parasol/run_qc_targeting.py [--n 25] [--seed 0]
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
from lemaitre.initial_data.applications import qc_targeting as T

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "reports", "3D_parametric", "models",
                     "surrogate_smolyak_d4_qc_L4.npz")
REPDIR = os.path.join(HERE, "reports", "P3")
FIGDIR = os.path.join(HERE, "figures")

BOX = np.array([[1.5, 1.0, -0.4, -0.4], [4.0, 3.0, 0.4, 0.4]])
TARGETS = ["M_ADM", "J"]
ACTIVE = (0, 1)                     # adjust (b, q); spins fixed at 0
THETA0 = np.array([3.2, 1.15, 0.0, 0.0])   # a fixed generic start for every target
METHODS = ("cold", "broyden", "gradient")


def draw_targets(n, seed):
    """``n`` random known-answer configurations ``θ* = (b,q,0,0)`` in the interior."""
    rng = np.random.default_rng(seed)
    lo, hi = BOX[0], BOX[1]
    pad = 0.08 * (hi - lo)                       # stay off the box edges (endpoint nodes)
    stars = []
    for _ in range(n):
        b = rng.uniform(lo[0] + pad[0], hi[0] - pad[0])
        q = rng.uniform(lo[1] + pad[1], hi[1] - pad[1])
        stars.append(np.array([b, q, 0.0, 0.0]))
    return stars


def run_one(model, prob, theta_star):
    target = T.make_target(model, prob, theta_star, TARGETS)
    out = {}
    for m in METHODS:
        if m == "gradient":
            r = T.gauss_newton_target(model, prob, target, THETA0, TARGETS, BOX,
                                      active=ACTIVE)
        else:
            r = T.broyden_target(model, prob, target, THETA0, TARGETS, BOX,
                                 mode=("cold" if m == "cold" else "interp"),
                                 active=ACTIVE)
        out[m] = dict(n_solves=r.n_certified_solves, ctrl=r.ctrl_residual,
                      cert=r.certified_residual, wall=r.wall_s,
                      converged=r.converged, history=r.history,
                      theta=r.theta.tolist(), target=target.tolist(),
                      theta_star=theta_star.tolist())
    return out


def main(n=25, seed=0):
    os.makedirs(REPDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    t0 = time.time()
    print(f"[qc-target] load model + problem ...", flush=True)
    prob = s3.make_problem(Na=44, Nb=32, Nphi=8)
    model = T.load_model(MODEL, prob)
    print(f"[qc-target]   nodes={model.n_solver_nodes}", flush=True)

    stars = draw_targets(n, seed)
    runs = []
    for i, ts in enumerate(stars):
        r = run_one(model, prob, ts)
        runs.append(r)
        print(f"  [{i+1}/{n}] b*={ts[0]:.2f} q*={ts[1]:.2f}  "
              f"solves cold/broy/grad = {r['cold']['n_solves']}/"
              f"{r['broyden']['n_solves']}/{r['gradient']['n_solves']}  "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate ----
    def agg(key):
        d = {}
        for m in METHODS:
            v = np.array([r[m][key] for r in runs], float)
            d[m] = dict(min=float(v.min()), median=float(np.median(v)),
                        mean=float(v.mean()), max=float(v.max()))
        return d

    worst_cert = {m: float(max(r[m]["cert"] for r in runs)) for m in METHODS}
    all_conv = {m: bool(all(r[m]["converged"] for r in runs)) for m in METHODS}
    summary = dict(n=n, seed=seed, box=BOX.tolist(), targets=TARGETS,
                   active=list(ACTIVE), theta0=THETA0.tolist(),
                   n_solver_nodes=int(model.n_solver_nodes),
                   solves=agg("n_solves"), wall=agg("wall"),
                   worst_certified_residual=worst_cert,
                   all_converged=all_conv, wall_clock_s=time.time() - t0)
    out = os.path.join(REPDIR, f"qc_targeting_{n}.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "runs": runs}, f, indent=2, default=float)

    # ---- report ----
    print(f"\n=== QC parameter targeting: (M_ADM,J) via (b,q), {n} random targets ===")
    print(f"model: Smolyak L=4 ({model.n_solver_nodes} solves)\n")
    print(f"{'method':<12}{'solves(med)':>12}{'solves(max)':>12}"
          f"{'wall(med) s':>13}{'worst||R||':>13}{'all conv':>10}")
    for m in METHODS:
        s, w = summary["solves"][m], summary["wall"][m]
        print(f"{m:<12}{s['median']:>12.0f}{s['max']:>12.0f}{w['median']:>13.1f}"
              f"{worst_cert[m]:>13.1e}{str(all_conv[m]):>10}")
    med_bb = summary["solves"]["cold"]["median"]
    med_gr = summary["solves"]["gradient"]["median"]
    print(f"\nheadline: gradient needs {med_gr:.0f} certified solves vs "
          f"{med_bb:.0f} for the black box  ({med_bb/max(med_gr,1):.1f}x fewer)")
    print(f"[qc-target] DONE {time.time()-t0:.0f}s -> {out}", flush=True)

    _figure(runs, summary, os.path.join(FIGDIR, "fig_qc_targeting.png"))
    return summary


def _figure(runs, summary, path):
    """Single panel: the convergence trace of *every* target, per method —
    target residual vs cumulative certified elliptic solves.  Faint lines are
    the individual runs; the bold line is the across-runs median residual at
    each solve count (drawn only while a majority of runs are still active)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"cold": "#c44e52", "broyden": "#dd8452", "gradient": "#4c72b0"}
    LB = {"cold": "black-box", "broyden": "warm black-box (Broyden)",
          "gradient": "differentiable (gradient)"}
    # cold and warm coincide in solve count (warm-starting cuts only wall time),
    # so the figure shows one black-box family (cold) against the gradient method.
    PLOT = ("cold", "gradient")
    TOL = 1e-8
    n = len(runs)

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.2))
    for m in PLOT:
        # every run, faint
        for r in runs:
            h = np.array(r[m]["history"], float)          # (k, [n_solves, resid])
            ax.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-16),
                        color=C[m], alpha=0.13, lw=0.8, solid_capstyle="round")
        # across-runs median residual at each cumulative-solve count
        by_x = {}
        for r in runs:
            for ns, res in r[m]["history"]:
                by_x.setdefault(int(ns), []).append(max(float(res), 1e-16))
        xm = [x for x in sorted(by_x) if len(by_x[x]) >= n // 2]
        ym = [np.median(by_x[x]) for x in xm]
        ax.semilogy(xm, ym, "o-", color=C[m], lw=2.4, ms=4.5, label=LB[m],
                    zorder=5)
    ax.axhline(TOL, color="grey", ls=":", lw=1)
    ax.text(ax.get_xlim()[1], TOL, " tol", color="grey", va="bottom", ha="right",
            fontsize=8)
    ax.set_xlabel("certified elliptic solves")
    ax.set_ylabel(r"target residual $\|F-F_\star\|_\infty$")
    ax.set_title(f"convergence over {n} random targets", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[qc-target] figure -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-json", type=str, default=None,
                    help="regenerate the figure from an existing results JSON "
                         "(no solves)")
    args = ap.parse_args()
    if args.from_json:
        with open(args.from_json) as f:
            d = json.load(f)
        _figure(d["runs"], d["summary"],
                os.path.join(FIGDIR, "fig_qc_targeting.png"))
    else:
        main(n=args.n, seed=args.seed)
