"""LM-initial-data — the certified-and-differentiable PARAMETER-TARGETING demonstrator
(paper §VI), on the shipped PRODUCTION 4-D quasi-circular model.

The box, the axes and the grid all come from ``production_box`` — the same
production model the rest of the paper's 4-D results use:

    b       in [B_MIN, B_MAX] M      (separations D = 2b in [6, 20] M)
    q       in [Q_MIN, Q_MAX]
    chi_Ay  in [-CHI_MAX, CHI_MAX]   (DIMENSIONLESS aligned spin, χ = S/m²)
    chi_By  in [-CHI_MAX, CHI_MAX]

against the shipped 4-D χ Smolyak model (isotropic level L=5, 1105 solves):

    reports/3D_parametric/models_chi/surrogate_smolyak_d4_qc_chi_prod_L5.npz

This replaces the earlier narrow-separation, dimensionful-spin model
(``surrogate_smolyak_d4_qc_L4.npz``, b in [1.5,4], S_Ay/S_By in [-0.4,0.4], L=4),
which is superseded; the protocol below is otherwise unchanged, so the two runs are
directly comparable.

Hit a physical target ``(M_ADM, J)`` by adjusting ``(b, q)`` (spins fixed), three
ways, over ``N`` random known-answer targets, and tabulate the honest cost metric
— the **number of certified elliptic solves** to reach the target:

  * ``cold``     — black-box Broyden, each F-eval a cold certified solve;
  * ``broyden``  — black-box Broyden, each F-eval warm-started from the surrogate
                   (standard NR practice: cuts per-solve cost, NOT solve count);
  * ``gradient`` — Gauss–Newton on the *free* differentiable surrogate (analytic
                   ∂F/∂θ, no solve per step) + a certified last-mile.

Every emitted configuration is certified to ``‖R‖∞ ≤ 1e-10``.  Writes
``reports/P3/qc_targeting_chi_prod_<N>.json`` and the figure
``figures/fig_qc_targeting.png``.

Each target is an independent work unit (its own known-answer target and its own
three control loops), so the ``--n 100`` study strides across a job array and merges:

  sbatch --array=0-9 --export=ALL,DRIVER=run_qc_targeting,ARGS="--n 100" \
      slurm/ivs/submit_lm_initial_data_cpu_array_hi.slurm
  python -m lm.initial_data.pipeline.run_qc_targeting --n 100 --assemble

``--ntasks 1`` (the default off SLURM) runs every target in one process and writes the
results JSON directly; the two paths share ``_aggregate``, so they agree exactly.

Run: python -m lm.initial_data.pipeline.run_qc_targeting [--n 25] [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.applications import qc_targeting as T
from lm.initial_data.pipeline import production_box as pb

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
MODEL = os.path.join(REPORTS, "3D_parametric", "models_chi",
                     "surrogate_smolyak_d4_qc_chi_prod_L5.npz")
REPDIR = os.path.join(REPORTS, "P3")
FIGDIR = os.path.join(HERE, "figures")

# Fixed-budget mode writes a SEPARATE artifact (``..._fixed_<N>.json``) so the
# early-exit run stays on disk beside it; the two answer different questions
# (cost-to-tolerance vs residual-vs-budget) and share the same first crossings.
STEM = "qc_targeting_chi_prod"


def _stem(budgets):
    return STEM + ("_fixed" if budgets else "")


def _partdir(budgets):
    return os.path.join(REPDIR, _stem(budgets) + "_parts")

# --- the production 4-D quasi-circular box, χ parameterization ---------------
_AXES = pb.aligned_box()                                  # (b, q, chi_Ay, chi_By)
BOX = np.array([[a["min"] for a in _AXES], [a["max"] for a in _AXES]])
NAMES = tuple(a["name"] for a in _AXES)
assert NAMES == T.NAMES, (NAMES, T.NAMES)                 # axis order must match
NA, NB, NPHI = pb.PROD_GRID
LEVEL = pb.SMOLYAK_LEVEL

TARGETS = ["M_ADM", "J"]
ACTIVE = (0, 1)                     # adjust (b, q); spins fixed at χ=0
# One fixed generic start for every target, interior to the box and deliberately
# off the CGL node superset (so the surrogate Jacobian never needs the nudge).
THETA0 = np.array([5.0, 1.35, 0.0, 0.0])
METHODS = ("cold", "broyden", "gradient")


def _check_model_box(model):
    """Refuse a model whose provenance is not the production χ box.

    The figure this producer feeds spent a revision on a superseded model whose box
    (b in [1.5,4]) and spin parameterization (dimensionful S, not χ) silently
    disagreed with the rest of the paper.  A mismatched box is invisible in the
    output — the run completes and the numbers look fine — so it is checked here
    rather than left to a reader (HISTORY_AND_FINDINGS §2.6).
    """
    meta = getattr(model, "meta", None) or {}
    got_names = tuple(meta.get("axis_names", ()))
    if got_names != NAMES:
        raise ValueError(f"model axis_names {got_names} != production {NAMES}")
    got_box = np.asarray(model.axes, dtype=float)          # (d, 2) [min, max]
    want_box = BOX.T
    if got_box.shape != want_box.shape or not np.allclose(got_box, want_box):
        raise ValueError(f"model box {got_box.tolist()} != production {want_box.tolist()}")
    got_grid = (meta.get("Na"), meta.get("Nb"), meta.get("Nphi"))
    if got_grid != (NA, NB, NPHI):
        raise ValueError(f"model grid {got_grid} != production {(NA, NB, NPHI)}")
    if int(meta.get("level", -1)) != LEVEL:
        raise ValueError(f"model level {meta.get('level')} != production {LEVEL}")
    if dict(meta.get("fixed", {})) != dict(T.FIXED):
        raise ValueError(f"model fixed {meta.get('fixed')} != {dict(T.FIXED)}")
    print(f"[qc-target]   provenance OK: production box, L={LEVEL}, "
          f"git_commit={meta.get('git_commit')}", flush=True)
    return meta


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


def run_one(model, prob, theta_star, budgets=None):
    """One target through all three strategies.

    ``budgets`` = ``{method: n}`` runs that method to a fixed ``n`` certified solves
    instead of stopping at tolerance (see ``qc_targeting.broyden_target``)."""
    budgets = budgets or {}
    target = T.make_target(model, prob, theta_star, TARGETS)
    out = {}
    for m in METHODS:
        if m == "gradient":
            r = T.gauss_newton_target(model, prob, target, THETA0, TARGETS, BOX,
                                      active=ACTIVE, budget=budgets.get(m))
        else:
            r = T.broyden_target(model, prob, target, THETA0, TARGETS, BOX,
                                 mode=("cold" if m == "cold" else "interp"),
                                 active=ACTIVE, budget=budgets.get(m))
        out[m] = dict(n_solves=r.n_certified_solves, ctrl=r.ctrl_residual,
                      n_solves_to_tol=r.n_solves_to_tol,
                      cert=r.certified_residual, wall=r.wall_s,
                      converged=r.converged, history=r.history,
                      theta=r.theta.tolist(), target=target.tolist(),
                      theta_star=theta_star.tolist())
    return out


def _aggregate(runs, n, seed, *, n_solver_nodes, model_git_commit, wall_clock_s,
               assembled_from=None, budgets=None):
    """The summary block.  Shared by the single-process and ``--assemble`` paths so
    the two produce byte-identical statistics from the same per-target runs."""
    def agg(key):
        d = {}
        for m in METHODS:
            # n_solves_to_tol is None for a target that never reached tolerance
            v = np.array([r[m][key] for r in runs
                          if r[m].get(key) is not None], float)
            if v.size == 0:
                d[m] = None
                continue
            d[m] = dict(min=float(v.min()), median=float(np.median(v)),
                        mean=float(v.mean()), max=float(v.max()),
                        n=int(v.size))
        return d

    summary = dict(n=n, seed=seed, box=BOX.tolist(), targets=TARGETS,
                   active=list(ACTIVE), theta0=THETA0.tolist(),
                   n_solver_nodes=int(n_solver_nodes),
                   # provenance: which model/box these numbers are FROM
                   axis_names=list(NAMES), level=LEVEL, grid=[NA, NB, NPHI],
                   model=os.path.basename(MODEL),
                   model_git_commit=model_git_commit,
                   spin_parameterization="chi",
                   budgets=dict(budgets or {}),
                   solves=agg("n_solves"), wall=agg("wall"),
                   # the COST metric: first solve count reaching tol_ctrl.  Equal to
                   # ``solves`` in early-exit mode; the meaningful one under a budget.
                   solves_to_tol=agg("n_solves_to_tol"),
                   worst_certified_residual={
                       m: float(max(r[m]["cert"] for r in runs)) for m in METHODS},
                   all_converged={
                       m: bool(all(r[m]["converged"] for r in runs)) for m in METHODS},
                   wall_clock_s=wall_clock_s)
    if assembled_from is not None:
        summary["assembled_from"] = int(assembled_from)
        # in array mode the elapsed wall is meaningless (tasks ran concurrently), so
        # the recorded figure is the SUM over tasks; the per-method ``wall`` blocks
        # are per-target and unaffected by sharding.
        summary["wall_clock_is_sum_of_tasks"] = True
    return summary


def _report(summary):
    n = summary["n"]
    bud = summary.get("budgets") or {}
    print(f"\n=== QC parameter targeting: (M_ADM,J) via (b,q), {n} random targets ===")
    print(f"model: production χ box, Smolyak L={summary['level']} "
          f"({summary['n_solver_nodes']} solves)"
          + (f"   FIXED BUDGET {bud}" if bud else "") + "\n")
    print(f"{'method':<12}{'run':>5}{'to_tol(med)':>12}{'to_tol(max)':>12}"
          f"{'wall(med) s':>13}{'worst||R||':>13}{'all conv':>10}")
    for m in METHODS:
        s, w = summary["solves"][m], summary["wall"][m]
        t = summary.get("solves_to_tol", {}).get(m)
        tm = f"{t['median']:.0f}" if t else "-"
        tx = f"{t['max']:.0f}" if t else "-"
        print(f"{m:<12}{s['median']:>5.0f}{tm:>12}{tx:>12}{w['median']:>13.1f}"
              f"{summary['worst_certified_residual'][m]:>13.1e}"
              f"{str(summary['all_converged'][m]):>10}")
    tt = summary.get("solves_to_tol", {})
    if tt.get("cold") and tt.get("gradient"):
        med_bb, med_gr = tt["cold"]["median"], tt["gradient"]["median"]
        print(f"\nheadline (cost = solves to tolerance): gradient {med_gr:.0f} vs "
              f"black box {med_bb:.0f}  ({med_bb/max(med_gr,1):.1f}x fewer)")


def _finalize(runs, n, seed, *, n_solver_nodes, model_git_commit, wall_clock_s,
              assembled_from=None, budgets=None):
    summary = _aggregate(runs, n, seed, n_solver_nodes=n_solver_nodes,
                         model_git_commit=model_git_commit,
                         wall_clock_s=wall_clock_s, assembled_from=assembled_from,
                         budgets=budgets)
    out = os.path.join(REPDIR, f"{_stem(budgets)}_{n}.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "runs": runs}, f, indent=2, default=float)
    _report(summary)
    print(f"[qc-target] -> {out}", flush=True)
    _figure(runs, summary, os.path.join(FIGDIR, "fig_qc_targeting.png"))
    return summary


def assemble(n=100, seed=0, budgets=None):
    """Merge the per-task partials of an array run into the single results JSON.

    Each target is an INDEPENDENT unit (its own targets, its own three control
    loops), so sharding cannot change any per-target number; the stripe indices are
    stored and used to restore the single-process target ORDER, and the aggregate is
    computed by the same ``_aggregate`` the single-process path uses.
    """
    partdir = _partdir(budgets)
    parts = sorted(glob.glob(os.path.join(partdir, "part_*.json")))
    if not parts:
        raise FileNotFoundError(f"no partials in {partdir}")
    by_idx, nodes, commit, wall = {}, None, None, 0.0
    for pf in parts:
        with open(pf) as f:
            p = json.load(f)
        if p["n"] != n or p["seed"] != seed:
            raise ValueError(f"{os.path.basename(pf)}: (n,seed)=({p['n']},{p['seed']})"
                             f" != ({n},{seed}) — partials from a different run")
        if dict(p.get("budgets") or {}) != dict(budgets or {}):
            raise ValueError(f"{os.path.basename(pf)}: budgets {p.get('budgets')}"
                             f" != {budgets} — partials from a different run")
        nodes = nodes or p["n_solver_nodes"]
        commit = commit or p.get("model_git_commit")
        wall += float(p.get("wall_clock_s", 0.0))
        for i, r in zip(p["indices"], p["runs"]):
            if i in by_idx:
                raise ValueError(f"target {i} appears in two partials")
            by_idx[i] = r
    missing = [i for i in range(n) if i not in by_idx]
    if missing:
        raise ValueError(f"{len(missing)} target(s) missing from partials: "
                         f"{missing[:12]}{' ...' if len(missing) > 12 else ''}\n"
                         f"  re-submit those array indices before assembling")
    runs = [by_idx[i] for i in range(n)]
    print(f"[assemble] {len(parts)} partials -> {n} targets "
          f"(total task wall {wall/3600:.2f} h)", flush=True)
    return _finalize(runs, n, seed, n_solver_nodes=nodes, model_git_commit=commit,
                     wall_clock_s=wall, assembled_from=len(parts), budgets=budgets)


def main(n=25, seed=0, taskid=-1, ntasks=1, budgets=None):
    """Run the targeting study.  ``ntasks>1`` runs only this task's stripe of targets
    and writes a partial (merge with ``--assemble``); ``ntasks==1`` runs all and
    writes the results JSON directly."""
    os.makedirs(REPDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    sharded = ntasks > 1
    if sharded and not (0 <= taskid < ntasks):
        raise SystemExit(f"--taskid {taskid} out of range for --ntasks {ntasks}")
    t0 = time.time()
    print(f"[qc-target] load model + problem ...", flush=True)
    prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)
    model = T.load_model(MODEL, prob)
    print(f"[qc-target]   model={os.path.basename(MODEL)} nodes={model.n_solver_nodes}"
          f" axes={list(NAMES)} grid=({NA},{NB},{NPHI})", flush=True)
    meta = _check_model_box(model)

    # Every task draws the FULL target list from the same seed and takes a stride of
    # it, so target i is the same configuration at any --ntasks.
    stars = draw_targets(n, seed)
    idx = list(range(n))[taskid::ntasks] if sharded else list(range(n))
    if sharded:
        print(f"[qc-target] task {taskid}/{ntasks}: {len(idx)} of {n} targets "
              f"{idx[:6]}{' ...' if len(idx) > 6 else ''}", flush=True)

    runs = []
    for k, i in enumerate(idx):
        ts = stars[i]
        r = run_one(model, prob, ts, budgets=budgets)
        runs.append(r)
        tol = "/".join(str(r[m]["n_solves_to_tol"]) for m in METHODS)
        print(f"  [{k+1}/{len(idx)}] target {i}: b*={ts[0]:.2f} q*={ts[1]:.2f}  "
              f"solves cold/broy/grad = {r['cold']['n_solves']}/"
              f"{r['broyden']['n_solves']}/{r['gradient']['n_solves']}  "
              f"to_tol = {tol}  (elapsed {time.time()-t0:.0f}s)", flush=True)

    if sharded:
        partdir = _partdir(budgets)
        os.makedirs(partdir, exist_ok=True)
        out = os.path.join(partdir, f"part_{taskid:03d}.json")
        with open(out, "w") as f:
            json.dump(dict(n=n, seed=seed, taskid=taskid, ntasks=ntasks,
                           budgets=dict(budgets or {}),
                           indices=idx, runs=runs,
                           n_solver_nodes=int(model.n_solver_nodes),
                           model_git_commit=meta.get("git_commit"),
                           wall_clock_s=time.time() - t0), f, indent=2, default=float)
        print(f"[qc-target] task {taskid} DONE {time.time()-t0:.0f}s -> {out}",
              flush=True)
        return None

    summary = _finalize(runs, n, seed, n_solver_nodes=model.n_solver_nodes,
                        model_git_commit=meta.get("git_commit"),
                        wall_clock_s=time.time() - t0, budgets=budgets)
    print(f"[qc-target] DONE {time.time()-t0:.0f}s", flush=True)
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
    ap.add_argument("--taskid", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)),
                    help="array-mode stripe index (default: SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--ntasks", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)),
                    help="array-mode stripe count (default: SLURM_ARRAY_TASK_COUNT); "
                         "1 runs every target in this process")
    ap.add_argument("--assemble", action="store_true",
                    help="merge the per-task partials into the results JSON (no solves)")
    ap.add_argument("--budget-grad", type=int, default=0,
                    help="fixed-budget mode: certified solves for the gradient method "
                         "(0 = stop at tolerance, the default study)")
    ap.add_argument("--budget-bb", type=int, default=0,
                    help="fixed-budget mode: certified solves for both black-box "
                         "variants (0 = stop at tolerance)")
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
        budgets = {}
        if args.budget_grad:
            budgets["gradient"] = args.budget_grad
        if args.budget_bb:
            budgets["cold"] = args.budget_bb
            budgets["broyden"] = args.budget_bb
        budgets = budgets or None
        if args.assemble:
            assemble(n=args.n, seed=args.seed, budgets=budgets)
        else:
            main(n=args.n, seed=args.seed, taskid=args.taskid, ntasks=args.ntasks,
                 budgets=budgets)
