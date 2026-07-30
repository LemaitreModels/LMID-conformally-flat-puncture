"""PARASOL — JOINT held-out DISTRIBUTION (best/median/worst) over 1000 random
representative binaries vs Smolyak node count, DIMENSIONLESS-spin (chi) model
(paper revision R1 for the joint convergence figure).

The committed joint blocks (run_qc_walls_sweep_chi.py block_D, run_qc_dense_stats.py,
run_qc_wide_build_stats.py) report the WORST held-out error over a handful of points
(6-50).  R1 (notes/paper_revision_2.md line 15) asks for the full DISTRIBUTION
(best/median/worst) over ~1000 random points.  This is cheap for the JOINT model
because the model is built ONCE per Smolyak level from the shared solve corpus
(all store hits: S3 populated the 4D d4_qc_chi_b27 L=5 pool, S6 the 8D one), so the
only new cost is the ~1000 direct reference solves (shared across all levels).

For each Smolyak level L the model is (re)built from the store and evaluated at the
same 1000 random off-node points; the per-point held-out errors
|model.evaluate(theta) - truth(theta)| give best/median/worst at that level's node
count.

Boxes (from build_surrogate_chi.py, verbatim):
  d4_qc_chi_b27    = (b in [2,7], q in [1,3], chi_Ay, chi_By in [-0.99,0.99])   [S3 -- READY]
  spin8_qc_chi_b27 = (b, q, chi_Ax..chi_Bz in [-0.99,0.99])                     [S6 -- gated on assembly]

Modes:
  (default)   1000 truth solves + build L=1..5 from store + best/median/worst.
  --smoke     8 points, levels 1-2 (~a few min).

Run (single node; ~1000 modified-Newton truth solves ~2-4 h):
  sbatch --time=12:00:00 \
    --export=ALL,DRIVER=run_qc_joint_dist_chi.py,ARGS=--box d4_qc_chi_b27,\
JOB_DIR=sandbox/parasol/reports/3D_parametric/qc_chi/_mark_jointdist \
    slurm/ivs/submit_parasol_cpu_hi.slurm
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3
from lm.initial_data.parametric import solve_store as ss

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "3D_parametric", "qc_chi")
STORE = os.path.join(HERE, "reports", "3D_parametric", "solve_store_chi")
PARTDIR = os.path.join(REPDIR, "joint_dist_truth_parts")  # array truth-solve partials
os.makedirs(REPDIR, exist_ok=True)
os.makedirs(PARTDIR, exist_ok=True)

NA, NB, NPHI = 44, 32, 8
CODE_TAG = "chi-rebuild"
QC = {"qc": 1.0}
CHI = 0.99

# boxes verbatim from build_surrogate_chi.py (production b in [2,7])
BOXES = {
    "d4_qc_chi_b27": [
        {"name": "b", "min": 2.0, "max": 7.0},
        {"name": "q", "min": 1.0, "max": 3.0},
        {"name": "chi_Ay", "min": -CHI, "max": CHI},
        {"name": "chi_By", "min": -CHI, "max": CHI},
    ],
    "spin8_qc_chi_b27": [
        {"name": "b", "min": 2.0, "max": 7.0},
        {"name": "q", "min": 1.0, "max": 3.0},
        {"name": "chi_Ax", "min": -CHI, "max": CHI},
        {"name": "chi_Ay", "min": -CHI, "max": CHI},
        {"name": "chi_Az", "min": -CHI, "max": CHI},
        {"name": "chi_Bx", "min": -CHI, "max": CHI},
        {"name": "chi_By", "min": -CHI, "max": CHI},
        {"name": "chi_Bz", "min": -CHI, "max": CHI},
    ],
}


def _t(m): print(m, flush=True)


def random_points(box, n, seed):
    """n genuinely random uniform points in the box (off-node w.p. 1)."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, len(box)))
    for j, a in enumerate(box):
        pts[:, j] = rng.uniform(a["min"], a["max"], n)
    return pts


def _distribution(prob, store, box, levels, hold, U_dir):
    """Build the Smolyak model per level (store hits) and evaluate at the held-out
    points -> best/median/worst joint held-out error per level."""
    joint = []
    for L in levels:
        store.n_hits = store.n_misses = 0
        t0 = time.time()
        smsolver = ss.from_problem_smolyak_3d_cached(prob, box, store=store, fixed=QC,
                                                     solver="modified", retry_tol=1e-6)
        sm = smsolver.build_isotropic(L, tol=1e-12, max_iter=30)
        errs = np.array([float(np.max(np.abs(sm.evaluate(th) - U_dir[i])))
                         for i, th in enumerate(hold)])
        rec = dict(level=L, nodes=int(sm.n_solver_nodes),
                   best=float(errs.min()), median=float(np.median(errs)),
                   worst=float(errs.max()),
                   p05=float(np.percentile(errs, 5)), p95=float(np.percentile(errs, 95)),
                   store_hits=int(store.n_hits), store_misses=int(store.n_misses))
        joint.append(rec)
        _t(f"   L={L}: {rec['nodes']} nodes  best={rec['best']:.2e} "
           f"median={rec['median']:.2e} worst={rec['worst']:.2e}  "
           f"(store {rec['store_hits']}h/{rec['store_misses']}m)  [{time.time()-t0:.0f}s]")
    return joint


def _write_results(box_name, names, n_pts, seed, levels, joint, t_start):
    results = dict(meta=dict(box=box_name, axes=names, n_points=n_pts, seed=seed,
                            levels=levels, Na=NA, Nb=NB, Nphi=NPHI, code_tag=CODE_TAG,
                            wall_s=time.time() - t_start),
                   joint=joint)
    out = os.path.join(REPDIR, f"joint_dist_{box_name}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _t(f"\nWrote {out}")
    make_figure(results)
    _t(f"TOTAL {time.time()-t_start:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="d4_qc_chi_b27", choices=list(BOXES))
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--levels", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--taskid", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    ap.add_argument("--ntasks", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = ap.parse_args()

    box = BOXES[args.box]
    names = [a["name"] for a in box]
    levels = [int(x) for x in args.levels.split(",")]
    n_pts = args.n_points
    if args.smoke:
        n_pts = 8; levels = [1, 2]
    t_start = time.time()
    hold = random_points(box, n_pts, args.seed)
    part_tag = f"{args.box}_seed{args.seed}_n{n_pts}"

    # -- assemble: load array truth partials -> distribution + figure --
    if args.assemble:
        _t(f"=== joint dist ASSEMBLE  box={args.box}  n_points={n_pts} ===")
        prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)
        store = ss.SolveStore(STORE, grid_meta=(NA, NB, NPHI), code_tag=CODE_TAG, reuse_tol=1e-6)
        U_dir = [None] * n_pts
        parts = sorted(glob.glob(os.path.join(PARTDIR, f"truth_{part_tag}_task*.npz")))
        got = 0
        for pf in parts:
            d = np.load(pf)
            for j, idx in enumerate(d["idx"]):
                U_dir[int(idx)] = d["U"][j]; got += 1
        missing = [i for i, u in enumerate(U_dir) if u is None]
        if missing:
            _t(f"[assemble] ERROR: {len(missing)} truth solves missing "
               f"(e.g. {missing[:5]}) from {len(parts)} parts"); sys.exit(1)
        _t(f"[assemble] loaded {got} truth fields from {len(parts)} parts")
        joint = _distribution(prob, store, box, levels, hold, U_dir)
        _write_results(args.box, names, n_pts, args.seed, levels, joint, t_start)
        return

    array_mode = args.taskid >= 0 and args.ntasks > 1 and not args.smoke
    _t(f"=== joint held-out distribution {'(SMOKE)' if args.smoke else ''} "
       f"{'(truth array %d/%d)'%(args.taskid,args.ntasks) if array_mode else ''} ===")
    _t(f"    box={args.box} axes={names}  n_points={n_pts}  levels={levels}")
    prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)
    solve_fn, _ = p3.make_solve_fn(prob, names, fixed=QC, solver="modified")

    # -- array mode: solve a stride of truth points -> npz partial (no models) --
    if array_mode:
        mine = list(range(args.taskid, n_pts, args.ntasks))
        _t(f"    truth stride: {len(mine)} points [{args.taskid}::{args.ntasks}]")
        t0 = time.time(); Us = []
        for c, i in enumerate(mine):
            U, _ = solve_fn(hold[i].astype(float), None, 1e-12, 30)
            Us.append(np.asarray(U))
            if (c + 1) % 20 == 0 or c == len(mine) - 1:
                rate = (time.time() - t0) / (c + 1)
                _t(f"      {c+1}/{len(mine)}  {rate:.1f}s/solve  ETA {rate*(len(mine)-c-1):.0f}s")
        part = os.path.join(PARTDIR, f"truth_{part_tag}_task{args.taskid}.npz")
        np.savez(part, idx=np.array(mine), U=np.array(Us))
        _t(f"\n[array {args.taskid}/{args.ntasks}] wrote {part} "
           f"({len(mine)} solves)  [{time.time()-t0:.0f}s]")
        return

    # -- single-node (or smoke): all truth + distribution + figure --
    store = ss.SolveStore(STORE, grid_meta=(NA, NB, NPHI), code_tag=CODE_TAG, reuse_tol=1e-6)
    _t(f"    store: {store.n_entries} entries, code_tag={store.code_tag}")
    _t(f"    computing {n_pts} reference solves ...")
    U_dir = []; t0 = time.time()
    for i, th in enumerate(hold):
        U, _ = solve_fn(th.astype(float), None, 1e-12, 30)
        U_dir.append(np.asarray(U))
        if (i + 1) % 50 == 0 or i == n_pts - 1:
            rate = (time.time() - t0) / (i + 1)
            _t(f"      truth {i+1}/{n_pts}  {rate:.1f}s/solve  ETA {rate*(n_pts-i-1):.0f}s")
    joint = _distribution(prob, store, box, levels, hold, U_dir)
    if args.smoke:
        _t(f"\n[SMOKE] ran without error in {time.time()-t_start:.0f}s — safe to launch full.")
        return
    _write_results(args.box, names, n_pts, args.seed, levels, joint, t_start)


def make_figure(results):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        _t(f"[fig] matplotlib unavailable ({e})"); return
    J = results["joint"]; box = results["meta"]["box"]
    nodes = [r["nodes"] for r in J]
    best = [r["best"] for r in J]; med = [r["median"] for r in J]; worst = [r["worst"] for r in J]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.fill_between(nodes, best, worst, color="#4477aa", alpha=0.2,
                    label="best..worst")
    ax.semilogy(nodes, worst, "^--", color="#4477aa", alpha=0.8, label="worst")
    ax.semilogy(nodes, med, "o-", color="#222", label="median")
    ax.semilogy(nodes, best, "v--", color="#4477aa", alpha=0.8, label="best")
    ax.set_xlabel("solver node count")
    ax.set_ylabel(f"joint held-out error over {results['meta']['n_points']} points")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REPDIR, f"fig_qc_joint_dist_{box}.png"), dpi=160)
    plt.close(fig)
    _t(f"[fig] wrote fig_qc_joint_dist_{box}.png to {REPDIR}")


if __name__ == "__main__":
    main()
