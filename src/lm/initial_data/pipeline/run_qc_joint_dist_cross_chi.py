"""LM-initial-data — JOINT held-out DISTRIBUTION (best/median/worst) for the
value+gradient+CROSS (full-bilinear Hermite-Smolyak) 4-D chi model, over the
Smolyak level.  The gradient-enhanced companion to run_qc_joint_dist_chi.py
(the BARE-interpolant left panel of paper Fig. 5).

Add-only.  Mirrors run_qc_joint_dist_chi.py EXACTLY -- SAME box d4_qc_chi_prod,
SAME 1000 seed-0 random off-node points (random_points, verbatim), SAME max-abs
held-out metric max|model(theta) - u_true(theta)|, SAME levels 1..5 -- but the
per-level model is the full-bilinear cross Hermite-Smolyak model (enhanced
chi_Ay,chi_By + the mixed 2nd partial), NOT the bare value interpolant.

Both panels are served from the SHARED committed corpus with ZERO new NODE solves:
the per-level cross sub-models are assembled from the committed L=5 cross corpus's
node pool (nested Clenshaw-Curtis: level-L nodes subset of the L=5 pool).  The only
cost is the 1000 direct reference solves (u_true), warm-started by the L=5 cross
model guess (modified-Newton to tol 1e-12 -- field-identical to a cold solve, and
to run_qc_joint_dist_chi's truth).

Data (read-only):
  reports/P2/models_chi/hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz
Writes:
  reports/3D_parametric/qc_chi/joint_dist_cross_d4_qc_chi_prod.json

Run (background, ~1-2 h for the 1000 truth solves):
  caffeinate -ims ~/Software/micromamba/micromamba run -n BBHFM python \\
      -m lm.initial_data.pipeline.run_qc_joint_dist_cross_chi
Smoke (a few min):
  ... run_qc_joint_dist_cross_chi.py --smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3
from lm.initial_data.parametric import parametric_nd_smolyak as sm
from lm.initial_data.pipeline import production_box as pb
from lm.initial_data.parametric.hermite_smolyak_cross import (
    load_hermite_smolyak_cross, build_cross_from_pool)

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REPDIR = os.path.join(REPORTS, "3D_parametric", "qc_chi")
CKDIR = os.path.join(REPDIR, "joint_dist_cross_parts")
os.makedirs(REPDIR, exist_ok=True)
os.makedirs(CKDIR, exist_ok=True)

NA, NB, NPHI = 44, 32, 8
QC = dict(pb.FIXED_QC)
D = 4
CHI = pb.CHI_MAX
CROSS_MODEL = os.path.join(REPORTS, "P2", "models_chi",
                           "hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz")

# box d4_qc_chi_prod
BOX = pb.aligned_box()


def _t(m): print(m, flush=True)


def random_points(box, n, seed):
    """VERBATIM from run_qc_joint_dist_chi.random_points -- identical points."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, len(box)))
    for j, a in enumerate(box):
        pts[:, j] = rng.uniform(a["min"], a["max"], n)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--levels", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    n_pts = args.n_points
    if args.smoke:
        n_pts = 12; levels = [1, 2, 5]
    t_start = time.time()
    names = [a["name"] for a in BOX]
    _t(f"=== joint held-out distribution (VALUE+GRAD+CROSS) {'(SMOKE)' if args.smoke else ''} ===")
    _t(f"    box=d4_qc_chi_prod axes={names}  n_points={n_pts}  levels={levels}")

    prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)
    solve_fn, _ = p3.make_solve_fn(prob, names, fixed=QC, solver="modified")

    # -- load the committed L=5 cross corpus; build per-level sub-models (no solves) --
    mc = load_hermite_smolyak_cross(CROSS_MODEL)
    axes, enh = mc.axes, tuple(mc.enhanced)
    pool = dict(mc.pool)
    _t(f"    cross corpus: {mc.n_solver_nodes} nodes, enhanced={enh}")
    models = {}
    for L in levels:
        iset = sm.isotropic_index_set(D, L)
        m = build_cross_from_pool(axes, iset, enh, pool)
        models[L] = (m, int(len(m._dedup_pool())))
        _t(f"    built cross L={L}: {models[L][1]} nodes")

    # -- 1000 held-out truth solves (warm-started by the L=5 cross model) --
    hold = random_points(BOX, n_pts, args.seed)
    ck = os.path.join(CKDIR, f"truth_seed{args.seed}_n{n_pts}.npz")
    U_dir = [None] * n_pts
    start = 0
    if os.path.exists(ck):
        z = np.load(ck)
        if z["pts"].shape == hold.shape and np.allclose(z["pts"], hold):
            for i in range(n_pts):
                if bool(z["done"][i]):
                    U_dir[i] = z["U"][i]
            start = int(np.argmin(z["done"])) if not z["done"].all() else n_pts
            _t(f"    resumed truth from checkpoint: {start}/{n_pts}")
    Ustack = np.array([(u if u is not None else np.zeros(prob.Ntot2d * NPHI))
                       for u in U_dir], dtype=float) if U_dir[0] is not None \
        else np.zeros((n_pts, prob.Ntot2d * NPHI))
    done = np.array([u is not None for u in U_dir])

    t0 = time.time()
    for i in range(start, n_pts):
        th = hold[i]
        guess = np.asarray(mc.evaluate(th))          # excellent warm start
        U, _info = solve_fn(th, guess, 1e-12, 30)
        Ua = np.asarray(U).ravel()
        U_dir[i] = Ua
        Ustack[i] = Ua
        done[i] = True
        if (i + 1) % 25 == 0 or i == n_pts - 1:
            np.savez(ck, pts=hold, U=Ustack, done=done)
            el = time.time() - t0
            rate = el / (i + 1 - start)
            _t(f"    truth {i+1}/{n_pts}  {el:.0f}s  {rate:.2f}s/pt  "
               f"ETA {rate*(n_pts-1-i)/60:.1f} min")

    # -- per-level distribution: max-abs held-out error vs u_true --
    joint = []
    for L in levels:
        m, n_lvl = models[L]
        errs = np.array([float(np.max(np.abs(np.asarray(m.evaluate(hold[i])).ravel()
                                             - U_dir[i]))) for i in range(n_pts)])
        rec = dict(level=L, nodes=n_lvl,
                   best=float(errs.min()), median=float(np.median(errs)),
                   worst=float(errs.max()),
                   p05=float(np.percentile(errs, 5)), p95=float(np.percentile(errs, 95)))
        joint.append(rec)
        _t(f"    L={L}: {n_lvl} nodes  best={rec['best']:.2e} "
           f"median={rec['median']:.2e} worst={rec['worst']:.2e}")

    results = dict(
        meta=dict(box="d4_qc_chi_prod", model="value+grad+cross", axes=names,
                  n_points=n_pts, seed=args.seed, levels=levels,
                  Na=NA, Nb=NB, Nphi=NPHI, enhanced=[names[e] for e in enh],
                  cross_model=os.path.basename(CROSS_MODEL),
                  wall_s=time.time() - t_start),
        joint=joint)
    tag = "_smoke" if args.smoke else ""
    out = os.path.join(REPDIR, f"joint_dist_cross_d4_qc_chi_prod{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _t(f"\nWrote {out}")
    _t(f"TOTAL {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
