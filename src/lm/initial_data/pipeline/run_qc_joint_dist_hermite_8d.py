"""PARASOL — JOINT held-out DISTRIBUTION (best/median/worst) for the 8-D
VALUE+GRADIENT model, over the Smolyak level.

Per the user directive, the 8-D "value+gradient" model IS the y-pair CROSS model:
value + gradient in the spin-y pair (chi_Ay, chi_By) + the mixed 2nd partial
∂²U/∂χ_Ay∂χ_By.  So this is the 8-D analog of the committed 4-D
run_qc_joint_dist_cross_chi.py (the CROSS joint-dist), pointed at the 8-D y-pair
cross model — NOT the plain 6-spin gradient Hermite.  Produces figure-data source
``joint_dist_hermite_8d`` pinned by manuscript/figures/registry.py to EXACTLY:
    reports/3D_parametric/qc_chi/joint_dist_hermite_spin8_qc_chi_b27.json

FAITHFUL + CHEAP:
  * SAME box spin8_qc_chi_b27, SAME 1000 seed-0 ``random_points`` as
    run_qc_joint_dist_chi.py (b, the BARE value panel) -> the SAME held-out points,
    and we REUSE b's certified truth (``joint_dist_truth_parts/truth_spin8_qc_chi_
    b27_seed0_n1000_task*.npz``) -> ZERO new solves.
  * SAME max-abs held-out metric max|model(theta) - u_true(theta)|, levels 1..5,
    SAME per-level assembly (build_cross_from_pool from the L=5 cross corpus's node
    pool; nested Clenshaw-Curtis => level-L nodes subset of the L=5 pool) as the 4-D
    cross driver.
  * MEMORY-FRUGAL: builds ONE per-level CROSS sub-model at a time (an 8-D L=5
    cross sub-model is ~85 GB; holding all five like the 4-D template would OOM the
    256 GB node) — build, evaluate at all points, free, next level.

Add-only.  Imports committed ``parasol`` modules read-only; edits nothing.

Run (compute node, via sbatch):
  DRIVER=run_qc_joint_dist_hermite_8d.py sbatch ... slurm/ivs/submit_parasol_cpu_hi.slurm
Smoke:  ARGS="--smoke"
"""
from __future__ import annotations
import argparse, gc, glob, itertools, json, os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from lm.initial_data.parametric.parametric_nd_smolyak import isotropic_index_set
from lm.initial_data.parametric.hermite_smolyak_cross import (
    load_hermite_smolyak_cross, build_cross_from_pool)
# SAME box + SAME points + SAME truth-part location as the value panel (b)
from lm.initial_data.pipeline.run_qc_joint_dist_chi import BOXES, random_points, PARTDIR

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "3D_parametric", "qc_chi")
MODELS = os.path.join(HERE, "reports", "P2", "models_chi")
# the 8-D y-pair CROSS model (value + gradient in chi_Ay,chi_By + cross term)
CROSS_8D = os.path.join(MODELS, "hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross.npz")
BOX_NAME = "spin8_qc_chi_b27"
NA, NB, NPHI = 44, 32, 8


def _t(m): print(m, flush=True)


def _rss():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return f"peak={int(line.split()[1])/1e6:.1f}G"
    except Exception:
        pass
    return ""


def load_truth_parts(box_name, seed, n_pts):
    """Reuse b's certified truth (identical seed-0 points)."""
    tag = f"{box_name}_seed{seed}_n{n_pts}"
    parts = sorted(glob.glob(os.path.join(PARTDIR, f"truth_{tag}_task*.npz")))
    if not parts:
        raise SystemExit(f"no truth parts for {tag} in {PARTDIR}; run (b) first")
    U_dir = [None] * n_pts
    got = 0
    for pf in parts:
        d = np.load(pf)
        for j, idx in enumerate(d["idx"]):
            U_dir[int(idx)] = np.asarray(d["U"][j]); got += 1
    missing = [i for i, u in enumerate(U_dir) if u is None]
    if missing:
        raise SystemExit(f"{len(missing)} truth solves missing (e.g. {missing[:5]}) "
                         f"from {len(parts)} parts")
    _t(f"    reused {got} certified truth fields from {len(parts)} parts")
    return U_dir


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
        n_pts = 12; levels = [1, 2]
    t_start = time.time()

    box = BOXES[BOX_NAME]
    names = [a["name"] for a in box]
    D = len(box)
    _t(f"=== joint held-out distribution (VALUE+GRAD y-pair CROSS, 8-D) "
       f"{'(SMOKE)' if args.smoke else ''} ===")
    _t(f"    box={BOX_NAME} axes={names}  n_points={n_pts}  levels={levels}")

    # -- load the 8-D y-pair CROSS node pool (value + gradient(chi_Ay,chi_By) + cross) --
    _t(f"    loading 8-D cross model ({os.path.getsize(CROSS_8D)/1e9:.1f} GB) ...")
    mc = load_hermite_smolyak_cross(CROSS_8D)
    axes = list(mc.axes)
    enh = tuple(mc.enhanced)
    pool = dict(mc.pool)
    _t(f"    cross corpus: {mc.n_solver_nodes} nodes  enhanced={enh}="
       f"{[names[e] for e in enh]}  cross_pairs={mc.cross_pairs_global}  {_rss()}")
    del mc; gc.collect()

    # -- SAME held-out points as (b); REUSE (b)'s certified truth --
    hold = random_points(box, n_pts, args.seed)
    U_dir = None if args.smoke else load_truth_parts(BOX_NAME, args.seed, n_pts)

    # -- per-level CROSS sub-model (FRUGAL: one at a time) --
    joint = []
    for L in levels:
        t0 = time.time()
        iset = isotropic_index_set(D, L)
        m = build_cross_from_pool(axes, iset, enh, pool)
        n_lvl = int(len(m._dedup_pool()))
        if U_dir is None:
            _ = np.asarray(m.evaluate(hold[0]))    # smoke: exercise evaluate
            _t(f"    [smoke] built+evaluated L={L}: {n_lvl} nodes  {_rss()}  "
               f"[{time.time()-t0:.0f}s]")
            del m; gc.collect(); continue
        errs = np.array([float(np.max(np.abs(np.asarray(m.evaluate(hold[i])).ravel()
                                             - U_dir[i].ravel()))) for i in range(n_pts)])
        rec = dict(level=L, nodes=n_lvl,
                   best=float(errs.min()), median=float(np.median(errs)),
                   worst=float(errs.max()),
                   p05=float(np.percentile(errs, 5)), p95=float(np.percentile(errs, 95)))
        joint.append(rec)
        _t(f"    L={L}: {n_lvl} nodes  best={rec['best']:.2e} median={rec['median']:.2e} "
           f"worst={rec['worst']:.2e}  {_rss()}  [{time.time()-t0:.0f}s]")
        del m; gc.collect()

    if args.smoke:
        _t(f"\n[SMOKE] per-level cross build+evaluate ran in {time.time()-t_start:.0f}s — "
           f"safe to launch full.")
        return

    results = dict(
        meta=dict(box=BOX_NAME, model="value+grad y-pair cross", axes=names,
                  n_points=n_pts, seed=args.seed, levels=levels, Na=NA, Nb=NB, Nphi=NPHI,
                  enhanced=[names[e] for e in enh],
                  cross_model=os.path.basename(CROSS_8D),
                  cross_pairs=[[names[a], names[b]]
                               for (a, b) in itertools.combinations(sorted(enh), 2)],
                  truth="reused from joint_dist_8d (b) truth parts",
                  wall_s=time.time() - t_start),
        joint=joint)
    out = os.path.join(REPDIR, f"joint_dist_hermite_{BOX_NAME}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _t(f"\nWrote {out}")
    _t(f"TOTAL {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
