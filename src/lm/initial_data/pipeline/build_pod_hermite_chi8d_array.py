"""S7 (array) — CHUNKED 8-D χ gradient-enhanced (Hermite) build over a SLURM array.

Add-only parallelization of ``build_pod_hermite_model_chi_8d.py``'s reuse-value
tangent loop.  That single-process driver computes the 6 QC certified tangents for
ALL ~15.7k dedup nodes serially (~50-130 h wall) — infeasible under a deadline.
This driver splits the tangent loop across a job array (each task solves a disjoint
node-stride, like ``run_8d_chi_array.py`` does for the value solves) and a MERGE mode
finalizes + POD-compresses + certified-spot-checks the shipped model.

The physics/solve/tangent/POD/certify are all REUSED byte-for-byte from the committed
``build_pod_hermite_model`` (and the 8-D box + 6-spin defaults from
``build_pod_hermite_model_chi_8d``, imported for its module-level ``bh.BOX`` swap).
No committed module is modified.

Modes
-----
  chunk:  --mode chunk --taskid K --ntasks N --reuse-value <S6 8D npz> \
          --partial-dir <dir> [--enhanced ...] [--Na 44 --Nb 32 --Nphi 8]
      Loads the value corpus, computes the enhanced-only QC tangents for the node
      stride ``keys[K::N]``, and writes ``<dir>/tangents_task<K>.npz`` (thetas, dUs).
      taskid/ntasks default to SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT
      (override --ntasks for a subset resubmit, exactly as run_8d_chi_array.py).

  merge:  --mode merge --reuse-value <S6 8D npz> --partial-dir <dir> \
          --outdir <dir> [--enhanced ...] [--Na .. --Nb .. --Nphi ..]
      Combines every partial into the full ``key -> (U, dU, iters, resid)`` pool,
      finalizes the HermiteSmolyakSolutionND (bit-for-bit the from-scratch build —
      same index_set/nodes/deterministic tangents), saves it, POD-compresses (H5d),
      and runs the certified spot-check (worst ‖R‖ must be ≤ 1e-10).

Smoke (end-to-end, tiny): build a small 8-D value corpus first, e.g.
  python build_surrogate_chi.py --Na 16 --Nb 12 --Nphi 6 --box spin8_qc_chi_prod \
      --level 2 --solver modified --outdir /tmp/smk
then chunk (--ntasks 2, K=0 and 1) + merge, and check the certified ‖R‖.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lm.initial_data.pipeline import build_pod_hermite_model as bh  # committed builder
from lm.initial_data.pipeline import build_pod_hermite_model_chi_8d as bh8d  # noqa: F401
# ^ imported for its module-level side effect: bh.BOX -> the 8-D production box.
# Must go through the package path so it mutates the SAME bh module object
# imported above (a bare sibling import creates a second, unrelated copy).

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3
from lm.initial_data.parametric import parametric_nd_smolyak as sm
from lm.initial_data.parametric import hermite_smolyak as hsm
from lm.initial_data.parametric import hermite_smolyak_pod as hpod
from lm.initial_data.parametric.parametric_nd_3d import make_solve_fn
from lm.initial_data.parametric.parametric_nd import _git_commit

_node_key = sm._node_key
SPIN_AXES = bh8d.SPIN_AXES  # "chi_Ax,chi_Ay,chi_Az,chi_Bx,chi_By,chi_Bz"


def _t(m):
    print(m, flush=True)


def _human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def _common_args(ap):
    ap.add_argument("--reuse-value", default=None,
                    help="path to the 8-D value-only SmolyakSolutionND .npz (S6); "
                         "required for chunk/merge, unused by pod mode")
    ap.add_argument("--enhanced", default=SPIN_AXES,
                    help="comma-separated enhanced axes (default the 6 chi spins)")
    ap.add_argument("--Na", type=int, default=44)
    ap.add_argument("--Nb", type=int, default=32)
    ap.add_argument("--Nphi", type=int, default=8)
    ap.add_argument("--tangent-jac", choices=("nk", "modified"), default="nk")
    ap.add_argument("--tangent-gmres-rtol", type=float, default=1e-8)


def _setup(args):
    """Shared setup: names/enhanced/prob + the loaded value model + its dedup pool."""
    names = [a["name"] for a in bh.BOX]
    enhanced = [s for s in (x.strip() for x in args.enhanced.split(",")) if s]
    for e in enhanced:
        if e not in names:
            raise ValueError(f"enhanced axis {e!r} not in 8-D box {names}")
    prob = s3.make_problem(Na=args.Na, Nb=args.Nb, Nphi=args.Nphi)
    vm = sm.load_smolyak(args.reuse_value)
    if list(vm.axes) and len(vm.axes) != len(names):
        raise ValueError(f"value model dim {len(vm.axes)} != 8-D box dim {len(names)} "
                         f"— wrong --reuse-value (must be the 8-D spin8 corpus)")
    pool_v = vm._dedup_pool()                          # key -> (theta, U, iters, resid)
    keys = list(pool_v)                                # deterministic (npz insertion order)
    return names, enhanced, prob, vm, pool_v, keys


def run_chunk(args):
    taskid = args.taskid if args.taskid is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ntasks = args.ntasks if args.ntasks is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    names, enhanced, prob, vm, pool_v, keys = _setup(args)
    mine = keys[taskid::ntasks]
    os.makedirs(args.partial_dir, exist_ok=True)
    tf = bh.enhanced_only_tangent_fn(prob, names, enhanced, bh.M_TOT, bh.FIXED,
                                     tangent_jac=args.tangent_jac,
                                     gmres_rtol=args.tangent_gmres_rtol)
    _t(f"[S7-chunk] task {taskid}/{ntasks}: {len(mine)} of {len(keys)} dedup nodes "
       f"(stride [{taskid}::{ntasks}]); enhanced={enhanced} (first node compiles ~85s)")
    thetas, dUs = [], []
    t0 = time.time()
    for i, k in enumerate(mine):
        theta, U, _it, _rs = pool_v[k]
        dU = np.asarray(tf(theta, U))                  # (d, *field), zeros off enhanced
        thetas.append(np.asarray(theta, dtype=float))
        dUs.append(dU.astype(np.float64))
        if i == 0 or (i + 1) % 10 == 0 or i == len(mine) - 1:
            el = time.time() - t0
            rate = el / (i + 1)
            _t(f"[S7-chunk] {i+1}/{len(mine)}  {el:.0f}s  ~{rate:.1f}s/node  "
               f"ETA {rate*(len(mine)-i-1):.0f}s")
    out = os.path.join(args.partial_dir, f"tangents_task{taskid}.npz")
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:                        # file handle → no .npz auto-append
        np.savez(fh, thetas=np.asarray(thetas), dUs=np.asarray(dUs),
                 taskid=taskid, ntasks=ntasks)
    os.replace(tmp, out)                               # atomic
    _t(f"[S7-chunk] task {taskid} DONE → {out}  ({_human(os.path.getsize(out))}, "
       f"{len(mine)} nodes)")


def run_merge(args):
    commit = _git_commit()
    names, enhanced, prob, vm, pool_v, keys = _setup(args)
    enh_idx = [names.index(e) for e in enhanced]
    t_start = time.time()

    # ----- gather partials -> full hermite pool {key: (U, dU, iters, resid)} -----
    import glob
    parts = sorted(glob.glob(os.path.join(args.partial_dir, "tangents_task*.npz")))
    if not parts:
        raise FileNotFoundError(f"no tangents_task*.npz in {args.partial_dir}")
    _t(f"[S7-merge] {len(parts)} partial files; value pool has {len(keys)} nodes")
    pool = {}
    for p in parts:
        d = np.load(p, allow_pickle=False)
        th, dU = d["thetas"], d["dUs"]
        for j in range(th.shape[0]):
            k = _node_key(th[j])
            if k not in pool_v:
                raise KeyError(f"partial {os.path.basename(p)} node {j} not in value pool "
                               f"(theta={th[j]}) — grid/box mismatch?")
            _theta, U, it, rs = pool_v[k]
            pool[k] = (np.asarray(U, dtype=float), np.asarray(dU[j], dtype=float),
                       int(it), float(rs))
    missing = [k for k in keys if k not in pool]
    if missing:
        raise RuntimeError(f"{len(missing)}/{len(keys)} nodes missing tangents "
                           f"(incomplete array — resubmit the missing chunk tasks)")
    _t(f"[S7-merge] full pool assembled: {len(pool)} nodes (all covered) "
       f"in {time.time()-t_start:.0f}s")

    # ----- finalize the gradient-enhanced sparse model (reuses committed _finalize) -----
    solve_fn, _ = make_solve_fn(prob, names, M_tot=bh.M_TOT, fixed=bh.FIXED,
                                use_cache=True, solver=args.solver)
    builder = hsm.HermiteSmolyakSolverND(solve_fn=None, axes=list(vm.axes),
                                         tangent_fn=None, enhanced_axes=enh_idx)
    model = builder._finalize([tuple(l) for l in vm.index_set], pool)
    model._solve_fn = solve_fn
    os.makedirs(args.outdir, exist_ok=True)
    base_meta = dict(axis_names=names, box=[[a["min"], a["max"]] for a in bh.BOX],
                     enhanced=enhanced, Na=args.Na, Nb=args.Nb, Nphi=args.Nphi,
                     solver=args.solver, tangent_jac=args.tangent_jac,
                     fixed=bh.FIXED, git_commit=commit, level=args.level,
                     note=args.note, built_by="build_pod_hermite_chi8d_array.py(merge)")
    tag = f"spin8qc_L{args.level}_enh-{'-'.join(enhanced)}"
    m_path = os.path.join(args.outdir, f"hermite_smolyak_{tag}.npz")
    model.save(m_path, meta=base_meta)
    _t(f"[S7-merge] model: {model.n_solver_nodes} nodes, field {model.field_shape}, "
       f"{_human(os.path.getsize(m_path))} → {m_path}")

    # ----- H5d POD compression -----
    _t(f"[S7-merge] POD compression (tail={args.pod_tail:.0e}) ...")
    t0 = time.time()
    pod, diag = hpod.build_pod_hermite_smolyak(model, tail=args.pod_tail,
                                               solve_fn=model._solve_fn)
    nfeat = int(np.prod(model.field_shape))
    r = pod.r
    _t(f"   rank_value={diag['rank_value']}  rank_stacked={diag['rank_stacked']}  "
       f"shipped r={r}  nfeat/r={nfeat}/{r}={nfeat/r:.1f}x  ({time.time()-t0:.1f}s)")
    p_path = os.path.join(args.outdir, f"pod_hermite_smolyak_{tag}.npz")
    pod.save(p_path, meta=base_meta)
    _t(f"   raw {_human(os.path.getsize(m_path))} → POD {_human(os.path.getsize(p_path))}")

    # ----- certified spot-check (POD guess + committed polish) -----
    _t("[S7-merge] certified spot-check (POD guess + polish) ...")
    hold = p3.holdout_points_nd([dict(a, Q=8) for a in bh.BOX], n_points=3)
    p3.assert_off_node(hold, [dict(a, Q=8) for a in bh.BOX])
    worst = 0.0
    for th in hold:
        _ = pod.evaluate(th)
        _U, info = pod.evaluate_polished(th, newton_steps=args.newton_steps, tol=1e-10)
        worst = max(worst, float(info.residual_norm))
        _t(f"   θ={[round(float(x),3) for x in th]}  certified‖R‖={info.residual_norm:.2e}")
    ok = worst <= 1e-10
    _t(f"   worst certified ‖R‖ over {len(hold)} off-node θ = {worst:.2e}"
       + ("  ✓ ≤ 1e-10" if ok else "  ✗ > 1e-10"))
    _t(f"[S7-merge] TOTAL {time.time()-t_start:.0f}s\n   raw: {m_path}\n   pod: {p_path}")
    if not ok:
        raise SystemExit(f"CERTIFICATION FAILED: worst ‖R‖={worst:.2e} > 1e-10")


def run_pod(args):
    """POD-compress + certify an ALREADY-SAVED raw Hermite model.  Memory-lean: skips
    the assemble (partials + pool + finalized-tensor duplication) that inflated the
    full-merge peak past 128 G → OOM.  Peak ≈ model (~13 GB) + POD SVD workspace."""
    commit = _git_commit()
    names = [a["name"] for a in bh.BOX]
    enhanced = [s for s in (x.strip() for x in args.enhanced.split(",")) if s]
    prob = s3.make_problem(Na=args.Na, Nb=args.Nb, Nphi=args.Nphi)
    _t(f"[S7-pod] loading raw model {os.path.basename(args.raw_model)} ...")
    model = hsm.load_hermite_smolyak(args.raw_model)
    solve_fn, _ = make_solve_fn(prob, names, M_tot=bh.M_TOT, fixed=bh.FIXED,
                                use_cache=True, solver=args.solver)
    model._solve_fn = solve_fn
    os.makedirs(args.outdir, exist_ok=True)
    base_meta = dict(axis_names=names, box=[[a["min"], a["max"]] for a in bh.BOX],
                     enhanced=enhanced, Na=args.Na, Nb=args.Nb, Nphi=args.Nphi,
                     solver=args.solver, fixed=bh.FIXED, git_commit=commit,
                     level=args.level, note=args.note,
                     built_by="build_pod_hermite_chi8d_array.py(pod)")
    tag = f"spin8qc_L{args.level}_enh-{'-'.join(enhanced)}"
    _t(f"[S7-pod] POD compression (tail={args.pod_tail:.0e}, randomized={args.randomized}) ...")
    t0 = time.time()
    pod, diag = hpod.build_pod_hermite_smolyak(model, tail=args.pod_tail,
                                               solve_fn=model._solve_fn,
                                               randomized=args.randomized)
    nfeat = int(np.prod(model.field_shape))
    r = pod.r
    _t(f"   rank_value={diag['rank_value']}  rank_stacked={diag['rank_stacked']}  "
       f"shipped r={r}  nfeat/r={nfeat}/{r}={nfeat/r:.1f}x  ({time.time()-t0:.1f}s)")
    p_path = os.path.join(args.outdir, f"pod_hermite_smolyak_{tag}.npz")
    pod.save(p_path, meta=base_meta)
    _t(f"   POD → {_human(os.path.getsize(p_path))}  {p_path}")

    _t("[S7-pod] certified spot-check (POD guess + polish) ...")
    hold = p3.holdout_points_nd([dict(a, Q=8) for a in bh.BOX], n_points=3)
    p3.assert_off_node(hold, [dict(a, Q=8) for a in bh.BOX])
    worst = 0.0
    for th in hold:
        _ = pod.evaluate(th)
        _U, info = pod.evaluate_polished(th, newton_steps=args.newton_steps, tol=1e-10)
        worst = max(worst, float(info.residual_norm))
        _t(f"   θ={[round(float(x),3) for x in th]}  certified‖R‖={info.residual_norm:.2e}")
    ok = worst <= 1e-10
    _t(f"   worst certified ‖R‖ over {len(hold)} off-node θ = {worst:.2e}"
       + ("  ✓ ≤ 1e-10" if ok else "  ✗ > 1e-10"))
    if not ok:
        raise SystemExit(f"CERTIFICATION FAILED: worst ‖R‖={worst:.2e} > 1e-10")
    _t(f"[S7-pod] DONE\n   pod: {p_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("chunk", "merge", "pod"), required=True)
    ap.add_argument("--raw-model", default=None,
                    help="(pod mode) path to the already-saved raw Hermite model to "
                         "POD-compress + certify (skips assemble; memory-lean)")
    ap.add_argument("--randomized", action="store_true",
                    help="(pod mode) randomized SVD (lower memory) if the exact POD OOMs")
    ap.add_argument("--partial-dir", default=None,
                    help="shared dir for the per-task tangent partials "
                         "(required for chunk/merge, unused by pod mode)")
    ap.add_argument("--taskid", type=int, default=None)
    ap.add_argument("--ntasks", type=int, default=None)
    ap.add_argument("--outdir", default=os.path.join(bh.HERE, "reports", "P2", "models_chi"))
    ap.add_argument("--level", type=int, default=5, help="Smolyak level of the value corpus (metadata)")
    ap.add_argument("--solver", choices=("nk", "modified"), default="nk")
    ap.add_argument("--pod-tail", type=float, default=1e-6)
    ap.add_argument("--newton-steps", type=int, default=5,
                    help="certify polish Newton steps (8D sparse guess needs >2; "
                         "build_surrogate's 2 left the value model at ~1e-9)")
    ap.add_argument("--note", default="")
    _common_args(ap)
    args = ap.parse_args()
    if args.mode in ("chunk", "merge") and (not args.reuse_value or not args.partial_dir):
        ap.error(f"--mode {args.mode} requires --reuse-value and --partial-dir")
    if args.mode == "chunk":
        run_chunk(args)
    elif args.mode == "pod":
        if not args.raw_model:
            ap.error("--raw-model is required for --mode pod")
        run_pod(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
