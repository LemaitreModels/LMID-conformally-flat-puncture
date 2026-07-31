"""S6 — shipped 8D χ Smolyak L=5 (production box): SLURM job-array node-pool chunking.

The 8D ℓ=5 node pool (~15k nodes) is multi-day serial, so it is parallelised as a
SLURM job ARRAY: each task solves a DISJOINT STRIDE of the unique L=5 node pool
into the SHARED, lock-free ``solve_store_chi`` (atomic temp+os.replace writes, so
concurrent tasks are safe and dedup).  After the array completes, assemble the
model with the store fully populated (all hits):

    python -m lm.initial_data.pipeline.build_surrogate_chi --Na 44 --Nb 32 --Nphi 8 \
        --box spin8_qc_chi_prod --level 5 --solver modified --retry-tol 1e-6 \
        --store reports/3D_parametric/solve_store_chi \
        --code-tag chi-rebuild --outdir reports/3D_parametric/models_chi

Chunking is by stride (task k solves nodes[k::ntasks]), so tasks are disjoint (no
duplicate solves); the store additionally serves the in-plane=0 sub-slice from the
shared 4D corpus (S3), a free bonus.  Enumeration matches build_isotropic's node
set exactly (same isotropic_index_set + subgrid nodes + _node_key dedup), so the
array populates precisely the nodes the assembly needs.

Modes:
  --print-plan     enumerate + print the L=5 node count, then exit (no solves).
  --limit N        smoke: solve only the first N nodes of this task's stride.
  (default)        solve this task's full stride.

Task id / count come from ``SLURM_ARRAY_TASK_ID`` / ``SLURM_ARRAY_TASK_COUNT``
(overridable with --taskid / --ntasks).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import product

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import solve_store as ss
from lm.initial_data.parametric.parametric_nd_smolyak import isotropic_index_set, _node_key
from lm.initial_data.pipeline import production_box as pb

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
STORE = os.path.join(REPORTS, "3D_parametric", "solve_store_chi")
NA, NB, NPHI = 44, 32, 8
CHI = pb.CHI_MAX
CODE_TAG = "chi-rebuild"
FIXED = dict(pb.FIXED_QC)
BOX = pb.spin8_box()


def _t(m):
    print(m, flush=True)


def enumerate_nodes(solver, level):
    """The unique L=level Smolyak node θ-list for the 8D box (order-independent;
    the same set build_isotropic(level) would solve)."""
    seen = {}
    for l in isotropic_index_set(solver.d, level):
        nodes, _ = solver._subgrid_nodes(l)
        for combo in product(*nodes):
            th = np.array(combo, dtype=float)
            k = _node_key(th)
            if k not in seen:
                seen[k] = th
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--taskid", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--ntasks", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--print-plan", action="store_true",
                    help="enumerate + print node count, then exit (no solves)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: solve only the first N nodes of this task's stride")
    args = ap.parse_args()

    prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)
    store = ss.SolveStore(STORE, grid_meta=(NA, NB, NPHI), code_tag=CODE_TAG, reuse_tol=1e-6)
    solver = ss.from_problem_smolyak_3d_cached(prob, BOX, store=store, fixed=FIXED,
                                               solver="modified", retry_tol=1e-6)

    t0 = time.time()
    thetas = enumerate_nodes(solver, args.level)
    _t(f"[S6] L={args.level} 8D box b∈[{pb.B_MIN:g},{pb.B_MAX:g}]: {len(thetas)} unique nodes "
       f"(enumerated in {time.time()-t0:.0f}s)  store has {store.n_entries} entries")
    if args.print_plan:
        _t(f"[S6 PLAN] node_count={len(thetas)}")
        return

    mine = thetas[args.taskid::args.ntasks]
    if args.limit is not None:
        mine = mine[:args.limit]
    _t(f"[S6] task {args.taskid}/{args.ntasks}: solving {len(mine)} nodes "
       f"(stride [{args.taskid}::{args.ntasks}]{', limit '+str(args.limit) if args.limit else ''})")
    t0 = time.time()
    for i, th in enumerate(mine):
        _U, _info = solver.solve_fn(th, None, 1e-12, 40)
        if (i + 1) % 20 == 0 or i == len(mine) - 1:
            el = time.time() - t0
            rate = el / (i + 1)
            _t(f"[S6] task {args.taskid}: {i+1}/{len(mine)}  {rate:.1f}s/node  "
               f"store {store.n_hits}h/{store.n_misses}m  ETA {rate*(len(mine)-i-1):.0f}s")
    _t(f"[S6] task {args.taskid} DONE: {len(mine)} nodes, "
       f"{store.n_hits} hits / {store.n_misses} misses, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
