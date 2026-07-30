"""LM-initial-data — build & persist the production parametric surrogates as reusable artifacts.

Turns the in-memory sweeps (which build an interpolant, measure accuracy, then
DISCARD it) into standalone on-disk models — "we provide the solution, no need to
solve".  Builds both production models over the §5b family and saves each to a
single numpy ``.npz`` (numpy-only, no pickle):

  * **sparse** — isotropic Smolyak sparse grid at ``--level`` (default 3 → 137
    solver nodes at d=4), stored as the DEDUPLICATED node pool (~12 MiB, not the
    overlapping subgrids);
  * **dense** — the matched tensor-product Chebyshev model at ``--dense-Q``
    (default 5 → 1296 nodes at d=4), one full tensor.

Both reload as standalone predictors: ``evaluate`` needs only numpy + the
parametric modules (~10 ms); ``evaluate_polished`` reaches certified ‖R‖∞≤1e-10
after reattaching a solver (``parametric_nd.attach_solve_fn_3d``).

The default box is ``d4`` = the §5b production family ``(b, |S|, θ_S, q)``.

ETA (Na=44, Nb=32, Nφ=8): sparse L=3 ≈ 137 solves (~8 min); dense Q=5 ≈ 1296
solves (~70 min) — the dense build dominates.  Omit ``--dense-Q`` to skip it.
Run in the background under caffeinate:

    caffeinate -ims ~/micromamba/envs/BBHFM/bin/python -m lm.initial_data.pipeline.build_surrogate \
        --Na 44 --Nb 32 --Nphi 8 --box d4 --level 3 --dense-Q 5 --solver nk
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3
from lm.initial_data.parametric import parametric_nd_smolyak as sm
from lm.initial_data.parametric.parametric_nd import (
    load_parametric, attach_solve_fn_3d, _git_commit,
)
from lm.initial_data.parametric import solve_store as ss

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(HERE, "reports", "3D_parametric", "models")
DEFAULT_STORE = os.path.join(HERE, "reports", "3D_parametric", "solve_store")

# The §5b production families.
BOXES = {
    "d4": [{"name": "b", "min": 1.5, "max": 4.0},
           {"name": "S_mag", "min": 0.0, "max": 0.4},
           {"name": "theta_S", "min": 0.0, "max": 90.0},
           {"name": "q", "min": 1.0, "max": 3.0}],
    "d3": [{"name": "b", "min": 1.5, "max": 4.0},
           {"name": "theta_S", "min": 0.0, "max": 90.0},
           {"name": "q", "min": 1.0, "max": 3.0}],
    # genuine 8-D: separation, mass ratio, and BOTH full spin vectors (all
    # directions).  Head-on infall (P, P_x at defaults).  b first → D7 per-b cache.
    "spin8": [{"name": "b", "min": 1.5, "max": 4.0},
              {"name": "q", "min": 1.0, "max": 3.0},
              {"name": "S_Ax", "min": -0.4, "max": 0.4},
              {"name": "S_Ay", "min": -0.4, "max": 0.4},
              {"name": "S_Az", "min": -0.4, "max": 0.4},
              {"name": "S_Bx", "min": -0.4, "max": 0.4},
              {"name": "S_By", "min": -0.4, "max": 0.4},
              {"name": "S_Bz", "min": -0.4, "max": 0.4}],
    # 4-D aligned-spin QUASI-CIRCULAR workhorse (b, q, χ_A∥=S_Ay, χ_B∥=S_By).
    # Momenta are the deterministic PN quasi-circular momenta (FIXED["d4_qc"] =
    # {"qc": 1.0} → theta_to_slice3d's QC branch), NOT head-on infall.  Ranges are
    # chosen so d4_qc is EXACTLY the in-plane-spin=0 slice of spin8: same b,q ranges,
    # aligned axes S_Ay/S_By over the same [-0.4,0.4] and symmetric about 0 (their
    # nested level-0 midpoint is exactly 0), so the future 8-D precessing QC build
    # reuses this corpus node-for-node.  b first → D7 per-b cache.
    "d4_qc": [{"name": "b",    "min": 1.5, "max": 4.0},
              {"name": "q",    "min": 1.0, "max": 3.0},
              {"name": "S_Ay", "min": -0.4, "max": 0.4},
              {"name": "S_By", "min": -0.4, "max": 0.4}],
    # genuine 8-D PRECESSING QUASI-CIRCULAR: separation, mass ratio, and BOTH full
    # spin vectors (all directions), with the deterministic PN quasi-circular momenta
    # (FIXED["spin8_qc"] = {"qc": 1.0} → theta_to_slice3d's QC branch), NOT head-on
    # infall.  Same axes/ranges as the head-on `spin8` box; the in-plane spins
    # (S_Ax,S_Az,S_Bx,S_Bz) are symmetric about 0 so their nested level-0 midpoint is
    # exactly 0.  Hence spin8_qc's in-plane=0 sub-slice coincides node-for-node with
    # the d4_qc corpus: at in-plane=0 the QC branch builds S_A=(0,S_Ay,0),
    # S_B=(0,S_By,0) and the QC momenta depend only on the y-components ⇒ byte-
    # identical Slice3D ⇒ same store key.  Pin --code-tag fb4f07f so the shared
    # d4_qc nodes are served from the store.  b first → D7 per-b cache.
    "spin8_qc": [{"name": "b",    "min": 1.5, "max": 4.0},
                 {"name": "q",    "min": 1.0, "max": 3.0},
                 {"name": "S_Ax", "min": -0.4, "max": 0.4},
                 {"name": "S_Ay", "min": -0.4, "max": 0.4},
                 {"name": "S_Az", "min": -0.4, "max": 0.4},
                 {"name": "S_Bx", "min": -0.4, "max": 0.4},
                 {"name": "S_By", "min": -0.4, "max": 0.4},
                 {"name": "S_Bz", "min": -0.4, "max": 0.4}],
}
FIXED = {"d4": None, "d3": {"S_mag": 0.3}, "spin8": None, "d4_qc": {"qc": 1.0},
         "spin8_qc": {"qc": 1.0}}


def _t(m):
    print(m, flush=True)


def _human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def _reload_spotcheck(loader, path, box, seed, tag):
    """Reload the artifact and verify ``evaluate`` round-trips at a generic θ."""
    model = loader(path)
    rng = np.random.default_rng(seed)
    th = np.array([a["min"] + (a["max"] - a["min"]) * rng.random() for a in box])
    v = model.evaluate(th)
    assert np.all(np.isfinite(v)), f"{tag}: reloaded evaluate produced non-finite output"
    _t(f"   [{tag}] reload OK — evaluate(θ) finite, shape {np.asarray(v).shape}, "
       f"field_shape {model.field_shape}")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Na", type=int, default=44)
    ap.add_argument("--Nb", type=int, default=32)
    ap.add_argument("--Nphi", type=int, default=8)
    ap.add_argument("--box", choices=sorted(BOXES), default="d4")
    ap.add_argument("--level", type=int, default=3, help="isotropic Smolyak level (sparse)")
    ap.add_argument("--dense-Q", type=int, default=None,
                    help="dense tensor Q per axis; omit to SKIP the (expensive) dense build")
    ap.add_argument("--solver", choices=("nk", "modified"), default="nk",
                    help="'nk' (certified) or 'modified' (cheaper, field-identical)")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--gmres-rtol", type=float, default=1e-4,
                    help="NK inexact-Newton forcing term (GMRES rtol); default 1e-4 "
                         "is the documented stall-free sweet spot. Tighter (e.g. 1e-8) "
                         "gives more exact Newton steps — more robust outer convergence "
                         "on hard cold slices — at more GMRES iters/step (bounded by "
                         "gmres_atol=1e-12 and maxiter=60).")
    ap.add_argument("--retry-tol", type=float, default=None,
                    help="damped-Newton globalization threshold for the NK build "
                         "(default OFF): if an NK node solve stagnates above this "
                         "residual (global-convergence failure at an extreme corner, "
                         "e.g. b=7 + high q + strong spin), reach the basin with a "
                         "cold modified-Newton solve then NK-polish from it to the "
                         "certified floor. Recommended 1e-6 for wide/strong-spin boxes.")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--note", default="",
                    help="free-text note stored in the artifact metadata")
    ap.add_argument("--store", nargs="?", const=DEFAULT_STORE, default=None,
                    help="route builds through a persistent content-addressed "
                         "SolveStore so shared/nested nodes are reused across "
                         "builds (default off; bare --store uses "
                         "reports/3D_parametric/solve_store/)")
    ap.add_argument("--reuse-tol", type=float, default=1e-6,
                    help="store reuse-admission threshold: a cached entry is "
                         "reused iff its achieved residual ≤ this (decoupled from "
                         "--tol; default 1e-6 admits deeply-converged "
                         "modified/NK fields whose monitor reads ~1e-9)")
    ap.add_argument("--code-tag", default=None,
                    help="override the store's code_tag (default: the current git "
                         "HEAD via _git_commit()).  The code_tag is part of the "
                         "store KEY, so tiers that must share a node corpus have to "
                         "share a code_tag.  Pin it to the last SOLVE-RELEVANT "
                         "commit (solver/source/wiring) so ORCHESTRATION-only "
                         "commits (store/persistence/build script) do not fork the "
                         "key space and break cross-tier reuse.  QC tier builds "
                         "(d4_qc, and the future 8-D precessing QC) must all use "
                         "'--code-tag fb4f07f' — the P1 QC-wiring commit under "
                         "which the d4_qc corpus was built.")
    args = ap.parse_args()

    box = BOXES[args.box]
    fixed = FIXED[args.box]
    names = [a["name"] for a in box]
    os.makedirs(args.outdir, exist_ok=True)
    commit = _git_commit()
    code_tag = commit if args.code_tag is None else args.code_tag
    t_start = time.time()

    _t(f"LM-initial-data surrogate build — box={args.box} {names}  "
       f"grid Na={args.Na} Nb={args.Nb} Nφ={args.Nphi}  solver={args.solver}")
    _t(f"outdir: {args.outdir}  (gitignored — the artifacts are meant to be shared/hosted/cited)")
    _t(f"git HEAD: {commit}   store code_tag: {code_tag}"
       + ("  (pinned)" if args.code_tag is not None else "  (= HEAD)"))

    prob = s3.make_problem(Na=args.Na, Nb=args.Nb, Nphi=args.Nphi)
    store = None
    if args.store is not None:
        store = ss.SolveStore(args.store, grid_meta=(args.Na, args.Nb, args.Nphi),
                              code_tag=code_tag, reuse_tol=args.reuse_tol)
        _t(f"solve store: {store.root_dir}  (code_tag={store.code_tag}, "
           f"reuse_tol={store.reuse_tol:.0e}, {store.n_entries} existing entries)"
           "  — nested/shared nodes reused")
    base_meta = dict(axis_names=names,
                     box=[[a["min"], a["max"]] for a in box],
                     Na=args.Na, Nb=args.Nb, Nphi=args.Nphi,
                     solver=args.solver, tol=args.tol,
                     fixed=fixed, git_commit=commit, note=args.note)

    # ----- sparse (isotropic Smolyak) -----
    _t(f"\n===== SPARSE: isotropic Smolyak L={args.level} =====")
    t0 = time.time()
    if store is not None:
        sp_solver = ss.from_problem_smolyak_3d_cached(prob, box, store=store,
                                                      fixed=fixed, solver=args.solver,
                                                      gmres_rtol=args.gmres_rtol,
                                                      retry_tol=args.retry_tol)
    else:
        sp_solver = sm.from_problem_smolyak_3d(prob, box, fixed=fixed, solver=args.solver,
                                               gmres_rtol=args.gmres_rtol,
                                               retry_tol=args.retry_tol)
    sp = sp_solver.build_isotropic(args.level, tol=args.tol, max_iter=args.max_iter)
    dt = time.time() - t0
    if store is not None:
        _t(f"   store: {store.n_hits} hits / {store.n_misses} misses "
           f"→ {store.n_entries} entries on disk")
    sp_path = os.path.join(args.outdir, f"surrogate_smolyak_{args.box}_L{args.level}.npz")
    sp.save(sp_path, meta=dict(base_meta, level=args.level))
    sz = os.path.getsize(sp_path)
    _t(f"   built {sp.n_solver_nodes} solver nodes in {dt:.0f}s → {sp_path}")
    _t(f"   on-disk: {_human(sz)}  ({sp.n_solver_nodes} deduplicated node fields, "
       f"field_shape {sp.field_shape})")
    _reload_spotcheck(sm.load_smolyak, sp_path, box, seed=2024, tag="sparse")

    # ----- dense (tensor Chebyshev) -----
    dn_path = None
    if args.dense_Q is not None:
        _t(f"\n===== DENSE: tensor Chebyshev Q={args.dense_Q} =====")
        t0 = time.time()
        axes = [dict(a, Q=args.dense_Q) for a in box]
        if store is not None:
            dn_solver = ss.from_problem_nd_3d_cached(prob, axes, store=store,
                                                     fixed=fixed, solver=args.solver,
                                                     gmres_rtol=args.gmres_rtol,
                                                     retry_tol=args.retry_tol)
        else:
            dn_solver = p3.from_problem_nd_3d(prob, axes, fixed=fixed, solver=args.solver,
                                              gmres_rtol=args.gmres_rtol,
                                              retry_tol=args.retry_tol)
        dn = dn_solver.build(tol=args.tol, max_iter=args.max_iter)
        dt = time.time() - t0
        dn_path = os.path.join(args.outdir, f"surrogate_dense_{args.box}_Q{args.dense_Q}.npz")
        dn.save(dn_path, meta=dict(base_meta, Q=[args.dense_Q] * len(box)))
        sz = os.path.getsize(dn_path)
        _t(f"   built {dn.n_nodes} solver nodes in {dt:.0f}s → {dn_path}")
        _t(f"   on-disk: {_human(sz)}  (one full tensor, field_shape {dn.field_shape})")
        _reload_spotcheck(load_parametric, dn_path, box, seed=4048, tag="dense")
    else:
        _t("\n(dense build skipped — pass --dense-Q to build & persist the tensor model)")

    # ----- certified spot-check on the reloaded sparse model -----
    _t("\n===== certified prediction spot-check (reloaded sparse model + attached solver) =====")
    model = sm.load_smolyak(sp_path)
    attach_solve_fn_3d(model, prob, names, fixed=fixed, solver="nk")
    hold = p3.holdout_points_nd([dict(a, Q=8) for a in box], n_points=3)
    p3.assert_off_node(hold, [dict(a, Q=8) for a in box])
    worst = 0.0
    for th in hold:
        t0 = time.time()
        guess = model.evaluate(th)                 # ~10 ms standalone prediction
        t_eval = time.time() - t0
        _U, info = model.evaluate_polished(th, newton_steps=2, tol=1e-10)
        worst = max(worst, float(info.residual_norm))
        _t(f"   θ={[round(float(x), 3) for x in th]}  eval={t_eval*1e3:.1f} ms  "
           f"certified‖R‖={info.residual_norm:.2e}")
    _t(f"   worst certified ‖R‖ over {len(hold)} off-node θ = {worst:.2e}"
       + ("  ✓ ≤ 1e-10" if worst <= 1e-10 else "  ✗ > 1e-10"))

    _t(f"\nTOTAL {time.time() - t_start:.0f}s")
    _t("Artifacts:")
    _t(f"   sparse: {sp_path}")
    if dn_path:
        _t(f"   dense:  {dn_path}")


if __name__ == "__main__":
    main()
