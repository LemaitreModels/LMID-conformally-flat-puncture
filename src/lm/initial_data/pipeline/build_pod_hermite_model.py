"""LM-initial-data paper-integration P2 — build & persist the SHIPPED gradient-enhanced
(Hermite) quasi-circular model + its POD compression.

Add-only. Mirrors ``build_surrogate.py`` (the value-only shipped-model build) but on
the sparse gradient-enhanced path: it builds the 4-D quasi-circular model
``d4_qc = (b, q, S_Ay, S_By)`` as a :class:`HermiteSmolyakSolutionND`
**Hermite-enhanced on the two slow aligned-spin axes ``S_Ay``, ``S_By``** (D2, the
axes that dominate the offline solve count; ``b``,``q`` stay value-only), with the
Newton–Krylov solve and the **QC chain-rule** certified tangent (H5c
``sensitivity_3d_qc``); then compresses it with the H5d
:mod:`hermite_smolyak_pod` and saves both artifacts.

Why enhanced-only tangents: the two enhanced axes are the only ones the interpolant
consumes (``evaluate`` uses ``dU/dθ`` only for ``enhanced``), and the H5d POD only
compresses the enhanced derivative corpora, so we compute the certified QC tangent
for ``S_Ay``/``S_By`` and store zeros in the ``b``/``q`` slots — halving the per-node
back-solve cost versus computing the full 4-axis stack.

Reduce-to-committed: with ``--enhanced ''`` this builds the value-only
``SmolyakSolutionND`` limit (bit-for-bit, H5b) — a sanity mode, not the deliverable.

ETA (Na=44, Nb=32, Nφ=8, NK, enhanced S_Ay,S_By): the value-only ℓ=4 build is
~3.0e3 s / 401 nodes (paper Table timing); each node here adds two NK tangent
back-solves (reusing the node's factored per-m blocks) + the closed-form qc-momenta
source, so budget ~1.5–2.5× that, i.e. ~1.5–2.5 h at ℓ=4. Run in the background
under caffeinate:

    caffeinate -ims ~/Software/micromamba/micromamba run -n BBHFM python \\
        python -m lm.initial_data.pipeline.build_pod_hermite_model --Na 44 --Nb 32 --Nphi 8 --level 4

Smoke (fast, confirms the pipeline end-to-end): ``--Na 16 --Nb 12 --Nphi 6 --level 3``.
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
from lm.initial_data.parametric import hermite_smolyak as hsm
from lm.initial_data.parametric import hermite_smolyak_pod as hpod
from lm.initial_data.parametric.parametric_nd import _git_commit
from lm.initial_data.applications import sensitivity_3d_qc as qc

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
DEFAULT_OUTDIR = os.path.join(REPORTS, "P2", "models")

# The 4-D aligned-spin QUASI-CIRCULAR box (identical to build_surrogate.py::d4_qc).
# LEGACY default box, in the BARE-SPIN (S) parameterization -- NOT the production
# box.  The chi drivers (build_pod_hermite_model_chi{,_8d}.py) overwrite BOX with
# production_box.aligned_box()/spin8_box() before dispatching to main(); this
# default only applies if the base module is run directly.
BOX = [{"name": "b",    "min": 1.5, "max": 4.0},
       {"name": "q",    "min": 1.0, "max": 3.0},
       {"name": "S_Ay", "min": -0.4, "max": 0.4},
       {"name": "S_By", "min": -0.4, "max": 0.4}]
FIXED = {"qc": 1.0}
M_TOT = 1.0


def _t(m):
    print(m, flush=True)


def _human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def enhanced_only_tangent_fn(prob, active_names, enhanced_names, M_tot, fixed,
                             tangent_jac="nk", gmres_rtol=1e-8):
    """QC certified tangent stack ``(d,*field)`` computing the chain-rule tangent
    (``sensitivity_3d_qc.certified_tangent_3d_qc``) only for the ENHANCED axes and
    storing zeros in the rest — the interpolant/POD only consume the enhanced axes'
    tangents, so this halves the per-node back-solve cost.  One ``s3.assemble`` is
    shared across the enhanced axes at each node.

    ``gmres_rtol`` (default 1e-8) is the inner tangent-GMRES tolerance: the QC
    tangent's own default 1e-11 STALLS for thousands of iterations at stiff box
    corners (small ``b``, opposite-sign spins), producing ~28-min pathological
    nodes; 1e-8 is well below the field floor (~1e-9, empty-mode-limited at the
    production grid) so it does not degrade the stored tangent, and node-exactness
    reproduces the stored tangent regardless."""
    active_names = list(active_names)
    enh = set(enhanced_names)

    def tangent_fn(theta_vec, U):
        Ua = np.asarray(U)
        sl = p3.theta_to_slice3d(theta_vec, active_names, M_tot, fixed)
        asm = s3.assemble(prob, sl)
        stack = []
        for name in active_names:
            if name in enh:
                stack.append(np.asarray(qc.certified_tangent_3d_qc(
                    prob, Ua, sl, name, M_tot, asm=asm, jac=tangent_jac,
                    gmres_rtol=gmres_rtol)))
            else:
                stack.append(np.zeros(Ua.shape))
        return np.stack(stack, axis=0)

    return tangent_fn


def build_hermite_from_value(value_path, prob, names, enhanced, M_tot, fixed,
                             tangent_jac="nk", gmres_rtol=1e-8):
    """Build the gradient-enhanced Hermite-Smolyak model by REUSING an existing
    value-only ``SmolyakSolutionND`` corpus (its NK-converged fields ``U``) and
    computing ONLY the enhanced-axis QC certified tangents per node — skipping the
    expensive re-solve.  The resulting model is bit-for-bit the from-scratch build
    (same box/level/index-set/nested nodes; same deterministic NK fields; same
    tangents), just without re-solving.  Per-node progress + live ETA are logged."""
    from lm.initial_data.parametric import parametric_nd_smolyak as sm
    vm = sm.load_smolyak(value_path)
    pool_v = vm._dedup_pool()                      # key -> (theta, U, iters, resid)
    keys = list(pool_v)
    N = len(keys)
    tf = enhanced_only_tangent_fn(prob, names, enhanced, M_tot, fixed,
                                  tangent_jac=tangent_jac, gmres_rtol=gmres_rtol)
    _t(f"   reusing {N} NK value fields from {os.path.basename(value_path)}; "
       f"computing enhanced tangents {enhanced} (first node compiles ~85s)")
    pool = {}
    t0 = time.time()
    for i, k in enumerate(keys):
        theta, U, it, rs = pool_v[k]
        dU = np.asarray(tf(theta, U))              # (d,*field), zeros off the enhanced axes
        pool[k] = (np.asarray(U, dtype=float), dU, int(it), float(rs))
        if i == 0 or (i + 1) % 10 == 0 or i == N - 1:
            el = time.time() - t0
            rate = el / (i + 1)
            eta = rate * (N - i - 1)
            _t(f"   tangent {i+1}/{N}  elapsed {el:.0f}s  ~{rate:.1f}s/node  ETA {eta:.0f}s")
    enh_idx = [names.index(e) for e in enhanced]
    builder = hsm.HermiteSmolyakSolverND(solve_fn=None, axes=list(vm.axes),
                                         tangent_fn=None, enhanced_axes=enh_idx)
    model = builder._finalize([tuple(l) for l in vm.index_set], pool)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Na", type=int, default=44)
    ap.add_argument("--Nb", type=int, default=32)
    ap.add_argument("--Nphi", type=int, default=8)
    ap.add_argument("--level", type=int, default=4, help="isotropic Smolyak level")
    ap.add_argument("--enhanced", default="S_Ay,S_By",
                    help="comma-separated enhanced axis names (default the two slow "
                         "aligned-spin axes; '' → value-only limit)")
    ap.add_argument("--solver", choices=("nk", "modified"), default="nk")
    ap.add_argument("--tangent-jac", choices=("nk", "modified"), default="nk")
    ap.add_argument("--tangent-gmres-rtol", type=float, default=1e-8,
                    help="inner tangent-GMRES tolerance (default 1e-8; the QC "
                         "default 1e-11 stalls at stiff box corners)")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--pod-tail", type=float, default=1e-6,
                    help="POD singular-value tail for the shipped rank")
    ap.add_argument("--reuse-value", default=None,
                    help="path to an existing value-only SmolyakSolutionND .npz "
                         "(same box/level): reuse its NK fields and compute ONLY "
                         "the enhanced tangents (skips the re-solve; ~2.6s/node)")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    enhanced = [s for s in (x.strip() for x in args.enhanced.split(",")) if s]
    names = [a["name"] for a in BOX]
    for e in enhanced:
        if e not in names:
            raise ValueError(f"enhanced axis {e!r} not in {names}")
    os.makedirs(args.outdir, exist_ok=True)
    commit = _git_commit()
    t_start = time.time()

    _t(f"LM-initial-data P2 Hermite-Smolyak build — box=d4_qc {names}  enhanced={enhanced}")
    _t(f"grid Na={args.Na} Nb={args.Nb} Nφ={args.Nphi}  solver={args.solver}  "
       f"tangent_jac={args.tangent_jac}  level={args.level}")
    _t(f"outdir: {args.outdir}  (gitignored)   git HEAD: {commit}")

    prob = s3.make_problem(Na=args.Na, Nb=args.Nb, Nphi=args.Nphi)

    # ----- build the gradient-enhanced sparse model -----
    # The enhanced-only tangent (halves back-solves) — only the enhanced axes'
    # tangents are consumed by evaluate/POD.  The committed solve_fn is used for the
    # certified polish + POD certify below.
    from lm.initial_data.parametric.parametric_nd_3d import make_solve_fn
    solve_fn, _ = make_solve_fn(prob, names, M_tot=M_TOT, fixed=FIXED,
                                use_cache=True, solver=args.solver)

    _t(f"\n===== BUILD: isotropic Hermite-Smolyak L={args.level} =====")
    t0 = time.time()
    if args.reuse_value:
        model = build_hermite_from_value(args.reuse_value, prob, names, enhanced,
                                         M_TOT, FIXED, tangent_jac=args.tangent_jac,
                                         gmres_rtol=args.tangent_gmres_rtol)
    else:
        tf = enhanced_only_tangent_fn(prob, names, enhanced, M_TOT, FIXED,
                                      tangent_jac=args.tangent_jac,
                                      gmres_rtol=args.tangent_gmres_rtol)
        enh_idx = [names.index(e) for e in enhanced]
        spec = [(a["min"], a["max"]) for a in BOX]
        builder = hsm.HermiteSmolyakSolverND(solve_fn, spec, tf, enhanced_axes=enh_idx)
        model = builder.build_isotropic(args.level, tol=args.tol,
                                        max_iter=args.max_iter, verbose=True)
    model._solve_fn = solve_fn                     # for evaluate_polished (certify + POD)
    dt = time.time() - t0
    base_meta = dict(axis_names=names, box=[[a["min"], a["max"]] for a in BOX],
                     enhanced=enhanced, Na=args.Na, Nb=args.Nb, Nphi=args.Nphi,
                     solver=args.solver, tangent_jac=args.tangent_jac, tol=args.tol,
                     fixed=FIXED, git_commit=commit, level=args.level, note=args.note)
    tag = f"d4qc_L{args.level}_enh-{'-'.join(enhanced) or 'none'}"
    m_path = os.path.join(args.outdir, f"hermite_smolyak_{tag}.npz")
    model.save(m_path, meta=base_meta)
    _t(f"   built {model.n_solver_nodes} solver nodes in {dt:.0f}s "
       f"({dt/max(model.n_solver_nodes,1):.1f}s/node)")
    _t(f"   total Newton iters over pool: {model.total_iters}")
    _t(f"   raw model on-disk: {_human(os.path.getsize(m_path))}  "
       f"(value + {len(enhanced)} tangent fields/node, field_shape {model.field_shape})")

    # ----- H5d POD compression (stacked value+derivative corpus) -----
    _t(f"\n===== POD compression (H5d) tail={args.pod_tail:.0e} =====")
    t0 = time.time()
    pod, diag = hpod.build_pod_hermite_smolyak(model, tail=args.pod_tail,
                                               solve_fn=model._solve_fn)
    dt_pod = time.time() - t0
    nfeat = int(np.prod(model.field_shape))
    _t(f"   rank_value : {diag['rank_value']}")
    _t(f"   rank_stacked: {diag['rank_stacked']}")
    _t(f"   dU_on_value_basis_resid: {diag['dU_on_value_basis_resid']}")
    r = pod.r
    _t(f"   shipped r(tail={args.pod_tail:.0e}) = {r}   per-field compression "
       f"nfeat/r = {nfeat}/{r} = {nfeat/r:.1f}x   (POD built in {dt_pod:.1f}s)")
    p_path = os.path.join(args.outdir, f"pod_hermite_smolyak_{tag}.npz")
    pod.save(p_path, meta=base_meta)
    _t(f"   raw {_human(os.path.getsize(m_path))} → POD {_human(os.path.getsize(p_path))}  "
       f"({os.path.getsize(m_path)/max(os.path.getsize(p_path),1):.1f}x smaller on disk)")

    # ----- certified spot-check (POD-decoded guess → committed polish) -----
    _t("\n===== certified spot-check (POD guess + polish) =====")
    hold = p3.holdout_points_nd([dict(a, Q=8) for a in BOX], n_points=3)
    p3.assert_off_node(hold, [dict(a, Q=8) for a in BOX])
    worst = 0.0
    for th in hold:
        t0 = time.time()
        _ = pod.evaluate(th)
        t_eval = (time.time() - t0) * 1e3
        _U, info = pod.evaluate_polished(th, newton_steps=3, tol=1e-10)
        worst = max(worst, float(info.residual_norm))
        _t(f"   θ={[round(float(x),3) for x in th]}  eval={t_eval:.1f} ms  "
           f"certified‖R‖={info.residual_norm:.2e}")
    _t(f"   worst certified ‖R‖ over {len(hold)} off-node θ = {worst:.2e}"
       + ("  ✓ ≤ 1e-10" if worst <= 1e-10 else "  ✗ > 1e-10"))

    _t(f"\nTOTAL {time.time() - t_start:.0f}s")
    _t(f"Artifacts:\n   raw: {m_path}\n   pod: {p_path}")


if __name__ == "__main__":
    main()
