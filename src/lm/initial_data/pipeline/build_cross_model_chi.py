"""PARASOL — build the FULL BILINEAR (cross-enhanced) 4-D χ Hermite–Smolyak model.

Add-only POST-PROCESS of a shipped gradient-only Hermite–Smolyak model with TWO
enhanced axes: it reuses the stored per-node value ``node_U`` and first tangents
``node_dU`` (NO Newton re-solves), computes the missing mixed second partial
``∂²U/∂θ_{e0}∂θ_{e1}`` at every node via the certified second-order cross tangent
(``applications.sensitivity_3d_cross.cross_tangent_3d_qc`` — one back-solve against
the node's re-assembled+factored Jacobian), assembles the augmented node pool, and
writes the with-cross model (``hermite_smolyak_cross`` kind) to a DISTINCT path
(never overwriting the shipped model).

Resumable: checkpoints ``node_cross`` to ``<out>.cross.partial.npy`` every
``--checkpoint`` nodes; a re-run resumes from the checkpoint.

Run (from repo root):
  caffeinate -i python sandbox/parasol/build_cross_model_chi.py \
      --model sandbox/parasol/reports/P2/models_chi/hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By.npz \
      --out   sandbox/parasol/reports/P2/models_chi/hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz
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
from lemaitre.initial_data.parametric.parametric_nd import _load_npz, _unpack_meta
from lemaitre.initial_data.parametric.parametric_nd_smolyak import _node_key
from lemaitre.initial_data.parametric.parametric_nd_3d import theta_to_slice3d
from lemaitre.initial_data.applications import sensitivity_3d_cross as cross
from lemaitre.initial_data.parametric.hermite_smolyak_cross import (
    build_cross_from_pool, _global_pairs)


def main(model_path, out_path, jac="nk", checkpoint=50, M_tot=1.0, enhanced_names=None,
         cross_fn="committed"):
    # ADD-ONLY: select the second-order cross-tangent routine.  Default "committed"
    # calls sensitivity_3d_cross.cross_tangent_3d_qc EXACTLY as before (byte-for-byte
    # for a spin-spin enhanced set — the only case the committed routine supports).
    # "bq" routes to the 4-axis dispatcher sensitivity_3d_cross_bq.cross_tangent_3d_qc_bq
    # (spin-spin pairs fall through to the committed routine bit-for-bit; pairs
    # touching b/q use the M1 analytic path) — needed for the full 4-axis cross.
    if cross_fn == "committed":
        _cross_tan = cross.cross_tangent_3d_qc
    elif cross_fn == "bq":
        from lemaitre.initial_data.applications import sensitivity_3d_cross_bq as _cbq
        _cross_tan = _cbq.cross_tangent_3d_qc_bq
    else:
        raise ValueError(f"cross_fn must be 'committed' or 'bq', got {cross_fn!r}")
    t0 = time.time()
    data = _load_npz(model_path)
    meta = _unpack_meta(data)
    names = list(meta["axis_names"])
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])

    th = np.asarray(data["node_thetas"], dtype=float)     # (N, d)
    U = np.asarray(data["node_U"], dtype=float)           # (N, *fs)
    dU = np.asarray(data["node_dU"], dtype=float)         # (N, d, *fs)
    node_iters = np.asarray(data["node_iters"])
    node_resids = np.asarray(data["node_resids"], dtype=float)
    index_set = [tuple(int(x) for x in row) for row in np.asarray(data["index_set"])]
    axes = [(float(a[0]), float(a[1])) for a in np.asarray(data["axes"], dtype=float)]
    if enhanced_names:            # override the interpolant's enhanced set (e.g. the
        bad = [n for n in enhanced_names if n not in names]   # 8D "same-as-4D" y-pair
        if bad:
            raise ValueError(f"--enhanced {bad} not among axes {names}")
        enhanced = tuple(names.index(n) for n in enhanced_names)
    else:
        enhanced = tuple(int(e) for e in np.asarray(data["enhanced"], dtype=np.int64))
    fs = tuple(int(x) for x in np.asarray(data["field_shape"], dtype=np.int64))

    gp = _global_pairs(enhanced)                          # [(e0,e1), ...] all C(n,2) pairs
    npair = len(gp)
    if npair < 1:
        raise ValueError(f"need >=2 enhanced axes for a cross; enhanced={enhanced}")
    N = th.shape[0]
    print(f"[cross] model={os.path.basename(model_path)}  N={N} nodes  d={len(names)}",
          flush=True)
    print(f"[cross] axes={names} enhanced={[names[e] for e in enhanced]} "
          f"pairs={[(names[a], names[b]) for (a, b) in gp]} (n_pairs={npair}) "
          f"jac={jac} grid Na={Na},Nb={Nb},Nphi={Nphi}", flush=True)

    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    ckpt = out_path + ".cross.partial.npy"
    node_cross = np.full((N, len(gp)) + fs, np.nan)
    start = 0
    if os.path.exists(ckpt):
        saved = np.load(ckpt)
        if saved.shape == node_cross.shape:
            node_cross = saved
            done = np.all(np.isfinite(node_cross.reshape(N, -1)), axis=1)
            start = int(np.argmin(done)) if not done.all() else N
            print(f"[cross] resumed from checkpoint: {start}/{N} already done", flush=True)

    tev = time.time()
    for i in range(start, N):
        theta = th[i]
        sl = theta_to_slice3d(theta, names, M_tot, fixed)
        asm = s3.assemble(prob, sl)                       # shared across all pairs at node i
        for pj, (e0, e1) in enumerate(gp):
            Uij = _cross_tan(prob, U[i], sl, names[e0], names[e1], M_tot,
                             dU_i=dU[i, e0], dU_j=dU[i, e1], asm=asm, jac=jac)
            node_cross[i, pj] = np.asarray(Uij)
        if (i + 1) % checkpoint == 0 or i == N - 1:
            np.save(ckpt, node_cross)
            el = time.time() - tev
            rate = el / (i + 1 - start)
            eta = rate * (N - 1 - i)
            print(f"   {i+1}/{N}  ({el:.0f}s, {rate:.2f}s/node, ETA {eta/60:.1f} min)",
                  flush=True)

    # assemble the pool (6-tuple: theta, U, dU, cross, iters, resid) and finalize
    pool = {}
    for i in range(N):
        pool[_node_key(th[i])] = (th[i], U[i], dU[i], node_cross[i],
                                  int(node_iters[i]), float(node_resids[i]))
    sol = build_cross_from_pool(axes, index_set, enhanced, pool, solve_fn=None)

    out_meta = dict(meta)
    out_meta["kind"] = "hermite_smolyak_cross"
    out_meta["enhanced"] = [names[e] for e in enhanced]     # reflect any override
    out_meta["cross_pairs"] = [[names[a], names[b]] for (a, b) in gp]
    out_meta["cross_jac"] = jac
    out_meta["note"] = ((meta.get("note", "") or "")
                        + " | full bilinear cross ∂²U/∂χ_Ay∂χ_By added "
                          "(post-process of the shipped gradient-only model)")
    sol.save(out_path, meta=out_meta)
    print(f"[cross] wrote {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB) "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    # keep the checkpoint until the caller confirms; remove on clean finish
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jac", default="nk", choices=["nk", "modified"])
    ap.add_argument("--checkpoint", type=int, default=50)
    ap.add_argument("--enhanced", default=None,
                    help="comma-separated axis NAMES to enhance for the cross "
                         "(default: the shipped model's enhanced set). E.g. "
                         "'chi_Ay,chi_By' to restrict the 8D corpus to the 4D y-pair.")
    ap.add_argument("--cross-fn", default="committed", choices=["committed", "bq"],
                    help="cross-tangent routine: 'committed' (default, byte-for-byte "
                         "sensitivity_3d_cross.cross_tangent_3d_qc — spin pairs only) "
                         "or 'bq' (the 4-axis dispatcher supporting b/q pairs).")
    args = ap.parse_args()
    enh = [s.strip() for s in args.enhanced.split(",")] if args.enhanced else None
    main(args.model, args.out, jac=args.jac, checkpoint=args.checkpoint,
         enhanced_names=enh, cross_fn=args.cross_fn)
