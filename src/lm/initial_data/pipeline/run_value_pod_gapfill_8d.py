"""LM-initial-data — VALUE-only POD bare-guess EQUILIBRATED-RESIDUAL rank sweep (8-D).

Produces the fig06 figure-data source ``gvm_8d_value`` pinned by
manuscript/figures/registry.py to EXACTLY:
    reports/P3/guess_vs_memory_8d_value_gapfill_1000.json

The 8-D VALUE-only analog of the committed CROSS gapfill producer
``run_cross_pod_figuredata.py`` (which writes guess_vs_memory_4d_cross_gapfill_*.json):
same schema (dim, r_full, N, nfeat, d, norm="equilibrated", pod_curve of
{r, mem_bytes, min/median/mean/p95/max}), same seed-0 off-node sampling
(``run_cross_fielderror_chi.offnode_points``), same EQUILIBRATED bare-guess
constraint residual (``solver_3d_nk.equil_residual_inf`` — the paper convention),
but the swept model is the VALUE-only plain POD of the 8-D corpus.

NO certified solve is needed (the constraint residual is a function of the guess +
the PDE only): one shared assembly + per-m scales per point, then equil_residual_inf
of each rank-r decoded value-only POD guess.

The VALUE-only POD is built EXACTLY as ``run_hermite_fielderr_sweep_8d.py``'s
value flavor: ``pod_basis_pool(value_view, include_derivatives=False)`` on the
structural value-only view of the shipped plain 8-D Hermite model, and the rank-r
guess is ``mean + Phi[:,:r] @ (Phi.T (u_value - mean))[:r]`` (memory-lean; equals
``build_pod_hermite_smolyak`` + coeff-model to roundoff).  This reuses that
committed-style construction verbatim (imported), so gvm_8d_value and gvm_8d_field
share one value-only POD — only the metric differs (residual here vs field error).

Add-only.  Imports committed ``lm.initial_data`` modules + the add-only 8-D field-sweep
helpers read-only; edits nothing.

Run:
  python -m lm.initial_data.pipeline.run_value_pod_gapfill_8d                 # full 1000 pt
  python -m lm.initial_data.pipeline.run_value_pod_gapfill_8d --n-points 5    # smoke test
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


from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.solver import solver_3d_nk as s3nk
from lm.initial_data.parametric.parametric_nd import _load_npz, _unpack_meta
from lm.initial_data.parametric.parametric_nd_3d import theta_to_slice3d
from lm.initial_data.parametric.hermite_smolyak import load_hermite_smolyak
from lm.initial_data.parametric.hermite_smolyak_pod import pod_basis_pool
from lm.initial_data.pipeline.run_cross_fielderror_chi import offnode_points, stats
# reuse the EXACT value-only view + value mem formula + model path of the 8-D
# field sweep, so gvm_8d_value and gvm_8d_field use ONE identical value-only POD.
from lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d import (
    _value_only_view, _pod_mem_bytes_value, PLAIN)

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REP3 = os.path.join(REPORTS, "P3")


def main(n_points=1000, seed=0):
    os.makedirs(REP3, exist_ok=True)
    t0 = time.time()
    meta = _unpack_meta(_load_npz(PLAIN))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    print(f"[vpod8d] loading plain 8-D model ({os.path.getsize(PLAIN)/1e9:.0f} GB) ...",
          flush=True)
    mc = load_hermite_smolyak(PLAIN)
    N = int(mc.n_solver_nodes)
    nfeat = int(np.prod(mc.field_shape))
    d = int(mc.d)
    view = _value_only_view(mc)                       # value-only interpolant view
    print(f"[vpod8d] building VALUE-only POD basis (include_derivatives=False) ...",
          flush=True)
    Phi, mean, diag = pod_basis_pool(view, r=nfeat, include_derivatives=False)
    r_full = int(Phi.shape[1])
    mean = np.asarray(mean, dtype=float).reshape(-1)
    print(f"[vpod8d] Phi {Phi.shape}  r_full={r_full}  N={N} nfeat={nfeat} d={d}  "
          f"rank_value(1e-6)={diag['rank_value'].get(1e-6)}  ({time.time()-t0:.0f}s)",
          flush=True)

    ranks = sorted(set(int(round(x)) for x in np.geomspace(1, r_full, 10)))
    ranks = [r for r in ranks if 1 <= r <= r_full]
    print(f"[vpod8d] sweep ranks={ranks}", flush=True)

    pts = offnode_points(box, n_points, seed)
    res = {r: [] for r in ranks}
    tev = time.time()
    for i, th in enumerate(pts):
        sl = theta_to_slice3d(th, names, 1.0, fixed)
        asm = s3.assemble(prob, sl)
        scales = s3nk._block_scales(asm)
        u_full = np.asarray(view.evaluate(th)).reshape(-1)
        c = Phi.T @ (u_full - mean)
        for r in ranks:
            u = (mean + Phi[:, :r] @ c[:r]).reshape(prob.Ntot2d, prob.Nphi)
            res[r].append(s3nk.equil_residual_inf(asm, u, scales))
        if (i + 1) % 50 == 0 or i == n_points - 1:
            el = time.time() - tev
            rate = el / (i + 1)
            print(f"   sweep {i+1}/{n_points} ({el:.0f}s, {rate:.3f}s/pt, "
                  f"ETA {rate*(n_points-i-1)/60:.1f} min)  "
                  f"med resid@r_full={np.median(res[ranks[-1]]):.2e}", flush=True)

    pod_curve = [dict(r=int(r), mem_bytes=_pod_mem_bytes_value(r, N, nfeat), **stats(res[r]))
                 for r in ranks]
    out = dict(dim=8, metric="equilibrated_residual", r_full=int(r_full), r_cap=int(r_full),
               N=int(N), nfeat=int(nfeat), d=int(d), n_points=int(n_points), seed=int(seed),
               n_ranks=len(ranks), norm="equilibrated", flavor="value-only",
               model=os.path.basename(PLAIN),
               bare_mem_bytes=8.0 * N * nfeat, pod_curve=pod_curve)
    outp = os.path.join(REP3, f"guess_vs_memory_8d_value_gapfill_{n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[vpod8d] wrote {os.path.basename(outp)}  ({(time.time()-t0)/60:.1f} min)",
          flush=True)
    print(f"[vpod8d] median resid [r=1..r_full]: "
          f"{[f'{np.median(res[r]):.1e}' for r in ranks]}", flush=True)
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(n_points=args.n_points, seed=args.seed)
