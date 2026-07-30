"""PARASOL — figure data for the 4D "gradient-enhanced := cross" paper update.

Produces the two JSONs the cross model needs so fig06_guess_vs_memory and
fig01_polish_staircase show the 4D value+gradient family as the FULL-BILINEAR
CROSS model (`hermite_smolyak_cross`), re-encoded by POD:

  1. reports/P3/guess_vs_memory_4d_cross_gapfill_1000.json  — the POD rank-sweep
     of the cross model (bare-guess EQUILIBRATED residual vs stored memory);
     schema identical to guess_vs_memory_4d_gapfill_1000.json (the gradient one),
     with the cross POD memory formula (Φ + value + d tangents + n_pairs cross).
  2. reports/P2/models_chi/pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r75.npz
     — the cross POD truncated to r=75, so run_polish_table (which now knows the
     `pod_hermite_smolyak_cross` kind) produces the r=75 polish staircase JSON.

Add-only.  Uses the same seed-0 off-node sampling as run_polish_table (via
run_cross_fielderror_chi.offnode_points), the equilibrated residual
(solver_3d_nk.equil_residual_inf — the paper convention, notes/conventions.md),
and the committed hermite_smolyak_pod_cross POD layer.

Run:  python sandbox/parasol/run_cross_pod_figuredata.py
"""
from __future__ import annotations

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
from lm.initial_data.parametric.hermite_smolyak_cross import load_hermite_smolyak_cross
from lm.initial_data.parametric.hermite_smolyak_pod_cross import (
    build_pod_hermite_smolyak_cross, truncate_pod_cross)
from lm.initial_data.pipeline.run_cross_fielderror_chi import offnode_points, stats

HERE = os.path.dirname(os.path.abspath(__file__))
REP3 = os.path.join(HERE, "reports", "P3")
MODELS = os.path.join(HERE, "reports", "P2", "models_chi")
CROSS = os.path.join(MODELS, "hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz")


def _pod_mem_bytes(r, N, nfeat, d, npair):
    """Stored float memory of a rank-r cross POD: Phi(nfeat*r) + value coeff(N*r)
    + tangent coeff(N*d*r) + cross coeff(N*npair*r) + mean(nfeat)."""
    return 8.0 * (nfeat * r + N * r + N * d * r + N * npair * r + nfeat)


def main(n_points=1000, seed=0, r75_out=None):
    os.makedirs(REP3, exist_ok=True)
    t0 = time.time()
    meta = _unpack_meta(_load_npz(CROSS))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    print("[figdata] loading cross model + building full-rank cross POD ...", flush=True)
    mc = load_hermite_smolyak_cross(CROSS)
    N = mc.n_solver_nodes
    nfeat = int(np.prod(mc.field_shape))
    d = mc.d
    npair = len(mc.cross_pairs_global)
    r_full = (1 + d + npair) * N                          # value + d tangents + npair cross snapshots
    pod, diag = build_pod_hermite_smolyak_cross(mc, r=r_full)
    r_full = pod.r
    print(f"[figdata] cross POD r_full={r_full}  N={N} nfeat={nfeat} d={d} npair={npair} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- save the r=75 cross POD for run_polish_table (the r=75 staircase) ----
    if r75_out is None:
        r75_out = os.path.join(MODELS, "pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r75.npz")
    truncate_pod_cross(pod, 75).save(r75_out, meta={
        "axis_names": names, "box": [list(b) for b in box], "fixed": fixed,
        "Na": Na, "Nb": Nb, "Nphi": Nphi, "level": int(meta.get("level", 5)),
        "enhanced": list(meta.get("enhanced", [])), "r_shipped": int(r_full)})
    print(f"[figdata] wrote r=75 cross POD -> {os.path.basename(r75_out)}", flush=True)

    # ---- guess-vs-memory rank sweep (equilibrated bare-guess residual) ----
    ranks = sorted(set(int(round(x)) for x in np.geomspace(1, r_full, 10)))
    ranks = [r for r in ranks if 1 <= r <= r_full]
    print(f"[figdata] sweep ranks={ranks}", flush=True)
    pts = offnode_points(box, n_points, seed)
    Phi, mean = pod.Phi, pod.mean
    fs = mc.field_shape
    res = {r: [] for r in ranks}
    tev = time.time()
    for i, th in enumerate(pts):
        sl = theta_to_slice3d(th, names, 1.0, fixed)
        asm = s3.assemble(prob, sl)
        scales = s3nk._block_scales(asm)
        c = pod.coeffs(th)                                # (r_full,)
        for r in ranks:
            u = (mean + Phi[:, :r] @ c[:r]).reshape(prob.Ntot2d, prob.Nphi)
            res[r].append(s3nk.equil_residual_inf(asm, u, scales))
        if (i + 1) % 100 == 0 or i == n_points - 1:
            el = time.time() - tev
            print(f"   sweep {i+1}/{n_points} ({el:.0f}s, {el/(i+1):.2f}s/pt)  "
                  f"med@r_full={np.median(res[r_full] if r_full in res else res[ranks[-1]]):.2e}",
                  flush=True)

    pod_curve = []
    for r in ranks:
        st = stats(res[r])
        pod_curve.append(dict(r=int(r), mem_bytes=_pod_mem_bytes(r, N, nfeat, d, npair), **st))
    out = dict(dim=4, r_full=int(r_full), r_cap=int(r_full), N=int(N), nfeat=int(nfeat),
               d=int(d), npair=int(npair), n_points=int(n_points), seed=int(seed),
               n_ranks=len(ranks), norm="equilibrated",
               bare_mem_bytes=8.0 * N * (1 + d + npair) * nfeat, pod_curve=pod_curve)
    outp = os.path.join(REP3, f"guess_vs_memory_4d_cross_gapfill_{n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[figdata] wrote {os.path.basename(outp)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[figdata] sweep median [r=1..r_full]: "
          f"{[f'{np.median(res[r]):.1e}' for r in ranks]}", flush=True)
    return outp, r75_out


if __name__ == "__main__":
    main()
