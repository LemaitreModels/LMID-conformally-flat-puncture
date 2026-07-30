"""8D y-pair-cross residual POD gapfill — residual sibling of guess_vs_memory_8d_hermite_field.
Add-only clone of run_cross_pod_figuredata.py (4D) for the 8D cross model, so fig05 top-right
value+gradient matches the bottom-right (same model, same ranks/memory)."""
from __future__ import annotations
import argparse, json, os, sys, time
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.solver import solver_3d_nk as s3nk
from lm.initial_data.parametric.parametric_nd import _load_npz, _unpack_meta
from lm.initial_data.parametric.parametric_nd_3d import theta_to_slice3d
from lm.initial_data.parametric.hermite_smolyak_cross import load_hermite_smolyak_cross
from lm.initial_data.parametric.hermite_smolyak_pod_cross import build_pod_hermite_smolyak_cross
from lm.initial_data.pipeline.run_cross_fielderror_chi import offnode_points, pod_rank_ladder, stats
HERE = os.path.dirname(os.path.abspath(__file__)); REP3 = os.path.join(HERE, "reports", "P3")
def _mem(r, N, nfeat, d, npair):
    return 8.0 * (nfeat * r + N * r + N * d * r + N * npair * r + nfeat)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-model", required=True)
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(REP3, exist_ok=True); t0 = time.time()
    meta = _unpack_meta(_load_npz(a.cross_model)); names = list(meta["axis_names"])
    box = [(float(x[0]), float(x[1])) for x in meta["box"]]; fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    print("[resid8d] load cross + build full-rank POD ...", flush=True)
    mc = load_hermite_smolyak_cross(a.cross_model)
    N = int(mc.n_solver_nodes); nfeat = int(np.prod(mc.field_shape)); d = int(mc.d)
    npair = len(mc.cross_pairs_global); r_full = (1 + d + npair) * N
    pod, _ = build_pod_hermite_smolyak_cross(mc, r=r_full); r_full = int(pod.r)
    print(f"[resid8d] r_full={r_full} N={N} nfeat={nfeat} d={d} npair={npair} ({time.time()-t0:.0f}s)", flush=True)
    ranks = pod_rank_ladder(r_full)
    print(f"[resid8d] ranks={ranks}", flush=True)
    pts = offnode_points(box, a.n_points, a.seed)
    Phi = pod.Phi; mean = np.asarray(pod.mean, dtype=float).reshape(-1)
    res = {r: [] for r in ranks}; tev = time.time()
    for i, th in enumerate(pts):
        sl = theta_to_slice3d(th, names, 1.0, fixed); asm = s3.assemble(prob, sl)
        scales = s3nk._block_scales(asm); c = pod.coeffs(th)
        for r in ranks:
            u = (mean + Phi[:, :r] @ c[:r]).reshape(prob.Ntot2d, prob.Nphi)
            res[r].append(s3nk.equil_residual_inf(asm, u, scales))
        if (i + 1) % 50 == 0 or i == a.n_points - 1:
            el = time.time() - tev
            print(f"   {i+1}/{a.n_points} ({el:.0f}s, {el/(i+1):.2f}s/pt) med@r_full={np.median(res[ranks[-1]]):.2e}", flush=True)
    pod_curve = [dict(r=int(r), mem_bytes=_mem(r, N, nfeat, d, npair), **stats(res[r])) for r in ranks]
    out = dict(dim=8, metric="equilibrated_residual", flavor="value+grad-cross",
               model=os.path.basename(a.cross_model), r_full=int(r_full), r_cap=int(r_full),
               N=int(N), nfeat=int(nfeat), d=int(d), npair=int(npair), n_points=int(a.n_points),
               seed=int(a.seed), n_ranks=len(ranks), norm="equilibrated",
               bare_mem_bytes=8.0 * N * (1 + d + npair) * nfeat, pod_curve=pod_curve)
    p = os.path.join(REP3, f"guess_vs_memory_8d_hermite_gapfill_{a.n_points}.json")
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"[resid8d] wrote {os.path.basename(p)} ({(time.time()-t0)/60:.1f} min)", flush=True)
if __name__ == "__main__":
    main()
