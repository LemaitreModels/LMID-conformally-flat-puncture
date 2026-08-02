"""LM-initial-data — VALUE-only POD bare-guess EQUILIBRATED-RESIDUAL rank sweep (4-D).

Produces the fig05 figure-data source ``gvm_4d_value`` pinned by
paper/figures/registry.py to EXACTLY:
    reports/P3/guess_vs_memory_4d_value_gapfill_1000.json

The 4-D sibling of ``run_value_pod_gapfill_8d.py``, and the VALUE-flavour partner
of ``run_cross_pod_figuredata.py`` (which writes the CROSS gapfill,
``guess_vs_memory_4d_cross_gapfill_*.json``).  Together they are fig05's
top-left panel: value vs value+gradient, same metric, same points, same ladder
convention.

Everything but the model, the dimension label and the output name is the 8-D
producer verbatim: the same seed-0 off-node sampling
(``run_cross_fielderror_chi.offnode_points``), the same EQUILIBRATED bare-guess
constraint residual (``solver_3d_nk.equil_residual_inf`` — the paper convention),
the same value-only POD construction (``_value_only_view`` +
``pod_basis_pool(include_derivatives=False)``), and the same output schema
(dim, r_full, N, nfeat, d, norm="equilibrated", pod_curve of
{r, mem_bytes, min/median/mean/p95/max}).  Reusing the 8-D helpers rather than
re-deriving them is deliberate: it keeps the 4-D and 8-D panels of one figure on
one code path.

Why a separate module rather than a ``--dim`` flag on the 8-D one: the package
already pairs per-dimension producers this way (``run_cross_pod_figuredata`` /
``run_cross_pod_resid_8d``; ``run_polish_fielderr`` / ``run_polish_fielderr_8d``),
and it keeps this add-only with respect to a module the campaign is running.

NO certified solve is needed — the constraint residual is a function of the guess
and the PDE alone: one shared assembly + per-m scales per point, then
``equil_residual_inf`` of each rank-r decoded value-only POD guess.  The rank
ladder is the dense 10-point geomspace; ``fig05``'s ``_thin`` reduces it at figure
time (its docstring: dense ladders are thinned there, pre-thinned ones pass
through), so this stays consistent with its panel partner.

Add-only.  Imports committed ``lm.initial_data`` modules + the 8-D field-sweep
helpers read-only; edits nothing.

Run:
  python -m lm.initial_data.pipeline.run_value_pod_gapfill_4d                 # full 1000 pt
  python -m lm.initial_data.pipeline.run_value_pod_gapfill_4d --n-points 5    # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
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
# the value-only view and value memory formula are dimension-agnostic; reuse them
# so gvm_4d_value and gvm_8d_value are the same construction at two dimensions.
from lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d import (
    _value_only_view, _pod_mem_bytes_value)

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REP3 = os.path.join(REPORTS, "P3")

#: The shipped plain (value+gradient) 4-D Hermite corpus.  Only its VALUE-only
#: view is swept here; the box/axes/grid come from its metadata, never redeclared.
PLAIN_4D = os.path.join(REPORTS, "P2", "models_chi",
                        "hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By.npz")


def main(n_points=1000, seed=0, model=None):
    path = model or PLAIN_4D
    os.makedirs(REP3, exist_ok=True)
    t0 = time.time()
    meta = _unpack_meta(_load_npz(path))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    print(f"[vpod4d] box={dict(zip(names, box))} fixed={fixed} "
          f"grid={Na}x{Nb}x{Nphi}", flush=True)

    print(f"[vpod4d] loading plain 4-D model ({os.path.getsize(path)/1e6:.0f} MB) ...",
          flush=True)
    mc = load_hermite_smolyak(path)
    N = int(mc.n_solver_nodes)
    nfeat = int(np.prod(mc.field_shape))
    d = int(mc.d)
    view = _value_only_view(mc)                       # value-only interpolant view
    print("[vpod4d] building VALUE-only POD basis (include_derivatives=False) ...",
          flush=True)
    Phi, mean, diag = pod_basis_pool(view, r=nfeat, include_derivatives=False)
    r_full = int(Phi.shape[1])
    mean = np.asarray(mean, dtype=float).reshape(-1)
    print(f"[vpod4d] Phi {Phi.shape}  r_full={r_full}  N={N} nfeat={nfeat} d={d}  "
          f"rank_value(1e-6)={diag['rank_value'].get(1e-6)}  ({time.time()-t0:.0f}s)",
          flush=True)

    ranks = sorted(set(int(round(x)) for x in np.geomspace(1, r_full, 10)))
    ranks = [r for r in ranks if 1 <= r <= r_full]
    print(f"[vpod4d] sweep ranks={ranks}", flush=True)

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
    out = dict(dim=4, metric="equilibrated_residual", r_full=int(r_full), r_cap=int(r_full),
               N=int(N), nfeat=int(nfeat), d=int(d), n_points=int(n_points), seed=int(seed),
               n_ranks=len(ranks), norm="equilibrated", flavor="value-only",
               model=os.path.basename(path),
               bare_mem_bytes=8.0 * N * nfeat, pod_curve=pod_curve)
    outp = os.path.join(REP3, f"guess_vs_memory_4d_value_gapfill_{n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[vpod4d] wrote {os.path.basename(outp)}  ({(time.time()-t0)/60:.1f} min)",
          flush=True)
    print("[vpod4d] median resid [r=1..r_full]: "
          f"{[f'{np.median(res[r]):.1e}' for r in ranks]}", flush=True)
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None,
                    help="override the plain 4-D Hermite corpus (default: PLAIN_4D)")
    args = ap.parse_args()
    main(n_points=args.n_points, seed=args.seed, model=args.model)
