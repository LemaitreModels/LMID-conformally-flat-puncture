"""LM-initial-data — CROSS-model FIELD-ERROR POD-rank sweep (4-D).

Companion to ``run_cross_pod_figuredata.py`` (which produces the cross bare-guess
EQUILIBRATED-RESIDUAL POD sweep, ``guess_vs_memory_4d_cross_gapfill_1000.json``).
This computes the *field error* analogue: the relative-L2 field error
``||decode_r - u_true||_2 / ||u_true||_2`` of the rank-r full-bilinear CROSS POD
guess against the certified solve ``u_true``, as the POD rank r is swept, over the
IDENTICAL 1000 seed-0 off-node points.

Why: the on-disk field-error sweep ``guess_vs_memory_4d_field_1000.json`` only
carries the VALUE and PLAIN-HERMITE (gradient-only) corpora. The gradient-only
model drops the mixed 2nd partial d^2U/dchi_Ay dchi_By and so *loses jointly*
(field error ~3.8e-4 vs value ~7.9e-5). The shipped 4-D "value + gradient" model
is the CROSS model (commits 4f78e98 / 055c722), whose field error is ~2.4e-5,
BELOW value. fig05's field-error bottom row must therefore show value + CROSS
(matching its residual top row), not value + plain-Hermite.

u_true is the cross model's own certified NK solve (warm from the full-rank cross
guess -> fast); field-error convention matches ``run_cross_fielderror_chi.field_err``.

Writes ``reports/P3/guess_vs_memory_4d_cross_field_1000.json`` with schema
mirroring ``guess_vs_memory_4d_cross_gapfill_1000.json`` (pod_curve of
{r, mem_bytes, min, median, mean, p95, max}; metric ``field_error_relL2``).

Add-only.  Imports committed ``lm.initial_data`` modules read-only; edits nothing.

Run:
  python -m lm.initial_data.pipeline.run_cross_fielderr_sweep                 # full 1000 pt
  python -m lm.initial_data.pipeline.run_cross_fielderr_sweep --n-points 5    # smoke test
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
from lm.initial_data.parametric.parametric_nd import _load_npz, _unpack_meta, attach_solve_fn_3d
from lm.initial_data.parametric.hermite_smolyak_cross import load_hermite_smolyak_cross
from lm.initial_data.parametric.hermite_smolyak_pod_cross import build_pod_hermite_smolyak_cross
# reuse the EXACT seed-shared off-node sampling + stats of the residual sweep
from lm.initial_data.pipeline.run_cross_fielderror_chi import (
    offnode_points, pod_rank_ladder, stats)

HERE = os.path.dirname(os.path.abspath(__file__))
REP3 = os.path.join(HERE, "reports", "P3")
MODELS = os.path.join(HERE, "reports", "P2", "models_chi")
CROSS = os.path.join(MODELS, "hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz")


def _pod_mem_bytes(r, N, nfeat, d, npair):
    """Cross POD stored float memory — IDENTICAL formula to run_cross_pod_figuredata:
    Phi(nfeat*r) + value coeff(N*r) + d tangents(N*d*r) + npair cross(N*npair*r) + mean(nfeat)."""
    return 8.0 * (nfeat * r + N * r + N * d * r + N * npair * r + nfeat)


def _field_err(u, ut):
    """Relative Frobenius L2 (matches run_cross_fielderror_chi.field_err)."""
    u = np.asarray(u).reshape(-1)
    ut = np.asarray(ut).reshape(-1)
    return float(np.linalg.norm(u - ut) / max(np.linalg.norm(ut), 1e-300))


def main(n_points=1000, seed=0, u_tol=1e-11, u_steps=12):
    os.makedirs(REP3, exist_ok=True)
    t0 = time.time()
    meta = _unpack_meta(_load_npz(CROSS))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)

    print(f"[xfield] loading cross model + building full-rank cross POD ...", flush=True)
    mc = load_hermite_smolyak_cross(CROSS)
    N = mc.n_solver_nodes
    nfeat = int(np.prod(mc.field_shape))
    d = mc.d
    npair = len(mc.cross_pairs_global)
    r_full = (1 + d + npair) * N
    pod, _diag = build_pod_hermite_smolyak_cross(mc, r=r_full)
    r_full = pod.r
    Phi, mean = pod.Phi, pod.mean
    print(f"[xfield] cross POD r_full={r_full}  N={N} nfeat={nfeat} d={d} npair={npair} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # certified u_true via the cross model's own NK solve (warm from the full guess)
    attach_solve_fn_3d(mc, prob, names, M_tot=1.0, fixed=fixed, use_cache=False, solver="nk")

    # SAME rank grid as run_cross_pod_figuredata (so mem_bytes align with the residual sweep)
    ranks = pod_rank_ladder(r_full)
    print(f"[xfield] sweep ranks={ranks}", flush=True)

    pts = offnode_points(box, n_points, seed)
    fe = {r: [] for r in ranks}
    ures = []
    tev = time.time()
    for i, th in enumerate(pts):
        ut, info = mc.evaluate_polished(th, newton_steps=u_steps, tol=u_tol)
        ut = np.asarray(ut)
        ures.append(float(info.residual_norm))
        c = pod.coeffs(th)                                    # (r_full,)
        for r in ranks:
            u = (mean + Phi[:, :r] @ c[:r]).reshape(prob.Ntot2d, prob.Nphi)
            fe[r].append(_field_err(u, ut))
        if (i + 1) % 50 == 0 or i == n_points - 1:
            el = time.time() - tev
            rate = el / (i + 1)
            print(f"   sweep {i+1}/{n_points} ({el:.0f}s, {rate:.2f}s/pt, "
                  f"ETA {rate*(n_points-i-1)/60:.1f} min)  "
                  f"med field-err@r_full={np.median(fe[ranks[-1]]):.2e}  "
                  f"(u_true res med={np.median(ures):.1e})", flush=True)

    pod_curve = []
    for r in ranks:
        pod_curve.append(dict(r=int(r), mem_bytes=_pod_mem_bytes(r, N, nfeat, d, npair),
                              **stats(fe[r])))
    out = dict(dim=4, metric="field_error_relL2", r_full=int(r_full), r_cap=int(r_full),
               N=int(N), nfeat=int(nfeat), d=int(d), npair=int(npair),
               n_points=int(n_points), seed=int(seed), n_ranks=len(ranks),
               u_tol=u_tol, u_steps=u_steps,
               bare_mem_bytes=8.0 * N * (1 + d + npair) * nfeat, pod_curve=pod_curve)
    outp = os.path.join(REP3, f"guess_vs_memory_4d_cross_field_{n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[xfield] wrote {os.path.basename(outp)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[xfield] median field err [r=1..r_full]: "
          f"{[f'{np.median(fe[r]):.1e}' for r in ranks]}", flush=True)
    return outp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(n_points=args.n_points, seed=args.seed)
