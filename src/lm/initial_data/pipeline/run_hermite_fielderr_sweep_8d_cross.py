"""LM-initial-data — 8-D FIELD-ERROR POD-rank sweep: VALUE-only + y-pair CROSS (value+grad).

Rework of ``run_hermite_fielderr_sweep_8d.py`` per the 8-D "value+gradient = the
y-pair CROSS model" directive.  Produces the two fig06 bottom-right sources at the
registry paths:

  gvm_8d_field         -> reports/P3/guess_vs_memory_8d_field_<n>.json
                          VALUE-only plain POD (basis + value-only interp)
  gvm_8d_hermite_field -> reports/P3/guess_vs_memory_8d_hermite_field_<n>.json
                          the 8-D y-pair CROSS field-error POD sweep — i.e. the
                          committed 4-D ``run_cross_fielderr_sweep.py`` cross path
                          (load_hermite_smolyak_cross + build_pod_hermite_smolyak_cross
                          + rank sweep) pointed at the 8-D cross model, dim=8.

The 8-D "value+gradient" model is the y-pair CROSS
``hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross.npz`` (value + gradient in
chi_Ay,chi_By + the mixed 2nd partial d^2U/dchi_Ay dchi_By), NOT the plain 6-spin
gradient-only Hermite (which drops all mixed partials and REGRESSES below value —
the plain-Hermite behavior; the cross fixes it, mirroring the 4-D story).

BIG OPTIMIZATION — reuse the saved certified truth (no ~14 h re-solve).  The
IDENTICAL ``offnode_points(box,1000,0)`` certified ``u_true`` is on disk in
``reports/P2/cross/shards/shard_seed0_n1000_*of16.npz`` (16 files, arrays
``idx_off`` + ``UT``; 1000/1000 coverage).  ``UT[i]`` == the 8-D y-pair cross model's
``evaluate_polished(pts[i], newton_steps=12, tol=1e-11)`` at
``offnode_points(cross_box,1000,0)[i]`` (same model/points/sampler/tol as
``run_cross_fielderr_sweep``).  We reassemble ``UT[1000]`` by ``idx_off``, VALIDATE by
re-solving 2-3 points with the cross model's ``evaluate_polished`` (assert rel-L2 <
1e-8), then use it as ``u_true`` for BOTH sweeps (the same true PDE solution).

Memory-frugal — ONE model/POD at a time (never both models):
  * CROSS flavor: load the cross model, validate u_true, build the CROSS POD
    (``build_pod_hermite_smolyak_cross(mc, r=r_full)`` — SVD of ~nfeat x (10*N)
    snapshots, ~40 GB peak), sweep via ``pod.coeffs`` (the committed pattern), free.
  * VALUE-only flavor: load the plain 6-spin model, build the VALUE-only POD
    (``pod_basis_pool(view, include_derivatives=False)`` on a value-only view
    [enhanced=()] — bit-for-bit the value-only Smolyak interp), lean-project, free.
Request --mem=200G.

Schema mirrors ``guess_vs_memory_4d_cross_field_*.json`` (pod_curve of
{r, mem_bytes, min/median/mean/p95/max}; metric ``field_error_relL2``; dim=8).

Add-only.  Reuses committed ``lm.initial_data`` helpers verbatim; edits nothing.

Run:
  python -m lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d_cross                # full 1000 pt
  python -m lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d_cross --n-points 5   # smoke test
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric.parametric_nd import _load_npz, _unpack_meta, attach_solve_fn_3d
from lm.initial_data.parametric.hermite_smolyak import (
    load_hermite_smolyak,
    HermiteSmolyakSolutionND,
)
from lm.initial_data.parametric.hermite_smolyak_pod import pod_basis_pool
from lm.initial_data.parametric.hermite_smolyak_cross import load_hermite_smolyak_cross
from lm.initial_data.parametric.hermite_smolyak_pod_cross import build_pod_hermite_smolyak_cross
# reuse the EXACT seed-shared off-node sampling + stats of the residual/field sweeps
from lm.initial_data.pipeline.run_cross_fielderror_chi import offnode_points, stats

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REP3 = os.path.join(REPORTS, "P3")
MODELS = os.path.join(REPORTS, "P2", "models_chi")
CROSS_MODEL = os.path.join(
    MODELS, "hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross.npz")
PLAIN_MODEL = os.path.join(
    MODELS,
    "hermite_smolyak_spin8qc_L5_enh-chi_Ax-chi_Ay-chi_Az-chi_Bx-chi_By-chi_Bz.npz")
SHARD_GLOB = os.path.join(REPORTS, "P2", "cross", "shards",
                          "shard_seed0_n1000_*of16.npz")
N_SAVED = 1000            # the saved-truth point count (offnode_points(box, 1000, 0))


def _pod_mem_bytes_cross(r, N, nfeat, d, npair):
    """CROSS POD stored floats — IDENTICAL to run_cross_fielderr_sweep:
    Phi(nfeat*r) + value coeff(N*r) + d tangents(N*d*r) + npair cross(N*npair*r) + mean(nfeat)."""
    return 8.0 * (nfeat * r + N * r + N * d * r + N * npair * r + nfeat)


def _pod_mem_bytes_value(r, N, nfeat):
    """VALUE-only POD stored floats: Phi(nfeat*r) + value coeff(N*r) + mean(nfeat)."""
    return 8.0 * (nfeat * r + N * r + nfeat)


def _field_err(u, ut):
    """Relative Frobenius L2 (matches run_cross_fielderror_chi.field_err)."""
    u = np.asarray(u).reshape(-1)
    ut = np.asarray(ut).reshape(-1)
    return float(np.linalg.norm(u - ut) / max(np.linalg.norm(ut), 1e-300))


def load_saved_utrue():
    """Reassemble the seed-0 certified truth ``UT[N_SAVED, nfeat]`` from the 16
    cross shard files (arrays ``idx_off`` + ``UT``).  Asserts full 0..N_SAVED-1
    coverage."""
    files = sorted(glob.glob(SHARD_GLOB))
    if not files:
        raise FileNotFoundError(f"no shard files matching {SHARD_GLOB}")
    UT_full = None
    seen = np.zeros(N_SAVED, dtype=bool)
    for p in files:
        d = np.load(p, allow_pickle=True)
        idx = np.asarray(d["idx_off"]).astype(int)
        UT = np.asarray(d["UT"], dtype=float)
        if UT_full is None:
            UT_full = np.zeros((N_SAVED, UT.shape[1]), dtype=float)
        UT_full[idx] = UT
        seen[idx] = True
    if not seen.all():
        raise ValueError(f"shard coverage incomplete: {int(seen.sum())}/{N_SAVED}")
    print(f"[hf8dx] reassembled saved u_true: {UT_full.shape} from {len(files)} shards "
          f"(coverage {int(seen.sum())}/{N_SAVED})", flush=True)
    return UT_full


def _value_only_view(mc: HermiteSmolyakSolutionND) -> HermiteSmolyakSolutionND:
    """Structural VALUE-only view of ``mc``: subgrid arrays SHARED (dataclasses.replace)
    but enhanced=() so ``evaluate`` is the barycentric value-only interpolant."""
    value_subs = [replace(sub, enhanced=()) for sub in mc.subgrids]
    return HermiteSmolyakSolutionND(
        axes=list(mc.axes), index_set=[tuple(l) for l in mc.index_set],
        coeffs=list(mc.coeffs), subgrids=value_subs, enhanced=(),
        n_solver_nodes=mc.n_solver_nodes, total_iters=mc.total_iters, _solve_fn=None)


def _write(outp, out):
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[hf8dx] wrote {os.path.basename(outp)}", flush=True)


def sweep_cross(pts, ut, prob, names, fixed, n_points, seed, u_tol, u_steps):
    """The 8-D y-pair CROSS field-error POD sweep (the committed cross path)."""
    t0 = time.time()
    print(f"[hf8dx:cross] loading cross model "
          f"({os.path.getsize(CROSS_MODEL)/1e9:.0f} GB) ...", flush=True)
    mc = load_hermite_smolyak_cross(CROSS_MODEL)
    N = int(mc.n_solver_nodes)
    nfeat = int(np.prod(mc.field_shape))
    d = int(mc.d)
    npair = len(mc.cross_pairs_global)
    print(f"[hf8dx:cross] N={N} nfeat={nfeat} d={d} npair={npair} "
          f"pairs={mc.cross_pairs_global}  ({time.time()-t0:.0f}s)", flush=True)

    # validate the reused saved u_true against the cross model's own certified solve
    attach_solve_fn_3d(mc, prob, names, M_tot=1.0, fixed=fixed, use_cache=False,
                       solver="nk")
    nval = min(3, n_points)
    print(f"[hf8dx:cross] validating reused u_true on {nval} re-solves "
          f"(assert rel-L2 < 1e-8) ...", flush=True)
    vmax = 0.0
    for k in range(nval):
        u, info = mc.evaluate_polished(pts[k], newton_steps=u_steps, tol=u_tol)
        rel = _field_err(u, ut[k])
        vmax = max(vmax, rel)
        print(f"   validate pt {k}: rel-L2(re-solve, saved UT)={rel:.2e} "
              f"(res={info.residual_norm:.1e})", flush=True)
    if vmax >= 1e-8:
        raise AssertionError(f"reused u_true validation FAILED: max rel-L2={vmax:.2e} >= 1e-8")
    print(f"[hf8dx:cross] u_true reuse VALIDATED (max rel-L2={vmax:.2e})", flush=True)

    r_req = (1 + d + npair) * N
    print(f"[hf8dx:cross] building CROSS POD (r_req={r_req}) ...", flush=True)
    pod, _diag = build_pod_hermite_smolyak_cross(mc, r=r_req)
    r_full = int(pod.r)
    Phi, mean = pod.Phi, np.asarray(pod.mean, dtype=float).reshape(-1)
    print(f"[hf8dx:cross] Phi {Phi.shape}  r_full={r_full}  ({time.time()-t0:.0f}s)",
          flush=True)

    ranks = sorted(set(int(round(x)) for x in np.geomspace(1, r_full, 10)))
    ranks = [r for r in ranks if 1 <= r <= r_full]
    print(f"[hf8dx:cross] sweep ranks={ranks}", flush=True)

    fe = {r: [] for r in ranks}
    tev = time.time()
    for i, th in enumerate(pts):
        c = pod.coeffs(th)                                    # (r_full,)
        for r in ranks:
            u = mean + Phi[:, :r] @ c[:r]
            fe[r].append(_field_err(u, ut[i]))
        if (i + 1) % 25 == 0 or i == n_points - 1 or n_points <= 10:
            el = time.time() - tev
            rate = el / (i + 1)
            print(f"   [cross] sweep {i+1}/{n_points} ({el:.0f}s, {rate:.3f}s/pt, "
                  f"ETA {rate*(n_points-i-1)/60:.1f} min)  "
                  f"med field-err@r_full={np.median(fe[ranks[-1]]):.2e}", flush=True)

    pod_curve = [dict(r=int(r), mem_bytes=_pod_mem_bytes_cross(r, N, nfeat, d, npair),
                      **stats(fe[r])) for r in ranks]
    out = dict(dim=8, metric="field_error_relL2", flavor="value+grad-cross",
               r_full=int(r_full), r_cap=int(r_full),
               N=int(N), nfeat=int(nfeat), d=int(d), npair=int(npair),
               n_points=int(n_points), seed=int(seed), n_ranks=len(ranks),
               u_tol=u_tol, u_steps=u_steps, u_true_source="reports/P2/cross/shards",
               u_true_reuse_val_relL2=float(vmax), model=os.path.basename(CROSS_MODEL),
               bare_mem_bytes=8.0 * N * (1 + d + npair) * nfeat, pod_curve=pod_curve)
    outp = os.path.join(REP3, f"guess_vs_memory_8d_hermite_field_{n_points}.json")
    _write(outp, out)
    med = {r: float(np.median(fe[r])) for r in ranks}
    print(f"[hf8dx:cross] median field err: {[f'{med[r]:.2e}' for r in ranks]}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    del mc, pod, Phi, mean, fe
    gc.collect()
    return outp, r_full, med


def sweep_value(pts, ut, prob, n_points, seed, u_tol, u_steps):
    """The VALUE-only plain POD field-error sweep (value basis + value-only interp)."""
    t0 = time.time()
    print(f"[hf8dx:value] loading plain model "
          f"({os.path.getsize(PLAIN_MODEL)/1e9:.0f} GB) ...", flush=True)
    mc = load_hermite_smolyak(PLAIN_MODEL)
    N = int(mc.n_solver_nodes)
    nfeat = int(np.prod(mc.field_shape))
    d = int(mc.d)
    view = _value_only_view(mc)
    print(f"[hf8dx:value] N={N} nfeat={nfeat} d={d}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"[hf8dx:value] building VALUE-only POD (include_derivatives=False) ...",
          flush=True)
    Phi, mean, diag = pod_basis_pool(view, r=nfeat, include_derivatives=False)
    r_full = int(Phi.shape[1])
    mean = np.asarray(mean, dtype=float).reshape(-1)
    print(f"[hf8dx:value] Phi {Phi.shape}  r_full={r_full}  "
          f"rank_value(1e-6)={diag['rank_value'].get(1e-6)}  ({time.time()-t0:.0f}s)",
          flush=True)

    ranks = sorted(set(int(round(x)) for x in np.geomspace(1, r_full, 10)))
    ranks = [r for r in ranks if 1 <= r <= r_full]
    print(f"[hf8dx:value] sweep ranks={ranks}", flush=True)

    fe = {r: [] for r in ranks}
    tev = time.time()
    for i, th in enumerate(pts):
        u_full = np.asarray(view.evaluate(th)).reshape(-1)    # value-only interpolant
        c = Phi.T @ (u_full - mean)                           # (r_full,) POD coeffs
        for r in ranks:
            u = mean + Phi[:, :r] @ c[:r]
            fe[r].append(_field_err(u, ut[i]))
        if (i + 1) % 25 == 0 or i == n_points - 1 or n_points <= 10:
            el = time.time() - tev
            rate = el / (i + 1)
            print(f"   [value] sweep {i+1}/{n_points} ({el:.0f}s, {rate:.3f}s/pt, "
                  f"ETA {rate*(n_points-i-1)/60:.1f} min)  "
                  f"med field-err@r_full={np.median(fe[ranks[-1]]):.2e}", flush=True)

    pod_curve = [dict(r=int(r), mem_bytes=_pod_mem_bytes_value(r, N, nfeat),
                      **stats(fe[r])) for r in ranks]
    out = dict(dim=8, metric="field_error_relL2", flavor="value-only",
               r_full=int(r_full), r_cap=int(r_full),
               N=int(N), nfeat=int(nfeat), d=int(d),
               n_points=int(n_points), seed=int(seed), n_ranks=len(ranks),
               u_tol=u_tol, u_steps=u_steps, u_true_source="reports/P2/cross/shards",
               model=os.path.basename(PLAIN_MODEL),
               bare_mem_bytes=8.0 * N * nfeat, pod_curve=pod_curve)
    outp = os.path.join(REP3, f"guess_vs_memory_8d_field_{n_points}.json")
    _write(outp, out)
    med = {r: float(np.median(fe[r])) for r in ranks}
    print(f"[hf8dx:value] median field err: {[f'{med[r]:.2e}' for r in ranks]}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    del mc, view, Phi, mean, fe
    gc.collect()
    return outp, r_full, med


def main(n_points=1000, seed=0, u_tol=1e-11, u_steps=12):
    os.makedirs(REP3, exist_ok=True)
    t0 = time.time()
    if n_points > N_SAVED:
        raise ValueError(f"n_points={n_points} > saved truth {N_SAVED}")

    meta = _unpack_meta(_load_npz(CROSS_MODEL))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    print(f"[hf8dx] 8-D qc  d={len(box)}  grid={Na}x{Nb}x{Nphi}  axes={names}  "
          f"fixed={fixed}", flush=True)

    # SAME sampler/seed as the shards + all other 8-D field sweeps
    pts = offnode_points(box, n_points, seed)
    UT_full = load_saved_utrue()
    ut = [UT_full[i] for i in range(n_points)]   # pts[:n]==offnode(1000)[:n] (same rng stream)

    # CROSS first (validates the reused u_true via the cross model's own solve), then value
    cross_out, cross_rfull, cross_med = sweep_cross(
        pts, ut, prob, names, fixed, n_points, seed, u_tol, u_steps)
    val_out, val_rfull, val_med = sweep_value(
        pts, ut, prob, n_points, seed, u_tol, u_steps)

    print(f"\n[hf8dx] === SUMMARY ({n_points} pts, {(time.time()-t0)/60:.1f} min) ===")
    print(f"[hf8dx]   VALUE-only       -> {os.path.basename(val_out)}  (r_full={val_rfull})")
    print(f"[hf8dx]   CROSS value+grad -> {os.path.basename(cross_out)}  (r_full={cross_rfull})")
    common = sorted(set(val_med) & set(cross_med))
    below = sum(1 for r in common if cross_med[r] <= val_med[r] * 1.05)
    print(f"[hf8dx]   cross <= 1.05*value-only at {below}/{len(common)} shared ranks "
          f"(cross should be BELOW value-only — the plain-Hermite regression fixed)")
    for r in common:
        flag = "  <-- cross below" if cross_med[r] <= val_med[r] else ""
        print(f"      r={r:6d}  value-only={val_med[r]:.3e}  cross={cross_med[r]:.3e}{flag}")
    return val_out, cross_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(n_points=args.n_points, seed=args.seed)
