"""LM-initial-data — 8-D FIELD-ERROR POD-rank sweep: the y-pair CROSS (value+grad).

The 8-D sibling of the 4-D cross field sweep ``run_cross_fielderr_sweep.py``, and
the producer of the paper's 8-D "value+gradient" field-error curve:

  gvm_8d_cross_field -> reports/P3/guess_vs_memory_8d_cross_field_<n>.json
                        the 8-D y-pair CROSS field-error POD sweep — the committed
                        4-D cross path (load_hermite_smolyak_cross +
                        build_pod_hermite_smolyak_cross + rank sweep) pointed at the
                        8-D cross model, dim=8.

  gvm_8d_cross_value_field (``--with-value``, off by default)
                     -> reports/P3/guess_vs_memory_8d_cross_value_field_<n>.json
                        the VALUE-only companion recomputed against the same truth.
                        Redundant now that the truth is cached and shared: the
                        registry's value-only source ``gvm_8d_field`` is the same
                        quantity from the same model against the same cached
                        u_true.  Kept as an opt-in cross-check (~1 h).

The 8-D "value+gradient" model is the y-pair CROSS
``hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross.npz`` (value + gradient in
chi_Ay,chi_By + the mixed 2nd partial d^2U/dchi_Ay dchi_By), NOT the plain 6-spin
gradient-only Hermite (which drops all mixed partials and REGRESSES below value —
the plain-Hermite behavior; the cross fixes it, mirroring the 4-D story).  It is
the model fig03, the 4-D bottom-left panel and the 8-D residual panel all use.

**Output names are load-bearing.**  Until 2026-08-02 this driver wrote the SAME
two filenames as ``run_hermite_fielderr_sweep_8d.py`` (the plain-Hermite sweep),
with no registry entry of its own.  Whichever ran last won.  On the retargeted-box
rebuild the plain sweep completed (20.6 h) and this one died in 11 s on a missing
shard path, so fig05's bottom-right orange curve silently became the six-axis
plain-Hermite model — 1.31e-2 where the cross model reads ~1.35e-3, flipping the
panel's ordering and contradicting fig03.  Do not point two producers at one path.

CERTIFIED TRUTH — cached, not hand-sharded.  ``u_true`` belongs to the PDE + box +
sampler, so ``fielderr_shared.certified_truth`` caches it on a key pinning
box/grid/sampler/seed/tolerance and re-validates on load.  The plain sweep over the
same box populates the same cache, so this driver reuses its ~15 h solve (measured
54.9 s/pt at 1000 points) instead of repeating it.  This replaces the old
``reports/P2/cross/shards/shard_seed0_n1000_*of16.npz`` reader: those shards are
the OLD box (b in [2,7], chi in [-0.99,0.99]; 2026-07-23), so on the production box
they are a different point set entirely — the reader had no way to know, and its
absence under $LM_REPORTS is what killed the rebuild run.

Memory-frugal — ONE model/POD at a time (never both models):
  * CROSS flavor: load the cross model, resolve u_true, build the CROSS POD
    (``build_pod_hermite_smolyak_cross(mc, r=r_full)`` — SVD of ~nfeat x (10*N)
    snapshots, ~40 GB peak), sweep via ``pod.coeffs`` (the committed pattern), free.
  * VALUE-only flavor (``--with-value``): load the plain 6-spin model, build the
    VALUE-only POD (``pod_basis_pool(view, include_derivatives=False)`` on a
    value-only view [enhanced=()] — bit-for-bit the value-only Smolyak interp),
    lean-project, free.
Request --mem=200G.

Schema mirrors ``guess_vs_memory_4d_cross_field_*.json`` (pod_curve of
{r, mem_bytes, min/median/mean/p95/max}; metric ``field_error_relL2``; dim=8), plus
the ``gate`` block from the held-out accuracy check (HISTORY_AND_FINDINGS 2.7).
The gate is FATAL here: a cross model that does not beat value-only must not be
consumed as the paper's value+gradient curve.

Run:
  python -m lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d_cross                # full 1000 pt
  python -m lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d_cross --n-points 5   # smoke test
  python -m lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d_cross --with-value   # + value-only
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
# shared certified-truth cache + the held-out accuracy gate (see that module's docstring)
from lm.initial_data.pipeline.fielderr_shared import (
    attach_gate, certified_truth, enhanced_vs_value, truth_key)

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


def _value_only_medians(n_points):
    """Median-per-rank of the registry's value-only 8-D field curve, or ``None``.

    ``gvm_8d_field`` (written by ``run_hermite_fielderr_sweep_8d.py``) is the same
    value-only quantity, from the same plain model, and — now that the certified
    truth is cached and shared — against the same ``u_true``.  So the gate reads it
    rather than spending an hour recomputing it.  Its rank ladder is the thinned
    subset of this driver's, which is what ``enhanced_vs_value`` intersects on.
    """
    p = os.path.join(REP3, f"guess_vs_memory_8d_field_{n_points}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return {int(e["r"]): float(e["median"]) for e in d["pod_curve"]}


def sweep_cross(pts, key, prob, names, fixed, n_points, seed):
    """The 8-D y-pair CROSS field-error POD sweep (the committed cross path).

    Resolves the certified truth (shared cache) with the cross model loaded, then
    sweeps.  Returns ``(outpath, r_full, median_per_rank, ut, ures)`` so the
    optional value-only companion can reuse the same truth without reloading.
    """
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

    # certified u_true — from the shared cache when the plain sweep over this box has
    # already paid for it, else solved here and cached.  certified_truth re-validates a
    # cache hit against this model's own solve, which is what makes a stale (wrong-box)
    # truth set raise instead of silently scoring against the wrong points.
    attach_solve_fn_3d(mc, prob, names, M_tot=1.0, fixed=fixed, use_cache=False,
                       solver="nk")
    ut, ures, src = certified_truth(mc, pts, key, tag="hf8dx:cross")

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
               u_tol=key["u_tol"], u_steps=key["u_steps"], u_true_source=src,
               u_true_res_med=float(np.median(ures)),
               model=os.path.basename(CROSS_MODEL),
               bare_mem_bytes=8.0 * N * (1 + d + npair) * nfeat, pod_curve=pod_curve)
    outp = os.path.join(REP3, f"guess_vs_memory_8d_cross_field_{n_points}.json")
    _write(outp, out)
    med = {r: float(np.median(fe[r])) for r in ranks}
    print(f"[hf8dx:cross] median field err: {[f'{med[r]:.2e}' for r in ranks]}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    del mc, pod, Phi, mean, fe
    gc.collect()
    return outp, r_full, med, ut, ures


def sweep_value(pts, ut, ures, key, n_points, seed):
    """The VALUE-only plain POD field-error sweep (value basis + value-only interp).

    Opt-in (``--with-value``): the registry's value-only source ``gvm_8d_field`` is
    this same quantity from the same model, and now against the same cached truth,
    so this is a cross-check rather than a distinct measurement.  It writes its own
    path (``..._8d_cross_value_field_<n>.json``) — never the registry's.
    """
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
               u_tol=key["u_tol"], u_steps=key["u_steps"], u_true_source="shared cache",
               u_true_res_med=float(np.median(ures)),
               model=os.path.basename(PLAIN_MODEL),
               bare_mem_bytes=8.0 * N * nfeat, pod_curve=pod_curve)
    outp = os.path.join(REP3, f"guess_vs_memory_8d_cross_value_field_{n_points}.json")
    _write(outp, out)
    med = {r: float(np.median(fe[r])) for r in ranks}
    print(f"[hf8dx:value] median field err: {[f'{med[r]:.2e}' for r in ranks]}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    del mc, view, Phi, mean, fe
    gc.collect()
    return outp, r_full, med


def main(n_points=1000, seed=0, u_tol=1e-11, u_steps=12, with_value=False):
    os.makedirs(REP3, exist_ok=True)
    t0 = time.time()

    meta = _unpack_meta(_load_npz(CROSS_MODEL))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    print(f"[hf8dx] 8-D qc  d={len(box)}  grid={Na}x{Nb}x{Nphi}  axes={names}  "
          f"fixed={fixed}", flush=True)

    # SAME sampler/seed as every other 8-D field sweep, so the cached certified truth
    # (keyed on box/grid/sampler/seed/tol) is shared with run_hermite_fielderr_sweep_8d.
    pts = offnode_points(box, n_points, seed)
    key = truth_key(box=box, Na=Na, Nb=Nb, Nphi=Nphi, sampler="offnode",
                    seed=seed, u_steps=u_steps, u_tol=u_tol, fixed=fixed)

    # CROSS first: it resolves the truth with the cross model loaded and hands it back.
    cross_out, cross_rfull, cross_med, ut, ures = sweep_cross(
        pts, key, prob, names, fixed, n_points, seed)

    val_out = val_rfull = None
    if with_value:
        val_out, val_rfull, val_med = sweep_value(pts, ut, ures, key, n_points, seed)

    print(f"\n[hf8dx] === SUMMARY ({n_points} pts, {(time.time()-t0)/60:.1f} min) ===")
    print(f"[hf8dx]   CROSS value+grad -> {os.path.basename(cross_out)}  (r_full={cross_rfull})")
    if val_out:
        print(f"[hf8dx]   VALUE-only (opt) -> {os.path.basename(val_out)}  (r_full={val_rfull})")

    # ---- held-out accuracy gate (HISTORY_AND_FINDINGS 2.7), FATAL here ----
    # The cross completion is exactly what is supposed to fix the plain-Hermite
    # regression; if it does not, the artifact must not become the paper's curve.
    if not with_value:
        val_med = _value_only_medians(n_points)
    if val_med is None:
        print("[hf8dx] gate SKIPPED: no value-only curve to compare against — rerun the "
              "plain sweep (run_hermite_fielderr_sweep_8d.py) or pass --with-value.",
              flush=True)
    else:
        gate = enhanced_vs_value(val_med, cross_med, tag="hf8dx",
                                 label="8-D y-pair cross (value+grad+cross)",
                                 expect_below=True, fatal=True)
        attach_gate(cross_out, gate)
        if val_out:
            attach_gate(val_out, gate)
    return cross_out, val_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-value", action="store_true",
                    help="also recompute the value-only companion into its OWN path "
                         "(guess_vs_memory_8d_cross_value_field_<n>.json); redundant "
                         "with gvm_8d_field now that the truth is shared")
    args = ap.parse_args()
    main(n_points=args.n_points, seed=args.seed, with_value=args.with_value)
