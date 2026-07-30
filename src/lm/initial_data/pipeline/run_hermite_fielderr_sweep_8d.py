"""LM-initial-data — PLAIN 8-D FIELD-ERROR POD-rank sweep (value-only AND value+gradient).

The 8-D analogue of the 4-D CROSS field-error sweep ``run_cross_fielderr_sweep.py``.
Produces the TWO fig05 bottom-right figure-data sources (registry-pinned):

  gvm_8d_field         -> reports/P3/guess_vs_memory_8d_field_<n>.json
                          (VALUE-only plain POD — value basis + value-only interp)
  gvm_8d_hermite_field -> reports/P3/guess_vs_memory_8d_hermite_field_<n>.json
                          (VALUE+GRADIENT plain-Hermite POD — stacked value+deriv
                           basis + the gradient-enhanced Hermite-Smolyak interp)

Both compute the relative-L2 field error ``||decode_r - u_true|| / ||u_true||`` of
the rank-r plain-Hermite-Smolyak POD guess against the certified NK solve
``u_true``, as the POD rank r is swept over ``pod_rank_ladder(r_full)`` (every other
rung of a 10-point geomspace(1..r_full), plus r_full itself), over
the IDENTICAL 1000 seed-0 off-node points as the other 8-D field sweeps
(``offnode_points`` imported from ``run_cross_fielderror_chi``, same sampler/seed).

Model: the shipped PLAIN gradient (value+gradient) 8-D model
``hermite_smolyak_spin8qc_L5_enh-chi_Ax-chi_Ay-chi_Az-chi_Bx-chi_By-chi_Bz.npz``
(d=8 axes [b,q,chi_A{x,y,z},chi_B{x,y,z}]; 6 enhanced spin axes; N=15713 nodes;
field (45,32,8) -> nfeat=11520; fixed={'qc':1.0}).  NOT the cross model, NOT the
b-q-enhanced one.  Loaded with the PLAIN loader
``hermite_smolyak.load_hermite_smolyak``.

u_true is the plain model's OWN certified NK solve (warm from its full guess), the
same convention as the 4-D driver: ``mc.evaluate_polished(theta, newton_steps=12,
tol=1e-11)``.  ``u_true`` is solved ONCE per point and reused for BOTH flavors.

The two POD flavors differ in BOTH the basis and the parametric interpolant:
  * value-only : POD basis from the VALUE corpus (include_derivatives=False), and
                 the coeff interpolant is VALUE-only (a HermiteSmolyakSolutionND
                 view with enhanced=() -> barycentric, bit-for-bit the value-only
                 Smolyak interpolant).  Stored floats: Phi(nfeat*r)+value coeff
                 (N*r)+mean(nfeat)  -> no tangent-coeff term.
  * value+grad : POD basis from the STACKED value+derivative corpus
                 (include_derivatives=True), and the coeff interpolant is the FULL
                 gradient-enhanced Hermite-Smolyak interpolant (mc itself).  Stored
                 floats: +N*d*r tangent coeffs (all d axes, matching the on-disk
                 node_dU (N,d,*fs)).

Memory-lean: by linearity + the combination property (Sum c_l = 1), the rank-r POD
coeff vector equals ``Phi^T (interp(theta) - mean)`` truncated (this is exactly what
``build_pod_hermite_smolyak`` + ``pod.coeff_model.evaluate`` produce, to roundoff),
so we never materialize a full PODHermiteSmolyak coeff model (~13 GB) — we hold only
Phi (nfeat*r_full, ~1 GB) per flavor + the single loaded interpolant ``mc`` (shared
via a value-only structural view for the value flavor).  ONE Phi at a time.

Writes the two JSONs with schema mirroring ``guess_vs_memory_4d_cross_field_*.json``
(pod_curve of {r, mem_bytes, min/median/mean/p95/max}; metric ``field_error_relL2``;
dim=8; npair dropped).

Add-only.  Imports committed ``lm.initial_data`` modules read-only; edits nothing.

Run:
  python -m lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d                # full 1000 pt
  python -m lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d --n-points 5   # smoke test
"""
from __future__ import annotations

import argparse
import gc
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
# reuse the EXACT seed-shared off-node sampling + stats of the other field sweeps
from lm.initial_data.pipeline.run_cross_fielderror_chi import (
    offnode_points, pod_rank_ladder, stats)

HERE = os.path.dirname(os.path.abspath(__file__))
REP3 = os.path.join(HERE, "reports", "P3")
MODELS = os.path.join(HERE, "reports", "P2", "models_chi")
PLAIN = os.path.join(
    MODELS,
    "hermite_smolyak_spin8qc_L5_enh-chi_Ax-chi_Ay-chi_Az-chi_Bx-chi_By-chi_Bz.npz",
)


def _pod_mem_bytes_value(r, N, nfeat):
    """VALUE-only plain POD stored floats: Phi(nfeat*r) + value coeff(N*r) + mean(nfeat).
    (The cross formula with the d-tangent and npair-cross terms dropped.)"""
    return 8.0 * (nfeat * r + N * r + nfeat)


def _pod_mem_bytes_grad(r, N, nfeat, d):
    """VALUE+GRADIENT plain POD stored floats: Phi(nfeat*r) + value coeff(N*r)
    + d tangent coeffs(N*d*r) + mean(nfeat).  (The cross formula with npair dropped;
    d = all axes, matching the on-disk node_dU (N,d,*fs).)"""
    return 8.0 * (nfeat * r + N * r + N * d * r + nfeat)


def _field_err(u, ut):
    """Relative Frobenius L2 (matches run_cross_fielderror_chi.field_err)."""
    u = np.asarray(u).reshape(-1)
    ut = np.asarray(ut).reshape(-1)
    return float(np.linalg.norm(u - ut) / max(np.linalg.norm(ut), 1e-300))


def _value_only_view(mc: HermiteSmolyakSolutionND) -> HermiteSmolyakSolutionND:
    """A structural VALUE-only view of ``mc``: same subgrids (arrays SHARED via
    dataclasses.replace) but enhanced=() so ``evaluate`` is the barycentric
    value-only interpolant (bit-for-bit the value-only Smolyak limit — the
    HermiteSolutionND/HermiteSmolyakSolutionND docstrings guarantee this)."""
    value_subs = [replace(sub, enhanced=()) for sub in mc.subgrids]
    return HermiteSmolyakSolutionND(
        axes=list(mc.axes), index_set=[tuple(l) for l in mc.index_set],
        coeffs=list(mc.coeffs), subgrids=value_subs, enhanced=(),
        n_solver_nodes=mc.n_solver_nodes, total_iters=mc.total_iters, _solve_fn=None)


def _sweep_flavor(flavor, view, include_derivatives, mem_fn, bare_mem_bytes,
                  pts, ut_flat, ures, N, nfeat, d, n_points, seed, u_tol, u_steps):
    """Build the flavor's POD basis, sweep POD rank, write the JSON.  Returns
    (outpath, r_full, median_field_err_per_rank)."""
    t0 = time.time()
    print(f"[hf8d:{flavor}] building POD basis "
          f"(include_derivatives={include_derivatives}) ...", flush=True)
    # full-rank basis: request r=nfeat (the hard cap min(nfeat, ncols)); read back r_full
    Phi, mean, diag = pod_basis_pool(view, r=nfeat,
                                     include_derivatives=include_derivatives)
    r_full = int(Phi.shape[1])
    mean = np.asarray(mean, dtype=float).reshape(-1)
    print(f"[hf8d:{flavor}] Phi {Phi.shape}  r_full={r_full}  "
          f"rank_value(1e-6)={diag['rank_value'].get(1e-6)}  "
          f"rank_stacked(1e-6)={diag['rank_stacked'].get(1e-6)}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    ranks = pod_rank_ladder(r_full)
    print(f"[hf8d:{flavor}] sweep ranks={ranks}", flush=True)

    fe = {r: [] for r in ranks}
    tev = time.time()
    for i, th in enumerate(pts):
        u_full = np.asarray(view.evaluate(th)).reshape(-1)        # the flavor's interpolant
        c = Phi.T @ (u_full - mean)                               # (r_full,) POD coeffs
        for r in ranks:
            u = mean + Phi[:, :r] @ c[:r]
            fe[r].append(_field_err(u, ut_flat[i]))
        if (i + 1) % 50 == 0 or i == n_points - 1:
            el = time.time() - tev
            rate = el / (i + 1)
            print(f"   [{flavor}] sweep {i+1}/{n_points} ({el:.0f}s, {rate:.3f}s/pt, "
                  f"ETA {rate*(n_points-i-1)/60:.1f} min)  "
                  f"med field-err@r_full={np.median(fe[ranks[-1]]):.2e}", flush=True)

    pod_curve = [dict(r=int(r), mem_bytes=mem_fn(r), **stats(fe[r])) for r in ranks]
    out = dict(dim=8, metric="field_error_relL2", flavor=flavor,
               include_derivatives=bool(include_derivatives),
               r_full=int(r_full), r_cap=int(r_full),
               N=int(N), nfeat=int(nfeat), d=int(d),
               n_points=int(n_points), seed=int(seed), n_ranks=len(ranks),
               u_tol=u_tol, u_steps=u_steps, u_true_res_med=float(np.median(ures)),
               model=os.path.basename(PLAIN),
               bare_mem_bytes=float(bare_mem_bytes), pod_curve=pod_curve)
    tag = "hermite_field" if include_derivatives else "field"
    outp = os.path.join(REP3, f"guess_vs_memory_8d_{tag}_{n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    med = [float(np.median(fe[r])) for r in ranks]
    print(f"[hf8d:{flavor}] wrote {os.path.basename(outp)}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[hf8d:{flavor}] median field err [r=1..r_full]: "
          f"{[f'{m:.2e}' for m in med]}", flush=True)

    del Phi, mean, c, u_full, fe
    gc.collect()
    return outp, r_full, dict(zip(ranks, med))


def main(n_points=1000, seed=0, u_tol=1e-11, u_steps=12):
    os.makedirs(REP3, exist_ok=True)
    t0 = time.time()
    meta = _unpack_meta(_load_npz(PLAIN))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    print(f"[hf8d] PLAIN 8-D qc  d={len(box)}  grid={Na}x{Nb}x{Nphi}  "
          f"axes={names}  fixed={fixed}", flush=True)

    print(f"[hf8d] loading plain model ({os.path.getsize(PLAIN)/1e9:.0f} GB) ...",
          flush=True)
    mc = load_hermite_smolyak(PLAIN)
    N = int(mc.n_solver_nodes)
    nfeat = int(np.prod(mc.field_shape))
    d = int(mc.d)
    print(f"[hf8d] model: N={N} nfeat={nfeat} d={d} field_shape={mc.field_shape} "
          f"enhanced={mc.enhanced}  ({time.time()-t0:.0f}s)", flush=True)

    # certified u_true via the plain model's own NK solve (warm from its full guess)
    attach_solve_fn_3d(mc, prob, names, M_tot=1.0, fixed=fixed, use_cache=False,
                       solver="nk")

    pts = offnode_points(box, n_points, seed)

    # ---- phase 1: certified truth, ONCE per point, reused for both flavors ----
    print(f"[hf8d] solving certified u_true at {n_points} off-node points "
          f"(seed={seed}, newton_steps={u_steps}, tol={u_tol:.0e}) ...", flush=True)
    ut_flat, ures = [], []
    tt = time.time()
    for i, th in enumerate(pts):
        ut, info = mc.evaluate_polished(th, newton_steps=u_steps, tol=u_tol)
        ut_flat.append(np.asarray(ut, dtype=float).reshape(-1))
        ures.append(float(info.residual_norm))
        if (i + 1) % 25 == 0 or i == n_points - 1 or n_points <= 10:
            el = time.time() - tt
            rate = el / (i + 1)
            print(f"   u_true {i+1}/{n_points} ({el:.0f}s, {rate:.2f}s/pt, "
                  f"ETA {rate*(n_points-i-1)/60:.1f} min)  "
                  f"res med={np.median(ures):.1e}", flush=True)
    print(f"[hf8d] u_true done: {n_points} pts, {(time.time()-tt)/60:.1f} min, "
          f"{(time.time()-tt)/n_points:.2f}s/pt, res med={np.median(ures):.1e}",
          flush=True)

    # ---- flavor sweeps: ONE Phi held at a time (mc shared; value view free) ----
    # value+gradient first (heavier basis), then value-only.
    grad_out, grad_rfull, grad_med = _sweep_flavor(
        "value+grad", mc, True, lambda r: _pod_mem_bytes_grad(r, N, nfeat, d),
        8.0 * N * (1 + d) * nfeat,
        pts, ut_flat, ures, N, nfeat, d, n_points, seed, u_tol, u_steps)

    value_view = _value_only_view(mc)
    val_out, val_rfull, val_med = _sweep_flavor(
        "value-only", value_view, False, lambda r: _pod_mem_bytes_value(r, N, nfeat),
        8.0 * N * nfeat,
        pts, ut_flat, ures, N, nfeat, d, n_points, seed, u_tol, u_steps)

    # ---- summary + the "value+grad at/below value-only" sanity check ----
    print(f"\n[hf8d] === SUMMARY ({n_points} pts, {(time.time()-t0)/60:.1f} min) ===")
    print(f"[hf8d]   VALUE-only  -> {os.path.basename(val_out)}  (r_full={val_rfull})")
    print(f"[hf8d]   VALUE+GRAD  -> {os.path.basename(grad_out)}  (r_full={grad_rfull})")
    common = sorted(set(val_med) & set(grad_med))
    below = sum(1 for r in common if grad_med[r] <= val_med[r] * 1.05)
    print(f"[hf8d]   value+grad <= 1.05*value-only at {below}/{len(common)} shared ranks")
    for r in common:
        print(f"      r={r:6d}  value-only={val_med[r]:.3e}  value+grad={grad_med[r]:.3e}")
    return val_out, grad_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(n_points=args.n_points, seed=args.seed)
