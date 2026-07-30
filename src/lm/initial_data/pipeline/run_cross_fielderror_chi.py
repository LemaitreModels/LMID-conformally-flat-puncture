"""PARASOL — M4 verification: JOINT bare-guess field error of the value-only,
value+gradient (shipped), and value+gradient+cross (full bilinear) 4-D χ models.

Add-only.  At the SAME 1000 seed-0 off-node points (``run_polish_table``'s
``random_offnode_points``), computes the raw-interpolant field error
``‖interp − u_true‖₂ / ‖u_true‖₂`` for each model, where ``u_true`` is the
certified solve (``evaluate_polished`` to ``tol=1e-11``, one per point, reused
across models).  Reports the joint-error distribution per model + the per-axis
error on the two enhanced axes, and gates that the with-cross joint floor drops to
≤ the value-only joint floor (the ~5× gradient-only gap closes) while per-axis
stays as good.  Resumable (checkpoints the per-point errors).

Run (from repo root):
  caffeinate -i python sandbox/parasol/run_cross_fielderror_chi.py \
      --grad  sandbox/parasol/reports/P2/models_chi/hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By.npz \
      --cross sandbox/parasol/reports/P2/models_chi/hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz
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
from lm.initial_data.parametric.parametric import cheb_param_nodes
from lm.initial_data.parametric.parametric_nd import _load_npz, _unpack_meta
from lm.initial_data.parametric.parametric_nd_smolyak import combination_coeffs  # noqa: F401
from lm.initial_data.parametric.parametric_nd import attach_solve_fn_3d
from lm.initial_data.parametric.parametric_nd_3d import theta_to_slice3d
from lm.initial_data.parametric.hermite_smolyak import load_hermite_smolyak, HermiteSmolyakSolverND
from lm.initial_data.parametric.hermite_smolyak_cross import load_hermite_smolyak_cross

GAP_MIN = 1e-4
HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "P2", "cross")


def offnode_points(box, n, seed):
    """Mirror run_polish_table.random_offnode_points EXACTLY (same seed/box/gap)."""
    rng = np.random.default_rng(seed)
    ns = [cheb_param_nodes(lo, hi, 16)[0] for lo, hi in box]
    d = len(box)
    pts = np.empty((n, d))
    for i in range(n):
        while True:
            theta = np.array([rng.uniform(lo, hi) for lo, hi in box])
            if all(np.min(np.abs(theta[k] - ns[k])) > GAP_MIN for k in range(d)):
                break
        pts[i] = theta
    return pts


def per_axis_points(box, names, axis_names, n, seed, center):
    """n off-node points varying ONE axis over its box, others fixed at ``center``."""
    out = {}
    for axname in axis_names:
        k = names.index(axname)
        lo, hi = box[k]
        rng = np.random.default_rng(seed + 100 + k)
        ns = cheb_param_nodes(lo, hi, 16)[0]
        pts = np.tile(np.asarray(center, float), (n, 1))
        col = []
        while len(col) < n:
            v = rng.uniform(lo, hi)
            if np.min(np.abs(v - ns)) > GAP_MIN:
                col.append(v)
        pts[:, k] = np.array(col)
        out[axname] = pts
    return out


def stats(a):
    a = np.asarray(a, float)
    return dict(min=float(a.min()), median=float(np.median(a)), mean=float(a.mean()),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def thin_ranks(dense, every=2):
    """Keep every ``every``-th rung of a POD rank ladder, always keeping the last.

    ``every=1`` returns the dense ladder.  The default ``every=2`` halves the sweep
    cost (and the number of plotted points in Fig. 5, which was too crowded at the
    dense ladder) while keeping both endpoints, so the curve still spans the whole
    memory range.  The final rung is always kept because the full-rank point supplies
    the bare-guess reference for the field-error panels.
    """
    dense = sorted(set(int(r) for r in dense))
    return sorted(set(dense[:-1][::every] + dense[-1:]))


def pod_rank_ladder(r_full, n=10, every=2):
    """The shared POD rank sweep: ``thin_ranks`` of an ``n``-point geomspace(1, r_full)."""
    dense = [int(round(x)) for x in np.geomspace(1, r_full, n)]
    return [r for r in thin_ranks(dense, every) if 1 <= r <= r_full]


def field_err(interp, u_true):
    d = np.asarray(interp) - np.asarray(u_true)
    return float(np.linalg.norm(d) / max(np.linalg.norm(u_true), 1e-300))


def build_value_only(grad_model):
    """Value-only interpolant from the SAME node pool (enhanced=()) — the
    reduce-to-committed value baseline (committed HermiteSmolyakSolverND)."""
    pool = {k: (v[1], v[2], v[3], v[4])          # (U, dU, iters, resid)
            for k, v in grad_model._dedup_pool().items()}
    solver = HermiteSmolyakSolverND(None, grad_model.axes, None, enhanced_axes=())
    return solver._finalize(grad_model.index_set, pool)


def main(cross_path, n_points=1000, seed=0, n_peraxis=60,
         u_tol=1e-11, u_steps=12, grad_path=None):
    os.makedirs(REPDIR, exist_ok=True)
    t0 = time.time()
    # The cross model carries the full corpus + meta; derive value / gradient /
    # cross ALL from it so the gradient baseline uses the SAME enhanced set as the
    # cross (matches the shipped model for 4D; the y-pair for the 8D "same-as-4D"
    # run).  grad_path is accepted for provenance only.
    meta = _unpack_meta(_load_npz(cross_path))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    enhanced_names = list(meta.get("enhanced", []))
    print(f"[m4] names={names} box={box} enhanced={enhanced_names} "
          f"grid Na={Na},Nb={Nb},Nphi={Nphi} fixed={fixed}", flush=True)

    prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=Nphi)
    print(f"[m4] loading cross model + deriving value/gradient baselines ...", flush=True)
    mc = load_hermite_smolyak_cross(cross_path)          # value+gradient+cross
    enh = tuple(mc.enhanced)
    pool_grad = {k: (v[1], v[2], v[4], v[5]) for k, v in mc.pool.items()}   # (U,dU,iters,resid)
    mg = HermiteSmolyakSolverND(None, mc.axes, None, enhanced_axes=enh
                                )._finalize(mc.index_set, pool_grad)         # value+gradient
    mv = HermiteSmolyakSolverND(None, mc.axes, None, enhanced_axes=()
                                )._finalize(mc.index_set, pool_grad)         # value-only
    # certified u_true via the cross model's solve_fn (best guess); use_cache=False
    attach_solve_fn_3d(mc, prob, names, M_tot=1.0, fixed=fixed, use_cache=False, solver="nk")

    def u_true_at(theta):
        U, info = mc.evaluate_polished(theta, newton_steps=u_steps, tol=u_tol)
        return np.asarray(U), float(info.residual_norm)

    # -------- JOINT field error AND bare-guess EQUILIBRATED residual --------
    # field error = ‖interp − u_true‖₂/‖u_true‖₂ (vs the certified solve);
    # constraint residual = the EQUILIBRATED ‖R‖_equil (solver_3d_nk, row-scaled —
    # the paper convention, notes/conventions.md; NOT the raw nodal residual, which
    # is roundoff-dominated near the inner axis). Needs NO reference solve — one
    # shared assembly + scales per point.
    pts = offnode_points(box, n_points, seed)
    ck = os.path.join(REPDIR, f"m4_joint_seed{seed}_n{n_points}.npz")
    EV = np.full(n_points, np.nan); EG = np.full(n_points, np.nan); EC = np.full(n_points, np.nan)
    RV = np.full(n_points, np.nan); RG = np.full(n_points, np.nan); RC = np.full(n_points, np.nan)
    RES = np.full(n_points, np.nan)
    start = 0
    if os.path.exists(ck):
        z = np.load(ck)
        if (z["pts"].shape == pts.shape and np.allclose(z["pts"], pts) and "RC" in z.files):
            EV, EG, EC = z["EV"].copy(), z["EG"].copy(), z["EC"].copy()
            RV, RG, RC = z["RV"].copy(), z["RG"].copy(), z["RC"].copy()
            RES = z["RES"].copy()
            done = np.isfinite(EC) & np.isfinite(RC)
            start = int(np.argmin(done)) if not done.all() else n_points
            print(f"[m4] resumed joint from checkpoint: {start}/{n_points}", flush=True)

    ntot2d = prob.Ntot2d
    tev = time.time()
    for i in range(start, n_points):
        th = pts[i]
        ut, res = u_true_at(th)
        gv = mv.evaluate(th); gg = mg.evaluate(th); gc = mc.evaluate(th)
        EV[i] = field_err(gv, ut); EG[i] = field_err(gg, ut); EC[i] = field_err(gc, ut)
        sl = theta_to_slice3d(th, names, 1.0, fixed)
        asm = s3.assemble(prob, sl)                       # shared across the 3 guesses
        scales = s3nk._block_scales(asm)                  # equilibrated norm (paper convention)
        RV[i] = s3nk.equil_residual_inf(asm, np.asarray(gv).reshape(ntot2d, Nphi), scales)
        RG[i] = s3nk.equil_residual_inf(asm, np.asarray(gg).reshape(ntot2d, Nphi), scales)
        RC[i] = s3nk.equil_residual_inf(asm, np.asarray(gc).reshape(ntot2d, Nphi), scales)
        RES[i] = res
        if (i + 1) % 50 == 0 or i == n_points - 1:
            np.savez(ck, pts=pts, EV=EV, EG=EG, EC=EC, RV=RV, RG=RG, RC=RC, RES=RES)
            el = time.time() - tev
            rate = el / (i + 1 - start)
            print(f"   joint {i+1}/{n_points} ({el:.0f}s, {rate:.2f}s/pt, "
                  f"ETA {rate*(n_points-1-i)/60:.1f} min)  "
                  f"fielderr med[V,G,C]=[{np.nanmedian(EV):.2e},{np.nanmedian(EG):.2e},"
                  f"{np.nanmedian(EC):.2e}]  resid med[V,G,C]=[{np.nanmedian(RV):.2e},"
                  f"{np.nanmedian(RG):.2e},{np.nanmedian(RC):.2e}]", flush=True)

    joint = {
        "field_error": {"value": stats(EV), "value+grad": stats(EG),
                        "value+grad+cross": stats(EC)},
        "constraint_residual": {"value": stats(RV), "value+grad": stats(RG),
                                "value+grad+cross": stats(RC),
                                "frac_certified_le_1e-10": {
                                    "value": float(np.mean(RV <= 1e-10)),
                                    "value+grad": float(np.mean(RG <= 1e-10)),
                                    "value+grad+cross": float(np.mean(RC <= 1e-10))}},
        "u_true_res_max": float(np.nanmax(RES))}

    # ---------- per-axis field error on the two enhanced axes ----------
    center = np.array([0.5 * (lo + hi) for (lo, hi) in box])
    pax = per_axis_points(box, names, enhanced_names, n_peraxis, seed, center)
    peraxis = {}
    for axname, P in pax.items():
        ev = np.empty(len(P)); eg = np.empty(len(P)); ec = np.empty(len(P))
        for j, th in enumerate(P):
            ut, _ = u_true_at(th)
            ev[j] = field_err(mv.evaluate(th), ut)
            eg[j] = field_err(mg.evaluate(th), ut)
            ec[j] = field_err(mc.evaluate(th), ut)
        peraxis[axname] = {"value": stats(ev), "value+grad": stats(eg),
                           "value+grad+cross": stats(ec)}
        print(f"[m4] per-axis {axname}: median V={np.median(ev):.2e} "
              f"G={np.median(eg):.2e} C={np.median(ec):.2e}", flush=True)

    out = {"config": {"grad_model": os.path.basename(grad_path),
                      "cross_model": os.path.basename(cross_path),
                      "n_points": n_points, "seed": seed, "gap_min": GAP_MIN,
                      "u_tol": u_tol, "n_peraxis": n_peraxis,
                      "box": box, "names": names, "enhanced": enhanced_names},
           "joint": joint, "per_axis": peraxis, "wall_clock_s": time.time() - t0}
    outp = os.path.join(REPDIR, f"m4_fielderror_seed{seed}_n{n_points}.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---------- report + gate ----------
    def _table(title, block):
        print(f"\n=== M4: JOINT {title} (n={n_points} seed-{seed} off-node points) ===")
        print(f"{'model':<22}{'min(floor)':>12}{'median':>12}{'mean':>12}{'p95':>12}{'max':>12}")
        for lab, key in [("value", "value"), ("value+gradient", "value+grad"),
                         ("value+grad+cross", "value+grad+cross")]:
            s = block[key]
            print(f"{lab:<22}" + "".join(f"{s[c]:>12.3e}"
                  for c in ("min", "median", "mean", "p95", "max")))

    fe = joint["field_error"]; cr = joint["constraint_residual"]
    _table("bare-guess FIELD ERROR ‖interp−u_true‖₂/‖u_true‖₂", fe)
    print(f"u_true residual max = {joint['u_true_res_max']:.2e} (target ≤ {u_tol})")
    vmed, gmed, cmed = fe["value"]["median"], fe["value+grad"]["median"], fe["value+grad+cross"]["median"]
    vflo, gflo, cflo = fe["value"]["min"], fe["value+grad"]["min"], fe["value+grad+cross"]["min"]
    print(f"FIELD-ERR floor(min):  value={vflo:.2e} grad={gflo:.2e} cross={cflo:.2e}"
          f"  (grad/value={gflo/vflo:.1f}x, cross/value={cflo/vflo:.2f}x)")
    print(f"FIELD-ERR median:      value={vmed:.2e} grad={gmed:.2e} cross={cmed:.2e}"
          f"  (grad/value={gmed/vmed:.1f}x, cross/value={cmed/vmed:.2f}x)")

    _table("bare-guess EQUILIBRATED constraint residual ‖R‖_equil", cr)
    frac = cr["frac_certified_le_1e-10"]
    rvmed, rgmed, rcmed = cr["value"]["median"], cr["value+grad"]["median"], cr["value+grad+cross"]["median"]
    print(f"RESID median:  value={rvmed:.2e} grad={rgmed:.2e} cross={rcmed:.2e}"
          f"  (cross/grad={rcmed/rgmed:.2f}x)   %certified≤1e-10 "
          f"[V,G,C]=[{100*frac['value']:.0f}%,{100*frac['value+grad']:.0f}%,"
          f"{100*frac['value+grad+cross']:.0f}%]")

    gate_floor = cflo <= 1.10 * vflo
    gate_med = cmed <= 1.10 * vmed
    resid_ok = rcmed <= 1.25 * rgmed          # sanity: residual not degraded by the cross
    print(f"\nM4 GATE — field-error floor≤value: {'PASS' if gate_floor else 'FAIL'}; "
          f"median≤value: {'PASS' if gate_med else 'FAIL'}; "
          f"gap grad/value={gmed/vmed:.1f}x closes (cross/value={cmed/vmed:.2f}x); "
          f"constraint-residual sanity (cross≲grad): {'PASS' if resid_ok else 'FAIL'}")
    print(f"[m4] wrote {outp}  ({out['wall_clock_s']/60:.1f} min)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", required=True, help="the with-cross model .npz "
                    "(value + gradient baselines are derived from it)")
    ap.add_argument("--grad", default=None, help="(optional, provenance only)")
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-peraxis", type=int, default=60)
    args = ap.parse_args()
    main(args.cross, n_points=args.n_points, seed=args.seed,
         n_peraxis=args.n_peraxis, grad_path=args.grad)
