"""PARASOL — QC 4-D per-axis + joint convergence STUDY on the WIDE separation box
b∈[2,7] (chi rebuild; matches the b∈[2,7] shipped model).

Add-only b∈[2,7] twin of run_qc_walls_sweep_chi.py.  It does NOT copy that
module's body; it imports it and (1) swaps the module BOX to b∈[2,7], (2)
redefines block_A (per-axis) with the b-axis studied over [2,7], and block_B
(merger wall) with b_max=7, then dispatches to its main().  block_C (spin wall,
fixed b) and block_D (joint 4-D Smolyak + cost model, over the patched BOX) are
reused verbatim; block_D shares solve_store_chi with the b∈[2,7] value model
build (build_surrogate_chi --box d4_qc_chi_b27), so their L<=4 nodes are shared.

Outputs go to reports/3D_parametric/qc_chi_b27/ (separate from the [1.5,4] study).

Held representative for the q / spin per-axis rates stays b=2.5 (interior to
[2,7], protocol-consistent with the [1.5,4] study for comparability); only the
b-AXIS rate is genuinely range-dependent and is now measured over [2,7].

Run: python sandbox/parasol/run_qc_walls_sweep_chi_b27.py
"""
from __future__ import annotations

import os
import sys
import time


import run_qc_walls_sweep_chi as w  # noqa: E402  (committed study driver, reused)
from lm.initial_data.parametric import parametric_nd_3d as p3  # noqa: E402

CHI_MAX = w.CHI_MAX
QC = w.QC
CHI_REP = w.CHI_REP
B_MAX = 7.0
B_MIN = 2.0

# (1) wide-separation box (block_D + holdout sample from this)
w.BOX = [{"name": "b", "min": B_MIN, "max": B_MAX},
         {"name": "q", "min": 1.0, "max": 3.0},
         {"name": "chi_Ay", "min": -CHI_MAX, "max": CHI_MAX},
         {"name": "chi_By", "min": -CHI_MAX, "max": CHI_MAX}]

# (2) separate output dir so the [1.5,4] study's files are not clobbered
w.REPDIR = os.path.join(w.HERE, "reports", "3D_parametric", "qc_chi_b27")
os.makedirs(w.REPDIR, exist_ok=True)


def block_A_b27(prob):
    """Per-axis held-out rates with the b-axis studied over [2,7]."""
    w._t("\n########## A: per-axis held-out rates (d4_qc chi b∈[2,7], QC) ##########")
    studies = [
        ("b",      "b",      B_MIN, B_MAX, [4, 8, 12, 16], dict(QC, q=2.0, chi_Ay=CHI_REP, chi_By=CHI_REP)),
        ("q",      "q",      1.0, 3.0,     [4, 8, 12, 16], dict(QC, b=2.5, chi_Ay=CHI_REP, chi_By=CHI_REP)),
        ("chi_Ay", "chi_Ay", -CHI_MAX, CHI_MAX, [4, 6, 8, 10], dict(QC, b=2.5, q=2.0, chi_By=CHI_REP)),
        ("chi_By", "chi_By", -CHI_MAX, CHI_MAX, [4, 6, 8, 10], dict(QC, b=2.5, q=2.0, chi_Ay=CHI_REP)),
    ]
    out = {}
    for (label, name, lo, hi, Qs, fixed) in studies:
        t0 = time.time()
        rows, _ = p3.held_out_convergence_1axis(prob, name, lo, hi, Qs, fixed=fixed)
        Q_arr = [r[0] for r in rows]; e_arr = [float(r[1]) for r in rows]
        rate = w._rate(Q_arr, e_arr)
        out[label] = dict(name=name, p_min=lo, p_max=hi,
                          fixed={k: v for k, v in fixed.items() if k != "qc"},
                          Qs=Q_arr, errs=e_arr, rate=rate)
        w._t(f"\n=== A: {label}  rate={rate:.3f} dec/Q  [{time.time()-t0:.0f}s] ===")
        for q, e in zip(Q_arr, e_arr):
            w._t(f"   Q={q:>3}  err={e:.3e}")
    return out


def block_B_b27(prob):
    """Merger (b) wall with b_max=7 (b_min swept below the box toward merger)."""
    w._t("\n########## B: merger (b) wall — QC (risk R3), b_max=7 ##########")
    b_max = B_MAX
    Qs = [4, 8, 12, 16]
    out = []
    for b_min in [2.0, 1.5, 1.2, 1.0]:
        t0 = time.time()
        rows, _ = p3.held_out_convergence_1axis(prob, "b", b_min, b_max, Qs,
                                                fixed=dict(QC, q=1.0, chi_Ay=0.0, chi_By=0.0))
        Q_arr = [r[0] for r in rows]; e_arr = [float(r[1]) for r in rows]
        rate = w._rate(Q_arr, e_arr)
        pred0 = p3.bernstein_rate_from_zero(b_min, b_max)
        p_star = p3.infer_real_singularity(b_min, b_max, rate, side="left")
        out.append(dict(b_min=b_min, b_max=b_max, Qs=Q_arr, errs=e_arr, rate=rate,
                        rate_pred_b0=pred0, inferred_sing=float(p_star)))
        w._t(f"\n=== B: b_min={b_min}  rate={rate:.3f}  (b=0 Bernstein {pred0:.3f})  "
             f"inferred nearest sing b*={p_star:.3f}  [{time.time()-t0:.0f}s] ===")
        for q, e in zip(Q_arr, e_arr):
            w._t(f"   Q={q:>3}  err={e:.3e}")
    return out


# (2) install the b∈[2,7] blocks (main() reads block_A/block_B as module globals)
w.block_A = block_A_b27
w.block_B = block_B_b27


if __name__ == "__main__":
    w.main()
