"""LM-initial-data — QC 4-D per-axis + joint convergence STUDY on the PRODUCTION
separation box of ``production_box`` (chi rebuild; matches the shipped model).

The production (wide-separation) twin of run_qc_walls_sweep_chi.py.  It does NOT
copy that module's body; it imports it and (1) swaps the module BOX to the
production box, (2) redefines block_A (per-axis) with the b-axis studied over
[B_MIN, B_MAX], and block_B (merger wall) with b_max=B_MAX, then dispatches to its
main().  block_C (spin wall, fixed b -- BOTH the inside-box and super-extremal
range sets) and block_D (joint 4-D Smolyak + cost model, over the patched BOX) are
reused verbatim; block_D shares solve_store_chi with the production value-model
build (build_surrogate_chi --box d4_qc_chi_prod), so their L<=4 nodes are shared.

Outputs go to reports/3D_parametric/qc_chi_prod/ (separate from the narrow-b study).

Held representative for the q / spin per-axis rates stays b=pb.B_REP (interior to the
box, protocol-consistent with the narrow-b study for comparability); only the
b-AXIS rate is genuinely range-dependent and is measured over the full box.

Run: caffeinate -i python -m lm.initial_data.pipeline.run_qc_walls_sweep_chi_prod
"""
from __future__ import annotations

import os
import sys
import time


from lm.initial_data.pipeline import run_qc_walls_sweep_chi as w  # noqa: E402
from lm.initial_data.parametric import parametric_nd_3d as p3  # noqa: E402
from lm.initial_data.pipeline import production_box as pb  # noqa: E402

CHI_MAX = pb.CHI_MAX
QC = w.QC
CHI_REP = pb.CHI_REP
B_MAX = pb.B_MAX
B_MIN = pb.B_MIN

# (1) wide-separation box (block_D + holdout sample from this)
w.BOX = pb.aligned_box()

# (2) separate output dir so the [1.5,4] study's files are not clobbered
w.REPDIR = os.path.join(w.HERE, "reports", "3D_parametric", "qc_chi_prod")
os.makedirs(w.REPDIR, exist_ok=True)


def block_A_prod(prob):
    """Per-axis held-out rates with the b-axis studied over the full production box."""
    w._t(f"\n########## A: per-axis held-out rates (d4_qc chi b∈[{B_MIN:g},{B_MAX:g}], QC) ##########")
    studies = [
        ("b",      "b",      B_MIN, B_MAX, [4, 8, 12, 16], dict(QC, q=2.0, chi_Ay=CHI_REP, chi_By=CHI_REP)),
        ("q",      "q",      pb.Q_MIN, pb.Q_MAX, [4, 8, 12, 16], dict(QC, b=pb.B_REP, chi_Ay=CHI_REP, chi_By=CHI_REP)),
        ("chi_Ay", "chi_Ay", -CHI_MAX, CHI_MAX, [4, 6, 8, 10], dict(QC, b=pb.B_REP, q=2.0, chi_By=CHI_REP)),
        ("chi_By", "chi_By", -CHI_MAX, CHI_MAX, [4, 6, 8, 10], dict(QC, b=pb.B_REP, q=2.0, chi_Ay=CHI_REP)),
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


def block_B_prod(prob):
    """Merger (b) wall at b_max=B_MAX (b_min swept below the box toward merger)."""
    w._t(f"\n########## B: merger (b) wall — QC (risk R3), b_max={B_MAX:g} ##########")
    b_max = B_MAX
    Qs = [4, 8, 12, 16]
    out = []
    for b_min in pb.WALL_B_MIN_SWEEP:
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


# (2) install the production-box blocks (main() reads block_A/block_B as module globals)
w.block_A = block_A_prod
w.block_B = block_B_prod


if __name__ == "__main__":
    w.main()
