"""LM-initial-data §VII (recast) — QC 4-D per-axis convergence + walls, DIMENSIONLESS-SPIN
parameterization (chi = S/m^2 axes).

Same machinery, same grid, same protocol as the bare-spin study it replaced; the
spin axes are the Kerr-like dimensionless spins chi_Ay, chi_By (via the chi_*
axes in ``parametric_nd_3d``), carried on the production box of
``production_box`` (chi in [-CHI_MAX, CHI_MAX]).

Why chi: at fixed q it is an affine relabel of S (identical per-axis rate), but
the target region (bounded dimensionless spin) is a rectangle in (q, chi) and a
q-dependent fan in (q, S), so the (q, chi) tensor/Smolyak grid adapts node density
to each hole's mass instead of over-covering the small hole toward extremal chi.
A CONSISTENT representative interior point chi_other = CHI_REP is used for every
one-axis sweep, so the whole per-axis table is internally consistent.

Blocks (-> reports/3D_parametric/qc_chi/walls_d4_qc_chi.json + 4 figures):
  A. per-axis held-out geometric rates -- b, q, chi_Ay, chi_By.
  B. MERGER (b) wall -- rate at several b_min + b=0 Bernstein prediction (spins 0).
  C. SPIN wall -- rate at several chi_max + inferred nearest real singularity,
     swept DIRECTLY in chi.  Reported in TWO parts (see :func:`block_C`):
     ``C_wall_spin_inside`` over ranges inside the production box (the rates that
     govern the shipped model) and ``C_wall_spin`` over super-extremal ranges
     (the only place the wall CHARACTER can be established).
  Q. MASS-RATIO wall -- rate at several q_max + inferred nearest real q*
     (``Q_wall_q``); decides whether q_max = Q_MAX is near a hard limit.
  D. joint 4-D convergence (Smolyak) + cost model over the chi box.

    caffeinate -i python -m lm.initial_data.pipeline.run_qc_walls_sweep_chi
"""
from __future__ import annotations
import json, os, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3
from lm.initial_data.parametric import solve_store as ss
from lm.initial_data.pipeline import production_box as pb

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "3D_parametric", "qc_chi")
STORE = os.path.join(HERE, "reports", "3D_parametric", "solve_store_chi")
os.makedirs(REPDIR, exist_ok=True)

NA, NB, NPHI = 44, 32, 8
CODE_TAG = "chi-rebuild"
QC = dict(pb.FIXED_QC)
CHI_MAX = pb.CHI_MAX
CHI_REP = pb.CHI_REP   # representative interior dimensionless spin for the held axes
BOX = pb.aligned_box(b_min=1.5, b_max=pb.B_MAX_NARROW)


def _t(m): print(m, flush=True)


def _rate(Qs, errs, floor=1e-9, ceil=1.0):
    Qs = np.asarray(Qs, float); errs = np.asarray(errs, float)
    m = (errs > floor) & (errs < ceil) & np.isfinite(errs)
    if m.sum() < 2:
        m = (errs > 0) & np.isfinite(errs)
    return -float(np.polyfit(Qs[m], np.log10(errs[m]), 1)[0])


# ==========================================================================
# A — per-axis held-out rates (consistent chi_rep hold throughout)
# ==========================================================================
def block_A(prob):
    _t("\n########## A: per-axis held-out rates (d4_qc chi, QC momenta) ##########")
    studies = [
        ("b",      "b",      1.5, 4.0,  [4, 8, 12, 16], dict(QC, q=2.0, chi_Ay=CHI_REP, chi_By=CHI_REP)),
        ("q",      "q",      1.0, 3.0,  [4, 8, 12, 16], dict(QC, b=pb.B_REP, chi_Ay=CHI_REP, chi_By=CHI_REP)),
        ("chi_Ay", "chi_Ay", -CHI_MAX, CHI_MAX, [4, 6, 8, 10], dict(QC, b=pb.B_REP, q=2.0, chi_By=CHI_REP)),
        ("chi_By", "chi_By", -CHI_MAX, CHI_MAX, [4, 6, 8, 10], dict(QC, b=pb.B_REP, q=2.0, chi_Ay=CHI_REP)),
    ]
    out = {}
    for (label, name, lo, hi, Qs, fixed) in studies:
        t0 = time.time()
        rows, _ = p3.held_out_convergence_1axis(prob, name, lo, hi, Qs, fixed=fixed)
        Q_arr = [r[0] for r in rows]; e_arr = [float(r[1]) for r in rows]
        rate = _rate(Q_arr, e_arr)
        out[label] = dict(name=name, p_min=lo, p_max=hi,
                          fixed={k: v for k, v in fixed.items() if k != "qc"},
                          Qs=Q_arr, errs=e_arr, rate=rate)
        _t(f"\n=== A: {label}  rate={rate:.3f} dec/Q  [{time.time()-t0:.0f}s] ===")
        for q, e in zip(Q_arr, e_arr):
            _t(f"   Q={q:>3}  err={e:.3e}")
    return out


# ==========================================================================
# B — merger (b) wall (spins zero: chi==S label-irrelevant)
# ==========================================================================
def block_B(prob):
    _t("\n########## B: merger (b) wall — QC (risk R3) ##########")
    b_max = 4.0
    Qs = [4, 8, 12, 16]
    out = []
    for b_min in [2.0, 1.5, 1.2, 1.0]:
        t0 = time.time()
        rows, _ = p3.held_out_convergence_1axis(prob, "b", b_min, b_max, Qs,
                                                fixed=dict(QC, q=1.0, chi_Ay=0.0, chi_By=0.0))
        Q_arr = [r[0] for r in rows]; e_arr = [float(r[1]) for r in rows]
        rate = _rate(Q_arr, e_arr)
        pred0 = p3.bernstein_rate_from_zero(b_min, b_max)
        p_star = p3.infer_real_singularity(b_min, b_max, rate, side="left")
        out.append(dict(b_min=b_min, b_max=b_max, Qs=Q_arr, errs=e_arr, rate=rate,
                        rate_pred_b0=pred0, inferred_sing=float(p_star)))
        _t(f"\n=== B: b_min={b_min}  rate={rate:.3f}  (b=0 Bernstein {pred0:.3f})  "
           f"inferred nearest sing b*={p_star:.3f}  [{time.time()-t0:.0f}s] ===")
        for q, e in zip(Q_arr, e_arr):
            _t(f"   Q={q:>3}  err={e:.3e}")
    return out


# ==========================================================================
# C — spin wall, swept directly in chi, INSIDE and OUTSIDE the production box
# ==========================================================================
# The sweep interval is ONE-SIDED, [0, chi_max]; the mapped Bernstein semi-major
# axis is a = 2*chi*/chi_max - 1, and rate = log10(a + sqrt(a^2-1)).
#
# Inside-box Qs start lower and are sampled denser: the rate there is faster (a
# fixed distant singularity predicts ~0.9-1.3 dec/Q, while linear extrapolation of
# the measured super-extremal rates predicts only ~0.5 -- a factor-2 disagreement
# that is itself one reason to measure it), so the clean geometric window closes
# earlier against the ~1e-12 spatial floor.  [3,4,5,6,8,10] leaves >=4 usable
# points at either extreme; _rate() masks the floored tail.
QS_SPIN_INSIDE = [3, 4, 5, 6, 8, 10]
QS_SPIN_OUTSIDE = [4, 6, 8, 10]


def _spin_wall_sweep(prob, chi_ranges, Qs, tag):
    """Held-out rate + inferred nearest real chi* over one set of [0,chi_max] ranges."""
    out = []
    for chi_max in chi_ranges:
        t0 = time.time()
        rows, _ = p3.held_out_convergence_1axis(prob, "chi_Ay", 0.0, chi_max, Qs,
                                                fixed=dict(QC, b=pb.B_REP, q=1.0, chi_By=0.0))
        Q_arr = [r[0] for r in rows]; e_arr = [float(r[1]) for r in rows]
        rate = _rate(Q_arr, e_arr)
        chi_star = p3.infer_real_singularity(0.0, chi_max, rate, side="right")
        out.append(dict(chi_max=float(chi_max), Qs=Q_arr, errs=e_arr, rate=rate,
                        chi_star=float(chi_star)))
        _t(f"\n=== C[{tag}]: chi_Ay in [0,{chi_max:g}]  rate={rate:.3f}  "
           f"nearest sing chi*={chi_star:.2f}  [{time.time()-t0:.0f}s] ===")
        for q, e in zip(Q_arr, e_arr):
            _t(f"   Q={q:>3}  err={e:.3e}")
    return out


QS_Q_WALL = [4, 6, 8, 12, 16]


def block_Q(prob):
    """MASS-RATIO wall, swept in q over [Q_MIN, q_max] (one-sided, like block_C).

    The q axis had no wall block: block_A measures a single rate on the production
    range and block_B/block_C cover b and chi.  It needs one, because inverting the
    measured q rate puts the nearest inferred singularity at q* ~ 4.1 -- only just
    outside the box, far closer in relative terms than the spin wall.  Whether that
    is a real branch point (chi* pinned as the range grows => q_max = Q_MAX is near a
    hard limit) or a complex pair (marching => q_max could be raised to cover the
    asymmetric mass ratios) is exactly the pinned-vs-marching test block_C applies to
    the spins, and it decides whether the box can ever be widened in q.
    """
    _t("\n########## Q: mass-ratio (q) wall ##########")
    out = []
    for q_max in pb.WALL_Q_MAX:
        t0 = time.time()
        rows, _ = p3.held_out_convergence_1axis(prob, "q", pb.Q_MIN, q_max, QS_Q_WALL,
                                                fixed=dict(QC, b=pb.B_REP, chi_Ay=0.0, chi_By=0.0))
        Q_arr = [r[0] for r in rows]; e_arr = [float(r[1]) for r in rows]
        rate = _rate(Q_arr, e_arr)
        q_star = p3.infer_real_singularity(pb.Q_MIN, q_max, rate, side="right")
        out.append(dict(q_max=float(q_max), Qs=Q_arr, errs=e_arr, rate=rate,
                        q_star=float(q_star)))
        _t(f"\n=== Q: q in [{pb.Q_MIN:g},{q_max:g}]  rate={rate:.3f}  "
           f"nearest sing q*={q_star:.2f}  [{time.time()-t0:.0f}s] ===")
        for q, e in zip(Q_arr, e_arr):
            _t(f"   Q={q:>3}  err={e:.3e}")
    return out


def block_C(prob):
    """Spin wall in two parts.

    ``inside``  -- ranges within the production box (``production_box.wall_chi_inside``).
                   These are the rates that actually govern the shipped model, and
                   they are what a statement about the model's own spin range may be
                   based on.
    ``outside`` -- super-extremal ranges (``production_box.WALL_CHI_OUTSIDE``).
                   Required for the wall CHARACTER: the pinned-vs-marching test only
                   discriminates once the interval approaches the singularity, so a
                   fixed complex pair would move chi* by only ~2% across the inside
                   ranges but ~40% across these.  Not a validity claim.
    """
    _t("\n########## C: spin (chi) wall — inside the production box ##########")
    inside = _spin_wall_sweep(prob, pb.wall_chi_inside(), QS_SPIN_INSIDE, "inside")
    _t("\n########## C: spin (chi) wall — super-extremal (character) ##########")
    outside = _spin_wall_sweep(prob, pb.WALL_CHI_OUTSIDE, QS_SPIN_OUTSIDE, "outside")
    return {"inside": inside, "outside": outside}


# ==========================================================================
# D — joint 4-D Smolyak convergence + cost model (chi box)
# ==========================================================================
def block_D(prob, blockA):
    _t("\n########## D: joint 4-D Smolyak convergence + cost model (chi) ##########")
    store = ss.SolveStore(STORE, grid_meta=(NA, NB, NPHI), code_tag=CODE_TAG, reuse_tol=1e-6)
    _t(f"   store: {store.n_entries} entries, code_tag={store.code_tag}")
    hold = p3.holdout_points_nd(BOX, n_points=6)
    p3.assert_off_node(hold, [dict(a, Q=8) for a in BOX])
    solve_fn, _ = p3.make_solve_fn(prob, [a["name"] for a in BOX], fixed=QC, solver="modified")
    U_dir = [np.asarray(solve_fn(th, None, 1e-12, 30)[0]) for th in hold]
    joint = []
    for L in (1, 2, 3, 4):
        store.n_hits = store.n_misses = 0
        t0 = time.time()
        smsolver = ss.from_problem_smolyak_3d_cached(prob, BOX, store=store, fixed=QC,
                                                     solver="modified")
        sm = smsolver.build_isotropic(L, tol=1e-12, max_iter=30)
        err = max(float(np.max(np.abs(sm.evaluate(th) - U_dir[i]))) for i, th in enumerate(hold))
        joint.append(dict(level=L, nodes=int(sm.n_solver_nodes), err=err,
                          store_hits=int(store.n_hits), store_misses=int(store.n_misses)))
        _t(f"   L={L}: {sm.n_solver_nodes} nodes  held-out err={err:.3e}  "
           f"(store {store.n_hits} hits / {store.n_misses} misses)  [{time.time()-t0:.0f}s]")
    models = {}
    for label in ("b", "q", "chi_Ay", "chi_By"):
        rows = list(zip(blockA[label]["Qs"], blockA[label]["errs"]))
        rate, logC = p3.fit_error_model(rows, q_lo=4)
        models[label] = (rate, logC)
    cost = {}
    for eps in (1e-6, 1e-9):
        tbl, Qa, Qi = p3.cost_table(models, eps=eps)
        d4 = [r for r in tbl if r["d"] == 4][0]
        cost[f"eps_{eps:.0e}"] = dict(Q_aniso=Qa, Q_iso=Qi,
                                      n_aniso_d4=d4["n_aniso"], n_iso_d4=d4["n_iso"])
        _t(f"   cost eps={eps:.0e}: Q_aniso={Qa} Q_iso={Qi}  "
           f"tensor n(d=4) aniso={d4['n_aniso']} iso={d4['n_iso']}")
    return dict(smolyak=joint, cost_model={k: list(v) for k, v in models.items()},
                cost=cost)


def make_figures(results):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        _t(f"[fig] matplotlib unavailable ({e})"); return
    A = results["A_per_axis"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for label in ("b", "q", "chi_Ay", "chi_By"):
        d = A[label]
        ax.semilogy(d["Qs"], d["errs"], "o-", label=f"{label} ({d['rate']:.2f} dec/Q)")
    ax.set_xlabel("per-axis order Q"); ax.set_ylabel("held-out error")
    ax.set_title(r"QC $d4_{qc}$ ($\chi$) — per-axis convergence"); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(REPDIR, "fig_qc_per_axis_chi.png"), dpi=140); plt.close(fig)

    B = results["B_wall_b"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for w in B:
        ax1.semilogy(w["Qs"], w["errs"], "o-", label=f"b_min={w['b_min']}")
    ax1.set_xlabel("Q_b"); ax1.set_ylabel("held-out error")
    ax1.set_title("B: QC merger (b) wall"); ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=8)
    bm = [w["b_min"] for w in B]
    ax2.plot(bm, [w["rate"] for w in B], "o-", label="measured")
    ax2.plot(bm, [w["rate_pred_b0"] for w in B], "s--", label="b=0 Bernstein")
    ax2.set_xlabel("b_min"); ax2.set_ylabel("rate (dec/Q)")
    ax2.set_title("rate vs b=0 prediction (R3)"); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(REPDIR, "fig_qc_wall_b_chi.png"), dpi=140); plt.close(fig)

    C_in = results.get("C_wall_spin_inside", [])
    C_out = results["C_wall_spin"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for w in C_in:
        ax1.semilogy(w["Qs"], w["errs"], "o--", label=f"in-box chi_max={w['chi_max']:g}")
    for w in C_out:
        ax1.semilogy(w["Qs"], w["errs"], "o-", label=f"chi_max={w['chi_max']:g}")
    ax1.set_xlabel("Q_chi"); ax1.set_ylabel("held-out error")
    ax1.set_title(r"C: QC spin ($\chi$) wall"); ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=8)
    for lbl, C, st in (("in-box", C_in, "s--"), ("super-extremal", C_out, "o-")):
        if C:
            ax2.plot([w["chi_max"] for w in C], [w["rate"] for w in C], st, label=lbl)
    ax2.axvline(CHI_MAX, color="k", lw=0.8, ls=":", label=r"box edge $\chi_{\max}$")
    ax2.set_xlabel(r"$\chi$ range max"); ax2.set_ylabel("rate (dec/Q)")
    ax2.set_title("spin-axis rate vs range"); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(REPDIR, "fig_qc_spin_wall_chi.png"), dpi=140); plt.close(fig)

    D = results["D_joint_cost"]["smolyak"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.semilogy([r["nodes"] for r in D], [r["err"] for r in D], "o-", label="Smolyak sparse")
    ax.set_xlabel("solver node count"); ax.set_ylabel("joint held-out error")
    ax.set_title(r"D: QC 4-D joint convergence ($\chi$, Smolyak)"); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(REPDIR, "fig_qc_joint_chi.png"), dpi=140); plt.close(fig)
    _t(f"[fig] wrote 4 figures to {REPDIR}")


def main():
    t0 = time.time()
    prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)
    results = {"meta": dict(Na=NA, Nb=NB, Nphi=NPHI, box=[[a["min"], a["max"]] for a in BOX],
                            axes=[a["name"] for a in BOX], chi_rep=CHI_REP, code_tag=CODE_TAG)}
    results["A_per_axis"] = block_A(prob)
    results["B_wall_b"] = block_B(prob)
    C = block_C(prob)
    # "C_wall_spin" keeps its historical shape (the list of super-extremal ranges)
    # so downstream readers -- paper/figures/fig02_walls_data.py -- are unaffected;
    # the inside-box sweep lands under a new key.
    results["C_wall_spin"] = C["outside"]
    results["C_wall_spin_inside"] = C["inside"]
    results["Q_wall_q"] = block_Q(prob)
    results["D_joint_cost"] = block_D(prob, results["A_per_axis"])
    results["meta"]["wall_s"] = time.time() - t0
    out = os.path.join(REPDIR, "walls_d4_qc_chi.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _t(f"\nWrote {out}")
    make_figures(results)
    _t(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
