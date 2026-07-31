"""LM-initial-data — per-axis held-out convergence as a DISTRIBUTION over random base
points (the "reduced option (B)" companion to ``run_qc_peraxis_chi6.py``).

Paper Fig. 2 (``fig:peraxis``) shows the per-axis held-out interpolation
error vs. the number of parameter nodes Q for the eight quasi-circular axes, with
the seven NON-swept axes frozen at ONE representative interior base point
(b=2.5, q=2, every spin at chi=0.5).  Fig. 5 (``fig:joint``), by contrast, reports
the JOINT error as a distribution over 1000 random off-node points.  This driver
adds the missing distributional view of the per-axis study: it repeats the exact
per-axis 1-D convergence measurement of ``run_qc_peraxis_chi6.py`` for each of
``N_SAMPLES`` RANDOM base points (the other seven axes drawn uniformly in their
boxes), so each (axis, Q) yields a distribution of held-out errors rather than a
single number.  The plotter then draws best/median/worst bands (Fig.-5 style).

Reduced scope (per the paper request; ~weekend budget):
  * ``N_SAMPLES = 100`` random base points (shared 8-D draw; for each axis the
    OTHER seven coordinates are used, the swept coordinate is ignored);
  * ``Q_LADDER = [4, 6, 8, 10, 12]`` — the first five nodes of the chi6 ladder;
  * everything ELSE identical to ``run_qc_peraxis_chi6.py``: grid (44,32,8),
    code_tag "chi-rebuild", the 21 golden-ratio held-out fractions ALONG the swept
    axis, the max-over-held-out metric, value-only + gradient-enhanced (Hermite)
    interpolants sharing the same nodal solves, and the certified chi tangent
    (``tangent_qc_chi``, jac='nk', inner GMRES rtol=1e-8).

Work unit = one (axis, sample) pair (8 x N_SAMPLES of them).  Array mode stripes
the flat unit list across ``--ntasks`` tasks (``units[taskid::ntasks]``); each task
writes ``peraxis_dist_chi_parts/part_<taskid>.json``; ``--assemble`` merges them to
``reports/3D_parametric/qc_chi/peraxis_dist_chi.json`` (order stats + raw samples).

Add-only: no existing module or driver is modified.  Output ->
  reports/3D_parametric/qc_chi/peraxis_dist_chi.json
  reports/3D_parametric/qc_chi/peraxis_dist_chi_parts/part_<taskid>.json
  reports/3D_parametric/qc_chi/fig_qc_per_axis_dist_chi.png   (2x4 best/median/worst)

Smoke (1 axis, 2 samples, Q=[4,6]; times value-solve vs tangent for the ETA):
  python run_qc_peraxis_dist_chi.py --smoke

Full run (cluster job array; e.g. 40 tasks, <=20 concurrent):
  sbatch --array=0-39%20 \
    --export=ALL,DRIVER=run_qc_peraxis_dist_chi.py \
    slurm/ivs/submit_lm_initial_data_cpu_array_hi.slurm
  # then, after all tasks finish:
  python run_qc_peraxis_dist_chi.py --assemble
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.parametric import parametric_nd_3d as p3
from lm.initial_data.parametric.parametric import cheb_param_nodes
from lm.initial_data.parametric.hermite import cardinal_deriv_at_nodes
from lm.initial_data.parametric.hermite_nd import HermiteSolutionND
from lm.initial_data.applications import sensitivity_3d_qc as qc
from lm.initial_data.pipeline import production_box as pb

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
REPDIR = os.path.join(REPORTS, "3D_parametric", "qc_chi")
PARTDIR = os.path.join(REPDIR, "peraxis_dist_chi_parts")
os.makedirs(PARTDIR, exist_ok=True)

NA, NB, NPHI = 44, 32, 8
CODE_TAG = "chi-rebuild"
QC = dict(pb.FIXED_QC)
M_TOT = 1.0
CHI_MAX = pb.CHI_MAX
GMRES_RTOL = 1e-8

# production box edges (all from production_box)
B_LO, B_HI = pb.B_MIN, pb.B_MAX
Q_LO, Q_HI = pb.Q_MIN, pb.Q_MAX

STUDIES = [(a["name"], a["min"], a["max"]) for a in pb.spin8_box()]
AXES = [s[0] for s in STUDIES]
BOX = {n: (lo, hi) for n, lo, hi in STUDIES}

# ---- reduced ladder / sample count (the two knobs the paper request sets) ----
N_SAMPLES = 100
Q_LADDER = [4, 6, 8, 10, 12]
SEED = 20260724   # deterministic base-point draw (reproducible, array-consistent)

# dense held-out fractions ALONG the swept axis — 21 golden-ratio off-node points
# (identical to run_qc_peraxis_chi6.py / the paper's run_qc_dense_stats.py).
GOLDEN = 0.6180339887498949
FRACS_DENSE = np.array(sorted(0.02 + 0.96 * ((np.arange(1, 22) * GOLDEN) % 1.0)))

LBL = {"b": r"$b$", "q": r"$q$",
       "chi_Ax": r"$\chi_{Ax}$", "chi_Ay": r"$\chi_{Ay}$", "chi_Az": r"$\chi_{Az}$",
       "chi_Bx": r"$\chi_{Bx}$", "chi_By": r"$\chi_{By}$", "chi_Bz": r"$\chi_{Bz}$"}


def _t(m): print(m, flush=True)


def base_points():
    """The shared set of N_SAMPLES random 8-D base points (uniform in each box).
    Drawn once with a fixed seed so every array task agrees on sample s."""
    rng = np.random.default_rng(SEED)
    pts = []
    for _ in range(N_SAMPLES):
        pts.append({n: float(rng.uniform(lo, hi)) for n, lo, hi in STUDIES})
    return pts


def _tangent_fn(prob, name, fixed):
    """1-axis certified chi-parameterized tangent (jac='nk', relaxed inner GMRES).

    Calls ``certified_tangent_3d_qc`` DIRECTLY.  NB: run_qc_peraxis_chi6.py uses the
    ``qc_chi_tangent.tangent_qc_chi`` wrapper, which ADDS the mass->spin chain dS/dq on
    top of the base tangent.  That was correct when the base was buggy (pre-25d120e,
    2026-07-20), but commit 25d120e fixed dS/dq INSIDE ``certified_tangent_3d_qc`` — so
    the wrapper now DOUBLE-COUNTS on the q axis (confirmed: wrapper q-Hermite ~2.7e-2 vs
    base-direct ~2.6e-7, the committed peraxis_chi6.json value).  The fixed base is
    correct for every axis; the wrapper is obsolete."""
    def tf(theta_vec, U):
        sl = p3.theta_to_slice3d(theta_vec, [name], M_TOT, fixed)
        asm = s3.assemble(prob, sl)
        dU = qc.certified_tangent_3d_qc(prob, np.asarray(U), sl, name, M_TOT,
                                        asm=asm, jac="nk", gmres_rtol=GMRES_RTOL)
        return np.asarray(dU)[None, ...]
    return tf


def build_axis(name, lo, hi, Q, solve_fn, tf):
    """Solve the Q+1 CGL nodes once (warm-started); return (hermite, value_only)
    1-axis interpolants sharing the same nodal fields.  Identical machinery to
    run_qc_peraxis_chi6.py::build_axis."""
    nodes, weights = cheb_param_nodes(lo, hi, Q)
    U_nodes = dU_nodes = None
    guess = None
    for i, p in enumerate(nodes):
        U, _ = solve_fn(np.array([float(p)]), guess, 1e-12, 30)
        Ua = np.asarray(U)
        dU = np.asarray(tf(np.array([float(p)]), Ua))
        if U_nodes is None:
            fs = Ua.shape
            U_nodes = np.empty((nodes.size,) + fs)
            dU_nodes = np.empty((nodes.size, 1) + fs)
        U_nodes[i] = Ua; dU_nodes[i] = dU
        guess = Ua
    cvec = [cardinal_deriv_at_nodes(nodes)]
    common = dict(axes=[(lo, hi, Q)], nodes=[nodes], weights=[weights],
                  U_nodes=U_nodes, cvec=cvec,
                  iters=np.zeros(nodes.size, int), residuals=np.zeros(nodes.size))
    her = HermiteSolutionND(dU_nodes=dU_nodes, enhanced=(0,), **common)
    val = HermiteSolutionND(dU_nodes=np.zeros_like(dU_nodes), enhanced=(), **common)
    return her, val


def measure_unit(prob, name, lo, hi, base, Qs, fracs, timing=None):
    """The full per-axis 1-D held-out measurement at ONE base point.
    Returns {Q: (err_value, err_hermite)}; NaN on any solver failure (robust:
    one bad base point never kills the task).  ``timing`` (optional dict) collects
    per-value-solve and per-tangent wall times for the ETA estimate."""
    fixed = dict(QC, **{k: v for k, v in base.items() if k != name})
    solve_fn, _ = p3.make_solve_fn(prob, [name], M_tot=M_TOT, fixed=fixed,
                                   use_cache=True, solver="modified")
    tf = _tangent_fn(prob, name, fixed)
    hold = lo + fracs * (hi - lo)

    if timing is not None:
        # instrumented held-out reference solves (value solver)
        Ud = {}
        for p in hold:
            t0 = time.time()
            Ud[float(p)] = np.asarray(solve_fn(np.array([float(p)]), None, 1e-12, 30)[0])
            timing.setdefault("value_solve_s", []).append(time.time() - t0)
        tf_timed = tf
        def tf(theta_vec, U, _orig=tf_timed):   # noqa: F811 (wrap for timing)
            t0 = time.time()
            r = _orig(theta_vec, U)
            timing.setdefault("tangent_s", []).append(time.time() - t0)
            return r
    else:
        Ud = {float(p): np.asarray(solve_fn(np.array([float(p)]), None, 1e-12, 30)[0])
              for p in hold}

    out = {}
    for Q in Qs:
        her, val = build_axis(name, lo, hi, Q, solve_fn, tf)
        e_v = max(float(np.max(np.abs(val.evaluate([float(p)]) - Ud[float(p)]))) for p in hold)
        e_h = max(float(np.max(np.abs(her.evaluate([float(p)]) - Ud[float(p)]))) for p in hold)
        out[Q] = (e_v, e_h)
    return out


def work_units(only_axis=None):
    """Flat list of (axis_index, sample_index) — 8 x N_SAMPLES units, or just the
    N_SAMPLES units of ``only_axis`` (for a single-axis re-run)."""
    if only_axis is not None:
        ai = AXES.index(only_axis)
        return [(ai, s) for s in range(N_SAMPLES)]
    return [(ai, s) for ai in range(len(STUDIES)) for s in range(N_SAMPLES)]


def run_units(prob, units, pts, Qs, fracs, timing=None):
    """Run a set of (axis, sample) units; group results by axis name."""
    A = {}   # name -> {Q: [(e_v, e_h, sample_idx), ...]}
    for k, (ai, s) in enumerate(units):
        name, lo, hi = STUDIES[ai]
        t0 = time.time()
        try:
            res = measure_unit(prob, name, lo, hi, pts[s], Qs, fracs, timing=timing)
            row = {Q: (res[Q][0], res[Q][1], s) for Q in Qs}
            tag = f"v/h@Q{Qs[-1]}={res[Qs[-1]][0]:.2e}/{res[Qs[-1]][1]:.2e}"
        except Exception as e:   # robust: record NaN, keep going
            row = {Q: (float("nan"), float("nan"), s) for Q in Qs}
            tag = f"FAILED ({type(e).__name__}: {str(e)[:60]})"
        A.setdefault(name, {Q: [] for Q in Qs})
        for Q in Qs:
            A[name][Q].append(row[Q])
        _t(f"   [{k+1}/{len(units)}] {name} sample={s}  {tag}  [{time.time()-t0:.0f}s]")
    return A


def _stats(vals):
    """best/median/worst/p05/p95/n_ok over the finite samples of a list."""
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return dict(best=None, median=None, worst=None, p05=None, p95=None, n_ok=0)
    return dict(best=float(a.min()), median=float(np.median(a)),
                worst=float(a.max()), p05=float(np.percentile(a, 5)),
                p95=float(np.percentile(a, 95)), n_ok=int(a.size))


def _meta(extra=None):
    m = dict(Na=NA, Nb=NB, Nphi=NPHI, code_tag=CODE_TAG, axes=AXES,
             boxes={n: [lo, hi] for n, lo, hi in STUDIES}, chi_max=CHI_MAX,
             Q_ladder=Q_LADDER, gmres_rtol=GMRES_RTOL,
             n_holdout=len(FRACS_DENSE), n_samples=N_SAMPLES, seed=SEED,
             base_point="RANDOM (uniform in box; other 7 axes per swept axis)")
    if extra:
        m.update(extra)
    return m


def _merge_parts(paths):
    """name -> {Q(str): [(e_v,e_h,s), ...]}  merged over a list of part_*.json."""
    merged = {}
    for pf in paths:
        with open(pf) as f:
            A = json.load(f)["A"]
        for name, byQ in A.items():
            m = merged.setdefault(name, {})
            for Qs_, rows in byQ.items():
                m.setdefault(Qs_, []).extend(rows)
    return merged


def assemble():
    """Merge per-task partials -> peraxis_dist_chi.json (order stats + raw) + figure.

    Top-level PARTDIR/part_*.json is the main (all-axes) run; PARTDIR/<axis>/part_*.json
    (a single-axis ``--only-axis`` re-run) OVERRIDES that axis — used to replace the
    q axis with the fixed-tangent re-run without redoing b/spins."""
    parts = sorted(glob.glob(os.path.join(PARTDIR, "part_*.json")))
    if not parts:
        _t(f"[assemble] no partials in {PARTDIR}"); return
    merged = _merge_parts(parts)
    for name in AXES:
        sub = sorted(glob.glob(os.path.join(PARTDIR, name, "part_*.json")))
        if sub:
            merged[name] = _merge_parts(sub)[name]
            _t(f"[assemble] axis {name}: OVERRIDDEN from {len(sub)} re-run partial(s)")
    A_out = {}
    for name in AXES:
        if name not in merged:
            _t(f"[assemble] WARNING: axis {name} absent from partials"); continue
        Qs = sorted(int(q) for q in merged[name])
        val_stats = [_stats([r[0] for r in merged[name][str(Q)]]) for Q in Qs]
        her_stats = [_stats([r[1] for r in merged[name][str(Q)]]) for Q in Qs]
        A_out[name] = dict(
            name=name, Qs=Qs, n_samples=N_SAMPLES,
            value={k: [s[k] for s in val_stats] for k in val_stats[0]},
            hermite={k: [s[k] for s in her_stats] for k in her_stats[0]},
            raw={str(Q): merged[name][str(Q)] for Q in Qs},   # (e_v,e_h,sample_idx)
        )
    results = {"meta": _meta(dict(assembled_from=len(parts),
                                  axes_present=list(A_out),
                                  base_points=base_points())),
               "A_per_axis": A_out}
    out = os.path.join(REPDIR, "peraxis_dist_chi.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _t(f"[assemble] {len(parts)} partials -> {out}  ({len(A_out)}/{len(AXES)} axes)")
    make_figure(results)


def make_figure(results):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        _t(f"[fig] matplotlib unavailable ({e})"); return
    A = results["A_per_axis"]
    names = [n for n in AXES if n in A]
    YLIM = (1e-12, 1e-2)
    ncol = 4; nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
    for k, name in enumerate(names):
        ax = axs[k // ncol][k % ncol]
        d = A[name]; Qs = d["Qs"]
        for key, color, mk, lab in (("value", "#888", "o--", "value"),
                                    ("hermite", "#c0392b", "s-", "value+gradient")):
            st = d[key]
            med = np.array([np.nan if v is None else v for v in st["median"]], float)
            lo = np.array([np.nan if v is None else v for v in st["best"]], float)
            hi = np.array([np.nan if v is None else v for v in st["worst"]], float)
            ax.fill_between(Qs, lo, hi, color=color, alpha=0.2)
            ax.semilogy(Qs, med, mk, color=color, label=f"{lab} (median)")
        ax.set_title(LBL.get(name, name)); ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(*YLIM); ax.legend(fontsize=8); ax.set_xlabel("parameter nodes  $Q$")
    for k in range(len(names), nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    for r in range(nrow):
        axs[r][0].set_ylabel("held-out error")
    fig.tight_layout()
    p = os.path.join(REPDIR, "fig_qc_per_axis_dist_chi.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    _t(f"[fig] wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--taskid", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    ap.add_argument("--ntasks", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--only-axis", default=None,
                    help="re-run just this axis; partials go to PARTDIR/<axis>/ and "
                         "OVERRIDE that axis at --assemble (e.g. the q fixed-tangent re-run)")
    args = ap.parse_args()

    if args.assemble:
        assemble(); return

    t0 = time.time()
    pts = base_points()
    prob = s3.make_problem(Na=NA, Nb=NB, Nphi=NPHI)

    if args.smoke:
        # 1 axis (chi_Ay), 2 samples, Q=[4,6]; time value-solve vs tangent -> ETA.
        ai = AXES.index("chi_Ay")
        units = [(ai, 0), (ai, 1)]
        Qs = [4, 6]; fracs = FRACS_DENSE[:5]
        timing = {}
        _t(f"=== SMOKE: axis=chi_Ay  samples=2  Q={Qs}  holdout={len(fracs)} ===")
        run_units(prob, units, pts, Qs, fracs, timing=timing)
        v = np.array(timing.get("value_solve_s", [np.nan]))
        g = np.array(timing.get("tangent_s", [np.nan]))
        _t(f"\n[SMOKE timing] value_solve: {v.mean():.2f}s/solve (n={v.size}); "
           f"tangent: {g.mean():.2f}s/tangent (n={g.size})")
        # extrapolate the FULL run: per (axis,sample): 21 ref + sum(Q+1) node value
        # solves + sum(Q+1) tangents, over Q_LADDER.
        n_node = sum(Q + 1 for Q in Q_LADDER)
        val_per_unit = len(FRACS_DENSE) + n_node
        tan_per_unit = n_node
        units_total = len(STUDIES) * N_SAMPLES
        sec = units_total * (val_per_unit * v.mean() + tan_per_unit * g.mean())
        _t(f"[SMOKE ETA] full run = {units_total} units x "
           f"({val_per_unit} value-solves + {tan_per_unit} tangents)")
        _t(f"[SMOKE ETA] serial wall ~ {sec/3600:.1f} h  "
           f"(= {sec/3600/40:.2f} h/task on a 40-task array; "
           f"{sec/3600/80:.2f} h/task on 80).")
        _t(f"\n[SMOKE] ok in {time.time()-t0:.0f}s — safe to launch full.")
        return

    array_mode = args.taskid >= 0 and args.ntasks > 1
    partdir = PARTDIR if not args.only_axis else os.path.join(PARTDIR, args.only_axis)
    os.makedirs(partdir, exist_ok=True)
    units = work_units(only_axis=args.only_axis)
    if array_mode:
        units = units[args.taskid::args.ntasks]
    Qs = Q_LADDER; fracs = FRACS_DENSE
    _t(f"=== per-axis DISTRIBUTION (value+Hermite), 8-D chi axes "
       f"{'[only %s] ' % args.only_axis if args.only_axis else ''}"
       f"{'(array %d/%d)' % (args.taskid, args.ntasks) if array_mode else ''} ===")
    _t(f"    units={len(units)}  Q={Qs}  n_samples={N_SAMPLES}  holdout={len(fracs)}")
    A = run_units(prob, units, pts, Qs, fracs)

    if array_mode:
        part = os.path.join(partdir, f"part_{args.taskid}.json")
        with open(part, "w") as f:
            json.dump({"A": A}, f, indent=2)
        _t(f"\n[array {args.taskid}/{args.ntasks}] wrote {part}  [{time.time()-t0:.0f}s]")
        return
    # single-process run: write one part then assemble (unless a single-axis re-run,
    # which only writes its override partial — assemble separately after).
    part = os.path.join(partdir, "part_0.json")
    with open(part, "w") as f:
        json.dump({"A": A}, f, indent=2)
    _t(f"\nWrote {part}  [{time.time()-t0:.0f}s]")
    if not args.only_axis:
        assemble()


if __name__ == "__main__":
    main()
