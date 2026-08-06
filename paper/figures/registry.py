"""LM-initial-data paper — figure/data registry (single source of truth).

Every paper figure ``figNN_<name>`` has:
  * a PLOTTER   ``figNN_<name>_plot.py``       — reads ONLY ``figdata/figNN_<name>.json`` and draws
                                            (no ``reports/``, no ``jax``).
  * a DATA SCRIPT ``figNN_<name>_data.py`` — distills the arrays the figure plots out of one or
                                            more raw SOURCE artifacts into that committed json.

This file declares two maps:

  SOURCES   canonical source key -> where the raw run output lives under ``reports/``, the
            command that produces it, whether that command runs on the laptop or the cluster,
            and which figures consume it.  This is the DEDUP graph: a source shared by several
            figures is listed once and produced once.

  FIGURES   figure stem -> the source keys it needs + its data-script filename.  The committed
            output is always ``figdata/<stem>.json``.

The raw SOURCE artifacts are NOT committed (``reports/`` and ``*.npz`` are gitignored, and the
spectral corpora are multi-GB).  The small distilled ``figdata/*.json`` ARE committed, so every
figure rebuilds from the repo alone with no solves, no models, and no jax.  ``make_figdata.py``
uses this registry for the presence check, the dedup, and the cluster-command hints.
"""
from __future__ import annotations

# --- raw run outputs (under reports/); NOT committed --------------------------
# where: "laptop"  -> the distill step only reshapes json already on disk (no solves)
#        "cluster" -> the source is produced by a heavy CPU run on IVS (see the cluster prompt)
# status: "ready"   -> present on the laptop today
#         "pending" -> an 8D artifact still to be produced on the cluster
SOURCES = {
    # ---- fig01 (per-axis Hermite, DISTRIBUTION over random base points) ----
    "peraxis_dist_chi":     dict(reports="3D_parametric/qc_chi/peraxis_dist_chi.json",
                                 producer="run_qc_peraxis_dist_chi.py --assemble", where="cluster",
                                 status="ready", figures=["fig01_peraxis_hermite"]),

    # ---- fig02 (all three analyticity walls, merged) ----
    # Was 3D_parametric/qc/walls_d4_qc_dense.json, which no producer in this package
    # writes: it is the monorepo's run_qc_dense_stats.py, on the superseded narrow box
    # b in [1.5,4] in the OLD bare-spin (S_Ay,S_By) parameterization, and it carries no
    # mass-ratio block.  The production producer is the one named here; "dense" now
    # refers to its wall blocks' 21-point held-out sets and 5-level Q ladders.
    "walls_dense":          dict(reports="3D_parametric/qc_chi_prod/walls_d4_qc_chi.json",
                                 producer="run_qc_walls_sweep_chi_prod.py", where="cluster",
                                 status="ready",
                                 figures=["fig02_walls"]),

    # ---- fig03 (joint held-out distribution) ----
    "joint_dist_4d":        dict(reports="3D_parametric/qc_chi/joint_dist_d4_qc_chi_prod.json",
                                 producer="run_qc_joint_dist_chi.py --box d4_qc_chi_prod", where="cluster",
                                 status="ready", figures=["fig03_joint_dist"]),
    "joint_dist_cross_4d":  dict(reports="3D_parametric/qc_chi/joint_dist_cross_d4_qc_chi_prod.json",
                                 producer="run_qc_joint_dist_cross_chi.py", where="cluster",
                                 status="ready", figures=["fig03_joint_dist"]),
    "joint_dist_8d":        dict(reports="3D_parametric/qc_chi/joint_dist_spin8_qc_chi_prod.json",
                                 producer="run_qc_joint_dist_chi.py --box spin8_qc_chi_prod  (appendix b)",
                                 where="cluster", status="ready", figures=["fig03_joint_dist"]),
    "joint_dist_hermite_8d": dict(reports="3D_parametric/qc_chi/joint_dist_hermite_spin8_qc_chi_prod.json",
                                 producer="run_qc_joint_dist_hermite_8d.py  (appendix c)",
                                 where="cluster", status="ready", figures=["fig03_joint_dist"]),

    # ---- fig04 (certified refinement staircase) ----
    "polish_cold_4d":       dict(reports="P3/polish_cold_chi4d_1000.json",
                                 producer="run_polish_cold.py --dim 4", where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_cold_8d":       dict(reports="P3/polish_cold_chi8d_1000.json",
                                 producer="run_polish_cold.py --dim 8", where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_pod_4d":        dict(reports="P3/polish_table_chi4d_pod_r75_cross_1000.json",
                                 producer="run_polish_podrank.py --dim 4 --rank 75 (cross)", where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_pod_8d":        dict(reports="P3/polish_table_chi8d_pod_r250_1000.json",
                                 producer="run_polish_podrank.py --dim 8 --rank 250", where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_fielderr_4d":   dict(reports="P3/polish_fielderr_chi4d_1000.json",
                                 producer="run_polish_fielderr.py", where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_fielderr_8d":   dict(reports="P3/polish_fielderr_chi8d_1000.json",
                                 producer="run_polish_fielderr_8d.py  (appendix a)", where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_fielderr_value_4d": dict(reports="P3/polish_fielderr_value_chi4d_1000.json",
                                 producer="run_polish_fielderr_value.py --dim 4", where="laptop",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_fielderr_value_8d": dict(reports="P3/polish_fielderr_value_chi8d_1000.json",
                                 producer="run_polish_fielderr_value.py --dim 8", where="laptop",
                                 status="ready", figures=["fig04_polish_staircase"]),
    # value-only POD warm start (same shipped basis + rank as the value+gradient POD curve);
    # one run_family sweep carries BOTH fig04 rows (residual_rows + field_rows). Falls back to
    # polish_table_{4d,8d_value} + polish_fielderr_value_{4,8}d until these land.
    "polish_value_pod_4d":  dict(reports="P3/polish_fielderr_value_pod_chi4d_r75_1000.json",
                                 producer="run_polish_fielderr_value_pod.py --dim 4 --rank 75",
                                 where="cluster", status="pending",
                                 figures=["fig04_polish_staircase"]),
    "polish_value_pod_8d":  dict(reports="P3/polish_fielderr_value_pod_chi8d_r250_1000.json",
                                 producer="run_polish_fielderr_value_pod.py --dim 8 --rank 250",
                                 where="cluster", status="pending",
                                 figures=["fig04_polish_staircase"]),

    # ---- fig05 (POD compression vs memory) ----
    "gvm_4d_value":         dict(reports="P3/guess_vs_memory_4d_value_gapfill_1000.json",
                                 producer="run_guess_vs_memory.py (4d value gapfill)", where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_4d_cross":         dict(reports="P3/guess_vs_memory_4d_cross_gapfill_1000.json",
                                 producer="run_cross_pod_figuredata.py", where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_4d_field":         dict(reports="P3/guess_vs_memory_4d_field_1000.json",
                                 producer="run_guess_vs_memory.py (4d field)", where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_4d_cross_field":   dict(reports="P3/guess_vs_memory_4d_cross_field_1000.json",
                                 producer="run_cross_fielderr_sweep.py", where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "polish_table_4d":      dict(reports="P3/polish_table_qc_chi_prod_1000.json",
                                 producer="run_polish_table_qc_chi.py", where="cluster",
                                 status="ready",
                                 figures=["fig05_guess_vs_memory", "fig04_polish_staircase"]),
    "polish_table_4d_cross": dict(reports="P3/polish_table_qc_chi_prod_cross_1000.json",
                                 producer="run_polish_table_qc_chi.py (cross)", where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "polish_table_8d_value": dict(reports="P3/polish_table_chi8d_value_1000.json",
                                 producer="run_polish_table (8d value)", where="cluster",
                                 status="ready",
                                 figures=["fig05_guess_vs_memory", "fig04_polish_staircase"]),
    "gvm_8d_value":         dict(reports="P3/guess_vs_memory_8d_value_gapfill_1000.json",
                                 producer="run_guess_vs_memory (8d value)  (appendix d)", where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_8d_field":         dict(reports="P3/guess_vs_memory_8d_field_1000.json",
                                 producer="run_hermite_fielderr_sweep_8d.py (value)  (appendix e)",
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),
    # The 8-D field sweeps come in TWO enhanced flavours, and they are not
    # interchangeable.  gvm_8d_hermite_field is the PLAIN Hermite (gradient-only on all
    # six spin axes, no cross), which HISTORY_AND_FINDINGS 2.4 predicts regresses below
    # value-only — it does (1.31e-2 vs 1.80e-3 at full rank).  gvm_8d_cross_field is the
    # y-pair CROSS, the model fig03, the 4-D bottom-left panel and gvm_8d_cross (the
    # residual sibling) all use.  Both producers wrote the *_hermite_field path until
    # 2026-08-02, with only the first registered, so whichever ran last won; see
    # fig05_guess_vs_memory_data.BR_8D_ENHANCED for which one the figure plots.
    "gvm_8d_hermite_field": dict(reports="P3/guess_vs_memory_8d_hermite_field_1000.json",
                                 producer="run_hermite_fielderr_sweep_8d.py (value+grad, PLAIN Hermite: 6 spin axes, no cross)  (appendix e)",
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_8d_cross_field":   dict(reports="P3/guess_vs_memory_8d_cross_field_1000.json",
                                 producer="run_hermite_fielderr_sweep_8d_cross.py (8D y-pair CROSS FIELD sweep)",
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_8d_cross":         dict(reports="P3/guess_vs_memory_8d_hermite_gapfill_1000.json",
                                 producer="run_cross_pod_resid_8d.py (8D y-pair cross RESIDUAL sweep)",
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),

    # ---- fig06 (physical-parameter targeting) ----
    # Was P3/qc_targeting_100.json, produced against the SUPERSEDED narrow model
    # surrogate_smolyak_d4_qc_L4.npz (b in [1.5,4], DIMENSIONFUL bare spins
    # S_Ay/S_By in [-0.4,0.4], L=4, 401 nodes) — the one figure left off the
    # production box, and excluded from the reports bundle as stale.  run_qc_targeting
    # now takes its box, axes, grid and level from production_box and refuses any
    # model whose stored provenance disagrees (_check_model_box), so this artifact is
    # the production 4-D chi model (1105 nodes) the rest of the 4-D results use.
    # FIXED-BUDGET run (gradient 4, black box 14): every target is carried to a common
    # solve count, so each plotted point is the full sample.  The early-exit run
    # (``qc_targeting_chi_prod_100.json``, same seed) is kept beside it and its shared
    # prefix is bit-identical, which is what makes the cost metric (solves to
    # tolerance) identical between the two — see the data script.
    "qc_targeting":         dict(reports="P3/qc_targeting_chi_prod_fixed_100.json",
                                 producer="run_qc_targeting.py --n 100 "
                                          "--budget-grad 4 --budget-bb 14",
                                 where="cluster",
                                 status="ready", figures=["fig06_targeting"]),

    # ---- fig07 (effective-potential eccentricity) — needs a MODEL, distilled to json ----
    "qc_effpot":            dict(reports="P3/qc_effpot_Jsweep.json",
                                 producer="run_qc_effpot.py", where="cluster",
                                 status="ready", figures=["fig07_eccentricity"]),
    "effpot_model":         dict(reports="3D_parametric/models/surrogate_bpt_ecc.npz",
                                 producer="run_qc_effpot.py (parametric-model build)", where="cluster",
                                 status="ready", model=True, figures=["fig07_eccentricity"]),

    # ---- superseded as figure sources, RETAINED for provenance ----
    # Neither feeds a figure any more (both fed the former fig08 / the former
    # single-configuration fig09).  They are kept because the appendix still quotes numbers
    # produced by them: the ADM-angular-momentum diagnostics (theta_J vs theta_S to
    # ~1e-14 deg, J_y = 2 b P_x, TP agreement <= 1.4e-14) from `sweep_3d`, and the
    # axisymmetric-limit code-to-code anchor (psi to 4.7e-12, M_ADM to 1.0e-11 at b=3,
    # P=0.5) from `tp_validation`.  Deleting the entries would strand those numbers.
    "sweep_3d":             dict(reports="3D/sweep_results.json",
                                 producer="run_3d_sweep.py", where="cluster",
                                 status="ready", figures=[]),
    "tp_validation":        dict(reports="3D_parametric/qc/tp_validation_qc.json",
                                 producer="run_qc_tp_validation.py", where="cluster",
                                 status="ready", figures=[]),

    # ---- fig08 + fig09 (the TwoPunctures validation: DISTRIBUTIONS over the box) ----
    # One sweep, SHARED by two figures because it carries two abscissae: fig08 walks the
    # meridional resolution ladder (pointwise + integral agreement, plus the certified
    # residual), fig09 walks the azimuthal mode index at the best-resolved rung.  Every
    # point in both is a median with min--max whiskers over configurations sampled from the
    # production box, so no panel depends on one arbitrary parameter point.
    #   The predecessor of this sweep was two single-configuration figures, one of them a
    # separate non-axisymmetric check.  That check is redundant: the quasi-circular data are
    # ALREADY non-axisymmetric -- the tangential momentum puts ~2% of the field in the m=2
    # azimuthal mode and generic spins add ~2% at m=1 -- so the QC family exercises the
    # Fourier-in-phi solver by itself, which is what fig09 now shows.  The axisymmetric-limit
    # checks QC cannot provide (aligned-spin m>=1 suppression, the head-on code-to-code
    # anchor) are carried as quantitative statements in the appendix text.
    "tp_band_sweep":        dict(reports="3D_parametric/qc/tp_band_sweep.json",
                                 producer="run_tp_random_sweep.py --n 100 --workers 6",
                                 where="cluster", status="ready",
                                 figures=["fig08_tp_validation", "fig09_tp_spectrum"]),
}

# --- figure -> the source keys it distills + its data-script filename -------------------------
# The committed output is always figdata/<stem>.json.  "inline" figures carry their own numbers
# in the data script (no external source).
#
# ``keys`` are the top-level figdata keys the PLOTTER reads.  They are checked by
# ``_figdata.load`` and reported by ``make_figdata.py --check``, because "the json exists" is
# NOT the same as "the json is current": a figdata built before a block was added to its
# producer loads fine and then fails deep inside the plotter with a bare KeyError.  That is
# exactly how fig02 broke — its committed PDF carries the mass-ratio panel while an older local
# figdata (no ``Q_wall_q``) cannot rebuild it.  Keep this list in step with the plotter.
FIGURES = {
    "fig01_peraxis_hermite":    dict(sources=["peraxis_dist_chi"], keys=["A_per_axis"]),
    "fig02_walls":              dict(sources=["walls_dense"],
                                     keys=["B_wall_b", "Q_wall_q", "C_wall_spin"]),
    "fig03_joint_dist":         dict(sources=["joint_dist_4d", "joint_dist_cross_4d",
                                              "joint_dist_8d", "joint_dist_hermite_8d"],
                                     keys=["left", "right"]),
    "fig04_polish_staircase":   dict(sources=["polish_cold_4d", "polish_cold_8d", "polish_pod_4d",
                                              "polish_pod_8d", "polish_fielderr_4d",
                                              "polish_fielderr_8d", "polish_table_4d",
                                              "polish_table_8d_value", "polish_fielderr_value_4d",
                                              "polish_fielderr_value_8d",
                                              "polish_value_pod_4d", "polish_value_pod_8d"],
                                     keys=["cols"]),
    # Both 8-D enhanced field flavours are listed: the panel plots one of them
    # (fig05_guess_vs_memory_data.BR_8D_ENHANCED) and the guard there compares it
    # against the residual panel's model, so the graph must know about both.
    "fig05_guess_vs_memory":    dict(sources=["gvm_4d_value", "gvm_4d_cross",
                                              "gvm_4d_field", "gvm_4d_cross_field",
                                              "polish_table_4d", "polish_table_4d_cross",
                                              "polish_table_8d_value", "gvm_8d_value",
                                              "gvm_8d_field", "gvm_8d_hermite_field",
                                              "gvm_8d_cross_field", "gvm_8d_cross"],
                                     keys=["panels"]),
    # ``meta`` carries the box/level/model the run was measured on, so a figdata built
    # against the superseded narrow model cannot be replotted silently (the caption
    # states the box).
    "fig06_targeting":          dict(sources=["qc_targeting"], keys=["methods", "meta"]),
    "fig07_eccentricity":       dict(sources=["qc_effpot", "effpot_model"],
                                     keys=["Jlist", "per_J", "bg", "n_scan", "n_grad"]),
    # one shared source, two figures: the resolution ladder and the azimuthal spectrum share
    # no abscissa, so each distills its own block of tp_band_sweep (see SOURCES above)
    "fig08_tp_validation":      dict(sources=["tp_band_sweep"],
                                     keys=["ladder", "meta"]),
    "fig09_tp_spectrum":        dict(sources=["tp_band_sweep"],
                                     keys=["spectrum", "meta"]),
    # RECOMPUTE, not a distillation: the data script runs the solve and the FD constraint
    # monitor itself (both cheap) and queries the TwoPunctures binary directly, so there is no
    # reports/ artifact to declare -- hence ``inline``.  Its one heavy input is the oracle
    # binary (make oracle), which is why the script also has a --no-tp mode.  ``meta`` carries
    # the configuration and the Cartesian ladder the caption states.
    "fig10_constraints":        dict(sources=[], inline=True,
                                     keys=["curves", "meta"]),
}


def figure_stems():
    return list(FIGURES)


def sources_for(stem):
    return FIGURES[stem].get("sources", [])


def keys_for(stem):
    """Top-level figdata keys the plotter of ``stem`` requires (empty if undeclared)."""
    return FIGURES.get(stem, {}).get("keys", [])
