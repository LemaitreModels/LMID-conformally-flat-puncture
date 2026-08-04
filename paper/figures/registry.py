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
            figures (e.g. ``sweep_3d`` -> fig08+fig09) is listed
            once and produced once.

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
    "gvm_8d_hermite_field": dict(reports="P3/guess_vs_memory_8d_hermite_field_1000.json",
                                 producer="run_hermite_fielderr_sweep_8d.py (value+grad)  (appendix e)",
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_8d_cross":         dict(reports="P3/guess_vs_memory_8d_hermite_gapfill_1000.json",
                                 producer="run_cross_pod_resid_8d.py (8D y-pair cross RESIDUAL sweep)",
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),

    # ---- fig06 (physical-parameter targeting) ----
    "qc_targeting":         dict(reports="P3/qc_targeting_100.json",
                                 producer="run_qc_targeting.py", where="cluster",
                                 status="ready", figures=["fig06_targeting"]),

    # ---- fig07 (effective-potential eccentricity) — needs a MODEL, distilled to json ----
    "qc_effpot":            dict(reports="P3/qc_effpot_Jsweep.json",
                                 producer="run_qc_effpot.py", where="cluster",
                                 status="ready", figures=["fig07_eccentricity"]),
    "effpot_model":         dict(reports="3D_parametric/models/surrogate_bpt_ecc.npz",
                                 producer="run_qc_effpot.py (surrogate build)", where="cluster",
                                 status="ready", model=True, figures=["fig07_eccentricity"]),

    # ---- fig08 (3D validation) + fig09 (consolidated TP validation) — SHARED source ----
    "sweep_3d":             dict(reports="3D/sweep_results.json",
                                 producer="run_3d_validation_sweep.py -> plot_3d_sweep.py", where="cluster",
                                 status="ready",
                                 figures=["fig08_3d_validation", "fig09_tp_validation"]),

    # ---- fig09 (consolidated TwoPunctures validation: ADM-J + quasi-circular psi/M_ADM) ----
    "tp_validation":        dict(reports="3D_parametric/qc/tp_validation_qc.json",
                                 producer="run_qc_tp_validation.py", where="cluster",
                                 status="ready", figures=["fig09_tp_validation"]),

    # ---- fig10 (TP agreement as a DISTRIBUTION over the production box) ----
    # Complements, not replaces, fig08/fig09: those are convergence ladders at a FIXED
    # configuration (x axis = resolution), which is what shows the difference to be
    # resolution-limited.  Neither samples the box the models are claimed over -- fig08
    # sits at b=1.5, BELOW production B_MIN=3, and no external check anywhere reaches
    # q != 1 or production spin magnitudes.  This source closes that.
    "tp_random_sweep":      dict(reports="3D_parametric/qc/tp_random_sweep.json",
                                 producer="run_tp_random_sweep.py --n 100 --workers 6",
                                 where="cluster", status="ready",
                                 figures=["fig10_tp_box_sample"]),
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
    "fig05_guess_vs_memory":    dict(sources=["gvm_4d_value", "gvm_4d_cross",
                                              "gvm_4d_field", "gvm_4d_cross_field",
                                              "polish_table_4d", "polish_table_4d_cross",
                                              "polish_table_8d_value", "gvm_8d_value",
                                              "gvm_8d_field", "gvm_8d_hermite_field",
                                              "gvm_8d_cross"],
                                     keys=["panels"]),
    "fig06_targeting":          dict(sources=["qc_targeting"], keys=["methods"]),
    "fig07_eccentricity":       dict(sources=["qc_effpot", "effpot_model"],
                                     keys=["Jlist", "per_J", "bg", "n_scan", "n_grad"]),
    "fig08_3d_validation":      dict(sources=["sweep_3d"], keys=["ladder", "spectrum"]),
    "fig09_tp_validation":      dict(sources=["sweep_3d", "tp_validation"], keys=["C_psi_adm"]),
    "fig10_tp_box_sample":      dict(sources=["tp_random_sweep"],
                                     keys=["interior", "edge", "summary", "meta"]),
}


def figure_stems():
    return list(FIGURES)


def sources_for(stem):
    return FIGURES[stem].get("sources", [])


def keys_for(stem):
    """Top-level figdata keys the plotter of ``stem`` requires (empty if undeclared)."""
    return FIGURES.get(stem, {}).get("keys", [])
