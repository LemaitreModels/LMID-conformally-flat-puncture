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

            ``producer`` is STRUCTURED — ``_prod(module, *argv, dim=, note=)`` — not a
            free-text string, and ``producer_cmd(key)`` renders it into a command that can
            be pasted into a shell.  It used to be prose, and four entries had drifted off
            the code: ``gvm_4d_value``, ``gvm_4d_field`` and ``gvm_8d_value`` named
            ``run_guess_vs_memory``, which writes one flat ``guess_vs_memory.json`` and
            cannot emit their filenames, and ``effpot_model`` named ``run_qc_effpot``, which
            CONSUMES that model — ``build_surrogate`` builds it.  ``make_figdata.py --check``
            prints these as the rebuild command, so a wrong one actively misdirects.
            ``tests/test_paper_figures.py`` now pins every module against ``pipeline/`` and
            every rank against ``production_model.SHIPPED_RANK``.

  FIGURES   figure stem -> the source keys it needs + its data-script filename.  The committed
            output is always ``figdata/<stem>.json``.

The raw SOURCE artifacts are NOT committed (``reports/`` and ``*.npz`` are gitignored, and the
spectral corpora are multi-GB).  The small distilled ``figdata/*.json`` ARE committed, so every
figure rebuilds from the repo alone with no solves, no models, and no jax.  ``make_figdata.py``
uses this registry for the presence check, the dedup, and the cluster-command hints.
"""
from __future__ import annotations

from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm
# ^ pure stdlib + production_box: no jax, no solver, so the "plotters need no jax"
#   property of this tier survives.  It is here so the shipped rank and the shipped
#   artifact names are read from their single source rather than restated.

PIPELINE = "lemaitre.initial_data.conformally_flat_puncture.pipeline"


def _prod(module, *argv, dim=None, note=""):
    """A producer: the ``pipeline/`` module that WRITES this source, plus its argv.

    ``module`` is the basename under ``pipeline/`` without ``.py`` (``None`` when
    nothing in this package produces the source).  ``dim`` is the model dimension
    the source measures, and it is what resolves the argv placeholders below, so
    the shipped rank and the shipped artifact names are never restated here —
    ``production_model`` remains their single source (CLAUDE.md).

    Placeholders, substituted by :func:`producer_cmd`:

    ==================  =========================================================
    ``{dim}``           ``4`` / ``8``
    ``{rank}``          ``production_model.SHIPPED_RANK[dim]``
    ``{model_stem}``    the untruncated shipped cross corpus, ``.npz``
    ``{pod_stem}``      the shipped rank-``{rank}`` POD artifact, ``.npz``
    ==================  =========================================================
    """
    return dict(module=module, argv=list(argv), dim=dim, note=note)


def _tokens(dim):
    if dim is None:
        return {}
    return {"{dim}": str(dim),
            "{rank}": str(pm.SHIPPED_RANK[dim]),
            "{model_stem}": pm.model_stem(dim) + ".npz",
            "{pod_stem}": pm.pod_stem(dim) + ".npz"}


def producer_cmd(key):
    """The runnable command that regenerates ``SOURCES[key]``.

    Raises ``KeyError`` on an argv placeholder the entry's ``dim`` cannot resolve —
    so a half-declared producer fails the registry test rather than printing a
    command with a literal ``{rank}`` in it.
    """
    p = SOURCES[key]["producer"]
    note = f"   # {p['note']}" if p["note"] else ""
    if p["module"] is None:
        return (p["note"] or "no producer in this package")
    tok = _tokens(p["dim"])
    argv = []
    for a in p["argv"]:
        for t, v in tok.items():
            a = a.replace(t, v)
        if "{" in a:
            raise KeyError(f"{key}: unresolved placeholder in {a!r} (dim={p['dim']!r})")
        argv.append(a)
    return f"python -m {PIPELINE}.{p['module']}" + "".join(f" {a}" for a in argv) + note


# --- raw run outputs (under reports/); NOT committed --------------------------
# where: "laptop"  -> the distill step only reshapes json already on disk (no solves)
#        "cluster" -> the source is produced by a heavy CPU run on a cluster (see docs/DATA.md)
# status: "ready"   -> present on the laptop today
#         "pending" -> an 8D artifact still to be produced on the cluster
SOURCES = {
    # ---- fig01 (per-axis Hermite, DISTRIBUTION over random base points) ----
    "peraxis_dist_chi":     dict(reports="3D_parametric/qc_chi/peraxis_dist_chi.json",
                                 producer=_prod("run_qc_peraxis_dist_chi", "--assemble"),
                                 where="cluster",
                                 status="ready", figures=["fig01_peraxis_hermite"]),

    # ---- fig02 (all three analyticity walls, merged) ----
    # Was 3D_parametric/qc/walls_d4_qc_dense.json, which no producer in this package
    # writes: it is the monorepo's run_qc_dense_stats.py, on the superseded narrow box
    # b in [1.5,4] in the OLD bare-spin (S_Ay,S_By) parameterization, and it carries no
    # mass-ratio block.  The production producer is the one named here; "dense" now
    # refers to its wall blocks' 21-point held-out sets and 5-level Q ladders.
    "walls_dense":          dict(reports="3D_parametric/qc_chi_prod/walls_d4_qc_chi.json",
                                 producer=_prod("run_qc_walls_sweep_chi_prod", dim=4),
                                 where="cluster",
                                 status="ready",
                                 figures=["fig02_walls"]),

    # ---- fig03 (joint held-out distribution) ----
    "joint_dist_4d":        dict(reports="3D_parametric/qc_chi/joint_dist_d4_qc_chi_prod.json",
                                 producer=_prod("run_qc_joint_dist_chi",
                                                "--box", "d4_qc_chi_prod", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig03_joint_dist"]),
    "joint_dist_cross_4d":  dict(reports="3D_parametric/qc_chi/joint_dist_cross_d4_qc_chi_prod.json",
                                 producer=_prod("run_qc_joint_dist_cross_chi", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig03_joint_dist"]),
    "joint_dist_8d":        dict(reports="3D_parametric/qc_chi/joint_dist_spin8_qc_chi_prod.json",
                                 producer=_prod("run_qc_joint_dist_chi",
                                                "--box", "spin8_qc_chi_prod",
                                                dim=8, note="appendix b"),
                                 where="cluster", status="ready", figures=["fig03_joint_dist"]),
    "joint_dist_hermite_8d": dict(reports="3D_parametric/qc_chi/joint_dist_hermite_spin8_qc_chi_prod.json",
                                 producer=_prod("run_qc_joint_dist_hermite_8d",
                                                dim=8, note="appendix c"),
                                 where="cluster", status="ready", figures=["fig03_joint_dist"]),

    # ---- fig04 (certified refinement staircase) ----
    "polish_cold_4d":       dict(reports="P3/polish_cold_chi4d_1000.json",
                                 producer=_prod("run_polish_cold", "--dim", "{dim}", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_cold_8d":       dict(reports="P3/polish_cold_chi8d_1000.json",
                                 producer=_prod("run_polish_cold", "--dim", "{dim}", dim=8),
                                 where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    # All six fig04 POD curves share ONE model family: the y-pair full-bilinear CROSS
    # POD (the model the paper ships, cf. sec:model:enhanced / fig:joint), at r=250 in
    # 4D and r=500 in 8D -- the rank at which fig05's ladder shows the compression is
    # no longer the limiting error.  Before this revision the 8D residual row used the
    # SIX-axis non-cross POD while its field-error row used the cross POD; the two rows
    # of that column were therefore different models, masked by a shared r=250.
    "polish_pod_4d":        dict(reports="P3/polish_table_chi4d_pod_r250_cross_1000.json",
                                 producer=_prod("run_polish_table",
                                                "--model", "{pod_stem}",
                                                "--tag", "chi{dim}d_pod_r{rank}_cross", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_pod_8d":        dict(reports="P3/polish_table_chi8d_pod_r500_cross_1000.json",
                                 producer=_prod("run_polish_table",
                                                "--model", "{pod_stem}",
                                                "--tag", "chi{dim}d_pod_r{rank}_cross", dim=8),
                                 where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_fielderr_4d":   dict(reports="P3/polish_fielderr_chi4d_r250_1000.json",
                                 producer=_prod("run_polish_fielderr", "--rank", "{rank}", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_fielderr_8d":   dict(reports="P3/polish_fielderr_chi8d_r500_1000.json",
                                 producer=_prod("run_polish_fielderr_8d", "--rank", "{rank}",
                                                dim=8, note="appendix a"),
                                 where="cluster",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_fielderr_value_4d": dict(reports="P3/polish_fielderr_value_chi4d_1000.json",
                                 producer=_prod("run_polish_fielderr_value", "--dim", "{dim}", dim=4),
                                 where="laptop",
                                 status="ready", figures=["fig04_polish_staircase"]),
    "polish_fielderr_value_8d": dict(reports="P3/polish_fielderr_value_chi8d_1000.json",
                                 producer=_prod("run_polish_fielderr_value", "--dim", "{dim}", dim=8),
                                 where="laptop",
                                 status="ready", figures=["fig04_polish_staircase"]),
    # value-only POD warm start: the SAME cross-POD basis Phi[:, :r] and the same rank as
    # the value+gradient curve above, differing ONLY in whether the coefficient interpolant
    # uses the certified tangents -- which is what the fig04 caption claims.  (Before this
    # revision it was built on the non-cross basis, so the two matched in rank but not in
    # basis; hence the explicit --model.)  One run_family sweep carries BOTH fig04 rows
    # (residual_rows + field_rows).  Falls back to polish_table_{4d,8d_value} +
    # polish_fielderr_value_{4,8}d if these are absent.
    "polish_value_pod_4d":  dict(reports="P3/polish_fielderr_value_pod_chi4d_r250_1000.json",
                                 producer=_prod("run_polish_fielderr_value_pod",
                                                "--dim", "{dim}", "--rank", "{rank}",
                                                "--model", "{pod_stem}", dim=4),
                                 where="cluster", status="ready",
                                 figures=["fig04_polish_staircase"]),
    "polish_value_pod_8d":  dict(reports="P3/polish_fielderr_value_pod_chi8d_r500_1000.json",
                                 producer=_prod("run_polish_fielderr_value_pod",
                                                "--dim", "{dim}", "--rank", "{rank}",
                                                "--model", "{pod_stem}", dim=8),
                                 where="cluster", status="ready",
                                 figures=["fig04_polish_staircase"]),

    # ---- fig05 (POD compression vs memory) ----
    # `run_guess_vs_memory` was named here for the two `*_gapfill` sources and for
    # `gvm_4d_field`, and it produces none of them: it writes a single flat
    # `P3/guess_vs_memory.json`.  The real producers are below (docs/DATA.md carries
    # the same verified mapping and the reasoning behind the non-obvious ones).
    "gvm_4d_value":         dict(reports="P3/guess_vs_memory_4d_value_gapfill_1000.json",
                                 producer=_prod("run_value_pod_gapfill_4d", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_4d_cross":         dict(reports="P3/guess_vs_memory_4d_cross_gapfill_1000.json",
                                 producer=_prod("run_cross_pod_figuredata", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    # One `run_cross_fielderr_sweep` run writes BOTH flavours and shares the expensive
    # certified `u_true` solves between them, which is why the value curve lives in the
    # cross sweep: both fig05 bottom-left curves are then measured against one truth.
    # Its default `--flavours cross,value` produces this source and the next together.
    "gvm_4d_field":         dict(reports="P3/guess_vs_memory_4d_field_1000.json",
                                 producer=_prod("run_cross_fielderr_sweep",
                                                "--flavours", "value", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_4d_cross_field":   dict(reports="P3/guess_vs_memory_4d_cross_field_1000.json",
                                 producer=_prod("run_cross_fielderr_sweep",
                                                "--flavours", "cross", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "polish_table_4d":      dict(reports="P3/polish_table_qc_chi_prod_1000.json",
                                 producer=_prod("run_polish_table_qc_chi", dim=4),
                                 where="cluster",
                                 status="ready",
                                 figures=["fig05_guess_vs_memory", "fig04_polish_staircase"]),
    # The UNTRUNCATED cross corpus, not a POD of it -- corroborated by the
    # 8*N*(1+d+npair)*nfeat memory fig05 applies to this curve (docs/DATA.md).
    "polish_table_4d_cross": dict(reports="P3/polish_table_qc_chi_prod_cross_1000.json",
                                 producer=_prod("run_polish_table",
                                                "--model", "{model_stem}",
                                                "--tag", "qc_chi_prod_cross", dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    # The 8-D VALUE surrogate (no gradient blocks), so its model is not one
    # production_model names -- corroborated by fig05's 8*N*nfeat value-only memory.
    "polish_table_8d_value": dict(reports="P3/polish_table_chi8d_value_1000.json",
                                 producer=_prod("run_polish_table",
                                                "--model", "surrogate_smolyak_spin8_qc_chi_prod_L5.npz",
                                                "--tag", "chi{dim}d_value", dim=8),
                                 where="cluster",
                                 status="ready",
                                 figures=["fig05_guess_vs_memory", "fig04_polish_staircase"]),
    "gvm_8d_value":         dict(reports="P3/guess_vs_memory_8d_value_gapfill_1000.json",
                                 producer=_prod("run_value_pod_gapfill_8d",
                                                dim=8, note="appendix d"),
                                 where="cluster",
                                 status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_8d_field":         dict(reports="P3/guess_vs_memory_8d_field_1000.json",
                                 producer=_prod("run_hermite_fielderr_sweep_8d", dim=8,
                                                note="value flavour; appendix e"),
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),
    # The 8-D field sweeps come in TWO enhanced flavours, and they are not
    # interchangeable.  gvm_8d_hermite_field is the PLAIN Hermite (gradient-only on all
    # six spin axes, no cross), which regresses below
    # value-only — it does (1.31e-2 vs 1.80e-3 at full rank).  gvm_8d_cross_field is the
    # y-pair CROSS, the model fig03, the 4-D bottom-left panel and gvm_8d_cross (the
    # residual sibling) all use.  Both producers wrote the *_hermite_field path until
    # 2026-08-02, with only the first registered, so whichever ran last won; see
    # fig05_guess_vs_memory_data.BR_8D_ENHANCED for which one the figure plots.
    "gvm_8d_hermite_field": dict(reports="P3/guess_vs_memory_8d_hermite_field_1000.json",
                                 producer=_prod("run_hermite_fielderr_sweep_8d", dim=8,
                                                note="value+grad, PLAIN Hermite: 6 spin axes, "
                                                     "no cross; written by the same run as "
                                                     "gvm_8d_field; appendix e"),
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_8d_cross_field":   dict(reports="P3/guess_vs_memory_8d_cross_field_1000.json",
                                 producer=_prod("run_hermite_fielderr_sweep_8d_cross", dim=8,
                                                note="8D y-pair CROSS FIELD sweep"),
                                 where="cluster", status="ready", figures=["fig05_guess_vs_memory"]),
    "gvm_8d_cross":         dict(reports="P3/guess_vs_memory_8d_hermite_gapfill_1000.json",
                                 producer=_prod("run_cross_pod_resid_8d",
                                                "--cross-model", "{model_stem}", dim=8,
                                                note="8D y-pair cross RESIDUAL sweep; builds "
                                                     "its own full-rank POD from the corpus"),
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
                                 producer=_prod("run_qc_targeting", "--n", "100",
                                                "--budget-grad", "4", "--budget-bb", "14",
                                                dim=4),
                                 where="cluster",
                                 status="ready", figures=["fig06_targeting"]),

    # ---- fig07 (effective-potential eccentricity) — needs a MODEL, distilled to json ----
    "qc_effpot":            dict(reports="P3/qc_effpot_Jsweep.json",
                                 producer=_prod("run_qc_effpot"), where="cluster",
                                 status="ready", figures=["fig07_eccentricity"]),
    # `run_qc_effpot` CONSUMES this model, it does not build it: it loads the fixed
    # basename through parametric_nd.load_parametric, which rejects a Smolyak file
    # outright, so the build must pass --dense-Q/--dense-name.  Derivation of the
    # box edges, Q=16 and the FIXED overrides: docs/DATA.md, "the eccentricity family".
    "effpot_model":         dict(reports="3D_parametric/models/surrogate_bpt_ecc.npz",
                                 producer=_prod("build_surrogate", "--box", "bpt_ecc",
                                                "--level", "5", "--dense-Q", "16",
                                                "--dense-name", "surrogate_bpt_ecc.npz"),
                                 where="cluster",
                                 status="ready", model=True, figures=["fig07_eccentricity"]),

    # ---- superseded as figure sources, RETAINED as standalone diagnostics ----
    # Neither feeds a figure any more (both fed the former fig08 / the former
    # single-configuration fig09), and neither supplies a paper numeral any more either:
    # the appendix's axisymmetric-limit anchor (psi to 2.6e-10, M_ADM to 5.0e-11, certified
    # residual 1.2e-12 at b=3, P=0.5) and its "all m>=1 below 1e-16" statement now come from
    # `tp_band_sweep` itself -- the `anchor` and `axisym_m_ge1_max` entries of fig08's
    # committed figdata meta.  The ADM-angular-momentum diagnostics these once supplied
    # (theta_J vs theta_S, J_y = 2 b P_x) appear nowhere in the manuscript.
    # They are kept because their producers are still useful run-by-hand diagnostics and
    # this is the only record of where their output lands; retiring the producers is a
    # separate decision from retiring these entries.
    "sweep_3d":             dict(reports="3D/sweep_results.json",
                                 producer=_prod("run_3d_sweep"), where="cluster",
                                 status="ready", figures=[]),
    "tp_validation":        dict(reports="3D_parametric/qc/tp_validation_qc.json",
                                 producer=_prod("run_qc_tp_validation"), where="cluster",
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
                                 producer=_prod("run_tp_random_sweep", "--n", "100",
                                                "--workers", "6"),
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
                                     keys=["B_wall_b", "Q_wall_q", "C_wall_spin",
                                           "meta"]),
    "fig03_joint_dist":         dict(sources=["joint_dist_4d", "joint_dist_cross_4d",
                                              "joint_dist_8d", "joint_dist_hermite_8d"],
                                     keys=["left", "right", "meta"]),
    "fig04_polish_staircase":   dict(sources=["polish_cold_4d", "polish_cold_8d", "polish_pod_4d",
                                              "polish_pod_8d", "polish_fielderr_4d",
                                              "polish_fielderr_8d", "polish_table_4d",
                                              "polish_table_8d_value", "polish_fielderr_value_4d",
                                              "polish_fielderr_value_8d",
                                              "polish_value_pod_4d", "polish_value_pod_8d"],
                                     keys=["cols", "meta"]),
    # Both 8-D enhanced field flavours are listed: the panel plots one of them
    # (fig05_guess_vs_memory_data.BR_8D_ENHANCED) and the guard there compares it
    # against the residual panel's model, so the graph must know about both.
    "fig05_guess_vs_memory":    dict(sources=["gvm_4d_value", "gvm_4d_cross",
                                              "gvm_4d_field", "gvm_4d_cross_field",
                                              "polish_table_4d", "polish_table_4d_cross",
                                              "polish_table_8d_value", "gvm_8d_value",
                                              "gvm_8d_field", "gvm_8d_hermite_field",
                                              "gvm_8d_cross_field", "gvm_8d_cross"],
                                     # 'meta' is the model-provenance block (which model,
                                     # which enhanced axes, which rank, what memory
                                     # accounting).  Declaring it makes a figdata built
                                     # before the shipped-model fix read as STALE rather
                                     # than silently feeding the paper old memory numbers.
                                     keys=["panels", "meta"]),
    # ``meta`` carries the box/level/model the run was measured on, so a figdata built
    # against the superseded narrow model cannot be replotted silently (the caption
    # states the box).
    "fig06_targeting":          dict(sources=["qc_targeting"], keys=["methods", "meta"]),
    # ``meta`` names the model artifact the curves were evaluated from (file, build
    # commit, box, dense Q, grid) -- the fig06 failure mode, since that artifact is
    # gitignored.  Pinned here in the same commit as the re-distill that produced it.
    # NOTE this pins top-level PRESENCE only.  The per-J numerals added alongside it
    # (b_circ_scan, d_bcirc_abs/_rel, scan_argmin_b, dEb_db_certified, and the
    # ecc block) live INSIDE per_J and so cannot be declared here at all; see
    # tests/test_paper_figures.py, which asserts them structurally instead.
    "fig07_eccentricity":       dict(sources=["qc_effpot", "effpot_model"],
                                     keys=["Jlist", "per_J", "bg", "n_scan", "n_grad",
                                           "meta"]),
    # one shared source, two figures: the resolution ladder and the azimuthal spectrum share
    # no abscissa, so each distills its own block of tp_band_sweep (see SOURCES above)
    "fig08_tp_validation":      dict(sources=["tp_band_sweep"],
                                     keys=["ladder", "meta"]),
    "fig09_tp_spectrum":        dict(sources=["tp_band_sweep"],
                                     keys=["spectrum", "meta"]),
    # DISTILLATION of an EXTERNAL run tree, not a recompute: the constraint norms are measured
    # by GRTeclyn, so the data script only reads that code's
    # constraint_norms.json output.  The tree lives outside the repo and outside reports/ --
    # pass --runs or set $LM_GRTECLYN_RUNS -- hence still ``inline`` (no reports/ key to
    # declare).  ``meta`` carries the box, ladder, exclusion radius and spectral grid the
    # caption states; ``amr`` the separately-quoted refined-hierarchy numbers with their
    # per-level breakdown; ``variants`` the resolution tests that attribute an observed floor.
    "fig10_constraints":        dict(sources=[], inline=True,
                                     keys=["curves", "meta", "amr", "variants"]),
}


def figure_stems():
    return list(FIGURES)


def sources_for(stem):
    return FIGURES[stem].get("sources", [])


def keys_for(stem):
    """Top-level figdata keys the plotter of ``stem`` requires (empty if undeclared)."""
    return FIGURES.get(stem, {}).get("keys", [])
