#!/usr/bin/env python
"""Data for fig05_guess_vs_memory: distill the POD memory<->accuracy curves to figdata/.

A 2x2 grid: rows = (bare-guess constraint residual, bare-guess field error), cols = (4D, 8D).
Each panel carries a value-only (C0) and a value+gradient+cross (full-bilinear, C1) curve, each a
POD rank sweep + a full-rank bare-guess star. This script computes, per panel/curve, the exact
plot_family inputs (the rank-sweep points, the bare star + its memory, and the annotations), so the
plotter draws from figdata alone. Rank ladders are thinned to every other rung (see ``_thin``) to
keep the panels legible.

Curve LABELS live in the plot script (``CURVE_LABELS``), not here: they are presentation, so a
notation change needs only a re-plot, not a solver recompute. The two curves are written in the
fixed order (value, value+gradient+cross) that ``CURVE_LABELS`` assumes.

Sources (raw): gvm_all, gvm_4d_{value,cross,field,cross_field},
               gvm_8d_{value,field,cross}, one of gvm_8d_{hermite_field,cross_field}
               (see ``BR_8D_ENHANCED``),
               polish_table_{4d,4d_cross,8d_hermite,8d_value}  (registry keys).

The two 8-D value+gradient curves (residual, top right; field error, bottom right) must
come from ONE model — ``_check_8d_enhanced`` enforces it.  See that function for why.

MEMORY IS RECOMPUTED HERE, not read from the raw sweeps.  Every stored byte count is
``pipeline.production_model`` applied to the number of fields the model actually
stores per node -- ``1 + n_ENHANCED + n_pairs`` (:func:`production_model.blocks_of_model`).
The sweeps themselves recorded ``1 + d + npair``, which inflated the shipped y-pair
cross model by 1.5x (4-D) / 2.5x (8-D) because it counted a tangent for every axis
rather than for the two enhanced ones; the accuracy statistics were never affected,
so the correction is a re-distill (``make figdata``) and needs no re-sweep.  Raw files
written after that fix carry their own ``blocks`` and are cross-checked against it.

  value curves     blocks = 1                                  (node_U only)
  cross curves     blocks = 1 + n_enh + npair = 4              (the shipped model)

Run:  python fig05_guess_vs_memory_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump
from lm.initial_data.pipeline import production_model as pm

VAL_C, HERM_C = "C0", "C1"
POD5 = ("r", "mem_bytes", "min", "median", "max")
DENSE_LADDER = 10   # rungs a pre-thinning run produced; see pipeline thin_ranks()

# Which 8-D value+gradient model the BOTTOM-RIGHT (field-error) panel plots.
#
#   "gvm_8d_cross_field"   the y-pair CROSS model — the model the TOP-RIGHT residual
#                          panel, the BOTTOM-LEFT 4-D panel and fig03 all use.
#   "gvm_8d_hermite_field" the PLAIN Hermite model (gradient-only on all six spin
#                          axes, no cross term), which regresses below value-only
#                          exactly as HISTORY_AND_FINDINGS 2.4 predicts.
#
# These are different models, and until 2026-08-02 both producers wrote the
# *_hermite_field path, so the panel silently carried whichever ran last.  The two
# curves in the right-hand column must describe ONE model — _check_8d_enhanced()
# enforces that below.  Changing this key is a scientific choice about what the
# figure claims, so it is a single explicit line rather than an implicit default.
#
# The CROSS is the shipped model (HISTORY_AND_FINDINGS 2.4), and the one fig03, the
# 4-D panel and the residual panel above already plot.  Measured on the production
# box it beats value-only by 1.692x (1.0641e-3 vs 1.8009e-3), matching fig03's
# independent 1.700x to three digits.
BR_8D_ENHANCED = "gvm_8d_cross_field"


def _thin(cur):
    """Keep every other rung of a dense rank ladder, plus the full-rank one.

    Matches ``pipeline.run_cross_fielderror_chi.thin_ranks`` (which the producers now
    apply at sweep time, so newer runs arrive already thinned and pass through here
    untouched); this keeps the panels legible for runs made with the dense ladder.
    """
    return cur if len(cur) < DENSE_LADDER else [*cur[:-1][::2], cur[-1]]


def blocks_of(raw, kind):
    """Stored fields per node for a raw sweep, in ``production_model``'s accounting.

    ``kind`` is ``"value"`` (1 block) or ``"cross"`` (the shipped y-pair cross model,
    ``1 + n_enh + npair``).  Newer sweeps record ``blocks`` themselves; when they do,
    it must agree -- a mismatch means the sweep measured a different model from the
    one this panel claims, which is exactly the confusion this guard exists to catch.
    """
    want = 1 if kind == "value" else 1 + pm.n_enhanced() + int(raw.get("npair", 1))
    got = raw.get("blocks")
    if got is not None and int(got) != want:
        raise SystemExit(
            f"fig05: sweep stores {int(got)} blocks/node but this panel treats it as "
            f"'{kind}' ({want} blocks). The raw sweep is a different model than the "
            f"panel claims -- check its 'model' field against production_model.model_stem().")
    return want


def _pod5(cur, N, nfeat, blocks):
    """Distil a rank ladder, RECOMPUTING each point's memory from the block count."""
    out = []
    for c in _thin(cur):
        e = {k: c[k] for k in POD5}
        e["mem_bytes"] = pm.pod_bytes_of(int(c["r"]), N, nfeat, blocks)
        out.append(e)
    return out


def _mmm(rec):
    return {k: rec[k] for k in ("median", "min", "max")}


def _guess(key):
    """bare-guess {median,min,max} from a polish_table json (rows.guess)."""
    return _mmm(load_source(key)["rows"]["guess"])


def curve(raw, bare, color, marker, ann, drop_last, *, kind):
    """One panel curve: the thinned rank ladder + its full-rank bare star.

    Both memories come from :mod:`production_model`; the raw sweep supplies only
    ``N``/``nfeat``/``npair`` and the accuracy statistics.
    """
    N, nfeat = int(raw["N"]), int(raw["nfeat"])
    blocks = blocks_of(raw, kind)
    return dict(cur=_pod5(raw["pod_curve"], N, nfeat, blocks), bare=bare,
                bare_mem=pm.bare_bytes_of(N, nfeat, blocks) / 1e6,
                bare_r=raw["r_full"], blocks=blocks,
                color=color, marker=marker, ann=ann, drop_last=drop_last)


def _check_8d_enhanced(resid, field):
    """The two 8-D value+gradient curves must describe the SAME model.

    The right-hand column shows one model measured two ways: residual on top, field
    error below.  They therefore share a bare-guess memory (the x-position of the
    star) and a model name.  On 2026-08-01 they did not: the field sweep carried the
    six-axis plain-Hermite model while the residual sweep carried the y-pair cross,
    because both 8-D field producers wrote one filename and the plain one ran last.
    Nothing caught it — the figure built cleanly and the panel simply showed a
    different model from the one its neighbours and fig03 show.

    Memory is the sharp discriminator: the cross model stores one extra block per
    node, so bare_mem differs by exactly (1+d+npair)/(1+d) = 10/9 at d=8.
    """
    rm, fm = resid.get("model"), field.get("model")
    rb, fb = resid.get("bare_mem_bytes"), field.get("bare_mem_bytes")
    if rm != fm or (rb and fb and abs(rb - fb) > 1e-6 * max(rb, fb)):
        raise SystemExit(
            "fig05: the two 8-D value+gradient curves are DIFFERENT models.\n"
            f"  TOP-RIGHT  (residual, gvm_8d_cross)   model={rm}  bare_mem={rb}\n"
            f"  BOTTOM-RIGHT (field, {BR_8D_ENHANCED})  model={fm}  bare_mem={fb}\n"
            "  They must be one model measured two ways (see this function's docstring).\n"
            "  Either:\n"
            "    (a) set BR_8D_ENHANCED = 'gvm_8d_cross_field' and produce it with\n"
            "        python -m lm.initial_data.pipeline.run_hermite_fielderr_sweep_8d_cross\n"
            "        (~6-7 h with the shared certified-truth cache warm, ~21 h cold), or\n"
            "    (b) keep the plain-Hermite field curve and repoint the TOP-RIGHT residual\n"
            "        panel at the matching plain-Hermite residual sweep.\n"
            "  Option (a) matches the 4-D bottom-left panel and fig03; (b) does not.")


def _meta(gh4, gx8):
    """Provenance: WHICH model each enhanced curve is, so a caption can never drift
    from the measurement again (the fig06 lesson).  Read by tests/test_paper_figures."""
    return dict(
        model={str(dim): pm.model_stem(dim) for dim in (4, 8)},
        enhanced_axes=list(pm.ENHANCED_AXES),
        n_enhanced=pm.n_enhanced(), n_pairs=pm.n_pairs(),
        blocks_per_node=pm.stored_blocks("shipped"),
        shipped_rank={str(dim): pm.SHIPPED_RANK[dim] for dim in (4, 8)},
        smolyak_level=pm.pb.SMOLYAK_LEVEL,
        nodes={"4": int(gh4["N"]), "8": int(gx8["N"])},
        bare_mib={"4": pm.bare_bytes(4) / 2**20, "8": pm.bare_bytes(8) / 2**20},
        shipped_pod_mib={str(dim): pm.pod_bytes(pm.SHIPPED_RANK[dim], dim) / 2**20
                         for dim in (4, 8)},
        compression={str(dim): pm.compression_factor(dim) for dim in (4, 8)},
    )


def build():
    # ---- TOP-LEFT: 4D residual (value + value+gradient cross) ----
    gv = load_source("gvm_4d_value")
    gh = load_source("gvm_4d_cross")
    TL = [curve(gv, _guess("polish_table_4d"), VAL_C, "o", "below", True, kind="value"),
          curve(gh, _guess("polish_table_4d_cross"), HERM_C, "s", "above", True,
                kind="cross")]

    # ---- TOP-RIGHT: 8D residual (value gapfill + value+gradient y-pair-CROSS gapfill) ----
    # Both to r_full over the SAME 1000 seed-0 points; the value+gradient curve is the RESIDUAL
    # sibling of the bottom-right field sweep (same y-pair cross model, same ranks/memory), so the
    # two 8D value+gradient curves share identical x-positions (model, ranks, bare_mem).
    gv8 = load_source("gvm_8d_value")
    gx8 = load_source("gvm_8d_cross")
    TR = [curve(gv8, _guess("polish_table_8d_value"), VAL_C, "o", "below", True,
                kind="value"),
          curve(gx8, _mmm(gx8["pod_curve"][-1]), HERM_C, "s", "above", True, kind="cross")]

    # ---- BOTTOM-LEFT: 4D field error (value + value+gradient cross) ----
    # Flat schema, matching the 8-D siblings below.  Was ["value"]: a leftover
    # from when run_guess_vs_memory wrote ONE combined two-flavour file; the
    # --flavours path now writes one flat file per flavour, so gvm_4d_field has
    # pod_curve/N/nfeat/r_full at top level exactly like gvm_8d_field.
    fv = load_source("gvm_4d_field")
    gc = load_source("gvm_4d_cross_field")
    BL = [curve(fv, _mmm(fv["pod_curve"][-1]), VAL_C, "o", "below", True, kind="value"),
          curve(gc, _mmm(gc["pod_curve"][-1]), HERM_C, "s", "above", True, kind="cross")]

    # ---- BOTTOM-RIGHT: 8D field error (value + value+gradient cross) ----
    # BR_8D_ENHANCED selects which enhanced model this panel plots; the guard requires
    # it to be the same model the TOP-RIGHT residual panel plots.
    gvf8 = load_source("gvm_8d_field")
    gvhf8 = load_source(BR_8D_ENHANCED)
    _check_8d_enhanced(gx8, gvhf8)
    BR = [curve(gvf8, _mmm(gvf8["pod_curve"][-1]), VAL_C, "o", "below", True, kind="value"),
          curve(gvhf8, _mmm(gvhf8["pod_curve"][-1]), HERM_C, "s", "above", True,
                kind="cross")]

    panels = {
        # panel titles + curve labels live in the plot script (presentation, not data).
        # The 1000-point sample size is stated once in the caption, so no in-panel note.
        "TL": dict(curves=TL),
        "TR": dict(curves=TR),
        "BL": dict(title=None, curves=BL),
        "BR": dict(title=None, curves=BR),
    }
    p = dump("fig05_guess_vs_memory", dict(panels=panels, meta=_meta(gh, gx8)))
    print(f"wrote {os.path.relpath(p)}  (4 panels x value+gradient+cross)")
    print(f"  {pm.describe(4)}\n  {pm.describe(8)}")


if __name__ == "__main__":
    build()
