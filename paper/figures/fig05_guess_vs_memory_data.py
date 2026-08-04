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

Memory formulas (float64 stored arrays), matching the run drivers:
  value POD bare   = 8 * N * nfeat                          (node_U only)
  value+grad bare  = 8 * N * (1+d+npair) * nfeat            (node_U + d tangents + npair cross)
  POD rank-r point = each pod_curve entry's own mem_bytes.

Run:  python fig05_guess_vs_memory_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump

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


def _pod5(cur):
    return [{k: c[k] for k in POD5} for c in _thin(cur)]


def _mmm(rec):
    return {k: rec[k] for k in ("median", "min", "max")}


def _guess(key):
    """bare-guess {median,min,max} from a polish_table json (rows.guess)."""
    return _mmm(load_source(key)["rows"]["guess"])


def curve(cur, bare, bare_mem_mb, color, marker, bare_r, ann, drop_last):
    return dict(cur=_pod5(cur), bare=bare, bare_mem=bare_mem_mb, bare_r=bare_r,
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


def build():
    # ---- TOP-LEFT: 4D residual (value + value+gradient cross) ----
    gv = load_source("gvm_4d_value")
    gh = load_source("gvm_4d_cross")
    N, nfeat, d = gh["N"], gh["nfeat"], gh["d"]
    npair = gh.get("npair", 1)
    TL = [curve(gv["pod_curve"], _guess("polish_table_4d"), 8.0 * N * nfeat / 1e6,
                VAL_C, "o", gv["r_full"], "below", True),
          curve(gh["pod_curve"], _guess("polish_table_4d_cross"),
                8.0 * N * (1 + d + npair) * nfeat / 1e6,
                HERM_C, "s", gh["r_full"], "above", True)]

    # ---- TOP-RIGHT: 8D residual (value gapfill + value+gradient y-pair-CROSS gapfill) ----
    # Both to r_full over the SAME 1000 seed-0 points; the value+gradient curve is the RESIDUAL
    # sibling of the bottom-right field sweep (same y-pair cross model, same ranks/memory), so the
    # two 8D value+gradient curves share identical x-positions (model, ranks, bare_mem).
    gv8 = load_source("gvm_8d_value")
    gx8 = load_source("gvm_8d_cross")
    TR = [curve(gv8["pod_curve"], _guess("polish_table_8d_value"),
                8.0 * gv8["N"] * gv8["nfeat"] / 1e6,
                VAL_C, "o", gv8["r_full"], "below", True),
          curve(gx8["pod_curve"], _mmm(gx8["pod_curve"][-1]), gx8["bare_mem_bytes"] / 1e6,
                HERM_C, "s", gx8["r_full"], "above", True)]

    # ---- BOTTOM-LEFT: 4D field error (value + value+gradient cross) ----
    # Flat schema, matching the 8-D siblings below.  Was ["value"]: a leftover
    # from when run_guess_vs_memory wrote ONE combined two-flavour file; the
    # --flavours path now writes one flat file per flavour, so gvm_4d_field has
    # pod_curve/N/nfeat/r_full at top level exactly like gvm_8d_field.
    fv = load_source("gvm_4d_field")
    gc = load_source("gvm_4d_cross_field")
    BL = [curve(fv["pod_curve"], _mmm(fv["pod_curve"][-1]), 8.0 * fv["N"] * fv["nfeat"] / 1e6,
                VAL_C, "o", fv["r_full"], "below", True),
          curve(gc["pod_curve"], _mmm(gc["pod_curve"][-1]), gc["bare_mem_bytes"] / 1e6,
                HERM_C, "s", gc["r_full"], "above", True)]

    # ---- BOTTOM-RIGHT: 8D field error (value + value+gradient cross) ----
    # BR_8D_ENHANCED selects which enhanced model this panel plots; the guard requires
    # it to be the same model the TOP-RIGHT residual panel plots.
    gvf8 = load_source("gvm_8d_field")
    gvhf8 = load_source(BR_8D_ENHANCED)
    _check_8d_enhanced(gx8, gvhf8)
    BR = [curve(gvf8["pod_curve"], _mmm(gvf8["pod_curve"][-1]),
                8.0 * gvf8["N"] * gvf8["nfeat"] / 1e6,
                VAL_C, "o", gvf8["r_full"], "below", True),
          curve(gvhf8["pod_curve"], _mmm(gvhf8["pod_curve"][-1]), gvhf8["bare_mem_bytes"] / 1e6,
                HERM_C, "s", gvhf8["r_full"], "above", True)]

    panels = {
        # panel titles + curve labels live in the plot script (presentation, not data).
        # The 1000-point sample size is stated once in the caption, so no in-panel note.
        "TL": dict(curves=TL),
        "TR": dict(curves=TR),
        "BL": dict(title=None, curves=BL),
        "BR": dict(title=None, curves=BR),
    }
    p = dump("fig05_guess_vs_memory", dict(panels=panels))
    print(f"wrote {os.path.relpath(p)}  (4 panels x value+gradient+cross)")


if __name__ == "__main__":
    build()
