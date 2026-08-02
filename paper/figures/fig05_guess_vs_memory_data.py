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

Sources (raw): gvm_all, gvm_4d_{value,cross,field,cross_field}, gvm_8d_{value,field,hermite_field},
               polish_table_{4d,4d_cross,8d_hermite,8d_value}  (registry keys).

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
    gvf8 = load_source("gvm_8d_field")
    gvhf8 = load_source("gvm_8d_hermite_field")
    BR = [curve(gvf8["pod_curve"], _mmm(gvf8["pod_curve"][-1]),
                8.0 * gvf8["N"] * gvf8["nfeat"] / 1e6,
                VAL_C, "o", gvf8["r_full"], "below", True),
          curve(gvhf8["pod_curve"], _mmm(gvhf8["pod_curve"][-1]), gvhf8["bare_mem_bytes"] / 1e6,
                HERM_C, "s", gvhf8["r_full"], "above", True)]

    panels = {
        # panel titles + curve labels live in the plot script (presentation, not data)
        "TL": dict(note="1000 off-node pts", note_loc="bl", curves=TL),
        "TR": dict(note="1000 off-node pts", note_loc="br", curves=TR),
        "BL": dict(title=None, note="1000 off-node pts", note_loc="tr", curves=BL),
        "BR": dict(title=None, note="1000 off-node pts", note_loc="tr", curves=BR),
    }
    p = dump("fig05_guess_vs_memory", dict(panels=panels))
    print(f"wrote {os.path.relpath(p)}  (4 panels x value+gradient+cross)")


if __name__ == "__main__":
    build()
