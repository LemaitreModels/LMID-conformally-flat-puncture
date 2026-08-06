#!/usr/bin/env python
"""Data for fig04_polish_staircase: distill the per-Newton-step staircases to figdata/.

Two columns (4D | 8D), two rows (constraint residual | field error), each with a cold-start and a
value+gradient POD-warm-start curve (median + min/max over 1000 off-node points per step).

Each panel additionally carries a value-ONLY model curve, so the three warm-start families
cold | value | value+gradient are compared on both rows.  The value-only curve is built on the
SAME cross-POD basis Phi[:, :r] at the SAME rank as the value+gradient POD curve, differing only
in whether the coefficient interpolant uses the certified tangents -- which is what makes the
pair apples-to-apples (and what the caption claims).  If that cluster source is absent it falls
back to the full un-compressed value-only Smolyak model.

All six POD curves share ONE model family: the y-pair full-bilinear CROSS POD (the model the
paper ships, cf. sec:model:enhanced / fig:joint), at r=250 in 4D and r=500 in 8D -- the rank at
which fig05's ladder shows the compression is no longer the limiting error.  Two mismatches were
removed in that revision: the 8D residual row previously used the SIX-axis non-cross POD while
its own field-error row used the cross POD, and both value-only curves were built on the
non-cross basis (matching the value+gradient curve in rank but not in basis).

Sources (raw), per dimension X in {4,8} with rank R = 250 (4D) / 500 (8D):
  polish_cold_Xd            P3/polish_cold_chiXd_1000.json                    (cold residual staircase)
  polish_pod_Xd             P3/polish_table_chiXd_pod_rR_cross_1000.json      (value+grad POD residual)
  polish_value_pod_Xd       P3/polish_fielderr_value_pod_chiXd_rR_1000.json   (value-only POD; both rows)
  polish_table_{4d,8d_value} P3/polish_table_{qc_chi_prod,chi8d_value}_1000.json  (value-only residual, fallback)
  polish_fielderr_Xd        P3/polish_fielderr_chiXd_rR_1000.json             (cold+POD field-error stairs)
  polish_fielderr_value_Xd  P3/polish_fielderr_value_chiXd_1000.json          (value-only field-error, fallback)

The value-only POD source carries both rows (residual_rows + field_rows from run_family); when it is
absent the value curve falls back to the two full-value tables.  The 8D field-error file shares the
4D schema, so the bottom-right panel mirrors the bottom-left.

Run:  python fig04_polish_staircase_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, have_source, dump
from lm.initial_data.pipeline import production_model as pm


def _stair(d):
    """residual staircase from a polish_cold / polish_table json (rows.{guess,after k})."""
    K = int(d["config"]["max_steps"])
    keys = ["guess"] + [f"after{k}" for k in range(1, K + 1)]
    return dict(K=K, med=[d["rows"][k]["median"] for k in keys],
                lo=[d["rows"][k]["min"] for k in keys], hi=[d["rows"][k]["max"] for k in keys])


def _field(fam):
    """field-error staircase from a polish_fielderr family (cold/pod -> field_rows.{...})."""
    K = int(fam["max_steps"])
    keys = ["guess"] + [f"after{k}" for k in range(1, K + 1)]
    return dict(K=K, med=[fam["field_rows"][k]["median"] for k in keys],
                lo=[fam["field_rows"][k]["min"] for k in keys],
                hi=[fam["field_rows"][k]["max"] for k in keys])


def _rows_stair(rows, K):
    """staircase from a run_polish_fielderr ``run_family`` rows dict (residual_rows /
    field_rows), keyed guess/after1.. -> {median,min,max}."""
    keys = ["guess"] + [f"after{k}" for k in range(1, K + 1)]
    return dict(K=K, med=[rows[k]["median"] for k in keys],
                lo=[rows[k]["min"] for k in keys], hi=[rows[k]["max"] for k in keys])


VALUE_TABLE = {4: "polish_table_4d", 8: "polish_table_8d_value"}  # value-only residual staircases
# value-ONLY *POD* warm start (same shipped basis + rank as the value+gradient POD), when present
VALUE_POD = {4: "polish_value_pod_4d", 8: "polish_value_pod_8d"}


def _value_curves(dim):
    """(res_value, fld_value): the value-only model warm-start staircases.  Prefers the
    value-only *POD* family (run_polish_fielderr_value_pod, at the same rank as the
    value+gradient POD curve) when its cluster source is present, so the two parametric-model
    curves are apples-to-apples; falls back to the full un-compressed value-only Smolyak
    tables until that source lands (keeps the figure buildable today)."""
    key = VALUE_POD[dim]
    if have_source(key):
        d = load_source(key)
        fam = d["value_pod"]
        K = int(fam["max_steps"])
        res = _rows_stair(fam["residual_rows"], K)
        res["r"] = d["config"].get("r")            # POD rank -> plot label "value-only POD (r=..)"
        fld = _rows_stair(fam["field_rows"], K)
        return res, fld
    # fallback: full (un-compressed) value-only Smolyak model
    return _stair(load_source(VALUE_TABLE[dim])), _field(load_source(f"polish_fielderr_value_{dim}d")["value"])


def _check_rank(dim, r):
    """The POD curves must be at the SHIPPED rank.

    A staircase measured at some other rank is a different model from the one the
    caption and Sec. sec:model:pod describe -- the failure mode that put r=76/359
    numbers in the paper while the figure showed r=250/500.  production_model is the
    only place the rank is defined.
    """
    want = pm.SHIPPED_RANK[dim]
    if r is not None and int(r) != want:
        raise SystemExit(
            f"fig04: the {dim}-D POD staircase is at r={int(r)} but the shipped rank is "
            f"{want} (production_model.SHIPPED_RANK). Either rebuild the sweep at the "
            f"shipped rank, or change SHIPPED_RANK if the model really moved.")


def build():
    cols = []
    for dim in (4, 8):
        pod = load_source(f"polish_pod_{dim}d")
        fe = load_source(f"polish_fielderr_{dim}d")
        res_pod = _stair(pod)
        res_pod["r"] = pod["config"].get("r")
        _check_rank(dim, res_pod["r"])
        res_value, fld_value = _value_curves(dim)
        cols.append(dict(
            dim=dim,   # panel title lives in the plot script (presentation, not data)
            res_cold=_stair(load_source(f"polish_cold_{dim}d")),
            res_value=res_value,
            res_pod=res_pod,
            fld_cold=_field(fe["cold"]),
            fld_value=fld_value,
            fld_pod=_field(fe["pod"]),
        ))
    p = dump("fig04_polish_staircase", dict(cols=cols))
    print(f"wrote {os.path.relpath(p)}  (4D + 8D, residual + field rows, cold|value|value+grad)")


if __name__ == "__main__":
    build()
