#!/usr/bin/env python
"""Data for fig10_tp_box_sample: distill the plotted arrays to figdata/.

Source (raw): reports/3D_parametric/qc/tp_random_sweep.json (key ``tp_random_sweep``,
produced by ``run_tp_random_sweep.py``).

Keeps, per configuration, only what the figure draws: the two agreement metrics
(pointwise psi and the integral ADM mass), the certified residual, and the three box
coordinates the tail is attributed to (b, q, max|chi|).  The interior sample and the
edge stress set are kept SEPARATE -- the interior distribution is the unbiased estimate,
the edge set is the deliberate worst case -- together with the oracle self-convergence
floor, which is the smallest difference the comparison can resolve.

Run:  python fig10_tp_box_sample_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump

KEEP = ("b", "q", "chi_absmax", "max_dpsi", "M_ADM_rel_diff", "residual")


def _rows(rows, edge):
    out = []
    for r in rows:
        if not r.get("ok"):
            continue
        if (r.get("label") is not None) != edge:
            continue
        d = {k: r[k] for k in KEEP}
        if edge:
            d["label"] = r["label"]
        out.append(d)
    return out


def build():
    R = load_source("tp_random_sweep")
    m, s = R["meta"], R["summary"]
    p = dump("fig10_tp_box_sample", dict(
        interior=_rows(R["rows"], edge=False),
        edge=_rows(R["rows"], edge=True),
        summary=s,
        meta=dict(n=m["n"], sampler=m["sampler"], seed=m["seed"], grid=m["grid"],
                  tp_res=m["tp_res"], cert_tol=m["cert_tol"], box=m["box"],
                  axes=m["axes"]),
        oracle_floor=s.get("tp_selfconv_dpsi_max")))
    print(f"wrote {os.path.relpath(p)}  "
          f"({len(_rows(R['rows'], False))} interior, {len(_rows(R['rows'], True))} edge)")


if __name__ == "__main__":
    build()
