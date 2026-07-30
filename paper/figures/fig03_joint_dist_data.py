#!/usr/bin/env python
"""Data for fig03_joint_dist: distill the best/median/worst curves to figdata/.

Sources (raw):
  joint_dist_4d          reports/3D_parametric/qc_chi/joint_dist_d4_qc_chi_b27.json        (4D bare)
  joint_dist_cross_4d    reports/3D_parametric/qc_chi/joint_dist_cross_d4_qc_chi_b27.json  (4D value+grad+cross)
  joint_dist_8d          reports/3D_parametric/qc_chi/joint_dist_spin8_qc_chi_b27.json     (8D bare)
  joint_dist_hermite_8d  reports/3D_parametric/qc_chi/joint_dist_hermite_spin8_qc_chi_b27.json (8D value+grad+cross)

Both 8D files share the 4D schema (a "joint" list of {level, nodes, best, median, worst}), so the
8D right panel mirrors the 4D left panel exactly.

Run:  python fig03_joint_dist_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump


def _series(key):
    r = load_source(key)
    J = r["joint"]
    return dict(nodes=[j["nodes"] for j in J], best=[j["best"] for j in J],
                med=[j["median"] for j in J], worst=[j["worst"] for j in J],
                levels=[j["level"] for j in J], n_points=r["meta"]["n_points"])


def build():
    out = dict(
        left=dict(bare=_series("joint_dist_4d"), cross=_series("joint_dist_cross_4d")),
        right=dict(bare=_series("joint_dist_8d"), cross=_series("joint_dist_hermite_8d")),
    )
    p = dump("fig03_joint_dist", out)
    print(f"wrote {os.path.relpath(p)}  (4D + 8D, {len(out['left']['bare']['levels'])} levels)")


if __name__ == "__main__":
    build()
