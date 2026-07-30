#!/usr/bin/env python
"""Data for fig08_3d_validation: distill the plotted arrays to figdata/.

Source (raw): reports/3D/sweep_results.json  (key "sweep_3d"; SHARED with fig10, which distills
the angular-momentum blocks of the same file). Keeps the convergence ladder + the selected
phi-mode spectrum rows (|S|=0.3, b=1.5).

Run:  python fig08_3d_validation_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump


def build():
    R = load_source("sweep_3d")
    B = R["B_convergence"]["ladder"]
    ladder = dict(Na=[r["Na"] for r in B], resid=[r["resid"] for r in B],
                  dpsi=[r["dpsi_vs_TP"] for r in B])          # None kept as null
    sel = [r for r in R["A_spin_grid"]["rows"] if r["b"] == 1.5 and r["S_mag"] == 0.3]
    spectrum = [dict(tilt_deg=r["tilt_deg"], m_vals=r["m_vals"], phi_amps=r["phi_amps"])
                for r in sorted(sel, key=lambda r: r["tilt_deg"])]
    p = dump("fig08_3d_validation", dict(ladder=ladder, spectrum=spectrum))
    print(f"wrote {os.path.relpath(p)}  ({len(ladder['Na'])} ladder pts, {len(spectrum)} tilts)")


if __name__ == "__main__":
    build()
