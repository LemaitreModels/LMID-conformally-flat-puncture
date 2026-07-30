#!/usr/bin/env python
"""Data for fig09_tp_validation: distill the plotted arrays to figdata/.

Consolidated TwoPunctures validation figure (merges the former 3-D angular-momentum
and quasi-circular-validation figures). Two raw sources:
  * "sweep_3d"     (reports/3D/sweep_results.json)          -> ADM-J tilt vs spin tilt + TP anchors
  * "tp_validation"(reports/3D_parametric/qc/tp_validation_qc.json) -> psi/M_ADM vs grid

Keeps only the panels the consolidated figure draws: the ADM-J spin-tilt collapse
(with TwoPunctures anchors), and the quasi-circular psi-vs-TP and M_ADM-vs-TP
convergence. The orbital-J-recovery and Newtonian-momentum panels are dropped
(their results are stated in the text).

Run:  python fig09_tp_validation_data.py
"""
import os
import sys

import numpy as np

from _figdata import load_source, dump


def build():
    # --- ADM-J: tilt-vs-spin-tilt curves per |S| + TwoPunctures anchors (sweep_3d) ---
    R = load_source("sweep_3d")
    rows = R["A_spin_grid"]["rows"]
    S_mags = R["A_spin_grid"]["S_mags"]
    panelA = []
    for mag in S_mags:
        sel = sorted([r for r in rows if r["b"] == 1.5 and r["S_mag"] == mag],
                     key=lambda r: r["tilt_deg"])
        panelA.append(dict(S_mag=mag, ts=[r["tilt_deg"] for r in sel],
                           jt=[r["J_tilt_deg"] for r in sel]))

    D = R.get("D_anchors", {})
    available = bool(D.get("available"))
    anchors = []
    if available:
        for a in D["anchors"]:
            S = np.array(a["S_A"])
            if np.hypot(S[0], S[1]) > 0 or S[2] != 0:
                tS = float(np.rad2deg(np.arctan2(np.hypot(S[0], S[1]), S[2]))) if np.any(S) else 0.0
                # legacy corpora (pre-rename) store this as "J_tp_parasol"
                Jtp = np.array(a.get("J_tp_lm_initial_data", a.get("J_tp_parasol")))
                tJ = float(np.rad2deg(np.arctan2(np.hypot(Jtp[0], Jtp[1]), Jtp[2])))
                anchors.append([tS, tJ])

    # --- quasi-circular psi/M_ADM vs LM-initial-data grid (tp_validation) ---
    d = load_source("tp_validation")
    C = [dict(Na=r["Na"], Nb=r["Nb"], max_dpsi=r["max_dpsi"],
              M_ADM_rel_diff=r["M_ADM_rel_diff"]) for r in d["C_psi_adm"]["grids"]]

    p = dump("fig09_tp_validation",
             dict(panelA=panelA, anchors=anchors, anchors_available=available,
                  C_psi_adm=C))
    print(f"wrote {os.path.relpath(p)}  ({len(panelA)} |S|, {len(anchors)} anchors, {len(C)} grids)")


if __name__ == "__main__":
    build()
