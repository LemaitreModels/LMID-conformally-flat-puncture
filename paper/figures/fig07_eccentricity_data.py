#!/usr/bin/env python
"""Data for fig07_eccentricity: precompute the smooth E_b(b;J) curves to figdata/.

fig08 is the ONLY figure whose plotter used to import jax + the parasol package and evaluate a
surrogate model (surrogate_bpt_ecc.npz) at plot time. This script does that evaluation ONCE and
writes the smooth curves + the certified scan points + the gradient minima as plain arrays to
figdata/fig07_eccentricity.json, so the plotter (and every other figure) is pure-data.

Sources (raw):
  reports/P3/qc_effpot_Jsweep.json                          (key "qc_effpot")   — scan + minima
  reports/3D_parametric/models/surrogate_bpt_ecc.npz        (key "effpot_model") — for the curves
Needs the BBHFM env (jax + parasol); no solves are run.

Run:  python fig07_eccentricity_data.py
"""
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

from _figdata import load_source, source, dump
from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.applications import qc_effpot as E


def build():
    d = load_source("qc_effpot")
    Jlist = [float(J) for J in d["Jlist"]]
    n_scan = int(d["n_scan"])
    n_grad = int(d["per_J"][f"{Jlist[0]:.2f}"]["solves_gradient"])

    prob = s3.make_problem(Na=44, Nb=32, Nphi=8)
    model = E.load_model(source("effpot_model"), prob)
    V, _ = E.build_effpot_jax(model, prob)

    # global background grid = full scan-b span (same as the old plotter)
    b_all = np.concatenate([np.asarray(d["per_J"][f"{J:.2f}"]["scan_curve"]["b"], float)
                            for J in Jlist])
    bg = np.linspace(float(b_all.min()), float(b_all.max()), 240)

    per = {}
    for J in Jlist:
        pj = d["per_J"][f"{J:.2f}"]
        bc = float(pj["b_circ_gradient"])
        per[f"{J:.2f}"] = dict(
            scan_b=np.asarray(pj["scan_curve"]["b"], float),
            scan_Eb=np.asarray(pj["scan_curve"]["Eb"], float),
            b_circ=bc,
            Vg=[float(V(b, J)) for b in bg],   # smooth surrogate curve on bg
            Vc=float(V(bc, J)),                # surrogate value at the gradient minimum
        )
    p = dump("fig07_eccentricity",
             dict(Jlist=Jlist, n_scan=n_scan, n_grad=n_grad, bg=bg, per_J=per))
    print(f"wrote {os.path.relpath(p, SANDBOX)}  ({len(Jlist)} J-slices, |bg|={len(bg)})")


if __name__ == "__main__":
    build()
