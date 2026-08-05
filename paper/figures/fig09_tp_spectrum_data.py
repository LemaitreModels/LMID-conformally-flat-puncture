#!/usr/bin/env python
"""Data for fig09_tp_spectrum: distill the azimuthal-mode band to figdata/.

Source (raw): reports/3D_parametric/qc/tp_band_sweep.json (key ``tp_band_sweep``,
produced by ``run_tp_random_sweep.py``) -- the same sweep that feeds fig08, distilled
separately because the two figures share no abscissa: fig08 walks the meridional grid,
this one walks the azimuthal mode index at the best-resolved rung.

The quasi-circular data are non-axisymmetric by construction -- the tangential momentum
populates m=2 and generic spins populate m=1 -- so this band is what sets the azimuthal
truncation N_phi the production solves need.

Keeps two blocks:
  * ``spectrum`` per azimuthal mode m at the best-resolved rung: the |u_m|/|u_0| band;
  * ``meta``     sample size, the rung it was measured at, the box/sampler provenance, and
                 the aligned-spin head-on worst case (``axisym_m_ge1_max``), a text number
                 quoted in the appendix rather than plotted.

Run:  python fig09_tp_spectrum_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump

BAND = ("min", "median", "mean", "max")


def build():
    R = load_source("tp_band_sweep")
    m, s = R["meta"], R["summary"]
    I = s["interior"]

    spectrum = [dict(m=r["m"], **{k: r[k] for k in BAND}) for r in I["spectrum_top"]]

    p = dump("fig09_tp_spectrum", dict(
        spectrum=spectrum,
        meta=dict(n_interior=I["n"], ladder=m["ladder"], top_grid=m["ladder"][-1],
                  box=m["box"], sampler=m["sampler"], seed=m["seed"],
                  axisym_m_ge1_max=(s.get("axisym") or {}).get("m_ge1_max"))))
    print(f"wrote {os.path.relpath(p)}  ({len(spectrum)} modes at "
          f"{m['ladder'][-1]}, n_interior={I['n']})")


if __name__ == "__main__":
    build()
