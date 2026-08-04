#!/usr/bin/env python
"""Data for fig09_tp_validation: distill the plotted bands to figdata/.

Source (raw): reports/3D_parametric/qc/tp_band_sweep.json (key ``tp_band_sweep``,
produced by ``run_tp_random_sweep.py``).

This figure was two figures and one configuration each: a misaligned-spin head-on slice at
b=1.5 (the former fig08) and a nonspinning equal-mass quasi-circular slice at b=4.  It is
now ONE figure whose every curve is a min/median/max band over configurations drawn from
the production box, so no panel depends on an arbitrary parameter point.  The separate
non-axisymmetric figure was dropped because the quasi-circular data are already
non-axisymmetric -- measured, ~2% of the field sits in m=2 from the tangential momentum and
generic spins add ~2% at m=1 -- so the QC family exercises the Fourier-in-phi solver by
itself; the ``spectrum`` block below is what replaces that figure's spectrum panel.

Keeps three blocks:
  * ``ladder``   per rung: the psi, ADM-mass and certified-residual bands + how many
                 configurations fail the certification gate at that rung;
  * ``spectrum`` per azimuthal mode m at the best-resolved rung: the |u_m|/|u_0| band;
  * ``meta``     sample size, ladder, oracle resolution and its self-convergence floor,
                 and the axisymmetry-check worst case (a text number, not plotted).

Run:  python fig09_tp_validation_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import load_source, dump

BAND = ("min", "median", "mean", "max")


def _band(d):
    return {k: d[k] for k in BAND} if d else None


def build():
    R = load_source("tp_band_sweep")
    m, s = R["meta"], R["summary"]
    I = s["interior"]

    # psi_l2 and psi_legacy6 are carried for provenance, not plotted: the L2 is the stabler
    # statistic, and the six-probe estimate is what the predecessor reported, so keeping both
    # in the figdata makes the probe-density effect auditable from the committed artifact.
    ladder = [dict(grid=r["grid"], psi=_band(r["psi"]), M_ADM=_band(r["M_ADM"]),
                   residual=_band(r["residual"]), n_uncertified=r["n_uncertified"],
                   psi_l2=_band(r.get("psi_l2")), psi_legacy6=_band(r.get("psi_legacy6")))
              for r in I["ladder"]]
    spectrum = [dict(m=r["m"], **{k: r[k] for k in BAND}) for r in I["spectrum_top"]]

    # the edge stress set is carried as a band too, so the plotter can mark the deliberate
    # worst case without it contaminating the interior estimate
    E = s.get("edge") or {}
    edge = [dict(grid=r["grid"], psi=_band(r["psi"]), M_ADM=_band(r["M_ADM"]),
                 residual=_band(r["residual"]))
            for r in (E.get("ladder") or [])]

    p = dump("fig09_tp_validation", dict(
        ladder=ladder, spectrum=spectrum, edge=edge,
        meta=dict(n_interior=I["n"], n_edge=E.get("n", 0), ladder=m["ladder"],
                  tp_res=m["tp_res"], cert_tol=m["cert_tol"], box=m["box"],
                  sampler=m["sampler"], seed=m["seed"],
                  oracle_floor=s.get("tp_selfconv_dpsi_max"),
                  axisym_m_ge1_max=(s.get("axisym") or {}).get("m_ge1_max"),
                  anchor=s.get("anchor"),          # the axisymmetric code-to-code reference
                  n_failed=s.get("n_failed", 0))))
    print(f"wrote {os.path.relpath(p)}  ({len(ladder)} rungs, {len(spectrum)} modes, "
          f"n_interior={I['n']}, n_edge={E.get('n', 0)})")


if __name__ == "__main__":
    build()
