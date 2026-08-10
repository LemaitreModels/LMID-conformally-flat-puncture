#!/usr/bin/env python
"""Data for fig08_tp_validation: distill the resolution-ladder bands to figdata/.

Source (raw): reports/3D_parametric/qc/tp_band_sweep.json (key ``tp_band_sweep``,
produced by ``run_tp_random_sweep.py``).

This is the RESOLUTION-LADDER half of the TwoPunctures validation: how the agreement with
the oracle behaves as the meridional grid is refined, as a min/median/max distribution over
configurations drawn from the production box, so no panel depends on an arbitrary parameter
point.  The AZIMUTHAL-SPECTRUM half of the same sweep (which has a different abscissa -- the
mode index, not the grid -- and answers a different question, how many phi modes the data
need) is distilled separately by ``fig09_tp_spectrum_data.py`` from this same source.

Keeps three blocks:
  * ``ladder``   per rung: the psi, ADM-mass and certified-residual bands + how many
                 configurations fail the certification gate at that rung;
  * ``edge``     the same bands for the deliberate box-edge stress set, kept out of the
                 interior statistics;
  * ``meta``     sample size, ladder, oracle resolution and its self-convergence floor,
                 the axisymmetric code-to-code anchor, and the axisymmetry-check worst case
                 (both text numbers, not plotted).

Run:  python fig08_tp_validation_data.py
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

    # the edge stress set is carried as a band too, so the plotter can mark the deliberate
    # worst case without it contaminating the interior estimate
    E = s.get("edge") or {}
    edge = [dict(grid=r["grid"], psi=_band(r["psi"]), M_ADM=_band(r["M_ADM"]),
                 residual=_band(r["residual"]))
            for r in (E.get("ladder") or [])]

    p = dump("fig08_tp_validation", dict(
        ladder=ladder, edge=edge,
        meta=dict(n_interior=I["n"], n_edge=E.get("n", 0), ladder=m["ladder"],
                  tp_res=m["tp_res"], cert_tol=m["cert_tol"], box=m["box"],
                  sampler=m["sampler"], seed=m["seed"],
                  oracle_floor=s.get("tp_selfconv_dpsi_max"),
                  axisym_m_ge1_max=(s.get("axisym") or {}).get("m_ge1_max"),
                  anchor=s.get("anchor"),          # the axisymmetric code-to-code reference
                  n_failed=s.get("n_failed", 0))))
    print(f"wrote {os.path.relpath(p)}  ({len(ladder)} rungs, "
          f"n_interior={I['n']}, n_edge={E.get('n', 0)})")


if __name__ == "__main__":
    build()
