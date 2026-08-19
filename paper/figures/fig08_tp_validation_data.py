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


# The probe set behind ``max_dpsi``.  ``run_tp_random_sweep._probe_points`` builds
# ``6 + N_PROBE_DENSE`` points (the six legacy probes of the predecessor, kept as a
# reproducible subset, plus 64 well-spread ones), i.e. 70 -- see that module's
# ll. 211-235.  Recorded here rather than imported because the figure tier is
# deliberately jax-free, and read out of the code rather than remembered because the
# paper quoted 69.  The raw artifact predates the producer emitting it; the durable
# fix is for ``run_tp_random_sweep`` to write ``n_probe`` into its own meta, which
# would catch a drift at build time instead of at review time.
N_PROBE_LEGACY = 6
N_PROBE_DENSE = 64
N_PROBE = N_PROBE_LEGACY + N_PROBE_DENSE


def _n_nonmonotone(rows, value):
    """Configurations whose refinement sequence is not monotonically decreasing.

    ``value`` maps one rung dict (plus its row) to the statistic being refined.
    The count is statistic-SPECIFIC and definition-specific -- that is the whole
    reason it is recorded per statistic here rather than as one number: the raw
    ``M_ADM`` is non-monotone for 75 of the 100 interior configurations, while its
    *error against the oracle* ``|M_ADM - tp_E|`` is non-monotone for 32.  Quoting
    "the ADM mass is non-monotone for N" without saying which is meaningless.
    """
    n = 0
    for r in rows:
        v = [value(g, r) for g in r["rungs"]]
        v = [x for x in v if x is not None]
        if len(v) > 1 and any(v[i + 1] > v[i] for i in range(len(v) - 1)):
            n += 1
    return n


def _max_nonmonotone_steps(rows, value):
    """Worst number of upward steps inside a single configuration's sequence."""
    best = 0
    for r in rows:
        v = [value(g, r) for g in r["rungs"]]
        best = max(best, sum(1 for i in range(len(v) - 1) if v[i + 1] > v[i]))
    return best


def build():
    R = load_source("tp_band_sweep")
    m, s = R["meta"], R["summary"]
    I = s["interior"]
    # the interior sample the bands are built from, per configuration
    interior_rows = [r for r in R["rows"] if r.get("ok") and r.get("label") is None]

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
                  # RENAMED from ``oracle_floor``: this is the oracle's azimuthal
                  # self-convergence floor specifically, and the bare name invited
                  # reading it as a floor on the whole comparison.
                  oracle_floor_azimuthal=s.get("tp_selfconv_dpsi_max"),
                  # how far the closest measured agreement sits ABOVE that floor
                  # (min over rungs of the psi band minimum / the floor): the
                  # comparison is floor-limited only if this approaches 1.
                  oracle_floor_margin_min=min(
                      r["psi"]["min"] / s["tp_selfconv_dpsi_max"]
                      for r in I["ladder"]),
                  n_probe=N_PROBE, n_probe_legacy=N_PROBE_LEGACY,
                  # named by statistic; see _n_nonmonotone
                  n_nonmonotone_psi=_n_nonmonotone(
                      interior_rows, lambda g, r: g["max_dpsi"]),
                  n_nonmonotone_psi_legacy6=_n_nonmonotone(
                      interior_rows, lambda g, r: g["max_dpsi_legacy6"]),
                  n_nonmonotone_l2=_n_nonmonotone(
                      interior_rows, lambda g, r: g["l2_dpsi"]),
                  n_nonmonotone_M_ADM=_n_nonmonotone(
                      interior_rows, lambda g, r: abs(g["M_ADM"] - r["tp_E"])),
                  n_nonmonotone_M_ADM_raw=_n_nonmonotone(
                      interior_rows, lambda g, r: g["M_ADM"]),
                  n_nonmonotone_max_steps=_max_nonmonotone_steps(
                      interior_rows, lambda g, r: g["max_dpsi"]),
                  axisym_m_ge1_max=(s.get("axisym") or {}).get("m_ge1_max"),
                  anchor=s.get("anchor"),          # the axisymmetric code-to-code reference
                  n_failed=s.get("n_failed", 0))))
    print(f"wrote {os.path.relpath(p)}  ({len(ladder)} rungs, "
          f"n_interior={I['n']}, n_edge={E.get('n', 0)})")


if __name__ == "__main__":
    build()
