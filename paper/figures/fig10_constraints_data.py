#!/usr/bin/env python
"""Data for fig10_constraints: FD constraint violation on an evolution grid.

RECOMPUTE figure — this is the one data script with no ``reports/`` source.  It solves the
initial data itself and measures the constraints, because both steps are cheap (one
axisymmetric two-centre solve plus four Cartesian evaluations, seconds each); the only slow
input is the external oracle.

What it measures (App. A of the paper, ``sec:validation:constraints``).  The spectral
conformal factor is interpolated onto a uniform Cartesian grid, as an evolution code would
read it, and the Hamiltonian and momentum constraints are evaluated by
``validation.constraints.fd_constraints_generic`` — a GENERIC second-order finite-difference
monitor that builds the Christoffel symbols and the Ricci tensor from the nodal metric and
makes no use of conformal flatness.  The measurement is therefore independent of the spectral
discretization that produced the data.

Two curves per constraint, on the SAME Cartesian grids:
  * ``lm``  — psi from this package's solver;
  * ``tp``  — psi from TwoPunctures at the same physical points (the external oracle).
Both feed the identical monitor, so the comparison isolates the initial data: if the two
curves coincide, the measured violation is the monitor's truncation error rather than a
property of either solution.

Configuration: the axisymmetric anchor of App.~A, equal masses at half-separation ``b=3``
with axial momentum ``P=0.5`` — the configuration the preceding subsection compares against
TwoPunctures, and the one with the fewest confounds.

Cost.  The solver leg is memory-bound and cheap: ~1.5 GB and ~7 s per million Cartesian
points, so the production ladder is ~25 GB and ~4 min.  The oracle leg dominates: the
binary is queried at every Cartesian point of every rung (~26M points, ~1.8 ms each,
~13 h serial).  It is embarrassingly parallel over chunks -- ``--workers 32`` brings the
whole build to well under an hour, which is what the cluster wrapper
(``slurm/ivs/submit_fig10_constraints.slurm``) runs.  ``--no-tp`` drops it entirely and
leaves the solver curve alone, in minutes, on a laptop.

Run:  python fig10_constraints_data.py --workers 32          # production (cluster)
      python fig10_constraints_data.py --no-tp               # solver curve only
      python fig10_constraints_data.py --n-list 40,56,72,88  # the coarse ladder only
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figdata import dump

from lm.initial_data.solver import solver_abt as sa
from lm.initial_data.validation import constraints as cst
from lm.initial_data.validation import twopunctures as tp

# --- the configuration (the App. A axisymmetric anchor) --------------------------------------
B, M_A, M_B, P = 3.0, 0.5, 0.5, 0.5
SPECTRAL = (52, 36)          # (N_A, N_B) meridional grid of the two-centre solve
NEWTON_TOL = 1e-12
# Cartesian rungs.  The ladder spans 6.5x in h (0.46 -> 0.071 M) and stays in the asymptotic
# second-order regime throughout -- verified: the local pairwise orders scatter about 2 with no
# sign of a floor.  It cannot reach the finest level of an evolution's AMR hierarchy
# (h ~ m/25 ~ 0.02 M): the punctures sit at z = +-3, so a uniform box wide enough to hold them
# needs N ~ 900 there, and no reachable spacing exposes the initial data's OWN error anyway --
# with a solver residual of ~1e-10 and spectral interpolation onto the grid, second-order
# truncation only falls that far at h ~ 2e-4 M.  This measurement is therefore an upper bound
# on the initial-data error by construction, which is what the appendix claims.
NS = (40, 56, 72, 88, 128, 176, 256)
L = 9.0                      # half-width of the Cartesian box
R_EXCL = 1.5                 # puncture exclusion radius of the bulk mask
TP_RES = (48, 48, 4)         # oracle resolution (nA, nB, nphi)
# The oracle is queried point by point through a pipe, at ~1.8 ms/point, so the finest rung is
# ~8 h serial -- well past the wrapper's 600 s default timeout.  Query it in chunks: each call
# re-runs the (cheap, ~7 s) spectral solve and stays inside a generous timeout, the rung reports
# progress instead of dying at the end of it, and the chunks parallelise across ``--workers``.
# The binary is deterministic to ~1e-10 across invocations (docs/DATA.md), which is what makes
# splitting one rung over many calls legitimate: far below the ~1e-3 truncation being measured.
TP_CHUNK = 200_000
TP_TIMEOUT = 3600


def _order(hs, errs):
    """Least-squares log-log slope: the measured FD convergence order."""
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def _psi_tp(X, Y, Z, workers=1, chunk=TP_CHUNK):
    """TwoPunctures psi at every point of a Cartesian grid, queried in chunks.

    Threads, not processes: every chunk is one ``subprocess.run`` of the oracle binary, so
    the GIL is released for the whole call and the workers genuinely run in parallel.
    """
    # TP's native frame puts the punctures on x, ours on z; the slice is axisymmetric,
    # so (z, rho, 0) in the oracle frame is the same physical point.
    pts = np.stack([Z.ravel(), np.sqrt(X ** 2 + Y ** 2).ravel(),
                    np.zeros(Z.size)], axis=1)
    out = np.empty(pts.shape[0])
    slices = [slice(i0, min(i0 + chunk, pts.shape[0]))
              for i0 in range(0, pts.shape[0], chunk)]
    done = [0]

    def run(sl):
        t0 = time.time()
        out[sl] = tp.solve_tp(B, M_A, M_B, P, pts[sl], nA=TP_RES[0], nB=TP_RES[1],
                              nphi=TP_RES[2], timeout=TP_TIMEOUT).psi
        done[0] += 1
        print(f"      oracle chunk {done[0]:>3d}/{len(slices)} "
              f"({sl.stop - sl.start} points, {time.time() - t0:.0f} s)", flush=True)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run, slices))
    else:
        for sl in slices:
            run(sl)
    return out.reshape(X.shape)


def _norms(psi, X, Y, Z, h):
    """Bulk L2-RMS of the generic FD Hamiltonian and momentum constraints."""
    A = cst.A_tensor_3d(X, Y, Z, B, P)
    H, Mvec = cst.fd_constraints_generic(psi, A, h)
    mask = cst.interior_mask(X, Y, Z, h, b=B, r_excl=R_EXCL)
    return cst.norms(H, mask)[1], cst.vec_norms(Mvec, mask)[1]


def build(with_tp=True, ns=NS, workers=1, chunk=TP_CHUNK):
    if with_tp and not tp.available():
        raise SystemExit(
            f"TwoPunctures binary not found at {tp.binary_path()} — build the oracle "
            f"(make oracle; see docs/DATA.md) or run with --no-tp")

    prob = sa.make_problem(Na=SPECTRAL[0], Nb=SPECTRAL[1], P=P)
    sl = sa.Slice(B, M_A, M_B)
    U, info = sa.newton_solve(prob, sl, tol=NEWTON_TOL, max_iter=25)

    hs, H_lm, M_lm, H_tp, M_tp = [], [], [], [], []
    for N in ns:
        _, X, Y, Z, h = cst.cartesian_grid(L, N)
        hs.append(h)

        eH, eM = _norms(cst.psi_on_grid(prob, U, sl, X, Y, Z), X, Y, Z, h)
        H_lm.append(eH)
        M_lm.append(eM)

        if with_tp:
            t0 = time.time()
            eHt, eMt = _norms(_psi_tp(X, Y, Z, workers, chunk), X, Y, Z, h)
            H_tp.append(eHt)
            M_tp.append(eMt)
            print(f"  N={N:3d}  h={h:.4f}  H={eH:.4e} (TP {eHt:.4e})  "
                  f"M={eM:.4e} (TP {eMt:.4e})  [oracle {time.time() - t0:.0f} s]",
                  flush=True)
        else:
            print(f"  N={N:3d}  h={h:.4f}  H={eH:.4e}  M={eM:.4e}", flush=True)

    hs = np.array(hs)
    curves = dict(N=list(ns), h=hs,
                  H_lm=H_lm, M_lm=M_lm,
                  order_H_lm=_order(hs, np.array(H_lm)),
                  order_M_lm=_order(hs, np.array(M_lm)))
    if with_tp:
        curves.update(H_tp=H_tp, M_tp=M_tp,
                      order_H_tp=_order(hs, np.array(H_tp)),
                      order_M_tp=_order(hs, np.array(M_tp)),
                      # how far the two initial-data sets are apart, per rung, as seen by
                      # the monitor: the number behind "identical constraint violations"
                      rel_H=list(np.abs(np.array(H_lm) - np.array(H_tp)) / np.array(H_tp)),
                      rel_M=list(np.abs(np.array(M_lm) - np.array(M_tp)) / np.array(M_tp)))

    meta = dict(b=B, m_A=M_A, m_B=M_B, P=P, spectral_grid=list(SPECTRAL),
                newton_residual=float(info.residual_norm), newton_tol=NEWTON_TOL,
                L=L, r_excl=R_EXCL, Ns=list(ns), tp_res=list(TP_RES),
                norm="bulk L2-RMS", monitor="generic FD (no conformal flatness)",
                has_tp=bool(with_tp))

    p = dump("fig10_constraints", dict(curves=curves, meta=meta))
    print(f"wrote {p}")
    print(f"  measured order:  H {curves['order_H_lm']:.3f}   M {curves['order_M_lm']:.3f}")
    # The local (pairwise) orders are what shows the ladder is still in the asymptotic
    # regime at the fine end; a single least-squares slope over a long ladder can hide a
    # softening rate.  Printed, not stored: the figure plots the fit.
    for i in range(len(hs) - 1):
        pH = np.log(H_lm[i] / H_lm[i + 1]) / np.log(hs[i] / hs[i + 1])
        pM = np.log(M_lm[i] / M_lm[i + 1]) / np.log(hs[i] / hs[i + 1])
        print(f"    local order {hs[i]:.4f} -> {hs[i + 1]:.4f}:  H {pH:5.3f}   M {pM:5.3f}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tp", action="store_true",
                    help="skip the TwoPunctures curve and write the solver curve only")
    ap.add_argument("--n-list", default=",".join(str(n) for n in NS),
                    help=f"comma-separated Cartesian rungs (default: {','.join(str(n) for n in NS)})")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel oracle chunks (threads; 32 on a cluster node)")
    ap.add_argument("--chunk", type=int, default=TP_CHUNK,
                    help=f"oracle query points per call (default {TP_CHUNK})")
    a = ap.parse_args()
    build(with_tp=not a.no_tp,
          ns=tuple(int(s) for s in a.n_list.split(",") if s.strip()),
          workers=a.workers, chunk=a.chunk)
