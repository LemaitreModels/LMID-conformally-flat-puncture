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

Cost: the LM curve is ~10 s; the oracle curve is ~40 min (the binary is queried at every
Cartesian point of every rung, 1.3M points in total).  ``--no-tp`` skips it.

Run:  python fig10_constraints_data.py            # both curves (needs the TP binary)
      python fig10_constraints_data.py --no-tp    # LM curve only
"""
import argparse
import os
import sys
import time

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
NS = (40, 56, 72, 88)        # Cartesian rungs; fine enough to sit in the asymptotic regime
L = 9.0                      # half-width of the Cartesian box
R_EXCL = 1.5                 # puncture exclusion radius of the bulk mask
TP_RES = (48, 48, 4)         # oracle resolution (nA, nB, nphi)
# The oracle is queried point by point through a pipe, at ~1.8 ms/point, so the finest rung
# alone is ~20 min -- well past the wrapper's 600 s default timeout.  Query it in chunks: each
# call re-runs the (cheap, ~7 s) spectral solve and stays inside a generous timeout, and the
# rung reports progress instead of dying at the end of it.
TP_CHUNK = 100_000
TP_TIMEOUT = 3600


def _order(hs, errs):
    """Least-squares log-log slope: the measured FD convergence order."""
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def _psi_tp(X, Y, Z):
    """TwoPunctures psi at every point of a Cartesian grid, queried in chunks."""
    # TP's native frame puts the punctures on x, ours on z; the slice is axisymmetric,
    # so (z, rho, 0) in the oracle frame is the same physical point.
    pts = np.stack([Z.ravel(), np.sqrt(X ** 2 + Y ** 2).ravel(),
                    np.zeros(Z.size)], axis=1)
    out = np.empty(pts.shape[0])
    for i0 in range(0, pts.shape[0], TP_CHUNK):
        sl = slice(i0, min(i0 + TP_CHUNK, pts.shape[0]))
        t0 = time.time()
        out[sl] = tp.solve_tp(B, M_A, M_B, P, pts[sl], nA=TP_RES[0], nB=TP_RES[1],
                              nphi=TP_RES[2], timeout=TP_TIMEOUT).psi
        print(f"      oracle {sl.stop:>7d}/{pts.shape[0]} points "
              f"({time.time() - t0:.0f} s)", flush=True)
    return out.reshape(X.shape)


def _norms(psi, X, Y, Z, h):
    """Bulk L2-RMS of the generic FD Hamiltonian and momentum constraints."""
    A = cst.A_tensor_3d(X, Y, Z, B, P)
    H, Mvec = cst.fd_constraints_generic(psi, A, h)
    mask = cst.interior_mask(X, Y, Z, h, b=B, r_excl=R_EXCL)
    return cst.norms(H, mask)[1], cst.vec_norms(Mvec, mask)[1]


def build(with_tp=True):
    if with_tp and not tp.available():
        raise SystemExit(
            f"TwoPunctures binary not found at {tp.binary_path()} — build the oracle "
            f"(make oracle; see docs/DATA.md) or run with --no-tp")

    prob = sa.make_problem(Na=SPECTRAL[0], Nb=SPECTRAL[1], P=P)
    sl = sa.Slice(B, M_A, M_B)
    U, info = sa.newton_solve(prob, sl, tol=NEWTON_TOL, max_iter=25)

    hs, H_lm, M_lm, H_tp, M_tp = [], [], [], [], []
    for N in NS:
        _, X, Y, Z, h = cst.cartesian_grid(L, N)
        hs.append(h)

        eH, eM = _norms(cst.psi_on_grid(prob, U, sl, X, Y, Z), X, Y, Z, h)
        H_lm.append(eH)
        M_lm.append(eM)

        if with_tp:
            t0 = time.time()
            eHt, eMt = _norms(_psi_tp(X, Y, Z), X, Y, Z, h)
            H_tp.append(eHt)
            M_tp.append(eMt)
            print(f"  N={N:3d}  h={h:.4f}  H={eH:.4e} (TP {eHt:.4e})  "
                  f"M={eM:.4e} (TP {eMt:.4e})  [oracle {time.time() - t0:.0f} s]",
                  flush=True)
        else:
            print(f"  N={N:3d}  h={h:.4f}  H={eH:.4e}  M={eM:.4e}", flush=True)

    hs = np.array(hs)
    curves = dict(N=list(NS), h=hs,
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
                L=L, r_excl=R_EXCL, Ns=list(NS), tp_res=list(TP_RES),
                norm="bulk L2-RMS", monitor="generic FD (no conformal flatness)",
                has_tp=bool(with_tp))

    p = dump("fig10_constraints", dict(curves=curves, meta=meta))
    print(f"wrote {p}")
    print(f"  measured order:  H {curves['order_H_lm']:.2f}   M {curves['order_M_lm']:.2f}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tp", action="store_true",
                    help="skip the TwoPunctures curve (~40 min) and write the LM curve only")
    build(with_tp=not ap.parse_args().no_tp)
