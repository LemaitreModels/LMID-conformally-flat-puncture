"""Produce the initial-data files the GRTeclyn constraint check consumes.

Step 2 of ``docs/GRTECLYN_CONSTRAINTS_PLAN.md``: run the certified Newton-Krylov
solve for each configuration and write it out in the plain format of
``validation.export_grteclyn``, together with a table of reference ``psi`` /
``Ahat`` values that the C++ port is validated against.

Configurations
--------------
``anchor``   the axisymmetric anchor of App. A (equal bare masses, b = 3,
             axial momentum P = 0.5) — what the appendix currently uses and
             where our solution and TwoPunctures differ least, so it carries the
             fewest confounds.  Primary.
``single``   one puncture (m_B = 0, P = 0).  Here psi = 1 + m_A/2 r_A is
             harmonic and K_ij = 0, so the continuum constraints vanish
             identically and any measured violation is pure truncation error —
             the sharpest possible check of the consumer.  Note this
             configuration is NOT expressible in GRTeclyn's own analytic
             initial data, whose boost term divides by the puncture mass.
``p010``     the same head-on configuration at P = 0.1.  GRTeclyn's analytic
             initial data enforces |P| < 0.3 m, so it cannot represent the
             anchor at all; P = 0.1 is inside that guard, making this the one
             configuration where both initial-data sets can be handed to the
             same code.
``qc``       an equal-mass quasi-circular slice with aligned spins, i.e. a
             genuinely non-axisymmetric case of the kind the parametric model
             actually emits.  Secondary/robustness.

With ``--tp`` each configuration is also exported with TwoPunctures' conformal
factor on the identical grid, for the cross-code parity gate.

Run:
    python -m lm.initial_data.pipeline.run_export_grteclyn --out DIR
    python -m lm.initial_data.pipeline.run_export_grteclyn --which anchor --tp
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from lm.initial_data.parametric import quasicircular as qc
from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.solver import solver_3d_nk as nk
from lm.initial_data.validation import export_grteclyn as eg

# The anchor's spectral grid, matching paper/figures/fig10_constraints_data.py.
SPECTRAL = (52, 36)
B_ANCHOR, M_A, M_B, P_ANCHOR = 3.0, 0.5, 0.5, 0.5


def _configs(which):
    """(name, Slice3D, Nphi) for each requested configuration."""
    out = []
    if which in ("all", "anchor"):
        out.append(("anchor",
                    s3.Slice3D.head_on(B_ANCHOR, M_A, M_B, P_ANCHOR), 8))
    if which in ("all", "single"):
        out.append(("single",
                    s3.Slice3D(b=B_ANCHOR, m_A=1.0, m_B=0.0), 1))
    if which in ("all", "p010"):
        # The same head-on configuration at P = 0.1.  GRTeclyn's own analytic
        # initial data refuses |P| >= 0.3 m (its O(P^2) small-boost guard), so
        # the anchor at P = 0.5 cannot be run with it at all.  P = 0.1 is inside
        # the guard, which makes this the one configuration where BOTH initial
        # data sets can be handed to the same code — the direct comparison of an
        # O(P^2)-accurate analytic solution against a solved one.
        out.append(("p010", s3.Slice3D.head_on(B_ANCHOR, M_A, M_B, 0.1), 8))
    if which in ("all", "qc"):
        # equal-mass, aligned spins along the orbital angular momentum (+y),
        # quasi-circular momenta from the PN closure
        S_A = (0.0, 0.15, 0.0)
        S_B = (0.0, 0.15, 0.0)
        P_A, P_B = qc.qc_momenta(B_ANCHOR, M_A, M_B, S_A, S_B)
        out.append(("qc", s3.Slice3D(b=B_ANCHOR, m_A=M_A, m_B=M_B,
                                     P_A_vec=P_A, P_B_vec=P_B,
                                     S_A_vec=S_A, S_B_vec=S_B), 8))
    if not out:
        raise SystemExit(f"unknown configuration {which!r}")
    return out


def tp_export(prob, sl, name, out_dir, n_ref):
    """Export TwoPunctures' conformal factor in the SAME format, on the SAME grid.

    This is what makes the cross-code comparison sharp.  GRTeclyn's own
    TwoPunctures path is an unported stub (its `initData` branch still uses the
    pre-AMReX API and is marked `todo`) and in any case needs an external Cactus
    thorn, so it cannot be the vehicle.  Instead the external oracle is queried
    for ``psi`` at the ABT spectral nodes, ``u = psi - psi_BL`` is formed with
    *our* Brill-Lindquist split, and the result is written in the same format —
    so the identical C++ class, interpolation, variable conversion, stencils and
    grid consume both data sets, and the only difference is the initial data.

    A side benefit: the oracle is queried at a few thousand spectral nodes
    instead of the ~26 million Cartesian points the in-house monitor's
    comparison needed (~13 h serial), which is why that sweep is retired.
    """
    from lm.initial_data.validation import twopunctures as tp

    if not tp.available():
        print(f"[{name}] TwoPunctures binary not found "
              f"({tp.binary_path()}) — skipping the parity export")
        return None

    # Node coordinates.  i = 0 is A = 1, i.e. spatial infinity, where the solve
    # imposes Dirichlet u = 0 (operators_abt.apply_bcs) and where the oracle
    # cannot be queried; fill that row with the same boundary value.
    from lm.initial_data.solver import operators_abt as ops
    A2, B2 = np.meshgrid(prob.A, prob.B, indexing="ij")
    rho2, z2 = ops.abt_map(A2, B2, sl.b)
    nA, nB, nphi = prob.shape
    U = np.zeros((nA, nB, nphi))

    rho_q, z_q, phi_q, idx = [], [], [], []
    for i in range(1, nA):                       # skip the A = 1 (infinity) row
        for j in range(nB):
            for p in range(nphi):
                rho_q.append(rho2[i, j])
                z_q.append(z2[i, j])
                phi_q.append(2.0 * np.pi * p / nphi)
                idx.append((i, j, p))
    t0 = time.time()
    res = tp.solve_lm_initial_data_points_3d(
        sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec,
        np.array(rho_q), np.array(z_q), np.array(phi_q))
    dt = time.time() - t0

    from lm.initial_data.solver import source
    psi_bl = np.asarray(source.psi_BL_2c(np.array(rho_q), np.array(z_q),
                                         sl.b, sl.m_A, sl.m_B))
    u_q = np.asarray(res.psi) - psi_bl
    for (i, j, p), val in zip(idx, u_q):
        U[i, j, p] = val

    exp = eg.from_solution(prob, sl, U,
                           note=f"config={name} SOURCE=TwoPunctures "
                                f"(E={res.E:.12g})")
    path = exp.write(os.path.join(out_dir, f"{name}.lmid"))
    ref = eg.dump_reference_table(
        exp, os.path.join(out_dir, f"{name}_reference.dat"), n=n_ref,
        half_width=9.0)
    print(f"[{name}] TwoPunctures: {len(idx)} node queries in {dt:.1f} s, "
          f"E_ADM={res.E:.12g}, max|u|={np.max(np.abs(U)):.6e}")
    return dict(name=name, path=path, reference=ref, source="TwoPunctures",
                b=sl.b, m_A=sl.m_A, m_B=sl.m_B, E_adm=float(res.E),
                Na=prob.Na, Nb=prob.Nb, Nphi=prob.Nphi,
                n_queries=len(idx), max_abs_u=float(np.max(np.abs(U))),
                wall_s=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/grteclyn_export")
    ap.add_argument("--which", default="all",
                    choices=["all", "anchor", "single", "qc", "p010"])
    ap.add_argument("--tp", action="store_true",
                    help="also export the TwoPunctures conformal factor on the "
                         "same grid, for the cross-code parity gate")
    ap.add_argument("--na", type=int, default=SPECTRAL[0])
    ap.add_argument("--nb", type=int, default=SPECTRAL[1])
    ap.add_argument("--nphi", type=int, default=None,
                    help="override the azimuthal resolution (default: "
                         "per-configuration)")
    ap.add_argument("--suffix", default="",
                    help="appended to the output names, so a "
                         "resolution variant does not overwrite the "
                         "production export")
    # The paper's certification threshold.  1e-12 is below the equilibrated
    # residual's roundoff floor (~2e-12 at this grid), so it never "converges".
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--n-ref", type=int, default=256,
                    help="reference-table points for the C++ port test")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = []

    for name, sl, nphi in _configs(args.which):
        if args.nphi is not None:
            nphi = args.nphi
        name = name + args.suffix
        t0 = time.time()
        prob = s3.make_problem(Na=args.na, Nb=args.nb, Nphi=nphi)
        U, info = nk.newton_solve_nk(prob, sl, tol=args.tol, verbose=True)
        dt = time.time() - t0
        if not info.converged:
            raise SystemExit(
                f"{name}: NOT converged (equilibrated residual "
                f"{info.residual_norm:.3e} after {info.iters} iters)")

        exp = eg.from_solution(prob, sl, U, residual=info.residual_norm,
                               raw_residual=info.raw_residual_norm,
                               note=f"config={name}")
        path = exp.write(os.path.join(args.out, f"{name}.lmid"))
        ref = eg.dump_reference_table(
            exp, os.path.join(args.out, f"{name}_reference.dat"),
            n=args.n_ref, half_width=9.0)

        # Self-check: the written file reproduces the solver's own evaluator.
        back = eg.Export.read(path)
        tab = np.loadtxt(ref)
        rho = np.hypot(tab[:, 0], tab[:, 1])
        phi = np.arctan2(tab[:, 1], tab[:, 0])
        u_solver = s3.evaluate_field(prob, U, rho, tab[:, 2], phi, sl.b)
        u_file = eg.eval_u(back, tab[:, 0], tab[:, 1], tab[:, 2])
        roundtrip = float(np.max(np.abs(u_solver - u_file)))
        if not roundtrip < 1e-11:
            raise SystemExit(f"{name}: export round-trip {roundtrip:.3e}")

        rec = dict(name=name, path=path, reference=ref,
                   b=sl.b, m_A=sl.m_A, m_B=sl.m_B,
                   P_A=list(sl.P_A_vec), P_B=list(sl.P_B_vec),
                   S_A=list(sl.S_A_vec), S_B=list(sl.S_B_vec),
                   Na=prob.Na, Nb=prob.Nb, Nphi=prob.Nphi,
                   iters=info.iters,
                   residual_equilibrated=info.residual_norm,
                   residual_raw=info.raw_residual_norm,
                   export_roundtrip_inf=roundtrip,
                   max_abs_u=float(np.max(np.abs(U))),
                   sine_content=float(np.max(np.abs(exp.S))) if exp.S.size
                   else 0.0,
                   wall_s=dt)
        summary.append(rec)
        print(f"[{name}] iters={info.iters} residual(equil)="
              f"{info.residual_norm:.3e} raw={info.raw_residual_norm:.3e} "
              f"roundtrip={roundtrip:.2e} max|u|={rec['max_abs_u']:.3e} "
              f"({dt:.1f} s)")

        if args.tp:
            tp_rec = tp_export(prob, sl, f"{name}_tp", args.out, args.n_ref)
            if tp_rec is not None:
                summary.append(tp_rec)
                # How far apart are the two initial-data solutions themselves?
                # This is the number the appendix currently quotes (<= 1e-6
                # relative); the GRTeclyn run then measures whether an
                # independent code can tell them apart at all.
                tp_exp = eg.Export.read(tp_rec["path"])
                tabp = np.loadtxt(rec["reference"])
                psi_us = eg.eval_psi(exp, tabp[:, 0], tabp[:, 1], tabp[:, 2])
                psi_tp = eg.eval_psi(tp_exp, tabp[:, 0], tabp[:, 1], tabp[:, 2])
                rel = float(np.max(np.abs(psi_us - psi_tp)
                                   / np.abs(psi_us)))
                tp_rec["psi_rel_vs_solver"] = rel
                print(f"[{name}_tp] max relative |psi_us - psi_tp| = "
                      f"{rel:.3e}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"configs": summary, "spectral": [args.na, args.nb]}, f,
                  indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
