"""B1 Step 4 — PARASOL vs TwoPunctures comparison + figures (v)/(vi).

Produces, for an equal-mass head-on slice:
  * psi-on-shared-points agreement vs PARASOL resolution (PARASOL -> TP as N grows);
  * an ADM table (total ADM mass, individual puncture ADM masses, ADM linear
    momentum, angular momentum) PARASOL vs TwoPunctures;
  * the oracle-independent constraint-violation-vs-grid-spacing curve (Step 5),
    optionally overlaid with a TwoPunctures-sourced baseline on the same grid.

The external oracle is invoked only through :mod:`parasol.validation.twopunctures`.
"""

from __future__ import annotations

import os

import numpy as np

from ..solver import solver_abt as sa, source
from . import adm, conventions, constraints as cst
from . import twopunctures as tp


# --------------------------------------------------------------------------
# Step 4a — psi on shared points: PARASOL converges to TwoPunctures
# --------------------------------------------------------------------------
def _query_points(b):
    """Interior meridian probes scaled to b: near A, mid, near B, off-axis, far."""
    return (np.array([0.30, 0.60, 0.30, 0.90, 2.0]) * b,        # rho
            np.array([1.10, 0.50, -1.10, 0.60, 0.40]) * b)      # z


def psi_agreement(b, m_A, m_B, P, parasol_grids, tp_res=(64, 64, 4)):
    """max|psi_PARASOL - psi_TP| at shared points, vs PARASOL grid resolution."""
    rho, z = _query_points(b)
    res = tp.solve_parasol_points(b, m_A, m_B, P, rho, z,
                                  nA=tp_res[0], nB=tp_res[1], nphi=tp_res[2])
    rows = []
    for (Na, Nb) in parasol_grids:
        prob = sa.make_problem(Na=Na, Nb=Nb, P=P)
        sl = sa.Slice(b, m_A, m_B)
        U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=30)
        u = np.asarray(sa.evaluate_field_phys(prob, U, rho, z, b))
        psi = np.asarray(source.psi_BL_2c(rho, z, b, m_A, m_B)) + u
        rows.append((Na, Nb, float(np.max(np.abs(psi - res.psi))),
                     info.residual_norm))
    return rows, res


# --------------------------------------------------------------------------
# Step 4b — ADM table: total mass, puncture masses, momenta, J
# --------------------------------------------------------------------------
def adm_table(b, m_A, m_B, P, parasol_grid=(64, 44), tp_res=(64, 64, 4)):
    prob = sa.make_problem(Na=parasol_grid[0], Nb=parasol_grid[1], P=P)
    sl = sa.Slice(b, m_A, m_B)
    U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=30)
    res = tp.solve_tp(b, m_A, m_B, P, np.array([[b, 0, 0]]),
                      nA=tp_res[0], nB=tp_res[1], nphi=tp_res[2])
    return {
        "newton_residual": info.residual_norm,
        "M_ADM": dict(parasol=adm.adm_mass_spectral(prob, U, sl),
                      tp=res.E),
        "M_A": dict(parasol=adm.puncture_adm_mass(prob, U, sl, "A"),
                    tp=res.mp_adm),
        "M_B": dict(parasol=adm.puncture_adm_mass(prob, U, sl, "B"),
                    tp=res.mm_adm),
        "P_A_z": dict(parasol=adm.by_momentum_gauss(b, P, "A"), tp=-P),
        "P_B_z": dict(parasol=adm.by_momentum_gauss(b, P, "B"), tp=+P),
        "P_total_z": dict(parasol=adm.adm_linear_momentum_total(b, P),
                          tp=0.0),
        "J": dict(parasol=0.0, tp=float(np.max(np.abs(res.J)))),
    }


# --------------------------------------------------------------------------
# Step 5 — constraint violation vs grid spacing (oracle-independent + TP baseline)
# --------------------------------------------------------------------------
def constraint_curve(b, m_A, m_B, P, Ns, L, parasol_grid=(52, 36),
                     tp_baseline_N=None, tp_res=(48, 48, 4), r_excl=1.0):
    """FD Hamiltonian + momentum constraint (L2-RMS, bulk) vs grid spacing h.

    Returns (rows, tp_rows): rows = [(N, h, H_L2, M_L2)] for PARASOL ID;
    tp_rows = same on a TwoPunctures-sourced grid at ``tp_baseline_N`` (or None).
    """
    prob = sa.make_problem(Na=parasol_grid[0], Nb=parasol_grid[1], P=P)
    sl = sa.Slice(b, m_A, m_B)
    U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=30)
    rows = []
    for N in Ns:
        x, X, Y, Z, h = cst.cartesian_grid(L, N)
        psi = cst.psi_on_grid(prob, U, sl, X, Y, Z)
        A = cst.A_tensor_3d(X, Y, Z, b, P)
        H, Mvec = cst.fd_constraints_generic(psi, A, h)
        mask = cst.interior_mask(X, Y, Z, h, b=b, r_excl=r_excl)
        rows.append((N, h, cst.norms(H, mask)[1], cst.vec_norms(Mvec, mask)[1]))

    tp_rows = None
    if tp_baseline_N is not None:
        x, X, Y, Z, h = cst.cartesian_grid(L, tp_baseline_N)
        # TP psi on the SAME grid (axial=z -> x_TP, radius=rho -> y_TP, z_TP=0)
        rho = np.sqrt(X ** 2 + Y ** 2)
        pts = np.stack([Z.ravel(), rho.ravel(), np.zeros(Z.size)], axis=1)
        rtp = tp.solve_tp(b, m_A, m_B, P, pts,
                          nA=tp_res[0], nB=tp_res[1], nphi=tp_res[2])
        psi_tp = rtp.psi.reshape(X.shape)
        A = cst.A_tensor_3d(X, Y, Z, b, P)
        H, Mvec = cst.fd_constraints_generic(psi_tp, A, h)
        mask = cst.interior_mask(X, Y, Z, h, b=b, r_excl=r_excl)
        tp_rows = [(tp_baseline_N, h, cst.norms(H, mask)[1],
                    cst.vec_norms(Mvec, mask)[1])]
    return rows, tp_rows


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _fig_agreement(rows, table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = [f"{Na}x{Nb}" for Na, Nb, _, _ in rows]
    dpsi = [d for _, _, d, _ in rows]
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(11, 4.6),
                                  gridspec_kw={"width_ratios": [1.2, 1]})
    ax.semilogy(range(len(rows)), dpsi, "o-", color="C0")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(Ns, rotation=30)
    ax.axhline(5e-9, ls="--", color="gray", alpha=0.6)
    ax.text(0, 6e-9, "two-centre spatial floor", color="gray", fontsize=8)
    ax.set(xlabel="PARASOL grid $N_A\\times N_B$",
           ylabel=r"$\max|\psi_{\rm PARASOL}-\psi_{\rm TwoPunctures}|$",
           title="(v) PARASOL $\\to$ TwoPunctures agreement on shared points")
    ax.grid(True, which="both", alpha=0.3)

    # ADM table panel
    axt.axis("off")
    lines = [["quantity", "PARASOL", "TwoPunctures", "rel.diff"]]
    for key in ("M_ADM", "M_A", "M_B"):
        p, t = table[key]["parasol"], table[key]["tp"]
        lines.append([key, f"{p:.9f}", f"{t:.9f}", f"{abs(p-t)/abs(t):.1e}"])
    for key in ("P_A_z", "P_B_z", "P_total_z", "J"):
        p, t = table[key]["parasol"], table[key]["tp"]
        lines.append([key, f"{p:.3e}", f"{t:.3e}", "-"])
    tbl = axt.table(cellText=lines[1:], colLabels=lines[0], loc="center",
                    cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.4)
    axt.set_title("ADM / quasi-local quantities", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_constraints(rows, tp_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = np.array([h for _, h, _, _ in rows])
    eH = np.array([e for _, _, e, _ in rows])
    eM = np.array([e for _, _, _, e in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.loglog(hs, eH, "o-", color="C0", label="Hamiltonian $\\|H\\|_2$")
    ax.loglog(hs, eM, "s-", color="C2", label="momentum $\\|M\\|_2$")
    # 2nd-order reference slope anchored at the coarsest point
    ref = eH[0] * (hs / hs[0]) ** 2
    ax.loglog(hs, ref, "--", color="gray", alpha=0.7, label="$\\propto h^2$")
    if tp_rows:
        for (N, h, eHt, eMt) in tp_rows:
            ax.loglog(h, eHt, "*", color="C0", ms=13, mec="k",
                      label="TwoPunctures-sourced $H$")
            ax.loglog(h, eMt, "*", color="C2", ms=13, mec="k",
                      label="TwoPunctures-sourced $M$")
    ax.set(xlabel="grid spacing $h$",
           ylabel="constraint violation (bulk $L_2$)",
           title="(vi) FD constraint violation vs grid spacing")
    ax.legend(fontsize=8.5)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(outdir=None, b=3.0, m_A=0.5, m_B=0.5, P=0.5,
         parasol_grids=((28, 20), (36, 24), (44, 30), (52, 36), (64, 44)),
         constraint_Ns=(40, 56, 72, 88), L=9.0, r_excl=1.5,
         tp_baseline_N=40, verbose=True):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = outdir or os.path.join(here, "figures")
    os.makedirs(outdir, exist_ok=True)

    rows, _ = psi_agreement(b, m_A, m_B, P, parasol_grids)
    table = adm_table(b, m_A, m_B, P)
    crows, tp_rows = constraint_curve(b, m_A, m_B, P, constraint_Ns, L,
                                      tp_baseline_N=tp_baseline_N, r_excl=r_excl)

    f_v = os.path.join(outdir, "tp_agreement.png")
    f_vi = os.path.join(outdir, "constraint_convergence.png")
    _fig_agreement(rows, table, f_v)
    _fig_constraints(crows, tp_rows, f_vi)

    if verbose:
        print("=== (v) psi-on-shared-points agreement (PARASOL -> TwoPunctures) ===")
        for Na, Nb, d, rn in rows:
            print(f"  {Na}x{Nb}: max|dpsi|={d:.3e}  (||R||={rn:.1e})")
        print("=== ADM / quasi-local table ===")
        for k, v in table.items():
            if isinstance(v, dict):
                print(f"  {k:10s}: PARASOL={v['parasol']:.10g}  TP={v['tp']:.10g}")
        print("=== (vi) constraint violation vs h (PARASOL ID) ===")
        for N, h, eH, eM in crows:
            print(f"  N={N}: h={h:.4f}  |H|2={eH:.3e}  |M|2={eM:.3e}")
        if tp_rows:
            for N, h, eH, eM in tp_rows:
                print(f"  TP-sourced N={N}: h={h:.4f}  |H|2={eH:.3e}  |M|2={eM:.3e}")
    return dict(agreement=rows, table=table, constraints=crows, tp_rows=tp_rows,
                figures=[f_v, f_vi])


if __name__ == "__main__":
    main()
