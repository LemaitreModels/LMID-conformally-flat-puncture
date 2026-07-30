"""PARASOL §VIII — QC ↔ TwoPunctures validation.

Validates the quasi-circular (QC) initial data against the TwoPunctures oracle
and the closed-form anchors.  QC has tangential momentum along x (orbital L along
+y), so the field is genuinely non-axisymmetric ⇒ the PARASOL solves use the 3-D
solver at Nφ=8 and the oracle is called through the vector-momentum path
(``solve_parasol_points_3d`` / ``solve_tp_3d``, PARASOL→TP proper rotation
z^P→x^TP so PARASOL-y ↔ TP-z).

Blocks (→ reports/3D_parametric/qc/tp_validation_qc.json + fig_qc_tp.png):
  A.  Large-b Newtonian anchor  p_t → μ√(M/2b)  and eccentricity proxy p_r/p_t.
  B.  Angular momentum J = 2b·p_t (+ spins) along y — closed form vs TP's reported
      J (rotated to the PARASOL frame); net linear momentum P_A+P_B.
  C.  ψ spectral convergence to TP on a QC slice as the PARASOL grid refines +
      ADM-mass relative agreement (3-D φ-averaged spectral M_ADM vs TP.E).

    caffeinate -i ~/micromamba/envs/BBHFM/bin/python sandbox/parasol/run_qc_tp_validation.py
"""
from __future__ import annotations
import json, math, os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from lm.initial_data.solver import solver_3d as s3, solver_3d_nk as s3nk, source
from lm.initial_data.parametric import parametric_nd_3d as p3, quasicircular as qc
from lm.initial_data.validation import twopunctures as tp, conventions as cv

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "3D_parametric", "qc")
os.makedirs(REPDIR, exist_ok=True)
NAMES = ["b", "q", "S_Ay", "S_By"]


def _t(m): print(m, flush=True)


def qc_slice(b, q, S_Ay=0.0, S_By=0.0, M=1.0):
    return p3.theta_to_slice3d([b, q, S_Ay, S_By], NAMES, M_tot=M, fixed={"qc": 1.0})


def adm_mass_3d(prob, U, sl):
    """M_ADM = (m_A+m_B) + 2c, c = -b·<∂_A u_0>|_{A=1}, u_0 = φ-average of U.

    The φ-average is the m=0 mode (the m≠0 angular modes integrate to zero on the
    sphere at infinity, contributing nothing to the ADM monopole); the ABT
    spectral boundary read at A=1 is the 3-D twin of validation.adm.adm_mass_spectral.
    """
    u0 = np.asarray(U).mean(axis=2)                     # (Na+1, Nb) φ-average
    dUdA_inf = (np.asarray(prob.DA1) @ u0)[0, :]        # A[0]=1 -> infinity edge
    c = -sl.b * float(np.mean(dUdA_inf))
    return float(sl.M + 2.0 * c)


# ==========================================================================
# A — Newtonian anchor + eccentricity proxy (closed form, q=1 no spin)
# ==========================================================================
def block_A(M=1.0):
    _t("\n=== A: Newtonian anchor p_t→μ√(M/2b) + eccentricity proxy (q=1, no spin) ===")
    rows = []
    for b in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
        sl = qc_slice(b, 1.0)
        m_A, m_B = sl.m_A, sl.m_B
        p_t, p_r = qc.qc_scalar_momenta(b, m_A, m_B)
        p_newt = qc.pt_newtonian(b, m_A, m_B)
        ecc = qc.eccentricity_proxy(b, m_A, m_B)
        rows.append(dict(b=b, p_t=p_t, p_t_newt=p_newt, ratio=p_t / p_newt,
                         ecc_proxy=ecc))
        _t(f"   b={b:5.1f}  p_t={p_t:.6f}  p_t^Newt={p_newt:.6f}  "
           f"p_t/p_Newt={p_t/p_newt:.6f}  p_r/p_t={ecc:.2e}")
    return rows


# ==========================================================================
# B — J closed form vs TP + net momentum
# ==========================================================================
def block_B():
    _t("\n=== B: angular momentum J (closed form vs TwoPunctures, PARASOL frame) ===")
    cases = [(4.0, 1.0, 0.0, 0.0), (3.0, 2.0, 0.0, 0.0),
             (5.0, 1.0, 0.3, 0.3), (4.0, 1.0, -0.2, 0.2)]
    rows = []
    for (b, q, SAy, SBy) in cases:
        sl = qc_slice(b, q, SAy, SBy)
        p_t, _ = qc.qc_scalar_momenta(b, sl.m_A, sl.m_B, SAy / sl.m_A**2, SBy / sl.m_B**2)
        J_orbital = qc.orbital_angular_momentum(b, p_t)        # 2b·p_t (y)
        J_total = J_orbital + SAy + SBy                        # + aligned spins
        r = tp.solve_parasol_points_3d(b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec,
                                       sl.S_A_vec, sl.S_B_vec, rho=[1.3], z=[0.7],
                                       phi=[0.4], nA=48, nB=48, nphi=8)
        J_tp = cv.tp_vec_to_parasol(r.J)                        # -> PARASOL frame
        Jy_tp = float(J_tp[1])
        net = tuple(float(x) for x in np.add(sl.P_A_vec, sl.P_B_vec))
        reld = abs(J_total - Jy_tp) / abs(Jy_tp) if Jy_tp else float("nan")
        rows.append(dict(b=b, q=q, S_Ay=SAy, S_By=SBy, p_t=p_t,
                         J_orbital=J_orbital, J_total_closed=J_total,
                         J_tp_parasol_frame=[float(x) for x in J_tp], Jy_tp=Jy_tp,
                         rel_diff=reld, net_momentum=net, tp_E=r.E))
        _t(f"   b={b} q={q} S=({SAy},{SBy}): J_closed={J_total:.6f}  "
           f"J_TP(y)={Jy_tp:.6f}  rel={reld:.2e}  |net P|={max(abs(x) for x in net):.1e}")
    return rows


# ==========================================================================
# C — ψ convergence + ADM mass agreement on a QC slice
# ==========================================================================
def block_C(b=4.0, q=1.0, tp_res=(64, 64, 12)):
    _t(f"\n=== C: ψ→TP convergence + ADM mass  (QC slice b={b}, q={q}, no spin) ===")
    sl = qc_slice(b, q)
    # shared probe points (interior; mixed φ to probe the non-axisymmetric field)
    rho = np.array([0.30, 0.60, 0.90, 0.30, 2.0, 0.7]) * b
    z = np.array([1.10, 0.50, 0.60, -1.10, 0.40, -0.5]) * b
    phi = np.array([0.0, math.pi / 2, math.pi / 4, math.pi, math.pi / 3, 1.0])
    # TP reference at high resolution + a TP nphi self-convergence check
    ref = tp.solve_parasol_points_3d(b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec,
                                     sl.S_A_vec, sl.S_B_vec, rho=rho, z=z, phi=phi,
                                     nA=tp_res[0], nB=tp_res[1], nphi=tp_res[2])
    ref8 = tp.solve_parasol_points_3d(b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec,
                                      sl.S_A_vec, sl.S_B_vec, rho=rho, z=z, phi=phi,
                                      nA=tp_res[0], nB=tp_res[1], nphi=8)
    tp_nphi_selfconv = float(np.max(np.abs(ref.psi - ref8.psi)))
    _t(f"   TP reference res={tp_res}; TP nφ=8 vs {tp_res[2]} ψ self-diff = {tp_nphi_selfconv:.2e}  "
       f"(reference nφ-converged); TP.E = {ref.E:.10f}")
    rows = []
    for (Na, Nb) in [(28, 20), (36, 24), (44, 30), (52, 36), (64, 44)]:
        prob = s3.make_problem(Na=Na, Nb=Nb, Nphi=8)
        t0 = time.time()
        U, info = s3nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=30)
        u = np.asarray(s3.evaluate_field(prob, U, rho, z, phi, b))
        psiBL = np.asarray(source.psi_BL_2c(rho, z, b, sl.m_A, sl.m_B))
        psi = psiBL + u
        dpsi = float(np.max(np.abs(psi - ref.psi)))
        M_adm = adm_mass_3d(prob, U, sl)
        reld_M = abs(M_adm - ref.E) / abs(ref.E)
        rows.append(dict(Na=Na, Nb=Nb, max_dpsi=dpsi, residual=float(info.residual_norm),
                         M_ADM=M_adm, M_ADM_rel_diff=reld_M))
        _t(f"   {Na}x{Nb}: max|Δψ|={dpsi:.3e}  ‖R‖={info.residual_norm:.1e}  "
           f"M_ADM={M_adm:.10f}  relΔM={reld_M:.2e}  [{time.time()-t0:.0f}s]")
    return dict(slice=dict(b=b, q=q), tp_res=list(tp_res), tp_E=ref.E,
                tp_nphi_selfconv=tp_nphi_selfconv,
                probe=dict(rho=[float(x) for x in rho], z=[float(x) for x in z],
                           phi=[float(x) for x in phi]),
                grids=rows)


def make_figure(results):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        _t(f"[fig] matplotlib unavailable ({e})"); return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    A = results["A_newtonian"]
    bb = [r["b"] for r in A]
    ax1.semilogx(bb, [r["ratio"] for r in A], "o-", label=r"$p_t/p_t^{\rm Newt}$")
    ax1.axhline(1.0, ls="--", color="gray", alpha=0.6)
    ax1.set_xlabel("half-separation b"); ax1.set_ylabel(r"$p_t/p_t^{\rm Newt}$")
    ax1.set_title("A: Newtonian anchor"); ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=9)
    ax1b = ax1.twinx()
    ax1b.loglog(bb, [r["ecc_proxy"] for r in A], "s--", color="C1", label=r"$p_r/p_t$")
    ax1b.set_ylabel(r"eccentricity proxy $p_r/p_t$", color="C1")
    C = results["C_psi_adm"]["grids"]
    ns = [f"{r['Na']}x{r['Nb']}" for r in C]
    ax2.semilogy(range(len(C)), [r["max_dpsi"] for r in C], "o-")
    ax2.set_xticks(range(len(C))); ax2.set_xticklabels(ns, rotation=30)
    ax2.set_xlabel("PARASOL grid $N_A\\times N_B$ (Nφ=8)")
    ax2.set_ylabel(r"$\max|\psi_{\rm PARASOL}-\psi_{\rm TP}|$")
    ax2.set_title("C: ψ → TwoPunctures (QC slice)"); ax2.grid(True, which="both", alpha=0.3)
    ax3.semilogy(range(len(C)), [r["M_ADM_rel_diff"] for r in C], "o-", color="C2")
    ax3.set_xticks(range(len(C))); ax3.set_xticklabels(ns, rotation=30)
    ax3.set_xlabel("PARASOL grid $N_A\\times N_B$")
    ax3.set_ylabel(r"$|M_{\rm ADM}-E_{\rm TP}|/E_{\rm TP}$")
    ax3.set_title("C: ADM mass agreement"); ax3.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REPDIR, "fig_qc_tp.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    _t(f"[fig] wrote {path}")


def main():
    t0 = time.time()
    results = {"meta": dict(tp_binary=tp.binary_path(), tp_available=tp.available())}
    results["A_newtonian"] = block_A()
    results["B_angular_momentum"] = block_B()
    results["C_psi_adm"] = block_C()
    results["meta"]["wall_s"] = time.time() - t0
    out = os.path.join(REPDIR, "tp_validation_qc.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    _t(f"\nWrote {out}")
    make_figure(results)
    _t(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
