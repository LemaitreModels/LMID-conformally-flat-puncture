"""LM-initial-data — the eccentricity-control demonstrator (paper §VI): accelerate the Cook
(1994) effective-potential circular-orbit finder with the differentiable, certified
surrogate.

Family: q=1, no spin, P_r=0, free (b, P_t); 2-D surrogate ``surrogate_bpt_ecc.npz``.
For each target angular momentum J:

  * classical SCAN — certified E_b at n_scan separations, parabola-fit the minimum;
  * gradient — Newton on ∂E_b/∂b|_J on the FREE surrogate, certify at the end;

both locate the same circular orbit; the gradient uses far fewer certified solves.
Sweeping several J shows the circular-orbit sequence (the effective-potential
minimum shifting with J), and the turning-point eccentricity e(b0;J) reads off the
same differentiable E_b (e→0 at the circular orbit).

Run: ~/micromamba/envs/BBHFM/bin/python -m lm.initial_data.pipeline.run_qc_effpot
     [--J 1.00 1.05 1.10] [--n-scan 13]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lm.initial_data.solver import solver_3d as s3
from lm.initial_data.applications import qc_effpot as E
from lm.initial_data.pipeline import production_box as pb

HERE = os.path.dirname(os.path.abspath(__file__))
from lm.initial_data.paths import reports_root
REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
MODEL = os.path.join(REPORTS, "3D_parametric", "models", "surrogate_bpt_ecc.npz")
REPDIR = os.path.join(REPORTS, "P3")
FIGDIR = os.path.join(HERE, "figures")
# The b-scan window.  Taken from production_box, NOT hardcoded: it must match the
# b range of the surrogate being scanned (``build_surrogate.BOXES["bpt_ecc"]``,
# whose b edges are the same constants), because ``circular_scan`` /
# ``eccentricity`` evaluate the model across this window and a surrogate cannot be
# extrapolated outside its box.  The historical value was (2.6, 6.4), whose lower
# edge predates the box retarget and now sits below B_MIN.
BOX_B = (pb.B_MIN, pb.B_MAX)

JLIST_DEFAULT = [1.00, 1.05, 1.10]
"""Angular momenta swept.  These also set the surrogate's P_t extent via
P_t = J/(2b) (see ``build_surrogate.ECC_J_MIN/ECC_J_MAX``), so changing them
invalidates the model.

GATE: the circular orbit is located as the INTERIOR minimum of dE_b/db|_J.  With
b_min raised to B_MIN, verify ``b_circ`` comes out strictly inside ``BOX_B`` for
every J here — if it lands on an edge the reported "circular orbit" is an edge
artifact, not a measurement, and the J list must move up with the box."""


def main(Jlist=None, n_scan=13):
    Jlist = list(JLIST_DEFAULT if Jlist is None else Jlist)
    os.makedirs(REPDIR, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)
    t0 = time.time()
    prob = s3.make_problem(Na=44, Nb=32, Nphi=8)
    model = E.load_model(MODEL, prob)
    print(f"[effpot] model {model.n_nodes} nodes; J sweep {Jlist}", flush=True)

    per_J = {}
    print(f"\n{'J':>6}{'b_circ(scan)':>14}{'b_circ(grad)':>14}{'|Δ|':>10}"
          f"{'solves scan/grad':>18}{'cert dEb/db':>14}")
    for J in Jlist:
        scan = E.circular_scan(model, prob, J, BOX_B, n_scan=n_scan)
        grad = E.circular_gradient(model, prob, J, b0=0.5 * (BOX_B[0] + BOX_B[1]),
                                   box_b=BOX_B)
        # eccentricity readout, outer apsis (in-box) parametrization
        b0s = np.linspace(grad.b_circ, BOX_B[1] - 0.1, 15)
        eccs = np.array([E.eccentricity(model, prob, float(b0), J, grad.b_circ, BOX_B)[0]
                         for b0 in b0s])
        per_J[J] = dict(scan=scan, grad=grad, b0s=b0s, eccs=eccs)
        print(f"{J:>6.2f}{scan.b_circ:>14.4f}{grad.b_circ:>14.4f}"
              f"{abs(scan.b_circ-grad.b_circ):>10.1e}"
              f"{f'{scan.n_certified_solves}/{grad.n_certified_solves}':>18}"
              f"{grad.dEb_db_certified:>14.1e}", flush=True)

    worst_R = max(max(p["scan"].certified_residual, p["grad"].certified_residual)
                  for p in per_J.values())
    n_scan_tot = sum(p["scan"].n_certified_solves for p in per_J.values())
    n_grad_tot = sum(p["grad"].n_certified_solves for p in per_J.values())
    print(f"\nacross {len(Jlist)} angular momenta: scan {n_scan_tot} certified solves "
          f"vs gradient {n_grad_tot}  ({n_scan_tot/n_grad_tot:.1f}x fewer)")
    print(f"circular-orbit sequence b_circ(J): "
          + ", ".join(f"J={J}:{per_J[J]['grad'].b_circ:.2f}" for J in Jlist))
    print(f"worst certified ||R||_inf = {worst_R:.1e}  (<= 1e-10)")

    res = dict(Jlist=Jlist, n_scan=n_scan, box_b=list(BOX_B),
               worst_certified_residual=worst_R,
               solves_scan_total=n_scan_tot, solves_gradient_total=n_grad_tot,
               per_J={f"{J:.2f}": dict(
                   b_circ_scan=per_J[J]["scan"].b_circ,
                   b_circ_gradient=per_J[J]["grad"].b_circ,
                   solves_scan=per_J[J]["scan"].n_certified_solves,
                   solves_gradient=per_J[J]["grad"].n_certified_solves,
                   dEb_db_certified=per_J[J]["grad"].dEb_db_certified,
                   ecc_b0=per_J[J]["b0s"].tolist(), ecc=per_J[J]["eccs"].tolist(),
                   scan_curve=per_J[J]["scan"].scan) for J in Jlist},
               wall_clock_s=time.time() - t0)
    out = os.path.join(REPDIR, "qc_effpot_Jsweep.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)

    n_grad = per_J[Jlist[0]]["grad"].n_certified_solves
    data = {J: dict(b_circ=per_J[J]["grad"].b_circ,
                    scan_b=np.asarray(per_J[J]["scan"].scan["b"], float),
                    scan_Eb=np.asarray(per_J[J]["scan"].scan["Eb"], float))
            for J in Jlist}
    _figure(model, prob, Jlist, data, n_scan, n_grad,
            os.path.join(FIGDIR, "fig_qc_eccentricity.png"))
    print(f"\n[effpot] DONE {time.time()-t0:.0f}s -> {out}", flush=True)
    return res


def _figure(model, prob, Jlist, data, n_scan, n_grad, path):
    """Single panel: the field-dependent binding energy $E_b(b)$ at each fixed $J$
    (the circular-orbit sequence).  For every $J$ the classical certified scan
    (squares) and the differentiable gradient minimum (star) are both shown, so the
    solve-count contrast is visible across the whole sequence — including the
    shallow small-$J$ well where the fixed scan mislocates the minimum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V, _ = E.build_effpot_jax(model, prob)
    PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#c44e52"]
    cols = {J: PALETTE[i % len(PALETTE)] for i, J in enumerate(Jlist)}
    b_all = np.concatenate([data[J]["scan_b"] for J in Jlist])
    bg = np.linspace(float(b_all.min()), float(b_all.max()), 240)

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for J in Jlist:
        c = cols[J]
        Vg = np.array([float(V(b, J)) for b in bg])
        ax.plot(bg, 1e3 * Vg, "-", color=c, lw=1.9, label=f"$J={J:.2f}$")
        ax.plot(data[J]["scan_b"], 1e3 * data[J]["scan_Eb"], "s", color=c, ms=3.8,
                mec="k", mew=0.3, alpha=0.85, zorder=3)
        bc = data[J]["b_circ"]
        ax.plot([bc], [1e3 * float(V(bc, J))], "*", color=c, ms=16, mec="k",
                mew=0.6, zorder=6)
    # neutral marker legend (shape = method, colour = J)
    ax.plot([], [], "k*", ms=13, label=f"gradient minimum ({n_grad} solves)")
    ax.plot([], [], "ks", ms=6, label=f"classical scan ({n_scan} solves each)")
    ax.set_xlabel(r"separation $b$  [$M$]")
    ax.set_ylabel(r"binding energy $E_b \times 10^{3}$")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center",
              columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[effpot] figure -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, nargs="+", default=None,
                    help="angular-momentum values to sweep (default 1.00 1.05 1.10)")
    ap.add_argument("--n-scan", type=int, default=13)
    ap.add_argument("--from-json", type=str, default=None,
                    help="regenerate the figure from an existing sweep JSON "
                         "(loads the model for the smooth curves; no solves)")
    args = ap.parse_args()
    if args.from_json:
        with open(args.from_json) as f:
            d = json.load(f)
        Jlist = list(d["Jlist"])
        n_scan = int(d["n_scan"])
        n_grad = int(d["per_J"][f"{Jlist[0]:.2f}"]["solves_gradient"])
        data = {J: dict(b_circ=d["per_J"][f"{J:.2f}"]["b_circ_gradient"],
                        scan_b=np.asarray(d["per_J"][f"{J:.2f}"]["scan_curve"]["b"], float),
                        scan_Eb=np.asarray(d["per_J"][f"{J:.2f}"]["scan_curve"]["Eb"], float))
                for J in Jlist}
        prob = s3.make_problem(Na=44, Nb=32, Nphi=8)
        model = E.load_model(MODEL, prob)
        _figure(model, prob, Jlist, data, n_scan, n_grad,
                os.path.join(FIGDIR, "fig_qc_eccentricity.png"))
    else:
        main(Jlist=args.J, n_scan=args.n_scan)
