"""LM-initial-data — TwoPunctures agreement over the PRODUCTION BOX (random sample).

Why this exists
---------------
The appendix's other TwoPunctures figures are **convergence ladders**: one fixed
configuration, refined in resolution, to show that ``max|psi - psi_TP|`` is limited by
spatial resolution rather than by the nonlinear solve.  That argument requires a fixed
configuration and cannot be replaced by a parameter scan.

What it cannot show is whether the agreement at that one configuration is
*representative*.  Two gaps in particular:

  * the ladders sit at ``b=1.5`` (head-on, **below** the production ``B_MIN=3``) and at
    ``b=4`` (quasi-circular, nonspinning, equal mass) — so the external check does not
    sample the box the parametric models are claimed over,
    ``b in [3,10] x q in [1,3] x chi in [-0.9,0.9]^6``;
  * there is no external comparison at ``q != 1`` anywhere, and none at production spin
    magnitudes.  Meanwhile the *internal* certified residual IS swept (45 configurations).

This producer closes that: it draws configurations from
:mod:`~lm.initial_data.pipeline.production_box`, solves each with the certified
Newton-Krylov solver at the production grid, and compares with TwoPunctures at a
higher resolution than the solve.  The deliverable is the empirical DISTRIBUTION of the
agreement (best / median / worst), matching the convention the rest of the paper already
uses for held-out accuracy, plus a documented set of box-edge stress configurations so
the reported worst case is not a sampling accident.

Design decisions
----------------
* **Latin hypercube, not i.i.d. uniform.**  100 i.i.d. points in 8-D leave large holes and
  essentially never reach the box edges where accuracy is worst (small ``b``, large ``q``,
  large ``|chi|`` — the analyticity-wall region).  LHS guarantees one-dimensional
  stratification at exactly the requested ``n``; Sobol is available via ``--sampler sobol``
  (use a power-of-two ``n`` for its balance property to hold).
* **The edges are sampled deliberately, not hoped for.**  ``EDGE_CASES`` names the corner
  and near-corner configurations; they are reported separately from the interior sample so
  the interior distribution stays an unbiased estimate.
* **chi, not S.**  The box coordinate is the dimensionless ``chi = S/m^2``; the physical
  Bowen-York spin ``S_X = chi_X m_X^2`` therefore MOVES with ``q``.  The conversion is
  q-coupled and must not be hand-rolled -- it is done by the canonical mapping
  ``parametric_nd_3d.theta_to_slice3d`` (see docs/HISTORY_AND_FINDINGS.md 2.3).
* **The residual is the EQUILIBRATED one** (``info.residual_norm``), never the raw nodal
  norm (2.1).  Certification gate: ``<= 1e-10``.
* **The reference must out-resolve the test, in EVERY direction including phi.**  The
  oracle's default ``nphi=16`` sits above every solve grid's ``Nphi``; ``--selfconv k``
  re-runs the first ``k`` samples at a lower oracle ``nphi`` to bound the oracle's OWN
  truncation, which is the floor the comparison can resolve.  Measured, that floor is
  ``~7e-12`` for nonspinning configurations but only ``~2e-8`` for spinning ones -- so an
  oracle at ``nphi=12`` would already floor a refined spinning comparison, and a reported
  agreement at or below the floor is a statement about the oracle, not about this solver.

Two measured facts this producer exists to report honestly (diagnostic: 7 configurations,
grid ladder ``(44,32,8) -> (52,36,12) -> (64,44,16)`` against a ``(64,64,12)`` oracle):

* **Spin, not the mass ratio, drives the disagreement -- and it converges.**  Nonspinning
  configurations agree at ``~1e-9`` at BOTH ``q=1`` and ``q=2.27`` (so the ``q``-coupled
  ``chi -> S`` mapping is right).  Spinning ones start at ``1.5e-6`` (equal mass) to
  ``1.5e-5`` (``q=2.27``, where ``S_A = chi_A m_A^2`` puts a large physical spin on the
  heavier puncture) at the production grid and fall by ~2-3 decades under refinement.
  The production grid is a corpus-building choice; it is NOT converged for a pointwise
  external comparison at generic spins, which is exactly why this sweep reports a grid
  ladder rather than one grid.
* **The certified residual and the field agreement peak on DIFFERENT grids.**  The
  equilibrated residual RISES with resolution -- ``1.8e-13 -> 1.0e-7`` (nonspinning),
  ``4e-12 -> 3e-6`` (spinning) over that ladder -- from roundoff amplification in
  unpopulated high-``m`` modes, the effect the Fig. 8 caption already documents, not from
  a loss of convergence (the field difference keeps falling).  Consequence: the
  ``<= 1e-10`` certification gate is met at the PRODUCTION grid and not at the refined
  ones, so ``certified`` is recorded per grid and the top-level convenience keys are taken
  from the FIRST (production) grid -- the one whose certificate is meaningful.  Do not
  quote a certified residual from a refined grid.

Output
------
``reports/3D_parametric/qc/tp_random_sweep.json`` (registry source ``tp_random_sweep``).

Run (~130 s per sample, dominated by the oracle; embarrassingly parallel)::

    caffeinate -i ~/micromamba/envs/BBHFM/bin/python -m lm.initial_data.pipeline.run_tp_random_sweep \
        --n 100 --workers 6
"""
from __future__ import annotations

# Thread-limit BEFORE jax is imported: this module is re-imported in every worker
# process, and an unrestricted jax would oversubscribe the cores the pool is using.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from lm.initial_data.paths import reports_root
from lm.initial_data.parametric import parametric_nd_3d as p3
from lm.initial_data.pipeline import production_box as pbox
from lm.initial_data.solver import solver_3d as s3, solver_3d_nk as s3nk, source
from lm.initial_data.validation import twopunctures as tp

REPDIR = os.path.join(reports_root(), "3D_parametric", "qc")

# The 8-D general-spin production box, in the order theta_to_slice3d expects.
AXES = pbox.spin8_box()
NAMES = [a["name"] for a in AXES]
LO = np.array([a["min"] for a in AXES])
HI = np.array([a["max"] for a in AXES])

FIXED_QC = dict(pbox.FIXED_QC)          # {"qc": 1.0} -- deterministic PN momenta
CERT_TOL = 1e-10                        # certification gate (equilibrated); 2.8


def _edge_cases():
    """Box-edge stress configurations, as ``(label, theta)`` pairs.

    Not a random sample: these are the places accuracy is expected to be worst -- the
    smallest separation combined with the largest mass ratio and the largest spins, both
    spin signs, and the aligned / fully-tilted extremes.  Reported separately from the
    interior sample.
    """
    b0, b1 = pbox.B_MIN, pbox.B_MAX
    q0, q1 = pbox.Q_MIN, pbox.Q_MAX
    c = pbox.CHI_MAX
    z6 = [0.0] * 6

    def th(b, q, chi):
        return [b, q] + list(chi)

    return [
        ("b_min_q_max_aligned_up",   th(b0, q1, [0, +c, 0, 0, +c, 0])),
        ("b_min_q_max_aligned_down", th(b0, q1, [0, -c, 0, 0, -c, 0])),
        ("b_min_q_max_antialigned",  th(b0, q1, [0, +c, 0, 0, -c, 0])),
        ("b_min_q_min_aligned_up",   th(b0, q0, [0, +c, 0, 0, +c, 0])),
        ("b_min_q_max_tilted",       th(b0, q1, [+c, 0, 0, 0, 0, +c])),
        ("b_min_q_max_generic",      th(b0, q1, [+c, +c, +c, -c, -c, -c])),
        ("b_max_q_max_aligned_up",   th(b1, q1, [0, +c, 0, 0, +c, 0])),
        ("b_max_q_max_generic",      th(b1, q1, [+c, +c, +c, -c, -c, -c])),
        ("b_min_q_max_nospin",       th(b0, q1, z6)),
        ("b_mid_q_mid_nospin",       th(0.5 * (b0 + b1), 0.5 * (q0 + q1), z6)),
    ]


def _sample(n, sampler, seed):
    """``n`` interior configurations, stratified over the production box."""
    from scipy.stats import qmc

    d = len(AXES)
    if sampler == "sobol":
        eng = qmc.Sobol(d=d, scramble=True, seed=seed)
        m = int(math.ceil(math.log2(max(n, 2))))
        unit = eng.random_base2(m=m)[:n]
        if 2 ** m != n:
            print(f"[warn] Sobol balance holds at powers of two; n={n} != 2^{m}. "
                  f"Consider --n {2 ** m} or --sampler lhs.", flush=True)
    else:
        unit = qmc.LatinHypercube(d=d, seed=seed).random(n)
    return LO + unit * (HI - LO)


def _probe_points(b):
    """Shared query points: interior, mixed ``phi`` so the comparison sees the
    non-axisymmetric field (the quasi-circular momenta put the orbital ``L`` along
    ``+y``).  Same construction as ``run_qc_tp_validation.block_C``, scaled by ``b``."""
    rho = np.array([0.30, 0.60, 0.90, 0.30, 2.00, 0.70]) * b
    z = np.array([1.10, 0.50, 0.60, -1.10, 0.40, -0.50]) * b
    phi = np.array([0.0, math.pi / 2, math.pi / 4, math.pi, math.pi / 3, 1.0])
    return rho, z, phi


def _adm_mass_3d(prob, U, sl):
    """``M_ADM = m_A + m_B + 2c``, ``c = -b <d_A u_0>|_{A=1}``, ``u_0`` the phi-average.

    The phi-average is the ``m=0`` mode; the ``m != 0`` modes integrate to zero on the
    sphere at infinity and so do not enter the ADM monopole.  Same reader as
    ``run_qc_tp_validation.adm_mass_3d``.
    """
    u0 = np.asarray(U).mean(axis=2)
    dUdA_inf = (np.asarray(prob.DA1) @ u0)[0, :]        # A[0] = 1 -> infinity edge
    c = -sl.b * float(np.mean(dUdA_inf))
    return float(sl.M + 2.0 * c)


def _one(job):
    """Solve + oracle for one configuration.  Runs in a worker process.

    Solves at EVERY grid in ``grids`` against the SAME oracle call.  The oracle is ~95%
    of the per-sample cost, so the extra grids are nearly free -- and they are what turn
    this into a distributional version of the convergence claim: if the whole
    distribution shifts down with resolution, the disagreement is resolution-limited
    across the box, not only at the one configuration of the ladders.
    """
    idx, label, theta, grids, tp_res, selfconv_nphi, tol, timeout = job
    t0 = time.time()
    chi = np.asarray(theta[2:], dtype=float)
    out = dict(idx=idx, label=label, theta=[float(x) for x in theta],
               b=float(theta[0]), q=float(theta[1]),
               chi=[float(x) for x in chi],
               # the box coordinate is PER COMPONENT, so the box is a hyper-rectangle and
               # its corners carry |chi| up to sqrt(3)*CHI_MAX ~ 1.56 -- beyond the
               # horizon-spin ceiling that sets the half-width.  Record both the box
               # coordinate and the physical magnitudes so the tail can be attributed.
               chi_absmax=float(np.max(np.abs(chi))),
               chi_mag_A=float(np.linalg.norm(chi[:3])),
               chi_mag_B=float(np.linalg.norm(chi[3:])),
               chi_mag_max=float(max(np.linalg.norm(chi[:3]), np.linalg.norm(chi[3:]))))
    try:
        sl = p3.theta_to_slice3d(theta, NAMES, M_tot=1.0, fixed=FIXED_QC)
        out.update(m_A=float(sl.m_A), m_B=float(sl.m_B),
                   S_A=[float(x) for x in sl.S_A_vec], S_B=[float(x) for x in sl.S_B_vec],
                   P_A=[float(x) for x in sl.P_A_vec], P_B=[float(x) for x in sl.P_B_vec])
        rho, z, phi = _probe_points(sl.b)

        ref = tp.solve_lm_initial_data_points_3d(
            sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec,
            rho=rho, z=z, phi=phi,
            nA=tp_res[0], nB=tp_res[1], nphi=tp_res[2], timeout=timeout)

        # Bound the ORACLE's own error on a subset: same grid, lower nphi.
        if selfconv_nphi:
            lo = tp.solve_lm_initial_data_points_3d(
                sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec,
                rho=rho, z=z, phi=phi,
                nA=tp_res[0], nB=tp_res[1], nphi=selfconv_nphi, timeout=timeout)
            out["tp_selfconv_dpsi"] = float(np.max(np.abs(ref.psi - lo.psi)))
            out["tp_selfconv_dE"] = float(abs(ref.E - lo.E) / abs(ref.E))

        psiBL = np.asarray(source.psi_BL_2c(rho, z, sl.b, sl.m_A, sl.m_B))
        per_grid = []
        for g in grids:
            prob = s3.make_problem(Na=g[0], Nb=g[1], Nphi=g[2])
            U, info = s3nk.newton_solve_nk(prob, sl, tol=tol, max_iter=40, gmres_rtol=1e-8)
            u = np.asarray(s3.evaluate_field(prob, U, rho, z, phi, sl.b))
            M_adm = _adm_mass_3d(prob, U, sl)
            per_grid.append(dict(
                grid=list(g),
                max_dpsi=float(np.max(np.abs(psiBL + u - ref.psi))),
                M_ADM=M_adm,
                M_ADM_rel_diff=float(abs(M_adm - ref.E) / abs(ref.E)),
                residual=float(info.residual_norm),      # EQUILIBRATED (2.1)
                raw_residual=float(info.raw_residual_norm),
                certified=bool(info.residual_norm <= CERT_TOL),
                iters=int(info.iters), converged=bool(info.converged)))

        out.update(ok=True, tp_E=float(ref.E), per_grid=per_grid)
        # Convenience keys = the FIRST (production) grid: the one whose certificate is
        # meaningful.  The residual on a refined grid is roundoff-dominated (see above),
        # so promoting the finest grid here would advertise an uncertified residual.
        out.update({k: per_grid[0][k] for k in
                    ("max_dpsi", "M_ADM", "M_ADM_rel_diff", "residual",
                     "raw_residual", "certified", "iters", "converged")})
    except Exception as e:                               # a failed sample is data, not a crash
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    out["dt"] = time.time() - t0
    return out


def _stats(rows, key, gi=None):
    """best / median / p90 / worst of ``key``; ``gi`` selects a grid from ``per_grid``."""
    v = np.array([(r["per_grid"][gi][key] if gi is not None else r[key])
                  for r in rows if r.get("ok")], dtype=float)
    if v.size == 0:
        return None
    return dict(n=int(v.size), best=float(v.min()), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)), worst=float(v.max()))


def _group_stats(rows, grids):
    return dict(
        psi=_stats(rows, "max_dpsi"), M_ADM=_stats(rows, "M_ADM_rel_diff"),
        residual=_stats(rows, "residual"),
        per_grid=[dict(grid=list(g), psi=_stats(rows, "max_dpsi", i),
                       M_ADM=_stats(rows, "M_ADM_rel_diff", i),
                       residual=_stats(rows, "residual", i),
                       n_uncertified=int(sum(1 for r in rows if r.get("ok")
                                             and not r["per_grid"][i]["certified"])))
                  for i, g in enumerate(grids)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100, help="interior sample size (default 100)")
    ap.add_argument("--sampler", choices=("lhs", "sobol"), default="lhs")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--grids", type=str, default="44,32,8;56,40,12",
                    help="';'-separated NA,NB,NPHI solve grids, coarsest first "
                         "(default: the production grid and one refinement). All share the "
                         "one oracle call, so extra grids are nearly free.")
    ap.add_argument("--tp-res", type=int, nargs=3, default=[64, 64, 16],
                    metavar=("NA", "NB", "NPHI"),
                    help="oracle resolution; must out-resolve every solve grid in phi too")
    ap.add_argument("--selfconv", type=int, default=6,
                    help="re-run the first K samples at a lower oracle nphi to bound "
                         "the oracle's own error (0 disables)")
    ap.add_argument("--selfconv-nphi", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-12, help="solver target (equilibrated)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-oracle-call timeout (s)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-edges", action="store_true", help="skip the edge stress set")
    ap.add_argument("--out", default=os.path.join(REPDIR, "tp_random_sweep.json"))
    args = ap.parse_args()

    if not tp.available():
        raise SystemExit(f"TwoPunctures oracle not found at {tp.binary_path()!r}; "
                         f"build it with `make oracle` (see docs/DATA.md).")

    grids = [tuple(int(x) for x in g.split(",")) for g in args.grids.split(";") if g.strip()]
    if any(len(g) != 3 for g in grids):
        raise SystemExit(f"--grids must be ';'-separated NA,NB,NPHI triples, got {args.grids!r}")
    if max(g[2] for g in grids) >= args.tp_res[2]:
        print(f"[warn] the oracle must OUT-RESOLVE the solve, but nphi_solve="
              f"{max(g[2] for g in grids)} >= nphi_oracle={args.tp_res[2]}: the reported "
              f"agreement would be floored by the oracle's own truncation.", flush=True)

    interior = _sample(args.n, args.sampler, args.seed)
    edges = [] if args.no_edges else _edge_cases()

    jobs = []
    for i, th in enumerate(interior):
        sc = args.selfconv_nphi if i < args.selfconv else 0
        jobs.append((i, None, list(th), grids, tuple(args.tp_res),
                     sc, args.tol, args.timeout))
    for j, (lab, th) in enumerate(edges):
        jobs.append((len(interior) + j, lab, list(th), grids,
                     tuple(args.tp_res), 0, args.tol, args.timeout))

    print(f"=== TwoPunctures agreement over the production box ===")
    print(f"  box      : " + ", ".join(f"{a['name']}in[{a['min']:g},{a['max']:g}]" for a in AXES))
    print(f"  sample   : {args.n} {args.sampler.upper()} interior + {len(edges)} edge "
          f"(seed {args.seed})")
    print(f"  solve    : NK certified, grids {grids}, tol {args.tol:g}")
    print(f"  oracle   : TwoPunctures {tuple(args.tp_res)}"
          f"{f', nphi={args.selfconv_nphi} self-conv on first {args.selfconv}' if args.selfconv else ''}")
    print(f"  workers  : {args.workers}   ({len(jobs)} samples)\n", flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, j): j[0] for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            tag = r.get("label") or f"#{r['idx']:03d}"
            if r.get("ok"):
                print(f"  [{k:3d}/{len(jobs)}] {tag:<24s} b={r['b']:5.2f} q={r['q']:4.2f} "
                      f"|chi|max={r['chi_absmax']:4.2f}  max|dpsi|={r['max_dpsi']:.2e}  "
                      f"relDM={r['M_ADM_rel_diff']:.2e}  ||R||={r['residual']:.1e}  "
                      f"[{r['dt']:.0f}s]", flush=True)
            else:
                print(f"  [{k:3d}/{len(jobs)}] {tag:<24s} b={r['b']:5.2f} q={r['q']:4.2f}  "
                      f"FAILED: {r['error']}", flush=True)
    rows.sort(key=lambda r: r["idx"])

    inter = [r for r in rows if r.get("label") is None]
    edge = [r for r in rows if r.get("label") is not None]
    sc = [r["tp_selfconv_dpsi"] for r in rows if "tp_selfconv_dpsi" in r]

    summary = dict(
        interior=_group_stats(inter, grids),
        edge=_group_stats(edge, grids),
        grids=[list(g) for g in grids],
        n_failed=int(sum(1 for r in rows if not r.get("ok"))),
        n_uncertified=int(sum(1 for r in rows if r.get("ok") and not r.get("certified"))),
        tp_selfconv_dpsi_max=(float(max(sc)) if sc else None),
        wall_s=time.time() - t0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(meta=dict(axes=AXES, names=NAMES, n=args.n, sampler=args.sampler,
                                 seed=args.seed, grids=[list(g) for g in grids],
                                 tp_res=list(args.tp_res),
                                 selfconv=args.selfconv, selfconv_nphi=args.selfconv_nphi,
                                 tol=args.tol, cert_tol=CERT_TOL,
                                 fixed=FIXED_QC, box="spin8_qc_chi_prod"),
                      summary=summary, rows=rows), f, indent=1)

    print(f"\n=== summary ({summary['wall_s'] / 60:.1f} min) ===")
    for grp in ("interior", "edge"):
        if summary[grp]["psi"] is None:
            continue
        print(f"  {grp} (n={summary[grp]['psi']['n']}):")
        for pg in summary[grp]["per_grid"]:
            s, m, R = pg["psi"], pg["M_ADM"], pg["residual"]
            print(f"    grid {str(tuple(pg['grid'])):>14s}  max|dpsi|: best {s['best']:.2e}  "
                  f"median {s['median']:.2e}  p90 {s['p90']:.2e}  worst {s['worst']:.2e}")
            print(f"    {'':19s}  relDM   : best {m['best']:.2e}  "
                  f"median {m['median']:.2e}  p90 {m['p90']:.2e}  worst {m['worst']:.2e}")
            print(f"    {'':19s}  ||R||   : median {R['median']:.2e}  worst {R['worst']:.2e}"
                  f"   uncertified(>{CERT_TOL:g}): {pg['n_uncertified']}")
    print(f"  failed: {summary['n_failed']}")
    print(f"  NOTE ||R|| rises with resolution (roundoff in unpopulated high-m modes, "
          f"not loss of convergence); quote the certificate from the production grid only.")
    if summary["tp_selfconv_dpsi_max"] is not None:
        print(f"  oracle self-convergence (nphi {args.selfconv_nphi} vs {args.tp_res[2]}): "
              f"max {summary['tp_selfconv_dpsi_max']:.2e}  "
              f"(the floor the comparison can resolve)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
