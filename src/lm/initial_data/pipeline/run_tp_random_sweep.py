"""LM-initial-data — TwoPunctures validation as BANDS over the production box.

Why this exists
---------------
The appendix's TwoPunctures validation used to be two figures, each a convergence ladder
at ONE hand-picked configuration: a misaligned-spin head-on slice at ``b=1.5`` (below the
production ``B_MIN=3``, and not quasi-circular), and a nonspinning equal-mass
quasi-circular slice at ``b=4``.  The ladder structure is right -- refining at fixed
parameters is what shows ``max|psi - psi_TP|`` to be limited by spatial resolution rather
than by the nonlinear solve -- but the choice of parameters was arbitrary, and neither
figure sampled the box the parametric models are actually claimed over
(``b in [3,10] x q in [1,3] x chi in [-0.9,0.9]^6``).  In particular there was no external
comparison at ``q != 1`` anywhere, nor at production spin magnitudes.

This producer keeps the ladder and removes the parameter specifics: it walks the SAME
resolution ladder for each of ``n`` configurations drawn from the production box, so every
reported curve becomes a min/median/max band over configurations.

It also absorbs what the separate non-axisymmetric figure carried.  Measured (production
grid, ``Nphi=8``, ``b=4``), the quasi-circular data are ALREADY non-axisymmetric: the
tangential momentum puts ~2% of the field in ``m=2``, and generic spins add ~2% at ``m=1``,
whereas a head-on slice with spin along the collision axis keeps every ``m>=1`` mode at
~3e-17.  So the QC family exercises the Fourier-in-phi solver by itself -- hence the
``spectrum`` block here (which justifies the production ``Nphi`` and explains where the
spin/mass-ratio sensitivity comes from) and the ``axisym`` block (the one check QC cannot
provide, because QC is never axisymmetric: that the phi machinery does not manufacture
spurious non-axisymmetry).

Design decisions
----------------
* **Latin hypercube, not i.i.d. uniform.**  100 i.i.d. points in 8-D leave large holes and
  essentially never reach the box edges where accuracy is worst.  LHS gives
  one-dimensional stratification at exactly the requested ``n``; ``--sampler sobol`` is
  available (use a power-of-two ``n`` for its balance property to hold).
* **The edges are sampled deliberately.**  ``EDGE_CASES`` names the corner and near-corner
  configurations, reported separately so the interior band stays an unbiased estimate.
* **chi, not S.**  The box coordinate is the dimensionless ``chi = S/m^2``, so the physical
  Bowen-York spin ``S_X = chi_X m_X^2`` MOVES with ``q``.  That conversion is q-coupled and
  must not be hand-rolled: it goes through the canonical mapping
  ``parametric_nd_3d.theta_to_slice3d`` (docs/HISTORY_AND_FINDINGS.md 2.3).
* **The residual is the EQUILIBRATED one** (``info.residual_norm``), never the raw nodal
  norm (2.1).  Certification gate: ``<= 1e-10``.
* **One oracle call per configuration, shared by every rung.**  The oracle is ~95% of the
  per-sample cost, which is what makes a whole ladder per configuration affordable.
* **The reference must out-resolve the test in EVERY direction, phi included.**  The
  oracle's ``nphi`` default sits above every rung's ``Nphi``; ``--selfconv k`` re-runs the
  first ``k`` configurations at a lower oracle ``nphi`` to bound the oracle's OWN
  truncation, which is the floor the comparison can resolve.  Measured, that floor is
  ~4e-13 at ``nphi=12`` vs ``16``, so the oracle does not limit anything reported here.

Three measured facts to read the output with
--------------------------------------------
* **The sup-norm needs a DENSE probe set.**  The predecessor estimated
  ``||psi - psi_TP||_inf`` from six query points.  Measured against a converged oracle,
  that underestimates the sup-norm by up to 14x and makes the convergence sequence
  non-monotone (a six-point maximum collapses when the error at the dominant point changes
  sign).  Six and 64 points agree exactly on coarse grids and diverge in the well-resolved
  regime -- on exactly the converged numbers one quotes.  ``max_dpsi`` is the dense
  estimate; ``max_dpsi_legacy6`` keeps the old one for comparison.
* **Spin, not the mass ratio alone, drives the disagreement -- and it converges.**
  Nonspinning configurations agree at ``~2e-8`` on the coarsest rung and reach ``~2e-9``,
  at both ``q=1`` and ``q=3``; spinning ones start near ``1e-4`` and fall by 2-3 decades
  over the ladder (316x for a ``chi=0.7``, ``q=3`` slice).  At fixed spin the disagreement
  grows ~100x from ``q=1`` to ``q=3`` and a further ~10x from ``b=3`` to ``b=10``.
* **The certified residual RISES with resolution, and the DIRECTION decides how much.**
  Along this meridional ladder the rise is mild (~33x) and every rung stays under the
  ``1e-10`` gate; raising ``Nphi`` instead makes it rise by ~6e6 (to ``3.2e-7``) and breach
  the gate, because the mechanism is roundoff amplification in unpopulated high-``m``
  azimuthal modes.  Either way it is not a loss of convergence -- the field difference
  keeps falling.  ``certified`` is recorded per rung; never quote a certificate from a rung
  that fails the gate.

Output
------
``reports/3D_parametric/qc/tp_band_sweep.json`` (registry source ``tp_band_sweep``).

Run (~2-8 min of oracle per configuration, so budget hours; embarrassingly parallel)::

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

# The resolution ladder: MERIDIONAL refinement at FIXED Nphi=8.  Measured against a
# (72,72,20) oracle on three configurations, raising Nphi alongside (Na,Nb) changes the
# field convergence not at all -- a spinning q=3 slice gives 1.447e-4 -> 4.584e-7 (316x) at
# fixed Nphi=8 versus 1.446e-4 -> 4.568e-7 (317x) rising, agreeing to three digits at every
# rung -- so (Na,Nb), not Nphi, is what limits the agreement with TwoPunctures.  What DOES
# depend on the direction is the certified residual: it rises 33x along this ladder and
# stays under the 1e-10 gate at every rung, whereas raising Nphi makes it rise by ~6e6 (to
# 3.2e-7) and breach the gate from the third rung on, because the mechanism is roundoff
# amplification in unpopulated high-m azimuthal modes.  Refining the meridian therefore
# keeps every rung certified at no cost in convergence.  (This is also the ladder the
# previous single-configuration figure used.)
DEFAULT_LADDER = "28,20,8;36,24,8;44,30,8;52,36,8;64,44,8"


def _edge_cases():
    """Box-edge stress configurations, as ``(label, theta)`` pairs.

    Not a random sample: these are where accuracy is expected to be worst -- the smallest
    separation with the largest mass ratio and the largest spins, both spin signs, and the
    aligned / fully-tilted extremes.  Reported separately from the interior band.
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


def _axisym_cases():
    """Head-on slices with spin along the COLLISION AXIS: exactly axisymmetric.

    The one check the quasi-circular family cannot supply -- QC always carries a tangential
    momentum, so it is never axisymmetric.  Here every ``m >= 1`` mode must sit at roundoff,
    which is what shows the Fourier-in-phi machinery does not manufacture spurious
    non-axisymmetry.  No oracle call: this is an internal symmetry check.
    """
    c = pbox.CHI_MAX
    return [
        ("headon_chiz_q1",      [4.0, 1.0, 0, 0, 0.5, 0, 0, 0.5]),
        ("headon_chiz_q3",      [4.0, 3.0, 0, 0, 0.5, 0, 0, 0.5]),
        ("headon_chiz_max_q3",  [pbox.B_MIN, 3.0, 0, 0, c, 0, 0, c]),
        ("headon_nospin_q1",    [4.0, 1.0, 0, 0, 0, 0, 0, 0]),
    ]


def _anchor_case():
    """The axisymmetric code-to-code anchor: equal mass, ``b=3``, axial momentum ``P=0.5``.

    Head-on (``fixed=None`` takes the axial-momentum branch), so there is no azimuthal
    structure to limit either code -- which is why this is the sharpest comparison available
    and why the appendix quotes it.  It is measured HERE, through the same pipeline and the
    same dense probe set as everything else, rather than carried over: the previously quoted
    value came from the six-probe estimate this producer replaces.
    """
    return ("anchor_axisym_b3_P0.5", [3.0, 1.0, 0, 0, 0, 0, 0, 0])


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


N_PROBE_DENSE = 64          # beyond ~50 the sup-norm estimate stops moving
PROBE_SEED = 11             # fixed: the probe set must not vary between configurations


def _probe_points(b, n_dense=N_PROBE_DENSE, seed=PROBE_SEED):
    """Query points shared with the oracle: interior, all ``phi``, clear of both punctures.

    The first SIX points are the legacy set of ``run_qc_tp_validation.block_C``, kept so the
    previous numbers remain reproducible as a subset; the rest are a fixed well-spread
    sample.  The density is load-bearing, not cosmetic.  Measured on a spinning
    ``b=5.3``, ``q=2.9`` slice against a converged oracle, a six-point maximum
    UNDERESTIMATES the sup-norm by up to 14x (4.05e-7 against 5.76e-6) and turns the
    convergence sequence non-monotone, because with six samples the maximum collapses
    whenever the error at the dominant point changes sign.  The two agree exactly on coarse
    grids and diverge in the WELL-RESOLVED regime -- that is, on precisely the converged
    values one quotes.  With this set the sequence is monotone.
    """
    rho = [0.30 * b, 0.60 * b, 0.90 * b, 0.30 * b, 2.00 * b, 0.70 * b]
    z = [1.10 * b, 0.50 * b, 0.60 * b, -1.10 * b, 0.40 * b, -0.50 * b]
    phi = [0.0, math.pi / 2, math.pi / 4, math.pi, math.pi / 3, 1.0]
    rng = np.random.default_rng(seed)
    while len(rho) < 6 + n_dense:
        rr, zz = rng.uniform(0.12, 2.3) * b, rng.uniform(-1.35, 1.35) * b
        if math.hypot(rr, zz - b) > 0.18 * b and math.hypot(rr, zz + b) > 0.18 * b:
            rho.append(rr); z.append(zz); phi.append(rng.uniform(0.0, 2.0 * math.pi))
    return np.array(rho), np.array(z), np.array(phi)


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


def _phi_spectrum(U):
    """``|u_m|`` per azimuthal mode: max over the meridian of the real FFT in phi."""
    A = np.asarray(U)
    F = np.fft.rfft(A, axis=2) / A.shape[2]
    return np.max(np.abs(F), axis=(0, 1))


def _chi_fields(theta):
    chi = np.asarray(theta[2:], dtype=float)
    return dict(chi=[float(x) for x in chi],
                # the box coordinate is PER COMPONENT, so the box is a hyper-rectangle and
                # its corners carry |chi| up to sqrt(3)*CHI_MAX ~ 1.56 -- beyond the
                # horizon-spin ceiling that sets the half-width.  Record both.
                chi_absmax=float(np.max(np.abs(chi))),
                chi_mag_A=float(np.linalg.norm(chi[:3])),
                chi_mag_B=float(np.linalg.norm(chi[3:])),
                chi_mag_max=float(max(np.linalg.norm(chi[:3]),
                                      np.linalg.norm(chi[3:]))))


def _one(job):
    """Oracle + the whole resolution ladder for one configuration (worker process)."""
    idx, label, theta, ladder, tp_res, selfconv_nphi, tol, timeout, qc = job
    t0 = time.time()
    out = dict(idx=idx, label=label, theta=[float(x) for x in theta], qc=bool(qc),
               b=float(theta[0]), q=float(theta[1]), **_chi_fields(theta))
    try:
        sl = p3.theta_to_slice3d(theta, NAMES, M_tot=1.0,
                                 fixed=(FIXED_QC if qc else None))
        out.update(m_A=float(sl.m_A), m_B=float(sl.m_B),
                   S_A=[float(x) for x in sl.S_A_vec], S_B=[float(x) for x in sl.S_B_vec],
                   P_A=[float(x) for x in sl.P_A_vec], P_B=[float(x) for x in sl.P_B_vec])
        rho, z, phi = _probe_points(sl.b)

        ref = tp.solve_lm_initial_data_points_3d(
            sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec,
            rho=rho, z=z, phi=phi,
            nA=tp_res[0], nB=tp_res[1], nphi=tp_res[2], timeout=timeout)

        if selfconv_nphi:                  # bound the ORACLE's own error on a subset
            lo = tp.solve_lm_initial_data_points_3d(
                sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec,
                rho=rho, z=z, phi=phi,
                nA=tp_res[0], nB=tp_res[1], nphi=selfconv_nphi, timeout=timeout)
            out["tp_selfconv_dpsi"] = float(np.max(np.abs(ref.psi - lo.psi)))
            out["tp_selfconv_dE"] = float(abs(ref.E - lo.E) / abs(ref.E))

        psiBL = np.asarray(source.psi_BL_2c(rho, z, sl.b, sl.m_A, sl.m_B))
        rungs = []
        for g in ladder:
            prob = s3.make_problem(Na=g[0], Nb=g[1], Nphi=g[2])
            U, info = s3nk.newton_solve_nk(prob, sl, tol=tol, max_iter=40, gmres_rtol=1e-8)
            u = np.asarray(s3.evaluate_field(prob, U, rho, z, phi, sl.b))
            M_adm = _adm_mass_3d(prob, U, sl)
            amps = _phi_spectrum(U)
            a0 = float(amps[0]) if amps[0] > 0 else 1.0
            dpsi = np.abs(psiBL + u - ref.psi)
            rungs.append(dict(
                grid=list(g),
                max_dpsi=float(np.max(dpsi)),                  # the reported sup-norm
                max_dpsi_legacy6=float(np.max(dpsi[:6])),      # the old 6-probe estimate
                l2_dpsi=float(np.sqrt(np.mean(dpsi ** 2))),    # stabler, for cross-check
                M_ADM=M_adm,
                M_ADM_rel_diff=float(abs(M_adm - ref.E) / abs(ref.E)),
                residual=float(info.residual_norm),      # EQUILIBRATED (2.1)
                raw_residual=float(info.raw_residual_norm),
                certified=bool(info.residual_norm <= CERT_TOL),
                iters=int(info.iters), converged=bool(info.converged),
                spectrum=[float(a / a0) for a in amps]))
        out.update(ok=True, tp_E=float(ref.E), rungs=rungs)
    except Exception as e:                 # a failed sample is data, not a crash
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    out["dt"] = time.time() - t0
    return out


def _one_axisym(job):
    """Axisymmetry check for one head-on, collision-axis-spin slice (worker process)."""
    label, theta, grid, tol = job
    out = dict(label=label, theta=[float(x) for x in theta],
               b=float(theta[0]), q=float(theta[1]), **_chi_fields(theta))
    try:
        sl = p3.theta_to_slice3d(theta, NAMES, M_tot=1.0, fixed=None)   # head-on, not QC
        prob = s3.make_problem(Na=grid[0], Nb=grid[1], Nphi=grid[2])
        U, info = s3nk.newton_solve_nk(prob, sl, tol=tol, max_iter=40, gmres_rtol=1e-8)
        amps = _phi_spectrum(U)
        a0 = float(amps[0]) if amps[0] > 0 else 1.0
        rel = [float(a / a0) for a in amps]
        out.update(ok=True, grid=list(grid), spectrum=rel,
                   m_ge1_max=float(max(rel[1:])) if len(rel) > 1 else 0.0,
                   S_A=[float(x) for x in sl.S_A_vec],
                   P_A=[float(x) for x in sl.P_A_vec],
                   residual=float(info.residual_norm))
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


def _band(rows, key, ri):
    """min / median / mean / max of ``key`` at rung ``ri`` -- the band the figure draws."""
    v = np.array([r["rungs"][ri][key] for r in rows if r.get("ok")], dtype=float)
    if v.size == 0:
        return None
    return dict(n=int(v.size), min=float(v.min()), median=float(np.median(v)),
                mean=float(v.mean()), p90=float(np.percentile(v, 90)), max=float(v.max()))


def _spectrum_band(rows, ri, nm):
    """Per-``m`` min/median/max of ``|u_m|/|u_0|`` at rung ``ri``."""
    out = []
    for m in range(nm):
        v = np.array([r["rungs"][ri]["spectrum"][m] for r in rows
                      if r.get("ok") and len(r["rungs"][ri]["spectrum"]) > m], dtype=float)
        if v.size == 0:
            continue
        out.append(dict(m=m, n=int(v.size), min=float(v.min()),
                        median=float(np.median(v)), mean=float(v.mean()),
                        max=float(v.max())))
    return out


def _group(rows, ladder):
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return dict(n=0, ladder=[])
    nm = min(len(r["rungs"][-1]["spectrum"]) for r in ok)
    return dict(
        n=len(ok),
        ladder=[dict(grid=list(g),
                     psi=_band(ok, "max_dpsi", i),
                     psi_legacy6=_band(ok, "max_dpsi_legacy6", i),
                     psi_l2=_band(ok, "l2_dpsi", i),
                     M_ADM=_band(ok, "M_ADM_rel_diff", i),
                     residual=_band(ok, "residual", i),
                     raw_residual=_band(ok, "raw_residual", i),
                     n_uncertified=int(sum(1 for r in ok
                                           if not r["rungs"][i]["certified"])))
                for i, g in enumerate(ladder)],
        spectrum_top=_spectrum_band(ok, len(ladder) - 1, nm),
        spectrum_prod=_spectrum_band(ok, 0, nm))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100, help="interior sample size (default 100)")
    ap.add_argument("--sampler", choices=("lhs", "sobol"), default="lhs")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--ladder", type=str, default=DEFAULT_LADDER,
                    help="';'-separated NA,NB,NPHI rungs, coarsest first. Every rung shares "
                         "the one oracle call per configuration.")
    ap.add_argument("--tp-res", type=int, nargs=3, default=[72, 72, 12],
                    metavar=("NA", "NB", "NPHI"),
                    help="oracle resolution; must out-resolve every rung, phi included. "
                         "(72,72) leaves margin over the top rung's meridian (64,44) -- the "
                         "direction that limits the agreement -- and nphi=12 is phi-converged "
                         "(3.7e-13 against nphi=16), so the oracle floors nothing reported.")
    ap.add_argument("--selfconv", type=int, default=6,
                    help="re-run the first K configurations at a lower oracle nphi to "
                         "bound the oracle's own error (0 disables)")
    ap.add_argument("--selfconv-nphi", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-12, help="solver target (equilibrated)")
    ap.add_argument("--timeout", type=int, default=7200, help="per-oracle-call timeout (s)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-edges", action="store_true", help="skip the edge stress set")
    ap.add_argument("--no-axisym", action="store_true", help="skip the axisymmetry checks")
    ap.add_argument("--no-anchor", action="store_true",
                    help="skip the head-on axisymmetric code-to-code anchor")
    ap.add_argument("--out", default=os.path.join(REPDIR, "tp_band_sweep.json"))
    args = ap.parse_args()

    if not tp.available():
        raise SystemExit(f"TwoPunctures oracle not found at {tp.binary_path()!r}; "
                         f"build it with `make oracle` (see docs/DATA.md).")

    ladder = [tuple(int(x) for x in g.split(",")) for g in args.ladder.split(";") if g.strip()]
    if any(len(g) != 3 for g in ladder):
        raise SystemExit(f"--ladder must be ';'-separated NA,NB,NPHI triples, got {args.ladder!r}")
    if max(g[2] for g in ladder) >= args.tp_res[2]:
        print(f"[warn] the oracle must OUT-RESOLVE every rung, but max Nphi_rung="
              f"{max(g[2] for g in ladder)} >= nphi_oracle={args.tp_res[2]}: the reported "
              f"agreement would be floored by the oracle's own truncation.", flush=True)

    interior = _sample(args.n, args.sampler, args.seed)
    edges = [] if args.no_edges else _edge_cases()
    axisym = [] if args.no_axisym else _axisym_cases()

    jobs = []
    for i, th in enumerate(interior):
        sc = args.selfconv_nphi if i < args.selfconv else 0
        jobs.append((i, None, list(th), ladder, tuple(args.tp_res), sc, args.tol,
                     args.timeout, True))
    for j, (lab, th) in enumerate(edges):
        jobs.append((len(interior) + j, lab, list(th), ladder, tuple(args.tp_res),
                     0, args.tol, args.timeout, True))
    if not args.no_anchor:                 # head-on (qc=False): the axisymmetric anchor
        alab, ath = _anchor_case()
        jobs.append((len(interior) + len(edges), alab, list(ath), ladder,
                     tuple(args.tp_res), args.selfconv_nphi, args.tol, args.timeout, False))

    print("=== TwoPunctures validation: bands over the production box ===")
    print("  box      : " + ", ".join(f"{a['name']}in[{a['min']:g},{a['max']:g}]" for a in AXES))
    print(f"  sample   : {args.n} {args.sampler.upper()} interior + {len(edges)} edge "
          f"(seed {args.seed})")
    print(f"  ladder   : {ladder}  (NK certified, tol {args.tol:g})")
    print(f"  oracle   : TwoPunctures {tuple(args.tp_res)}"
          f"{f', nphi={args.selfconv_nphi} self-conv on first {args.selfconv}' if args.selfconv else ''}")
    print(f"  workers  : {args.workers}   ({len(jobs)} configurations + {len(axisym)} axisym)\n",
          flush=True)

    t0 = time.time()
    rows, ax_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, j): j[0] for j in jobs}
        for lab, th in axisym:             # cheap, no oracle
            futs[ex.submit(_one_axisym, (lab, th, ladder[0], args.tol))] = f"ax:{lab}"
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if "m_ge1_max" in r or (not r.get("ok") and "rungs" not in r and "grid" in r):
                ax_rows.append(r)
                print(f"  [{k:3d}/{len(futs)}] axisym {r['label']:<22s} "
                      f"max_m>=1 |u_m|/|u_0| = {r.get('m_ge1_max', float('nan')):.1e}", flush=True)
                continue
            rows.append(r)
            tag = r.get("label") or f"#{r['idx']:03d}"
            if r.get("ok"):
                first, last = r["rungs"][0], r["rungs"][-1]
                print(f"  [{k:3d}/{len(futs)}] {tag:<24s} b={r['b']:5.2f} q={r['q']:4.2f} "
                      f"|chi|={r['chi_mag_max']:4.2f}  dpsi {first['max_dpsi']:.1e}"
                      f"->{last['max_dpsi']:.1e}  ||R|| {first['residual']:.1e}"
                      f"->{last['residual']:.1e}  [{r['dt']:.0f}s]", flush=True)
            else:
                print(f"  [{k:3d}/{len(futs)}] {tag:<24s} FAILED: {r['error']}", flush=True)
    rows.sort(key=lambda r: r["idx"])
    ax_rows.sort(key=lambda r: r["label"])

    # the anchor is head-on, so it belongs to neither band -- keep it out of both
    anchor = next((r for r in rows if r.get("qc") is False), None)
    inter = [r for r in rows if r.get("label") is None and r.get("qc") is not False]
    edge = [r for r in rows if r.get("label") is not None and r.get("qc") is not False]
    sc = [r["tp_selfconv_dpsi"] for r in rows if "tp_selfconv_dpsi" in r]
    ax_ok = [r for r in ax_rows if r.get("ok")]

    anchor_sum = None
    if anchor is not None and anchor.get("ok"):
        fine = anchor["rungs"][-1]
        anchor_sum = dict(
            label=anchor["label"], theta=anchor["theta"], grid=fine["grid"],
            max_dpsi=fine["max_dpsi"], max_dpsi_legacy6=fine["max_dpsi_legacy6"],
            l2_dpsi=fine["l2_dpsi"], M_ADM_rel_diff=fine["M_ADM_rel_diff"],
            residual=fine["residual"], certified=fine["certified"],
            per_rung=[dict(grid=g["grid"], max_dpsi=g["max_dpsi"],
                           max_dpsi_legacy6=g["max_dpsi_legacy6"],
                           M_ADM_rel_diff=g["M_ADM_rel_diff"]) for g in anchor["rungs"]])

    summary = dict(
        interior=_group(inter, ladder), edge=_group(edge, ladder),
        ladder=[list(g) for g in ladder], anchor=anchor_sum,
        axisym=dict(n=len(ax_ok),
                    m_ge1_max=(float(max(r["m_ge1_max"] for r in ax_ok)) if ax_ok else None)),
        n_failed=int(sum(1 for r in rows if not r.get("ok"))),
        tp_selfconv_dpsi_max=(float(max(sc)) if sc else None),
        wall_s=time.time() - t0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(meta=dict(axes=AXES, names=NAMES, n=args.n, sampler=args.sampler,
                                 seed=args.seed, ladder=[list(g) for g in ladder],
                                 tp_res=list(args.tp_res), selfconv=args.selfconv,
                                 selfconv_nphi=args.selfconv_nphi, tol=args.tol,
                                 cert_tol=CERT_TOL, fixed=FIXED_QC,
                                 box="spin8_qc_chi_prod"),
                      summary=summary, rows=rows, axisym=ax_rows), f, indent=1)

    print(f"\n=== summary ({summary['wall_s'] / 60:.1f} min) ===")
    for grp in ("interior", "edge"):
        G = summary[grp]
        if not G.get("n"):
            continue
        print(f"  {grp} (n={G['n']}):")
        for rung in G["ladder"]:
            p, m, R = rung["psi"], rung["M_ADM"], rung["residual"]
            print(f"    {str(tuple(rung['grid'])):>16s}  dpsi min/med/max "
                  f"{p['min']:.2e}/{p['median']:.2e}/{p['max']:.2e}   relDM "
                  f"{m['min']:.2e}/{m['median']:.2e}/{m['max']:.2e}   ||R|| med "
                  f"{R['median']:.2e} max {R['max']:.2e}  uncert {rung['n_uncertified']}")
    A = summary.get("anchor")
    if A:
        print(f"  axisymmetric anchor ({A['label']}, {tuple(A['grid'])}): "
              f"max|dpsi|={A['max_dpsi']:.2e}  (legacy 6-probe {A['max_dpsi_legacy6']:.2e})  "
              f"relDM={A['M_ADM_rel_diff']:.2e}  ||R||={A['residual']:.1e}")
    if summary["axisym"]["m_ge1_max"] is not None:
        print(f"  axisymmetry (head-on, collision-axis spin, n={summary['axisym']['n']}): "
              f"worst m>=1 |u_m|/|u_0| = {summary['axisym']['m_ge1_max']:.2e}")
    print(f"  failed: {summary['n_failed']}")
    if summary["tp_selfconv_dpsi_max"] is not None:
        print(f"  oracle self-convergence (nphi {args.selfconv_nphi} vs {args.tp_res[2]}): "
              f"max {summary['tp_selfconv_dpsi_max']:.2e}  (floor of the comparison)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
