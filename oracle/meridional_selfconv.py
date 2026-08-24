"""Bound the TwoPunctures reference solve's own MERIDIONAL truncation error.

The paper's App.-A footnote states that the reference solve is converged well below
every difference it reports.  That needs a number, and the number has to be a bound
over the sampled box rather than one configuration: the difference grows steeply with
separation -- measured slope d(log10 max|dpsi|)/d(log10 b) ~= 5 -- so a value taken at
close separation understates the wide-separation end by three orders of magnitude.

The measurement is exactly what the sentence says.  For one configuration: solve the
oracle twice at the SAME probe points and the SAME ``nphi``, moving only ``(nA, nB)``
from 72 to 96.  Nothing else varies, so the difference IS the meridional truncation.

Configurations are a genuine SUBSET of fig08's sample.  The LHS draw is regenerated
with fig08's committed ``meta`` (``n=100``, ``sampler=lhs``, ``seed=20260804``) and rows
selected from it -- an LHS draw is not prefix-stable, so asking for fewer points would
give DIFFERENT configurations rather than a subset.

Needs the oracle binary: ``make oracle`` (see ``oracle/README.md``), then

    python oracle/meridional_selfconv.py --rows 0-19 --workers 3

Committed result: ``oracle/meridional_selfconv.json``, 21 configurations, max
1.38e-12 -- which is what ``tests/test_paper_numerals.py`` pins the footnote against.
Each configuration is two solves at 72^2 and 96^2 and costs minutes, so the committed
artifact exists precisely so a reader need not re-run it.
"""
import argparse
import json
import sys
import time

import numpy as np

from lemaitre.initial_data.conformally_flat_puncture.pipeline import (
    run_tp_random_sweep as sw,
)
from lemaitre.initial_data.conformally_flat_puncture.validation import twopunctures as tp

ANCHOR = [3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # fig08 meta.anchor.theta
BOUND = 1e-11                                         # the bound the paper footnote states


def one(theta, nphi, coarse, fine, timeout):
    """max|dpsi| between two oracle solves differing ONLY in meridional resolution."""
    sl = sw.p3.theta_to_slice3d(theta, sw.NAMES, M_tot=1.0, fixed=sw.FIXED_QC)
    rho, z, phi = sw._probe_points(sl.b)
    out = {}
    for tag, n in (("coarse", coarse), ("fine", fine)):
        t = time.time()
        r = tp.solve_lm_initial_data_points_3d(
            sl.b, sl.m_A, sl.m_B, sl.P_A_vec, sl.P_B_vec, sl.S_A_vec, sl.S_B_vec,
            rho=rho, z=z, phi=phi, nA=n, nB=n, nphi=nphi, timeout=timeout)
        out[tag] = r
        out[tag + "_s"] = time.time() - t
    d = np.abs(out["coarse"].psi - out["fine"].psi)
    return dict(b=float(sl.b),
                max_dpsi=float(np.max(d)),
                median_dpsi=float(np.median(d)),
                rel_dE=float(abs(out["coarse"].E - out["fine"].E) / abs(out["fine"].E)),
                psi_scale=float(np.max(np.abs(out["fine"].psi))),
                secs=out["coarse_s"] + out["fine_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="fig08 sample rows to measure")
    ap.add_argument("--rows", default=None,
                    help="explicit LHS row range 'a-b' (overrides --n); omits the anchor")
    ap.add_argument("--workers", type=int, default=1,
                    help="configurations in parallel; each TP solve is single-threaded")
    ap.add_argument("--nphi", type=int, default=12)      # fig08 meta.tp_res[2]
    ap.add_argument("--coarse", type=int, default=72)    # the footnote's 72x72
    ap.add_argument("--fine", type=int, default=96)      # the footnote's 96x96
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="b6_meridional_selfconv.json")
    a = ap.parse_args()

    full = sw._sample(100, "lhs", 20260804)              # fig08 meta: n=100, lhs, seed
    if a.rows:
        lo, hi = (int(x) for x in a.rows.split("-"))
        thetas = [(f"lhs[{i}]", list(full[i])) for i in range(lo, hi + 1)]
    else:
        thetas = [("anchor", ANCHOR)]
        thetas += [(f"lhs[{i}]", list(full[i])) for i in range(a.n)]

    print(f"bounding max|dpsi| between {a.coarse}^2 and {a.fine}^2 at nphi={a.nphi}; "
          f"the paper's footnote claims < {BOUND:.0e}")
    print(f"{'config':>10}  {'b':>6}  {'max|dpsi|':>11}  {'median':>11}  "
          f"{'rel dE':>10}  {'s':>6}")
    rows = []

    def record(label, th, r):
        r["label"], r["theta"] = label, list(map(float, th))
        rows.append(r)
        print(f"{label:>10}  {r['b']:6.2f}  {r['max_dpsi']:11.3e}  "
              f"{r['median_dpsi']:11.3e}  {r['rel_dE']:10.2e}  {r['secs']:6.1f}")
        sys.stdout.flush()

    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(one, th, a.nphi, a.coarse, a.fine, a.timeout): (lbl, th)
                    for lbl, th in thetas}
            for f in as_completed(futs):
                lbl, th = futs[f]
                try:
                    record(lbl, th, f.result())
                except Exception as e:                    # noqa: BLE001
                    print(f"{lbl:>10}  FAILED: {type(e).__name__}: {e}", flush=True)
    else:
        for label, th in thetas:
            try:
                r = one(th, a.nphi, a.coarse, a.fine, a.timeout)
            except Exception as e:                        # noqa: BLE001
                print(f"{label:>10}  FAILED: {type(e).__name__}: {e}")
                continue
            record(label, th, r)

    if not rows:
        print("\nno successful measurements")
        return 1
    mx = max(r["max_dpsi"] for r in rows)
    print(f"\nmax over {len(rows)} configurations: {mx:.3e}")
    print(f"footnote's stated bound             : {BOUND:.3e}")
    print(f"headroom against it                 : {BOUND / mx:.1f}x"
          f"{'' if mx < BOUND else '   *** BOUND EXCEEDED ***'}")
    with open(a.out, "w") as f:
        json.dump(dict(meta=dict(coarse=a.coarse, fine=a.fine, nphi=a.nphi,
                                 n_probe=6 + sw.N_PROBE_DENSE,
                                 probe_seed=sw.PROBE_SEED,
                                 sample="fig08 lhs seed=20260804 n=100, first rows",
                                 bound=BOUND),
                       max_over_sample=mx, rows=rows), f, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
