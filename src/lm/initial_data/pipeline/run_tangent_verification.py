"""LM-initial-data — verification of the exposed parameter sensitivities (paper Appendix B).

Two distinct statements are verified, and they need different references:

  (a) **the tangent equation is the sensitivity of the certified solve.**  On the
      production three-dimensional (Fourier-in-φ) operator, compare the
      implicit-function-theorem tangent ``J·(dU/dθ) = −∂R/∂θ``
      (``applications.sensitivity_3d.certified_tangent_3d``, full Jacobian via GMRES)
      against second-order central finite differences of the certified
      Newton–Krylov solve itself (``fd_tangent_3d``, the independent oracle).
      → paper Table I.

  (b) **the gradient the surrogate exposes is that same sensitivity.**  On the
      two-centre aligned-spin interpolants, compare ``jacfwd`` of the branchless
      barycentric map (``applications.sensitivity.nodal_dU_dtheta``) against
      (i) central finite differences of the *same* surrogate — which verifies the
      automatic differentiation and nothing more — and (ii) the
      implicit-function-theorem tangent computed independently from the certified
      solve at the same parameter point, which is the check that carries the
      certification claim.  → paper Table II.

The residual in (b)(ii) is the interpolation error of the derivative, largest along
the slowly converging ``b`` and ``q`` axes; it decreases exponentially with the
parameter order.

Both panels recompute from the solver in seconds — no model corpus, no cluster.

Run:  python -m lm.initial_data.pipeline.run_tangent_verification --out tangent.json
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from ..solver import solver_abt as sa                       # noqa: E402
from ..solver import solver_3d as s3                        # noqa: E402
from ..solver import solver_3d_nk as s3nk                   # noqa: E402
from ..parametric import parametric_nd_2c as p3             # noqa: E402
from ..applications import sensitivity as sen               # noqa: E402
from ..applications import sensitivity_3d as s3d            # noqa: E402

# --------------------------------------------------------------------------
# Panel (a) — Eq. (tangent) vs finite differences of the certified 3-D solve
# --------------------------------------------------------------------------
OP_GRID = dict(Na=18, Nb=14, Nphi=6)
# Non-axisymmetric quasi-circular slice: tangential momentum along x, so the
# orbital angular momentum is along y and the aligned spins are along y.
OP_SLICE = dict(b=2.4, q=1.7, P_t=0.5, chi_Ay=0.30, chi_By=0.20, M_tot=1.0)
OP_AXES = ("b", "q", "chi_Ay", "chi_By")
OP_H = 1e-4

# --------------------------------------------------------------------------
# Panel (b) — the exposed surrogate gradient vs both references
# --------------------------------------------------------------------------
SUR_GRID = dict(Na=36, Nb=24, P=0.5)
SUR_M_TOT = 1.0
SUR_H = 1e-6
SPEC_BQ = [{"name": "b", "min": 3.0, "max": 12.0, "Q": 12},
           {"name": "q", "min": 1.0, "max": 3.0, "Q": 10}]
SPEC_CHI = [{"name": "chi_A", "min": 0.0, "max": 0.6, "Q": 8},
            {"name": "chi_B", "min": 0.0, "max": 0.6, "Q": 8}]
FIXED_CHI = {"q": 1.5, "b": 4.0}
THETA_BQ = (6.337, 1.713)        # off-node
THETA_CHI = (0.413, 0.227)       # off-node, both spins non-zero


def _rel(a, b) -> float:
    """Maximum relative difference, normalised on the reference ``b``."""
    a = np.asarray(a); b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def operator_tangents(*, grid: Dict = None, slice_: Dict = None,
                      axes=OP_AXES, h: float = OP_H, verbose: bool = True) -> Dict:
    """Panel (a): Eq. (tangent) against central FD of the certified 3-D solve."""
    grid = dict(OP_GRID if grid is None else grid)
    cfg = dict(OP_SLICE if slice_ is None else slice_)
    t0 = time.time()
    prob = s3.make_problem(**grid)
    M, q = cfg["M_tot"], cfg["q"]
    m_A = M * q / (1.0 + q)
    m_B = M / (1.0 + q)
    sl = s3.Slice3D(b=cfg["b"], m_A=m_A, m_B=m_B,
                    P_A_vec=(cfg["P_t"], 0.0, 0.0), P_B_vec=(-cfg["P_t"], 0.0, 0.0),
                    S_A_vec=(0.0, cfg["chi_Ay"] * m_A ** 2, 0.0),
                    S_B_vec=(0.0, cfg["chi_By"] * m_B ** 2, 0.0))
    U, info = s3nk.newton_solve_nk(prob, sl, tol=1e-12, max_iter=40)
    asm = s3.assemble(prob, sl)
    rows = {}
    for name in axes:
        t_ift = s3d.certified_tangent_3d(prob, U, sl, name, M, asm=asm, jac="nk")
        fd = s3d.fd_tangent_3d(prob, sl, name, M, h=h, solver="nk",
                              tol=1e-12, max_iter=40)
        rows[name] = {"h": h, "rel_fd": _rel(t_ift, fd)}
        if verbose:
            print(f"  {name:8s} h={h:.0e}  Eq.(tangent) vs FD {rows[name]['rel_fd']:9.3e}",
                  flush=True)
    return {"panel": "operator", "config": {**grid, **cfg, "h": h,
                                            "residual_norm": float(info.residual_norm)},
            "rows": rows, "wall_clock_s": time.time() - t0}


def surrogate_tangents(*, grid: Dict = None, h: float = SUR_H,
                       verbose: bool = True) -> Dict:
    """Panel (b): jacfwd of the surrogate vs FD of the surrogate and vs Eq. (tangent)."""
    grid = dict(SUR_GRID if grid is None else grid)
    t0 = time.time()
    prob = sa.make_problem(**grid)
    if verbose:
        print("  building (b,q) interpolant ...", flush=True)
    ps_bq = p3.from_problem_nd(prob, SPEC_BQ, M_tot=SUR_M_TOT).build(tol=1e-12, max_iter=20)
    if verbose:
        print("  building (chi_A,chi_B) interpolant ...", flush=True)
    ps_chi = p3.from_problem_nd(prob, SPEC_CHI, M_tot=SUR_M_TOT,
                                fixed=FIXED_CHI).build(tol=1e-12, max_iter=20)

    cases = (("bq", ps_bq, ("b", "q"), None, np.array(THETA_BQ, dtype=float)),
             ("chi", ps_chi, ("chi_A", "chi_B"), FIXED_CHI,
              np.array(THETA_CHI, dtype=float)))
    rows, residuals = {}, {}
    for tag, ps, cn, fixed, theta in cases:
        dU = sen.nodal_dU_dtheta(ps, theta)                       # jacfwd of the surrogate
        sl = p3.theta_to_slice(theta, cn, SUR_M_TOT, fixed)
        U, info = sa.newton_solve(prob, sl, tol=1e-12, max_iter=25)
        residuals[tag] = float(info.residual_norm)
        for k, name in enumerate(cn):
            tp = theta.copy(); tp[k] += h
            tm = theta.copy(); tm[k] -= h
            fd = (np.asarray(ps.evaluate_jax(jnp.asarray(tp)))
                  - np.asarray(ps.evaluate_jax(jnp.asarray(tm)))) / (2.0 * h)
            t_ift = sen.certified_tangent(prob, U, sl, name, SUR_M_TOT)
            rows[name] = {"h": h,
                          "rel_fd": _rel(dU[..., k], fd),
                          "rel_ift": _rel(dU[..., k], t_ift),
                          "theta": theta.tolist()}
            if verbose:
                print(f"  {name:8s} h={h:.0e}  vs FD {rows[name]['rel_fd']:9.3e}"
                      f"   vs Eq.(tangent) {rows[name]['rel_ift']:9.3e}", flush=True)
    return {"panel": "surrogate",
            "config": {**grid, "M_tot": SUR_M_TOT, "h": h,
                       "spec_bq": SPEC_BQ, "spec_chi": SPEC_CHI, "fixed_chi": FIXED_CHI,
                       "theta_bq": list(THETA_BQ), "theta_chi": list(THETA_CHI),
                       "residual_norm": max(residuals.values()),
                       "residual_norm_per_case": residuals},
            "rows": rows, "wall_clock_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", choices=("operator", "surrogate", "both"), default="both")
    ap.add_argument("--out", help="write the result json here")
    args = ap.parse_args()

    out = {}
    if args.panel in ("operator", "both"):
        print("panel (a) — Eq. (tangent) vs FD of the certified 3-D solve", flush=True)
        out["operator"] = operator_tangents()
    if args.panel in ("surrogate", "both"):
        print("panel (b) — exposed surrogate gradient vs both references", flush=True)
        out["surrogate"] = surrogate_tangents()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
