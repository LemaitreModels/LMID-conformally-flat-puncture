#!/usr/bin/env python
"""Data for fig07_eccentricity: precompute the smooth E_b(b;J) curves to figdata/.

fig07 is the ONLY figure whose plotter used to import jax + the LM-initial-data package and evaluate a
parametric model (surrogate_bpt_ecc.npz) at plot time. This script does that evaluation ONCE and
writes the smooth curves + the certified scan points + the gradient minima as plain arrays to
figdata/fig07_eccentricity.json, so the plotter (and every other figure) is pure-data.

Sources (raw):
  reports/P3/qc_effpot_Jsweep.json                          (key "qc_effpot")   — scan + minima
  reports/3D_parametric/models/surrogate_bpt_ecc.npz        (key "effpot_model") — for the curves
Needs jax + the installed package; no solves are run.

The ``meta`` block records WHICH model artifact produced the curves (file, its build
commit, box, dense Q, grid), because that artifact is gitignored: without it a
re-distill against a different model is invisible in the committed figdata.

Run:  python fig07_eccentricity_data.py
"""
import json
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

from _figdata import load_source, source, dump
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3
from lemaitre.initial_data.conformally_flat_puncture.applications import qc_effpot as E


def _model_meta(path):
    """The shipped model's own provenance blob (``build_surrogate`` writes it).

    The leaf AGENTS.md rule: a figure that evaluates a model artifact records WHICH
    artifact, because the artifact is gitignored and a re-distill against a different
    one is otherwise invisible.  fig06 is the worked example of that going wrong.
    ``build_surrogate`` stores ``meta_json`` on every ``.npz`` it writes; the dense
    build additionally carries ``Q`` (the tensor-Chebyshev order per axis).
    """
    z = np.load(path, allow_pickle=True, mmap_mode="r")
    return json.loads(z["meta_json"].item())


def build():
    d = load_source("qc_effpot")
    Jlist = [float(J) for J in d["Jlist"]]
    n_scan = int(d["n_scan"])
    n_grad = int(d["per_J"][f"{Jlist[0]:.2f}"]["solves_gradient"])

    model_path = source("effpot_model")
    mm = _model_meta(model_path)
    # This script fixes the evaluation grid, while the model carries the grid it was
    # BUILT on.  Those must be the same grid or the curves are evaluated against a
    # model that never saw them, so disagreement is an error rather than a warning
    # (same policy as fig04's shipped-rank guard).
    grid = (44, 32, 8)
    built = tuple(int(mm[k]) for k in ("Na", "Nb", "Nphi")) if all(
        k in mm for k in ("Na", "Nb", "Nphi")) else None
    if built is not None and built != grid:
        raise SystemExit(
            f"fig07: {os.path.basename(model_path)} was built on Na,Nb,Nphi={built} "
            f"but this script evaluates on {grid}. Rebuild the model on {grid}, or "
            f"change `grid` here if the model really moved.")
    prob = s3.make_problem(Na=grid[0], Nb=grid[1], Nphi=grid[2])
    model = E.load_model(model_path, prob)
    V, _ = E.build_effpot_jax(model, prob)

    # global background grid = full scan-b span (same as the old plotter)
    b_all = np.concatenate([np.asarray(d["per_J"][f"{J:.2f}"]["scan_curve"]["b"], float)
                            for J in Jlist])
    bg = np.linspace(float(b_all.min()), float(b_all.max()), 240)
    # The scan spans the whole box, so bg's endpoints ARE Chebyshev-Lobatto nodes
    # of the parametric model, where the jax barycentric quotient is 0/0 and V returns
    # NaN (qc_effpot.off_node).  Evaluate a hair off any such node.  The shift is
    # `_NODE_SHIFT_FRAC` = 2e-3 of the box width (0.014 in b here) -- it is sized for
    # jacfwd accuracy through the barycentric quotient, not for plotting -- which is
    # ~0.2% of the axis and so below the line width, and it keeps the plotted curve
    # finite across the full span instead of dropping its end points.
    box_b = (float(bg[0]), float(bg[-1]))

    # Warm the parametric_nd._jax_consts cache CONCRETELY before anything jits.
    # That cache is populated on first use and keyed on the model; if the first use
    # is inside a jit trace it stores tracers, which escape and raise
    # UnexpectedTracerError on the next concrete call.  Ordering-dependent, so pin
    # it here rather than relying on the loop below running first.
    _ = float(V(0.5 * (box_b[0] + box_b[1]), Jlist[0]))

    per = {}
    for J in Jlist:
        pj = d["per_J"][f"{J:.2f}"]
        bc = float(pj["b_circ_gradient"])
        bs_scan = np.asarray(pj["scan_curve"]["b"], float)
        Ebs_scan = np.asarray(pj["scan_curve"]["Eb"], float)
        b_scan = float(pj["b_circ_scan"])
        # The two locators of the same circular orbit: the certified scan's
        # parabola fit and the surrogate Newton.  The paper quotes their
        # agreement, so carry BOTH the absolute and the relative form -- with one
        # only, a later reading can silently switch which it means.  Relative is
        # taken against the gradient value, the more accurate of the two.
        d_abs = abs(b_scan - bc)
        # The eccentricity ladder, REGENERATED here rather than read from the raw
        # source.  Two reasons, both provenance:
        #   * the committed raw source predates the `_off_node` guard on
        #     `eccentricity`'s bracket, so its `ecc` column is all zeros -- the
        #     pre-d332497 `b' = b0` fallback firing on a NaN box edge, i.e. an
        #     unmarked "perfectly circular" orbit at every rung;
        #   * re-running the producer would recompute the CERTIFIED scan through
        #     the elliptic solver, so its numbers would carry whatever solver state
        #     the tree happens to hold.  `eccentricity` is surrogate-only (no
        #     solve), so recomputing just this block here is reproducible from the
        #     shipped model alone, while `b_circ`/`dEb_db_certified` stay exactly
        #     the certified values the raw source recorded.
        b0s = E.ecc_ladder(bc, box_b)
        eccs = np.array([E.eccentricity(model, prob, float(b0), J, bc, box_b)[0]
                         for b0 in b0s])
        ok = np.isfinite(eccs)
        per[f"{J:.2f}"] = dict(
            scan_b=bs_scan,
            scan_Eb=Ebs_scan,
            b_circ=bc,
            Vg=[float(V(E.off_node(b, model, J, box_b), J)) for b in bg],
            Vc=float(V(bc, J)),                # model value at the gradient minimum
            b_circ_scan=b_scan,
            d_bcirc_abs=d_abs,
            d_bcirc_rel=d_abs / bc,
            # where the DISCRETE scan minimum sits, before the parabola fit: the
            # fit can move b_circ by most of a rung, and the caption's "the scan
            # locates it" is a statement about this, not about the fit.
            scan_argmin_b=float(bs_scan[int(np.argmin(Ebs_scan))]),
            dEb_db_certified=float(pj["dEb_db_certified"]),
            ecc_b0=b0s,
            # null, not NaN, for a rung whose second turning point leaves the box:
            # NaN is not valid JSON and no other committed figdata carries one.
            ecc=[float(e) if f else None for e, f in zip(eccs, ok)],
            n_ecc_measurable=int(ok.sum()),
            b0_max_measurable=float(b0s[ok][-1]) if ok.any() else None,
            e_max=float(eccs[ok].max()) if ok.any() else None,
        )
    meta = dict(
        model_file=os.path.basename(model_path),
        model_git_commit=mm.get("git_commit"),   # build_surrogate's git HEAD, not the
                                                 # solve store's code_tag
        box=mm.get("box"), axis_names=mm.get("axis_names"), fixed=mm.get("fixed"),
        dense_Q=mm.get("Q"),                     # tensor-Chebyshev order per axis
        grid=list(grid), solver=mm.get("solver"), build_tol=mm.get("tol"),
        n_J=len(Jlist),
        box_b=[float(box_b[0]), float(box_b[1])],
        n_scan=n_scan,
        worst_certified_residual=d.get("worst_certified_residual"),
        # `ecc_b0`/`ecc`/`b0_max_measurable`/`e_max` are recomputed here from the
        # model above; everything else per J is the raw source verbatim.  Stated
        # so a reader can tell which numerals the raw artifact backs and which
        # this script does.  (The solve store's `code_tag` is NOT reachable from
        # either source fig07 reads -- the model .npz carries `git_commit` only.)
        ecc_recomputed_from_model=True,
    )
    p = dump("fig07_eccentricity",
             dict(Jlist=Jlist, n_scan=n_scan, n_grad=n_grad, bg=bg, per_J=per,
                  meta=meta))
    print(f"wrote {os.path.relpath(p, SANDBOX)}  ({len(Jlist)} J-slices, |bg|={len(bg)})")
    print(f"   model {meta['model_file']}  commit {meta['model_git_commit']}  "
          f"Q={meta['dense_Q']}  grid={meta['grid']}")


if __name__ == "__main__":
    build()
