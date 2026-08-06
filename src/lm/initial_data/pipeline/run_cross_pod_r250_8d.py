"""PREREQ BUILD: the 8-D y-pair CROSS POD truncated to rank 250.

The 8-D sibling of ``run_cross_pod_figuredata.py`` (4-D), producing the ONE
artifact ``run_polish_fielderr_8d.py``'s POD warm-start family needs: the
full-bilinear y-pair CROSS Hermite-Smolyak model (value + gradient in
``chi_Ay``/``chi_By`` + the mixed 2nd partial ``∂²U/∂χ_Ay∂χ_By``), POD-re-encoded
and truncated to rank 250.

Mirrors the 4-D producer verbatim:
    load_hermite_smolyak_cross(CROSS_8D)
      -> build_pod_hermite_smolyak_cross(mc, r=r_full)     # full-rank cross POD
      -> truncate_pod_cross(pod, 250).save(OUT, meta=...)  # the r=250 slice

Input  : $LM_REPORTS/P2/models_chi/hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross.npz
Output : $LM_REPORTS/P2/models_chi/pod_hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross_r<rank>.npz

The output name carries ``--rank`` (:func:`out_path`), so building a second rank
does not clobber an existing artifact -- each is a 1--2 h, >100 GB build.  Both
paths resolve through :func:`lm.initial_data.paths.reports_root`, so ``r=250``
lands exactly where ``run_polish_fielderr_8d``'s default looks for it.

Cost (measured from the model's own shapes: N=15713 nodes, nfeat=11520,
d=8, n_enhanced=2, npair=1, 1287 subgrids).  The stacked SVD corpus is only
(11520 x 62852) = 5.8 GB, but the *combination-technique* expansion dominates:
``_assemble_subgrid`` materialises U/dU/cross per subgrid node, ~86 GB for the
field model, and the full-rank POD re-encoding (r_full = min((1+n_enh+npair)*N,
nfeat) = 11520 = nfeat, i.e. no compression) allocates the same again.

    peak RSS ~ 205 GB      runtime ~ 1-2 h      output ~ 340 MB

Request the memory accordingly (``--mem=240G`` on a 256 GB ivs-long node).
``--project-rank`` is the escape hatch if that much is unavailable: it projects
onto the leading ``r`` POD modes instead of all ``r_full`` of them, which drops
the peak to ~125 GB.  ``pod_basis`` always computes the *same* full SVD and then
slices ``Phi[:, :r]``, and ``truncate_pod_cross(pod, 250)`` is exactly that same
slice, so the shipped artifact is numerically the same model (to round-off in the
projection GEMMs); only the peak allocation differs.  ``r_shipped`` is still
recorded as the true full rank either way.

One-shot; NO solver, NO off-node sweep (that is a separate figure artifact).
Reloads the written .npz via ``load_pod_hermite_smolyak_cross`` and asserts
``r==250`` before exiting.

Run:
  python -m lm.initial_data.pipeline.run_cross_pod_r250_8d
  python -m lm.initial_data.pipeline.run_cross_pod_r250_8d --rank 250
  python -m lm.initial_data.pipeline.run_cross_pod_r250_8d --project-rank 250   # low-memory
  # fig04 r=500 revision (~125 GB peak with the matched projection rank):
  python -m lm.initial_data.pipeline.run_cross_pod_r250_8d --rank 500 --project-rank 500
"""
from __future__ import annotations

import argparse
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from lm.initial_data.parametric.parametric_nd import _load_npz, _unpack_meta
from lm.initial_data.parametric.hermite_smolyak_cross import load_hermite_smolyak_cross
from lm.initial_data.parametric.hermite_smolyak_pod_cross import (
    build_pod_hermite_smolyak_cross, truncate_pod_cross,
    load_pod_hermite_smolyak_cross)
from lm.initial_data.paths import reports_root

REPORTS = reports_root()          # heavy corpora root; $LM_REPORTS (see docs/DATA.md)
MODELS = os.path.join(REPORTS, "P2", "models_chi")
CROSS_8D = os.path.join(MODELS, "hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross.npz")


def out_path(rank):
    """Shipped-artifact path for a rank-``rank`` truncation.

    The name carries the rank so that building a second rank does NOT overwrite
    an existing artifact (each of these is a 1--2 h, >100 GB build).
    """
    return os.path.join(MODELS, "pod_hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_"
                                f"cross_r{int(rank)}.npz")


OUT_8D = out_path(250)          # the historical consumer path (fig04 r=250 revision)


def main(rank=250, project_rank=None):
    t0 = time.time()
    meta = _unpack_meta(_load_npz(CROSS_8D))
    names = list(meta["axis_names"])
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    fixed = meta.get("fixed", {}) or {}
    Na, Nb, Nphi = int(meta["Na"]), int(meta["Nb"]), int(meta["Nphi"])
    print(f"[r250-8d] input CROSS ({os.path.getsize(CROSS_8D)/1e9:.1f} GB) "
          f"d={len(box)} grid={Na}x{Nb}x{Nphi} enhanced={meta.get('enhanced')} "
          f"cross_pairs={meta.get('cross_pairs')} fixed={fixed}", flush=True)

    print("[r250-8d] loading cross model ...", flush=True)
    mc = load_hermite_smolyak_cross(CROSS_8D)
    N = mc.n_solver_nodes
    nfeat = int(np.prod(mc.field_shape))
    d = mc.d
    npair = len(mc.cross_pairs_global)
    r_full = (1 + d + npair) * N                          # requested; capped at nfeat by pod_basis
    r_req = r_full if project_rank is None else int(project_rank)
    print(f"[r250-8d] building cross POD (N={N} nfeat={nfeat} d={d} "
          f"npair={npair} project_rank={project_rank}) ... ({time.time()-t0:.0f}s)", flush=True)
    pod, diag = build_pod_hermite_smolyak_cross(mc, r=r_req)
    # ``diag['s']`` is the full stacked spectrum, so its length is the true
    # full rank even when the projection was done at a lower rank.
    r_full = int(len(diag["s"])) if project_rank is not None else int(pod.r)
    print(f"[r250-8d] cross POD r={pod.r} r_full={r_full} ({time.time()-t0:.0f}s)", flush=True)

    rank = int(min(rank, pod.r))
    out = out_path(rank)        # named AFTER the clamp, so it never mislabels the rank
    truncate_pod_cross(pod, rank).save(out, meta={
        "axis_names": names, "box": [list(b) for b in box], "fixed": fixed,
        "Na": Na, "Nb": Nb, "Nphi": Nphi, "level": int(meta.get("level", 5)),
        "enhanced": list(meta.get("enhanced", [])), "r_shipped": int(r_full)})
    print(f"[r250-8d] wrote r={rank} cross POD "
          f"({os.path.getsize(out)/1e6:.0f} MB) -> {os.path.basename(out)}",
          flush=True)

    # ---- reload-verify (zero solves) ----
    pod2 = load_pod_hermite_smolyak_cross(out)
    assert int(pod2.r) == rank, (int(pod2.r), rank)
    assert int(pod2.d) == len(box), (int(pod2.d), len(box))
    assert tuple(pod2.field_shape) == tuple(mc.field_shape), (pod2.field_shape, mc.field_shape)
    print(f"[r250-8d] reload OK: r={pod2.r} d={pod2.d} field_shape={pod2.field_shape}  "
          f"DONE in {(time.time()-t0)/60:.1f} min", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=250,
                    help="rank of the SHIPPED truncation (the consumer expects 250)")
    ap.add_argument("--project-rank", type=int, default=None,
                    help="project onto only the leading r POD modes (memory escape "
                         "hatch; must be >= --rank). Default: the full rank.")
    args = ap.parse_args()
    if args.project_rank is not None and args.project_rank < args.rank:
        ap.error("--project-rank must be >= --rank")
    main(rank=args.rank, project_rank=args.project_rank)
