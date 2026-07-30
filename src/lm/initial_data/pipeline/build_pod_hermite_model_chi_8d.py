"""PARASOL — build & persist the SHIPPED gradient-enhanced (Hermite) 8-D model in
the DIMENSIONLESS-SPIN (chi) parameterization (chi-rebuild S7).

Add-only 8-D twin of ``build_pod_hermite_model_chi.py`` (the 4-D S4 driver).  Like
that driver it does NOT copy the committed builder's body; it imports
``build_pod_hermite_model``, swaps the module-level box to the full 8-D chi spin box
``spin8_qc_chi_b27`` = the production 8-D box of ``production_box`` (matching
the S6 value corpus box), and dispatches to ``build_pod_hermite_model.main()``
verbatim — so the Newton–Krylov solve, the QC chain-rule certified tangent, the
H5d POD compression, the save, and the certified spot-check are reused byte-for-byte
and the committed builder is untouched.

The enhanced (gradient) axes default to the SIX dimensionless-spin components
``chi_Ax,chi_Ay,chi_Az,chi_Bx,chi_By,chi_Bz`` (b,q stay value-only, mirroring the
4-D design where only the slow spin axes carry tangents).  The chi certified tangent
(``∂/∂chi = m^2·∂/∂S``) is the committed ``sensitivity_3d[_qc]`` addition; it maps
every component generically (``sensitivity_3d.py``: ``chi_A{x,y,z}``→ ``dS_A[·]·m_A^2``,
``chi_B{x,y,z}``→ ``dS_B[·]·m_B^2``), so the in-plane components (x,z) are supported
by the same code path as the aligned (y) pair.  For the QC momenta the leading
aligned spin-orbit term depends only on the y-spins, so ``∂(momenta)/∂chi`` is zero
for the in-plane components and the in-plane tangent reduces (correctly, at this PN
order) to the direct Bowen–York spin-source tangent — this falls out of the
autodiff chain rule automatically.

Usage — fastest: reuse the S6 8-D value corpus (compute only tangents; no re-solve):
    python sandbox/parasol/build_pod_hermite_model_chi_8d.py \
        --Na 44 --Nb 32 --Nphi 8 --level 5 \
        --enhanced chi_Ax,chi_Ay,chi_Az,chi_Bx,chi_By,chi_Bz \
        --reuse-value sandbox/parasol/reports/3D_parametric/models_chi/surrogate_smolyak_spin8_qc_chi_b27_L5.npz \
        --outdir sandbox/parasol/reports/P2/models_chi

Smoke (fast, from scratch, small grid+level; exercises all six enhanced spin axes —
incl. the in-plane components — through the QC tangent, POD, and the certified
spot-check):
    --Na 16 --Nb 12 --Nphi 6 --level 2

Cost note: the reuse-value path is one process that loads the whole 8-D value
corpus and computes 6 tangent back-solves per dedup node (one shared ``s3.assemble``
per node).  Budget from the S4 4-D reuse (~2.6 s/node for 2 axes): ~4-7 s/node × the
8-D dedup pool.  The H5d POD then compresses a 7-field-per-node corpus (value + 6
tangents), so give the job generous ``--time`` and ``--mem`` (see SESSION_STATE §2).
"""
from __future__ import annotations

import os
import sys


import build_pod_hermite_model as bh  # noqa: E402  (committed builder, reused verbatim)

from lm.initial_data.pipeline import production_box as pb  # noqa: E402

CHI_MAX = pb.CHI_MAX

# swap the module-level box to the 8-D chi spin box (main() reads BOX/FIXED as
# globals).  The production separation range's feasibility was de-risked by
# derisk_b27.py (every hard corner certified <= 7.2e-12).  Must match the value
# corpus box for --reuse-value (spin8_qc_chi_b27), which is why both come from
# production_box.  All six spin components are box axes; b,q stay value-only.
bh.BOX = pb.spin8_box()
# FIXED = {"qc": 1.0} is identical for the chi QC family — no change needed.

# the six dimensionless-spin components (the enhanced/gradient axes; b,q value-only)
SPIN_AXES = ",".join(pb.SPIN8_AXES)


if __name__ == "__main__":
    # default the enhanced axes to the six chi spins if the caller did not specify
    # (the committed default "S_Ay,S_By" is not in the 8-D chi box).
    if not any(a == "--enhanced" or a.startswith("--enhanced=") for a in sys.argv[1:]):
        sys.argv += ["--enhanced", SPIN_AXES]
    bh.main()
