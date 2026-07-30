"""PARASOL — build & persist the SHIPPED gradient-enhanced (Hermite) model in the
DIMENSIONLESS-SPIN (chi) parameterization (chi-rebuild S4).

Add-only chi twin of ``build_pod_hermite_model.py``.  It does NOT copy that
module's body; it imports it, swaps the 4-D QC box to the chi box
``d4_qc_chi_b27`` = the production 4-D aligned box of ``production_box``
(rev-2 R4 separation range), and dispatches to ``build_pod_hermite_model.main()``
verbatim — so the Newton–Krylov solve, the QC chain-rule tangent, the H5d POD
compression, the save, and the certified spot-check are reused byte-for-byte and
the committed builder is untouched.

The enhanced (gradient) axes default to the two slow aligned dimensionless spins
``chi_Ay,chi_By`` (injected into sys.argv if you don't pass --enhanced).  The chi
certified tangent (``∂/∂chi = m^2·∂/∂S``) is the committed sensitivity_3d[_qc]
addition (verified: FD 1.1e-8, m^2-identity 1.2e-15).

Usage (mirrors build_pod_hermite_model.py):
    python sandbox/parasol/build_pod_hermite_model_chi.py \
        --Na 44 --Nb 32 --Nphi 8 --level 5 --enhanced chi_Ay,chi_By \
        --outdir sandbox/parasol/reports/P2/models_chi

    # fastest: reuse the S3 value corpus (skip the re-solve, compute only tangents)
    ... --reuse-value sandbox/parasol/reports/3D_parametric/models_chi/surrogate_smolyak_d4_qc_chi_L5.npz

Smoke: --Na 16 --Nb 12 --Nphi 6 --level 2 (enhanced chi_Ay,chi_By).
"""
from __future__ import annotations

import os
import sys


import build_pod_hermite_model as bh  # noqa: E402  (committed builder, reused verbatim)

from lm.initial_data.pipeline import production_box as pb  # noqa: E402

CHI_MAX = pb.CHI_MAX

# swap the module-level box to the chi box (main() reads BOX/FIXED as globals).
# The production separation range's feasibility was de-risked by derisk_b27.py
# (every hard corner certified <= 7.2e-12).  Must match the value corpus box for
# --reuse-value, which is why both come from production_box.
bh.BOX = pb.aligned_box()
# FIXED = {"qc": 1.0} is identical for the chi QC family — no change needed.


if __name__ == "__main__":
    # default the enhanced axes to the chi spins if the caller did not specify
    # (the committed default is "S_Ay,S_By", which is not in the chi box).
    if not any(a == "--enhanced" or a.startswith("--enhanced=") for a in sys.argv[1:]):
        sys.argv += ["--enhanced", "chi_Ay,chi_By"]
    bh.main()
