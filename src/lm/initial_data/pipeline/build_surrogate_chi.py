"""PARASOL — build & persist the DIMENSIONLESS-SPIN (chi) production surrogates.

Add-only chi twin of ``build_surrogate.py`` (S3 of the chi-rebuild ledger).  It
does NOT copy build_surrogate's body; it imports it and injects two chi boxes,
then dispatches to build_surrogate.main() verbatim — so the store/save/certified-
spot-check machinery is reused byte-for-byte and the committed module is
untouched.

Boxes added (all edges from ``production_box``).  The ``_b27`` pair is the
PRODUCTION box the shipped models are built on; the suffix is a historical
identifier (it encoded the old b in [2,7]) and no longer describes the range:
  * ``d4_qc_chi``       = legacy narrow b in [B_MIN_NARROW, B_MAX_NARROW]
  * ``spin8_qc_chi``    = the same legacy b range, all six chi components
  * ``d4_qc_chi_b27``   = PRODUCTION (b in [B_MIN, B_MAX], q, chi_Ay, chi_By)
  * ``spin8_qc_chi_b27``= PRODUCTION, all six chi components

The chi_* axes are value axes already handled by ``parametric_nd_3d.theta_to_slice3d``
(S_Xi = chi_Xi * m_X^2), so the value-only Smolyak/dense build works with only
the box swap — no tangent/derivative path is used (that is S4's separate concern).

Usage (mirrors build_surrogate.py; add --box d4_qc_chi / spin8_qc_chi):
    python sandbox/parasol/build_surrogate_chi.py \
        --Na 44 --Nb 32 --Nphi 8 --box d4_qc_chi --level 5 --solver modified \
        --store sandbox/parasol/reports/3D_parametric/solve_store_chi \
        --code-tag chi-rebuild \
        --outdir sandbox/parasol/reports/3D_parametric/models_chi

NOTE (store reuse): to reuse the d4_qc-chi Smolyak corpus that
run_qc_walls_sweep_chi.py (S1, block D) populates, pass the SAME store
(solve_store_chi) and the SAME --code-tag it uses ("chi-rebuild"); the code_tag
is part of the store key.  Match the build solver to S1's ("modified") so the
cached nodes are served.  The final certified spot-check always NK-polishes to
the ‖R‖ floor regardless of build solver.
"""
from __future__ import annotations

import os
import sys

# Make the sibling build_surrogate importable when run as
# ``python sandbox/parasol/build_surrogate_chi.py`` from the repo root.

import build_surrogate as bs  # noqa: E402  (committed module, reused verbatim)

from lm.initial_data.pipeline import production_box as pb  # noqa: E402

CHI_MAX = pb.CHI_MAX

# --- legacy narrow-separation variants (NOT production; historical edges) ---
bs.BOXES["d4_qc_chi"] = pb.aligned_box(b_min=pb.B_MIN_NARROW, b_max=pb.B_MAX_NARROW)
bs.BOXES["spin8_qc_chi"] = pb.spin8_box(b_min=pb.B_MIN_NARROW, b_max=pb.B_MAX_NARROW)
bs.FIXED["d4_qc_chi"] = dict(pb.FIXED_QC)
bs.FIXED["spin8_qc_chi"] = dict(pb.FIXED_QC)

# --- production wide-separation variants; feasibility de-risked by derisk_b27.py ---
bs.BOXES["d4_qc_chi_b27"] = pb.aligned_box()
bs.BOXES["spin8_qc_chi_b27"] = pb.spin8_box()
bs.FIXED["d4_qc_chi_b27"] = dict(pb.FIXED_QC)
bs.FIXED["spin8_qc_chi_b27"] = dict(pb.FIXED_QC)


if __name__ == "__main__":
    bs.main()
