"""PARASOL — build & persist the DIMENSIONLESS-SPIN (chi) production surrogates.

Add-only chi twin of ``build_surrogate.py`` (S3 of the chi-rebuild ledger).  It
does NOT copy build_surrogate's body; it imports it and injects two chi boxes,
then dispatches to build_surrogate.main() verbatim — so the store/save/certified-
spot-check machinery is reused byte-for-byte and the committed module is
untouched.

Boxes added:
  * ``d4_qc_chi``   = (b in [1.5,4], q in [1,3], chi_Ay in [-0.99,0.99],
                       chi_By in [-0.99,0.99]),  fixed {"qc":1.0}
  * ``spin8_qc_chi``= (b, q, chi_Ax..chi_Bz all in [-0.99,0.99]), fixed {"qc":1.0}

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

CHI_MAX = 0.99

bs.BOXES["d4_qc_chi"] = [
    {"name": "b", "min": 2.0, "max": 4.0},          # rev-2 R4 production range (b>=2)
    {"name": "q", "min": 1.0, "max": 3.0},
    {"name": "chi_Ay", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_By", "min": -CHI_MAX, "max": CHI_MAX},
]
bs.BOXES["spin8_qc_chi"] = [
    {"name": "b", "min": 2.0, "max": 4.0},          # rev-2 R4 production range (b>=2)
    {"name": "q", "min": 1.0, "max": 3.0},
    {"name": "chi_Ax", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_Ay", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_Az", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_Bx", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_By", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_Bz", "min": -CHI_MAX, "max": CHI_MAX},
]
bs.FIXED["d4_qc_chi"] = {"qc": 1.0}
bs.FIXED["spin8_qc_chi"] = {"qc": 1.0}

# --- wide-separation variants (b in [2,7]); feasibility de-risked by derisk_b27.py ---
bs.BOXES["d4_qc_chi_b27"] = [
    {"name": "b",      "min": 2.0, "max": 7.0},
    {"name": "q",      "min": 1.0, "max": 3.0},
    {"name": "chi_Ay", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_By", "min": -CHI_MAX, "max": CHI_MAX},
]
bs.BOXES["spin8_qc_chi_b27"] = [
    {"name": "b",      "min": 2.0, "max": 7.0},
    {"name": "q",      "min": 1.0, "max": 3.0},
    {"name": "chi_Ax", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_Ay", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_Az", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_Bx", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_By", "min": -CHI_MAX, "max": CHI_MAX},
    {"name": "chi_Bz", "min": -CHI_MAX, "max": CHI_MAX},
]
bs.FIXED["d4_qc_chi_b27"] = {"qc": 1.0}
bs.FIXED["spin8_qc_chi_b27"] = {"qc": 1.0}


if __name__ == "__main__":
    bs.main()
