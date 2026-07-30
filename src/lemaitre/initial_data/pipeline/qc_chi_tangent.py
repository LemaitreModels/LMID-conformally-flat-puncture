"""Add-only: the chi-parameterized QC tangent — now a thin PASS-THROUGH.

**Kept as a documented alias to ``sensitivity_3d_qc.certified_tangent_3d_qc`` so
existing imports keep working and so the double-count trap it once caused is on record.
It must add NOTHING.**

History (why this file exists and why it is now a pass-through):
  * Along the chi axis the spin source ``S_X = chi_X * m_X^2(q)`` varies with q, so the
    true ``dU/dq|_chi`` carries a mass->spin chain ``+ sum_i (dU/dS_Xi) dS_Xi/dq``
    beyond the fixed-physical-spin ``dU/dq|_S``.
  * ORIGINALLY ``certified_tangent_3d_qc("q")`` omitted that chain, so this wrapper
    ADDED it (a linear combination of the spin sub-tangents — valid because the shared
    IFT Jacobian makes ``dU/dtheta = J^{-1}(-dR/dtheta)`` linear in dR/dtheta).
  * Commit ``25d120e`` (2026-07-20) then fixed the chain INSIDE
    ``certified_tangent_3d_qc`` itself (``applications/sensitivity_3d_qc.py``).  From
    then on the base is correct for EVERY axis, and a wrapper still adding the chain
    would **double-count** — silently poisoning only the q-axis Hermite (measured
    q-Hermite ~2.7e-2 double-counted vs ~2.6e-7 correct; the committed
    ``peraxis_chi6.json`` was built 2026-07-16, before the base fix, so the wrapper
    was correct THEN — but any re-run after 25d120e was wrong until this change).

The base is now the single source of truth (FD-pinned by
``tests/test_sensitivity_3d_qc.py::test_qc_q_tangent_chi_nonzero``); this wrapper
delegates verbatim for ALL axes.  ``tests/test_qc_chi_tangent.py`` guards that the
wrapper stays bit-for-bit equal to the base (no correction re-introduced).
"""
from __future__ import annotations
import numpy as np
from lemaitre.initial_data.applications import sensitivity_3d_qc as qc

def tangent_qc_chi(prob, U, sl, name, M_tot, *, asm=None, jac="nk", gmres_rtol=1e-8):
    """Certified ``dU/dtheta`` for the chi-parameterized QC family — a PASS-THROUGH
    to ``sensitivity_3d_qc.certified_tangent_3d_qc``, which since commit 25d120e is
    correct for every axis INCLUDING the held-chi mass->spin chain on "q".  This
    wrapper adds nothing: adding the chain again would DOUBLE-COUNT on the q axis
    (see the module docstring).  Signature preserved for existing callers."""
    return np.asarray(qc.certified_tangent_3d_qc(prob, U, sl, name, M_tot,
                                                 asm=asm, jac=jac, gmres_rtol=gmres_rtol))
