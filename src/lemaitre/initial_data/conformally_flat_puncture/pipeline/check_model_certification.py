"""Refuse a built model whose nodes are not certified, or that carries another prescription.

A builder records every node's final residual (``residuals`` or ``node_resids``) but never rejects one: the
solve store admits a stored field up to its ``reuse_tol`` (1e-6 by default) and reports it
converged, and a warm-started NK solve that stalls returns its best iterate.  So a corpus can
hold a node above the certification gate without anything saying so.  This check reads the
artifact itself — the raw ``.npz``, no solver, no loader — and exits non-zero if

* any node residual exceeds the gate (``certification.CERT_TOL_U``, the ``u``-norm gate the
  3-D NK builders solve to), or is not finite, or
* the model's ``qc_prescription`` differs from ``quasicircular.QC_PRESCRIPTION`` (a model
  without the key predates the 2026-11-09 spin-orbit fix).

Run it after each model-building step of a rebuild chain::

    python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.check_model_certification \
        <model.npz> [<model.npz> ...] [--gate 1e-11] [--allow-prescription]

A model kind that stores no per-node residuals (a POD re-encoding) fails unless
``--allow-no-residuals`` says that is expected; it is then checked for its prescription only.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from ..parametric.certification import CERT_TOL_U
from ..parametric.parametric_nd import _unpack_meta
from ..parametric.quasicircular import QC_PRESCRIPTION

#: Where a model kind keeps its per-node final residuals: the dense/tensor kinds write
#: ``residuals``, the sparse (Smolyak, Hermite–Smolyak, cross) kinds ``node_resids``.
RESIDUAL_KEYS = ("node_resids", "residuals")


def _meta(data) -> dict:
    try:
        return _unpack_meta(data)
    except ValueError:
        return {}


def check(path: str, gate: float = CERT_TOL_U, check_prescription: bool = True,
          allow_no_residuals: bool = False) -> list:
    """Problems found in the model at ``path`` (empty list = passes); prints a summary."""
    problems = []
    with np.load(path, allow_pickle=False) as data:
        meta = _meta(data)
        key = next((k for k in RESIDUAL_KEYS if k in data), None)
        res = None if key is None else np.asarray(data[key], dtype=float).ravel()
    if res is None:
        if not allow_no_residuals:
            problems.append(f"no per-node residuals ({' / '.join(RESIDUAL_KEYS)}) — pass "
                            "--allow-no-residuals for a kind that stores none (a POD)")
        print(f"{path}: no per-node residuals stored")
    else:
        bad = ~np.isfinite(res) | (res > gate)
        print(f"{path}: {res.size} nodes, max residual {np.nanmax(res):.3e}, "
              f"{int(bad.sum())} above the gate {gate:.0e}")
        if bad.any():
            worst = np.argsort(np.where(np.isfinite(res), res, np.inf))[::-1][:5]
            problems.append(f"{int(bad.sum())} node(s) above {gate:.0e} or non-finite; "
                            f"worst indices {worst.tolist()} = {res[worst].tolist()}")
    if check_prescription:
        got = meta.get("qc_prescription")
        if got != QC_PRESCRIPTION:
            problems.append(f"qc_prescription {got!r} != current {QC_PRESCRIPTION!r}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--gate", type=float, default=CERT_TOL_U)
    ap.add_argument("--allow-prescription", action="store_true",
                    help="skip the qc_prescription check (a non-quasi-circular model)")
    ap.add_argument("--allow-no-residuals", action="store_true",
                    help="accept a model kind that stores no per-node residuals (a POD)")
    a = ap.parse_args(argv)
    failed = False
    for p in a.models:
        probs = check(p, a.gate, not a.allow_prescription, a.allow_no_residuals)
        for msg in probs:
            print(f"  FAIL {p}: {msg}")
        failed |= bool(probs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
