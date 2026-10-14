"""build_cross_model_chi: the cross-tangent GMRES tolerance reaches every back-solve.

On the "v2" (Gauss radial) grid the cross tangent's GMRES cannot reach the routine's
own default rtol 1e-11 — at the production grid it runs to ~11 650 iterations, about
three minutes a node, and returns the rtol-1e-9 answer — so the builder passes its own
``CROSS_GMRES_RTOL`` (1e-8, the first tangents' value), takes ``--gmres-rtol``, and
records it in the model's metadata.  Here the solver pieces are replaced by stand-ins
that record their arguments; the corpus is a tiny real Hermite–Smolyak model.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.pipeline import build_cross_model_chi as bcm
from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_box as pb
from lemaitre.initial_data.conformally_flat_puncture.parametric import hermite_smolyak as hsm


def _tiny_model(path, grid=(4, 4, 2)):
    box = pb.spin8_box()
    names = [a["name"] for a in box]
    fs = (grid[0] + 1, grid[1], grid[2])
    rng = np.random.default_rng(0)

    def solve(theta, guess, tol, max_iter):
        return rng.standard_normal(fs), SimpleNamespace(iters=1, residual_norm=1e-15)

    def tangent(theta, U):
        return rng.standard_normal((len(names),) + fs)

    enh = [names.index(a) for a in pb.SPIN8_AXES]
    model = hsm.HermiteSmolyakSolverND(solve, [(a["min"], a["max"]) for a in box],
                                       tangent_fn=tangent, enhanced_axes=enh
                                       ).build_isotropic(1)
    model.save(path, meta=dict(axis_names=names, Na=grid[0], Nb=grid[1], Nphi=grid[2],
                               fixed=dict(pb.FIXED_QC)))
    return model.n_solver_nodes


@pytest.mark.parametrize("rtol", [None, 3e-9])
def test_cross_rtol_reaches_every_backsolve(tmp_path, monkeypatch, rtol):
    src = str(tmp_path / "raw.npz")
    n = _tiny_model(src)
    seen = []

    def fake_cross(prob, U, sl, ni, nj, M_tot, dU_i=None, dU_j=None, asm=None, jac="nk",
                   **kw):
        seen.append(kw.get("gmres_rtol"))
        return 0.5 * (dU_i + dU_j)

    monkeypatch.setattr(bcm.s3, "assemble", lambda prob, sl: None)
    monkeypatch.setattr(bcm.cross, "cross_tangent_3d_qc", fake_cross)
    out = str(tmp_path / "cross.npz")
    kw = {} if rtol is None else dict(gmres_rtol=rtol)
    bcm.main(src, out, enhanced_names=["chi_Ay", "chi_By"], checkpoint=10 ** 6, **kw)

    want = bcm.CROSS_GMRES_RTOL if rtol is None else rtol
    assert bcm.CROSS_GMRES_RTOL == 1e-8
    assert len(seen) == n and set(seen) == {want}
    meta = json.loads(str(np.load(out)["meta_json"]))
    assert meta["cross_gmres_rtol"] == want
    assert meta["enhanced"] == ["chi_Ay", "chi_By"]
