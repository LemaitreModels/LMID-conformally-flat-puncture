"""The quasi-circular momentum prescription travels with a model's FIELDS, not with the code.

The 2026-11-09 spin-orbit fix (2/3 -> 3/4) changed every spinning quasi-circular slice
without changing the grid, the scheme or any key a loader checked, so a corpus built before
it was indistinguishable from a current one.  These gates pin the stamp's path:

* a model solved under the quasi-circular wiring records ``qc_prescription`` when saved;
* re-saving a LOADED model that has no stamp does not acquire today's (it stays legacy);
* ``attach_solve_fn_3d`` refuses a quasi-circular model whose stamp is missing or other;
* a non-quasi-circular model carries no stamp and is not refused.

One tiny Smolyak build on a coarse grid (seconds).
"""
import numpy as np
import pytest

from lemaitre.initial_data.conformally_flat_puncture.parametric import parametric_nd_smolyak as sm
from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd import (
    attach_solve_fn_3d)
from lemaitre.initial_data.conformally_flat_puncture.parametric.quasicircular import (
    QC_PRESCRIPTION)
from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d as s3

QC = {"qc": 1.0, "q": 1.0, "chi_By": 0.0}
BOX = [{"name": "b", "min": 4.0, "max": 6.0}, {"name": "chi_Ay", "min": -0.5, "max": 0.5}]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    prob = s3.make_problem(Na=16, Nb=12, Nphi=4)
    model = sm.from_problem_smolyak_3d(prob, BOX, fixed=QC, solver="nk").build_isotropic(
        1, max_iter=30)
    path = model.save(tmp_path_factory.mktemp("qc") / "m")
    return prob, path


def test_a_quasicircular_build_records_the_prescription(built):
    _, path = built
    assert sm.load_smolyak(path).meta.get("qc_prescription") == QC_PRESCRIPTION


def test_a_loaded_legacy_model_does_not_acquire_todays_stamp(built, tmp_path):
    prob, path = built
    m = sm.load_smolyak(path)
    m.meta = {k: v for k, v in m.meta.items() if k != "qc_prescription"}   # a pre-fix file
    p2 = m.save(tmp_path / "resaved")
    assert "qc_prescription" not in sm.load_smolyak(p2).meta


def test_attach_refuses_a_quasicircular_model_without_the_current_stamp(built, tmp_path):
    prob, path = built
    names = [a["name"] for a in BOX]
    attach_solve_fn_3d(sm.load_smolyak(path), prob, names, fixed=QC)        # current: fine
    old = sm.load_smolyak(path)
    old.meta = {k: v for k, v in old.meta.items() if k != "qc_prescription"}
    with pytest.raises(ValueError, match="momentum prescription"):
        attach_solve_fn_3d(old, prob, names, fixed=QC)
    # a family without the quasi-circular flag is not a quasi-circular model
    attach_solve_fn_3d(old, prob, names, fixed={"q": 1.0, "chi_By": 0.0})
