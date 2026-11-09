"""``pipeline.check_model_certification`` refuses an uncertified node and a stale prescription.

The rebuild chains run it after every model-building step, so it must fail on exactly the
two defects it exists for and pass a good artifact.  Synthetic ``.npz`` files with the real
``meta_json`` layout; no solve.
"""
import json

import numpy as np

from lemaitre.initial_data.conformally_flat_puncture.parametric.quasicircular import QC_PRESCRIPTION
from lemaitre.initial_data.conformally_flat_puncture.pipeline import check_model_certification as cmc


def _write(path, residuals, prescription=QC_PRESCRIPTION):
    meta = {"kind": "smolyak"}
    if prescription is not None:
        meta["qc_prescription"] = prescription
    arrays = {"meta_json": np.array(json.dumps(meta))}
    if residuals is not None:
        arrays["residuals"] = np.asarray(residuals, dtype=float)
    np.savez(path, **arrays)
    return str(path)


def test_passes_a_certified_current_model(tmp_path):
    p = _write(tmp_path / "ok.npz", [1e-13, 5e-12, 9e-12])
    assert cmc.check(p) == []
    assert cmc.main([p]) == 0


def test_fails_a_node_above_the_gate_or_not_finite(tmp_path):
    assert cmc.check(_write(tmp_path / "hi.npz", [1e-13, 2e-11])) != []
    assert cmc.check(_write(tmp_path / "nan.npz", [1e-13, np.nan])) != []
    assert cmc.main([_write(tmp_path / "hi2.npz", [3e-7])]) == 1


def test_fails_a_model_from_before_the_spin_orbit_fix(tmp_path):
    p = _write(tmp_path / "old.npz", [1e-13], prescription=None)
    assert any("qc_prescription" in m for m in cmc.check(p))
    assert cmc.check(p, check_prescription=False) == []


def test_a_model_without_residuals_fails_unless_that_is_declared(tmp_path):
    p = _write(tmp_path / "pod.npz", None)
    assert cmc.check(p) != []
    assert cmc.check(p, allow_no_residuals=True) == []
    assert cmc.check(_write(tmp_path / "pod_old.npz", None, prescription=None),
                     allow_no_residuals=True) != []


def test_reads_the_sparse_layout_node_resids(tmp_path):
    """Smolyak / Hermite-Smolyak / cross models store ``node_resids``, not ``residuals``."""
    meta = {"kind": "smolyak", "qc_prescription": QC_PRESCRIPTION}
    p = tmp_path / "sparse.npz"
    np.savez(p, meta_json=np.array(json.dumps(meta)), node_resids=np.array([1e-13, 1e-3]))
    assert cmc.check(str(p)) != []
    assert cmc.main([str(p)]) == 1


def test_a_tensor_of_residuals_is_flattened(tmp_path):
    p = _write(tmp_path / "dense.npz", np.array([[1e-13, 2e-12], [3e-11, 1e-14]]))
    probs = cmc.check(p)
    assert len(probs) == 1 and "3e-11" in probs[0].replace("3.0000000000000004e-11", "3e-11")
