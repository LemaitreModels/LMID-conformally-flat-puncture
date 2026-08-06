"""Acceptance — the shipped-model single source of truth and its slim storage.

Guards the five confusions catalogued in
:mod:`lm.initial_data.pipeline.production_model`:

  * the shipped model's identity (enhanced axes, cross term, ranks) is stated once
    and the axis INDICES differ between the two boxes;
  * ``stored_blocks`` reproduces every historical accounting, so no past number is
    orphaned, and the shipped one is independent of ``d``;
  * the memory formulas reproduce the numbers the paper and Fig. 5 have quoted;
  * the slim ``.npz`` layout is **lossless** — the dead tangent blocks it drops are
    provably never read, so a slim round-trip evaluates bit-for-bit like a full one
    while the file is smaller by exactly the predicted ratio.

The slim gate builds a genuine (small) cross model through the QC wiring rather
than a mock, so it exercises the real save/load path.
"""

import os
import tempfile

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from lm.initial_data.pipeline import production_box as pb
from lm.initial_data.pipeline import production_model as pm
from lm.initial_data.parametric.hermite_smolyak import isotropic_index_set
from lm.initial_data.parametric.parametric_nd_smolyak import _node_key
from lm.initial_data.parametric.hermite_smolyak_cross import (
    HermiteSmolyakCrossSolverND, build_cross_from_pool)
from lm.initial_data.parametric.hermite_smolyak_pod_cross import (
    build_pod_hermite_smolyak_cross, load_pod_hermite_smolyak_cross)

MiB, GiB = 2**20, 2**30


# ---------------------------------------------------------------- identity ----
def test_shipped_identity():
    """Enhanced axes, cross term and ranks are stated once, and indices are per-box."""
    assert pm.ENHANCED_AXES == ("chi_Ay", "chi_By") == pb.ALIGNED_SPIN_AXES
    assert pm.CROSS and pm.n_enhanced() == 2 and pm.n_pairs() == 1
    assert pm.SHIPPED_RANK == {4: 250, 8: 500}
    # same NAMES, different INDICES -- the trap enhanced_indices() exists to close
    assert pm.enhanced_indices(4) == (2, 3)
    assert pm.enhanced_indices(8) == (3, 6)
    for dim in (4, 8):
        names = [a["name"] for a in (pb.aligned_box() if dim == 4 else pb.spin8_box())]
        assert [names[i] for i in pm.enhanced_indices(dim)] == list(pm.ENHANCED_AXES)
    # the artifact name carries model + rank; the LAYOUT lives in the file's meta
    assert pm.pod_stem(4) == "pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r250"
    assert pm.pod_stem(8).endswith("_r500")


def test_stored_blocks_families():
    """Every historical accounting stays labelled; the shipped one is d-independent."""
    assert pm.stored_blocks("value") == 1
    assert pm.stored_blocks("shipped") == 4 == pm.stored_blocks("shipped", 8)
    assert pm.stored_blocks("gradient_all", 4) == 5
    assert pm.stored_blocks("cross_all", 4) == 6
    assert pm.stored_blocks("gradient_all", 8) == 9
    assert pm.stored_blocks("cross_all", 8) == 10
    with pytest.raises(ValueError):
        pm.stored_blocks("no_such_family")
    with pytest.raises(ValueError):       # d-dependent family needs a dim
        pm.stored_blocks("gradient_all")


def test_memory_reproduces_quoted_numbers():
    """The formulas regenerate each number that has appeared in the paper or Fig. 5."""
    # paper text before 2026-08: gradient_all, no cross
    assert pm.bare_bytes(4, "gradient_all") / MiB == pytest.approx(485.6, abs=0.1)
    assert pm.bare_bytes(8, "gradient_all") / GiB == pytest.approx(12.14, abs=0.01)
    # Fig. 5 bare stars before 2026-08: cross_all
    assert pm.bare_bytes(4, "cross_all") / MiB == pytest.approx(582.7, abs=0.1)
    assert pm.bare_bytes(8, "cross_all") / GiB == pytest.approx(13.49, abs=0.01)
    # superseded shipped ranks, in the accounting that produced 10.0 MiB / 420 MiB
    assert pm.pod_bytes(76, 4, "gradient_all") / MiB == pytest.approx(10.0, abs=0.1)
    assert pm.pod_bytes(359, 8, "gradient_all") / MiB == pytest.approx(419.0, abs=1.0)
    # what the paper now quotes: the shipped y-pair cross model at r=250 / r=500
    assert pm.bare_bytes(4) / MiB == pytest.approx(388.5, abs=0.1)
    assert pm.bare_bytes(8) / GiB == pytest.approx(5.395, abs=0.005)
    assert pm.pod_bytes(250, 4) / MiB == pytest.approx(30.5, abs=0.1)
    assert pm.pod_bytes(500, 8) / MiB == pytest.approx(283.8, abs=0.5)
    assert pm.compression_factor(4) == pytest.approx(12.7, abs=0.1)
    assert pm.compression_factor(8) == pytest.approx(19.5, abs=0.1)


# ------------------------------------------------------------ slim storage ----
# A synthetic analytic corpus, not an elliptic solve: the slim layout is a
# PERSISTENCE property, so the fastest test that exercises the real save/load path
# is the honest one.  The shipped shape is reproduced exactly where it matters --
# three axes with only the last two enhanced, so the layout genuinely has a dead
# block to drop, and one bilinear cross pair, as in production.
AXES = [(2.6, 3.2), (0.0, 0.3), (0.0, 0.3)]
ENH = (1, 2)
LEVEL = 2
FIELD_SHAPE = (3, 4)
_BASE = np.outer(np.linspace(1.0, 2.0, FIELD_SHAPE[0]),
                 np.linspace(0.5, 1.5, FIELD_SHAPE[1]))


def _u(th):
    b, x, y = th
    return _BASE * (np.sin(b) + 0.3 * x * x + 0.2 * y + 0.5 * x * y)


def _du(th):
    b, x, y = th
    return np.stack([_BASE * np.cos(b), _BASE * (0.6 * x + 0.5 * y),
                     _BASE * (0.2 + 0.5 * x)])


def _cross(th):
    return (0.5 * _BASE)[None, ...]          # one pair (1,2): d2u/dx dy


@pytest.fixture(scope="module")
def cross_pod():
    solver = HermiteSmolyakCrossSolverND(solve_fn=None, axes=AXES, tangent_fn=None,
                                         enhanced_axes=ENH)
    index_set = isotropic_index_set(len(AXES), LEVEL)
    pool = {}
    for l in sorted(index_set, key=sum):
        nodes, _ = solver._subgrid_nodes(l)
        for idx in np.ndindex(*(len(n) for n in nodes)):
            th = np.array([nodes[k][idx[k]] for k in range(len(AXES))], dtype=float)
            pool.setdefault(_node_key(th), (th, _u(th), _du(th), _cross(th), 1, 0.0))
    model = build_cross_from_pool(AXES, index_set, ENH, pool)
    pod, _diag = build_pod_hermite_smolyak_cross(model, r=8)
    return pod


def test_slim_roundtrip_is_lossless(cross_pod):
    """Slim and full layouts load to the same model and evaluate bit-for-bit.

    The dropped blocks are the tangents at NON-enhanced axes, which
    ``HermiteCrossSolutionND.evaluate`` never reads; dropping them must therefore
    change nothing observable.  A byte-size check confirms they really left the file.
    """
    d, enh = cross_pod.d, tuple(sorted(int(e) for e in cross_pod.enhanced))
    with tempfile.TemporaryDirectory() as tmp:
        p_slim = cross_pod.save(os.path.join(tmp, "slim.npz"), slim=True)
        p_full = cross_pod.save(os.path.join(tmp, "full.npz"), slim=False)

        raw_slim = np.load(p_slim)
        raw_full = np.load(p_full)
        assert raw_slim["node_dU"].shape[1] == len(enh)      # only enhanced blocks
        assert raw_full["node_dU"].shape[1] == d
        assert os.path.getsize(p_slim) < os.path.getsize(p_full)

        m_slim = load_pod_hermite_smolyak_cross(p_slim)
        m_full = load_pod_hermite_smolyak_cross(p_full)
        assert m_slim.meta["dU_layout"] == "enhanced"
        assert m_full.meta["dU_layout"] == "full"

        rng = np.random.default_rng(0)
        for _ in range(4):
            th = np.array([rng.uniform(lo, hi) for lo, hi in AXES])
            u_slim, u_full, u_mem = (m_slim.evaluate(th), m_full.evaluate(th),
                                     cross_pod.evaluate(th))
            assert np.array_equal(u_slim, u_full)            # bit-for-bit
            np.testing.assert_allclose(u_slim, u_mem, rtol=0, atol=1e-13)


def test_slim_drops_only_dead_blocks(cross_pod):
    """Teeth: slimming preserves the ENHANCED tangent blocks exactly and zeros the
    others -- and the others are precisely the slots the evaluator never reads."""
    enh = set(int(e) for e in cross_pod.enhanced)
    assert enh and enh != set(range(cross_pod.d)), "test needs a dead block to drop"
    with tempfile.TemporaryDirectory() as tmp:
        m = load_pod_hermite_smolyak_cross(
            cross_pod.save(os.path.join(tmp, "s.npz"), slim=True))
        for key, (_, _, dU_mem, _, _, _) in cross_pod.coeff_model.pool.items():
            dU_disk = m.coeff_model.pool[key][2]
            for a in range(m.d):
                if a in enh:
                    assert np.array_equal(dU_disk[a], dU_mem[a]), "enhanced tangent altered"
                else:
                    assert np.all(dU_disk[a] == 0.0), "a dead block survived the slimming"


# ------------------------------------------- fig05 distiller (no corpora needed) ----
def _fig05_mod():
    """Import the fig05 data script without the heavy sources it would load at build()."""
    import importlib.util
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "paper", "figures")
    spec = importlib.util.spec_from_file_location(
        "fig05_data", os.path.join(here, "fig05_guess_vs_memory_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw(npair=1, blocks=None, r=(1, 250)):
    d = dict(N=1105, nfeat=pm.N_FEAT, d=4, npair=npair, r_full=4420,
             pod_curve=[dict(r=int(x), mem_bytes=-1.0, min=1e-4, median=1e-3, max=1e-2)
                        for x in r])
    if blocks is not None:
        d["blocks"] = blocks
    return d


def test_fig05_recomputes_memory_from_the_shipped_accounting():
    """The distiller must IGNORE the sweeps' stored (d-keyed) memory and recompute it.

    This is what corrects Fig. 5 from the existing sweeps without re-running them, so
    it is gated here rather than only at build time (the 4-D sources are cluster-side).
    """
    f5 = _fig05_mod()
    raw = _raw()
    c = f5.curve(raw, dict(min=1.0, median=2.0, max=3.0), "C1", "s", "above", True,
                 kind="cross")
    assert c["blocks"] == 4                                     # 1 + n_enh + npair
    # bare star and every ladder point are the shipped accounting, not the raw's
    assert c["bare_mem"] * 1e6 == pytest.approx(pm.bare_bytes(4), rel=1e-12)
    assert c["bare_mem"] * 1e6 / MiB == pytest.approx(388.5, abs=0.1)
    for pt in c["cur"]:
        assert pt["mem_bytes"] == pytest.approx(
            pm.pod_bytes_of(pt["r"], 1105, pm.N_FEAT, 4), rel=1e-12)
        assert pt["mem_bytes"] > 0                              # the raw's -1 is gone
    # the value curve stores one field per node
    cv = f5.curve(_raw(), dict(min=1.0, median=2.0, max=3.0), "C0", "o", "below", True,
                  kind="value")
    assert cv["blocks"] == 1
    assert cv["bare_mem"] * 1e6 == pytest.approx(pm.bare_bytes(4, "value"), rel=1e-12)


def test_fig05_rejects_a_sweep_of_a_different_model():
    """Teeth: a sweep that stored a different block count is a DIFFERENT model, and
    must abort the build rather than quietly rescale its memory axis."""
    f5 = _fig05_mod()
    assert f5.blocks_of(_raw(blocks=4), "cross") == 4            # agreeing raw is fine
    with pytest.raises(SystemExit, match="different model"):
        f5.blocks_of(_raw(blocks=6), "cross")                    # the all-axis corpus


def test_fig05_meta_pins_the_model_and_ranks():
    """The provenance block records what a caption would otherwise only assert."""
    f5 = _fig05_mod()
    m = f5._meta(_raw(), dict(N=15713))
    assert m["enhanced_axes"] == list(pm.ENHANCED_AXES)
    assert m["blocks_per_node"] == 4
    assert m["shipped_rank"] == {"4": 250, "8": 500}
    assert m["model"]["8"].endswith("_cross")
    assert m["compression"]["4"] == pytest.approx(12.7, abs=0.1)
    assert m["compression"]["8"] == pytest.approx(19.5, abs=0.1)
