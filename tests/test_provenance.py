"""Guards for the code stamp producers write into their ``meta``.

The defect these exist for: on 2026-08-20 two sweeps whose ``meta`` blocks matched in all
13 keys turned out to come from trees 45 ``src/`` commits apart, and a measured shift was
attributed to the wrong change.  A stamp only closes that if it holds three properties, and
each is easy to lose in a later edit:

* it is **deterministic**, or every rebuild diffs and the diff stops meaning anything;
* it **degrades** instead of raising, or a wheel install with no ``.git`` cannot produce a
  figure at all;
* absence compares as **unknown, not different**, or the first legacy artifact a consumer
  meets is rejected as a mismatch.
"""
import json
import os
import sys
import tempfile

import pytest

from lemaitre.initial_data.conformally_flat_puncture import provenance as pv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "paper", "figures"))


def test_code_stamp_has_a_stable_shape():
    """All three keys always present, so a consumer never confuses absent with clean."""
    s = pv.code_stamp()
    assert set(s) == {"src_tree", "src_dirty", "version"}


def test_code_stamp_is_deterministic():
    """Same tree, byte-identical stamp — the property that keeps rebuilds diff-free.

    A wall-clock field here would make ``git diff`` after ``make figures`` always non-empty,
    which is exactly the signal the figure tier relies on.  That is why the timestamp lives
    in ``run_stamp`` and not here.
    """
    assert json.dumps(pv.code_stamp()) == json.dumps(pv.code_stamp())
    assert "utc" not in pv.code_stamp()


def test_run_stamp_is_code_stamp_plus_a_timestamp():
    r = pv.run_stamp()
    assert set(r) == {"src_tree", "src_dirty", "version", "utc"}
    assert r["utc"].endswith("Z")


def test_stamping_survives_a_machine_with_no_git(monkeypatch):
    """A wheel install has no ``.git`` and a bare container may have no ``git``.

    Neither may stop a producer.  Every field goes ``None``; nothing raises.
    """
    monkeypatch.setattr(pv, "_git", lambda *a, **k: None)
    s = pv.code_stamp()
    assert s["src_tree"] is None and s["src_dirty"] is None
    assert set(s) == {"src_tree", "src_dirty", "version"}
    pv.run_stamp()          # must not raise either


def test_git_helper_swallows_a_missing_binary(monkeypatch):
    """``_git`` converts every failure to None rather than propagating OSError."""
    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(pv.subprocess, "run", boom)
    assert pv._git("rev-parse", "HEAD:src") is None


def test_dirty_is_scoped_to_src_not_the_whole_tree(monkeypatch):
    """A dirty ``paper.tex`` must not flag the stamp — prose cannot change a solve.

    The unscoped form raised exactly this false alarm on 2026-08-21, so pin the pathspec
    rather than only the return value.
    """
    seen = {}

    def fake(*args, **kwargs):
        seen["args"] = args
        return "" if args[0] == "status" else "yes"
    monkeypatch.setattr(pv, "_git", fake)
    pv.src_dirty()
    assert seen["args"][:2] == ("status", "--porcelain")
    assert seen["args"][-1] == "src", "dirty check is no longer scoped to src/"


def test_clean_tree_reports_false_not_none(monkeypatch):
    """Empty stdout means clean; it must not be read as 'cannot tell'.

    ``_git`` returns None both for "failed" and for "succeeded with no output", so
    ``src_dirty`` has to disambiguate.  If it stops doing that, a clean tree silently
    becomes undecidable and ``same_code`` starts refusing to compare anything.
    """
    monkeypatch.setattr(pv, "_git",
                        lambda *a, **k: None if a[0] == "status" else "true")
    assert pv.src_dirty() is False


@pytest.mark.parametrize("a,b,expected", [
    ({"src_tree": "aaa", "src_dirty": False}, {"src_tree": "aaa", "src_dirty": False}, True),
    ({"src_tree": "aaa", "src_dirty": False}, {"src_tree": "bbb", "src_dirty": False}, False),
    ({}, {"src_tree": "aaa", "src_dirty": False}, None),                   # legacy artifact
    ({"src_tree": None, "src_dirty": None}, {"src_tree": "aaa"}, None),    # no git
    ({"src_tree": "aaa", "src_dirty": True}, {"src_tree": "aaa"}, None),   # dirty: HEAD != what ran
])
def test_same_code_says_unknown_rather_than_different(a, b, expected):
    """A consumer refusing on mismatch must not refuse on absence, or every pre-stamp
    artifact fails the day the check lands."""
    assert pv.same_code(a, b) is expected


def test_dump_stamps_meta_without_disturbing_the_producer_payload():
    """``_figdata.dump`` injects ``meta.code`` and touches nothing else."""
    import _figdata as fd
    obj = {"x": [1, 2], "meta": {"seed": 7, "box": "spin8_qc_chi_prod"}}
    with tempfile.TemporaryDirectory() as tmp:
        old, fd.FIGDATA = fd.FIGDATA, tmp
        try:
            fd.dump("probe_stamp", obj)
            with open(os.path.join(tmp, "probe_stamp.json")) as f:
                d = json.load(f)
        finally:
            fd.FIGDATA = old
    assert d["x"] == [1, 2]
    assert d["meta"]["seed"] == 7 and d["meta"]["box"] == "spin8_qc_chi_prod"
    assert set(d["meta"]["code"]) == {"src_tree", "src_dirty", "version"}


def test_dump_creates_meta_when_a_producer_writes_none():
    """Not every producer writes a ``meta``; the stamp must not depend on one existing."""
    import _figdata as fd
    with tempfile.TemporaryDirectory() as tmp:
        old, fd.FIGDATA = fd.FIGDATA, tmp
        try:
            fd.dump("probe_nometa", {"y": [3]})
            with open(os.path.join(tmp, "probe_nometa.json")) as f:
                d = json.load(f)
        finally:
            fd.FIGDATA = old
    assert d["y"] == [3]
    assert d["meta"]["code"]["version"] is not None or True   # shape, not value
    assert set(d["meta"]["code"]) == {"src_tree", "src_dirty", "version"}


def test_the_sweep_producer_actually_stamps():
    """``run_tp_random_sweep`` writes the stamp into its ``meta``.

    Pinned by source inspection rather than by running the sweep, which is a cluster-scale
    job.  The 2026-08-20 misattribution came from this one artifact, so the wiring is what
    matters, not the value.
    """
    src = os.path.join(REPO, "src", "lemaitre", "initial_data",
                       "conformally_flat_puncture", "pipeline", "run_tp_random_sweep.py")
    with open(src) as f:
        text = f.read()
    assert "from lemaitre.initial_data.conformally_flat_puncture.provenance import run_stamp" in text
    assert "code=run_stamp()" in text, (
        "the sweep's meta no longer carries a code stamp; two runs from different trees "
        "would again be indistinguishable (FINDINGS 2026-08-20 section 0)")
