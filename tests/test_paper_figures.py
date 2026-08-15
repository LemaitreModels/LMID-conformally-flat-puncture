"""Paper figures — every committed figdata json is present and current for its plotter.

The figure tier is two-step (``figNN_*_data.py`` -> ``figdata/figNN_*.json`` -> ``figNN_*_plot.py``).
All ten ``figdata/*.json`` are **committed** (see ``.gitignore``, which says so explicitly): they are
the distilled few-kB artifacts that let a clone rebuild every figure with no ``reports/``, no
models, no solves and no jax.  So there are two failure modes, and this file guards both.

*Stale*: a figdata built before a block was added to its producer loads fine and then dies deep
inside the plotter with a bare ``KeyError``.  That is exactly how Fig. 2 broke — its PDF carries the
mass-ratio panel while an older local figdata had no ``Q_wall_q``.

*Absent*: a committed artifact that has gone missing.  This used to ``skip``, on the since-corrected
premise that figdata was a gitignored build output — so deleting one passed the suite silently while
``make figures`` would then plot nothing.  It is a failure now.

A third guard pins ``registry``'s producer commands, because ``make_figdata.py --check`` prints them
as the rebuild command and three of them named a module that could not write the file (§9.2 of the
2026-08-15 review): the drift class had happened twice with no test on it.

Everything here reads files only — no solves, no jax — so it belongs in the fast tier.  Rebuild a
stale one with ``python paper/figures/make_figdata.py --fig NN --force``.
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIGURES = os.path.join(ROOT, "paper", "figures")
PIPELINE_DIR = os.path.join(
    ROOT, "src", "lemaitre", "initial_data", "conformally_flat_puncture", "pipeline")
sys.path.insert(0, FIGURES)

import _figdata as fd  # noqa: E402
import registry as reg  # noqa: E402


@pytest.mark.parametrize("stem", reg.figure_stems())
def test_committed_figdata_is_current(stem):
    """figdata/<stem>.json is present, and has every key the plotter reads."""
    p = fd.figdata_path(stem)
    assert os.path.exists(p), (
        f"{os.path.relpath(p, ROOT)} is MISSING.  Every figdata is committed — a clone must "
        f"rebuild all ten figures from the repo alone — so an absent one is a deleted artifact, "
        f"not an un-run build.  Restore it (git checkout) or rebuild: "
        f"python paper/figures/make_figdata.py --fig {stem} --force")
    with open(p) as f:
        d = json.load(f)
    miss = fd.missing_keys(stem, d)
    assert not miss, (
        f"{os.path.relpath(p, ROOT)} is stale: missing {miss} (has {sorted(d)}). "
        f"Rebuild: python paper/figures/make_figdata.py --fig {stem} --force")


def test_every_figure_declares_its_plotter_keys():
    """The registry must declare keys for every figure, else the guard above is vacuous."""
    undeclared = [s for s in reg.figure_stems() if not reg.keys_for(s)]
    assert not undeclared, f"registry.FIGURES lacks 'keys' for {undeclared}"


def test_declared_keys_are_grounded_in_the_plotter():
    """At least one declared key is named in the plotter, so a wholesale typo cannot pass.

    Deliberately weak in one direction: a figure may declare the json's full contract (keys its
    data script writes for provenance, e.g. ``meta``/``summary``) even when the plotter reads only
    some of them.  The reverse direction is not asserted because top-level and nested accesses are
    both spelled ``d[...]`` and cannot be told apart by reading the source.
    """
    for stem in reg.figure_stems():
        src = os.path.join(FIGURES, f"{stem}_plot.py")
        keys = reg.keys_for(stem)
        if not os.path.exists(src) or not keys:
            continue
        with open(src) as f:
            text = f.read()
        assert any(f'"{k}"' in text for k in keys), (
            f"{stem}_plot.py names none of its declared keys {keys}")


# --------------------------------------------------------------------------
# registry provenance: the producer commands must be runnable and current
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(reg.SOURCES))
def test_producer_module_exists_and_ranks_match_the_shipped_model(key):
    """Each source names a real ``pipeline/`` module, and every rank it passes is the shipped one.

    Two independent drifts this pins, both of which had already happened:

    * a producer string naming a module that cannot write the artifact
      (``run_guess_vs_memory`` writes one flat ``guess_vs_memory.json``, never the
      ``*_gapfill_1000.json`` three entries claimed) — ``--check`` prints these as the
      rebuild command, so a wrong one sends the reader to the wrong script;
    * a rank restated beside ``production_model.SHIPPED_RANK`` and left behind when the
      shipped rank moved (docs/DATA.md still carries an r75 row from that era).

    ``producer_cmd`` resolves ``{rank}``/``{pod_stem}``/``{model_stem}`` from
    ``SHIPPED_RANK``, so this asserts the wiring rather than a copy: an entry that
    hard-codes a rank instead of using the placeholder fails the second half.
    """
    p = reg.SOURCES[key]["producer"]
    cmd = reg.producer_cmd(key)                      # raises on an unresolved placeholder

    if p["module"] is None:
        return
    src = os.path.join(PIPELINE_DIR, p["module"] + ".py")
    assert os.path.exists(src), (
        f"source {key!r} names producer module {p['module']!r}, which does not exist under "
        f"pipeline/.  Declared command: {cmd}")

    for r in re.findall(r"--(?:cross-)?rank\s+(\d+)", cmd):
        assert p["dim"] is not None, f"{key}: passes --rank but declares no dim"
        assert int(r) == pm.SHIPPED_RANK[p["dim"]], (
            f"{key}: producer passes --rank {r} but production_model.SHIPPED_RANK"
            f"[{p['dim']}] is {pm.SHIPPED_RANK[p['dim']]}.  The shipped rank is defined "
            f"there and nowhere else (CLAUDE.md); use the '{{rank}}' placeholder.")


def test_producer_ranks_are_not_hard_coded():
    """No entry restates a shipped rank as a literal where the placeholder belongs.

    Deliberately narrow, so an unrelated numeric argument (``--n-points 500``, say)
    cannot trip it: only the two positions a rank is ever written in are checked —
    the value after a ``--rank`` flag, and the ``_r<N>`` suffix of a POD artifact name.
    """
    ranks = {str(r) for r in pm.SHIPPED_RANK.values()}
    offenders = []
    for key, spec in reg.SOURCES.items():
        argv = spec["producer"]["argv"]
        for i, a in enumerate(argv):
            after_rank_flag = i and argv[i - 1] in ("--rank", "--cross-rank") and a in ranks
            pod_name_suffix = any(re.search(rf"_r{r}(\D|$)", a) for r in ranks)
            if after_rank_flag or pod_name_suffix:
                offenders.append((key, a))
    assert not offenders, (
        f"producer argv restates a shipped rank literally: {offenders}.  Use '{{rank}}' / "
        f"'{{pod_stem}}' so production_model stays the single source (CLAUDE.md).")
