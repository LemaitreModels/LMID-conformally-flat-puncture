"""Paper figures — every committed figdata json is current for its plotter.

The figure tier is two-step (``figNN_*_data.py`` -> ``figdata/figNN_*.json`` -> ``figNN_*_plot.py``),
and ``figdata/*.json`` is a gitignored build output rebuilt from heavy ``reports/`` sources, most of
them cluster-side.  So the failure mode is not a missing json but a **stale** one: a figdata built
before a block was added to its producer loads fine and then dies deep inside the plotter with a
bare ``KeyError``.  That is exactly how Fig. 2 broke — its PDF carries the mass-ratio panel while an
older local figdata had no ``Q_wall_q``.

This guard asserts that whatever figdata is present carries the top-level keys its plotter reads
(``registry.FIGURES[stem]["keys"]``).  It reads files only — no solves, no jax — so it belongs in
the fast tier.  Rebuild a stale one with ``python paper/figures/make_figdata.py --fig NN --force``.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIGURES = os.path.join(ROOT, "paper", "figures")
sys.path.insert(0, FIGURES)

import _figdata as fd  # noqa: E402
import registry as reg  # noqa: E402


@pytest.mark.parametrize("stem", reg.figure_stems())
def test_committed_figdata_is_current(stem):
    """If figdata/<stem>.json exists, it has every key the plotter reads."""
    p = fd.figdata_path(stem)
    if not os.path.exists(p):
        pytest.skip(f"{stem} figdata absent (make figdata; most sources are cluster-side)")
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
