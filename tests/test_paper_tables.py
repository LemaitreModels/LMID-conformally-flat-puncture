"""Paper tables — the rendered LaTeX agrees with the module it is generated from.

The numeric rows are generated (``paper/tables/tabNN_*_tex.py`` -> ``tabNN_*.tex``,
``\\input`` by paper.tex), so they cannot drift from their source.  What can still
drift is the rendered ``.tex`` going stale relative to a retargeted producer, which
is what this guard covers.  It reads only files — no solves — so it belongs in the
fast tier.  Recompute with ``make tabdata`` and re-render with ``make tables``.

Also guarded here: ``pipeline/run_tangent_verification.py``, the standalone
parameter-sensitivity cross-check.  It no longer feeds a paper table (the numbers it
produced measured the interpolation error of the derivative on small value-only
verification interpolants, not on either shipped model), but it stays runnable
alongside ``tests/test_sensitivity*.py``.
"""
from __future__ import annotations

import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TABLES = os.path.join(ROOT, "paper", "tables")
PAPER = os.path.join(ROOT, "paper", "paper.tex")


def _tex(stem):
    p = os.path.join(TABLES, f"{stem}.tex")
    if not os.path.exists(p):
        pytest.skip(f"{stem}.tex not rendered yet (make tables)")
    with open(p) as f:
        return f.read()


def _paper():
    with open(PAPER) as f:
        return f.read()


def test_no_orphaned_table_inputs():
    """Every ``\\input{tables/...}`` in paper.tex resolves to a rendered body."""
    for stem in re.findall(r"\\input\{tables/(.+?)\.tex\}", _paper()):
        assert os.path.exists(os.path.join(TABLES, f"{stem}.tex")), (
            f"paper.tex inputs tables/{stem}.tex, which does not exist")


def test_production_box_table_matches_canonical_box():
    """Table I's rendered rows are the canonical box, level and grid the models share.

    The guard is that the rendered .tex still agrees with ``production_box`` — i.e.
    nobody retargeted an edge without re-rendering. The table carries only what the two
    quasi-circular models have in common (one value column, no per-model column), so the
    per-model node counts are guarded where they are stated instead: the body text.
    """
    from lm.initial_data.parametric.parametric_nd_2c import smolyak_points
    from lm.initial_data.pipeline import production_box as pb

    tex = _tex("tab01_production_box")
    paper = _paper()
    assert r"\input{tables/tab01_production_box.tex}" in paper
    assert rf"$[{2 * pb.B_MIN:g},\,{2 * pb.B_MAX:g}]$" in tex
    assert rf"$[{pb.Q_MIN:g},\,{pb.Q_MAX:g}]$" in tex
    assert rf"$[{-pb.CHI_MAX:g},\,{pb.CHI_MAX:g}]$" in tex
    assert rf"$\ell$ & ${pb.SMOLYAK_LEVEL}$" in tex
    assert rf"$({pb.PROD_GRID[0]},{pb.PROD_GRID[1]},{pb.PROD_GRID[2]})$" in tex
    # one value column only: no row may carry a second value
    assert r"\begin{tabular}{lc}" in tex
    for d in (4, 8):
        n = smolyak_points(d, pb.SMOLYAK_LEVEL)
        # the draft groups thousands only past four digits ($1105$ but $15{,}713$)
        forms = (f"${n}$", f"${n:,}$".replace(",", "{,}"))
        assert any(f in paper for f in forms), f"d={d} node count {n} not stated"


def test_producer_is_importable_and_standalone():
    """The canonical producer imports cleanly and does not reach outside the package."""
    from lm.initial_data.pipeline import run_tangent_verification as rtv
    assert callable(rtv.operator_tangents) and callable(rtv.surrogate_tangents)
    with open(rtv.__file__) as f:
        src = f.read()
    for forbidden in ("bbhfm", "sandbox", "import torch", "import nrpy"):
        assert forbidden not in src, forbidden
