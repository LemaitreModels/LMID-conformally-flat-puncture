"""Paper tables — the rendered LaTeX and the hand-written captions agree with the data.

The numeric rows of Tables II and III are generated (``paper/tables/tabNN_*_tex.py`` ->
``tabNN_*.tex``, ``\\input`` by paper.tex), so they cannot drift from
``tabdata/tabNN_*.json``.  Two things still can:

  * the rendered ``.tex`` can go stale relative to a recomputed json;
  * the CAPTION in paper.tex is author-owned prose and repeats the configuration
    (grid, slice, interpolation orders, certified residual) by hand.

This guard covers both.  It reads only files — no solves — so it belongs in the fast
tier.  Recompute the data with ``make tabdata`` and re-render with ``make tables``.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TABLES = os.path.join(ROOT, "paper", "tables")
PAPER = os.path.join(ROOT, "paper", "paper.tex")
sys.path.insert(0, TABLES)

import _tabdata as td  # noqa: E402

STEMS = ("tab02_tangent_operator", "tab03_tangent_surrogate")


def _tex(stem):
    p = os.path.join(TABLES, f"{stem}.tex")
    if not os.path.exists(p):
        pytest.skip(f"{stem}.tex not rendered yet (make tables)")
    with open(p) as f:
        return f.read()


def _data(stem):
    try:
        return td.load(stem)
    except FileNotFoundError:
        pytest.skip(f"{stem} tabdata absent (make tabdata)")


def _paper():
    with open(PAPER) as f:
        return f.read()


@pytest.mark.parametrize("stem", STEMS)
def test_rendered_rows_match_data(stem):
    """Every value in the rendered .tex is the json value at the table's precision."""
    d, tex = _data(stem), _tex(stem)
    keys = ["rel_fd"] + (["rel_ift"] if stem.endswith("surrogate") else [])
    for axis, row in d["rows"].items():
        assert td.axis(axis) in tex, f"{axis} missing from {stem}.tex"
        for k in keys:
            assert td.sci(row[k]) in tex, (
                f"{stem}.tex is stale: {axis}/{k}={row[k]:.4e} renders as "
                f"{td.sci(row[k])}, not found. Re-run: make tables")
        assert td.pow10(row["h"]) in tex


@pytest.mark.parametrize("stem", STEMS)
def test_paper_inputs_generated_table(stem):
    """paper.tex \\input s the generated body rather than hand-written rows."""
    assert rf"\input{{tables/{stem}.tex}}" in _paper()


def test_operator_caption_matches_config():
    """Table II's caption repeats the slice, grid, and certified residual by hand."""
    d = _data("tab02_tangent_operator")
    c, cap = d["config"], _paper()
    for frag in (rf"$b={c['b']}$", rf"$q={c['q']}$", rf"$P_t={c['P_t']}$",
                 rf"$\chi^{{A}}_{{y}}={c['chi_Ay']:.2f}$",
                 rf"$\chi^{{B}}_{{y}}={c['chi_By']:.2f}$",
                 rf"$(N_A,N_B,N_\phi)=({c['Na']},{c['Nb']},{c['Nphi']})$"):
        assert frag in cap, f"Table II caption disagrees with tabdata: expected {frag}"
    # certified residual, quoted to two significant figures
    assert td.sci(c["residual_norm"]).strip("$") in cap.replace(r"\|R\|_\infty=", "")


def test_surrogate_caption_matches_config():
    """Table III's caption repeats the grid, boxes, orders, and evaluation points."""
    d = _data("tab03_tangent_surrogate")
    c, cap = d["config"], _paper()
    qb = [s["Q"] for s in c["spec_bq"]]
    bmin, bmax = c["spec_bq"][0]["min"], c["spec_bq"][0]["max"]
    qmin, qmax = c["spec_bq"][1]["min"], c["spec_bq"][1]["max"]
    for frag in (rf"$(N_A,N_B)=({c['Na']},{c['Nb']})$",
                 rf"$Q=({qb[0]},{qb[1]})$",
                 rf"$b\in[{bmin:g},{bmax:g}]$", rf"$q\in[{qmin:g},{qmax:g}]$",
                 rf"$Q={c['spec_chi'][0]['Q']}$",
                 rf"$q={c['fixed_chi']['q']}$", rf"$b={c['fixed_chi']['b']:g}$",
                 rf"$(b,q)=({c['theta_bq'][0]:g},{c['theta_bq'][1]:g})$",
                 rf"$(\chi^{{A}}_{{y}},\chi^{{B}}_{{y}})=({c['theta_chi'][0]:g},{c['theta_chi'][1]:g})$"):
        assert frag in cap, f"Table III caption disagrees with tabdata: expected {frag}"
    # the caption bounds the reference residual; the bound must actually hold
    m = re.search(r"reference solves are certified to\s*\n?\s*"
                  r"\$\\\|R\\\|_\\infty\\le(\d)\\times10\^\{(-\d+)\}\$", cap)
    assert m, "Table III caption no longer states a certified-residual bound"
    bound = float(m.group(1)) * 10.0 ** int(m.group(2))
    assert c["residual_norm"] <= bound, (
        f"reference residual {c['residual_norm']:.2e} exceeds the caption's bound {bound:.0e}")


def test_production_box_table_matches_canonical_box():
    """Table I's rendered rows are the canonical box, level and grid the models share.

    Unlike Tables II--III this one recomputes from ``production_box`` rather than from a
    solve, so the guard is that the rendered .tex still agrees with that module — i.e.
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
