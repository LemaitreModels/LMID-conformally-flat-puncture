"""Pin the numerals the prose quotes to the figdata they were measured from.

**Why this file exists.** ``test_paper_figures.py`` checks that a figdata exists
and carries its plotter's keys, and ``test_paper_tables.py`` checks the rendered
table rows -- so a figure can be re-distilled, move every number it carries, and
leave the sentence quoting it untouched with the whole suite green.  That has now
happened three times on this leaf:

* 2026-08-19, the publication audit: fig03's caption claimed a ``35x`` deeper
  floor against a figdata measuring ``49.3x``, and all three fig07 eccentricity
  numerals were unreproducible from the committed curves.
* 2026-08-20: four appendix numerals were falsified by the re-run TwoPunctures
  sweep while all 284 gates passed, *because fig08 had no numeral gate*.
* 2026-08-21: ``docs/DATA.md`` still restated fig08's anchor with the three
  values ``paper.tex`` had been corrected away from the day before.
* 2026-08-21, the ``docs/`` audit: ``MODELS.md`` gave a POD artifact name with a
  ``_slim`` suffix ``production_model.pod_stem`` deliberately does not emit, and
  promised that a stale figdata "reads as stale" -- which the staleness check,
  being key-presence only, cannot do.  ``STRUCTURE.md``'s layout block predated
  ``oracle/``, and its "the threshold is not restated anywhere else" was false of
  four producers.

The first of those had a gate, and it is gone: the caption claiming a deeper
best-case floor was deleted on 2026-08-23, so ``49.3x`` is now a measurement fig03
carries and the paper does not quote.  A gate over a claim nobody makes fails on
the *prose* being shortened, which is not the defect class this file is for -- so
it was retired with the claim rather than kept green by re-adding the sentence.

Every one of those was a *prose* defect over a *correct* artifact, which is the
one thing none of the other gates look at.  So these tests read the artifact,
render the number the way the prose does, and assert the prose says it.

**The pattern to copy when you add a numeral.** Derive the expected string from
the figdata -- never hard-code it twice -- and assert both that the current form
is present *and* that the specific stale form is absent.  The second half is what
catches a revert or a bad merge, and it costs one line.

These tests read only committed JSON and committed prose: no ``$LM_REPORTS``, no
solver, no oracle.  They are milliseconds, so there is no excuse for a numeral
not to be here.
"""
from __future__ import annotations

import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper", "paper.tex")
FIGDATA = os.path.join(ROOT, "paper", "figures", "figdata")
DATA_MD = os.path.join(ROOT, "docs", "DATA.md")
MODELS_MD = os.path.join(ROOT, "docs", "MODELS.md")
STRUCTURE_MD = os.path.join(ROOT, "docs", "STRUCTURE.md")
PYPROJECT = os.path.join(ROOT, "pyproject.toml")


def _figdata(stem):
    with open(os.path.join(FIGDATA, stem)) as f:
        return json.load(f)


def _text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _sci_tex(x, digits=1):
    """Render ``x`` as the paper renders it: ``m.m\\times10^{e}``."""
    exp = 0
    m = abs(float(x))
    while m >= 10.0:
        m /= 10.0
        exp += 1
    while m < 1.0:
        m *= 10.0
        exp -= 1
    return f"{m:.{digits}f}\\times10^{{{exp}}}"


# --------------------------------------------------------------------------
# fig03 -- the joint held-out convergence ratio
# --------------------------------------------------------------------------

def test_fig03_ratio_is_non_monotonic_so_the_qualifier_is_load_bearing():
    """The shape of fig03's ratio curve, which the paper no longer describes.

    This began as the reason the caption had to say "reaches" rather than quote a
    bare number: the by-level ratios rise 1.5, 3.9, 13.3, **12.6**, 49.3 -- they
    dip at level 4, so a figure quoted without its level would have been
    ambiguous rather than merely terse.  The caption that made the claim was
    removed on 2026-08-23 along with its gate, so nothing in the prose depends on
    this any more and it pins the ARTIFACT alone: the dip is real, and the
    deepest level is still the best ratio.  Kept because both facts are what a
    reader of fig03 would reconstruct, and a re-drawn panel that lost either
    would be a different figure.
    """
    d = _figdata("fig03_joint_dist.json")
    r = [b / c for b, c in zip(d["left"]["bare"]["best"], d["left"]["cross"]["best"])]
    assert any(r[i + 1] < r[i] for i in range(len(r) - 1)), (
        f"the by-level best-case ratios are now monotone ({r}); the level-4 dip "
        "is gone, so fig03 is measuring something other than what it did -- "
        "check the sweep before redrawing the panel")
    assert r[-1] == max(r), (
        f"the deepest level is no longer the best ratio ({r}); fig03's left "
        "panel no longer bottoms out at $\\ell=5$, which is a change in the "
        "measurement and not in the wording")


# --------------------------------------------------------------------------
# fig07 -- the three eccentricity numerals
# --------------------------------------------------------------------------

def _fig07_well_resolved(d):
    """The J whose well the scan resolves -- i.e. all but the shallow smallest.

    The shallow J is the one with a single measurable eccentricity rung: its
    second turning point leaves the box, which is exactly the regime the paper
    says the scan struggles in.
    """
    return {J: pj for J, pj in d["per_J"].items() if pj["n_ecc_measurable"] > 1}


def test_fig07_agreement_bound_is_true_and_tight():
    """"agreeing with it to 6e-3 where the well is well resolved".

    Asserted two ways on purpose: the bound must HOLD for every well-resolved J,
    and the next-tighter round bound must FAIL -- otherwise a correct-but-loose
    number (say 1e-2) would pass and the sentence would understate the method.
    The old 5e-3 failed the first half by a hair at both J, which is exactly the
    kind of error no reader recomputes.
    """
    d = _figdata("fig07_eccentricity.json")
    rel = {J: pj["d_bcirc_rel"] for J, pj in _fig07_well_resolved(d).items()}
    assert rel, "no J has a resolved well; fig07 needs re-distilling"
    worst = max(rel.values())
    paper = _text(PAPER)
    assert "6\\times10^{-3}" in paper, (
        f"the well-resolved agreement is {worst:.2e} (per J: "
        f"{ {J: f'{v:.2e}' for J, v in rel.items()} }), which the paper should "
        "bound by 6e-3")
    assert worst < 6e-3, (
        f"the agreement has degraded to {worst:.2e}, above the 6e-3 the paper "
        "claims -- the sentence is now false")
    assert worst > 5e-3, (
        f"the agreement improved to {worst:.2e}, so the paper's 6e-3 is now "
        "loose and understates the method; tighten it deliberately")
    assert "$5\\times10^{-3}$ where the well is well resolved" not in paper, (
        "the 5e-3 claim is back; the committed figdata does not support it at "
        "either well-resolved J")


def test_fig07_scan_mislocation_matches_the_shallow_well():
    """"the scan mislocates the minimum by 0.05 M in b".

    In the SHALLOW well -- the sentence's own regime, so the gate must pick that
    J rather than the worst or the mean.  The old wording, "half a mass", was
    ~10x too large under the paper's own M = m_A + m_B = 1, and ambiguous
    besides: the error factor depended on which mass the reader assumed, so it
    could not be checked at all.  Hence the explicit units here.
    """
    d = _figdata("fig07_eccentricity.json")
    shallow = [(J, pj) for J, pj in d["per_J"].items() if pj["n_ecc_measurable"] <= 1]
    assert len(shallow) == 1, (
        f"expected exactly one shallow-well J, found {[J for J, _ in shallow]}; "
        "the sentence's regime is no longer unambiguous")
    J, pj = shallow[0]
    d_abs = pj["d_bcirc_abs"]
    want = f"${d_abs:.2f}\\,M$ in $b$"
    paper = _text(PAPER)
    assert want in paper, (
        f"the shallow well (J={J}) mislocation is {d_abs:.4f} in b, so the paper "
        f"should say {want!r}")
    assert "half a mass" not in paper, (
        f"'half a mass' is back; the measured mislocation is {d_abs:.4f} in b, "
        "roughly a tenth of that under M = m_A + m_B = 1")


def test_fig07_box_edge_eccentricity_matches_the_ladder_that_reaches_it():
    """"from e=0 at the circular orbit to e~0.42 at the box edge".

    "At the box edge" is only meaningful for the J whose ladder actually reaches
    it; the others go NaN first.  The old 0.23 was a MID-ladder rung of that same
    J presented as an endpoint, which no amount of internal figdata consistency
    would have caught.
    """
    d = _figdata("fig07_eccentricity.json")
    b_hi = d["meta"]["box_b"][1]
    reach = {J: pj for J, pj in d["per_J"].items()
             if pj["e_max"] is not None and pj["b0_max_measurable"] is not None}
    assert reach, "no J measures an eccentricity at all"
    J = max(reach, key=lambda J: reach[J]["b0_max_measurable"])
    pj = reach[J]
    assert pj["b0_max_measurable"] > 0.95 * b_hi, (
        f"the furthest measurable rung is b0={pj['b0_max_measurable']:.3f} against "
        f"a box edge of {b_hi}; no ladder reaches the edge any more, so the "
        "phrase 'at the box edge' is no longer supported at all")
    e_edge = pj["ecc"][pj["n_ecc_measurable"] - 1]
    want = f"e\\approx{e_edge:.2f}"
    paper = _text(PAPER)
    assert want in paper, (
        f"at the box edge (J={J}, b0={pj['b0_max_measurable']:.3f}) the "
        f"eccentricity is {e_edge:.4f}, so the paper should say {want!r}")
    assert "e\\approx0.23" not in paper, (
        f"e~0.23 is back; that is a mid-ladder rung, not the box edge "
        f"(edge is {e_edge:.4f} at J={J})")


# --------------------------------------------------------------------------
# fig08 -- the axisymmetric code-to-code anchor, quoted in TWO places
# --------------------------------------------------------------------------

#: ``meta.anchor`` key -> (what the prose calls it, form used in docs/DATA.md)
ANCHOR = {
    "max_dpsi": ("psi agreement", "{m}e{e}"),
    "M_ADM_rel_diff": ("relative ADM-mass difference", "{m}e{e}"),
    "residual": ("certified residual", "{m}e{e}"),
}


def _anchor():
    a = _figdata("fig08_tp_validation.json")["meta"]["anchor"]
    missing = [k for k in ANCHOR if k not in a]
    assert not missing, f"fig08 meta.anchor missing {missing} -- re-distill fig08"
    return a


def test_fig08_anchor_numerals_are_in_the_paper():
    """The three numerals of the appendix's most stringent oracle comparison.

    All three were falsified by the 2026-08-20 sweep re-run and passed 284 green
    tests, because this gate did not exist.  It is the cheapest test in the file.
    """
    a = _anchor()
    paper = _text(PAPER)
    for key, (what, _) in ANCHOR.items():
        want = _sci_tex(a[key])
        assert f"${want}$" in paper, (
            f"fig08's {what} is {a[key]:.4e}, so the appendix should quote "
            f"${want}$ and does not")
    for stale in ("2.6\\times10^{-10}", "5.0\\times10^{-11}", "1.2\\times10^{-12}"):
        assert stale not in paper, (
            f"{stale} is back in paper.tex: that is a pre-2026-08-20 anchor "
            "value, superseded by the sweep re-run")


def test_fig08_anchor_numerals_are_also_right_in_DATA_md():
    """``docs/`` quotes them too, and drifted for a day without anyone noticing.

    The 2026-08-20 lane corrected paper.tex and did not know DATA.md restated the
    same three numbers, so the repo shipped a doc contradicting its own paper.
    A numeral gate that reads only paper.tex would not have caught it.
    """
    a = _anchor()
    doc = _text(DATA_MD)
    for key, (what, _) in ANCHOR.items():
        m, e = f"{a[key]:.1e}".split("e")
        want = f"{m}e-{int(e[1:])}" if e.startswith("-") else f"{m}e{int(e)}"
        assert want in doc, (
            f"docs/DATA.md should quote fig08's {what} as {want} "
            f"(measured {a[key]:.4e}) and does not")
    for stale in ("2.6e-10", "5.0e-11", "1.2e-12"):
        assert stale not in doc, (
            f"{stale} is back in docs/DATA.md: a pre-2026-08-20 anchor value")


def test_the_anchor_is_still_certified():
    """A guard the numerals alone would not give.

    The anchor is only quotable as a certified result; if the residual drifts
    above the gate the sentence needs rewriting, not renumbering.
    """
    a = _anchor()
    from lemaitre.initial_data.conformally_flat_puncture.parametric import certification
    assert a.get("certified") is True, "fig08's anchor is no longer marked certified"
    assert a["residual"] < certification.CERT_TOL, (
        f"the anchor residual {a['residual']:.2e} is no longer below CERT_TOL "
        f"{certification.CERT_TOL:.0e}")


# ==========================================================================
# docs/ -- the same failure class, one directory over
# ==========================================================================
# The 2026-08-21 audit found MODELS.md and STRUCTURE.md carrying numbers and
# mechanisms nothing checked.  Everything below is derived from the tree at test
# time, so the docs cannot drift from the code without going red.


def test_models_md_memory_table_is_the_one_production_model_prints():
    """MODELS.md says "regenerate it rather than hand-editing" -- so enforce that.

    ``production_model.table()`` is the printer the page's own header tells you to
    run.  Comparing the whole block at once pins the four family rows, both bare
    and both POD columns in one assertion, and makes a hand-edit fail rather than
    quietly disagree with the module that defines the model.
    """
    from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm
    doc = _text(MODELS_MD)
    table = pm.table().strip()
    assert table in doc, (
        "docs/MODELS.md's memory table is not the one production_model prints. "
        "Regenerate it:\n  python -m lemaitre.initial_data."
        "conformally_flat_puncture.pipeline.production_model\n\nexpected:\n"
        + table)


def test_models_md_headline_numbers_come_from_production_model():
    """The ranks, node counts, level and compression, each read from the module.

    Derived here rather than transcribed: if ``SHIPPED_RANK`` or ``N_NODES``
    moves, this fails on the doc instead of leaving the paper's narrative page
    describing the previous model.
    """
    from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm
    doc = _text(MODELS_MD)
    for dim in (4, 8):
        assert f"{pm.N_NODES[dim]}" in doc, f"{dim}-D node count missing from MODELS.md"
        assert f"**{pm.SHIPPED_RANK[dim]}**" in doc, f"{dim}-D shipped rank missing"
        assert f"**{pm.compression_factor(dim):.1f}\u00d7**" in doc, (
            f"{dim}-D compression is {pm.compression_factor(dim):.1f}x and "
            "MODELS.md does not say so")
    assert f"| Smolyak level | {pm.pb.SMOLYAK_LEVEL} | {pm.pb.SMOLYAK_LEVEL} |" in doc
    for dim in (4, 8):
        idx = ", ".join(str(i) for i in pm.enhanced_indices(dim))
        assert f"({idx})" in doc, (
            f"{dim}-D enhanced indices are ({idx}); MODELS.md must not carry a "
            "literal that disagrees with production_model.enhanced_indices")


def test_models_md_pod_artifact_name_is_exactly_what_pod_stem_emits():
    """The example filename, pinned to the function the doc credits for it.

    It carried a ``_slim`` suffix through a revision.  ``pod_stem``'s docstring
    says the layout is deliberately NOT in the name -- it lives in
    ``meta['dU_layout']`` so a name cannot go stale against its contents -- so the
    doc was advertising a guarantee the code refuses to give.
    """
    from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm
    doc = _text(MODELS_MD)
    assert f"{pm.pod_stem(4)}.npz" in doc, (
        f"MODELS.md should show {pm.pod_stem(4)}.npz as the artifact name")
    assert "_slim.npz" not in doc, (
        "the '_slim' suffix is back in MODELS.md; pod_stem does not emit it, and "
        "the layout is recorded inside the file as meta['dU_layout']")


def test_structure_md_pool_and_slot_counts_are_reproducible():
    """The pooled-evaluation redundancy figures, recomputed from the index set.

    ``slots`` counts only the NONZERO-coefficient subgrids -- ``combination_coeffs``
    drops the rest -- which is exactly the 13-slot difference between the naive
    4-D count (5,270) and the true one (5,257).  Getting that wrong is how a
    reader concludes the doc is off by a rounding error when it is not.
    """
    import numpy as np
    from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd_smolyak import (
        isotropic_index_set, combination_coeffs)
    from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm

    doc = _text(STRUCTURE_MD)
    L = pm.pb.SMOLYAK_LEVEL
    for dim in (4, 8):
        kept = combination_coeffs(isotropic_index_set(dim, L))
        slots = sum(int(np.prod([1 if i == 0 else 2 ** i + 1 for i in l])) for l in kept)
        pool = pm.N_NODES[dim]
        assert f"{pool:,}" in doc, f"{dim}-D pool node count {pool:,} missing"
        assert f"{slots:,}" in doc, (
            f"{dim}-D expands {pool:,} pool nodes into {slots:,} subgrid slots "
            f"({slots / pool:.1f}x) and STRUCTURE.md does not say {slots:,}")


def test_the_slow_test_count_is_stated_identically_in_both_places():
    """One number, written down twice, and never compared.

    ``pyproject.toml``'s marker description states it and ``docs/STRUCTURE.md``
    restates it, which is the shape every defect in this file has had.  The mark
    count itself is deliberately NOT re-derived here: parametrized marks expand at
    collection (21 ``@pytest.mark.slow`` decorators become 30 collected tests), so
    a static count would pin the wrong number and read as authoritative.  Check
    that with ``pytest --collect-only -q -m slow``; what this pins is that the two
    prose copies cannot drift apart.
    """
    m = re.search(r"slow: needs an external oracle[^\"]*?\((\d+) tests", _text(PYPROJECT))
    assert m, "pyproject.toml's `slow` marker description no longer states a count"
    n = int(m.group(1))
    assert f"{n} `slow`" in _text(STRUCTURE_MD) or f"{n} slow" in _text(STRUCTURE_MD), (
        f"pyproject.toml says the suite has {n} slow tests; docs/STRUCTURE.md "
        "states a different number, or none. Re-measure with "
        "`pytest --collect-only -q -m slow` and fix both.")


def test_cert_tol_is_stated_once_and_correctly_in_the_docs():
    """The paper's threshold, in the two docs that name it.

    Pinned as a rendered string so a doc cannot keep 1e-10 after the module moves.
    """
    from lemaitre.initial_data.conformally_flat_puncture.parametric import certification
    want = f"CERT_TOL = {certification.CERT_TOL:.0e}"
    assert want in _text(STRUCTURE_MD), (
        f"STRUCTURE.md should state {want} (read from parametric.certification)")
