r"""LM-initial-data paper — shared figure geometry (single source of truth for figure size).

UNIFORMITY.  Every figure sizes itself through ``figdims`` so panels share ONE aspect ratio
(``PANEL_W : PANEL_H``) across the whole paper, regardless of the panel grid.  A figure with an
(nrow, ncol) grid uses ``figsize=figdims(nrow, ncol)``; LaTeX then scales it to ``\\columnwidth``
(single column) or ``\\textwidth`` (``figure*``), which preserves that per-panel aspect ratio.

STACKED FIGURES.  The 3:2 default is right for a figure ONE panel tall, but a figure whose grid is
two panels tall renders (at a fixed LaTeX width) twice as high, and the float then eats most of a
page: Figs. 4, 5 (2x2 at ``\\textwidth``) and 8 (3x1 at ``\\columnwidth``) came out ~4.6 in tall.
Those three therefore pass ``panel_h=PANEL_H_STACK``, a flatter panel.  They all use the SAME
flatter value, so they stay consistent with each other, and the log-scaled panels they carry lose
nothing by it.  Fig. 1 (2x4) is left on the default: its panels are already short because it is four
columns wide.  Fig. 2 (3x1) predates this and keeps its own local ``PANEL_H_SHORT``, a height fixed
by a measured RevTeX float-fitting limit rather than by appearance — see its plotter.

FONTS.  The font *family* IS forced globally, here, as an import-time side effect: every plotter
imports this module for ``figdims``, so this is the one place that keeps all ten figures on a single
typeface — and that puts a new figure on it automatically.  matplotlib's default is DejaVu Sans,
which clashes with the serif RevTeX sets for the body text.

That body face is **Computer Modern**, not Times.  ``revtex4-2`` with ``aps,prd`` loads no font
package, and ``paper.tex`` adds none, so the compiled paper is CMR10/CMMI10/CMSY10 throughout —
check it rather than assuming, with ``pdffonts paper.pdf``.  (This module set STIXGeneral until
2026-08-19 on the stated premise that RevTeX gives a Times-like serif; that premise was wrong, and
STIX's Times metrics are exactly what made the figures read as a different face from the page.)

``cmr10`` with the ``cm`` mathtext set is therefore the matching pair, and matplotlib bundles the
whole CM family (``cmr10``, ``cmmi10``, ``cmsy10``, ``cmex10``, ``cmti10``, ``cmb10``, ``cmtt10``) —
so this needs neither a system font nor a LaTeX install (``text.usetex`` stays off) and the figures
build identically everywhere.  Latin Modern would cover more glyphs, but it ships only inside a TeX
distribution, which would trade that portability away.

The two companion rcParams are not cosmetic, they are what make ``cmr10`` usable.  ``cmr10`` is a
Type-1 conversion carrying the TeX text encoding: it has **no** U+2212 MINUS, no en/em dash and no
Greek, so a negative tick label renders as a missing-glyph box.  ``axes.formatter.use_mathtext``
routes numeric tick labels through mathtext, where the minus comes from ``cmsy10`` and typesets as
the page's own minus (matplotlib emits a UserWarning naming this rcParam if you set ``cmr10``
without it); ``axes.unicode_minus = False`` catches any remaining non-mathtext text by falling back
to the ASCII hyphen, which ``cmr10`` does have.  Fig. 7's negative binding-energy axis is the case
that exercises both.  Literal non-ASCII in a plotter's *rendered* strings will still be a box —
write it as mathtext (``$\times$``, ``$\chi$``), as the plotters already do.

Two glyphs still fall back to STIX, and they are the only ones: matplotlib bundles no bold math
italic and no ``cmex``-side star, so ``\boldsymbol{\chi}`` (``MODEL_TITLES[8]``, Figs. 3--5) draws
from STIXGeneral-BoldItalic and ``\star`` (Fig. 6's ``\|F-F_\star\|_\infty``) from
STIXGeneral-Regular — while ``paper.tex`` sets the same two from CMMIB10 and CMSY10.  Plain
``\chi`` is unaffected: mathtext resolves it through the TeX name into ``cmmi10``, not by codepoint.
Neither is worth fixing by changing the notation, which must track the paper.  ``pdffonts`` on each
figure is the check; a *third* STIX name appearing there means a new label needs a CM-covered symbol.

Font *sizes*, by contrast, are deliberately NOT globally forced.  Each plotter keeps matplotlib's
natural per-element hierarchy (title >= axis labels / ticks > legend > small in-panel data labels),
which reads better than one flat size.  An earlier experiment that authored figures at their true
render width and fixed every figure's text to the 9 pt caption size was REJECTED — it looked too
large and unnatural.  So: keep the per-element ``fontsize=`` values in the plotters, and let LaTeX's
mild downscale of the ``figdims`` size make the effective text a little smaller than the caption
(the usual, natural look).
"""
import matplotlib as _mpl

_mpl.rcParams["font.family"] = "serif"
_mpl.rcParams["font.serif"] = ["cmr10"]
_mpl.rcParams["mathtext.fontset"] = "cm"
_mpl.rcParams["axes.formatter.use_mathtext"] = True   # cmr10 has no U+2212; see FONTS above
_mpl.rcParams["axes.unicode_minus"] = False

PANEL_W = 4.5          # inches per panel (width)
PANEL_H = 3.0          # inches per panel (height);  PANEL_W : PANEL_H = 3 : 2  (~golden)
PANEL_H_STACK = 2.1    # flatter panel for two-row figures (see STACKED FIGURES above)


def figdims(nrow=1, ncol=1, panel_h=PANEL_H):
    """Uniform (width, height) for an nrow x ncol panel grid: (ncol*PANEL_W, nrow*panel_h)."""
    return (ncol * PANEL_W, nrow * panel_h)


# Model titles are shared by Figs. 3, 4 and 5, which draw the same two models side by side; keeping
# them here rather than per-plotter is what stops the three from drifting apart (they had).
MODEL_TITLES = {
    4: r"4D quasi-circular model: $\theta=(b,q,\chi^{A}_{y},\chi^{B}_{y})$",
    8: r"8D quasi-circular model: $\theta=(b,q,\boldsymbol{\chi}^{A},\boldsymbol{\chi}^{B})$",
}
