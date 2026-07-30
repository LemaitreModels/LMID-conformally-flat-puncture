"""LM-initial-data paper — shared figure geometry (single source of truth for figure size).

UNIFORMITY.  Every figure sizes itself through ``figdims`` so panels share ONE aspect ratio
(``PANEL_W : PANEL_H``) across the whole paper, regardless of the panel grid.  A figure with an
(nrow, ncol) grid uses ``figsize=figdims(nrow, ncol)``; LaTeX then scales it to ``\\columnwidth``
(single column) or ``\\textwidth`` (``figure*``), which preserves that per-panel aspect ratio.

FONTS.  Font sizes are deliberately NOT globally forced.  Each plotter keeps matplotlib's natural
per-element hierarchy (title >= axis labels / ticks > legend > small in-panel data labels), which
reads better than one flat size.  An earlier experiment that authored figures at their true render
width and fixed every figure's text to the 9 pt caption size was REJECTED — it looked too large and
unnatural.  So: keep the per-element ``fontsize=`` values in the plotters, and let LaTeX's mild
downscale of the ``figdims`` size make the effective text a little smaller than the caption (the
usual, natural look).
"""
PANEL_W = 4.5    # inches per panel (width)
PANEL_H = 3.0    # inches per panel (height);  PANEL_W : PANEL_H = 3 : 2  (~golden)


def figdims(nrow=1, ncol=1):
    """Uniform (width, height) for an nrow x ncol panel grid: (ncol*PANEL_W, nrow*PANEL_H)."""
    return (ncol * PANEL_W, nrow * PANEL_H)
