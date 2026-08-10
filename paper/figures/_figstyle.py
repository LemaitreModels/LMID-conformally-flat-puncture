"""LM-initial-data paper — shared figure geometry (single source of truth for figure size).

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

FONTS.  Font sizes are deliberately NOT globally forced.  Each plotter keeps matplotlib's natural
per-element hierarchy (title >= axis labels / ticks > legend > small in-panel data labels), which
reads better than one flat size.  An earlier experiment that authored figures at their true render
width and fixed every figure's text to the 9 pt caption size was REJECTED — it looked too large and
unnatural.  So: keep the per-element ``fontsize=`` values in the plotters, and let LaTeX's mild
downscale of the ``figdims`` size make the effective text a little smaller than the caption (the
usual, natural look).
"""
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
