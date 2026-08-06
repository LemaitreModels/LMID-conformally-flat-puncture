# Paper figures

This folder contains **only** the figures used in `../paper.tex`. Each figure `figNN_<name>`
now comes as a **data/plot split** so it rebuilds from the repo alone — no solves, no
`reports/`, no `jax`, no models at plot time:

| file | role |
|---|---|
| `figNN_<name>_plot.py`  | **plotter** — reads **only** `figdata/figNN_<name>.json` and draws the `.pdf`. |
| `figNN_<name>_data.py`  | **data script** — distills the arrays the figure plots out of the raw run artifacts under `../../reports/` into that committed json. This is the only step that ever touches `reports/`, heavy models, or `jax`. |
| `figdata/figNN_<name>.json` | the committed, plot-ready data (≈46 kB total for all figures). |

One exception to "distills": `fig10_constraints_data.py` **recomputes** — it runs the
two-centre solve and the finite-difference constraint monitor itself (both cheap) and queries
the TwoPunctures binary directly, so it has no `reports/` source and is registered
`inline=True`. Its oracle leg takes ~40 min (1.3M query points); `--no-tp` skips it.

Supporting the split:

| file | role |
|---|---|
| `registry.py`     | single source of truth: `SOURCES` (each raw run artifact → its producer command, laptop/cluster, status, and the figures that consume it) + `FIGURES` (figure → its source keys). Encodes the **shared-data graph**, so a source used by two figures is listed and built once. |
| `_figdata.py`     | I/O helper: `load(stem)` for plotters; `source(key)`/`dump(stem, …)` for data scripts. A missing raw input raises an actionable error naming the producer and where it runs. |
| `make_figdata.py` | driver over the registry (see below). |

## Colour convention (keep this consistent across all figures)

The first three default-matplotlib colours are **reserved for the paper's standard surrogate
models**, so a colour means the same model in every figure:

| colour | code | meaning |
|---|---|---|
| `tab:blue`   | `C0` | **value** model |
| `tab:orange` | `C1` | **value+gradient** (Hermite) model |

Only `C0` and `C1` are reserved. `C2` (`tab:green`) is **not** — an earlier version of this
convention held it back for a third standard model, and no figure needs it.

Everything else is a non-model colour:

- **Any other categorical colour coding** — sampling ranges, spin/`J`/`|S|` values, tilt angles,
  per-panel diagnostics (residual, `ψ`-vs-TP error, …) — **must start at `C2` (`tab:green`)** and step
  through the default cycle (`tab:green`, `tab:red`, `tab:purple`, `tab:brown`, `tab:pink`,
  `tab:gray`, …). Never reuse `C0`–`C1` for a non-model series. In code this is just
  `color=f"C{2 + i}"` in the loop, or a palette that starts at `tab:green`.
- A figure of **single-series panels** takes `C2` in *every* panel: with one series per panel there
  is nothing to distinguish, and stepping the colour would imply a coding that is not there. Fig. 8
  is the example.
- **Grey** (`"0.6"`) is the cold-start / black-box baseline.
- **Black** is for reference lines and oracle anchors (identity diagonals, TwoPunctures markers).

When adding or editing a figure, check it against this table before committing.

## Distributions — median marker with min–max whiskers (not a shaded band)

Wherever a curve summarises a *sample* (held-out base points, sampled configurations, Newton
polish steps), the paper draws it as a **median marker with whiskers spanning the sample minimum
to maximum**, never as a shaded envelope. In code that is one `errorbar`:

```python
ax.errorbar(x + off, med, yerr=[med - lo, hi - med], fmt="-o", color=C,
            ms=5, lw=1.7, capsize=3.5, elinewidth=1.1, capthick=1.1, label=...)
```

Two series in one panel are nudged apart in `x` (`off = ±0.10`) so their whiskers stay readable.
Figures 1, 3, 4, 5, 6 and 8 all use this idiom; the whisker caps make the extremes of the sample
legible in a way a translucent band does not, which matters because several claims in the text are
about the *minimum* or *maximum* of the sample rather than its median. Do not add a separate
"max" line — the upper whisker already carries it.

## Lines are solid; no y sub-ticks

Data series are drawn with **solid** lines (`fmt="-o"`), never dashed or dash-dotted: series are
distinguished by colour and marker, and dashing a data line reads as a fit or a reference rather
than a measurement. Broken lines mark things that are *not* measurements — dotted grey (`ls=":"`)
for reference lines (thresholds, gates, Nyquist limits, identity diagonals) and dashed for fitted
rate lines (Fig. 2).

Log y axes carry **major ticks and gridlines only**:

```python
ax.yaxis.set_minor_locator(NullLocator())   # no y sub-ticks
ax.grid(True, which="major", alpha=0.3)
```

Figures 2, 3, 8 and 9 all do this. It matters only for panels spanning few decades, which is
where matplotlib's `LogLocator` emits sub-decade minors — measured, ~19 of them over 2.6 decades
and ~35 over 4.6, dense enough to read as grey hatching behind the data. Wide-range panels
(Fig. 6 spans 13.6 decades) get none from the locator anyway, so setting `NullLocator`
unconditionally costs nothing and keeps every log panel in the paper looking the same.

## Figure size / aspect ratio (uniform across the paper)

Every figure uses **one uniform per-panel aspect ratio**, so panels look consistent throughout
the paper no matter how many panels a figure has. The geometry lives in one place, `_figstyle.py`:

- `PANEL_W`, `PANEL_H` — per-panel width and height in inches (currently `4.5 × 3.0`, a 3:2 ≈
  golden landscape panel).
- `figdims(nrow, ncol)` → `(ncol*PANEL_W, nrow*PANEL_H)`.

A plotter **must** size its figure through `figdims`, never a hand-picked `figsize`:

```python
from _figstyle import figdims
fig, axes = plt.subplots(nrow, ncol, figsize=figdims(nrow, ncol))
```

LaTeX scales each figure to `\columnwidth` (single-column) or `\textwidth` (`figure*`), which
preserves the matplotlib aspect, so the **rendered per-panel aspect ratio is identical in every
figure**. To restyle sizing paper-wide, change `PANEL_W`/`PANEL_H` in that one file and re-run the
plotters.

### Fonts — keep the natural per-element hierarchy (do NOT flatten)

Font sizes are **not** globally forced. Each plotter keeps matplotlib's natural per-element
hierarchy — title ≥ axis labels / ticks > legend > small in-panel data labels (e.g. title 10,
legend 8, node/annotation labels 6–8) — which reads better than one flat size. Because `figdims`
is authored a bit larger than the render width, LaTeX's mild downscale makes the effective text a
little smaller than the 9 pt caption, which is the usual, natural look.

> Rejected experiment (do not redo): authoring each figure at its exact render width and fixing
> every element to the 9 pt caption size made the fonts look **too large and unnatural**. Keep the
> per-element `fontsize=` values in the plotters. (If a caption-exact size is ever wanted again, it
> is achievable — author at true `\columnwidth`/`\textwidth`, set the fonts via rcParams, and drop
> `bbox_inches="tight"` — but it was deliberately reverted.)

## Regenerating the figures (from committed data — no solves)

```bash
PY=~/micromamba/envs/BBHFM/bin/python          # the BBHFM env interpreter
cd paper/figures
for f in fig??_*_plot.py; do "$PY" "$f"; done
```

The plotters read `figdata/` only, so this needs nothing under `reports/` and no `jax`.

## Rebuilding the data (`figdata/`)

```bash
$PY make_figdata.py --check          # present/MISSING matrix for every figure + source (dedup); runs nothing
$PY make_figdata.py --all            # (re)build every figdata json whose raw sources are present
$PY make_figdata.py --fig fig03      # one figure (stem, number, or prefix); --force to overwrite
```

`--check` reports, per figure, whether its `figdata/*.json` exists and whether its raw sources are
present under `reports/`; blocked figures list the exact producer command and whether it runs on the
laptop or the cluster. Most raw sources are produced by heavy CPU runs on the IVS cluster (see the
paper cluster prompt); once produced under `reports/`, the matching `*_data.py` distills them.
`fig07`'s data script (eccentricity) is the one that needs `jax` + the `lm.initial_data` package (it evaluates the
surrogate once to precompute the smooth curves).

## Figure → data script → raw source(s)

Sources marked **(cluster)** are produced by a run on IVS; **shared** sources are built once and
consumed by both listed figures.

| Fig | data script | raw source key(s) (`registry.SOURCES`) |
|----:|-------------|------------------------------------------|
| 1  | `fig01_peraxis_hermite_data.py`    | `peraxis_dist_chi` *(distribution over random base points)* |
| 2  | `fig02_walls_data.py`              | `walls_dense` *(both walls — separation + spin — merged into one figure)* |
| 3  | `fig03_joint_dist_data.py`         | `joint_dist_4d`, `joint_dist_cross_4d`, `joint_dist_8d` *(pending)*, `joint_dist_hermite_8d` *(pending)* |
| 4  | `fig04_polish_staircase_data.py`   | `polish_cold_{4,8}d`, `polish_pod_{4,8}d`, `polish_fielderr_4d`, `polish_fielderr_8d` *(pending)* |
| 5  | `fig05_guess_vs_memory_data.py`    | `gvm_all`, `gvm_4d_{value,cross,field,cross_field}`, `polish_table_{4d,4d_cross,8d_hermite}`, `gvm_8d_{value,field,hermite_field}` *(pending)* |
| 6  | `fig06_targeting_data.py`          | `qc_targeting` |
| 7  | `fig07_eccentricity_data.py`       | `qc_effpot` + `effpot_model` (surrogate `.npz`; distilled to json) |
| 8  | `fig08_tp_validation_data.py`      | `tp_band_sweep` *(shared with fig 9)* — the resolution ladder |
| 9  | `fig09_tp_spectrum_data.py`        | `tp_band_sweep` *(shared with fig 8)* — the azimuthal spectrum |

**Status.** Figures 2, 3, 6, 7, 8, 9 are fully data-split and their `figdata/` json is
committed; each regenerates pixel-identical to the shipped figure. Figures **1, 4, 5** still read
`reports/` directly and are pending the 8D cluster runs (see `registry.py`; the `_data.py`/plotter
split lands for them once the 8D bundle returns).
