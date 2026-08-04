# CLAUDE.md

Guidance for Claude Code when working in **LM-initial-data** (`lm.initial_data`).

> Keep this file current: when you add/rename a module, change the figure
> pipeline, or add a doc, update the relevant section before finishing.

## What this repo is

The code + paper for a **certified, differentiable,
parametric** reduced-order model of the binary-black-hole *constraint (initial
data)* solve — quasi-circular Bowen–York punctures up to the 8-D general-spin
model θ₈ = (b, q, χ_A, χ_B). It is the `initial_data` member of the **Lemaitre**
package family (see `README.md` for the namespace).

This repo was migrated out of the BBHFM monorepo (`sandbox/parasol/`) and cleaned
up: **the old add-only policy is retired.** Edit modules in place; keep exactly
one canonical version of each model. Normal engineering hygiene applies.

> **Read `docs/HISTORY_AND_FINDINGS.md` before touching the parametric ROM,
> rebuilding a corpus, or comparing models.** It records the project's origin and
> the hard-won findings (equilibrated-vs-raw residual, field error as a separate
> metric, "enhance only a small axis set", the χ q-tangent bug, the stale-corpus
> trap, the missing held-out-accuracy gate, 8-D needing ≥3 Newton steps) — several
> were discovered twice. §2 there is "do not re-litigate".

## Ground rules (load-bearing)

- **Standalone.** Depend only on `jax`, `numpy`, `scipy`, `matplotlib`. Never
  import `bbhfm`, `src.*`, `context`, `torch`, or `nrpy`. This is enforced by a
  test guard (`tests/test_*` self-containment asserts) — keep it passing.
- **float64 everywhere.** `jax.config.update("jax_enable_x64", True)` before any
  jax use. The solver is spectral; **no neural networks in the solver.**
- **Intra-package imports are relative** (`from . import ...`, `from ..solver
  import ...`). Absolute imports use the full `lm.initial_data.*` path
  (producers, tests, figure scripts). Do not reintroduce `sys.path` bootstraps —
  the package is pip-installed.
- **caffeinate long jobs** (macOS): wrap any local run >a few seconds in
  `caffeinate -i <cmd>` (tests, solves, sweeps). Not `sbatch` (cluster).
- **Estimate + report duration** for jobs >~30 s; run heavy ones in the
  background with an ETA.
- **User-owned prose is authoritative** — never regenerate author/title/abstract
  blocks in `paper/paper.tex`; make targeted edits only, and the paper
  edits come LAST.

## Commands

```bash
pip install -e ".[dev]"                       # install
caffeinate -i pytest -q                        # full acceptance suite
caffeinate -i pytest tests/test_solver_3d.py -v
make figures                                   # regenerate figure data (recompute) + plot
```

## Architecture

`src/lm/initial_data/`

- **`solver/`** — spatial elliptic (xCFC) solver. Production 3-D stack:
  `spectral` (1-D Chebyshev primitives), `operators_3d`/`source_3d` (Fourier-in-φ
  non-axisymmetric operator + Bowen–York source), `solver_3d` (modified-Newton
  build), `solver_3d_nk` (Newton–Krylov, the *certified* solve), `diagnostics_3d`
  (ADM diagnostics + `convergence_table`). The axisymmetric two-centre base
  (`operators_abt`, `source`, `solver_abt`) is a **transitively-required base
  layer** of the 3-D stack — not dead code.
- **`parametric/`** — the ROM. Production: `parametric_nd_smolyak` (sparse-grid
  value model), `hermite_smolyak`/`hermite_smolyak_pod`/`hermite_smolyak_pod_cross`
  (gradient-enhanced + POD + full-bilinear cross term), `quasicircular` (PN QC
  momenta), `solve_store` (content-addressed solve cache). The `parametric`,
  `parametric_nd`, `hermite`, `hermite_nd`, `hermite_pod`, `parametric_nd_2c/_3d`
  layers are the base rungs the Smolyak/POD models build on.
- **`applications/`** — `qc_targeting` (gradient parameter targeting),
  `qc_effpot` (eccentricity / Cook effective potential), `control` (accelerated
  parameter control), `sensitivity_3d`/`_qc`/`_cross`/`_cross_bq` (differentiable
  tangents dU/dθ, incl. the full-bilinear cross term).
- **`validation/`** — `twopunctures` (external oracle wrapper), `conventions`
  (convention map), `adm`, `constraints`, `compare` (LM-initial-data-vs-TwoPunctures).

`pipeline/figures/` and `pipeline/models/` hold the canonical producers/builders;
`paper/figures/` holds the recompute+plot scripts. See `docs/STRUCTURE.md`.

## Figure pipeline (two-tier, recompute-by-default)

`paper/figures/figNN_*_data.py` **recomputes** each figure's numbers from
the solver/ROM (loading a shipped surrogate model artifact), writing
`figdata/NN.json` as a build output; `figNN_*_plot.py` draws the PDF from it.
Driver: `make figdata` / `make figures`. Heavy inputs (χ corpora, TwoPunctures)
are the `make models` / `make oracle` tier — see `docs/DATA.md`.

`registry.FIGURES[stem]["keys"]` lists the top-level figdata keys a figure needs;
`tests/test_paper_figures.py` fails on any figdata missing one, which is how a
figdata predating a producer change is caught instead of dying inside the plotter.
A figure whose **caption states the box** should also write a `meta` provenance
block (box, axes, level, node count, model file) and declare it there — fig06 was
measured on a superseded model for a whole revision without that being visible
anywhere in its figdata.

The paper's **tables** follow the same two tiers in `paper/tables/`:
`tabNN_*_data.py` recomputes from the solver into `tabdata/NN.json` (gitignored),
`tabNN_*_tex.py` renders the `ruledtabular` body into a committed `tabNN_*.tex`
that `paper.tex` `\input`s, so no number is hand-transcribed. Driver:
`make tabdata` / `make tables`; the canonical producer is
`pipeline/run_tangent_verification.py`; `tests/test_paper_tables.py` guards both
the rendered rows and the hand-written captions against drift.

## Working cadence

Self-verifying + report-then-wait at milestone boundaries: keep each committed
unit independently tested; when a phase is large/end-to-end-only, stop and report
rather than pushing on. Run the test suite after significant changes, then
propose a commit message and **wait for the user before committing**.
