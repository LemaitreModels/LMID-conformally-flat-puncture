# CLAUDE.md

Guidance for Claude Code when working in **LMID-conformally-flat-puncture**
(`lemaitre.initial_data.conformally_flat_puncture`).

> Keep this file current: when you add/rename a module, change the figure
> pipeline, or add a doc, update the relevant section before finishing.

## What this repo is

The code + paper for a **certified, differentiable,
parametric** reduced-order model of the binary-black-hole *constraint (initial
data)* solve — quasi-circular Bowen–York punctures up to the 8-D general-spin
model θ₈ = (b, q, χ_A, χ_B). It is the `conformally_flat_puncture` model of the
`initial_data` domain of the **Lemaitre** package family (see `README.md` for the
namespace, and `docs/STRUCTURE.md` for who owns which namespace level). Its
sibling `lemaitre.initial_data.curved_puncture` (repo `LMID-curved-puncture`) is the
non-conformally-flat successor and **depends on this package** — it reuses the
ABT chart, the Newton–Krylov solver and the ROM. Never invert that: nothing here
may import `lemaitre.initial_data.curved_puncture`.

This repo was migrated out of the BBHFM monorepo (`sandbox/parasol/`) and cleaned
up: **the old add-only policy is retired.** Edit modules in place; keep exactly
one canonical version of each model. Normal engineering hygiene applies.

> **Which model is shipped is defined in exactly one place:**
> `src/lemaitre/initial_data/conformally_flat_puncture/pipeline/production_model.py` (narrative: `docs/MODELS.md`).
> Never restate the enhanced axes, POD ranks or stored-memory numbers anywhere else
> — producers, figures, tests and the paper all read them from there.
>
> **Several design choices here were learned the hard way** — the equilibrated
> (not raw) residual as the certification metric, field error as a metric separate
> from residual, the small gradient-enhanced axis set with its cross term, and the
> held-out accuracy gate a model must pass before it is consumed. They are enforced
> in the producers and tests rather than restated in prose. Do not re-tune them or
> rebuild a corpus without checking with the maintainers first.
>
> **The certification gate lives in exactly one place:**
> `src/lemaitre/initial_data/conformally_flat_puncture/parametric/certification.py`.
> `CERT_TOL = 1e-10` is the paper's threshold and the default `tol` of every
> `evaluate_polished`; never restate the number elsewhere. The gate is **opt-in**
> (`strict=True`) because the polish-history producers query below it on purpose —
> but it is **closed at every point where a datum leaves the package** (`qc_targeting`,
> `qc_effpot`, the GRTeclyn export). If you add such an exit, close it there too.

## Ground rules (load-bearing)

- **Standalone.** Depend only on `jax`, `numpy`, `scipy`, `matplotlib` (plus the
  dependency-free `lemaitre` / `LM-initial-data` namespace packages). Never
  import `bbhfm`, `src.*`, `context`, `torch`, or `nrpy`. Enforced by
  `tests/test_self_containment.py` (AST-parses the whole `src/` tree, and pins
  the namespace ownership) plus per-module `test_standalone_imports`
  spot-checks — keep them passing.
- **float64 everywhere.** `jax.config.update("jax_enable_x64", True)` before any
  jax use. The solver is spectral; **no neural networks in the solver.**
- **Intra-package imports are relative** (`from . import ...`, `from ..solver
  import ...`). Absolute imports use the full `lemaitre.initial_data.conformally_flat_puncture.*` path
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
pip install -e ".[dev]" --config-settings editable_mode=compat   # install
caffeinate -i pytest -q                        # full acceptance suite
caffeinate -i pytest tests/test_solver_3d.py -v
make figures                                   # regenerate figure data (recompute) + plot
```

**The full suite is a ~40-minute job.** Measured 2026-08-13: **611 tests,
36m35s** on an M-series laptop (with another pytest run competing for part of it).
Start it in the background and do other work — and do not pipe it through
`tail`/`head`, which buffers until pytest exits so a running suite looks hung.

That figure has moved twice, so treat it as a measurement and not a constant.
Earlier the same day it was **589 tests, 2h01m**, and before that "542 tests,
~36 min". The drop back to ~36 min is not a stale note: the suite is dominated by
elliptic solves, and the solver's assembly and per-step linear algebra were made
several times cheaper (see `solver/operators_3d.py` and `solver/separable.py`).
Re-measure rather than quoting this line if the number matters to a decision.

`tests/test_source_spin.py::test_sympy_exact_spin_closed_form` needs `sympy`,
which is a **test-only** dependency — the standalone guard forbids importing it
from `src/`. It is declared in this leaf's `[dev]` extra and in the workspace
`environment.yml`; keep the two in step. It used to be declared in neither, and
because the test fails rather than skips without it, a clean environment reported
a failure unrelated to whatever was being changed. If you see that failure, the
environment is stale — `pip install -e ".[dev]"`, or `./scripts/create_env.sh`
from the workspace root.

For a fast check while iterating, these three cover the structural invariants in
about four seconds:

```bash
caffeinate -i pytest -q tests/test_self_containment.py tests/test_certification.py \
                       tests/test_qc_wiring.py          # 255 tests, ~2 s
```

## Architecture

`src/lemaitre/initial_data/conformally_flat_puncture/`

- **`solver/`** — spatial elliptic (xCFC) solver. Production 3-D stack:
  `spectral` (1-D Chebyshev primitives), `operators_3d`/`source_3d` (Fourier-in-φ
  non-axisymmetric operator + Bowen–York source), `separable` (the same operator
  as 1-D Kronecker factors + its exact inverse), `solver_3d` (modified-Newton
  build), `solver_3d_nk` (Newton–Krylov, the *certified* solve), `diagnostics_3d`
  (ADM diagnostics + `convergence_table`). The axisymmetric two-centre base
  (`operators_abt`, `source`, `solver_abt`) is a **transitively-required base
  layer** of the 3-D stack — not dead code.

  **Who owns the linear operator.** `operators_3d` owns the *dense* per-m blocks
  and is the only place they are built; `operators_3d.mode_operators_cached`
  hands out shared, **read-only** blocks memoized on `(Na, Nb, Nφ, b)`, so a
  second slice at the same separation costs nothing (Newton copies before adding
  its nonlinear diagonal). `separable` owns the *matrix-free* representation: the
  1-D Kronecker factors of each block, its fast-diagonalization inverse, and the
  row-equilibration scales in `O(Na·Nb)` — built once per grid and reused at every
  separation and every parameter point. `solver_3d.linear_apply` is the single
  place that knows which of the two an assembly holds; the residual, the Jacobian
  action and the tangents all go through it. Which representation an assembly gets
  is chosen once, in `solver_3d.assemble(..., separable=)`.

  > **`operators_abt` is the axisymmetric regression oracle, and the m=0 3-D block
  > must equal its operator BIT-FOR-BIT.** That is what makes the Nφ=1 reduction
  > reproduce the 2-D Newton solve to ~1e-16 rather than ~1e-13, and it is a
  > tighter constraint than "don't edit `operators_abt`": any change to how
  > `operators_3d` assembles the m=0 block breaks it just as effectively. Pinned by
  > `tests/test_solver_3d_fast.py::test_m0_block_is_the_frozen_axisymmetric_operator`.
  > In particular the assembly does **not** use
  > `kron(D,I) @ kron(D,I) = kron(D@D, I)`, which would be ~40× cheaper but agrees
  > only to ~1e-16: taking it needs the same change in `operators_abt` in the same
  > commit, and that is a maintainer decision.

  > **The separable preconditioner is opt-in, not the default.** `separable=True`
  > drops the nonlinear diagonal from the preconditioner — that is exactly what
  > makes it b- and θ-independent — so it is no longer the exact Jacobian at
  > Nφ=1, and GMRES takes a handful of iterations there instead of one. The
  > default (dense) route keeps that property, and
  > `test_solver_3d.py::test_nk_axisym_reduction_reproduces_2d` keeps asserting
  > it. The separable path has its own same-answer gate in
  > `tests/test_solver_3d_fast.py`. Flipping the default would change the Krylov
  > behaviour of every production solve, the corpus and the figure data — check
  > with the maintainers first.
- **`parametric/`** — the ROM. Production: `parametric_nd_smolyak` (sparse-grid
  value model), `hermite_smolyak`/`hermite_smolyak_pod`/`hermite_smolyak_pod_cross`
  (gradient-enhanced + POD + full-bilinear cross term), `quasicircular` (PN QC
  momenta), `solve_store` (content-addressed solve cache), `certification`
  (`CERT_TOL` + the `‖R‖∞ ≤ tol` gate — see below). The `parametric`,
  `parametric_nd`, `hermite`, `hermite_nd`, `hermite_pod`, `parametric_nd_2c/_3d`
  layers are the base rungs the Smolyak/POD models build on.
- **`applications/`** — `qc_targeting` (gradient parameter targeting),
  `qc_effpot` (eccentricity / Cook effective potential), `control` (accelerated
  parameter control), `sensitivity_3d`/`_qc`/`_cross`/`_cross_bq` (differentiable
  tangents dU/dθ, incl. the full-bilinear cross term).
- **`validation/`** — `twopunctures` (external oracle wrapper), `conventions`
  (convention map), `adm`, `constraints`, `compare` (LM-initial-data-vs-TwoPunctures).

`pipeline/` holds the canonical producers/builders directly — `run_*.py` producers
and `build_*.py` model builders, no subdirectories; `paper/figures/` holds the
recompute+plot scripts. See `docs/STRUCTURE.md`.

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
