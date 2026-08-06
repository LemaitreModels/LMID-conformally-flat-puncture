# Repository structure & migration notes

`lm.initial_data` (the paper package) was migrated out of the BBHFM
monorepo (`sandbox/parasol/`) into this standalone repo, de-cluttered and
restructured. This document is the package map and the record of what was kept,
dropped, and deferred.

## Layout

```
src/lm/__init__.py             shared namespace root (pkgutil.extend_path + lazy __getattr__)
src/lm/initial_data/
  solver/         spectral elliptic (xCFC) solver
  parametric/     parameter-space collocation / Hermite / Smolyak / POD (the ROM)
  applications/   qc_targeting, qc_effpot, control, sensitivity_3d{,_qc,_cross,_cross_bq}
  validation/     twopunctures, conventions, adm, constraints, compare
  pipeline/       canonical figure producers + model builders (runnable + importable)
tests/            acceptance suite (float64, CPU)
paper/            paper.tex + references + figures/ (data+plot scripts, helpers, registry)
docs/             this file · DATA.md · STAGE2_HANDOFF.md · GRTECLYN_CONSTRAINTS_PLAN.md
```

## Namespace

`lm` is a shared namespace across the LemaitreModels family. The meta
`lm/__init__.py` uses `pkgutil.extend_path` (so sibling repos —
`lm.early_inspiral`, `lm.ringdown`, `lm.artwork` — merge under
one `lm`) and a PEP 562 `__getattr__` for lazy submodule access:

```python
import lm
lm.initial_data.solver.solver_3d_nk    # resolves lazily
```

**Family bookkeeping:** exactly one installed distribution may own
`lm/__init__.py`. While LM-initial-data is the only sibling it lives here;
when a second sibling repo is created, factor it into a dedicated `lm-core`
distribution all siblings depend on.

## What was kept

- The full `solver` / `parametric` / `applications` / `validation` module
  hierarchy. The production QC/χ stack (`solver_3d`, `solver_3d_nk`,
  `operators_3d`, `source_3d`, `diagnostics_3d`; `parametric_nd_smolyak`,
  `hermite_smolyak{,_pod,_pod_cross}`, `quasicircular`, `solve_store`;
  `sensitivity_3d{,_qc,_cross,_cross_bq}`, `qc_targeting`, `qc_effpot`,
  `control`) plus its **transitively-required base layers** (the axisymmetric
  two-centre ABT rungs `operators_abt`/`source`/`solver_abt`; the Hermite/ND base
  rungs `parametric`, `parametric_nd`, `hermite`, `hermite_nd`, `hermite_pod`,
  `parametric_nd_2c/_3d`). These are the paper's method ladder — each a distinct
  model, all test-covered — not redundant copies.
- The 32-file acceptance suite (fast tier: **262 passing**; 29 `slow` tests need
  the external TwoPunctures oracle).
- The canonical figure producers + χ model builders, as `…pipeline`.
- The paper source + figure scripts + the 9 figure PDFs.

## What was dropped (add-only clutter, not migrated)

- **236 byte-identical `… 2.*` duplicate files** (a committed cloud-sync
  conflicted-copy accident in the source tree).
- ~60 add-only / scratch root scripts: pre-χ and bare-mass predecessors, the
  q-tangent ablation drivers, the earliest head-on/prototype drivers, exploratory
  sweeps, and old one-off plotters (superseded by the `pipeline` + registry set).
- `experiments/ml/` (a POD/ML side-quest), `notes/`, historical handoff docs
  (`HANDOFF*.md`, `GRADIENT_ENHANCED_PLAN.md`), LaTeX build artifacts,
  `__pycache__`, `manuscript/papers/` (reference PDFs), and the gitignored
  bare-mass `reports/*/models/` corpora (superseded by the χ models).

## Deferred (not done in this migration)

- **Marginal module prune.** The import-closure trace found only 3 tiny 1-D
  pedagogical modules (`solver/solver.py`, `operators.py`, `diagnostics.py`) plus
  `parametric/parametric_2c.py` are *truly* unreachable by the production closure.
  Dropping them needs minor test surgery (relocate a `convergence_table` printer;
  drop/repoint a few 1-D/2c tests). `applications/sensitivity.py` is NOT droppable
  — surviving Hermite-ND tests use it via `hermite_nd.from_problem_nd_hermite`.
  Left in for a green baseline; can be pruned on request.
- **Figure recompute (Stage 2).** The `figNN_*_data.py` scripts still carry the
  old "read `reports/` cache" logic. Rewiring them to genuinely recompute from the
  solver/ROM (two-tier) is Stage 2 — see `STAGE2_HANDOFF.md` and `DATA.md`.
- Cosmetic docstring cleanup (a few module/producer docstrings still say
  "add-only"); the stale figures `README.md`. The old `sandbox/parasol/…`
  invocation strings have been repaired to real `-m lm.initial_data.pipeline.…`
  (or script-path) commands.

## Which model is shipped

`src/lm/initial_data/pipeline/production_model.py` is the single source of truth for the shipped surrogate (enhanced axes, cross term, POD ranks, stored-memory accounting); `production_box.py` is its sibling for the parameter box. Narrative and tables: [`MODELS.md`](MODELS.md).
