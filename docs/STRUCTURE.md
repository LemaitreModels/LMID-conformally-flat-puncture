# Repository structure & migration notes

`lemaitre.initial_data.conformally_flat_puncture` (the paper package) was migrated out of the BBHFM
monorepo (`sandbox/parasol/`) into this standalone repo, de-cluttered and
restructured. This document is the package map and the record of what was kept,
dropped, and deferred.

## Layout

```
src/lemaitre/                  namespace level, owned by the `lemaitre` core (no __init__.py here)
src/lemaitre/initial_data/     namespace level, owned by `LM-initial-data` (no __init__.py here)
src/lemaitre/initial_data/conformally_flat_puncture/
  solver/         spectral elliptic (xCFC) solver
  parametric/     parameter-space collocation / Hermite / Smolyak / POD (the ROM)
  applications/   qc_targeting, qc_effpot, control, sensitivity_3d{,_qc,_cross,_cross_bq}
  validation/     twopunctures, conventions, adm, constraints, compare
  pipeline/       canonical figure producers + model builders (runnable + importable)
tests/            acceptance suite (float64, CPU)
paper/            paper.tex + references + figures/ (data+plot scripts, helpers, registry)
docs/             this file · MODELS.md (shipped model) · DATA.md (data regeneration)
```

## Namespace

`lemaitre` is a shared namespace across the LemaitreModels family, and it is
**two levels deep** here. Both `lemaitre/__init__.py` and
`lemaitre/initial_data/__init__.py` use `pkgutil.extend_path` (so sibling repos
merge under one `lemaitre`) and a PEP 562 `__getattr__` for lazy submodule
access:

```python
import lemaitre as lm
lm.initial_data.conformally_flat_puncture.solver.solver_3d_nk    # resolves lazily
```

**Family bookkeeping:** exactly one installed distribution may own each
namespace `__init__.py`, and **neither belongs to this repo**:

| namespace | `__init__.py` owned by | repository |
|---|---|---|
| `lemaitre` | `lemaitre` (the core) | `Lemaitre` |
| `lemaitre.initial_data` | `LM-initial-data` (the umbrella) | `LM-initial-data` |
| `lemaitre.initial_data.conformally_flat_puncture` | `LMID-conformally-flat-puncture` | **this repo** |
| `lemaitre.initial_data.curved_puncture` | `LMID-curved-puncture` | `LMID-curved-puncture` |

This repo ships only its own leaf package; `src/lemaitre/` and
`src/lemaitre/initial_data/` are bare directories that `setuptools`
`find_namespace` walks through (`namespaces = true`, `include` scoped to
`lemaitre.initial_data.conformally_flat_puncture*`). Shipping a second copy of either
namespace `__init__.py` would shadow the owner non-deterministically.

### Editable installs must use the static-path mode

```bash
pip install -e . --config-settings editable_mode=compat
```

**Measured, and the reason the flag is not optional.** setuptools' *modern*
editable mode installs each distribution behind a meta-path finder. Those
finders are **appended** to `sys.meta_path`, i.e. *after* `PathFinder` — so when
Python resolves `lemaitre.initial_data`, `PathFinder` runs first, walks
`lemaitre.__path__`, finds this repo's and the curved repo's bare
`lemaitre/initial_data/` directories, and builds a **PEP 420 namespace package**
out of them. It never reaches the umbrella's finder, which is the only thing
that knows where the real `lemaitre/initial_data/__init__.py` lives.

The damage is *partial and therefore easy to miss*: `import
lemaitre.initial_data.conformally_flat_puncture` still works, because the leaf's own
finder resolves it. What breaks is everything the umbrella's `__init__.py`
provides — chiefly the lazy `__getattr__`, so `lm.initial_data.conformally_flat_puncture`
raises `AttributeError` while the equivalent `import` succeeds.

`editable_mode=compat` installs a plain `.pth` that puts each `src/` on
`sys.path`, so `pkgutil.extend_path` finds the umbrella's real directory and the
regular package wins over the namespace portions. **Non-editable installs are
unaffected** — all four distributions unpack into one `site-packages/lemaitre/`
tree, so the question never arises.

`tests/test_self_containment.py::test_two_level_namespace_is_live` and
`::test_lazy_attribute_access` fail loudly, with this fix in the message, rather
than letting a degraded install pass silently.

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
  solver/ROM (two-tier) is Stage 2 — see `DATA.md` and `paper/figures/registry.py`.
- Cosmetic docstring cleanup (a few module/producer docstrings still say
  "add-only"); the stale figures `README.md`. The old `sandbox/parasol/…`
  invocation strings have been repaired to real `-m lemaitre.initial_data.conformally_flat_puncture.pipeline.…`
  (or script-path) commands.

## Which model is shipped

`src/lemaitre/initial_data/conformally_flat_puncture/pipeline/production_model.py` is the single source of truth for the shipped surrogate (enhanced axes, cross term, POD ranks, stored-memory accounting); `production_box.py` is its sibling for the parameter box. Narrative and tables: [`MODELS.md`](MODELS.md).
