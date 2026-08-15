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
  validation/     twopunctures, conventions, adm, constraints, export_grteclyn
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

## The certification gate

`parametric/certification.py` owns the one number the paper's central claim rests
on — `CERT_TOL = 1e-10`, the threshold on the *equilibrated* discrete constraint
residual — plus the comparison that enforces it. It is the default `tol` of every
`evaluate_polished` in the package; the threshold is not restated anywhere else.

The split to know:

- **Measuring** is unconditional. Every `evaluate_polished` computes the residual
  of the field it returns (after the last Newton step, on the returned iterate)
  and hands it back as `info.residual_norm`.
- **Enforcing** is opt-in, via `strict=True`. It has to be: the polish-history
  producers (`run_polish_table.py`, `run_polish_cold.py`) query at `tol=1e-12`
  precisely to watch the residual fall *through* `1e-10`, and a library-level
  raise would make that measurement untakeable.
- The gate is **closed at every exit** — `applications/qc_targeting.py`,
  `applications/qc_effpot.py`, and `pipeline/run_export_grteclyn.py`. That is what
  makes the paper's "a gate checked before the datum is returned" true of the
  package rather than only of its bookkeeping.

Historical note, because the failure mode was silent: before 2026-08-13 the
default `tol` was `1e-12` — *below* the equilibrated residual's roundoff floor
(~2e-12 on the production grid) — so `info.converged` was False on every
default-tol query, the solver's own `if rn < tol: break` was unreachable, and no
caller compared the residual to anything. `tests/test_certification.py` pins all
of it, including that the `1e-12` default cannot creep back.

`attach_solve_fn_3d` (the loader-side wiring that makes a *shipped* model
queryable) defaults `retry_tol=CERT_TOL`, enabling the damped-Newton
globalization that `make_solve_fn` leaves off by default. The asymmetry is
deliberate: build-time behaviour stays bit-for-bit unchanged, while the query
path — the one the certification claim is about, and the one that meets the
extreme corners where Newton–Krylov stalls globally — gets the fallback.

## What was kept

- The full `solver` / `parametric` / `applications` / `validation` module
  hierarchy. The production QC/χ stack (`solver_3d`, `solver_3d_nk`,
  `operators_3d`, `separable`, `source_3d`, `diagnostics_3d`; `parametric_nd_smolyak`,
  `hermite_smolyak{,_pod,_pod_cross}`, `quasicircular`, `solve_store`;
  `sensitivity_3d{,_qc,_cross,_cross_bq}`, `qc_targeting`, `qc_effpot`,
  `control`) plus its **transitively-required base layers** (the axisymmetric
  two-centre ABT rungs `operators_abt`/`source`/`solver_abt`; the Hermite/ND base
  rungs `parametric`, `parametric_nd`, `hermite`, `hermite_nd`, `hermite_pod`,
  `parametric_nd_2c/_3d`). These are the paper's method ladder — each a distinct
  model, all test-covered — not redundant copies.
- The acceptance suite (measured 2026-08-15: **622 passed, 40m57s**, no failures
  and no skips, with a peer suite competing for the box at load 8–16; `sympy` is a
  declared `[dev]` extra, so the one long-standing expected failure is gone. 29
  `slow` tests need the external TwoPunctures oracle). Earlier measurements: 611 /
  36m35s and 589 / 2h01m on 2026-08-13. **The count is only partly accounted
  for** — the evaluate_field/GRTeclyn commit adds 7 gates to 611, reaching 618, and
  the remaining four tests are unexplained and unbisected. Re-measure rather than
  quoting this; the suite is dominated by elliptic solves and both the solver cost
  and the machine load move it.
- The canonical figure producers + χ model builders, as `…pipeline`.
- The paper source + figure scripts + the 10 figure PDFs.

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

## What the prune removed (2026-08-15)

The prune deferred at migration time (see "Deferred", below), executed. Every
deletion was grepped against the
sibling `LMID-curved-puncture`'s `src/` **and** `tests/` first — a leaf's own
suite is blind to its consumers — and the sibling imports none of it. What it
does import from here is unchanged: `solver.{operators_3d, operators_abt,
separable, solver_3d, solver_3d_nk, solver_abt, source, source_3d}`,
`validation.adm`, and `parametric.{quasicircular, parametric_nd, hermite_smolyak*,
parametric_nd_smolyak}`.

**Modules (7).** The 1-D pedagogical rung `solver/{solver.py, operators.py,
diagnostics.py}`, reachable only from `tests/test_spatial_2c.py` once
`parametric.from_problem` went; `parametric/parametric_2c.py`, test-only;
`validation/compare.py`, a full orphan; `pipeline/plot_3d_sweep.py`, which
plotted retired figure identities and ran `os.makedirs` at *import*, creating
junk inside the installed package; `pipeline/run_3d_validation_sweep.py`,
superseded by `run_tp_random_sweep`.

**Symbols.** `parametric.from_problem` (bit-rotted: it called
`solver.newton_solve(prob, m=q, …)`, a kwarg that never existed, and
`solver.tangent`, which does not exist — uncallable since the migration);
`ParametricSolverND`'s tangent-predictor branch, unreachable because no
construction site ever passed `tangent_fn`, and wrong for any non-1-D field;
`operators_3d.block_operator_m`; `solver_3d._interp1`; `solver_abt.{adm_mass,
residual_norm}`, superseded by `validation.adm`'s spectral extraction;
`validation.adm.psi_at`; `sensitivity_3d_qc.QC_TANGENT_AXES`;
`qc_targeting.M_ADM`'s unused `q=` parameter; a dead `scipy.linalg` import and a
stale `noqa: F401` on an import that *is* used.

**Tests.** `test_spatial_2c.py` and `test_parametric_2c.py` went with their
modules. No gate on a published claim was lost: the P=0 exact-fixed-point gate
survives at both kept layers (`test_abt_2c.py`, `test_solver_3d.py`), and the
analyticity-wall study behind the paper's fig02 is `parametric_nd_2c`'s — gated
twice over, in `test_parametric_nd.py` and `test_parametric_3d.py`, on the same
`b_min` ladder. The bit-for-bit 1-D reduction oracle moved into
`test_parametric_nd.py` as two local closures over `parametric.ParametricSolver`,
so the reduction gates still compare against the same construction.

**Relocations.** `convergence_table` moved from the pruned `diagnostics` to
`diagnostics_3d` — the attribution `CLAUDE.md` carried before 2026-08-15, and now
true rather than aspirational. `MODELS`,
`load_pod_truncated` and `GAP_MIN` moved from the *producer*
`pipeline/run_guess_vs_memory.py` into `pipeline/fielderr_shared.py`, where
library code belongs — three other producers were importing a producer's module
globals. `run_guess_vs_memory` re-exports them, so no consumer broke.

**Kept deliberately.** `pipeline/run_separable_sweep.py` is not dead: it is the
evidence producer behind the separable default, and was referenced nowhere in the
public tree. It is now documented in [`DATA.md`](DATA.md) with what it gates and
its pass criteria. Also kept: `validation/constraints.py`,
`pipeline/qc_chi_tangent.py`, and the whole of `applications/`.

## Deferred (not done in this migration)

- **Marginal module prune — done 2026-08-15.** The 3 tiny 1-D pedagogical modules
  (`solver/solver.py`, `operators.py`, `diagnostics.py`) and
  `parametric/parametric_2c.py` were unreachable by the production closure and are
  gone, with three orphans beside them (`validation/compare.py`,
  `pipeline/plot_3d_sweep.py`, `pipeline/run_3d_validation_sweep.py`) — see "What
  the prune removed", above. `applications/sensitivity.py` was NOT droppable and
  stays — surviving Hermite-ND tests use it via `hermite_nd.from_problem_nd_hermite`.
- **Figure recompute (Stage 2).** The `figNN_*_data.py` scripts still carry the
  old "read `reports/` cache" logic — every one of them distils via
  `_figdata.load_source`, and none runs the solver. Rewiring them to genuinely
  recompute from the solver/ROM (two-tier) is Stage 2 — see `DATA.md` and
  `paper/figures/registry.py`. Until it is done, do not write "recompute" in a
  doc: `README.md` and `CLAUDE.md` both claimed it, and both were corrected on
  2026-08-15.
- **Docstring cleanup — done 2026-08-15.** The retired add-only banners, the
  citations to private planning documents (`plan.md`, `PAPER_PLAN`,
  `GRADIENT_ENHANCED_PLAN.md`, `notes/`, `reports/*/analysis.md`,
  `experiments/ml/`), the private milestone jargon in prose, and the
  `~/micromamba/envs/BBHFM/bin/python` invocation strings were swept out of
  `src/`, the docs and `paper/figures/`. Two deliberate survivors: this file's and
  `CLAUDE.md`'s references to the BBHFM monorepo, which are migration *history*;
  and `solver/operators_3d.py:6`, which states that the add-only policy **is**
  retired. Still open: milestone tags inside runtime `print`/plot-title strings
  (`[S6]`, `[S7-merge]`, `risk R3`) — changing those changes program output, so
  they were left for whoever touches the code.
  The old `sandbox/parasol/…` invocation strings were repaired earlier to real
  `-m lemaitre.initial_data.conformally_flat_puncture.pipeline.…` commands.

## Which model is shipped

`src/lemaitre/initial_data/conformally_flat_puncture/pipeline/production_model.py` is the single source of truth for the shipped surrogate (enhanced axes, cross term, POD ranks, stored-memory accounting); `production_box.py` is its sibling for the parameter box. Narrative and tables: [`MODELS.md`](MODELS.md).
