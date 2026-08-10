# lemaitre.initial_data.conformally_flat (LMID-conformally-flat-puncture)

Certified, differentiable, **parametric** conformally-flat binary-black-hole
*initial data* via spectral collocation — the code and paper.

This is the `conformally_flat` model of the `initial_data` domain of the
**Lemaitre** package family. It installs under the shared `lemaitre` namespace,
so once installed:

```python
import lemaitre as lm

lm.initial_data.conformally_flat.solver.solver_3d   # the production 3-D xCFC solver
lm.initial_data.conformally_flat.parametric         # the certified/differentiable ROM layer
```

Its sibling `lemaitre.initial_data.curved` (repo `LMID-curved-puncture`) is the
non-conformally-flat successor, and reuses this package's ABT chart,
Newton–Krylov solver and ROM directly. Further family members
(`lemaitre.early_inspiral`, …) slot in under the same `lemaitre.` prefix when
installed alongside.

## Install

```bash
pip install -e ".[dev]" --config-settings editable_mode=compat
python -c "import lemaitre as lm; lm.initial_data.conformally_flat"   # smoke check
```

Pure Python: `jax`, `numpy`, `scipy`, `matplotlib` (float64 throughout), plus the
dependency-free `lemaitre` / `lemaitre-initial-data` namespace packages. No
machine-learning framework and no external solver code are required — the
package is self-contained (enforced by `tests/test_self_containment.py`).

## Layout

```
src/lemaitre/initial_data/conformally_flat/
  solver/         spatial elliptic (xCFC) solver — production 3-D stack + base layers
  parametric/     parameter-space collocation, Hermite/Smolyak, POD — the ROM
  applications/   parameter targeting, eccentricity control, differentiable sensitivity
  validation/     TwoPunctures oracle wrapper + ADM / constraint diagnostics
  pipeline/       canonical producers, runnable and importable:
                    run_*.py    per-figure / per-table data producers
                    build_*.py  surrogate-model builders (heavy; cluster)
tests/            acceptance suite (float64, CPU)
paper/            paper.tex + figures/ (recompute + plot scripts)
docs/             DATA.md (data regeneration + oracle) · STRUCTURE.md (package map)
```

Only `lemaitre/initial_data/conformally_flat/` is shipped by this distribution —
`lemaitre/` and `lemaitre/initial_data/` are namespace levels owned by the
`lemaitre` core and the `lemaitre-initial-data` umbrella respectively, so they
carry no `__init__.py` here.

## Reproduce the paper

```bash
make test        # run the acceptance suite
make figures     # regenerate every figure's data (recompute) then plot the PDFs
make tables      # recompute the table data then render the LaTeX bodies
```

Figure data is **recomputed** from the solver / ROM (not read from cached JSON).
Two tiers, see `docs/DATA.md`:

- **laptop tier** — fast figures rebuild from the shipped surrogate model artifacts;
- **heavy tier** — the χ surrogate corpora (`make models`, cluster) and the
  TwoPunctures validation binary (`make oracle`) back the two validation figures;
  a small committed `figdata/` fallback keeps `pdflatex` working without them.

## License

GPL-3.0 (see `LICENSE`).
