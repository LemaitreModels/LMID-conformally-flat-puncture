# lm.initial_data (LM-initial-data)

Certified, differentiable, **parametric** binary-black-hole *initial data* via
spectral collocation — the code and paper for PARASOL.

This is the `initial_data` member of the **Lemaitre** package family. It installs
under the shared `lm` namespace, so once installed:

```python
import lm
lm.initial_data.solver.solver_3d      # the production 3-D xCFC solver
lm.initial_data.parametric            # the certified/differentiable ROM layer
```

Sibling repos (`lm.early_inspiral`, `lm.ringdown`, `lm.artwork`)
slot in under the same `lm.` prefix when installed alongside.

## Install

```bash
pip install -e ".[dev]"        # editable install + pytest
python -c "import lm; lm.initial_data"   # smoke check
```

Pure Python: `jax`, `numpy`, `scipy`, `matplotlib` (float64 throughout). No
machine-learning framework and no external solver code are required — the
package is self-contained (enforced by a test guard).

## Layout

```
src/lm/initial_data/
  solver/         spatial elliptic (xCFC) solver — production 3-D stack + base layers
  parametric/     parameter-space collocation, Hermite/Smolyak, POD — the ROM
  applications/   parameter targeting, eccentricity control, differentiable sensitivity
  validation/     TwoPunctures oracle wrapper + ADM / constraint diagnostics
tests/            acceptance suite (float64, CPU)
pipeline/
  figures/        canonical per-figure data producers
  models/         canonical surrogate-model builders (heavy; cluster)
paper/            paper.tex + figures/ (recompute + plot scripts)
docs/             DATA.md (data regeneration + oracle) · STRUCTURE.md (package map)
```

## Reproduce the paper

```bash
make test        # run the acceptance suite
make figures     # regenerate every figure's data (recompute) then plot the PDFs
```

Figure data is **recomputed** from the solver / ROM (not read from cached JSON).
Two tiers, see `docs/DATA.md`:

- **laptop tier** — fast figures rebuild from the shipped surrogate model artifacts;
- **heavy tier** — the χ surrogate corpora (`make models`, cluster) and the
  TwoPunctures validation binary (`make oracle`) back the two validation figures;
  a small committed `figdata/` fallback keeps `pdflatex` working without them.

## License

GPL-3.0 (see `LICENSE`).
