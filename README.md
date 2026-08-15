# lemaitre.initial_data.conformally_flat_puncture (LMID-conformally-flat-puncture)

Certified, differentiable, **parametric** conformally-flat binary-black-hole
*initial data* via spectral collocation — the code and paper.

This is the `conformally_flat_puncture` model of the `initial_data` domain of the
**Lemaitre** package family. It installs under the shared `lemaitre` namespace,
so once installed:

```python
import lemaitre as lm

lm.initial_data.conformally_flat_puncture.solver.solver_3d   # the production 3-D xCFC solver
lm.initial_data.conformally_flat_puncture.parametric         # the certified/differentiable ROM layer
```

Its sibling `lemaitre.initial_data.curved_puncture` (repo `LMID-curved-puncture`) is the
non-conformally-flat successor, and reuses this package's ABT chart,
Newton–Krylov solver and ROM directly. Further family members
(`lemaitre.inspiral`, …) slot in under the same `lemaitre.` prefix when
installed alongside.

## Install

This distribution declares two namespace parents, `lemaitre` and
`LM-initial-data`. **Neither is published on PyPI**, so a bare
`pip install -e .` here cannot resolve them. Install all three from a checkout
of the family superproject, outermost first, and tell pip not to look upstream:

```bash
pip install numpy scipy jax matplotlib pytest sympy    # the solver stack + test deps

# from the Lemaitre superproject root, with LMID-conformally-flat-puncture checked out
for d in . LM-initial-data LM-initial-data/LMID-conformally-flat-puncture; do
  pip install -e "$d" --config-settings editable_mode=compat --no-deps
done
python -c "import lemaitre as lm; lm.initial_data.conformally_flat_puncture"   # smoke check
```

`--no-deps` and the **order** go together: pip must not try to resolve the two
unpublished parents from PyPI, and each must already be installed before its
dependents — so the runtime stack is installed first, by hand. `sympy` is the
`[dev]` extra, needed by one test that fails rather than skips without it.

`editable_mode=compat` is not optional either — the modern editable mode makes
the two namespace levels degrade into PEP 420 portions, and lazy attribute
access then fails while a direct `import` still works. See `docs/STRUCTURE.md`.

Pure Python: `jax`, `numpy`, `scipy`, `matplotlib` (float64 throughout), plus the
dependency-free `lemaitre` / `LM-initial-data` namespace packages. No
machine-learning framework and no external solver code are required — the
package is self-contained (enforced by `tests/test_self_containment.py`).

## Layout

```
src/lemaitre/initial_data/conformally_flat_puncture/
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
                  · MODELS.md (which surrogate is shipped, and what it stores)
```

Only `lemaitre/initial_data/conformally_flat_puncture/` is shipped by this distribution —
`lemaitre/` and `lemaitre/initial_data/` are namespace levels owned by the
`lemaitre` core and the `LM-initial-data` umbrella respectively, so they
carry no `__init__.py` here.

## Reproduce the paper

```bash
make test        # run the acceptance suite
make figures     # (re)build any missing figure data, then plot every PDF
make tables      # recompute the table data then render the LaTeX bodies
```

**What a bare clone can and cannot do.** The plot tier is fully reproducible: the
plotters read `paper/figures/figdata/*.json` and nothing else — no `reports/`, no
models, no jax — so `make figures` redraws all ten PDFs from committed data with
only matplotlib. `make tables` genuinely recomputes: Table I derives its node
counts and box edges from `pipeline/production_box.py` in seconds, with no solve.

The **data** tier is not reproducible from a clone, and the numbers are the
paper's:

- each `figNN_*_data.py` **distils** a raw run artifact under `$LM_REPORTS` — it
  reshapes numbers a heavy-tier run already produced, rather than recomputing
  them from the solver. `fig07` additionally evaluates a shipped surrogate model
  (`.npz`) through the ROM; `fig10` distils an external GRTeclyn run tree.
- those raw artifacts are the **heavy tier**: multi-GB χ surrogate corpora
  (`make models`, cluster), the external TwoPunctures binary (`make oracle`), and
  the GRTeclyn constraint runs. None is committed.
- `make figdata` **skips any figdata that already exists**, and all ten are
  committed — so on a clone it rebuilds nothing and `make figures` only replots.
  `--force` re-distils, and then needs the heavy tier.

Rewiring the data scripts to compute from the solver/ROM instead of distilling
cached run output is **Stage 2, and is not done**. See `docs/DATA.md`.

## License

GPL-3.0 (see `LICENSE`).
