# Regenerating the paper's data & figures

The design goal: figure scripts **recompute** their numbers from the solver / ROM,
not read a pre-baked cache. There are two tiers.

## Entry points

```bash
make figdata     # (re)compute manuscript/figures/figdata/*.json from the solver/ROM
make figures     # figdata, then plot every fig??_*_plot.py -> PDF
make test        # acceptance suite (fast tier)
make models      # heavy: (re)build the chi surrogate corpora  [cluster]
make oracle      # build the external TwoPunctures validation binary
```

## Two tiers

**Laptop tier (fast).** Most figures recompute in seconds–minutes from a *shipped
surrogate model artifact* (the χ Smolyak/Hermite/POD models) via the ROM. The
per-figure producers live in `src/lemaitre/initial_data/pipeline/` and are mapped
to figures by `manuscript/figures/registry.py` (the single source of truth for
the figure→producer→artifact graph).

**Heavy tier (cluster / oracle).**
- **χ surrogate corpora** — built by `pipeline/{build_surrogate_chi, run_8d_chi_array,
  build_pod_hermite_model_chi, build_pod_hermite_model_chi_8d,
  build_pod_hermite_chi8d_array, build_cross_model_chi}`. Multi-GB; produced on the
  cluster. Not committed (`reports/` is gitignored). Point the figure scripts at
  the built artifacts (see `registry.py` / Stage-2 wiring).
- **TwoPunctures validation** (fig08, fig09) — needs the external oracle binary
  (`make oracle`; the build script is bundled). Only these two figures depend on
  it.

## What is committed vs regenerated

- **Committed:** the 9 figure **PDFs** (so `pdflatex manuscript/main.tex` works
  out of the box) and the source scripts.
- **Regenerated (gitignored `figdata/*.json`):** figure data recomputes via
  `make figdata`. As a documented fallback, a small committed `figdata` snapshot
  for the two oracle/cluster-gated figures (fig08, fig09) keeps the paper building
  without the heavy tier — force-added despite the gitignore.

## Model-artifact location convention

Heavy artifacts are read from a `reports/` tree (gitignored) or a path the figure
scripts resolve; the exact convention is finalized in Stage 2 (see
`STAGE2_HANDOFF.md`). During the migration the source corpora live in the BBHFM
tree at `sandbox/parasol/reports/`.

> Status: the recompute wiring (rewriting `figNN_*_data.py` to compute from the
> solver/ROM instead of reading `reports/*.json`) is **Stage 2** — not yet done.
> Today the data scripts still carry the old cache-read logic.
