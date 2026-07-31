# Regenerating the paper's data & figures

The design goal: figure scripts **recompute** their numbers from the solver / ROM,
not read a pre-baked cache. There are two tiers.

## Entry points

```bash
make figdata     # (re)compute paper/figures/figdata/*.json from the solver/ROM
make figures     # figdata, then plot every fig??_*_plot.py -> PDF
make tabdata     # (re)compute paper/tables/tabdata/*.json from the solver
make tables      # tabdata, then render every tab??_*_tex.py -> .tex
make test        # acceptance suite (fast tier)
make models      # heavy: (re)build the chi surrogate corpora  [cluster]
make oracle      # build the external TwoPunctures validation binary
```

## Tables

The two appendix tables (the parameter-sensitivity verification) use the same
two-tier pattern as the figures, but sit entirely in the laptop tier: both
recompute from the solver in seconds with no corpus and no oracle. The canonical
producer is `pipeline/run_tangent_verification.py`; `tabdata/*.json` is a
gitignored build output while the rendered `paper/tables/tab??_*.tex` is committed
(as the figure PDFs are), and `paper.tex` `\input`s it. See
`paper/tables/README.md`.

## Two tiers

**Laptop tier (fast).** Most figures recompute in seconds–minutes from a *shipped
surrogate model artifact* (the χ Smolyak/Hermite/POD models) via the ROM. The
per-figure producers live in `src/lm/initial_data/pipeline/` and are mapped
to figures by `paper/figures/registry.py` (the single source of truth for
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

- **Committed:** the 9 figure **PDFs** (so `pdflatex paper/paper.tex` works
  out of the box) and the source scripts.
- **Regenerated (gitignored `figdata/*.json`):** figure data recomputes via
  `make figdata`. As a documented fallback, a small committed `figdata` snapshot
  for the two oracle/cluster-gated figures (fig08, fig09) keeps the paper building
  without the heavy tier — force-added despite the gitignore.

## Model-artifact location convention

**One setting: `$LM_REPORTS`.** Every producer in `src/lm/initial_data/pipeline/`
and `paper/figures/_figdata` resolves the heavy tree through
`lm.initial_data.paths.reports_root()`:

1. `$LM_REPORTS` — explicit, `~` expanded, absolutised. Use this always.
2. `<pipeline>/reports` — the producers' historical location, so their behaviour
   is unchanged when the variable is unset.

Before this, producers wrote to `<pipeline>/reports` (off their own `__file__`)
while the figure scripts read `<repo_root>/reports`, so a figure could not see
the output of the producer that fed it and every source read as absent.
`tests/test_paths.py` fails if either half regresses.

## Which source comes from which producer

`paper/figures/registry.py` names a producer per source, but several of those
strings were stale or wrong. The verified mapping, and the decisions behind the
non-obvious ones:

| source | produced by | note |
|---|---|---|
| `sweep_3d` | `run_3d_sweep` | **not** `run_3d_validation_sweep`, which writes a different artifact (`3D_parametric/validation_results.json`) and returns early without the oracle. `run_3d_sweep`'s oracle use is optional, so fig08 builds without TwoPunctures. |
| `polish_pod_4d` | `run_polish_table --model pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r75.npz --tag chi4d_pod_r75_cross` | `run_polish_podrank` hardcodes the *non-cross* POD per dimension, so it cannot emit the `_cross` name. |
| `polish_table_4d_cross` | `run_polish_table --model hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz --tag qc_chi_prod_cross` | the *untruncated* cross corpus — corroborated by the `8·N·(1+d+npair)·nfeat` memory fig05 applies to it. |
| `polish_table_8d_value` | `run_polish_table --model surrogate_smolyak_spin8_qc_chi_prod_L5.npz --tag chi8d_value` | the 8-D **value** surrogate — corroborated by fig05's `8·N·nfeat` value-only memory. |
| `polish_pod_8d` | `run_polish_podrank --dim 8 --rank 250` | podrank's tag `chi8d_pod_r250` already matches byte-for-byte. |
| `gvm_4d_value` | `run_value_pod_gapfill_4d` | the 4-D sibling of the 8-D producer; same value-only POD construction, so `gvm_4d_value` and `gvm_4d_field` describe one model at two metrics. |
| `gvm_4d_field` | `run_cross_fielderr_sweep --flavours value` | the value flavour lives in the *cross* sweep so both fig05 bottom-left curves are measured against the same certified `u_true` (the expensive part), mirroring the 8-D `_sweep_flavor` design. |
| `qc_targeting` | `run_qc_targeting --n 100` | a parameter, not a missing producer. `run_qc_targeting_hermite` writes `P6/qc_targeting_hermite.json` and is a different study. |

Producers that do **not** produce what the registry once implied:
`make_polish_summary` writes only figures and a `.tex` (no JSON at all);
`run_cross_pod_r250_8d` is a model builder; `run_qc_dense_stats` carries a
pre-χ box (`b∈[1.5,4]`, dimensionful `S_Ay∈[-0.4,0.4]`) and would silently build a
different model against the production corpus.

Still without any producer: `gvm_4d_value`/`gvm_4d_field` are now covered by the
two entries above; nothing else in `registry.SOURCES` is orphaned.

## The eccentricity family (fig07)

`run_qc_effpot` needs a 2-D `(b, P_t)` surrogate, `surrogate_bpt_ecc.npz`, which
does not exist anywhere and had no builder. `build_surrogate` now declares the
family as `bpt_ecc` — the only box with a **free momentum**; every other family
either fixes head-on infall or takes the deterministic PN quasi-circular momenta
(`FIXED[...]["qc"]=1.0`), and freeing `P_t` is precisely what makes eccentricity
measurable.

Both edges are derived, not chosen:

- `b` = `production_box.B_MIN..B_MAX`, so the study shares the separations of
  every other figure. (The historical `run_qc_effpot.BOX_B=(2.6,6.4)` predates
  the box retarget and its lower edge sits below `B_MIN`.)
- `P_t = J/(2b)`, because `qc_effpot` fixes `J = 2 b P_t`. The momentum axis is
  `P_x`: `theta_to_slice3d` builds `P_A=(P_x,0,−P)`, `P_B=(−P_x,0,P)`, so for
  punctures at `z=±b` the orbital term gives `J=(0, 2 b P_x, 0)` — hence
  `P_t ≡ P_x`. Covering the study's `J∈[1.00,1.10]` therefore needs
  `P_x ∈ [J_min/(2 B_MAX), J_max/(2 B_MIN)]`.
- `FIXED["bpt_ecc"] = {"P": 0.0, "q": 1.0}` — `P` is the *radial* momentum, and
  the apsis condition is `P_r=0`; the default is `P=0.5` head-on infall, which
  would not be an apsis at all, so it must be overridden explicitly.

**Verification gate before this model is used:** the study locates the circular
orbit as the *interior* minimum of `∂E_b/∂b|_J`. Raising `b_min` to `B_MIN` can
push `b_circ` onto the lower edge for the smallest `J`, in which case the
"circular orbit" is an edge artifact rather than a measurement. Check `b_circ` is
strictly interior for every `J` and stop if it is not.

## The TwoPunctures oracle (fig09)

fig09's panel (a) — ADM-`J` tilt vs spin tilt — comes from `sweep_3d` and needs no
oracle; the TP anchor overlay already self-disables via `D_anchors.available`.
Only panels (b) and (c), the quasi-circular ψ and `M_ADM` comparisons *against*
TwoPunctures, require the binary, and those are irreducibly a comparison.
`make oracle` only echoes: **the build script is not bundled** anywhere in this
repo, contrary to earlier revisions of this file. Building it means obtaining the
Einstein Toolkit TwoPunctures C source (NRPy port) and compiling against GSL,
then setting `LM_TP_BIN`.

> Status: the recompute wiring (rewriting `figNN_*_data.py` to compute from the
> solver/ROM instead of reading `reports/*.json`) is **Stage 2** — not yet done.
> Today the data scripts still carry the old cache-read logic.
