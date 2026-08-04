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
- **TwoPunctures validation** (fig08, fig09, fig10) — needs the external oracle binary
  (`make oracle`; the build script is bundled). Only these three figures depend on
  it.
- **TwoPunctures over the production box** (fig10) — `run_tp_random_sweep.py --n 100
  --workers 6`. One oracle call dominates each sample (~2–8 min, and markedly slower
  for spinning configurations), so budget a few hours wall-clock even parallel; every
  solve grid in `--grids` shares that one call, which is why the ladder is nearly free.
  Complements fig08/fig09 rather than replacing them: those are convergence ladders at
  a *fixed* configuration (x axis = resolution), which is what shows the difference to
  be resolution- rather than solver-limited; this samples the box the models are
  claimed over. See that producer's docstring for the two findings it exists to report
  (spin, not `q`, drives the disagreement and it converges; the certified residual and
  the field agreement peak on *different* grids).

## What is committed vs regenerated

- **Committed:** the figure **PDFs** (so `pdflatex paper/paper.tex` works out of
  the box), the source scripts, and **`paper/figures/figdata/*.json`**.
- **Why figdata is committed:** the plotters read `figdata/figNN_*.json` and
  nothing else — no `reports/`, no model corpus, no jax. Committing it (~150 kB
  for all of them) is what lets anyone **replot from a bare clone**, with only
  matplotlib and no copy step:

  ```bash
  cd paper/figures && for f in fig??_*_plot.py; do python "$f"; done
  ```

- **Regenerated:** `make figdata` **recomputes** that json from the raw sources,
  and needs the heavy tier (`$LM_REPORTS`, the multi-GB corpora, the cluster).
  That is the only step a clone cannot do.

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

**The consumer needs the DENSE artifact, under a fixed name.** `run_qc_effpot`
loads the model through `qc_effpot.load_model` →
`parametric_nd.load_parametric`, which asserts `meta["kind"] == "dense"` and so
rejects a Smolyak file outright; it reads the fixed basename
`surrogate_bpt_ecc.npz` (`run_qc_effpot.MODEL`, and the `effpot_model` entry in
`registry.SOURCES`). A `--level`-only build therefore does *not* feed fig07: it
writes `surrogate_smolyak_bpt_ecc_L5.npz`, which is both the wrong kind and the
wrong name. The build must pass `--dense-Q` and `--dense-name`:

```bash
python -m lm.initial_data.pipeline.build_surrogate \
    --box bpt_ecc --level 5 --dense-Q 16 --dense-name surrogate_bpt_ecc.npz \
    --Na 44 --Nb 32 --Nphi 8 --solver nk --store --code-tag chi-rebuild
```

`--dense-name` exists for exactly this: a consumer that hardcodes the filename.
`Q=16` (17 nodes/axis) is derived the same way as the edges — the historical
model was dense at `Q=7`/`Q=6` (8 and 7 nodes) over the narrower `b∈[2.5,6.5]`,
so 17 nodes holds that resolution density across the ~1.75× wider production `b`
range, and 17 is on the nested Chebyshev–Lobatto ladder (1,3,5,9,17,33) so the
sparse build's nodes are reused from the solve store rather than re-solved.

**Verification gate before this model is used:** the study locates the circular
orbit as the *interior* minimum of `∂E_b/∂b|_J`. Raising `b_min` to `B_MIN` can
push `b_circ` onto the lower edge for the smallest `J`, in which case the
"circular orbit" is an edge artifact rather than a measurement. Check `b_circ` is
strictly interior for every `J` and stop if it is not.

## The TwoPunctures oracle (fig09)

Both of fig09's panels — the quasi-circular ψ and `M_ADM` comparisons *against*
TwoPunctures — require the binary, and those are irreducibly a comparison. The
`panelA`/`anchors` blocks (ADM-`J` tilt vs spin tilt, from `sweep_3d`, needing no
oracle) are still emitted by the data script but are no longer plotted: measured,
θ_J tracked θ_S to ~1e-14 deg for every |S| and every TP anchor, so the panel was
three coincident curves on the line y=x, and the identity is now stated in the
appendix text instead.

### Where the source comes from, and how the binary is built

`make oracle` only echoes — the build script is **not** in this repo (it must not
be: the oracle is deliberately external, so that "agrees with TwoPunctures" means
something). It lives beside the binary it produces:

```bash
bash ~/.cache/bbhfm/parasol_tp_oracle/build.sh --check    # -> tp_solve  (+ self-test)
```

Provenance of everything that build script compiles:

| layer | origin |
|---|---|
| physics | Einstein Toolkit thorn **`TwoPunctures`** (M. Ansorg, E. Schnetter, F. Löffler) — the single-domain spectral puncture solver of Ansorg, Brügmann & Tichy, *PRD* **70**, 064011 (2004), arXiv:gr-qc/0404056. Upstream `https://bitbucket.org/einsteintoolkit/einsteininitialdata`. **LGPL v2.0+.** |
| C port | Z. B. Etienne's Cactus-free port, shipped inside the `nrpy` package as `nrpy/infrastructures/BHaH/general_relativity/TwoPunctures/` (`https://github.com/nrpy/nrpy`, PyPI `nrpy`). The build **pins** `nrpy==2.2026.6`. |
| numerics | GSL (BiCGStab + linear algebra); built against GSL 2.7.1, linked statically so the binary is node-portable. |

`build.sh` pip-installs the pinned `nrpy` into a throwaway venv, runs `emit_c.py`
to write the six TwoPunctures translation units to disk **verbatim** (no upstream
C is edited or retyped), and compiles them with two small local files:
`shim/BHaH_defines.h` (the ~40-line subset of BH@H's generated header the
solver actually uses — `REAL`, `derivs`, `ID_persist_struct`, transcribed from
nrpy's own `ID_persist_str()`) and `shim/tp_solve_main.c` (argv/stdin/stdout glue
only: it fills the struct, calls the unmodified `TP_solve()`, and evaluates the
result with the unmodified `PunctIntPolAtArbitPositionFast()`). BH@H's
`TP_Interp()` is not built — it only exists to fill a BH@H grid.

The build is **serial on purpose**: upstream parallelises the BiCGStab
line-relaxation preconditioner, which would make the Krylov path
schedule-dependent. `tests/test_validation_spin.py::test_spin_axisymmetry_nphi`
diffs ψ across two separate invocations at `1e-10`, and a paper oracle should be
bit-reproducible, so ~2× wall-clock is traded for determinism (`TP_OPENMP=1`
overrides).

`build.sh --check` verifies the binary against closed-form Brill–Lindquist: at
`P=0` the regular correction must vanish identically and `E = m_A + m_B`,
`m^ADM_± = m_± + m_+m_-/(4b)`, `ψ = 1 + m_A/2r_A + m_B/2r_B` — all reproduced to
**0.0** absolute. Aligned spins give `J = (S_A+S_B, 0, 0)` exactly. Resolution
`n = 32→64` shows clean spectral convergence (`E` settles by ~1e-12).

Set `LM_TP_BIN` to use a binary somewhere else; otherwise
`~/.cache/bbhfm/parasol_tp_oracle/tp_solve` is the default that
`validation/twopunctures.py` looks for.

> Status: the recompute wiring (rewriting `figNN_*_data.py` to compute from the
> solver/ROM instead of reading `reports/*.json`) is **Stage 2** — not yet done.
> Today the data scripts still carry the old cache-read logic.
