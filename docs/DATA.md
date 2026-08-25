# Regenerating the paper's data & figures

The design *goal* is that figure scripts recompute their numbers from the solver /
ROM rather than reading a pre-baked cache. **That is Stage 2, and it is not done
— see the status note at the end of this file.** What the scripts do today is
*distil*: each reads a raw run artifact a heavy-tier job already produced and
reshapes it into the committed `figdata/*.json`. State it that way in any doc you
write; `README.md` and `CLAUDE.md` claimed the recompute for a while and a referee
checking it would have found the opposite.

There are two tiers.

## Entry points

```bash
make figdata     # distil any MISSING paper/figures/figdata/*.json (--force to redo present ones)
make figures     # figdata, then plot every fig??_*_plot.py -> PDF
make tabdata     # recompute paper/tables/tabdata/*.json (instant; no solve)
make tables      # tabdata, then render every tab??_*_tex.py -> .tex
make test        # the FULL acceptance suite: ~40 min, not a fast tier
make models      # heavy: (re)build the chi surrogate corpora  [cluster]
make oracle      # build the external TwoPunctures validation binary
```

## Tables

There is **one** table, `tab01_production_box`, and it uses the same two-tier
pattern as the figures but sits entirely in the laptop tier — and unlike the
figures it genuinely recomputes: it derives every box edge, the Smolyak level, the
spatial grid and the enhanced axis set from `pipeline/production_box.py`, and the
sparse-grid node count from `parametric_nd_2c.smolyak_points`. No corpus, no
oracle, no solve; retargeting an edge in `production_box` moves the paper's number
with it. `tabdata/*.json` is a gitignored build output while the rendered
`paper/tables/tab??_*.tex` is committed (as the figure PDFs are), and `paper.tex`
`\input`s it. See `paper/tables/README.md`.

`pipeline/run_tangent_verification.py` is **not** the table producer — it fed a
paper appendix that was withdrawn, and its own docstring says so. It stays as the
runnable end-to-end cross-check behind the sensitivity claims of Sec. IV.

## Two tiers

**Laptop tier (fast).** The *distil* step — turning a raw run artifact into
`figdata/*.json` — is seconds per figure and needs only that artifact. Only fig07
touches the ROM at this tier, evaluating a *shipped surrogate model artifact* (the
χ Smolyak/Hermite/POD models) to precompute its smooth curves. The per-figure
producers of the raw artifacts live in
`src/lemaitre/initial_data/conformally_flat_puncture/pipeline/` and are mapped to
figures by `paper/figures/registry.py` (the single source of truth for the
figure→producer→artifact graph). Those producers are the heavy tier below.

**Heavy tier (cluster / oracle).**
- **χ surrogate corpora** — built by `pipeline/{build_surrogate_chi, run_8d_chi_array,
  build_pod_hermite_model_chi, build_pod_hermite_model_chi_8d,
  build_pod_hermite_chi8d_array, build_cross_model_chi}`. Multi-GB; produced on the
  cluster. Not committed (`reports/` is gitignored). Point the figure scripts at
  the built artifacts (see `registry.py` / Stage-2 wiring).
- **TwoPunctures validation** (the fig08/fig09 sweep) — `run_tp_random_sweep.py --n 100
  --workers 6`, and the external oracle binary (`make oracle`; the build script is
  bundled). One oracle call dominates each configuration
  (~2–8 min, markedly slower for spinning ones), so budget hours of wall-clock even in
  parallel; every rung of `--ladder` shares that one call, which is what makes a whole
  resolution ladder per configuration affordable.

  Each curve in fig08 is a **min/median/max band over configurations** sampled from the
  production box, so no panel depends on an arbitrary parameter point. This one source
  replaced two earlier figures: the former fig08 (a separate non-axisymmetric validation
  at `b=1.5`, head-on — below the production `B_MIN`) and the former
  single-configuration fig09.
  The separate non-axisymmetric figure was redundant because **the quasi-circular data
  are already non-axisymmetric**: measured, the tangential momentum puts ~2% of the field
  in `m=2` and generic spins add ~2% at `m=1`, so the QC family exercises the
  Fourier-in-φ solver by itself. The spectrum is now a figure of its own, fig09; the one
  check QC cannot supply (a head-on slice with spin along the collision axis keeps every
  `m≥1` mode below 1e-16 — `axisym_m_ge1_max` is 9.2e-17 in fig08's committed figdata) is
  computed by the same producer's `axisym` block and quoted as text.

  See that producer's docstring for the two findings needed to read the figure: spin
  combined with `q` drives the disagreement and it converges (~100× worse from `q=1` to
  `q=3` at fixed spin, a further ~10× from `b=3` to `b=10`); and the certified residual
  *rises* with resolution — mildly along a fixed-`Nφ` ladder, steeply when `Nφ` rises,
  because the mechanism is roundoff in unpopulated high-`m` modes, not lost convergence.

- **Constraints on an evolution grid** (fig10) — measured by **GRTeclyn**, an external
  numerical-relativity code, not by the in-house monitor (which stays as an internal
  check: `validation/constraints.py` + its tests). Three stages:

  1. `python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.run_export_grteclyn --out reports/grteclyn_export
     --tp` (~3 min; needs a machine with AVX, so on a cluster submit it as a batch
     job) — solves each
     configuration and writes `<name>.lmid` plus a `<name>_reference.dat` table of `psi`
     and `Ahat` values that the C++ evaluator is validated against. With `--tp` it also
     writes the TwoPunctures conformal factor **on the identical spectral grid**, so the
     cross-code comparison shares the interpolation operator too.
  2. the GRTeclyn runs (~30 min for the six-series ladder, plus the
     refined-hierarchy AMR runs). Each rung leaves one `constraint_norms.json`;
     the rungs of one series are collected into `<tag>/ladder.json` by

     ```bash
     LM_GRTECLYN_EXE=/path/to/BinaryBH3d.gnu.MPI.OMP.ex \
       ./grteclyn/run_ladder.sh lm_anchor "48 64 96 128 192" \
           LMID=<export>/qc_b3.lmid LMREF=<export>/qc_b3_reference.dat
     ```

     See [`../grteclyn/README.md`](../grteclyn/README.md) for the series, the
     configuration defaults and where each of their numbers is pinned. The
     script is account- and cluster-agnostic on purpose: it takes the executable
     and the launcher from the environment and hard-codes no paths. GRTeclyn
     itself is **not** vendored here — you need a build of our fork's
     `lm-initial-data-constraints` branch, which carries the `lm_id_file`
     runtime switch.

     > **Provenance of the committed numbers: they are format-1 measurements, and
     > that is deliberate.** The exporter now writes **format 2**, which divides
     > the axis factor `w_k(B) = (1-B^2)^{k/2}` out of the wavenumber-`k`
     > coefficients so that polynomial interpolation on the consumer side is
     > exact for every `k` rather than only for `k = 0`. The consumer learned to
     > restore it on 2026-08-19 (`510dd01` on `lm-initial-data-constraints`); the
     > ladder behind fig10 was run against the earlier format-1 consumer
     > (`94323cd`, 2026-08-06). So a re-run today would exercise a *different*,
     > strictly better consumer-side interpolation, and the committed figure is
     > the conservative measurement rather than the best one available. Re-running
     > it was weighed and declined for this paper. **The two halves cannot be
     > mixed:** `LMSpectralData` aborts unless the file says `format 2`, and
     > `Export.read` refuses the mismatch from this side, because reading a
     > newer file under an older rule is silently wrong rather than an error.
  3. `python paper/figures/fig10_constraints_data.py --runs <that tree>` (seconds, reads
     files only) then the plotter. Set `$LM_GRTECLYN_RUNS` instead of `--runs` if you
     prefer.

  The oracle cost collapses in this arrangement: it is queried at ~15 000 **spectral
  nodes** (~26 s) instead of the ~26 million Cartesian points the in-house comparison
  needed (~13 h serial). The cluster sweep that drove the old arrangement is retired
  along with it.

## What is committed vs regenerated

- **Committed:** the figure **PDFs** (so `pdflatex paper/paper.tex` works out of
  the box), the source scripts, and **`paper/figures/figdata/*.json`**.
- **Why figdata is committed:** the plotters read `figdata/figNN_*.json` and
  nothing else — no `reports/`, no model corpus, no jax. Committing it (~110 kB
  for all ten) is what lets anyone **replot from a bare clone**, with only
  matplotlib and no copy step:

  ```bash
  cd paper/figures && for f in fig??_*_plot.py; do python "$f"; done
  ```

- **Regenerated:** `make figdata` re-distils that json from the raw sources, and
  needs the heavy tier (`$LM_REPORTS`, the multi-GB corpora, the cluster). That is
  the only step a clone cannot do — and note it **skips every figdata already on
  disk**, so on a clone (where all ten are committed) it does nothing at all.
  `python paper/figures/make_figdata.py --all --force` is what actually rebuilds.

## Model-artifact location convention

**One setting: `$LM_REPORTS`.** Every producer in `src/lemaitre/initial_data/conformally_flat_puncture/pipeline/`
and `paper/figures/_figdata` resolves the heavy tree through
`lemaitre.initial_data.conformally_flat_puncture.paths.reports_root()`:

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
non-obvious ones.

The rank and artifact strings below are the **rendered** form of the registry's
structured producers, which take the rank and the POD filename from
`production_model.SHIPPED_RANK` / `production_model.pod_stem(dim)` — their single
source. If a shipped rank moves, re-render rather than hand-editing this table:

```bash
cd paper/figures && python -c "import registry as R; print(R.producer_cmd('polish_pod_8d'))"
```

| source | produced by | note |
|---|---|---|
| `tp_band_sweep` | `run_tp_random_sweep --n 100 --workers 6` | the single validation source. Supersedes `sweep_3d` (`run_3d_sweep`) and `tp_validation` (`run_qc_tp_validation`), which fed the former fig08 and the former single-configuration fig09; both are now unreferenced by any figure, and by any paper numeral. `run_3d_sweep` remains useful standalone for its ADM-`J` diagnostics, which the manuscript does not quote. |
| `polish_pod_4d` | `run_polish_table --model pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r250.npz --tag chi4d_pod_r250_cross` | the shipped rank-250 **cross** POD. `run_polish_podrank` hardcodes the *non-cross* POD per dimension, so it cannot emit the `_cross` name. |
| `polish_table_4d_cross` | `run_polish_table --model hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz --tag qc_chi_prod_cross` | the *untruncated* cross corpus — corroborated by the `8·N·(1+d+npair)·nfeat` memory fig05 applies to it. |
| `polish_table_8d_value` | `run_polish_table --model surrogate_smolyak_spin8_qc_chi_prod_L5.npz --tag chi8d_value` | the 8-D **value** surrogate — corroborated by fig05's `8·N·nfeat` value-only memory. |
| `polish_pod_8d` | `run_polish_table --model pod_hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross_r500.npz --tag chi8d_pod_r500_cross` | the shipped rank-500 **cross** POD — the same model family as the 4-D row. It was `run_polish_podrank --dim 8 --rank 250`, i.e. the six-axis *non-cross* POD, which made this column's residual row and its field-error row two different models, masked by a shared `r=250`. |
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

## Producers that gate a decision rather than feed a figure

Not every producer under `pipeline/` writes figure data. One gates a **solver
default**, and it is the one to re-run after any change to the preconditioner or
the linear algebra underneath it.

| producer | what it gates | pass criteria |
|---|---|---|
| `run_separable_sweep --n 240` | the **separable preconditioner being the default** (`solver_3d.assemble(separable=)` via `choose_separable`). Solves each sampled point twice from the *same* warm start — separable vs dense — through the shipped forward map `parametric_nd_3d.make_solve_fn`, so it measures production rather than a reimplementation. | three, all in the module docstring and enforced in code: (1) separable certifies **wherever dense does** (a point failing on *both* is a solver limitation, reported separately and not charged to this gate); (2) the two converged fields agree to `FIELD_TOL = 1e-9` relative; (3) extra Newton steps are at most **one** per point at no more than `MAX_EXTRA_FRAC = 0.05` of points. Exit status is 0 only on a clean pass, so it can gate a cluster job. |

The gate on extra Newton steps is a **rate, not an absolute**. It read "zero
extra steps" until 2026-08-15, a criterion inherited from a 24-point prototype
sweep; at 240 points three of them (1.25 %) cost one step, every one at or above
the 90th percentile of total spin `|χ_A|+|χ_B|` — exactly where dropping the
nonlinear diagonal from the preconditioner should first show. Do not re-tighten
it to zero without re-reading that reasoning.

The certification threshold is imported from
`parametric.certification.CERT_TOL_U`, never restated — see the single-source rule in
`CLAUDE.md`. Since 2026-08-25 that is the **`u`**-norm gate `1e-11`; the `v`-norm
`CERT_TOL = 1e-10` this line used to name is kept beside it as the submitted paper's
number and is no longer any default.

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
python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.build_surrogate \
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

Panels (a) and (b) of fig09 — the quasi-circular ψ and `M_ADM` comparisons *against*
TwoPunctures — require the binary, and those are irreducibly a comparison. Panel (c),
the azimuthal spectrum, and the `axisym` block do not (they are internal properties of
our own solve), but they ride along in the same producer because the expensive part is
one oracle call per configuration.

Two diagnostics that no longer have a panel of their own:

- the **axisymmetric-limit code-to-code anchor** at `b=3`, `P=0.5` head-on, quoted in the
  appendix text — the most stringent TwoPunctures number in the paper, and not obtainable
  from a quasi-circular configuration, which is never axisymmetric. It is measured by this
  same producer and read out of fig08's committed figdata meta (`anchor`): ψ to 2.5e-10 in
  the supremum norm, `M_ADM` to 1.2e-11 relative, certified residual 9.3e-15. The older
  4.7e-12 / 1.0e-11 pair came from the superseded `tp_validation` source and no longer
  appears anywhere in the manuscript.
- the **ADM-`J` tilt against the spin tilt** (measured, θ_J tracked θ_S to ~1e-14 deg for
  every |S| and every TP anchor, so the panel was three coincident curves on the line
  y=x). This one is *not* quoted in the manuscript at all; it survives only as a
  `run_3d_sweep` diagnostic.

### Where the source comes from, and how the binary is built

`make oracle` runs `oracle/build.sh`, which is **in this repo**. What is *not* in
this repo is the oracle's own source: the build generates it, so "agrees with
TwoPunctures" still means agreement with upstream code we did not write or edit.

```bash
make oracle                                   # -> ~/.cache/lemaitre/tp-oracle/tp_solve
LM_TP_PREFIX=/elsewhere ./oracle/build.sh     # install somewhere else
```

Provenance of everything the build compiles:

| layer | origin |
|---|---|
| physics | Einstein Toolkit thorn **`TwoPunctures`** (M. Ansorg, E. Schnetter, F. Löffler) — the single-domain spectral puncture solver of Ansorg, Brügmann & Tichy, *PRD* **70**, 064011 (2004), arXiv:gr-qc/0404056. Upstream `https://bitbucket.org/einsteintoolkit/einsteininitialdata`. **LGPL v2.0+.** |
| C port | Z. B. Etienne's Cactus-free port, shipped inside the `nrpy` package as `nrpy/infrastructures/BHaH/general_relativity/TwoPunctures/` (`https://github.com/nrpy/nrpy`, PyPI `nrpy`). The build pins **`nrpy==2.2026.6`**, the version the committed oracle numbers were produced with. |
| numerics | GSL (BiCGStab + linear algebra), taken from the build environment. The binary here was built against **GSL 2.8** and links it dynamically via `-rpath`, so it is tied to the environment that built it — rebuild rather than copying it between machines. |

`oracle/gen_tp.py` writes the six TwoPunctures translation units and their two
headers to `oracle/src/` **verbatim** from `nrpy` (no upstream C is edited or
retyped), plus a minimal `BHaH_defines.h` — the small subset of BH@H's generated
header the solver uses (`REAL`, `derivs`, `ID_persist_struct`), transcribed from
nrpy's own `ID_persist_str()`. Those generated files are **gitignored**: they are
third-party code and are produced at build time rather than vendored.

The one hand-written file, and the only one committed under `oracle/src/`, is
`main.c` — argv/stdin/stdout glue that fills `ID_persist_struct`, calls the
unmodified `TP_solve()`, and evaluates the result with the unmodified
`PunctIntPolAtArbitPosition()`. BH@H's `TP_Interp()` is not built; it exists only
to fill a BH@H grid.

The build passes no OpenMP flag, so the oracle is **serial and deterministic
across invocations** — which is what
`tests/test_validation_spin.py::test_spin_axisymmetry_nphi` relies on when it
diffs ψ across two separate runs at `1e-10`. Upstream parallelises the BiCGStab
line-relaxation preconditioner; building that would make the Krylov path
schedule-dependent, so a paper oracle should stay serial.

Set `LM_TP_BIN` to point at a binary somewhere else; otherwise
`~/.cache/lemaitre/tp-oracle/tp_solve` — where `oracle/build.sh` installs — is the
default `validation/twopunctures.py` looks for. If it is absent,
`validation.twopunctures.available()` returns `False` and the 29 oracle-dependent
tests skip rather than fail.

> Status: the recompute wiring (rewriting `figNN_*_data.py` to compute from the
> solver/ROM instead of reading `reports/*.json`) is **Stage 2** — not yet done.
> Today the data scripts still carry the old cache-read logic.
