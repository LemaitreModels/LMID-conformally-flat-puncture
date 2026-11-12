# Reproduce the paper

This page is for rebuilding the paper's models, data and figures. To *use* a shipped
model, see [Available models](MODELS.md) and [Choose a workflow](workflows.md).

The paper's figures are made in two steps, and a clone can do the second on its own:

1. **Distil.** Each `paper/figures/figNN_*_data.py` reads the raw output of a heavy
   run (a model build, a validation sweep, an evolution) and writes the numbers its
   figure needs to `paper/figures/figdata/figNN_*.json`. These scripts reshape
   existing run output; none of them reruns the solver. The figdata files are
   committed.
2. **Plot.** Each `figNN_*_plot.py` draws its PDF from that json and nothing else.

## Entry points

```bash
make figures     # plot every figure from the committed figdata (only matplotlib needed)
make figdata     # distil any MISSING figdata (--force to redo present ones; needs the heavy runs)
make tables      # recompute and render the table (no solve)
make test        # the full test suite, tens of minutes
make models      # build the model corpora (cluster)
make oracle      # build the TwoPunctures validation binary
```

`make figdata` skips every figdata file that already exists, and all ten are
committed, so on a clone it does nothing. `python paper/figures/make_figdata.py --all
--force` re-distils everything, and then needs the heavy runs below.

## What a clone can rebuild

- **Every figure PDF**, from the committed figdata:

  ```bash
  cd paper/figures && for f in fig??_*_plot.py; do python "$f"; done
  ```

- **The table**, `tab01_production_box`, from scratch. It derives every box edge,
  the Smolyak level, the spatial grid and the enhanced axes from
  `pipeline/production_box.py`, and the sparse-grid node count from
  `parametric_nd_2c.smolyak_points`, with no corpus and no solve. `paper.tex` inputs
  the rendered `paper/tables/tab01_*.tex`; see `paper/tables/README.md`.

## The heavy runs

Everything else needs runs measured in cluster hours. Their outputs live under one
directory, set by `$LM_REPORTS` (`paths.reports_root()`), separated by
discretisation: artifacts of the production scheme `"v2"` are under
`$LM_REPORTS/v2/`, those of the first submitted paper's `"paper-v1"` grid directly
under `$LM_REPORTS`. Every committed figdata records its `meta.scheme`, and
`make_figdata.py --check` reports one that is not on the current scheme as STALE.

`paper/figures/registry.py` maps each figure to its sources and each source to the
producer command that makes it:

```bash
cd paper/figures && python -c "import registry as R; print(R.producer_cmd('polish_pod_8d'))"
```

### Model corpora

Built by `pipeline/{build_surrogate_chi, run_8d_chi_array, build_pod_hermite_model_chi,
build_pod_hermite_model_chi_8d, build_pod_hermite_chi8d_array, build_cross_model_chi}`
on a cluster. They are multi-GB and not committed. fig07 is the one figure-data script
that evaluates a model (to draw its smooth curves); the others read sweep outputs.

### The TwoPunctures comparison (fig08, fig09)

`run_tp_random_sweep --n 100 --workers 6`, with the oracle binary from `make oracle`.
Each configuration costs one oracle call (2–8 minutes, longer with spins), shared by
every rung of its resolution ladder, so budget hours of wall-clock even in parallel.

- **fig08** compares ψ and $M_{\rm ADM}$ against TwoPunctures, as min/median/max
  bands over configurations sampled from the production box.
- **fig09** is the azimuthal spectrum of the same solves. The quasi-circular data are
  not axisymmetric: the tangential momentum puts about 2 % of the field in $m = 2$
  and generic spins about 2 % in $m = 1$, so this family exercises the Fourier
  directions of the solver.
- The same producer measures the **axisymmetric anchor** the appendix quotes, a
  head-on slice at $b = 3$, $P = 0.5$. On the `"v2"` grid (top rung 64 × 56 × 8) ψ
  agrees to 4.9e-14 in the supremum norm, $M_{\rm ADM}$ to 1.2e-12 relative, with
  certified residual 2.7e-16 in the `u` norm. It is stored in fig08's figdata
  (`meta.anchor`). A head-on slice with spin along the axis keeps every $m \ge 1$
  mode below 1e-16 (`axisym_m_ge1_max`).

### Constraints on an evolution grid (fig10)

The constraint violation is measured by **GRTeclyn**, an independent evolution code,
in three stages:

1. Export. `python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.run_export_grteclyn --tp`
   (a few minutes; needs AVX, so on a cluster submit it as a job) solves each
   configuration and writes `<name>.lmid`, plus a `<name>_reference.dat` table of ψ
   and $\hat A_{ij}$ values against which the C++ reader is checked. With `--tp` it
   also writes the TwoPunctures conformal factor on the same spectral grid, so the
   two codes share the interpolation.
2. Evolve. Each rung of a resolution ladder leaves one `constraint_norms.json`, and
   `grteclyn/run_ladder.sh` collects a series into `<tag>/ladder.json`:

   ```bash
   LM_GRTECLYN_EXE=/path/to/BinaryBH3d.gnu.MPI.OMP.ex \
     ./grteclyn/run_ladder.sh lm_anchor "48 64 96 128 192" \
         LMID=<export>/qc_b3.lmid LMREF=<export>/qc_b3_reference.dat
   ```

   GRTeclyn is not vendored: it needs a build of the fork's
   `lm-initial-data-constraints` branch at commit `087f827` or later.
   [`grteclyn/README.md`](https://github.com/LemaitreModels/LMID-conformally-flat-puncture/blob/main/grteclyn/README.md)
   lists the series, their parameters and the keys an older build would silently
   ignore.
3. Distil. `python paper/figures/fig10_constraints_data.py --runs <that tree>` (or set
   `$LM_GRTECLYN_RUNS`), then the plotter.

The export format follows the grid: `"paper-v1"` is written as format 2 (a polynomial
in $B$ on Lobatto $A$) and `"v2"` as format 4 (angular nodes in the ABT angle $s$,
Chebyshev–Gauss $A$, $u = (A - 1) V$). The GRTeclyn reader accepts 2 and 4, each with
its own rule, and refuses anything else, and `Export.read` refuses a mismatch from this
side, because reading a file under the wrong rule is silently wrong.

The committed fig10 is a `"paper-v1"` measurement, exported at 52 × 36 and read with
the earlier format-1 reader, which interpolates the angular factor less accurately
than format 2; it is therefore the conservative measurement. A `"v2"` fig10 needs new
ladder runs from format-4 exports on the production grid.

## Residual norms in the committed figdata

The certification gate is the `u`-norm `CERT_TOL_U = 1e-11`. Figdata produced while it
was the `v`-norm `CERT_TOL = 1e-10` carries `v`-norm residuals: `fig04_polish_staircase`
and `fig08_tp_validation` say so in `meta` (`certify_tol`, `cert_tol` = 1e-10), and the
`worst_certified_residual` values in `fig06_targeting` (9.88e-11, 5.64e-11, 3.64e-11)
are `v`-norm too. Do not read them against the `u` gate; no single factor converts
between the norms. New producers should stamp the norm of every residual they write.

## Which source comes from which producer

The rendered form of the registry's producers for the sources whose choice is not
obvious. The ranks and filenames come from `production_model.SHIPPED_RANK` and
`production_model.pod_stem(dim)`; re-render with `registry.producer_cmd` rather than
editing this table.

| source | produced by | note |
|---|---|---|
| `tp_band_sweep` | `run_tp_random_sweep --n 100 --workers 6` | the one validation source (fig08, fig09, the anchor). |
| `polish_pod_4d` | `run_polish_table --model pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r250.npz --tag chi4d_pod_r250_cross` | the shipped rank-250 cross POD model. |
| `polish_table_4d_cross` | `run_polish_table --model hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross.npz --tag qc_chi_prod_cross` | the uncompressed cross model. |
| `polish_table_8d_value` | `run_polish_table --model surrogate_smolyak_spin8_qc_chi_prod_L5.npz --tag chi8d_value` | the 8-D value-only model. |
| `polish_pod_8d` | `run_polish_table --model pod_hermite_smolyak_spin8qc_L5_enh-chi_Ay-chi_By_cross_r500.npz --tag chi8d_pod_r500_cross` | the shipped rank-500 cross POD model, the same family as the 4-D row. |
| `gvm_4d_value` | `run_value_pod_gapfill_4d` | the value-only POD model, the 4-D counterpart of the 8-D producer. |
| `gvm_4d_field` | `run_cross_fielderr_sweep --flavours value` | measured in the cross sweep, so both fig05 curves use the same certified reference fields. |
| `qc_targeting` | `run_qc_targeting --n 100` | |

## A producer that gates a solver default

`run_separable_sweep --n 240` checks that the separable preconditioner can stay the
default (`solver_3d.assemble(separable=)`, chosen by `choose_separable`). It solves each
sampled point twice from the same warm start, separable and dense, through the
production forward map `parametric_nd_3d.make_solve_fn`, and passes when:

1. separable certifies wherever dense does (a point that fails both is reported
   separately);
2. the two fields agree to `FIELD_TOL = 1e-9` relative;
3. separable needs at most one extra Newton step at a point, at no more than
   `MAX_EXTRA_FRAC = 0.05` of the points.

The last criterion is a rate because dropping the nonlinear term from the
preconditioner does cost a step at the largest spins: at 240 points, three (1.25 %)
needed one, all above the 90th percentile of $|\chi_A| + |\chi_B|$. Exit status 0 means
a clean pass, so it can gate a cluster job. Re-run it after any change to the
preconditioner or the linear algebra under it.

## The eccentricity model (fig07)

`run_qc_effpot` needs a dense 2-D model in $(b, P_t)$, the one family with a free
momentum; every other family takes the quasi-circular momenta. Its box is derived:
$b$ spans the production range `production_box.B_MIN..B_MAX`, and $P_t = J / (2b)$
covers the study's $J \in [1.00, 1.10]$, so
$P_x \in [J_{\min} / (2 B_{\max}), J_{\max} / (2 B_{\min})]$ (with punctures at
$z = \pm b$ the orbital angular momentum is $J = 2 b P_x$, so $P_t \equiv P_x$). The
radial momentum is fixed to zero, as an apsis requires. The consumer loads a *dense*
model under a fixed name, so build it with `--dense-Q` and `--dense-name`:

```bash
python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.build_surrogate \
    --box bpt_ecc --level 5 --dense-Q 16 --dense-name surrogate_bpt_ecc.npz \
    --solver nk --store --code-tag chi-rebuild
```

`Q = 16` gives 17 nodes per axis, on the nested Chebyshev–Lobatto ladder (1, 3, 5, 9,
17, 33), so the sparse build's solves are reused from the solve store. The grid is the
builder's default, `production_box.PROD_GRID`.

Before using the model, check that the circular orbit, the interior minimum of
$\partial E_b / \partial b|_J$, lies strictly inside the box for every $J$. On the
lower edge it would be an artifact of the box, not a measurement.

## The TwoPunctures oracle

`make oracle` runs `oracle/build.sh`, which is in this repository. The oracle's own
source is not: the build generates it, so "agrees with TwoPunctures" means agreement
with upstream code we did not write or edit.

```bash
make oracle                                   # -> ~/.cache/lemaitre/tp-oracle/tp_solve
LM_TP_PREFIX=/elsewhere ./oracle/build.sh     # install somewhere else
```

| layer | origin |
|---|---|
| physics | Einstein Toolkit thorn **`TwoPunctures`** (M. Ansorg, E. Schnetter, F. Löffler) — the single-domain spectral puncture solver of Ansorg, Brügmann & Tichy, *PRD* **70**, 064011 (2004), arXiv:gr-qc/0404056. Upstream `https://bitbucket.org/einsteintoolkit/einsteininitialdata`. **LGPL v2.0+.** |
| C port | Z. B. Etienne's Cactus-free port, shipped inside the `nrpy` package as `nrpy/infrastructures/BHaH/general_relativity/TwoPunctures/` (`https://github.com/nrpy/nrpy`, PyPI `nrpy`). The build pins **`nrpy==2.2026.6`**, the version the committed oracle numbers were produced with. |
| numerics | GSL (BiCGStab and linear algebra), from the build environment, linked dynamically, so rebuild rather than copy the binary between machines. |

`oracle/gen_tp.py` writes the six TwoPunctures translation units and their two headers
to `oracle/src/` verbatim from `nrpy`, plus a minimal `BHaH_defines.h` with the few
definitions the solver uses. These generated files are gitignored. The one hand-written
file, `oracle/src/main.c`, reads the configuration, calls the unmodified `TP_solve()`
and evaluates the result with the unmodified `PunctIntPolAtArbitPosition()`.

The build passes no OpenMP flag, so the oracle is serial and deterministic across runs,
which `tests/test_validation_spin.py::test_spin_axisymmetry_nphi` relies on when it
compares ψ across two runs at 1e-10.

Set `LM_TP_BIN` to use a binary elsewhere; otherwise `validation/twopunctures.py` looks
in `~/.cache/lemaitre/tp-oracle/tp_solve`. Without it,
`validation.twopunctures.available()` returns `False` and the oracle-dependent tests
skip rather than fail.
