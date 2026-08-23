# Which model is shipped

**Authoritative source: [`src/lemaitre/initial_data/conformally_flat_puncture/pipeline/production_model.py`](../src/lemaitre/initial_data/conformally_flat_puncture/pipeline/production_model.py).**
It is executable and tested (`tests/test_production_model.py`); this page is the
narrative. Every table below is printed by

```bash
python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.production_model
```

so regenerate it rather than hand-editing. If a number about "the model" appears in
a producer, a figure, a test or the paper and does **not** come from that module,
that is a bug.

---

## 1. The shipped model, in one line

> Gradient-enhanced sparse (Hermite–Smolyak) collocation on the production box,
> **enhanced on exactly two axes — the aligned spin components `chi_Ay`, `chi_By` —
> plus their one bilinear cross term**, reduced-basis (POD) compressed to
> **r = 250 (4-D) / r = 500 (8-D)**, stored in the **slim** layout.

Both the 4-D aligned-spin and the 8-D general-spin model enhance the *same two
axes*. The 8-D model does **not** enhance its other four spin components —
enhancing a larger axis set measurably degrades held-out accuracy — and the
paper's `sec:model:enhanced` reports the measurement.

| | 4-D aligned | 8-D general-spin |
|---|---|---|
| box | `(b, q, chi_Ay, chi_By)` | `(b, q, chi_A{x,y,z}, chi_B{x,y,z})` |
| enhanced axis **names** | `chi_Ay, chi_By` | `chi_Ay, chi_By` |
| enhanced axis **indices** | `(2, 3)` | `(3, 6)` |
| cross pairs | 1 | 1 |
| Smolyak level | 5 | 5 |
| solver nodes | 1105 | 15713 |
| shipped POD rank | **250** | **500** |

The names are the same in both boxes and the indices are **not** — the six spin
components sit between `q` and the aligned pair in 8-D. Use
`production_model.enhanced_indices(dim)`, never a literal.

---

## 2. What a node stores, and what that costs

The interpolant reads `dU_nodes` **only at the enhanced axes** — both
`HermiteCrossSolutionND.evaluate` and its jax twin do
`{e: take(dU_nodes, e) for e in enhanced}`. So the shipped model stores

```
1 value  +  n_enh (=2) tangents  +  n_pairs (=1) cross  =  4 fields per node
```

independently of the dimension `d`. Anything keyed on `1 + d` or `1 + d + npair`
is counting tangents the model never reads.

| family | blocks/node | 4-D bare | 4-D POD | 8-D bare | 8-D POD |
|---|---|---|---|---|---|
| `value` | 1 | 97 MiB | 24.2 MiB | 1,381 MiB | 104.0 MiB |
| `gradient_all` | 5 / 9 | 486 MiB | 32.6 MiB | 12,429 MiB | 583.5 MiB |
| `cross_all` | 6 / 10 | 583 MiB | 34.7 MiB | 13,810 MiB | 643.4 MiB |
| `shipped` **(shipped)** | 4 | 388 MiB | 30.5 MiB | 5,524 MiB | 283.8 MiB |

POD columns are at the shipped ranks. Compression: **12.7×** (4-D), **19.5×** (8-D).

The non-shipped families are kept *named* so that every number the project has ever
quoted stays reproducible and labelled — see §4.

### The slim layout

`PODHermiteSmolyakCross.save(..., slim=True)` (the default) writes the tangent block
only for the enhanced axes. This is **lossless**, not an approximation: the dropped
blocks are the ones `evaluate` never reads, and the loader restores them as zeros, so
a slim round-trip evaluates **bit-for-bit** like a full one
(`tests/test_production_model.py::test_slim_roundtrip_is_lossless`). Files record
their layout in `meta['dU_layout']`; pre-existing full-`d` artifacts still load.

Artifact names carry the model and the rank, and deliberately **not** the layout:

```
pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r250.npz
```

`production_model.pod_stem` builds that stem, and its docstring gives the reason the
layout is left out: the layout is recorded *inside* the file as `meta['dU_layout']`
and both layouts load, so a name cannot go stale against its contents. Read the
layout from the file, never from the filename.

---

## 3. Where each figure's model comes from

Provenance is recorded in each figure's figdata `meta` block, and unevenly:
`fig05_guess_vs_memory` carries the full set (`model`, `enhanced_axes`,
`shipped_rank`, `blocks_per_node`, `bare_mib`, `shipped_pod_mib`, `compression`),
`fig04_polish_staircase` carries `shipped_rank`, `smolyak_level` and a `code_tag`,
and `fig03_joint_dist` carries per-source box, grid and `code_tag` but **no model
stem at all**.

**What that does and does not catch.** `registry.FIGURES[stem]["keys"]` pins only the
*presence of top-level keys*: `make_figdata._figdata_tag` calls a figdata STALE when
it "lacks a top-level key its plotter reads", and nothing more. A figdata whose `meta`
names an older model therefore reads as `figdata OK`, and `make figdata` skips it. The
guards that do fire are producer-side, and only once the producer is re-run —
`fig04`'s rank guard and `fig05`'s blocks/node guard each `raise SystemExit` on a
sweep that is a different model than the panel claims, and `fig07` refuses a model
built on a different grid. In the suite,
`test_paper_figures.py::test_producer_module_exists_and_ranks_match_the_shipped_model`
pins the *registry's* rank to `SHIPPED_RANK`. **Nothing compares a committed figdata's
`meta` to `production_model`,** so a figdata can name a different model from the prose
that cites it with the suite green.

| figure | model plotted |
|---|---|
| `fig03_joint_dist` | value-only vs cross-enhanced, both **bare** — its sources (`run_qc_joint_dist_chi`, `run_qc_joint_dist_cross_chi`) pass no rank, and the json's two series are keyed `bare`/`cross` |
| `fig04_polish_staircase` | cold / value-only POD / shipped cross POD, at the shipped ranks |
| `fig05_guess_vs_memory` | value-only vs shipped cross, POD rank ladder |

`fig05_guess_vs_memory_data.py` **recomputes** every byte count from
`production_model` rather than trusting the sweeps' stored `mem_bytes`. That is what
lets a memory-accounting fix be a re-distill (`make_figdata.py --fig
fig05_guess_vs_memory --force`) instead of a re-sweep:
the accuracy statistics were never affected by it.

---

## 4. Superseded numbers (so they are never mistaken for current)

| number | what it actually was |
|---|---|
| 486 MiB / 12.1 GiB corpus | `gradient_all` — all-`d` tangents, **no** cross |
| 583 MiB / 13.5 GiB bare star | `cross_all` — all-`d` tangents **+** cross |
| 10.0 MiB / 420 MiB compressed | `gradient_all` at the **old** ranks r=76 / r=359 |
| compression 49× / 30× | the two rows above, combined |
| POD rank 75/76, 359 | the first shipped bases, below the Fig. 5 saturation knee |

The old bases held only 76 and 359 modes and the loaders truncate *downward only*,
so requesting a higher rank silently clamped; raising to 250/500 required rebuilding
the cross POD bases from the full Hermite corpora.

---

## 5. If you change the model

1. Change `production_model.py` — nothing else defines the model.
2. `pytest tests/test_production_model.py` (identity, accounting, slim losslessness).
3. Rebuild the affected figdata — `python paper/figures/make_figdata.py --fig <stem>
   --force`; most sources are cluster-side, see [`DATA.md`](DATA.md). **Do not wait for
   a guard to tell you.** Plain `make figdata` skips a figdata that is present, and its
   staleness check is key-presence only (§3), so a model change leaves the old numbers
   reading as `figdata OK`. `fig04` and `fig05` refuse a mismatched sweep, but only
   once their producer actually runs.
4. Update this page by re-running the self-report.
5. Only then touch `paper/paper.tex`.
