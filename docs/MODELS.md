# Which model is shipped

**Authoritative source: [`src/lm/initial_data/pipeline/production_model.py`](../src/lm/initial_data/pipeline/production_model.py).**
It is executable and tested (`tests/test_production_model.py`); this page is the
narrative. Every table below is printed by

```bash
python -m lm.initial_data.pipeline.production_model
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
axes*. The 8-D model does **not** enhance its other four spin components: §2.4 of
`HISTORY_AND_FINDINGS.md` is the measurement behind that ("enhance only a small
axis set"), and the paper's `sec:model:enhanced` reports it.

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

Artifact names carry both the rank and the layout, e.g.

```
pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By_cross_r250_slim.npz
```

so a reader can tell which model, which rank and which accounting a file obeys from
its name alone (`production_model.pod_stem`).

---

## 3. Where each figure's model comes from

Figures record provenance in their figdata `meta` block (model stem, enhanced axes,
rank, blocks/node, memory), and `registry.FIGURES[stem]["keys"]` declares it, so a
figdata built before a model change reads as **stale** instead of silently feeding
the paper old numbers.

| figure | model plotted |
|---|---|
| `fig03_joint_dist` | value-only vs shipped cross |
| `fig04_polish_staircase` | cold / value-only POD / shipped cross POD, at the shipped ranks |
| `fig05_guess_vs_memory` | value-only vs shipped cross, POD rank ladder |

`fig05_guess_vs_memory_data.py` **recomputes** every byte count from
`production_model` rather than trusting the sweeps' stored `mem_bytes`. That is what
lets a memory-accounting fix be a re-distill (`make figdata`) instead of a re-sweep:
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
3. Rebuild the affected figdata (`make figdata`; most sources are cluster-side, see
   [`DATA.md`](DATA.md)) — the provenance guard will fail until you do.
4. Update this page by re-running the self-report.
5. Only then touch `paper/paper.tex`.
