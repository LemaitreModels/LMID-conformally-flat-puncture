# Paper tables — two-tier, recompute-by-default

Same pattern as `paper/figures/`, one tier shorter because both tables recompute
from the solver on a laptop in seconds (no `reports/` corpus, no cluster):

```
tabNN_<name>_data.py   recomputes from the solver  ->  tabdata/tabNN_<name>.json   (gitignored)
tabNN_<name>_tex.py    reads ONLY that json        ->  tabNN_<name>.tex            (committed)
```

`paper.tex` keeps `\begin{table}`, `\caption`, and `\label`, and `\input`s the
generated `ruledtabular` body — so no numeric digit is hand-transcribed into the
manuscript.

| stem | paper | what it verifies | cost |
|---|---|---|---|
| `tab01_tangent_operator`  | Table I  | Eq. (tangent) vs. central FD of the certified 3-D Newton–Krylov solve | ~1 s |
| `tab02_tangent_surrogate` | Table II | the exposed surrogate gradient vs. FD of the surrogate and vs. Eq. (tangent) | ~15 s |

## Entry points

```bash
make tabdata                                   # recompute both tabdata/*.json
make tables                                    # tabdata, then re-render both .tex
python paper/tables/make_tabdata.py --check     # presence matrix
python paper/tables/make_tabdata.py --tab tab01 --force
```

The physics lives in one canonical producer,
`src/lm/initial_data/pipeline/run_tangent_verification.py`
(`operator_tangents()`, `surrogate_tangents()`), which the data scripts call and
which is importable and runnable on its own:

```bash
python -m lm.initial_data.pipeline.run_tangent_verification --out /tmp/tangent.json
```

## Guard

`tests/test_paper_tables.py` (fast tier, file reads only) asserts that the rendered
`.tex` is not stale relative to the json, that `paper.tex` `\input`s the generated
body rather than hand-written rows, and that the **captions** — author-owned prose
that repeats the grid, slice, interpolation orders, and certified residual by hand —
still agree with the recomputed configuration.
