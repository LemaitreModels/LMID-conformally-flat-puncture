# Paper tables — two-tier, recompute-by-default

Same pattern as `paper/figures/`, one tier shorter because every table recomputes
on a laptop in seconds (no `reports/` corpus, no cluster):

```
tabNN_<name>_data.py   recomputes from the solver  ->  tabdata/tabNN_<name>.json   (gitignored)
tabNN_<name>_tex.py    reads ONLY that json        ->  tabNN_<name>.tex            (committed)
```

`paper.tex` keeps `\begin{table}`, `\caption`, and `\label`, and `\input`s the
generated `ruledtabular` body — so no numeric digit is hand-transcribed into the
manuscript.

| stem | label | what it verifies | cost |
|---|---|---|---|
| `tab01_production_box`    | `tab:box`       | the production parameter box + shipped build configuration | instant |

## Entry points

```bash
make tabdata                                   # recompute every tabdata/*.json
make tables                                    # tabdata, then re-render every .tex
python paper/tables/make_tabdata.py --check     # presence matrix
python paper/tables/make_tabdata.py --tab tab01 --force
```

Table I needs no solve: it reads the box edges, Smolyak level, spatial grid and
enhanced axis set from `pipeline/production_box.py` — the canonical box definition —
and the sparse-grid node counts from `parametric_nd_2c.smolyak_points`, so retargeting
an edge there and re-running `make tables` moves the paper's numbers with it.

## Guard

`tests/test_paper_tables.py` (fast tier, file reads only) asserts that the rendered
`.tex` is not stale relative to its source module, and that `paper.tex` `\input`s the
generated body rather than hand-written rows. It also guards
`pipeline/run_tangent_verification.py`, the standalone parameter-sensitivity
verification tool: it no longer feeds a paper table, but it stays the runnable
cross-check behind the sensitivity claims of Sec. IV, alongside
`tests/test_sensitivity*.py`.
