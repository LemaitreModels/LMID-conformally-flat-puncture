# The GRTeclyn constraint check

The external evolution code this package's initial data is handed to, and the
recipe that runs the resolution ladder behind Fig. 10.

```bash
# 1. export the certified initial data (this package)
python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.run_export_grteclyn \
    --out /tmp/lmid_export --tp

# 2. run the ladder (GRTeclyn; external binary, MPI)
LM_GRTECLYN_EXE=/path/to/BinaryBH3d.gnu.MPI.OMP.ex \
  ./grteclyn/run_ladder.sh lm_anchor "48 64 96 128 192" \
      LMID=/tmp/lmid_export/qc_b3.lmid LMREF=/tmp/lmid_export/qc_b3_reference.dat

# 3. distil into figdata (this package again)
python paper/figures/fig10_constraints_data.py --runs <runs root>
```

## What is here, and what deliberately is not

| | |
|---|---|
| `run_ladder.sh` | **ours** — drives one resolution ladder and collects `<tag>/ladder.json` |
| `params_base.txt` | **ours** — the GRTeclyn parameter template, with `@N@`/`@L@`/… substituted per rung |
| GRTeclyn itself | **not here** — a separate code by the GRTL collaboration |

GRTeclyn is third-party and is not vendored, for the same reason `oracle/` does
not vendor NRPy's TwoPunctures sources: this repository does not carry code it
does not own. You need a build of it with the `lm_id_file` runtime switch, which
is what lets one executable read our spectral export instead of its own analytic
Bowen–York data. The paper's measurements were run with GRTeclyn commit
`2bfd19e` (cited as `\cite{GRTeclyn}`) — **that SHA belongs to the GRTL
collaboration's repository, not to this one, so it must never be restamped
against this repo's history.**

## The boundary this preserves

**The package never imports anything here, and GRTeclyn never imports the
package.** The contract between them is one file: a `.lmid` spectral export
written by `validation/export_grteclyn.py`, read by the C++ side, with the format
version checked by *equality* — a newer file under an older rule is silently
wrong rather than an error, so both halves must move together.

`run_ladder.sh` shells out to an external binary over files and is not importable
Python; nothing in `src/` reaches for it.

## The comparison this is arranged to make

Same executable, same fourth-order stencils, same conversion into evolution
variables — only the initial data differs. That is what makes the Hamiltonian and
momentum norms attributable to the initial data rather than to a code difference,
and it is why `LMID=` switches the data *inside* an otherwise identical run
rather than comparing two codes' outputs.

The `LMREF=` table closes the other half: it pins the C++ evaluation of the
export against this package's own evaluator to `LMTOL` (`1e-10` by default), so a
disagreement in the norms cannot be an interpolation bug on the consumer side.

## Configuration, and where its numbers are pinned

`run_ladder.sh`'s defaults are the validation appendix's axisymmetric anchor:
equal bare masses `0.5`, `b = 3`, `|P| = 0.5`, `L_full = 18` (a box of half-width
`9 M`), punctures excluded to `1.5 M`. Fig. 10's figdata records every one of
them in its `meta` block (`L_full`, `b`, `m_A`, `m_B`, `P_anchor`, `r_excl`,
`border_cells`), so changing a default here without re-distilling makes the
caption wrong. Check them against `paper/figures/figdata/fig10_constraints.json`
before you change one.

One asymmetry worth knowing before it aborts a run: GRTeclyn's `check_params`
rejects `|P| >= 0.3 m`, because its *own* initial data is an `O(P^2)` small-boost
approximation. The anchor is well outside that bound, so on the `LMID=` path the
script zeroes the params' momenta — they are read only by the analytic branch,
and the momenta that matter travel in the `.lmid` file.
