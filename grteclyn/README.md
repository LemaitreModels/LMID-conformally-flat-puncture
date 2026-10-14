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
| `params_base.txt` | **ours** — the GRTeclyn parameter template (prefixed keys, see below), with `@N@`/`@L@`/… substituted per rung |
| GRTeclyn itself | **not here** — a separate code by the GRTL collaboration |

GRTeclyn is third-party and is not vendored, for the same reason `oracle/` does
not vendor NRPy's TwoPunctures sources: this repository does not carry code it
does not own. You need a build of it with the `lm.id_file` runtime switch, which
is what lets one executable read our spectral export instead of its own analytic
Bowen–York data. The paper's `"paper-v1"` measurements were run with GRTeclyn
commit `2bfd19e` (cited as `\cite{GRTeclyn}`) — **that SHA belongs to the GRTL
collaboration's repository, not to this one, so it must never be restamped
against this repo's history.**

## Which GRTeclyn this driver targets

`params_base.txt` and `run_ladder.sh` are written for GRTeclyn's **prefixed
parameter keys** — the generation introduced upstream by PR #215
(`refactor_params2`): `amr.n_cell`, `geometry.prob_extent`, `bh1.mass`,
`evolution.max_steps`, `boundary.hi_condition`, … — plus our fork's `lm.*` keys
(`lm.id_file`, `lm.id_reference_file`, `lm.id_reference_tol`,
`lm.calculate_constraint_norms`, `lm.constraint_norm_exclusion_radius`,
`lm.constraint_norm_border_cells`). Build the `BinaryBH` example from our public
fork's `lm-initial-data-constraints` branch at commit **`087f827`** or later: that
is upstream `main` (through PRs #264 and #193) plus the LM reader, and it is the
first commit that reads a **format-4** (`"v2"`) export. Format-2 (`"paper-v1"`)
exports are read by the same binary.

**An older build will not fail on these keys — it will ignore them.** GRTeclyn
before PR #215, including `2bfd19e` above, reads *bare* keys (`N_full`, `L_full`,
`massA`, `max_steps`, `lm_id_file`, `calculate_constraint_norms`, …), and a key
from the other generation is not an error to `ParmParse`: it is never read and
the default takes its place. The same holds in reverse, so this directory's
history carries the bare-key template for those builds; do not mix the two.

That is why the template sets `amrex.abort_on_unused_inputs = 1` (AMReX then
aborts at the end of the run if any key in the file was never read), and why
`run_ladder.sh` checks every rung before it is collected: the binary must have
initialised, reported no unused key, opened (and, with `LMREF=`, validated) the
LM file, computed the norms in `post_init`, taken **no** evolution step, and
written a `constraint_norms.json` that echoes the `N`, exclusion radius and
border cells that were asked for. It reports "never got as far as the
measurement" separately from "ran but ignored keys", because the two need
opposite fixes. Note that `parameters_and_version.txt`, which GRTeclyn writes at
start-up, lists **every** key in the inputs file whether or not it is read, so
the presence of a key there proves nothing about whether the binary uses it.

Two keys are written only when the binary will read them, since an unread key
now fails the run: `puncture_tracking.level` (only with `PTRACK=1`) and
`weyl_extraction.center` (only with `MAXLEV>0`, where the refinement taggers run
and require it even with extraction disabled). The Sommerfeld asymptotic values
(`chi`, `h11`, `h22`, `h33`, lapse → 1) are compiled into this generation and are
no longer parameters.

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

One asymmetry worth knowing: GRTeclyn's parameter check flags `|P| >= 0.3 m`,
because its *own* initial data is an `O(P^2)` small-boost approximation — a hard
abort in the bare-key generation, a warning in the prefixed one. The anchor is
well outside that bound, so on the `LMID=` path the script zeroes the params'
momenta — they are read only by the analytic branch, and the momenta that matter
travel in the `.lmid` file. On that path the script also checks that the file's
`b` equals `BZ`: the `bh*.offset`s are what place the constraint-norm exclusion
balls, so a mismatch would mask the wrong region without any other symptom.
