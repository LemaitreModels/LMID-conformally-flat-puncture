#!/usr/bin/env bash
# Run a uniform-grid resolution ladder of t=0 constraint-norm measurements with
# GRTeclyn, and collect them into the `<tag>/ladder.json` that
# `paper/figures/fig10_constraints_data.py` distils.  See ./README.md and
# ../docs/DATA.md.
#
#   LM_GRTECLYN_EXE=/path/to/BinaryBH3d.<...>.ex \
#     ./grteclyn/run_ladder.sh <tag> "<N-list>" [KEY=VALUE ...]
#
# e.g.  ./grteclyn/run_ladder.sh lm_anchor "48 64 96 128 192" \
#           LMID=/path/to/qc_b3.lmid LMREF=/path/to/qc_b3_reference.dat
#
# Each rung runs in `<tag>/N<N>/` and leaves `constraint_norms.json` there;
# they are collected into `<tag>/ladder.json` at the end.
#
# Environment:
#   LM_GRTECLYN_EXE  the GRTeclyn BinaryBH executable (REQUIRED -- there is no
#                    default, because guessing it is how you silently measure a
#                    different build than the one you meant)
#   LM_MPIRUN        MPI launcher (default `mpirun`)
#   LM_NRANKS        ranks per rung (default 4)
#   LM_GRTECLYN_RUNS root of the run tree (default ./runs next to this script)
#
# Prereqs: a GRTeclyn build that reads the PREFIXED parameter keys of upstream
# PR #215 and later, with our fork's `lm.*` keys (the LM initial-data reader and
# the t = 0 constraint norms).  That is our fork's `lm-initial-data-constraints`
# branch; reading a format-4 ("v2") export needs commit 087f827 or later on it.
# See ./README.md.  The paper's "paper-v1" measurements were run with GRTeclyn
# commit 2bfd19e (cited as \cite{GRTeclyn}), which reads the OLDER, bare keys;
# that SHA belongs to the GRTL collaboration's repository, not to this one.
#
# Every rung is checked before it is believed (see check_rung below): the binary
# must have read every key we wrote (`amrex.abort_on_unused_inputs = 1` in
# params_base.txt), opened and validated the LM file when one is given, run
# post_init (where the norms are computed), taken no evolution step, and written
# a constraint_norms.json that echoes the N, r_excl and border cells we asked for.
#
# Nothing here is imported by the package: this drives an external binary over
# files, exactly as `oracle/` does for TwoPunctures.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXE="${LM_GRTECLYN_EXE:?set LM_GRTECLYN_EXE to the GRTeclyn BinaryBH executable}"
MPIRUN="${LM_MPIRUN:-mpirun}"
NRANKS="${LM_NRANKS:-4}"

[ -x "$EXE" ] || { echo "error: not executable: $EXE" >&2; exit 1; }
[ -f "$HERE/params_base.txt" ] || { echo "error: params_base.txt missing" >&2; exit 1; }

TAG="${1:?usage: run_ladder.sh <tag> \"<N-list>\" [KEY=VALUE ...]}"; shift
NLIST="${1:?usage: run_ladder.sh <tag> \"<N-list>\" [KEY=VALUE ...]}"; shift

# Defaults: the axisymmetric anchor of the validation appendix -- equal bare
# masses, b = 3, |P| = 0.5, on a box of half-width 9 M (L = 18) with the
# punctures excluded to 1.5 M.  These are the paper's numbers; fig10's figdata
# records them in `meta` (L_full, b, m_A, m_B, P_anchor, r_excl), so a change
# here without a re-distill makes the caption wrong.
L=18
MASSA=0.5
MASSB=0.5
BZ=3.0
PA=-0.5
PB=0.5
RXCL=1.5
# LMID: path to a spectral export from `pipeline.run_export_grteclyn`.  When set,
# the run uses that initial data instead of GRTeclyn's analytic Bowen-York --
# same executable, same stencils, same conversion into evolution variables, so
# the comparison isolates the initial data.  LMREF: the reference table the C++
# evaluation is validated against, to LMTOL.
LMID=""
LMREF=""
LMTOL=1e-10
MAXLEV=0
PTRACK=0
for kv in "$@"; do eval "$kv"; done

# `amr.regrid_int`: one entry per level below MAXLEV (a single shared value is
# also accepted by this generation; the per-level form is kept so the params are
# the same at every depth).
REGRID=""
for ((lev = 0; lev < MAXLEV; lev++)); do REGRID="$REGRID 0"; done
[ -z "$REGRID" ] && REGRID="0"
REGRID="${REGRID# }"

OUT="${LM_GRTECLYN_RUNS:-$HERE/runs}/$TAG"
mkdir -p "$OUT"

abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
# One record of a .lmid file, e.g. `lmid_record qc_b3.lmid b` -> 3.0000...e+00
lmid_record() { awk -v k="$2" '$1 == k { print $2; exit }' "$1"; }

# On the LM path the momenta live in the exported file.  The params' bh*.momentum
# are read only by the analytic branch (not taken here) and by GRTeclyn's
# parameter check, which flags |P| >= 0.3*mass because its own initial data is an
# O(P^2) small-boost approximation (a hard abort in the older key generation, a
# warning in this one).  They are zeroed on the LM path so the params carry no
# momentum the run does not use, and the run is warning-free.
if [ -n "$LMID" ]; then
    [ -f "$LMID" ] || { echo "error: LMID not found: $LMID" >&2; exit 1; }
    [ -z "$LMREF" ] || [ -f "$LMREF" ] || { echo "error: LMREF not found: $LMREF" >&2; exit 1; }
    if [ "$PA" != "0.0" ] || [ "$PB" != "0.0" ]; then
        echo "    note: zeroing params momenta ($PA/$PB) -- unused on the LM path"
        echo "          (the .lmid file carries them)."
    fi
    PA=0.0
    PB=0.0
    # The bh*.offset (= +-BZ on z) place the constraint-norm exclusion balls; the
    # LM data places the punctures at +-b from its own file.  If they disagree
    # the mask excludes the wrong region and nothing else notices.
    LMID_B="$(lmid_record "$LMID" b)"
    python3 -c "import sys; b, bz = float(sys.argv[1]), float(sys.argv[2]); \
sys.exit(0 if abs(b - bz) <= 1e-12 * max(1.0, abs(b)) else 1)" "${LMID_B:-nan}" "$BZ" \
        || { echo "error: $LMID has b = ${LMID_B:-<none>}, but BZ = $BZ places the" \
                  "punctures (and the exclusion balls) elsewhere" >&2; exit 1; }
fi

echo "=== ladder '$TAG': N = $NLIST ==="
echo "    exe=$EXE"
echo "    L=$L massA=$MASSA massB=$MASSB b=$BZ PA=$PA PB=$PB r_excl=$RXCL maxlev=$MAXLEV ptrack=$PTRACK"
[ -n "$LMID" ] && echo "    LM initial data: $LMID  ($(grep -m1 '^format' "$LMID"))"

# --- the per-rung guard ------------------------------------------------------
# check_rung <run dir> <N> <exit code>.  Three different failures look alike from
# outside, and are reported apart:
#   * the binary never got as far as the t = 0 measurement (wrong exe, bad
#     params value, MPI failure): nothing was tested;
#   * it ran but did not read some key we wrote: AMReX lists it as an "unused
#     ParmParse variable" and abort_on_unused_inputs turns that into a non-zero
#     exit AFTER constraint_norms.json was written -- so the json's existence
#     proves nothing on its own;
#   * it ran, read everything, and measured something other than what we asked
#     for (N, r_excl, border cells echoed back in the json), or stepped.
param() { sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*//p" "$1" | head -1; }
check_rung() {
    local run="$1" n="$2" rc="$3" log="$1/run.log"
    local fail=""
    if ! grep -q 'GRTeclyn (.*) initialized' "$log"; then
        echo "  FAILED: GRTeclyn never initialised (rc=$rc) -- nothing was measured;"
        echo "          check the executable and the launcher. Tail of log:"
        tail -25 "$log"; return 1
    fi
    echo "  binary: $(grep -m1 -o 'GRTeclyn (.*) initialized' "$log")"
    if grep -q -E 'Unused ParmParse Variables|unused ParmParse variables' "$log"; then
        echo "  FAILED: the binary did NOT READ these keys -- they are from another"
        echo "          parameter generation or misspelt, and took their defaults:"
        sed -n '/Unused ParmParse Variables/,/^$/p' "$log" | sed 's/^/          /'
        return 1
    fi
    if ! grep -q 'Constraint norms at t = ' "$log"; then
        echo "  FAILED: post_init printed no constraint norms (rc=$rc) -- either the run"
        echo "          died before post_init, or lm.calculate_constraint_norms was not"
        echo "          honoured. Why it stopped (a REQUIRED key missing is reported as"
        echo "          'not found in database', and is followed by a long table dump):"
        grep -m5 -E 'not found in database|Error from parameter|Abort' "$log" \
            | sed 's/^/          /' || true
        echo "          Tail of log:"
        tail -25 "$log"; return 1
    fi
    if [ "$rc" -ne 0 ]; then
        echo "  FAILED: exit $rc after the measurement. Tail of log:"
        tail -25 "$log"; return 1
    fi
    grep -q 'GRTeclyn finished' "$log" || fail="$fail no-'GRTeclyn finished'-line"
    local ns
    ns="$(grep -c '\[Level 0 step [0-9]*\] ADVANCE' "$log" || true)"
    [ "$ns" -eq 0 ] || fail="$fail took-$ns-coarse-steps(evolution.max_steps-not-0?)"
    if [ -n "$LMID" ]; then
        grep -q '^LM-ID:' "$log" || fail="$fail no-LM-ID-provenance(lm.id_file-not-read?)"
        if [ -n "$LMREF" ]; then
            grep -q 'LMInitialData::validate:' "$log" \
                || fail="$fail no-validate-line(lm.id_reference_file-not-read?)"
            grep -m1 'LMInitialData::validate:' "$log" | sed 's/^/  /' || true
        fi
    else
        ! grep -q '^LM-ID:' "$log" || fail="$fail LM-ID-read-on-the-analytic-path"
    fi
    if ! python3 - "$run/constraint_norms.json" "$n" \
        "$(param "$run/params.txt" lm.constraint_norm_exclusion_radius)" \
        "$(param "$run/params.txt" lm.constraint_norm_border_cells)" <<'PY'
import json, sys
path, n, r_excl, border = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
try:
    with open(path) as f:
        d = json.load(f)
except (OSError, ValueError) as e:
    sys.exit(f"  {path}: {e}")
bad = []
if d.get("n_cell_level0") != n:
    bad.append(f"n_cell_level0 = {d.get('n_cell_level0')} != N = {n}")
if d.get("time") != 0.0:
    bad.append(f"time = {d.get('time')} != 0")
if d.get("r_excl") is None or abs(d["r_excl"] - r_excl) > 1e-12:
    bad.append(f"r_excl = {d.get('r_excl')} != {r_excl}")
if d.get("border_cells") != border:
    bad.append(f"border_cells = {d.get('border_cells')} != {border}")
if not d.get("n_cells"):
    bad.append(f"n_cells = {d.get('n_cells')}")
for k in ("L2_Ham", "L2_Mom"):
    if k not in d:
        bad.append(f"no {k}")
if bad:
    sys.exit("  constraint_norms.json does not echo the request: " + "; ".join(bad))
PY
    then
        fail="$fail constraint_norms.json"
    fi
    if [ -n "$fail" ]; then
        echo "  FAILED:$fail"; return 1
    fi
    echo "  read-keys guard: OK (no unused keys, post_init measured, 0 steps, json echoes N/r_excl/border)"
}

for N in $NLIST; do
    RUN="$OUT/N$N"
    mkdir -p "$RUN"
    rm -f "$RUN/constraint_norms.json"        # a stale json must not pass the guard
    sed -e "s/@N@/$N/g" -e "s/@L@/$L/g" \
        -e "s/@MASSA@/$MASSA/g" -e "s/@MASSB@/$MASSB/g" \
        -e "s/@BZ@/$BZ/g" -e "s/@PA@/$PA/g" -e "s/@PB@/$PB/g" \
        -e "s/@RXCL@/$RXCL/g" \
        -e "s/@MAXLEV@/$MAXLEV/g" -e "s/@PTRACK@/$PTRACK/g" \
        -e "s/@REGRID@/$REGRID/g" \
        "$HERE/params_base.txt" > "$RUN/params.txt"
    # Keys appended only when this binary will read them: under
    # abort_on_unused_inputs a key nobody reads fails the run.
    if [ "$PTRACK" != "0" ]; then
        { echo ""
          echo "# puncture tracking on level 0 (read only when tracking is enabled)"
          echo "puncture_tracking.level = 0"
        } >> "$RUN/params.txt"
    fi
    if [ "$MAXLEV" -gt 0 ]; then
        C="$(python3 -c "import sys; print(float(sys.argv[1]) / 2)" "$L")"
        { echo ""
          echo "# read by the extraction tagger, which runs only when there are finer levels"
          echo "weyl_extraction.center = $C $C $C"
        } >> "$RUN/params.txt"
    fi
    if [ -n "$LMID" ]; then
        { echo ""
          echo "# LM-initial-data spectral initial data (runtime switch)"
          echo "lm.id_file = $(abspath "$LMID")"
          if [ -n "$LMREF" ]; then
            echo "lm.id_reference_file = $(abspath "$LMREF")"
            echo "lm.id_reference_tol = $LMTOL"
          fi
        } >> "$RUN/params.txt"
    fi
    echo "--- N = $N  (h = $(python3 -c "print($L/$N)")) ---"
    rc=0
    ( cd "$RUN" && $MPIRUN -n "$NRANKS" "$EXE" params.txt > run.log 2>&1 ) || rc=$?
    check_rung "$RUN" "$N" "$rc" || exit 1
    grep -E "L2_Ham|L2_Mom" "$RUN/run.log" | head -2 | sed 's/^/  /'
done

# --- collect into the schema fig10_constraints_data.py reads ----------------
LMID_ABS=""
[ -n "$LMID" ] && LMID_ABS="$(abspath "$LMID")"
python3 - "$OUT" "$L" "$LMID_ABS" $NLIST <<'PY'
import json, math, os, sys
out, L, lmid = sys.argv[1], float(sys.argv[2]), sys.argv[3]
ns = [int(a) for a in sys.argv[4:]]
# Which discretisation the LM initial data was exported from: the .lmid's own
# `format` record (2 = "paper-v1", 4 = "v2").  fig10_constraints_data refuses a
# series whose scheme is not production_model.SCHEME, so an old ladder can never
# be stamped as the current grid.  No LMID (GRTeclyn's own analytic data) -> None.
fmt, grid = None, {}
if lmid:
    with open(lmid) as f:
        for line in f:
            key = line.split(" ", 1)[0]
            if key == "format":
                fmt = int(line.split()[1])
            elif key in ("Na", "Nb", "Nphi"):
                grid[key] = int(line.split()[1])
scheme = {2: "paper-v1", 4: "v2"}.get(fmt)
if lmid and scheme is None:
    sys.exit(f"{lmid}: no known `format` record ({fmt!r}); refusing to label the ladder")
rows = []
for n in ns:
    with open(os.path.join(out, f"N{n}", "constraint_norms.json")) as f:
        d = json.load(f)
    d["N"] = n
    d["h"] = L / n
    rows.append(d)
with open(os.path.join(out, "ladder.json"), "w") as f:
    json.dump({"rungs": rows, "lm_format": fmt, "scheme": scheme,
               "spectral_grid": [grid.get("Na"), grid.get("Nb"), grid.get("Nphi")]
               if grid else None}, f, indent=2)


def show(v):
    # a non-finite norm is written as JSON null, so it must print, not crash
    return "        null" if v is None else f"{v:>13.4e}"


print(f"\n{'N':>5} {'h':>10} {'L2_Ham':>13} {'L2_Mom':>13} {'cells':>10}")
for r in rows:
    print(f"{r['N']:>5} {r['h']:>10.5f} {show(r['L2_Ham'])} "
          f"{show(r['L2_Mom'])} {r['n_cells']:>10d}")
for key in ("L2_Ham", "L2_Mom"):
    print(f"\nlocal orders, {key}:")
    for a, b in zip(rows[:-1], rows[1:]):
        if a[key] and b[key] and a[key] > 0 and b[key] > 0:
            p = math.log(a[key] / b[key]) / math.log(a["h"] / b["h"])
            print(f"  N {a['N']:>4} -> {b['N']:<4}  order = {p:6.3f}")
print(f"\nwritten: {os.path.join(out, 'ladder.json')}")
PY
