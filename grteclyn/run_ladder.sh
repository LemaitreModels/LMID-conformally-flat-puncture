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
#
# Prereqs: a built GRTeclyn with the `lm_id_file` runtime switch -- our fork's
# `lm-initial-data-constraints` branch.  The paper's measurements were run with
# GRTeclyn commit 2bfd19e (cited as \cite{GRTeclyn}); that SHA belongs to the
# GRTL collaboration's repository, not to this one.
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
# masses, b = 3, |P| = 0.5, on a box of half-width 9 M (L_full = 18) with the
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

# `regrid_interval` is a per-level array of MAXLEV values; a scalar aborts with
# "too many values requested", so it has to be built after MAXLEV is known.
REGRID=""
for ((lev = 0; lev < MAXLEV; lev++)); do REGRID="$REGRID 0"; done
[ -z "$REGRID" ] && REGRID="0"
REGRID="${REGRID# }"

OUT="${LM_GRTECLYN_RUNS:-$HERE/runs}/$TAG"
mkdir -p "$OUT"

# On the LM path the momenta live in the exported file.  The params' momentumA/B
# are read only by the analytic branch (not taken here) and by GRTeclyn's
# check_params, which enforces |P| < 0.3*mass because its own initial data is an
# O(P^2) small-boost approximation.  The appendix anchor (|P| = 0.5, m = 0.5) is
# well outside that, so leaving them set would abort a run whose initial data
# does not use them at all.  Zero them, and say so.
if [ -n "$LMID" ]; then
    if [ "$PA" != "0.0" ] || [ "$PB" != "0.0" ]; then
        echo "    note: zeroing params momenta ($PA/$PB) -- unused on the LM path"
        echo "          (the .lmid file carries them) and check_params would"
        echo "          reject them as too large a boost."
    fi
    PA=0.0
    PB=0.0
fi

echo "=== ladder '$TAG': N = $NLIST ==="
echo "    exe=$EXE"
echo "    L=$L massA=$MASSA massB=$MASSB b=$BZ PA=$PA PB=$PB r_excl=$RXCL"
[ -n "$LMID" ] && echo "    LM initial data: $LMID"

for N in $NLIST; do
    RUN="$OUT/N$N"
    mkdir -p "$RUN"
    sed -e "s/@N@/$N/g" -e "s/@L@/$L/g" \
        -e "s/@MASSA@/$MASSA/g" -e "s/@MASSB@/$MASSB/g" \
        -e "s/@BZ@/$BZ/g" -e "s/@PA@/$PA/g" -e "s/@PB@/$PB/g" \
        -e "s/@RXCL@/$RXCL/g" \
        -e "s/@MAXLEV@/$MAXLEV/g" -e "s/@PTRACK@/$PTRACK/g" \
        -e "s/@REGRID@/$REGRID/g" \
        "$HERE/params_base.txt" > "$RUN/params.txt"
    if [ -n "$LMID" ]; then
        { echo ""
          echo "# LM-initial-data spectral initial data (runtime switch)"
          echo "lm_id_file = $(cd "$(dirname "$LMID")" && pwd)/$(basename "$LMID")"
          if [ -n "$LMREF" ]; then
            echo "lm_id_reference_file = $(cd "$(dirname "$LMREF")" && pwd)/$(basename "$LMREF")"
            echo "lm_id_reference_tol = $LMTOL"
          fi
        } >> "$RUN/params.txt"
    fi
    echo "--- N = $N  (h = $(python3 -c "print($L/$N)")) ---"
    ( cd "$RUN" && $MPIRUN -n "$NRANKS" "$EXE" params.txt > run.log 2>&1 ) \
        || { echo "  FAILED -- tail of log:"; tail -25 "$RUN/run.log"; exit 1; }
    grep -E "L2_Ham|L2_Mom" "$RUN/run.log" | head -2 | sed 's/^/  /'
done

# --- collect into the schema fig10_constraints_data.py reads ----------------
python3 - "$OUT" "$L" $NLIST <<'PY'
import json, math, os, sys
out, L = sys.argv[1], float(sys.argv[2])
ns = [int(a) for a in sys.argv[3:]]
rows = []
for n in ns:
    with open(os.path.join(out, f"N{n}", "constraint_norms.json")) as f:
        d = json.load(f)
    d["N"] = n
    d["h"] = L / n
    rows.append(d)
with open(os.path.join(out, "ladder.json"), "w") as f:
    json.dump({"rungs": rows}, f, indent=2)


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
