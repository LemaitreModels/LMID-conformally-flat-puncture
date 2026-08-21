#!/usr/bin/env bash
# Reproducible build of the standalone TwoPunctures oracle this package
# validates against.  See ../docs/DATA.md and ./README.md.
#
# The oracle is the Ansorg-Brügmann-Tichy single-domain pseudospectral
# TwoPunctures solver (Phys. Rev. D 70, 064011), ported to C by Z. Etienne as
# nrpy.infrastructures.BHaH.general_relativity.TwoPunctures.  gen_tp.py emits
# those C sources; src/main.c (ours) is a thin driver over them.  Nothing here
# is imported by the package -- it only ever shells out to the built binary.
#
# Prereqs: a Python env with `nrpy` installed, plus GSL and a C compiler.
# With micromamba/conda:
#     micromamba create -y -n lm-tp-oracle -c conda-forge python=3.12 gsl make
#     micromamba run -n lm-tp-oracle pip install 'nrpy==2.2026.6' numpy
#     micromamba run -n lm-tp-oracle ./oracle/build.sh
#
# Environment:
#   LM_TP_PREFIX  where to install the binary
#                 (default ~/.cache/lemaitre/tp-oracle, which is also
#                  validation/twopunctures.py's default lookup path)
#   GSL_PREFIX    GSL location (default: the active conda/micromamba env,
#                 else /usr/local)
#   CC            compiler (default cc)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${LM_TP_PREFIX:-$HOME/.cache/lemaitre/tp-oracle}"
GSL="${GSL_PREFIX:-${CONDA_PREFIX:-/usr/local}}"
CC="${CC:-cc}"

command -v python >/dev/null || { echo "error: no python on PATH" >&2; exit 1; }
python -c 'import nrpy' 2>/dev/null || {
  echo "error: nrpy is not importable." >&2
  echo "       pip install 'nrpy==2.2026.6'  -- see the prereqs at the top." >&2
  exit 1; }
[ -f "$GSL/include/gsl/gsl_math.h" ] || {
  echo "error: GSL headers not found under $GSL." >&2
  echo "       Set GSL_PREFIX to the prefix containing include/gsl/." >&2
  exit 1; }

echo "[1/3] generating the TwoPunctures C sources via NRPy ..."
python "$HERE/gen_tp.py"

echo "[2/3] compiling tp_solve ($CC, GSL at $GSL) ..."
mkdir -p "$PREFIX"
"$CC" -O2 -std=c99 -w \
  -I "$HERE/src" -I "$GSL/include" \
  "$HERE"/src/*.c \
  -L "$GSL/lib" -Wl,-rpath,"$GSL/lib" -lgsl -lgslcblas -lm \
  -o "$PREFIX/tp_solve"

echo "[3/3] done: $PREFIX/tp_solve"
echo
echo "That is validation/twopunctures.py's default location, so the oracle tier"
echo "and the 29 'slow' tests will now find it with no further setup.  To use a"
echo "different path instead:  export LM_TP_BIN=$PREFIX/tp_solve"
