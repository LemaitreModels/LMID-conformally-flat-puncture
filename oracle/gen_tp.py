"""Generate a STANDALONE TwoPunctures solver from NRPy's BHaH TwoPunctures port.

Run with the dedicated build env (which has nrpy + GSL):

    python gen_tp.py        # in an env with nrpy installed (see build.sh)

This emits the 6 TwoPunctures C source files + the two provided headers + a
minimal BHaH_defines.h (REAL + ID_persist_struct only) into ./src/.  A hand-
written main.c (see Makefile) fills ID_persist_struct directly and calls
TP_solve + PunctIntPolAtArbitPosition, so we do NOT need NRPy's commondata /
NRPyPN / grid machinery (initialize_ID_persist_struct is intentionally skipped).

The generated solver is the Ansorg-Brügmann-Tichy single-domain pseudospectral
method (PRD 70, 064011) — the established TwoPunctures code, ported by
Z. Etienne (NRPy).  This script lives OUTSIDE src/ so the installed package
stays jax/numpy/matplotlib-only; the validation harness only ever shells out to
the compiled binary.
"""
import os
import shutil

import nrpy.c_function as cfc
from nrpy.infrastructures.BHaH.general_relativity import TwoPunctures as TP

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
os.makedirs(SRC, exist_ok=True)

# --- the persist-struct body is produced by ID_persist_str() (also registers a
#     few code params as a side effect, harmless here) -----------------------
id_body = TP.ID_persist_struct.ID_persist_str()

# --- register exactly the 6 C functions on the solve/interp(u) path ----------
TP.TP_utilities.register_CFunction_TP_utilities()
TP.CoordTransf.register_CFunction_TP_CoordTransf()
TP.Equations.register_CFunction_TP_Equations()
TP.FuncAndJacobian.register_CFunction_TP_FuncAndJacobian()
TP.Newton.register_CFunction_TP_Newton()
TP.TP_solve.register_CFunction_TP_solve()

for name, f in cfc.CFunction_dict.items():
    with open(os.path.join(SRC, f"{name}.c"), "w") as fh:
        fh.write(f.full_function)
    print("wrote", f"{name}.c")

# --- copy the two provided headers verbatim ----------------------------------
pkg = os.path.dirname(TP.__file__)
for h in ("TwoPunctures.h", "TP_utilities.h"):
    shutil.copy(os.path.join(pkg, h), os.path.join(SRC, h))
    print("copied", h)

# --- minimal BHaH_defines.h: just REAL + ID_persist_struct -------------------
#     (TP_utilities.h includes BHaH_defines.h then TwoPunctures.h; the latter
#      supplies REAL + the `derivs` struct referenced by ID_persist_struct.)
bhah = f"""// Minimal BHaH_defines.h for the STANDALONE TwoPunctures oracle.
// Only the symbols the solve/interp path touches: REAL + ID_persist_struct.
#ifndef BHAH_DEFINES_H
#define BHAH_DEFINES_H
#include "TwoPunctures.h"   // REAL (double) + the `derivs` struct
#include <stdbool.h>
typedef struct __ID_persist_struct__ {{
{id_body}
}} ID_persist_struct;
#endif // BHAH_DEFINES_H
"""
with open(os.path.join(SRC, "BHaH_defines.h"), "w") as fh:
    fh.write(bhah)
print("wrote BHaH_defines.h")
print("DONE -> %s" % SRC)
