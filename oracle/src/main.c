/* Standalone TwoPunctures driver for the LM-initial-data validation harness.
 *
 * Fills ID_persist_struct DIRECTLY (no NRPy commondata / NRPyPN), solves the
 * Ansorg-Brügmann-Tichy puncture equation (TP_solve), and reports:
 *   - the total ADM mass (par.E) and individual ADM masses (par.mp_adm/mm_adm),
 *   - the angular momenta J1,J2,J3,
 *   - the regular correction u = PunctIntPolAtArbitPosition(...) and the full
 *     conformal factor psi = 1 + mp/(2 r_+) + mm/(2 r_-) + u at query points.
 *
 * Native TwoPunctures frame: m+ at (+par_b,0,0), m- at (-par_b,0,0) on the
 * x-axis.  Usage:
 *     tp_solve  b  mA  mB  P  nA  nB  nphi  [Newton_tol] [Newton_maxit]
 * with query points "x y z" read one-per-line from stdin.  Numeric results go
 * to stdout (lines "SUMMARY ..." and "POINT ..."); TP's own chatter -> stderr.
 */
#include "TP_utilities.h" /* pulls BHaH_defines.h (ID_persist_struct) + TwoPunctures.h (REAL, derivs) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* TP_solve lives in TP_solve.c (not declared in TP_utilities.h) */
void TP_solve(ID_persist_struct *par);

int main(int argc, char **argv) {
  if (argc < 8) {
    fprintf(stderr, "usage: %s b mA mB P nA nB nphi [Newton_tol] [Newton_maxit] [SA SB] "
                    "[Pp0 Pp1 Pp2 Pm0 Pm1 Pm2 Sp0 Sp1 Sp2 Sm0 Sm1 Sm2]\n", argv[0]);
    return 1;
  }
  ID_persist_struct par;
  memset(&par, 0, sizeof(par));

  const REAL b = atof(argv[1]);
  const REAL mA = atof(argv[2]); /* +b puncture (hole A) -> m+ */
  const REAL mB = atof(argv[3]); /* -b puncture (hole B) -> m- */
  const REAL P = atof(argv[4]);  /* infall momentum magnitude */
  par.npoints_A = atoi(argv[5]);
  par.npoints_B = atoi(argv[6]);
  par.npoints_phi = atoi(argv[7]);
  par.Newton_tol = (argc > 8) ? atof(argv[8]) : 1.0e-12;
  par.Newton_maxit = (argc > 9) ? atoi(argv[9]) : 12;

  /* geometry + Bowen-York momenta (head-on infall, native x-axis frame) */
  par.par_b = b;
  par.par_m_plus = mA;
  par.par_m_minus = mB;
  par.give_bare_mass = true; /* par_m_* ARE the bare puncture masses */
  par.target_M_plus = mA;
  par.target_M_minus = mB;
  for (int i = 0; i < 3; i++) {
    par.par_P_plus[i] = par.par_P_minus[i] = 0.0;
    par.par_S_plus[i] = par.par_S_minus[i] = 0.0;
    par.center_offset[i] = 0.0;
  }
  par.par_P_plus[0] = -P; /* +b puncture moves in -x (toward origin) */
  par.par_P_minus[0] = +P;
  /* Optional aligned spins along the collision (x) axis — the TP-frame image of
   * this package's z-axis spins (S_X ẑ -> par_S_X[0]).  Default 0 (head-on,
   * non-spinning) keeps the B1 behaviour byte-identical.  (Milestone P2.) */
  par.par_S_plus[0] = (argc > 10) ? atof(argv[10]) : 0.0;  /* +b puncture (= S_A) */
  par.par_S_minus[0] = (argc > 11) ? atof(argv[11]) : 0.0; /* -b puncture (= S_B) */

  /* Full-VECTOR override (Test E, non-axisymmetric).  When all 12 trailing
   * components are supplied (argc >= 24) they REPLACE the scalar momentum/spin
   * set above with arbitrary per-puncture 3-vectors, already rotated into the
   * TP native (x-axis collision) frame by the Python wrapper.  Components are
   * read in the order  par_P_plus[0..2] par_P_minus[0..2] par_S_plus[0..2]
   * par_S_minus[0..2].  With argc < 24 the scalar path above is byte-identical
   * to the B1 binary, so the existing axisymmetric wrapper is unaffected. */
  if (argc >= 24) {
    for (int i = 0; i < 3; i++) {
      par.par_P_plus[i] = atof(argv[12 + i]);
      par.par_P_minus[i] = atof(argv[15 + i]);
      par.par_S_plus[i] = atof(argv[18 + i]);
      par.par_S_minus[i] = atof(argv[21 + i]);
    }
  }

  /* solver / regularization knobs (NRPy defaults) */
  par.adm_tol = 1.0e-10;
  par.TP_epsilon = 1.0e-6;
  par.TP_Tiny = 0.0;
  par.TP_Extend_Radius = 0.0;
  par.verbose = true;
  par.keep_u_around = false;
  par.use_sources = false;
  par.rescale_sources = true;
  par.do_residuum_debug_output = false;
  par.do_initial_debug_output = false;
  par.multiply_old_lapse = false;
  par.solve_momentum_constraint = false;
  par.initial_lapse_psi_exponent = -2.0;
  snprintf(par.grid_setup_method, 100, "evaluation");
  snprintf(par.initial_lapse, 100, "psi^n");

  TP_solve(&par);

  const int nvar = 1, n1 = par.npoints_A, n2 = par.npoints_B, n3 = par.npoints_phi;
  printf("SUMMARY b=%.17g mp=%.17g mm=%.17g E=%.17g mp_adm=%.17g mm_adm=%.17g J1=%.17g J2=%.17g J3=%.17g\n",
         (double)par.par_b, (double)par.mp, (double)par.mm, (double)par.E,
         (double)par.mp_adm, (double)par.mm_adm, (double)par.J1, (double)par.J2, (double)par.J3);

  /* query points: "x y z" per line on stdin -> u, psi */
  double x, y, z;
  char line[512];
  while (fgets(line, sizeof(line), stdin)) {
    if (line[0] == '#' || line[0] == '\n')
      continue;
    if (sscanf(line, "%lf %lf %lf", &x, &y, &z) != 3)
      continue;
    REAL u = PunctIntPolAtArbitPosition(par, 0, nvar, n1, n2, n3, par.v, x, y, z);
    REAL rp = sqrt((x - par.par_b) * (x - par.par_b) + y * y + z * z);
    REAL rm = sqrt((x + par.par_b) * (x + par.par_b) + y * y + z * z);
    REAL psi = 1.0 + par.par_m_plus / (2.0 * rp) + par.par_m_minus / (2.0 * rm) + u;
    printf("POINT %.17g %.17g %.17g %.17g %.17g\n", x, y, z, (double)u, (double)psi);
  }
  return 0;
}
