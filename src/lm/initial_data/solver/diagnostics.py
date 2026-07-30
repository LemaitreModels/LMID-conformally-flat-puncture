"""PARASOL-2C — diagnostics (adapted from the single-centre §6).

Field evaluation, residual norm, ADM mass, convergence-table printer.  On the
Stage-1 single A-centred grid these reuse the single-centre machinery verbatim
(modal radial functions about A × Legendre-in-μ_A synthesis); only the
solver-coupled ``residual_norm``/``adm_mass`` take the two-centre ``Slice``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from . import spectral


# --------------------------------------------------------------------------
# Field evaluation at arbitrary (r_A, mu_A)
# --------------------------------------------------------------------------
def x_of_r(r, L):
    """Inverse algebraic map: x = (r - L)/(r + L)  (r=0->-1, r=inf->+1)."""
    return (r - L) / (r + L)


def evaluate_field(prob, U, r_eval, mu_eval):
    """u(r_A, μ_A) on the grid r_eval (Nr,) x mu_eval (Nm,) -> (Nr, Nm)."""
    r_eval = np.asarray(r_eval, dtype=float).reshape(-1)
    mu_eval = np.asarray(mu_eval, dtype=float).reshape(-1)
    U = np.asarray(U)

    x_nodes = np.asarray(prob.x)
    w = spectral.cgl_bary_weights(prob.N)
    xe = x_of_r(r_eval, prob.L)
    B = spectral.bary_interp_matrix(xe, x_nodes, w)      # (Nr, N+1)

    u_a_eval = U @ B.T                                   # (L_theta, Nr)
    P = np.stack([spectral.legendre_P_eval(int(ell), mu_eval)
                  for ell in prob.ells])                 # (L_theta, Nm)
    return np.einsum("ar,am->rm", u_a_eval, P)           # (Nr, Nm)


def evaluate_field_points(prob, U, r_pts, mu_pts):
    """u at matched (r_pts[i], mu_pts[i]) points -> (Npts,)  (for heatmaps)."""
    r_pts = np.asarray(r_pts, dtype=float).reshape(-1)
    mu_pts = np.asarray(mu_pts, dtype=float).reshape(-1)
    U = np.asarray(U)
    x_nodes = np.asarray(prob.x)
    w = spectral.cgl_bary_weights(prob.N)
    B = spectral.bary_interp_matrix(x_of_r(r_pts, prob.L), x_nodes, w)
    u_a = U @ B.T
    P = np.stack([spectral.legendre_P_eval(int(ell), mu_pts)
                  for ell in prob.ells])
    return np.einsum("ap,ap->p", u_a, P)


# --------------------------------------------------------------------------
# Scalar observables
# --------------------------------------------------------------------------
def residual_norm(prob, U, sl) -> float:
    """||R||_inf for the (converged) two-centre field at slice ``sl``."""
    from . import solver
    R = solver.residual(prob, jnp.asarray(U), sl)
    return float(jnp.max(jnp.abs(R)))


def adm_mass(prob, U, sl, n_far: int = 4) -> float:
    """ADM mass  M_ADM = (m_A + m_B) + 2 lim_{r_A->inf} r_A u_0(r_A).

    The ℓ=0 mode u_0 carries the 1/r_A monopole tail; at large r_A both
    punctures and u look like a single point source of mass M_ADM.  c is the
    intercept of a linear fit r_A u_0 = c + d/r_A over the farthest nodes.
    """
    U = np.asarray(U)
    r = np.asarray(prob.r)
    finite = (r > 0) & np.isfinite(r)
    u0 = U[0]
    rf, u0f = r[finite], u0[finite]
    order = np.argsort(rf)
    rf, u0f = rf[order], u0f[order]
    k = min(max(2, n_far), rf.size)
    rsub = rf[-k:]
    y = rsub * u0f[-k:]
    t = 1.0 / rsub
    tn = t / t.max()
    d, c = np.polyfit(tn, y, 1)
    return float(sl.M + 2.0 * c)


# --------------------------------------------------------------------------
# Convergence-table printer (verbatim from single-centre)
# --------------------------------------------------------------------------
def convergence_table(rows, headers, title=None):
    lines = []
    if title:
        lines.append(title)
    widths = [max(len(str(h)), max((len(_fmt(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    lines.append("  ".join(str(h).rjust(widths[i]) for i, h in enumerate(headers)))
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        lines.append("  ".join(_fmt(r[i]).rjust(widths[i]) for i in range(len(r))))
    s = "\n".join(lines)
    print(s)
    return s


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) or isinstance(v, np.floating):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.3e}"
        return f"{v:.6f}"
    return str(v)
