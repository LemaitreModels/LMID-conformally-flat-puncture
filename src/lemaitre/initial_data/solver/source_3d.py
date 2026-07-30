"""PARASOL-3D — non-axisymmetric Bowen–York source over the (A,B,φ) node cloud.

The 2-D source machinery (``source.py``) is already point-wise and fully 3-D:
``_A_single_tensor(x, x0, P_vec)`` and ``_A_single_spin_tensor(x, x0, S_vec)``
build the full 3×3 Bowen–York momentum / spin tensors at an arbitrary Cartesian
point with **arbitrary vector** momentum and spin.  The axisymmetric closed
forms (``A2_2c``, ``A2_spin_extra``) are the on-axis / aligned specialisation.

For genuinely non-axisymmetric data (a misaligned spin, an off-axis momentum
component) we **stop calling the closed form** and instead contract the
point-wise summed tensor

    Â^{ij} = Â_{P,A}^{ij} + Â_{P,B}^{ij} + Â_{S,A}^{ij} + Â_{S,B}^{ij}

at every node ``x = (ρ cosφ, ρ sinφ, z)``.  Each piece is transverse, so the
momentum constraint stays analytic (∂_jÂ^{ij}=0 — verified to machine precision
by autodiff in :func:`divergence_3d_autodiff`).  Only ``Â_{ij}Â^{ij}`` enters
the Hamiltonian (Lichnerowicz) solve.

The summed tensor agrees with the axisymmetric closed form ``A2_2c`` to ~1e-12
on the meridian (the on-axis-momentum specialisation), which is what makes the
axisymmetric-reduction gate (Nφ=1, on-axis P, zero spin) reproduce the frozen
2-D solver.

Standalone: numpy + jax + the frozen sibling ``source``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp


# --------------------------------------------------------------------------
# Vectorised single-puncture tensors (the point-wise builders, over a cloud)
# --------------------------------------------------------------------------
def _mom_tensor_vec(X, x0, Pv):
    """Bowen–York momentum tensor Â_P^{ij} over points X (Npts,3).

    ``Â_P^{ij} = (3/2r²)[P^i n^j + n^i P^j − (δ^{ij} − n^i n^j)(P·n)]``,
    ``n = (x − x0)/r``.  Returns (Npts, 3, 3).
    """
    X = np.asarray(X, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    Pv = np.asarray(Pv, dtype=float)
    d = X - x0
    r = np.linalg.norm(d, axis=1)
    n = d / r[:, None]
    Pn = n @ Pv                                            # (Npts,)
    PnT = Pv[None, :, None] * n[:, None, :] + n[:, :, None] * Pv[None, None, :]
    proj = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
    return (3.0 / (2.0 * r ** 2))[:, None, None] * (PnT - proj * Pn[:, None, None])


def _spin_tensor_vec(X, x0, Sv):
    """Bowen–York spin tensor Â_S^{ij} over points X (Npts,3).

    ``Â_S^{ij} = (3/r³)(v^i n^j + n^i v^j)``, ``v = S × n``, ``n = (x − x0)/r``.
    Returns (Npts, 3, 3).
    """
    X = np.asarray(X, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    Sv = np.asarray(Sv, dtype=float)
    d = X - x0
    r = np.linalg.norm(d, axis=1)
    n = d / r[:, None]
    v = np.cross(np.broadcast_to(Sv, n.shape), n)          # (Npts,3)
    vn = v[:, :, None] * n[:, None, :] + n[:, :, None] * v[:, None, :]
    return (3.0 / r ** 3)[:, None, None] * vn


def A_full_tensor_vec(X, b, P_A_vec, P_B_vec, S_A_vec, S_B_vec):
    """Summed BY tensor Â^{ij} (momentum + spin) over points X (Npts,3).

    Punctures A at +b, B at −b.  Returns (Npts, 3, 3).
    """
    xA = np.array([0.0, 0.0, b])
    xB = np.array([0.0, 0.0, -b])
    return (_mom_tensor_vec(X, xA, P_A_vec) + _mom_tensor_vec(X, xB, P_B_vec)
            + _spin_tensor_vec(X, xA, S_A_vec) + _spin_tensor_vec(X, xB, S_B_vec))


# --------------------------------------------------------------------------
# Â² over the full (A,B,φ) node set
# --------------------------------------------------------------------------
def A2_at_nodes_3d(rho, z, phi, b, P_A_vec, P_B_vec, S_A_vec, S_B_vec):
    """Â_{ij}Â^{ij} of the summed BY tensor over the (A,B,φ) node cloud.

    ``rho, z`` are the flattened meridian node coordinates (Ntot2d,); ``phi``
    the φ-collocation nodes (Nφ,).  At each φ_k the Cartesian node is
    ``x = (ρ cosφ_k, ρ sinφ_k, z)``.  Returns ``A2`` of shape (Ntot2d, Nφ).

    Non-finite entries (the A=1 infinity edge, where ρ=∞) are returned as 0 —
    those rows are BC rows in the solve, with the source masked off.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    finite = np.isfinite(rho) & np.isfinite(z)
    rho_s = np.where(finite, rho, 1.0)
    z_s = np.where(finite, z, 0.0)
    Ntot, Nphi = rho.size, phi.size
    out = np.empty((Ntot, Nphi))
    for k in range(Nphi):
        X = np.stack([rho_s * np.cos(phi[k]), rho_s * np.sin(phi[k]), z_s], axis=1)
        T = A_full_tensor_vec(X, b, P_A_vec, P_B_vec, S_A_vec, S_B_vec)
        a2 = np.sum(T * T, axis=(1, 2))
        out[:, k] = np.where(finite & np.isfinite(a2), a2, 0.0)
    return out


# --------------------------------------------------------------------------
# Transversality oracle (autodiff; the momentum-constraint gate, test B)
# --------------------------------------------------------------------------
def _A_full_tensor_jax_vec(x_vec, b, P_A_vec, P_B_vec, S_A_vec, S_B_vec):
    """Summed BY tensor Â^{ij} as a jax (3,3) at a single point x (differentiable)."""
    x = jnp.asarray(x_vec, dtype=float)
    xA = jnp.array([0.0, 0.0, b])
    xB = jnp.array([0.0, 0.0, -b])

    def mom(x0, Pv):
        d = x - x0
        r = jnp.linalg.norm(d)
        n = d / r
        Pv = jnp.asarray(Pv, dtype=float)
        Pn = Pv @ n
        proj = jnp.eye(3) - jnp.outer(n, n)
        return (3.0 / (2.0 * r ** 2)) * (jnp.outer(Pv, n) + jnp.outer(n, Pv) - proj * Pn)

    def spin(x0, Sv):
        d = x - x0
        r = jnp.linalg.norm(d)
        n = d / r
        v = jnp.cross(jnp.asarray(Sv, dtype=float), n)
        return (3.0 / r ** 3) * (jnp.outer(v, n) + jnp.outer(n, v))

    return (mom(xA, P_A_vec) + mom(xB, P_B_vec)
            + spin(xA, S_A_vec) + spin(xB, S_B_vec))


def divergence_3d_autodiff(x_vec, b, P_A_vec, P_B_vec, S_A_vec, S_B_vec):
    """∂_j Â^{ij} of the full summed BY tensor via jax autodiff (EXACT).

    Returns the 3-vector; ~machine zero (transverse) for any vector momenta/spins,
    since each Bowen–York piece is transverse.  The momentum-constraint gate.
    """
    jac = jax.jacfwd(lambda x: _A_full_tensor_jax_vec(
        x, b, P_A_vec, P_B_vec, S_A_vec, S_B_vec))(jnp.asarray(x_vec, dtype=float))
    # jac[i,j,k] = ∂_k Â^{ij};  divergence^i = Σ_j ∂_j Â^{ij} = Σ_j jac[i,j,j]
    return np.array(jnp.einsum("ijj->i", jac))
