"""LM-initial-data — axisymmetric flat Laplacian (§3.2 of plan.md).

The axisymmetric flat Laplacian is *diagonal in the Legendre degree* ell:

    (Delta u)_ell = u_ell'' + (2/r) u_ell' - ell(ell+1)/r^2 * u_ell .

We build one (N+1)x(N+1) radial operator A_ell per even mode, impose the two
endpoint boundary conditions by *row replacement* (so the singular 1/r, 1/r^2
entries at r=0 are overwritten and never evaluated), and stack the per-mode
operators into a block-diagonal full Laplacian for the modal unknown ordering
``vec(U)[a*(N+1) + k] = u_{ell_a}(r_k)``.

BCs (plan §3.2):
  * r->inf (index 0): Dirichlet  u_ell = 0.
  * r=0  (index N):   ell=0 -> Neumann u_0'(0)=0 (the Dr row);
                      ell>=2 -> Dirichlet u_ell(0)=0  (from u_ell ~ r^ell).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp


def radial_operator(ell: int, r, Dr, Dr2):
    """Per-mode radial Laplacian A_ell with both endpoints as BC rows.

    ``r, Dr, Dr2`` are the outputs of ``spectral.radial_grid``.  Returns a
    numpy (N+1)x(N+1) matrix (caller stacks/converts).
    """
    r = np.asarray(r)
    Dr = np.asarray(Dr)
    Dr2 = np.asarray(Dr2)
    Np1 = r.shape[0]
    N = Np1 - 1

    # Safe inverse powers of r: 1/inf -> 0 automatically; force r=0 row -> 0
    # (it is a BC row, overwritten below) so no inf appears in construction.
    with np.errstate(divide="ignore"):
        invr = np.where(r > 0, 1.0 / r, 0.0)
    invr = np.where(np.isfinite(invr), invr, 0.0)     # kills 1/inf=0 cleanly
    invr[N] = 0.0                                      # r=0 BC row placeholder
    invr2 = invr ** 2

    A = Dr2 + 2.0 * invr[:, None] * Dr - ell * (ell + 1) * np.diag(invr2)

    # --- boundary rows (row replacement) ---
    # r -> inf  (index 0): Dirichlet u = 0
    A[0, :] = 0.0
    A[0, 0] = 1.0
    # r = 0  (index N)
    if ell == 0:
        A[N, :] = Dr[N, :]          # Neumann: u_0'(0) = 0
    else:
        A[N, :] = 0.0               # Dirichlet: u_ell(0) = 0
        A[N, N] = 1.0
    return A


def build_laplacian(ells, r, Dr, Dr2):
    """Build per-mode operators and the block-diagonal full Laplacian.

    Returns ``(A_list, Lap_const)``:
      * A_list: list of jax (N+1)x(N+1) per-mode operators (index a <-> ells[a]),
      * Lap_const: jax ((L_theta*(N+1)) x (L_theta*(N+1))) block-diagonal matrix
        for the modal ordering vec(U)[a*(N+1)+k].
    Constant across the Newton iteration (depends only on grid + ells).
    """
    A_blocks = [radial_operator(int(ell), r, Dr, Dr2) for ell in ells]
    Lap_const = _block_diag(A_blocks)
    A_list = [jnp.asarray(Ab) for Ab in A_blocks]
    return A_list, jnp.asarray(Lap_const)


def _block_diag(blocks):
    """Block-diagonal assembly (numpy; one-time construction)."""
    sizes = [b.shape[0] for b in blocks]
    M = int(sum(sizes))
    out = np.zeros((M, M))
    off = 0
    for b, s in zip(blocks, sizes):
        out[off:off + s, off:off + s] = b
        off += s
    return out
