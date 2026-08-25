"""LM-initial-data-3D — the per-mode block as a Kronecker sum, and its exact inverse.

``operators_3d`` builds each azimuthal-mode block as a dense ``(Na·Nb)²`` matrix,
and ``solver_3d_nk`` LU-factors it once per Newton step.  Neither is necessary:
**every per-m block is an exact generalized Kronecker sum of a 1-D A-operator and
a 1-D B-operator**, so it can be applied with two small matmuls and inverted
exactly by two one-dimensional eigendecompositions — the Fast Diagonalization
Method (Lynch, Rice & Thomas, *Numer. Math.* **6**, 185 (1964)).

Why the block separates
-----------------------
``operators_abt._coeffs`` returns all four prolate Laplacian coefficients as a
common row factor times a function of ``A`` alone or ``B`` alone.  With
``oa2 = 1−A²``, ``Den = (1+A²)² − B² oa2²``::

    pref  = oa2²/(b² Den)          <- the ONLY factor that mixes A and B
    alpha/pref = oa2²/4            gamma/pref = 1−B²
    pcoef/pref = oa2²/(4A)         qcoef/pref = −2B

so dividing each interior row by ``pref`` — an exact rescaling of the equations,
of the same kind as the row equilibration the solver already performs, which
therefore leaves the solution unchanged — removes all A–B mixing from the
derivative terms.

The centrifugal term ``−m²/ρ²`` looks two-dimensional and is not.  In the
underlying prolate-spheroidal coordinates ``ξ = (1+A²)/(1−A²)``, ``η = B``, one
has ``Den = oa2²(ξ²−η²)`` and ``ρ² = b²(ξ²−1)(1−η²)``, so::

    −m²/ρ² / pref = −m² (ξ²−η²) / [(ξ²−1)(1−η²)]
                  = −m² [(ξ²−1) + (1−η²)] / [(ξ²−1)(1−η²)]
                  = −m² [ 1/(1−η²) + 1/(ξ²−1) ]
                  = −m² [ 1/(1−B²)  + oa2²/(4A²) ] ,

using ``ξ²−η² = (ξ²−1) + (1−η²)`` and ``ξ²−1 = 4A²/oa2²``.  That identity is what
makes the prolate-spheroidal Laplacian separable in the first place — it is why
prolate spheroidal harmonics exist — and it splits the centrifugal coefficient
into a pure-A piece and a pure-B piece.  It is an identity, not an approximation.

With the associated-Legendre factoring ``u_m = W v_m``, ``W = (1−B²)^{|m|/2}`` (a
function of B alone, see ``operators_3d.bc_factor``), the whole interior block is
therefore, exactly::

    diag(1/pref) · Mv = kron(L_A^m, diag(W)) + kron(I, C_B^m)                (1)

with ``L_A^m`` of size (Na+1)² and ``C_B^m`` of size Nb².  Kronecker rank 2, for
every m.  Regime of validity: this is read off the FLAT prolate Laplacian of
``operators_abt._coeffs``.  It does not automatically carry over to a
non-conformally-flat operator — a successor with a different meridian operator
must re-derive (1) rather than assume it.

Why that gives an exact inverse
-------------------------------
Left-multiplying (1) by ``kron(I, W^-1)`` turns it into a true Kronecker SUM::

    kron(L_A^m, I) + kron(I, W^-1 C_B^m)

whose action on the array form ``V`` (shape (Na+1, Nb)) is the Sylvester operator
``L_A V + V (W^-1 C_B)^T``.  Diagonalising each 1-D factor once,
``L_A = S_A Λ_A S_A^-1`` and ``(W^-1 C_B)^T = S_B Λ_B S_B^-1``, turns the solve
into a division by ``λ_A^i + λ_B^j``: ``O(Na·Nb·(Na+Nb))`` per apply after an
``O(Na³+Nb³)`` setup, storing ``O(Na²+Nb²)`` numbers instead of ``O((Na·Nb)²)``.

The two A-direction boundary rows are A-only (Dirichlet at A=1; Neumann at A=0
for m=0, Dirichlet otherwise), so they are imposed inside the 1-D A-operator and
removed by exact static condensation — which is what preserves the Kronecker
form.  Nothing here is an approximation of the operator; ``apply`` reproduces the
dense block to roundoff and ``solve`` inverts it as well as float64 allows on a
matrix whose raw condition number is ~1e13–1e15.

Why the setup is done once for the whole model
----------------------------------------------
Every interior coefficient carries exactly one factor ``1/b²`` (through ``pref``,
and ``−m²/ρ²`` likewise) and the BC rows are b-free, so ``M0_m(b) = D(b)·M0_m(1)``
with ``D`` diagonal — a left scaling that row equilibration cancels.  The 1-D
factors here are built at ``b=1`` and the separation enters only as a scalar row
factor at apply time.  So the eigendecompositions are computed **once per grid**
and reused at every separation, every parameter point and every query — and,
because they depend on neither ``b`` nor the physical parameters θ nor the
iterate, they sit entirely outside the differentiation path of
``applications/sensitivity_3d``: no gradient is ever taken through ``eig``.

Standalone: numpy + the sibling modules (``operators_3d``, ``operators_abt``).
"""

from __future__ import annotations

import collections

import numpy as np

from . import operators_3d as ops3
from . import operators_abt as ops


# How many separations a SeparableModes memoizes its O(Na·Nb) per-b arrays for.
# Small on purpose: recomputing them is cheap, and a corpus sweep visits
# thousands of separations.
_PER_B_MEMO_MAX = 8


# --------------------------------------------------------------------------
# The row factor, and the two 1-D factors of equation (1)
# --------------------------------------------------------------------------
def pref_rows(A, B, b: float) -> np.ndarray:
    """``pref = (1−A²)²/(b² Den)`` at the flattened nodes, shaped (Na+1, Nb).

    The common row factor of ``operators_abt._coeffs``, reproduced here with the
    same edge guard: the A=0 and A=1 rows are BC rows, so ``_coeffs`` evaluates
    them at a dummy ``A=1/2`` and zeroes the coefficients.  The value on those
    rows is never used (they are replaced wholesale), but it must be finite.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    edge = (A <= 1e-14) | (A >= 1.0 - 1e-14)
    Asafe = np.where(edge, 0.5, A)
    oa2 = 1.0 - Asafe ** 2
    Den = (1.0 + Asafe ** 2)[:, None] ** 2 - (B ** 2)[None, :] * (oa2 ** 2)[:, None]
    return (oa2 ** 2)[:, None] / (b ** 2 * Den)


def kron_factors(A, B, DA1, DB1, m: int):
    """The pair ``(L_A^m, C_B^m)`` of equation (1), plus the B-factor ``W``.

    ``L_A^m`` is (Na+1)², ``C_B^m`` is Nb², ``W`` is (Nb,).  Both matrices are
    independent of the separation ``b`` — that is the whole point (see the module
    docstring) — and of every physical parameter.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    DA1 = np.asarray(DA1, dtype=float)
    DB1 = np.asarray(DB1, dtype=float)
    m = abs(int(m))

    # --- A direction.  Guard A=0 and A=1 exactly as _coeffs does; those rows are
    # replaced by boundary conditions below, so the dummy value never survives.
    edge = (A <= 1e-14) | (A >= 1.0 - 1e-14)
    Asafe = np.where(edge, 0.5, A)
    oa2 = 1.0 - Asafe ** 2
    L_A = (oa2 ** 2 / 4.0)[:, None] * (DA1 @ DA1) + (oa2 ** 2 / (4.0 * Asafe))[:, None] * DA1

    # --- B direction.  The associated-Legendre factoring u_m = W v_m, expanded
    # analytically in W and numerically in v (operators_3d.bc_factor).
    W, Wp, Wpp = ops3.bc_factor(B, m)
    ob2 = 1.0 - B ** 2
    C_B = (ob2[:, None] * (np.diag(Wpp) + 2.0 * Wp[:, None] * DB1 + W[:, None] * (DB1 @ DB1))
           + (-2.0 * B)[:, None] * (np.diag(Wp) + W[:, None] * DB1))

    # --- centrifugal: exactly separable, so it folds into the two factors above
    if m:
        L_A = L_A - (m ** 2) * np.diag(oa2 ** 2 / (4.0 * Asafe ** 2))
        C_B = C_B - (m ** 2) * np.diag(W / ob2)
    return L_A, C_B, W


# --------------------------------------------------------------------------
# The fast-diagonalization inverse of one mode block
# --------------------------------------------------------------------------
class _FastDiagonalization:
    """Exact inverse of one BC-applied per-m block, by two 1-D eigendecompositions.

    Built at ``b=1``; the separation enters ``solve`` as a scalar row factor.
    """

    def __init__(self, L_A, C_B, W, DA1, m: int, pref1: np.ndarray):
        n = L_A.shape[0]
        self.i_bc = np.array([0, n - 1])
        self.i_in = np.arange(1, n - 1)

        # --- impose the two A-only boundary rows inside the 1-D A operator ---
        L = np.array(L_A, dtype=float)
        L[0, :] = 0.0
        L[0, 0] = 1.0                                   # A=1 (infinity): Dirichlet
        if m == 0:
            L[-1, :] = np.asarray(DA1, dtype=float)[-1, :]   # A=0: Neumann
        else:
            L[-1, :] = 0.0
            L[-1, -1] = 1.0                             # A=0: Dirichlet for m≠0

        # --- exact static condensation of those two rows ---
        # They are A-only, so in the array form they read  L_bb V_b + L_bi V_i = F_b
        # with no B coupling at all; eliminating V_b leaves a Sylvester equation in
        # V_i alone.  This is what preserves the Kronecker structure — a boundary
        # condition that mixed A and B would destroy it.
        Lbb = L[np.ix_(self.i_bc, self.i_bc)]
        Lbi = L[np.ix_(self.i_bc, self.i_in)]
        Lib = L[np.ix_(self.i_in, self.i_bc)]
        Lii = L[np.ix_(self.i_in, self.i_in)]
        self.Lbb_inv = np.linalg.inv(Lbb)
        self.Kb = self.Lbb_inv @ Lbi                    # V_b = Lbb⁻¹ F_b − Kb V_i
        self.Lib_Lbbinv = Lib @ self.Lbb_inv
        L_red = Lii - self.Lib_Lbbinv @ Lbi

        # --- the two 1-D eigendecompositions (the entire setup cost) ---
        # Both operators are NON-symmetric (first-derivative terms), so these are
        # general eigendecompositions and the eigenvector matrices are not
        # orthogonal.  cond(S_A), cond(S_B) are recorded: they bound how much the
        # double diagonalization can amplify roundoff, and cond(S_B) grows with m.
        lamA, SA = np.linalg.eig(L_red)
        lamB, SB = np.linalg.eig((C_B / W[:, None]).T)   # (W⁻¹ C_B)ᵀ
        # NumPy 2 returns complex from eig unconditionally; these spectra are
        # real (imaginary parts EXACTLY zero — checked, not assumed), so the
        # factors are cast back to real.  The cast alone would be a REGRESSION:
        # the live operand is complex for every generic mode, and a real matrix
        # times a complex vector makes NumPy promote the matrix on every apply.
        # ``solve`` therefore pairs the cast with a re/im split — see there.
        cast = ops3._real_part_if_real
        reals = [cast(x) for x in (lamA, SA, lamB, SB)]
        self._real_factors = all(x is not None for x in reals)
        if self._real_factors:
            lamA, SA, lamB, SB = reals
        self.SA, self.SAi = SA, np.linalg.inv(SA)
        self.SB, self.SBi = SB, np.linalg.inv(SB)
        self.den = lamA[:, None] + lamB[None, :]
        self.cond = (float(np.linalg.cond(SA)), float(np.linalg.cond(SB)))

        self.W = W
        self.pref_in = pref1[self.i_in]                 # pref at b=1, interior rows
        self.n = n

    def solve(self, Y: np.ndarray, b: float) -> np.ndarray:
        """Solve ``M0_m X = Y`` for one mode.  ``Y`` is (Na+1, Nb), real or complex.

        With real factors (the generic case — see ``__init__``) the dtype of
        ``Y`` picks the route, mirroring ``solver_3d_nk._lu_solve_equilibrated``
        on the dense side: an exactly-real ``Y`` — m=0 always, the Nyquist mode
        for even Nφ — runs the one pure-real pass, and a genuinely complex ``Y``
        is solved as ``solve(Re Y) + i·solve(Im Y)``, exact because every factor
        is real.  Casting the factors WITHOUT this split is a measured
        regression: NumPy would promote them back to complex on every apply.
        Measured at (44,32,8), all 5 modes, per preconditioner apply: complex
        factors as-is 0.34 ms → split 0.20 ms; the two routes agree to ~1e-14
        relative (different BLAS accumulation order), so GMRES trajectories may
        shift by ulps — the separable same-answer gates pin the outcome.
        """
        Y = np.asarray(Y)
        if self._real_factors and np.iscomplexobj(Y):
            yr = ops3._real_part_if_real(Y)
            if yr is not None:
                return self._solve_one(yr, b).astype(complex)
            return self._solve_one(Y.real, b) + 1j * self._solve_one(Y.imag, b)
        return self._solve_one(Y, b)

    def _solve_one(self, Y: np.ndarray, b: float) -> np.ndarray:
        """One pass of the fast-diagonalization solve, dtype following ``Y``."""
        rhs_b = Y[self.i_bc]                            # BC rows carry no pref factor

        # Interior rows: undo the row factor pref = pref1/b², then divide by W,
        # which turns kron(L_A, W) + kron(I, C_B) into the Kronecker SUM
        # kron(L_A, I) + kron(I, W⁻¹C_B).  The static-condensation correction
        # −L_ib L_bb⁻¹ F_b enters AFTER that division: it comes from eliminating
        # V_b out of the already-W-divided interior equation, so dividing it by W
        # as well would be wrong for every m≠0 (and invisible at m=0, where W=1).
        F = (Y[self.i_in] * (b ** 2) / self.pref_in) / self.W[None, :]
        F = F - self.Lib_Lbbinv @ rhs_b

        # Sylvester solve  L_red V + V (W⁻¹C_B)ᵀ = F  by double diagonalization
        V_i = self.SA @ ((self.SAi @ F @ self.SB) / self.den) @ self.SBi
        out = np.empty((self.n, Y.shape[1]), dtype=V_i.dtype)
        out[self.i_in] = V_i
        out[self.i_bc] = self.Lbb_inv @ rhs_b - self.Kb @ V_i
        return out


# --------------------------------------------------------------------------
# The per-grid object: factors + inverses for the whole mode set
# --------------------------------------------------------------------------
class SeparableModes:
    """Matrix-free per-m operator and exact inverse for one ``(Na, Nb, Nφ)`` grid.

    Depends on the grid and on nothing else — not on ``b``, the masses, the
    momenta, the spins or the iterate — so one instance serves every node of a
    corpus and every certified query.  Build it through :func:`get_separable`,
    which caches per grid.

    Storage is ``O(Na²+Nb²)`` per mode against the dense route's ``O((Na·Nb)²)``:
    at the production grid (44, 32, 8) that is ~0.1 MiB against ~79 MiB.
    """

    def __init__(self, Na: int, Nb: int, Nphi: int):
        A, B, DA1, DB1 = ops.build_grid(Na, Nb)
        self.Na, self.Nb, self.Nphi = Na, Nb, Nphi
        self.A, self.B, self.DA1, self.DB1 = A, B, DA1, DB1
        self.Na1, self.Nb1 = A.size, B.size
        self.m_vals = ops3.fourier_modes(Nphi)
        self.pref1 = pref_rows(A, B, 1.0)               # (Na+1, Nb), b factored out

        self.L_A, self.C_B, self.W, self.fdm = [], [], [], []
        for m in self.m_vals:
            L_A, C_B, W = kron_factors(A, B, DA1, DB1, int(m))
            self.L_A.append(L_A)
            self.C_B.append(C_B)
            self.W.append(W)
            self.fdm.append(_FastDiagonalization(L_A, C_B, W, DA1, int(m), self.pref1))

        # Grid-only pieces of a separable ASSEMBLY, shared by every slice built
        # on this grid: the per-mode node-array B-factor (u_m = w · v_m, the same
        # values the dense route's mode_operators returns) and the PDE-row mask.
        # ``solver_3d.assemble(separable=True)`` was rebuilding both per slice
        # although neither depends on b, θ or the iterate.  Read-only because the
        # arrays are shared across assemblies.
        Bf = np.meshgrid(A, B, indexing="ij")[1].ravel()
        self.w_nodes = []
        for m in self.m_vals:
            w = ops3.bc_factor(Bf, int(m))[0]
            w.setflags(write=False)
            self.w_nodes.append(w)
        interior = np.ones(self.Na1 * self.Nb1, dtype=bool)
        interior[:self.Nb1] = False                 # A=1 (infinity) BC rows
        interior[-self.Nb1:] = False                # A=0 (inner axis) BC rows
        interior.setflags(write=False)
        self.interior = interior
        # small per-separation memos; the objects are O(Na·Nb), not O((Na·Nb)²),
        # but a sweep visits many separations so they are bounded rather than open
        self._scales = collections.OrderedDict()
        self._scales_u = collections.OrderedDict()
        self._pref = collections.OrderedDict()

    def _pref_at(self, b: float) -> np.ndarray:
        key = float(b)
        hit = self._pref.get(key)
        if hit is None:
            hit = self.pref1 / b ** 2
            self._pref[key] = hit
            while len(self._pref) > _PER_B_MEMO_MAX:
                self._pref.popitem(last=False)
        return hit

    # -- the exact operator action, matrix free -----------------------------
    def apply(self, mi: int, V: np.ndarray, b: float) -> np.ndarray:
        """``M0_m @ vec(V)`` in array form, reproducing the dense block exactly.

        ``V`` is (Na+1, Nb), real or complex.  Interior rows come from equation
        (1); the two BC row-blocks are written out in closed form, exactly as
        ``operators_3d.apply_bcs_m`` sets them.
        """
        m = int(self.m_vals[mi])
        out = (self.L_A[mi] @ V) * self.W[mi][None, :] + V @ self.C_B[mi].T
        out = out * self._pref_at(b)
        out[0, :] = V[0, :]                             # A=1 Dirichlet, every m
        out[-1, :] = (self.DA1[-1, :] @ V) if m == 0 else V[-1, :]
        return out

    # -- the exact inverse --------------------------------------------------
    def solve(self, mi: int, Y: np.ndarray, b: float) -> np.ndarray:
        """``M0_m⁻¹ @ vec(Y)`` in array form, by fast diagonalization."""
        return self.fdm[mi].solve(Y, b)

    # -- the row-equilibration scales, without forming a row -----------------
    def row_scales(self, b: float):
        """``[max|M0_m row|]`` per mode — the same numbers as
        ``solver_3d_nk._block_scales``, in ``O(Na·Nb)`` instead of ``O((Na·Nb)²)``.

        Row ``(i,j)`` of the interior block has entries ``pref_ij·L_A[i,k]·W_j`` at
        columns ``(k,j)`` and ``pref_ij·C_B[j,l]`` at columns ``(i,l)``; the two
        overlap only at ``(i,j)``, where they add.  So the row maximum is the
        largest of three quantities that are cheap to precompute per row and per
        column, and no ``(Na·Nb)²`` array is ever touched.
        """
        key = float(b)
        if key in self._scales:
            self._scales.move_to_end(key)
            return self._scales[key]
        pref = self._pref_at(b)
        scales = []
        for mi, m in enumerate(self.m_vals):
            L_A, C_B, W = self.L_A[mi], self.C_B[mi], self.W[mi]
            offA = np.max(np.abs(L_A - np.diag(np.diag(L_A))), axis=1)        # (Na+1,)
            offB = np.max(np.abs(C_B - np.diag(np.diag(C_B))), axis=1)        # (Nb,)
            both = np.abs(np.diag(L_A)[:, None] * W[None, :] + np.diag(C_B)[None, :])
            s = np.maximum(np.maximum(offA[:, None] * np.abs(W)[None, :],
                                      offB[None, :]), both) * pref
            # the two BC row-blocks, written out as apply_bcs_m sets them
            s[0, :] = 1.0                                                    # unit row
            s[-1, :] = (np.max(np.abs(self.DA1[-1, :])) if int(m) == 0 else 1.0)
            s = s.ravel()
            scales.append(np.where(s > 0.0, s, 1.0))
        self._scales[key] = scales
        while len(self._scales) > _PER_B_MEMO_MAX:
            self._scales.popitem(last=False)
        return scales


    # -- the same scales in the PHYSICAL u, still without forming a row -------
    def row_scales_u(self, b: float):
        """``[max_k |M0_m[j,k]|/w_k]`` per mode — the ``u``-space row scale.

        The companion of :meth:`row_scales`, and the reason it needs its own
        derivation rather than a division: the ``u`` norm divides **by column**,
        ``max_k |M0[j,k]|/w_k``, while a row scale is a max **over** columns — so
        ``scales_u`` is not ``scales`` divided by anything.  ``operators_3d.
        u_row_scales`` is the definition; this reproduces it in ``O(Na·Nb)``
        instead of ``O((Na·Nb)²)``, which is what lets the separable route (the
        default above ``Nφ = 1``) carry the norm at all — a separable assembly has
        ``M0 = None`` and no row to scan.

        **The derivation, from :meth:`apply`'s own structure.**  ``w`` comes from
        ``bc_factor(Bf, m)`` and so depends on the ``B`` index alone — asserted
        below, because the whole shortcut rests on it.  Writing a node index as
        ``(i, j)`` for ``(A_i, B_j)``, row ``(i,j)`` of the interior block has
        entries ``pref_ij·L_A[i,k]·W_j`` at columns ``(k,j)`` and ``pref_ij·C_B[j,l]``
        at columns ``(i,l)``, overlapping only at ``(i,j)`` where they add.  The
        two families sit at different ``B`` indices, so they divide by different
        ``w``:

        * columns ``(k,j)`` all share ``B_j``  -> one factor ``1/w_j``, and the row
          maximum over ``k != i`` is ``offA[i]·|W_j|/w_j`` as before;
        * columns ``(i,l)`` run over ``B_l``   -> the max over ``l != j`` must be
          taken of ``|C_B[j,l]|/w_l``, which is a **different vector** from
          ``offB``, not a rescaling of it;
        * the overlap at ``(i,j)`` divides by ``w_j``.

        The two BC row-blocks each have all their entries at a single ``B`` index
        (``apply`` sets ``out[0,:] = V[0,:]`` and ``out[-1,:] = DA1[-1,:] @ V`` or
        ``V[-1,:]``), so there they *are* the ``v`` scales over ``w_j``.  At ``m = 0``
        ``w ≡ 1`` and this returns :meth:`row_scales` exactly, which is what keeps
        the axisymmetric reduction independent of the norm.

        Gated against the dense definition by
        ``tests/test_solver_3d_fast.py::test_separable_row_scales_u`` — the only
        thing tying this shortcut to ``operators_3d.u_row_scales``, so keep it.
        """
        key = float(b)
        if key in self._scales_u:
            self._scales_u.move_to_end(key)
            return self._scales_u[key]
        pref = self._pref_at(b)
        scales_u = []
        for mi, m in enumerate(self.m_vals):
            L_A, C_B, W = self.L_A[mi], self.C_B[mi], self.W[mi]
            wR = np.asarray(self.w_nodes[mi]).reshape(self.Na1, self.Nb1)
            wB = wR[0, :]
            assert np.array_equal(wR, np.broadcast_to(wB, wR.shape)), (
                "row_scales_u assumes w depends on the B index alone (bc_factor(Bf, m)); "
                "it does not on this grid, so the O(Na·Nb) shortcut is invalid here")
            wB = np.where(np.abs(wB) > 0.0, np.abs(wB), 1.0)
            offA = np.max(np.abs(L_A - np.diag(np.diag(L_A))), axis=1)        # (Na+1,)
            CB_off = np.abs(C_B - np.diag(np.diag(C_B)))                      # (Nb, Nb)
            offB_u = np.max(CB_off / wB[None, :], axis=1)                     # (Nb,)
            both = np.abs(np.diag(L_A)[:, None] * W[None, :] + np.diag(C_B)[None, :])
            s = np.maximum(np.maximum(offA[:, None] * np.abs(W)[None, :] / wB[None, :],
                                      offB_u[None, :]), both / wB[None, :]) * pref
            # the two BC row-blocks: every entry of each sits at one B index, so
            # the v scale simply divides by w there
            s[0, :] = 1.0 / wB
            s[-1, :] = (np.max(np.abs(self.DA1[-1, :])) if int(m) == 0 else 1.0) / wB
            s = s.ravel()
            scales_u.append(np.where(s > 0.0, s, 1.0))
        self._scales_u[key] = scales_u
        while len(self._scales_u) > _PER_B_MEMO_MAX:
            self._scales_u.popitem(last=False)
        return scales_u


_SEPARABLE_CACHE = collections.OrderedDict()
_SEPARABLE_CACHE_MAX = 4      # ~0.1 MiB each at the production grid


def get_separable(Na: int, Nb: int, Nphi: int) -> SeparableModes:
    """The :class:`SeparableModes` for one grid, built at most once per process."""
    key = (int(Na), int(Nb), int(Nphi))
    hit = _SEPARABLE_CACHE.get(key)
    if hit is not None:
        _SEPARABLE_CACHE.move_to_end(key)
        return hit
    obj = SeparableModes(*key)
    _SEPARABLE_CACHE[key] = obj
    while len(_SEPARABLE_CACHE) > _SEPARABLE_CACHE_MAX:
        _SEPARABLE_CACHE.popitem(last=False)
    return obj


def clear_separable_cache():
    """Drop every cached :class:`SeparableModes`."""
    _SEPARABLE_CACHE.clear()
