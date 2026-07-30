"""PARASOL — POD (reduced-basis) re-encoding of the gradient-enhanced Hermite
surrogate + the Smolyak-compatibility decision (H3).

The H3 milestone of ``GRADIENT_ENHANCED_PLAN.md`` §4.  Two pieces.

POD (the main deliverable).
---------------------------
The committed value-only reduced-basis re-encoding (``experiments/ml/pod_surrogate.py``,
paper Sec.~\\ref{sec:param:pod}/\\ref{sec:model:pod}) compresses a corpus of
solved fields ``U_i(x)`` by proper orthogonal decomposition: an SVD of the stacked
(mean-subtracted) fields gives orthonormal spatial modes ``Φ``; keep the leading
``r`` and store the length-``r`` coefficient vector per node instead of the full
``nfeat``-value field.  Because barycentric/Hermite interpolation is *linear*,
interpolating the coefficients then decoding ``u = mean + Φ·c`` is **exactly** the
rank-``r`` truncation of the full interpolant (they differ only by the truncation
tail), and the certified polish reconstructs the discarded tail — the compression
is lossless in the only sense that matters.

The gradient-enhanced (Hermite) surrogate (H2, :mod:`hermite_nd`) carries, per node,
the value ``U_i`` **and** the certified parameter-tangents ``dU/dθ_k|_i`` (one field
per enhanced axis) — the R5 storage cost is ``(1 + n_enhanced)×`` the value-only
footprint.  This module's finding (the R5 mitigation): the derivative fields
``dU/dθ_k`` project onto **essentially the same low-rank spatial basis as** ``U``.
Concretely, the SVD of the *stacked value+derivative* corpus has a rank that barely
grows over the value-only corpus (the ``θ→field`` map is analytic, so its parameter
tangent lives in the same solution-manifold tangent space).  So we build ONE basis
``Φ`` (from the stacked corpus), project both ``U`` and every ``dU/dθ_k`` onto it,
and store ``(1 + n_enhanced)`` length-``r`` coefficient vectors per node — the value
storage compresses by ``nfeat/r`` and the derivative storage compresses by the SAME
factor, so the gradient enhancement stays cheap.

The reduced-basis Hermite (:class:`PODHermiteND`) interpolates the coefficient
vectors with the identical H2 :class:`hermite_nd.HermiteSolutionND` machinery (reused
**verbatim** in coefficient space) and decodes with ``Φ``.  By linearity it is
bit-identical to the POD projection ``mean + ΦΦᵀ(full_hermite − mean)`` of the full
Hermite interpolant, so the H2 node-exactness/reduce-to-committed properties and the
certified polish all carry over; the exposed parameter gradient of the compressed
model is the full Hermite gradient projected onto ``Φ`` (``P_r·∂U/∂θ``), preserved to
the truncation tail.

Smolyak compatibility (the decision).
-------------------------------------
**Decision: (b) — the gradient (Hermite) enhancement targets the dense/anisotropic
path only.**  The *value-only* interpolant already slots into the combination
technique bit-for-bit (a value-only :class:`hermite_nd.HermiteSolutionND` telescopes
identically to a :class:`parametric_nd_smolyak.SmolyakSolutionND` subgrid —
demonstrated in :func:`value_only_combination` and its test), so the interpolant
*family* is Smolyak-compatible.  The **enhancement** is dense-only, for two
code-grounded reasons:

  1. **No certified tangent exists for the Smolyak-wrapped solver.**  The gradient
     enhancement consumes the certified implicit-function tangent ``dU/dθ_k``, which
     lives only in ``applications.sensitivity.certified_tangent`` over the
     **axisymmetric** ``solver_abt`` family ``(q,b,χ_A,χ_B)`` — the DENSE
     ``ParametricSolverND`` path.  The sparse builders
     (``SmolyakSolverND`` / ``from_problem_smolyak_3d``) wrap
     ``parametric_nd_3d`` (the 3-D quasi-circular non-axisymmetric solver), which has
     **no** certified tangent (locked scope; a ``solver_3d`` IFT tangent is
     future work) — and the committed Smolyak build loop stores only values
     (``pool[key] = (U, iters, resid)``), never a tangent.  So a gradient-enhanced
     Smolyak model is not constructible today.
  2. **The Smolyak level-0 factor degenerates to the fragile 1-node Taylor on the
     enhanced axis.**  A Smolyak subgrid carries the enhanced hard axis at level 0
     (a single midpoint node) in every subgrid that spends its levels elsewhere.
     A value-only level-0 factor is the constant ``U_0``; the *Hermite* level-0
     factor is the 1-node linear Taylor ``U_0 + (θ−θ_0)·dU_0`` — precisely the R4
     ``Taylor radius`` mode the plan demotes to a fallback/diagnostic (it collapses
     near the merger wall ``b→0``, exactly the enhanced hard axis).  Enhancing a
     level-0 factor injects that fragile predictor into the combination sum.
     (:func:`level0_enhanced_is_taylor` demonstrates the degeneracy.)

The two features attack orthogonal cost axes (paper: Smolyak the parameter
resolution / offline solve count, POD the spatial rank, gradient-enhancement the
per-node convergence rate on the dense anisotropic path) — so this is a clean design
boundary, not a limitation of the algebra.

**Add-only.**  Reuses :mod:`hermite_nd` (``HermiteSolutionND`` verbatim),
:mod:`parametric_nd` (persistence helpers), and
:mod:`parametric_nd_smolyak` (``nested_levels``/``isotropic_index_set``/
``combination_coeffs``/``SmolyakSolutionND`` — only for the value-only demonstration)
verbatim; never edits a committed module.  Certification is unchanged — the compressed
object is only a *guess*; ``evaluate_polished`` reuses the committed ``solve_fn``.

Standalone: numpy + jax + the sibling ``parametric``/``parametric_nd``/``hermite``/
``hermite_nd`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric_nd import (              # persistence helpers, reused verbatim
    FORMAT_VERSION,
    _pack_meta,
    _unpack_meta,
    _git_commit,
    _load_npz,
    _check_meta,
)
from .hermite import cardinal_deriv_at_nodes           # node-set primitive (verbatim)
from .hermite_nd import HermiteSolutionND               # the H2 interpolant (verbatim)


# --------------------------------------------------------------------------
# SVD / rank helpers (mirror experiments/ml/pod_phase1.py; self-contained)
# --------------------------------------------------------------------------
def randomized_svd(A, n_modes, n_oversample=20, n_iter=2, seed=0):
    """Top-``n_modes`` left singular vectors + singular values of ``A`` (``m×n``).

    Halko–Martinsson–Tropp randomized range finder with a couple of power
    iterations (the POD spectrum decays geometrically, so ``n_iter=2`` suffices).
    Mirrors ``experiments/ml/pod_phase1.py`` verbatim.
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    k = min(n_modes + n_oversample, n)
    Om = rng.standard_normal((n, k))
    Y = A @ Om
    Q, _ = np.linalg.qr(Y)
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(A @ (A.T @ Q))
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :n_modes], s[:n_modes], Vt[:n_modes]


def rank_for_tail(s, tail):
    """Smallest ``r`` with ``sqrt(sum_{i>=r} s_i^2)/sqrt(sum s_i^2) <= tail`` (rel-L2)."""
    s = np.asarray(s, dtype=float)
    e = s ** 2
    total = e.sum()
    if total == 0.0:
        return 0
    cum_tail = np.sqrt(np.cumsum(e[::-1])[::-1] / total)     # tail energy from mode i on
    tail_after_r = np.append(cum_tail[1:], 0.0)              # keep r modes → tail from r..end
    idx = np.where(tail_after_r <= tail)[0]
    return int(idx[0] + 1) if len(idx) else len(s)


# --------------------------------------------------------------------------
# POD basis from the stacked value + derivative corpus of a HermiteSolutionND
# --------------------------------------------------------------------------
def _flatten_corpus(her: HermiteSolutionND):
    """Return ``(X, D_list, N, nfeat, field_shape)`` for a HermiteSolutionND.

    ``X`` : ``(N, nfeat)`` node values.
    ``D_list`` : list over the ENHANCED axes of ``(N, nfeat)`` node tangents
        ``dU/dθ_k`` (the fields the interpolant actually consumes).
    """
    d = her.d
    field_shape = her.field_shape
    nfeat = int(np.prod(field_shape)) if field_shape else 1
    grid = her.U_nodes.shape[:d]
    N = int(np.prod(grid))
    X = np.asarray(her.U_nodes, dtype=float).reshape(N, nfeat)
    D_list = []
    for e in her.enhanced:
        De = np.take(np.asarray(her.dU_nodes, dtype=float), int(e), axis=d)  # (grid, *fs)
        D_list.append(De.reshape(N, nfeat))
    return X, D_list, N, nfeat, field_shape


def pod_basis(her: HermiteSolutionND, *, r: Optional[int] = None,
              tail: Optional[float] = None, include_derivatives: bool = True,
              randomized: bool = False, seed: int = 0):
    """POD spatial modes ``Φ`` from the **stacked value+derivative** corpus of a
    :class:`hermite_nd.HermiteSolutionND`.

    The value fields are mean-subtracted (reconstruction is ``mean + Φ·c``); the
    derivative fields are **not** centered (``∂θ`` of the constant mean is 0, so the
    decode of a tangent is ``Φ·c`` with no mean).  With ``include_derivatives=True``
    the SVD corpus is ``[ (U−mean)ᵀ | (dU/dθ_{e_1})ᵀ | … ]`` over the enhanced axes,
    so ``Φ`` is guaranteed to represent both ``U`` and every ``dU/dθ_k`` at the same
    rank — the R5 storage mitigation.

    Returns ``(Phi, mean, diag)`` where ``Phi`` is ``(nfeat, r)``, ``mean`` is
    ``(nfeat,)`` and ``diag`` is a dict of diagnostics: the stacked and value-only
    singular values (``s``, ``s_value``), the ranks at standard tails
    (``rank_stacked``, ``rank_value``), and the relative projection residual of the
    derivative fields onto the value-only rank-``r`` basis
    (``dU_on_value_basis_resid`` — the "derivatives share the value basis" number).

    Exactly one of ``r`` / ``tail`` selects the kept rank (``tail`` default 1e-6).
    """
    X, D_list, N, nfeat, field_shape = _flatten_corpus(her)
    mean = X.mean(axis=0)
    Xc = X - mean

    # value-only basis (for the "rank barely grows" diagnostic)
    Av = Xc.T                                             # (nfeat, N)
    if randomized:
        nm = min(max(N, 1), nfeat)
        Phi_v, s_v, _ = randomized_svd(Av, min(nm, N), seed=seed)
    else:
        Phi_v, s_v, _ = np.linalg.svd(Av, full_matrices=False)

    # stacked value+derivative basis (the shipped modes)
    cols = [Av]
    if include_derivatives:
        cols += [De.T for De in D_list]
    A = np.hstack(cols) if len(cols) > 1 else Av          # (nfeat, N*(1+n_enh))
    if randomized:
        nm = min(A.shape[1], nfeat)
        Phi, s, _ = randomized_svd(A, nm, seed=seed)
    else:
        Phi, s, _ = np.linalg.svd(A, full_matrices=False)

    # rank selection
    if r is None:
        tail = 1e-6 if tail is None else float(tail)
        r = rank_for_tail(s, tail)
    r = int(max(1, min(r, Phi.shape[1])))
    Phi_r = np.ascontiguousarray(Phi[:, :r])

    # diagnostics
    diag = {
        "s": s, "s_value": s_v,
        "rank_stacked": {t: rank_for_tail(s, t) for t in (1e-3, 1e-4, 1e-6, 1e-8)},
        "rank_value": {t: rank_for_tail(s_v, t) for t in (1e-3, 1e-4, 1e-6, 1e-8)},
        "r": r, "N": N, "nfeat": nfeat, "n_enhanced": len(D_list),
    }
    # projection residual of dU onto the value-only rank-r basis (share-the-basis)
    if D_list:
        Pv = Phi_v[:, :min(r, Phi_v.shape[1])]
        resid = []
        for De in D_list:
            Dt = De.T
            proj = Pv @ (Pv.T @ Dt)
            num = np.linalg.norm(Dt - proj)
            den = np.linalg.norm(Dt)
            resid.append(float(num / den) if den else 0.0)
        diag["dU_on_value_basis_resid"] = resid
    return Phi_r, mean, diag


# --------------------------------------------------------------------------
# The POD (reduced-basis) gradient-enhanced Hermite surrogate
# --------------------------------------------------------------------------
@dataclass
class PODHermiteND:
    """Reduced-basis (POD) re-encoding of a :class:`hermite_nd.HermiteSolutionND`.

    Interpolates the length-``r`` POD **coefficient** vectors (value + one per
    enhanced axis) with the identical H2 Hermite machinery — an internal
    ``HermiteSolutionND`` over the coeff space, reused **verbatim** — and decodes
    ``u = mean + Φ·c``.  By linearity of barycentric/Hermite interpolation this is
    bit-identical (to roundoff) to the POD projection of the full Hermite
    interpolant, so the node-exactness / reduce-to-committed properties and the
    certified polish carry over; the exposed parameter gradient is the full Hermite
    gradient projected onto ``Φ`` (preserved to the truncation tail).

    Fields
    ------
    coeff_hermite : the internal :class:`hermite_nd.HermiteSolutionND` over the
        ``r``-dim coefficients (``field_shape == (r,)``; same axes/nodes/enhanced).
    Phi : ``(nfeat, r)`` POD spatial modes.
    mean : ``(nfeat,)`` mean field.
    field_shape : the decoded field shape.
    """

    coeff_hermite: HermiteSolutionND
    Phi: np.ndarray
    mean: np.ndarray
    field_shape: Tuple[int, ...]
    _solve_fn: Callable = field(repr=False, default=None)

    def __post_init__(self):
        self.Phi = np.asarray(self.Phi, dtype=float)
        self.mean = np.asarray(self.mean, dtype=float)
        self.field_shape = tuple(int(x) for x in self.field_shape)
        self._Phi_j = jnp.asarray(self.Phi)
        self._mean_j = jnp.asarray(self.mean)

    @property
    def d(self) -> int:
        return self.coeff_hermite.d

    @property
    def r(self) -> int:
        return int(self.Phi.shape[1])

    @property
    def enhanced(self):
        return self.coeff_hermite.enhanced

    @property
    def n_nodes(self) -> int:
        return self.coeff_hermite.n_nodes

    # ----- decode: interpolate the coeffs, then u = mean + Φ·c -----
    def evaluate(self, theta):
        """``ũ(θ)`` decoded from the interpolated POD coefficients (numpy, node-safe)."""
        c = np.asarray(self.coeff_hermite.evaluate(theta)).reshape(-1)
        u = self.mean + self.Phi @ c
        return u.reshape(self.field_shape)

    def evaluate_jax(self, theta):
        """``jnp`` twin of :meth:`evaluate` — the exposed-gradient hook
        (``jax.jacfwd`` gives ``P_r·∂U/∂θ``, the full gradient projected onto ``Φ``).
        Must NOT be queried exactly at a node."""
        c = jnp.reshape(self.coeff_hermite.evaluate_jax(theta), (-1,))
        u = self._mean_j + self._Phi_j @ c
        return jnp.reshape(u, self.field_shape)

    # ----- certified evaluation (unchanged; decode → committed solve_fn) -----
    def evaluate_polished(self, theta, newton_steps: int = 2, tol: float = 1e-12):
        """POD-decoded Hermite guess + 1–2 Newton steps → certified ``‖R‖≤tol``.

        Certification is unchanged: the compressed object is only a *guess*; the
        attached ``solve_fn`` → ``newton_solve`` is the certificate."""
        if self._solve_fn is None:
            raise RuntimeError(
                "no solve_fn attached; build via project_hermite_pod with a solver-backed "
                "HermiteSolutionND, or reattach a solve_fn")
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(np.asarray(theta, dtype=float), guess, tol, newton_steps)
        return U, info

    # ----- persistence (numpy-only .npz) -----
    def save(self, path, *, meta=None, coeff_dtype=np.float64, mode_dtype=np.float64):
        """Persist to a single ``.npz`` (numpy-only, no pickle).  Round-trips
        bit-for-bit via :func:`load_pod_hermite_nd`.

        ``Φ`` and the coeff pool are only a *warm start* for the certified polish,
        so they need not be float64 (the paper O1 note): ``mode_dtype`` /
        ``coeff_dtype`` may be ``float32`` to halve the on-disk footprint (reload
        upcasts to float64, so ``evaluate`` still runs in float64).  Defaults
        preserve the float64 artifact byte-for-byte.
        """
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        ch = self.coeff_hermite
        d = ch.d
        arrays = {}
        for k in range(d):
            arrays[f"nodes_{k}"] = np.asarray(ch.nodes[k], dtype=float)
            arrays[f"weights_{k}"] = np.asarray(ch.weights[k], dtype=float)
            arrays[f"cvec_{k}"] = np.asarray(ch.cvec[k], dtype=float)
        arrays["U_coeff"] = np.asarray(ch.U_nodes).astype(coeff_dtype)
        arrays["dU_coeff"] = np.asarray(ch.dU_nodes).astype(coeff_dtype)
        arrays["Phi"] = np.asarray(self.Phi).astype(mode_dtype)
        arrays["mean"] = np.asarray(self.mean, dtype=float)
        arrays["iters"] = np.asarray(ch.iters, dtype=np.int64)
        arrays["residuals"] = np.asarray(ch.residuals, dtype=float)
        arrays["axes"] = np.array([[float(lo), float(hi), float(Q)]
                                   for (lo, hi, Q) in ch.axes], dtype=float)
        arrays["enhanced"] = np.asarray(sorted(int(e) for e in ch.enhanced), dtype=np.int64)
        arrays["field_shape"] = np.asarray(self.field_shape, dtype=np.int64)
        arrays["r"] = np.asarray(self.r, dtype=np.int64)
        full_meta = {"d": int(d), "r": int(self.r), "git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION
        full_meta["kind"] = "pod_hermite_nd"
        arrays["meta_json"] = _pack_meta(full_meta)
        np.savez(path, **arrays)
        return path


def project_hermite_pod(her: HermiteSolutionND, Phi: np.ndarray, mean: np.ndarray,
                        *, solve_fn: Optional[Callable] = None) -> PODHermiteND:
    """Project a solver-backed :class:`hermite_nd.HermiteSolutionND` onto the POD
    basis ``Φ`` (``mean`` the value-mean) → a :class:`PODHermiteND`.

    Stores, per node, the value coefficient ``(U−mean)·Φ`` and the tangent
    coefficient ``dU/dθ_k·Φ`` for every axis (the enhanced axes' tangents are what
    ``evaluate`` consumes; value-only axes' are projected too, harmlessly, so the
    object is a faithful projection of the full one)."""
    d = her.d
    field_shape = her.field_shape
    nfeat = int(np.prod(field_shape)) if field_shape else 1
    grid = her.U_nodes.shape[:d]
    N = int(np.prod(grid))
    Phi = np.asarray(Phi, dtype=float)
    mean = np.asarray(mean, dtype=float)
    r = Phi.shape[1]

    U_coeff = ((np.asarray(her.U_nodes, dtype=float).reshape(N, nfeat) - mean) @ Phi
               ).reshape(grid + (r,))
    dU_coeff = np.empty(grid + (d, r), dtype=float)
    for k in range(d):
        Dk = np.take(np.asarray(her.dU_nodes, dtype=float), k, axis=d).reshape(N, nfeat)
        dU_coeff[..., k, :] = (Dk @ Phi).reshape(grid + (r,))

    coeff_hermite = HermiteSolutionND(
        axes=list(her.axes), nodes=[np.asarray(n, dtype=float) for n in her.nodes],
        weights=[np.asarray(w, dtype=float) for w in her.weights],
        U_nodes=U_coeff, dU_nodes=dU_coeff,
        cvec=[np.asarray(c, dtype=float) for c in her.cvec],
        enhanced=tuple(her.enhanced),
        iters=np.asarray(her.iters), residuals=np.asarray(her.residuals),
        _solve_fn=None)
    sf = solve_fn if solve_fn is not None else getattr(her, "_solve_fn", None)
    return PODHermiteND(coeff_hermite=coeff_hermite, Phi=Phi, mean=mean,
                        field_shape=field_shape, _solve_fn=sf)


def build_pod_hermite(her: HermiteSolutionND, *, r: Optional[int] = None,
                      tail: Optional[float] = None, include_derivatives: bool = True,
                      randomized: bool = False, seed: int = 0,
                      solve_fn: Optional[Callable] = None):
    """Convenience: :func:`pod_basis` then :func:`project_hermite_pod`.

    Returns ``(pod, diag)`` — the :class:`PODHermiteND` and the ``pod_basis``
    diagnostics (rank tables, share-the-basis residuals)."""
    Phi, mean, diag = pod_basis(her, r=r, tail=tail,
                                include_derivatives=include_derivatives,
                                randomized=randomized, seed=seed)
    pod = project_hermite_pod(her, Phi, mean, solve_fn=solve_fn)
    return pod, diag


def load_pod_hermite_nd(path) -> "PODHermiteND":
    """Load a :class:`PODHermiteND` saved by :meth:`PODHermiteND.save`.

    Reconstructs with ``_solve_fn=None`` (``evaluate``/``evaluate_jax`` work
    immediately; ``evaluate_polished`` raises until a solver is attached).  Parsed
    metadata is stored on the returned object as ``.meta``.
    """
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "pod_hermite_nd")
    try:
        d = int(meta["d"])
        nodes = [np.asarray(data[f"nodes_{k}"], dtype=float) for k in range(d)]
        weights = [np.asarray(data[f"weights_{k}"], dtype=float) for k in range(d)]
        cvec = [np.asarray(data[f"cvec_{k}"], dtype=float) for k in range(d)]
        axes = [(float(a[0]), float(a[1]), int(round(float(a[2]))))
                for a in np.asarray(data["axes"], dtype=float)]
        enhanced = tuple(int(e) for e in np.asarray(data["enhanced"], dtype=np.int64))
        U_coeff = np.asarray(data["U_coeff"], dtype=float)
        dU_coeff = np.asarray(data["dU_coeff"], dtype=float)
        Phi = np.asarray(data["Phi"], dtype=float)
        mean = np.asarray(data["mean"], dtype=float)
        iters = np.asarray(data["iters"])
        residuals = np.asarray(data["residuals"], dtype=float)
        field_shape = tuple(int(x) for x in np.asarray(data["field_shape"], dtype=np.int64))
        coeff_hermite = HermiteSolutionND(
            axes=axes, nodes=nodes, weights=weights, U_nodes=U_coeff, dU_nodes=dU_coeff,
            cvec=cvec, enhanced=enhanced, iters=iters, residuals=residuals, _solve_fn=None)
        pod = PODHermiteND(coeff_hermite=coeff_hermite, Phi=Phi, mean=mean,
                           field_shape=field_shape, _solve_fn=None)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt PARASOL pod_hermite_nd surrogate '{path}': {e}")
    pod.meta = meta
    return pod


# --------------------------------------------------------------------------
# Smolyak-decision demonstrations (used by the tests; see the module docstring)
# --------------------------------------------------------------------------
def value_only_hermite_subgrid(axes_lohi: Sequence[Tuple[float, float]],
                               levels: Sequence[int], U_nodes: np.ndarray,
                               iters=None, residuals=None) -> HermiteSolutionND:
    """A **value-only** :class:`hermite_nd.HermiteSolutionND` on nested CC ``levels``
    (``parametric_nd_smolyak.nested_levels``) — the Smolyak-subgrid interpolant of the
    Hermite family with ``enhanced=()``.

    Demonstrates that the Hermite interpolant slots into the combination technique in
    the value-only limit: this reduces bit-for-bit to the
    :class:`parametric_nd_smolyak.SmolyakSolutionND` subgrid
    (:class:`parametric_nd.ParametricSolutionND`) built from the identical nodal
    values (both use the committed barycentric contraction on the same nodes)."""
    from .parametric_nd_smolyak import nested_levels
    d = len(axes_lohi)
    nodes, weights = [], []
    for k, (lo, hi) in enumerate(axes_lohi):
        n, w = nested_levels(lo, hi, int(levels[k]))
        nodes.append(np.asarray(n, dtype=float))
        weights.append(np.asarray(w, dtype=float))
    grid = tuple(len(n) for n in nodes)
    U_nodes = np.asarray(U_nodes, dtype=float)
    field_shape = U_nodes.shape[d:]
    dU_nodes = np.zeros(grid + (d,) + field_shape)          # unused (enhanced=())
    axes_meta = [(lo, hi, len(nodes[k]) - 1) for k, (lo, hi) in enumerate(axes_lohi)]
    if iters is None:
        iters = np.zeros(grid, dtype=int)
    if residuals is None:
        residuals = np.zeros(grid, dtype=float)
    return HermiteSolutionND(
        axes=axes_meta, nodes=nodes, weights=weights, U_nodes=U_nodes, dU_nodes=dU_nodes,
        cvec=[cardinal_deriv_at_nodes(n) for n in nodes], enhanced=(),
        iters=iters, residuals=residuals, _solve_fn=None)


def level0_enhanced_is_taylor(lo: float, hi: float, U0, dU0, theta: float):
    """The Smolyak level-0 (single-midpoint-node) factor, ENHANCED, evaluated at
    ``theta`` — returns ``(hermite_value, taylor_value)``.

    A level-0 enhanced Hermite factor is the 1-node **linear Taylor**
    ``U_0 + (θ−θ_0)·dU_0`` (the R4 mode), NOT the value-only constant ``U_0`` — the
    concrete reason the enhancement is dense-only (see the module docstring)."""
    from .hermite import taylor_predict
    node = 0.5 * (lo + hi)                                  # nested CC level-0 midpoint
    U0 = np.asarray(U0, dtype=float)
    dU0 = np.asarray(dU0, dtype=float)
    her = HermiteSolutionND(
        axes=[(lo, hi, 0)], nodes=[np.array([node])], weights=[np.array([1.0])],
        U_nodes=U0[None, ...], dU_nodes=dU0[None, None, ...],
        cvec=[cardinal_deriv_at_nodes(np.array([node]))], enhanced=(0,),
        iters=np.zeros(1, int), residuals=np.zeros(1))
    hv = her.evaluate(np.array([float(theta)]))
    tv = taylor_predict(node, U0, dU0, float(theta) - node, order=1)
    return hv, tv
