"""PARASOL — Smolyak SPARSE-grid parametric collocation (the d≳3 cost fix).

The sparse-grid sibling of :mod:`parametric_nd` (the dense tensor-product
Chebyshev-in-parameter layer).  The dense layer pays a ``∏_k (Q_k+1) = O(Q^d)``
node count; the 4-D scaling run (`reports/3D_parametric/analysis.md` §8) showed
this needs ~30k anisotropic / ~65k isotropic solver calls to reach a held-out
1e-9 over the unequal-mass misaligned-spin head-on family ``(b, |S|, θ_S, q)``.
This module breaks that curse with a **Smolyak sparse grid** built by the
**combination technique** — a *signed sum of full anisotropic tensor
interpolants* on nested subgrids — so it reuses the committed dense layer
(:class:`parametric_nd.ParametricSolutionND`) verbatim rather than re-deriving a
hierarchical-surplus basis (open question 1 → combination, max reuse).

Design (the four open questions, answered):
  1. **Combination technique** over an explicit surplus basis: each subgrid
     interpolant is *exactly* a :class:`ParametricSolutionND`, so the validated
     barycentric ``evaluate`` is reused and the sparse interpolant is just
     ``Σ_i c(i)·ParametricSolutionND_i.evaluate(θ)`` over a shared solved-node
     pool.
  2. **Nested 1-D rule**: Clenshaw–Curtis doubling levels ``m(0)=1`` (the
     *midpoint*), ``m(i)=2^i+1`` (i≥1).  The CGL nodes ``cos(jπ/2^i)`` are nested
     (level i = the even-index nodes of level i+1), verified numerically.
     ``cheb_param_nodes(lo,hi,0)`` is NaN (Q=0 div-by-zero), so level 0 is
     special-cased to the midpoint (the nested CC level-0 node) by
     :func:`nested_levels` — ``cheb_param_nodes`` itself is untouched.
  3. **Warm-start ordering**: coarse→fine (by ``|l|₁``), snake-marching within
     each subgrid, caching solved nodes by value.  Shared nodes are solved ONCE;
     the unique-node count (the only solver-call cost) is exactly
     ``parametric_nd_2c.smolyak_points(d, level)``.
  4. **Dimension-adaptive** (the headline): a Gerstner–Griebel greedy refinement
     (:meth:`SmolyakSolverND.build_adaptive`) that spends levels on the hard axes
     (b, q) and starves the easy one (θ_S), plus a cheaper weighted-anisotropic
     simplex (:meth:`SmolyakSolverND.build_anisotropic`).

Solver-agnostic: it needs only the same callable the dense layer needs —
``solve_fn(theta_vec, guess, tol, max_iter) -> (U, info)``.  The 3-D wiring is the
thin :func:`from_problem_smolyak_3d` (it reuses ``parametric_nd_3d.make_solve_fn``
verbatim — same ``solve_fn`` interface, same D7 per-b cache).

Standalone: numpy + jax + the sibling ``parametric`` / ``parametric_nd`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as _iproduct
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .parametric import cheb_param_nodes          # the 1-D CGL layer (verbatim)
from .parametric_nd import ParametricSolutionND, snake_order   # the dense layer (verbatim)


# --------------------------------------------------------------------------
# Nested Clenshaw–Curtis 1-D rule (doubling levels, CGL nodes)
# --------------------------------------------------------------------------
def cc_m(i: int) -> int:
    """Clenshaw–Curtis node count at doubling level ``i``: ``m(0)=1``, ``m(i)=2^i+1``."""
    return 1 if i == 0 else (2 ** i + 1)


def nested_levels(lo: float, hi: float, i: int):
    """``(nodes, weights)`` for the nested CC level ``i`` on ``[lo, hi]``.

    Level ``i≥1`` is the full CGL set ``cheb_param_nodes(lo, hi, Q=2^i)`` (so the
    standard barycentric weights are exact for that complete set); level 0 is the
    single **midpoint** node — the nested CC level-0 point, which
    ``cheb_param_nodes(lo,hi,0)`` cannot supply (Q=0 → NaN).  The level-``i`` nodes
    are a subset of the level-``(i+1)`` nodes (numerically verified to 1e-13), so
    the union over a multi-index set is a genuine sparse grid.
    """
    if i == 0:
        return np.array([0.5 * (lo + hi)]), np.array([1.0])
    return cheb_param_nodes(lo, hi, 2 ** i)


# --------------------------------------------------------------------------
# Admissible multi-index sets (all downward-closed → valid combination grids)
# --------------------------------------------------------------------------
def isotropic_index_set(d: int, level: int) -> List[tuple]:
    """All 0-based levels ``l ≥ 0`` with ``|l|₁ ≤ level`` (the classic Smolyak set).

    The corresponding unique sparse-grid node count is
    ``parametric_nd_2c.smolyak_points(d, level)``.
    """
    out = []
    for idx in _iproduct(range(level + 1), repeat=d):
        if sum(idx) <= level:
            out.append(tuple(idx))
    return out


def anisotropic_index_set(d: int, level: float, weights: Optional[Sequence[float]] = None,
                          caps: Optional[Sequence[int]] = None) -> List[tuple]:
    """Weighted-simplex sparse set: ``Σ_k w_k·l_k ≤ level`` (and optional per-axis
    ``l_k ≤ caps_k``).  Downward-closed for any ``w_k > 0``.

    Smaller ``w_k`` → that axis is allowed *more* levels (resolve the hard axes).
    With ``weights=None`` this reduces to the isotropic set.
    """
    w = np.ones(d) if weights is None else np.asarray(weights, float)
    if np.any(w <= 0):
        raise ValueError("anisotropic weights must be strictly positive")
    cap = None if caps is None else [int(c) for c in caps]
    # per-axis level ceiling so the product enumeration is finite
    lmax = [int(np.floor(level / w[k] + 1e-9)) for k in range(d)]
    if cap is not None:
        lmax = [min(lmax[k], cap[k]) for k in range(d)]
    out = []
    for idx in _iproduct(*[range(lm + 1) for lm in lmax]):
        if float(np.dot(w, idx)) <= level + 1e-9:
            out.append(tuple(idx))
    return out


def combination_coeffs(index_set: Sequence[tuple]) -> Dict[tuple, int]:
    """General combination-technique coefficients for a downward-closed set.

    For each ``l`` in the set, ``c(l) = Σ_{z∈{0,1}^d, l+z∈set} (-1)^{|z|₁}`` — the
    telescoped difference-operator expansion ``A = Σ_{l∈set} ⊗_k Δ_{l_k}`` written
    as a signed sum of *full tensor* interpolators.  Only the nonzero
    coefficients are returned (interior indices telescope to 0).  Works for the
    isotropic, weighted-anisotropic, and greedy-adaptive (any downward-closed)
    sets alike.
    """
    iset = set(map(tuple, index_set))
    if not iset:
        return {}
    d = len(next(iter(iset)))
    coeffs: Dict[tuple, int] = {}
    for l in iset:
        c = 0
        for z in _iproduct((0, 1), repeat=d):
            ln = tuple(li + zi for li, zi in zip(l, z))
            if ln in iset:
                c += -1 if (sum(z) & 1) else 1
        if c != 0:
            coeffs[l] = c
    return coeffs


# --------------------------------------------------------------------------
# Sparse-grid solution container (combination of full-tensor interpolants)
# --------------------------------------------------------------------------
@dataclass
class SmolyakSolutionND:
    """Combination-technique sparse interpolant: ``Σ_i c_i · subgrid_i.evaluate(θ)``.

    Each ``subgrid`` is a full-tensor :class:`parametric_nd.ParametricSolutionND`
    on nested CC levels; they share one solved-node pool (so the solver-call cost
    is the *unique*-node count, :attr:`n_solver_nodes`, not the sum of subgrid
    sizes).
    """
    axes: List[Tuple[float, float]]        # [(p_min, p_max), ...] (no Q — Smolyak uses levels)
    index_set: List[tuple]                 # the admissible multi-index set (downward-closed)
    coeffs: List[int]                      # combination coefficient per kept subgrid
    subgrids: List[ParametricSolutionND]   # the kept (nonzero-coeff) full-tensor interpolants
    n_solver_nodes: int                    # unique solver calls (the sparse-grid node count)
    total_iters: int                       # total Newton iters over the node pool
    _solve_fn: Callable = field(repr=False, default=None)

    @property
    def d(self) -> int:
        return len(self.axes)

    @property
    def field_shape(self):
        return self.subgrids[0].field_shape

    @property
    def n_nodes(self) -> int:
        """Alias for :attr:`n_solver_nodes` (mirrors ``ParametricSolutionND.n_nodes``)."""
        return self.n_solver_nodes

    # ----- combination-technique barycentric interpolant -----
    def evaluate(self, theta):
        """U(θ) = Σ_i c_i · subgrid_i.evaluate(θ).  ``θ`` is a length-d vector."""
        out = None
        for c, sub in zip(self.coeffs, self.subgrids):
            v = sub.evaluate(theta)
            out = c * v if out is None else out + c * v
        return out

    # ----- JAX-differentiable twin (∂ID/∂θ hook; must not be queried at a node) -----
    def evaluate_jax(self, theta):
        out = None
        for c, sub in zip(self.coeffs, self.subgrids):
            v = sub.evaluate_jax(theta)
            out = c * v if out is None else out + c * v
        return out

    # ----- certified evaluation (the "cannot be silently wrong" gate) -----
    def evaluate_polished(self, theta, newton_steps: int = 2, tol: float = 1e-12):
        """Sparse prediction + 1–2 Newton steps → certified ‖R‖≤tol at θ.

        Returns ``(U, info)``; ``info.residual_norm`` is the certified constraint
        residual at θ, independent of the (sparse) interpolation error — exactly
        as for the dense :class:`ParametricSolutionND`.
        """
        if self._solve_fn is None:
            raise RuntimeError("no solve_fn attached; build via SmolyakSolverND/from_problem_smolyak_*")
        guess = jnp.asarray(self.evaluate(theta))
        U, info = self._solve_fn(np.asarray(theta, dtype=float), guess, tol, newton_steps)
        return U, info

    # ----- persistence: store the DEDUPLICATED node pool (numpy-only .npz) -----
    def _dedup_pool(self) -> Dict[tuple, tuple]:
        """Reconstruct the deduplicated node pool ``key -> (theta, U, iters, resid)``
        from the kept subgrids.

        The kept (nonzero-coefficient) subgrids always include every *maximal*
        multi-index, and — because the CC levels are nested — every lower
        subgrid's nodes are a subset of some maximal subgrid's nodes.  So the
        union of the kept subgrids' nodes is exactly the full solver-node pool
        (``len == n_solver_nodes``); no zero-coefficient subgrid is needed to
        recover a node.  This is what makes the sparse artifact store the
        deduplicated pool (~n_solver_nodes fields) rather than the overlapping
        per-subgrid tensors.
        """
        pool: Dict[tuple, tuple] = {}
        for sub in self.subgrids:
            nodes = sub.nodes
            shape = tuple(len(n) for n in nodes)
            for idx in np.ndindex(*shape):
                theta = np.array([nodes[k][idx[k]] for k in range(self.d)], dtype=float)
                key = _node_key(theta)
                if key in pool:
                    continue
                pool[key] = (theta, np.asarray(sub.U_nodes[idx], dtype=float),
                             int(sub.iters[idx]), float(sub.residuals[idx]))
        return pool

    def save(self, path, *, meta=None):
        """Persist this sparse-grid surrogate to a single ``.npz`` (numpy-only,
        no pickle).  Round-trips bit-for-bit via :func:`load_smolyak`.

        Stores the **deduplicated** node pool as parallel arrays —
        ``node_thetas`` (N×d), ``node_U`` (N×\\*field_shape), ``node_iters`` (N),
        ``node_resids`` (N) — plus ``index_set`` (M×d int), ``axes`` (d×2 float),
        ``field_shape``, and a ``meta_json`` 0-d string.  The combination
        coefficients are NOT stored: they are recomputed deterministically on
        load via :func:`combination_coeffs` (reusing :meth:`_finalize`).

        The artifact is a **standalone predictor**: after a reload ``evaluate`` /
        ``evaluate_jax`` work with only numpy + the parametric modules;
        ``evaluate_polished`` needs a ``solve_fn`` (reattach via
        ``parametric_nd.attach_solve_fn_3d``).  ``meta`` merges caller fields
        (axis_names, box, level, Na/Nb/Nφ, solver, tol, note, …);
        ``format_version``/``kind`` are set authoritatively.
        """
        from .parametric_nd import FORMAT_VERSION, _git_commit, _pack_meta
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        pool = self._dedup_pool()
        keys = list(pool)
        node_thetas = np.array([pool[k][0] for k in keys], dtype=float)     # N×d
        node_U = np.array([pool[k][1] for k in keys], dtype=float)          # N×*fs
        node_iters = np.array([pool[k][2] for k in keys], dtype=np.int64)   # N
        node_resids = np.array([pool[k][3] for k in keys], dtype=float)     # N
        index_set = np.array([[int(x) for x in l] for l in self.index_set],
                             dtype=np.int64)                                 # M×d
        axes = np.array([[float(lo), float(hi)] for (lo, hi) in self.axes], dtype=float)
        full_meta = {"d": int(self.d), "n_solver_nodes": int(self.n_solver_nodes),
                     "git_commit": _git_commit()}
        if meta:
            full_meta.update(meta)
        full_meta["format_version"] = FORMAT_VERSION      # authoritative
        full_meta["kind"] = "smolyak"
        np.savez(path,
                 node_thetas=node_thetas, node_U=node_U, node_iters=node_iters,
                 node_resids=node_resids, index_set=index_set, axes=axes,
                 field_shape=np.asarray(self.field_shape, dtype=np.int64),
                 meta_json=_pack_meta(full_meta))
        return path


# --------------------------------------------------------------------------
# Builder: solve the shared node pool once, assemble the subgrid interpolants
# --------------------------------------------------------------------------
def _node_key(theta_vec) -> tuple:
    return tuple(round(float(x), 12) for x in theta_vec)


class SmolyakSolverND:
    """Drives the sparse-grid continuation sweep and builds the combination interpolant.

    Needs only ``solve_fn(theta_vec, guess, tol, max_iter) -> (U, info)`` (the same
    contract as :class:`parametric_nd.ParametricSolverND`).  Three entry points:
    :meth:`build_isotropic`, :meth:`build_anisotropic` (weighted simplex / caps),
    and :meth:`build_adaptive` (Gerstner–Griebel greedy — the dimension-adaptive
    headline).
    """

    def __init__(self, solve_fn: Callable, axes: Sequence[Tuple[float, float]]):
        self.solve_fn = solve_fn
        self.axes = [tuple(a) for a in axes]
        self.d = len(self.axes)

    # ----- nested per-axis nodes/weights for one multi-index -----
    def _subgrid_nodes(self, l: Sequence[int]):
        nodes, weights = [], []
        for k, (lo, hi) in enumerate(self.axes):
            n, w = nested_levels(lo, hi, int(l[k]))
            nodes.append(n)
            weights.append(w)
        return nodes, weights

    # ----- solve every UNIQUE node of a set of subgrids, warm-started -----
    def _solve_pool(self, index_set, tol, max_iter, pool=None, guess0=None, verbose=False):
        """Solve each unique node across the subgrids in ``index_set`` exactly once.

        Coarse→fine traversal (by ``|l|₁``), snake-march within each subgrid, so
        each fresh node warm-starts from a genuine neighbour; already-solved nodes
        (shared between subgrids — the nesting payoff) are skipped.  Mutates and
        returns ``pool`` (dict ``key -> (U, iters, resid)``).
        """
        pool = {} if pool is None else pool
        guess = guess0
        for l in sorted(index_set, key=lambda t: sum(t)):
            nodes, _ = self._subgrid_nodes(l)
            shape = tuple(len(n) for n in nodes)
            for idx in snake_order(shape):
                theta = np.array([nodes[k][idx[k]] for k in range(self.d)], dtype=float)
                key = _node_key(theta)
                if key in pool:
                    guess = jnp.asarray(pool[key][0])
                    continue
                U, info = self.solve_fn(theta, guess, tol, max_iter)
                Ua = np.asarray(U)
                pool[key] = (Ua, int(info.iters), float(info.residual_norm))
                guess = jnp.asarray(Ua)
        if verbose:
            print(f"[smolyak] pool: {len(pool)} unique nodes")
        return pool

    # ----- assemble one subgrid's ParametricSolutionND from the solved pool -----
    def _assemble_subgrid(self, l, pool) -> ParametricSolutionND:
        nodes, weights = self._subgrid_nodes(l)
        shape = tuple(len(n) for n in nodes)
        field_shape = pool[next(iter(pool))][0].shape
        U_nodes = np.empty(shape + field_shape, dtype=float)
        iters = np.zeros(shape, dtype=int)
        resids = np.zeros(shape, dtype=float)
        for idx in np.ndindex(*shape):
            key = _node_key([nodes[k][idx[k]] for k in range(self.d)])
            Ua, it, rs = pool[key]
            U_nodes[idx] = Ua
            iters[idx] = it
            resids[idx] = rs
        axes_meta = [(lo, hi, len(nodes[k]) - 1) for k, (lo, hi) in enumerate(self.axes)]
        return ParametricSolutionND(axes=axes_meta, nodes=nodes, weights=weights,
                                    U_nodes=U_nodes, iters=iters, residuals=resids,
                                    _solve_fn=None)

    # ----- assemble the combination interpolant from a set + a solved pool -----
    def _finalize(self, index_set, pool) -> SmolyakSolutionND:
        coeffs = combination_coeffs(index_set)
        kept = sorted(coeffs)                          # deterministic order
        subgrids = [self._assemble_subgrid(l, pool) for l in kept]
        total_iters = int(sum(v[1] for v in pool.values()))
        return SmolyakSolutionND(
            axes=self.axes, index_set=[tuple(l) for l in index_set],
            coeffs=[coeffs[l] for l in kept], subgrids=subgrids,
            n_solver_nodes=len(pool), total_iters=total_iters,
            _solve_fn=self.solve_fn,
        )

    # ----- public builders -----------------------------------------------
    def build_isotropic(self, level: int, tol: float = 1e-12, max_iter: int = 20,
                         verbose: bool = False) -> SmolyakSolutionND:
        """Classic isotropic Smolyak at total level ``level`` (``|l|₁ ≤ level``).

        The unique-node count equals ``parametric_nd_2c.smolyak_points(d, level)``.
        """
        index_set = isotropic_index_set(self.d, level)
        pool = self._solve_pool(index_set, tol, max_iter, verbose=verbose)
        return self._finalize(index_set, pool)

    def build_anisotropic(self, level: float, weights: Optional[Sequence[float]] = None,
                          caps: Optional[Sequence[int]] = None, tol: float = 1e-12,
                          max_iter: int = 20, verbose: bool = False) -> SmolyakSolutionND:
        """Weighted-simplex sparse grid ``Σ_k w_k l_k ≤ level`` (optional per-axis
        ``caps``).  A cheap, static way to spend levels on the hard axes (smaller
        ``w_k`` → more levels on axis k)."""
        index_set = anisotropic_index_set(self.d, level, weights=weights, caps=caps)
        pool = self._solve_pool(index_set, tol, max_iter, verbose=verbose)
        return self._finalize(index_set, pool)

    def build_from_index_set(self, index_set: Sequence[tuple], tol: float = 1e-12,
                             max_iter: int = 20, verbose: bool = False) -> SmolyakSolutionND:
        """Build from an arbitrary **downward-closed** multi-index set (escape hatch)."""
        index_set = [tuple(int(x) for x in l) for l in index_set]
        _assert_downward_closed(index_set)
        pool = self._solve_pool(index_set, tol, max_iter, verbose=verbose)
        return self._finalize(index_set, pool)

    # ----- dimension-adaptive (Gerstner–Griebel greedy) -------------------
    def build_adaptive(self, max_nodes: int = 200, tol: float = 1e-12, max_iter: int = 20,
                       indicator_tol: float = 1e-13, max_level: int = 12,
                       verbose: bool = False,
                       # --- ADD-ONLY options (all default to the committed behaviour) ---
                       indicator: str = "surplus", seed_level: int = 0,
                       probe_points: Optional[Sequence] = None,
                       probe_values: Optional[Sequence] = None) -> SmolyakSolutionND:
        """Dimension-adaptive sparse grid (Gerstner & Griebel 2003).

        Greedily grows a downward-closed index set: at each step the **active**
        (admissible-to-refine) index with the largest local error indicator —
        the max-norm of the field surplus ``f − I_current`` over the index's NEW
        nodes — is moved to the **old** set and its forward neighbours become
        active.  Stops when the node budget ``max_nodes`` is hit, the largest
        active indicator drops below ``indicator_tol``, or no admissible index
        remains under ``max_level``.

        On strongly-anisotropic fields this spends levels on the hard axes
        automatically and beats both the dense tensor grid and the isotropic
        Smolyak.  On the *moderately*-anisotropic real head-on family the local
        surplus indicator over-invests in single hard axes ``(l,0,0,0)`` before
        adding the cross-coupling indices generic held-out points need
        (`smolyak_analysis.md` §4) — the three **add-only** knobs below target
        that failure mode (all default off → the committed behaviour is
        byte-for-byte unchanged):

        ``indicator``
            ``"surplus"`` (default) — the local max-norm field surplus.
            ``"profit"`` — Gerstner–Griebel's *cost-aware* indicator
            ``surplus / new_node_count`` (prefer cheap high-yield indices over
            expensive single-axis refinements).
            ``"heldout"`` — score each candidate by the **reduction of the
            max held-out error** on ``probe_points`` it produces when added to
            the current set (a steepest-descent on the real metric, re-scored
            every step).  Requires ``probe_points``; their truth is solved once
            (``probe_values`` to supply it), an extra cost exposed as
            ``result.n_probe_solves``.
        ``seed_level``
            ``> 0`` → seed the **old** set with the isotropic simplex
            ``|l|₁ ≤ seed_level`` so every pairwise cross-term ``(1,1,0,0)`` is
            present before the greedy starts (cross-coupling is never starved).
        """
        if indicator not in ("surplus", "profit", "heldout"):
            raise ValueError(f"indicator must be 'surplus'|'profit'|'heldout', got {indicator!r}")
        if indicator == "heldout" and probe_points is None:
            raise ValueError("indicator='heldout' requires probe_points")

        d = self.d
        pool: Dict[tuple, tuple] = {}
        indicators: Dict[tuple, float] = {}     # raw surplus (surplus / profit modes)
        new_counts: Dict[tuple, int] = {}       # new-node count per index (profit mode)

        def _fwd(l, k):
            return tuple(l[j] + (1 if j == k else 0) for j in range(d))

        # ---- the held-out-driven greedy is a separate loop (re-scores every step) ----
        if indicator == "heldout":
            sol = self._build_adaptive_heldout(
                max_nodes, tol, max_iter, indicator_tol, max_level, verbose,
                seed_level, probe_points, probe_values, pool, _fwd)
            return sol

        # ---- surplus / profit greedy (the committed loop; profit only re-keys) ----
        def _score(t):
            if indicator == "profit":
                return indicators.get(t, 0.0) / max(new_counts.get(t, 1), 1)
            return indicators.get(t, 0.0)

        def _score_candidate(nb, base_old):
            indicators[nb] = self._surplus_indicator(nb, base_old, pool, tol, max_iter)
            if indicator == "profit":
                new_counts[nb] = self._count_new_nodes(nb, base_old)

        if seed_level > 0:
            old: set = set(isotropic_index_set(d, seed_level))
            self._solve_pool(sorted(old), tol, max_iter, pool=pool)
            active: set = set()
            for l in sorted(old):
                for k in range(d):
                    nb = _fwd(l, k)
                    if nb in old or nb in active or nb[k] > max_level:
                        continue
                    if not _backward_neighbours_in(nb, old):
                        continue
                    _score_candidate(nb, old)
                    active.add(nb)
        else:
            root = (0,) * d
            old = set()
            active = {root}
            self._solve_pool([root], tol, max_iter, pool=pool)
            indicators[root] = self._surplus_indicator(root, set(), pool, tol, max_iter)

        while active:
            # pick the active index with the largest indicator
            l = max(active, key=_score)
            if _score(l) < indicator_tol and old:
                break
            active.remove(l)
            old.add(l)
            if verbose:
                print(f"[adaptive] add {l}  indicator={indicators.get(l,0.0):.2e}  "
                      f"nodes={len(pool)}")
            if len(pool) >= max_nodes:
                break
            # promote admissible forward neighbours into the active set
            for k in range(d):
                nb = _fwd(l, k)
                if nb in old or nb in active or nb[k] > max_level:
                    continue
                # admissible ⇔ all backward neighbours already refined (downward-closed)
                if not _backward_neighbours_in(nb, old):
                    continue
                # score nb: solve its NEW nodes and measure the surplus vs the
                # current (old) interpolant; the solved nodes are cached for accept
                _score_candidate(nb, old)
                active.add(nb)

        # ensure every scored/active node is in the pool, then finalize over `old`
        # (old is downward-closed; active leaves dangling forward neighbours out)
        _assert_downward_closed(sorted(old))
        sol = self._finalize(sorted(old), pool)
        sol.n_probe_solves = 0
        return sol

    # ----- held-out-driven greedy (indicator="heldout") -------------------
    def _build_adaptive_heldout(self, max_nodes, tol, max_iter, indicator_tol, max_level,
                                verbose, seed_level, probe_points, probe_values, pool, _fwd):
        """Greedy that, at every step, accepts the active index whose addition most
        reduces the max held-out error over ``probe_points`` (a steepest descent on
        the real interpolation metric, the §4-failure-mode fix)."""
        d = self.d
        probe = [np.asarray(p, dtype=float) for p in probe_points]
        n_probe_solves = 0
        if probe_values is not None:
            truth = [np.asarray(v, dtype=float) for v in probe_values]
        else:
            truth = []
            guess = None
            for p in probe:
                U, _ = self.solve_fn(p, guess, tol, max_iter)
                Ua = np.asarray(U)
                truth.append(Ua)
                guess = jnp.asarray(Ua)
                n_probe_solves += 1

        # seed
        if seed_level > 0:
            old: set = set(isotropic_index_set(d, seed_level))
            self._solve_pool(sorted(old), tol, max_iter, pool=pool)
        else:
            old = set()
        active: set = set()
        if not old:
            active.add((0,) * d)
        else:
            for l in sorted(old):
                for k in range(d):
                    nb = _fwd(l, k)
                    if nb in old or nb in active or nb[k] > max_level:
                        continue
                    if _backward_neighbours_in(nb, old):
                        active.add(nb)

        while active and len(pool) < max_nodes:
            err_old = self._heldout_err(old, pool, probe, truth)
            best, best_imp = None, None
            for nb in sorted(active):
                self._solve_pool([nb], tol, max_iter, pool=pool)   # solve (cached after 1st)
                imp = err_old - self._heldout_err(old | {nb}, pool, probe, truth)
                if best_imp is None or imp > best_imp:
                    best_imp, best = imp, nb
            if best_imp is not None and best_imp < indicator_tol and old:
                break
            l = best
            active.remove(l)
            old.add(l)
            if verbose:
                print(f"[adaptive-heldout] add {l}  improvement={best_imp:.2e}  "
                      f"nodes={len(pool)}")
            if len(pool) >= max_nodes:
                break
            for k in range(d):
                nb = _fwd(l, k)
                if nb in old or nb in active or nb[k] > max_level:
                    continue
                if _backward_neighbours_in(nb, old):
                    active.add(nb)

        _assert_downward_closed(sorted(old))
        sol = self._finalize(sorted(old), pool)
        sol.n_probe_solves = n_probe_solves
        return sol

    # held-out max-norm error of the combination interpolant over a probe set
    def _heldout_err(self, index_set, pool, probe, truth) -> float:
        iset = [tuple(l) for l in index_set]
        if not iset:                                   # empty set → prediction ≡ 0
            return max(float(np.max(np.abs(t))) for t in truth)
        coeffs = combination_coeffs(iset)
        kept = sorted(coeffs)
        sol = SmolyakSolutionND(
            axes=self.axes, index_set=iset, coeffs=[coeffs[k] for k in kept],
            subgrids=[self._assemble_subgrid(k, pool) for k in kept],
            n_solver_nodes=len(pool), total_iters=0, _solve_fn=None)
        worst = 0.0
        for p, t in zip(probe, truth):
            worst = max(worst, float(np.max(np.abs(sol.evaluate(p) - t))))
        return worst

    # number of subgrid-l nodes NOT already covered by ``old_set`` (profit mode)
    def _count_new_nodes(self, l, old_set) -> int:
        nodes, _ = self._subgrid_nodes(l)
        shape = tuple(len(n) for n in nodes)
        old_keys = set()
        for k in old_set:
            on, _ = self._subgrid_nodes(k)
            for oi in np.ndindex(*tuple(len(x) for x in on)):
                old_keys.add(_node_key([on[a][oi[a]] for a in range(self.d)]))
        cnt = 0
        for idx in np.ndindex(*shape):
            if _node_key([nodes[k][idx[k]] for k in range(self.d)]) not in old_keys:
                cnt += 1
        return max(cnt, 1)

    # surplus indicator: max |f - I_old| over the NEW nodes introduced by index l
    def _surplus_indicator(self, l, old_set, pool, tol, max_iter) -> float:
        nodes, _ = self._subgrid_nodes(l)
        shape = tuple(len(n) for n in nodes)
        # current interpolant over the OLD set (empty → 0)
        cur = None
        if old_set:
            coeffs = combination_coeffs(old_set)
            cur = SmolyakSolutionND(
                axes=self.axes, index_set=list(old_set),
                coeffs=[coeffs[k] for k in sorted(coeffs)],
                subgrids=[self._assemble_subgrid(k, pool) for k in sorted(coeffs)],
                n_solver_nodes=len(pool), total_iters=0, _solve_fn=None)
        # nodes already covered by the old set (no new info there)
        old_keys = set()
        for k in old_set:
            on, _ = self._subgrid_nodes(k)
            osh = tuple(len(x) for x in on)
            for oi in np.ndindex(*osh):
                old_keys.add(_node_key([on[a][oi[a]] for a in range(self.d)]))
        worst = 0.0
        guess = None
        for idx in snake_order(shape):
            theta = np.array([nodes[k][idx[k]] for k in range(self.d)], dtype=float)
            key = _node_key(theta)
            if key in pool:
                Ua = pool[key][0]
            else:
                U, info = self.solve_fn(theta, guess, tol, max_iter)
                Ua = np.asarray(U)
                pool[key] = (Ua, int(info.iters), float(info.residual_norm))
            guess = jnp.asarray(Ua)
            if key in old_keys:
                continue                                   # not a NEW node
            pred = 0.0 if cur is None else cur.evaluate(theta)
            worst = max(worst, float(np.max(np.abs(Ua - pred))))
        return worst


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _backward_neighbours_in(l, iset) -> bool:
    """True iff every backward neighbour ``l - e_k`` (l_k>0) is in ``iset``."""
    d = len(l)
    for k in range(d):
        if l[k] == 0:
            continue
        bk = tuple(l[j] - (1 if j == k else 0) for j in range(d))
        if bk not in iset:
            return False
    return True


def _assert_downward_closed(index_set):
    iset = set(map(tuple, index_set))
    for l in iset:
        d = len(l)
        for k in range(d):
            if l[k] > 0:
                bk = tuple(l[j] - (1 if j == k else 0) for j in range(d))
                assert bk in iset, f"index set not downward-closed: {l} missing {bk}"


# --------------------------------------------------------------------------
# Wiring: build a SmolyakSolverND around the 3-D non-axisymmetric solver
# --------------------------------------------------------------------------
def from_problem_smolyak_3d(prob, axes: Sequence[dict], M_tot: float = 1.0,
                            fixed: Optional[Dict[str, float]] = None, use_cache: bool = True,
                            solver: str = "nk", gmres_rtol: float = 1e-4,
                            retry_tol: Optional[float] = None) -> SmolyakSolverND:
    """``SmolyakSolverND`` over the active axes ``[{name,min,max}, ...]`` (NO Q —
    Smolyak uses doubling *levels*).

    Reuses ``parametric_nd_3d.make_solve_fn`` verbatim — the same
    ``solve_fn(theta, guess, tol, max_iter)`` contract and the same D7 per-b
    assembly cache the dense layer uses.  Put ``b`` first to keep the per-b cache
    warm.  ``solver='modified'`` for the (field-identical, cheaper) convergence
    studies, ``solver='nk'`` (default) for the certified-prediction gate.
    """
    from .parametric_nd_3d import make_solve_fn

    active_names = [a["name"] for a in axes]
    solve_fn, _ = make_solve_fn(prob, active_names, M_tot=M_tot, fixed=fixed,
                                use_cache=use_cache, solver=solver, gmres_rtol=gmres_rtol,
                                retry_tol=retry_tol)
    spec = [(a["min"], a["max"]) for a in axes]
    return SmolyakSolverND(solve_fn, spec)


# --------------------------------------------------------------------------
# Persistence: load a sparse-grid surrogate saved by SmolyakSolutionND.save
# --------------------------------------------------------------------------
def load_smolyak(path) -> SmolyakSolutionND:
    """Load a :class:`SmolyakSolutionND` saved by :meth:`SmolyakSolutionND.save`.

    Rebuilds the node pool by re-keying ``node_thetas`` with :func:`_node_key`,
    constructs a solver-less ``SmolyakSolverND(solve_fn=None, axes=…)``, and
    returns ``solver._finalize(index_set, pool)`` — reusing the committed
    combination-technique assembly (``combination_coeffs`` + ``_assemble_subgrid``)
    with **zero solves**.  ``evaluate`` / ``evaluate_jax`` work immediately;
    ``evaluate_polished`` keeps raising ``RuntimeError`` until a solver is
    attached (``parametric_nd.attach_solve_fn_3d``).  Parsed metadata is stored
    on the returned object as ``.meta``.
    """
    from .parametric_nd import _load_npz, _unpack_meta, _check_meta
    data = _load_npz(path)
    meta = _unpack_meta(data)
    _check_meta(meta, "smolyak")
    try:
        node_thetas = np.asarray(data["node_thetas"], dtype=float)     # N×d
        node_U = np.asarray(data["node_U"], dtype=float)               # N×*fs
        node_iters = np.asarray(data["node_iters"])
        node_resids = np.asarray(data["node_resids"], dtype=float)
        index_set = [tuple(int(x) for x in row)
                     for row in np.asarray(data["index_set"])]
        axes = [(float(a[0]), float(a[1]))
                for a in np.asarray(data["axes"], dtype=float)]
        pool: Dict[tuple, tuple] = {}
        for i in range(node_thetas.shape[0]):
            key = _node_key(node_thetas[i])
            pool[key] = (np.asarray(node_U[i], dtype=float),
                         int(node_iters[i]), float(node_resids[i]))
        solver = SmolyakSolverND(solve_fn=None, axes=axes)
        sol = solver._finalize(index_set, pool)      # combination coeffs recomputed
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"corrupt PARASOL smolyak surrogate '{path}': {e}")
    sol.meta = meta
    return sol
