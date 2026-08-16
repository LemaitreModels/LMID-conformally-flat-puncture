"""Shared plumbing for the field-error POD-rank sweeps (``run_*fielderr*``).

Two utilities that several producers need *identically*, factored here so they
cannot drift apart.

**Certified-truth cache.**  A field-error sweep measures
``||guess - u_true|| / ||u_true||`` against the certified NK solve over a fixed
held-out point set.  ``u_true`` is a property of the PDE, the box and the
sampler -- **not** of the surrogate being scored -- so two sweeps over the same
box, grid, sampler, seed and tolerance are entitled to share it.  At 8-D that
truth is ~15 h for 1000 points (measured: 54.9 s/pt), and each producer used to
solve its own copy and discard it.  The one producer that tried to reuse a saved
copy read a hand-made shard set that went silently stale when the box was
retargeted, and died on a missing path.

``certified_truth`` caches on a key that pins **every** input the solve depends
on, and re-validates a few points against a live solve whenever it loads from
disk, so a stale cache raises instead of quietly poisoning a sweep.  The key
omits the point count: the samplers draw from one sequential rng stream, so
``sampler(box, n, seed) == sampler(box, N, seed)[:n]`` for ``n <= N`` and a
short smoke run reuses the prefix of a long run's cache for free.

**Held-out accuracy gate.**  Certification
proves only that a Newton polish *from* the guess reaches the tolerance; it says
nothing about the raw interpolant, and skipping the held-out comparison shipped
bad models twice.  ``enhanced_vs_value`` prints the per-rank comparison, returns
a machine-readable verdict for the output JSON, and optionally exits non-zero
when the enhanced model fails to beat value-only.

**Shipped-model table and the rank-truncating loader.**  ``MODELS`` names the
per-dimension shipped POD artifact and the rank ladder the sweeps walk, and
``load_pod_truncated`` reads one back at a reduced rank.  Both lived in
``run_guess_vs_memory``, a *producer*, while three other producers imported them
as a library -- so a producer's module globals were load-bearing for figures it
does not make.  They are library, and they live here.

Standalone: numpy + stdlib only at import time (no jax).  ``load_pod_truncated``
imports the ROM lazily inside its body precisely to keep that true, so importing
this module for the truth cache alone stays cheap and jax-free.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from lemaitre.initial_data.conformally_flat_puncture.paths import reports_root

#: Where cached truth sets live under the reports root.
CACHE_SUBDIR = ("P2", "truth_cache")

#: Relative-L2 a cached ``u_true`` must reproduce when re-solved, or it is stale.
VALIDATE_TOL = 1e-8

#: How many points a cache load re-solves to prove itself.
VALIDATE_N = 3

#: Minimum parameter-space gap an off-node sample must keep from a grid node.
GAP_MIN = 1e-4

#: Heavy-corpus root; ``$LM_REPORTS`` (see ``docs/DATA.md``).
REPORTS = reports_root()

#: The shipped POD artifact per box dimension, with the rank ladder the
#: field-error and residual sweeps walk, and the reference reports they score
#: against.  ONE corpus per dimension, re-encoded three ways (value / Hermite /
#: POD) -- the ranks are the abscissa of fig05's memory axis.
MODELS = {
    4: dict(
        pod=os.path.join(REPORTS, "P2/models_chi/"
                         "pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By.npz"),
        ranks=[1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 75, 87],
        ref_value="polish_table_qc_chi_prod_1000.json",
        ref_hermite="polish_table_qc_chi_prod_hermite_1000.json",
    ),
    8: dict(
        pod=os.path.join(REPORTS, "P2/models_chi/pod_hermite_smolyak_"
                         "spin8qc_L5_enh-chi_Ax-chi_Ay-chi_Az-chi_Bx-chi_By-chi_Bz.npz"),
        ranks=[1, 5, 15, 30, 60, 100, 150, 200, 250, 300, 350, 394],
        ref_value="polish_table_chi8d_value_1000.json",
        ref_hermite="polish_table_chi8d_hermite_1000.json",   # pending (cluster)
    ),
}


def load_pod_truncated(path, r_new):
    """A ``PODHermiteSmolyak`` truncated to the leading ``r_new`` modes.

    Mirrors ``hermite_smolyak_pod.load_pod_hermite_smolyak`` verbatim but slices
    ``Phi[:, :r_new]`` / ``node_U[:, :r_new]`` / ``node_dU[..., :r_new]`` before
    the finalize.  ``r_new == r_shipped`` reproduces the committed loader
    bit-for-bit; ``evaluate`` at any ``r_new`` equals ``mean + Phi[:, :r_new] @
    c[:r_new]`` (leading-r' POD reconstruction).  NO re-solve, NO corpus.

    The ROM imports are deliberately local: they pull jax, and this module
    promises a jax-free import for callers that only want the truth cache.
    """
    from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd import (
        _load_npz, _unpack_meta, _check_meta,
    )
    from lemaitre.initial_data.conformally_flat_puncture.parametric.parametric_nd_smolyak import (
        _node_key,
    )
    from lemaitre.initial_data.conformally_flat_puncture.parametric.hermite_smolyak import (
        HermiteSmolyakSolverND,
    )
    from lemaitre.initial_data.conformally_flat_puncture.parametric.hermite_smolyak_pod import (
        PODHermiteSmolyak,
    )

    data = _load_npz(path)
    meta = _unpack_meta(data); _check_meta(meta, "pod_hermite_smolyak")
    r_full = int(data["r"]); r_new = int(min(r_new, r_full))
    Phi = np.asarray(data["Phi"], dtype=float)[:, :r_new]
    mean = np.asarray(data["mean"], dtype=float)
    field_shape = tuple(int(x) for x in np.asarray(data["field_shape"], dtype=np.int64))
    node_thetas = np.asarray(data["node_thetas"], dtype=float)
    node_U = np.asarray(data["node_U"], dtype=float)[:, :r_new]
    node_dU = np.asarray(data["node_dU"], dtype=float)[:, :, :r_new]
    node_iters = np.asarray(data["node_iters"])
    node_resids = np.asarray(data["node_resids"], dtype=float)
    index_set = [tuple(int(x) for x in row) for row in np.asarray(data["index_set"])]
    axes = [(float(a[0]), float(a[1])) for a in np.asarray(data["axes"], dtype=float)]
    enhanced = tuple(int(e) for e in np.asarray(data["enhanced"], dtype=np.int64))
    pool = {}
    for i in range(node_thetas.shape[0]):
        pool[_node_key(node_thetas[i])] = (
            np.asarray(node_U[i], dtype=float), np.asarray(node_dU[i], dtype=float),
            int(node_iters[i]), float(node_resids[i]))
    solver = HermiteSmolyakSolverND(solve_fn=None, axes=axes, tangent_fn=None,
                                    enhanced_axes=enhanced)
    pod = PODHermiteSmolyak(solver._finalize(index_set, pool), Phi, mean, field_shape)
    pod.meta = meta
    return pod


def rel_l2(u, ut):
    """Relative Frobenius L2 (matches ``run_cross_fielderror_chi.field_err``)."""
    u = np.asarray(u).reshape(-1)
    ut = np.asarray(ut).reshape(-1)
    return float(np.linalg.norm(u - ut) / max(np.linalg.norm(ut), 1e-300))


def polish_history(prob, sl, U0, max_steps, tol=1e-12, separable=None):
    """Instrumented NK polish: the per-step residual AND field-error staircase.

    This IS ``solver_3d_nk.newton_solve_nk`` — the production certified loop —
    hooked through its ``on_iterate`` callback to also store the field at every
    measured step, so the residuals recorded here reproduce the residual
    staircases (``run_polish_cold`` / ``run_polish_table``) per-step by
    construction rather than by a re-implementation kept in sync by hand.  It
    used to be a byte-identical copy of the loop in each field-error producer,
    assembled **dense** — a route the production solve no longer takes — so the
    field-error rows of fig04 were measured on different linear algebra than the
    residual rows they share a step axis with.

    ``separable`` is the route: ``None`` (default) resolves through
    ``choose_separable`` exactly as a production query does; pass ``True``/
    ``False`` to force.  Callers should resolve it themselves and record the
    route in the artifact meta.

    Returns ``(residuals[0..max_steps], field_error[0..max_steps])``; field
    error is relative Frobenius L2 against the best (converged) iterate.  Both
    lists are padded with their last value once the solve has converged or
    stagnated (matching the residual-staircase padding).

    The solver imports are local so this module keeps its jax-free import for
    callers that only want the truth cache.
    """
    from lemaitre.initial_data.conformally_flat_puncture.solver import solver_3d_nk as s3nk

    fields, hist = [], []

    def _store(_k, rn, U):
        hist.append(float(rn))
        fields.append(np.array(U, dtype=float, copy=True))

    U_best, _info = s3nk.newton_solve_nk(prob, sl, U0=U0, tol=tol,
                                         max_iter=max_steps, separable=separable,
                                         on_iterate=_store)

    while len(hist) < max_steps + 1:            # pad to the full step axis
        hist.append(hist[-1])
        fields.append(fields[-1])

    best = np.asarray(U_best, dtype=float).reshape(fields[0].shape)
    uref_norm = float(np.linalg.norm(best))
    ferr = [float(np.linalg.norm(f - best) / max(uref_norm, 1e-300)) for f in fields]
    return hist, ferr


# ------------------------------------------------------------------ truth cache ----------------
def truth_key(*, box, Na, Nb, Nphi, sampler, seed, u_steps, u_tol, fixed=None):
    """Canonical description of a certified-truth set.

    Everything the solve depends on and nothing else.  The point count is
    deliberately absent (see the module docstring: prefix reuse).
    """
    return dict(
        box=[[float(lo), float(hi)] for lo, hi in box],
        grid=[int(Na), int(Nb), int(Nphi)],
        fixed={str(k): float(v) for k, v in sorted((fixed or {}).items())},
        sampler=str(sampler),
        seed=int(seed),
        u_steps=int(u_steps),
        u_tol=float(u_tol),
    )


def _digest(key):
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def truth_cache_path(key):
    """Absolute path of the cache file for ``key`` (may not exist)."""
    root = os.path.join(reports_root(), *CACHE_SUBDIR)
    return os.path.join(root, f"{key['sampler']}_seed{key['seed']}_{_digest(key)}.npz")


def load_truth(key, n_points):
    """``(UT[:n], res[:n])`` from the cache, or ``None`` on a miss.

    A file whose stored key does not match ``key`` is a digest collision or a
    hand-edited file; that raises rather than returning the wrong truth.  A file
    holding fewer than ``n_points`` points is a miss (the caller re-solves the
    superset and overwrites).
    """
    p = truth_cache_path(key)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as d:
        stored = json.loads(d["key_json"].item())
        if stored != key:
            raise ValueError(
                f"truth cache key mismatch at {p}\n  stored: {stored}\n  wanted: {key}")
        UT = np.asarray(d["UT"], dtype=float)
        res = np.asarray(d["res"], dtype=float)
    if len(UT) < n_points:
        return None
    return UT[:n_points], res[:n_points]


def save_truth(key, UT, res):
    """Write ``(UT, res)`` to the cache for ``key`` (atomic replace).  Returns the path."""
    p = truth_cache_path(key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez(tmp, UT=np.asarray(UT, dtype=float), res=np.asarray(res, dtype=float),
             key_json=json.dumps(key, sort_keys=True))
    os.replace(tmp, p)
    return p


def certified_truth(mc, pts, key, *, tag, validate_n=VALIDATE_N, progress=25):
    """Certified ``u_true`` at ``pts``, from the cache when one is usable.

    ``mc`` needs only ``evaluate_polished(theta, newton_steps=, tol=)``.  On a
    cache hit ``validate_n`` points are re-solved and must agree to
    ``VALIDATE_TOL`` -- this is what catches a truth set built on a different
    box.  On a miss every point is solved and the result is cached.

    Returns ``(UT, res, source)`` with ``source`` in ``{"cache", "solve"}``.
    """
    u_steps, u_tol = int(key["u_steps"]), float(key["u_tol"])
    n = len(pts)
    p = truth_cache_path(key)

    hit = load_truth(key, n)
    if hit is not None:
        UT, res = hit
        print(f"[{tag}] certified u_true: cache HIT {os.path.basename(p)} "
              f"({n} pts); validating {min(validate_n, n)} re-solves "
              f"(assert rel-L2 < {VALIDATE_TOL:.0e}) ...", flush=True)
        vmax = 0.0
        for k in range(min(validate_n, n)):
            u, info = mc.evaluate_polished(pts[k], newton_steps=u_steps, tol=u_tol)
            rel = rel_l2(u, UT[k])
            vmax = max(vmax, rel)
            print(f"   validate pt {k}: rel-L2(re-solve, cached)={rel:.2e} "
                  f"(res={info.residual_norm:.1e})", flush=True)
        if vmax >= VALIDATE_TOL:
            raise AssertionError(
                f"cached u_true is STALE: max rel-L2={vmax:.2e} >= {VALIDATE_TOL:.0e}\n"
                f"  cache: {p}\n"
                f"  The cache key pins box/grid/sampler/seed/tol, so a mismatch here means "
                f"the solver or the model corpus changed under a fixed key.  Delete the "
                f"cache file to force a re-solve.")
        print(f"[{tag}] u_true reuse VALIDATED (max rel-L2={vmax:.2e}) — "
              f"skipped the certified-truth solve", flush=True)
        return UT, res, "cache"

    print(f"[{tag}] certified u_true: cache MISS; solving {n} points "
          f"(newton_steps={u_steps}, tol={u_tol:.0e}) ...", flush=True)
    UT, res = [], []
    t0 = time.time()
    for i, th in enumerate(pts):
        ut, info = mc.evaluate_polished(th, newton_steps=u_steps, tol=u_tol)
        UT.append(np.asarray(ut, dtype=float).reshape(-1))
        res.append(float(info.residual_norm))
        if (i + 1) % progress == 0 or i == n - 1 or n <= 10:
            el = time.time() - t0
            rate = el / (i + 1)
            print(f"   u_true {i+1}/{n} ({el:.0f}s, {rate:.2f}s/pt, "
                  f"ETA {rate*(n-i-1)/60:.1f} min)  res med={np.median(res):.1e}",
                  flush=True)
    UT = np.asarray(UT, dtype=float)
    res = np.asarray(res, dtype=float)
    print(f"[{tag}] u_true done: {n} pts, {(time.time()-t0)/60:.1f} min, "
          f"{(time.time()-t0)/n:.2f}s/pt, res med={np.median(res):.1e}", flush=True)
    print(f"[{tag}] cached -> {save_truth(key, UT, res)}", flush=True)
    return UT, res, "solve"


# ------------------------------------------------------------- held-out accuracy gate ----------
def enhanced_vs_value(val_med, enh_med, *, label, expect_below, fatal=False,
                      margin=1.05, tag="gate"):
    """Compare enhanced vs value-only median field error, rank by rank.

    ``val_med``/``enh_med`` map POD rank -> median field error.  The measurement
    is ``beats_value``: is the enhanced curve at or below value-only (within
    ``margin``) at a majority of shared ranks?

    ``expect_below`` declares what this model is *supposed* to do, so the log
    distinguishes an expected outcome from a surprise.  The gradient-only
    plain-Hermite models are expected to regress (every multi-axis enhanced set
    degrades without the cross term); the
    cross-completed models are expected to win.

    ``fatal=True`` raises ``SystemExit(1)`` on a regression -- use it wherever a
    regression means the artifact must not be consumed.

    Returns a JSON-able verdict block for the output artifact.
    """
    common = sorted(set(val_med) & set(enh_med))
    below = [r for r in common if enh_med[r] <= val_med[r] * margin]
    beats = len(below) > len(common) // 2
    verdict = "PASS" if beats else "REGRESSION"

    print(f"[{tag}] {label}: enhanced <= {margin:g}x value-only at "
          f"{len(below)}/{len(common)} shared ranks  -> {verdict}", flush=True)
    for r in common:
        mark = "  <-- enhanced below" if enh_med[r] <= val_med[r] else ""
        print(f"      r={r:6d}  value-only={val_med[r]:.3e}  "
              f"enhanced={enh_med[r]:.3e}{mark}", flush=True)

    if beats != bool(expect_below):
        note = ("expected a regression (gradient-only without the cross term) but the "
                "enhanced curve WINS" if beats else
                "expected the enhanced curve to WIN but it regresses")
        print(f"[{tag}] NOTE: {note} — a multi-axis enhanced set needs its cross "
              f"term, and the held-out comparison is the only check that catches it",
              flush=True)

    block = dict(metric="median_field_error_relL2", margin=float(margin),
                 n_ranks=len(common), below_at=len(below),
                 beats_value=bool(beats), expect_below=bool(expect_below),
                 verdict=verdict,
                 per_rank=[dict(r=int(r), value_only=float(val_med[r]),
                                enhanced=float(enh_med[r])) for r in common])

    if fatal and not beats:
        raise SystemExit(
            f"[{tag}] FAILED held-out accuracy gate: {label} regresses below value-only "
            f"({len(below)}/{len(common)} ranks at/below).  This artifact must not be "
            f"consumed as the paper's value+gradient curve: certification proves only "
            f"that a Newton polish from the guess converges, not that the raw "
            f"interpolant is accurate.")
    return block


def attach_gate(path, block):
    """Add a ``gate`` block to an already-written sweep JSON, in place."""
    with open(path) as f:
        out = json.load(f)
    out["gate"] = block
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    return path
