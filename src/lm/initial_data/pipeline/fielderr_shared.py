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

**Held-out accuracy gate.**  ``docs/HISTORY_AND_FINDINGS.md`` 2.7: certification
proves only that a Newton polish *from* the guess reaches the tolerance; it says
nothing about the raw interpolant, and skipping the held-out comparison shipped
bad models twice.  ``enhanced_vs_value`` prints the per-rank comparison, returns
a machine-readable verdict for the output JSON, and optionally exits non-zero
when the enhanced model fails to beat value-only.

Standalone: numpy + stdlib only (no jax).  The caller supplies the model, so
this module never loads a corpus.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from lm.initial_data.paths import reports_root

#: Where cached truth sets live under the reports root.
CACHE_SUBDIR = ("P2", "truth_cache")

#: Relative-L2 a cached ``u_true`` must reproduce when re-solved, or it is stale.
VALIDATE_TOL = 1e-8

#: How many points a cache load re-solves to prove itself.
VALIDATE_N = 3


def rel_l2(u, ut):
    """Relative Frobenius L2 (matches ``run_cross_fielderror_chi.field_err``)."""
    u = np.asarray(u).reshape(-1)
    ut = np.asarray(ut).reshape(-1)
    return float(np.linalg.norm(u - ut) / max(np.linalg.norm(ut), 1e-300))


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
    plain-Hermite models are expected to regress (HISTORY_AND_FINDINGS 2.4:
    every multi-axis enhanced set degrades without the cross term); the
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
        print(f"[{tag}] NOTE: {note} — see docs/HISTORY_AND_FINDINGS.md 2.4/2.7",
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
            f"consumed as the paper's value+gradient curve.  See "
            f"docs/HISTORY_AND_FINDINGS.md 2.7.")
    return block


def attach_gate(path, block):
    """Add a ``gate`` block to an already-written sweep JSON, in place."""
    with open(path) as f:
        out = json.load(f)
    out["gate"] = block
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    return path
