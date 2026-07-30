"""PARASOL — bare-guess constraint residual ||R||_inf vs stored model memory.

The memory<->accuracy tradeoff of the reduced-basis (POD) re-encoding
(manuscript Sec. sec:param:pod): as the POD truncation rank ``r`` is swept, the
shipped model shrinks (memory ~ linear in ``r``) but the barycentric warm-start
"guess" gets worse (its constraint residual ``||R||_inf`` before any Newton
polish rises).  This quantifies "how far can the shipped model be compressed
before the guess -- and hence the polish cost -- degrades."

For each dimension (4D, 8D) we truncate the shipped POD model to a ladder of
ranks ``r' <= r_shipped`` (a cheap slice of ``Phi``/coeffs -- NO re-solve), draw a
fixed set of seed-0 off-node points (the SAME sampling as ``run_polish_table.py``),
and record the (min, median, max) bare-guess ``||R||_inf`` over them.  The
value-only Smolyak and the full gradient-enhanced Hermite (same solved corpus,
un-compressed) are marked as annotated reference points.

The guess residual is the *equilibrated* constraint residual of the decoded field
(``solver_3d_nk.equil_residual_inf``) -- byte-identical to ``run_polish_table``'s
``history[0]`` but computed WITHOUT the wasted Newton step, and with the (guess-
independent) assembly shared across the whole rank ladder at each point.

Add-only.  Imports committed ``parasol`` modules read-only; edits nothing.

Run:
  python sandbox/parasol/run_guess_vs_memory.py               # sweep + plot
  python sandbox/parasol/run_guess_vs_memory.py --replot      # re-plot from JSON
  python sandbox/parasol/run_guess_vs_memory.py --n-points 300 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


from lemaitre.initial_data.solver import solver_3d as s3
from lemaitre.initial_data.solver import solver_3d_nk as s3nk
from lemaitre.initial_data.parametric.parametric import cheb_param_nodes
from lemaitre.initial_data.parametric.parametric_nd import (
    _load_npz, _unpack_meta, _check_meta,
)
from lemaitre.initial_data.parametric.parametric_nd_smolyak import _node_key
from lemaitre.initial_data.parametric.parametric_nd_3d import theta_to_slice3d, assemble_cached_3d
from lemaitre.initial_data.parametric.hermite_smolyak import HermiteSmolyakSolverND
from lemaitre.initial_data.parametric.hermite_smolyak_pod import (
    PODHermiteSmolyak, load_pod_hermite_smolyak,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPDIR = os.path.join(HERE, "reports", "P3")
GAP_MIN = 1e-4

# ---- shipped models (the SAME corpus, three re-encodings, per dimension) -----
MODELS = {
    4: dict(
        pod=os.path.join(HERE, "reports/P2/models_chi/"
                         "pod_hermite_smolyak_d4qc_L5_enh-chi_Ay-chi_By.npz"),
        ranks=[1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 75, 87],
        ref_value="polish_table_qc_chi_b27_1000.json",
        ref_hermite="polish_table_qc_chi_b27_hermite_1000.json",
    ),
    8: dict(
        pod=os.path.join(HERE, "reports/P2/models_chi/pod_hermite_smolyak_"
                         "spin8qc_L5_enh-chi_Ax-chi_Ay-chi_Az-chi_Bx-chi_By-chi_Bz.npz"),
        ranks=[1, 5, 15, 30, 60, 100, 150, 200, 250, 300, 350, 394],
        ref_value="polish_table_chi8d_value_1000.json",
        ref_hermite="polish_table_chi8d_hermite_1000.json",   # pending (cluster)
    ),
}


# ------------------------------------------------------------------- helpers
def load_pod_truncated(path, r_new) -> PODHermiteSmolyak:
    """A PODHermiteSmolyak truncated to the leading ``r_new`` modes.

    Mirrors ``hermite_smolyak_pod.load_pod_hermite_smolyak`` verbatim but slices
    ``Phi[:, :r_new]`` / ``node_U[:, :r_new]`` / ``node_dU[..., :r_new]`` before
    the finalize.  ``r_new == r_shipped`` reproduces the committed loader
    bit-for-bit; ``evaluate`` at any ``r_new`` equals ``mean + Phi[:, :r_new] @
    c[:r_new]`` (leading-r' POD reconstruction).  NO re-solve, NO corpus.
    """
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


def mem_pod_bytes(r, N, nfeat, d):
    """Analytic stored-float memory of a rank-``r`` POD model (float64, dominant
    arrays): Phi (nfeat*r) + node_U (N*r) + node_dU (N*d*r) + mean (nfeat).
    Matches the on-disk .npz size to <0.5% at r_shipped."""
    return 8.0 * (nfeat * r + N * r + N * d * r + nfeat)


def mem_value_bytes(N, nfeat):
    """value-only Smolyak: node_U (N*nfeat) -- the full field per node."""
    return 8.0 * (N * nfeat)


def mem_hermite_bytes(N, nfeat, d):
    """full gradient-enhanced Hermite: node_U (N*nfeat) + node_dU (N*d*nfeat)."""
    return 8.0 * (N * (1 + d) * nfeat)


def offnode_points(box, n, seed):
    """The IDENTICAL off-node sampling as run_polish_table.random_offnode_points."""
    rng = np.random.default_rng(seed)
    ns = [cheb_param_nodes(lo, hi, 16)[0] for lo, hi in box]
    d = len(box); pts = np.empty((n, d))
    for i in range(n):
        while True:
            th = np.array([rng.uniform(lo, hi) for lo, hi in box])
            if all(np.min(np.abs(th[k] - ns[k])) > GAP_MIN for k in range(d)):
                break
        pts[i] = th
    return pts


def _stats(a):
    a = np.asarray(a, float)
    return dict(min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), p95=float(np.percentile(a, 95)),
                max=float(a.max()))


def _ref_from_json(fn):
    """Read (guess-stats, median-steps) from an existing polish-table JSON, or
    None if it is not present yet (e.g. the 8D Hermite pending on the cluster)."""
    p = os.path.join(REPDIR, fn)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return dict(guess=d["rows"]["guess"],
                median_steps=d.get("median_steps_to_certify"),
                max_steps=d.get("max_steps_to_certify"),
                n_points=d["config"].get("n_points"))


def resolve_refs(dim, N, nfeat, d):
    """Reference points (value-only Smolyak, full Hermite) resolved FRESH from
    the polish-table JSONs at call time, so a newly-arrived table (e.g. the 8D
    Hermite from the cluster) is picked up by a plain ``--replot`` with no
    re-sweep.  Memory is analytic (matches on-disk); guess/steps come from the
    1000-point polish tables.  A missing Hermite table stays a memory-only
    ``pending`` marker."""
    cfg = MODELS[dim]
    refs = {}
    rv = _ref_from_json(cfg["ref_value"])
    if rv:
        refs["value"] = dict(mem_bytes=mem_value_bytes(N, nfeat), guess=rv["guess"],
                             median_steps=rv["median_steps"], max_steps=rv["max_steps"])
    rh = _ref_from_json(cfg["ref_hermite"])
    if rh:
        refs["hermite"] = dict(mem_bytes=mem_hermite_bytes(N, nfeat, d), guess=rh["guess"],
                               median_steps=rh["median_steps"], max_steps=rh["max_steps"])
    else:
        refs["hermite"] = dict(mem_bytes=mem_hermite_bytes(N, nfeat, d), guess=None,
                               pending=cfg["ref_hermite"])
    return refs


# ------------------------------------------------------------------- sweep
def sweep_dimension(dim, n_points, seed):
    cfg = MODELS[dim]
    z = np.load(cfg["pod"], allow_pickle=True, mmap_mode="r")
    meta = json.loads(z["meta_json"].item())
    r_ship = int(z["r"]); N = int(z["node_U"].shape[0])
    fs = tuple(int(x) for x in z["field_shape"]); nfeat = int(np.prod(fs))
    box = [(float(a[0]), float(a[1])) for a in meta["box"]]
    names = list(meta["axis_names"]); fixed = meta.get("fixed", {}) or {}
    d = len(box)
    ranks = [r for r in cfg["ranks"] if r <= r_ship]

    print(f"[{dim}D] loading POD ({os.path.getsize(cfg['pod'])/1e6:.0f} MB) "
          f"r_ship={r_ship} N={N} nfeat={nfeat} d={d} ...", flush=True)
    full = load_pod_hermite_smolyak(cfg["pod"])
    prob = s3.make_problem(Na=fs[0] - 1, Nb=fs[1], Nphi=fs[2])
    pts = offnode_points(box, n_points, seed)

    res = {r: [] for r in ranks}
    cache = {}
    t0 = time.time()
    for i, th in enumerate(pts):
        sl = theta_to_slice3d(th, names, 1.0, fixed)
        asm = assemble_cached_3d(prob, sl, cache)
        scales = s3nk._block_scales(asm)
        c = np.asarray(full.coeff_model.evaluate(th)).reshape(-1)     # (r_ship,)
        for r in ranks:
            u = (full.mean + full.Phi[:, :r] @ c[:r]).reshape(prob.Ntot2d, prob.Nphi)
            res[r].append(s3nk.equil_residual_inf(asm, u, scales))
        if (i + 1) % 50 == 0:
            print(f"   {dim}D {i+1}/{n_points}  ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"[{dim}D] sweep {n_points}pts x {len(ranks)}ranks in {dt:.0f}s", flush=True)

    curve = []
    for r in ranks:
        st = _stats(res[r])
        curve.append(dict(r=r, mem_bytes=mem_pod_bytes(r, N, nfeat, d),
                          residuals=[float(x) for x in res[r]], **st))
    refs = {}
    rv = _ref_from_json(cfg["ref_value"])
    if rv:
        refs["value"] = dict(mem_bytes=mem_value_bytes(N, nfeat),
                             guess=rv["guess"], median_steps=rv["median_steps"],
                             max_steps=rv["max_steps"], n_points=rv["n_points"])
    rh = _ref_from_json(cfg["ref_hermite"])
    if rh:
        refs["hermite"] = dict(mem_bytes=mem_hermite_bytes(N, nfeat, d),
                               guess=rh["guess"], median_steps=rh["median_steps"],
                               max_steps=rh["max_steps"], n_points=rh["n_points"])
    else:
        # residual pending (cluster); keep the memory-only marker
        refs["hermite"] = dict(mem_bytes=mem_hermite_bytes(N, nfeat, d),
                               guess=None, pending=cfg["ref_hermite"])
    return dict(dim=dim, r_ship=r_ship, N=N, nfeat=nfeat, d=d,
                pod_curve=curve, refs=refs)


def run_sweep(n_points, seed):
    os.makedirs(REPDIR, exist_ok=True)
    data = dict(n_points=n_points, seed=seed, gap_min=GAP_MIN,
                dims={str(dim): sweep_dimension(dim, n_points, seed)
                      for dim in sorted(MODELS)})
    out = os.path.join(REPDIR, "guess_vs_memory.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"[sweep] wrote {out}")
    return data


# ------------------------------------------------------------------- figure
DIM_STYLE = {4: dict(marker="o", ls="-"), 8: dict(marker="^", ls="--")}
POD_C, VAL_C, HERM_C = "C2", "C0", "C1"


def build_figure(data, out_png, az_band=None):
    """``az_band=(lo, hi, label)`` overlays a faint horizontal reference band
    (the re-solved Nphi=8->16 azimuthal-resolution floor), below which the
    reported guess residuals are not azimuthally limited.  Default None -> the
    figure is unchanged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.9, 5.1))

    def _steps(marker_x, marker_y, s):
        """Annotate median Newton steps-to-certify just above a shipped marker."""
        if s is None:
            return
        ax.annotate(f"{int(round(s))}", (marker_x, marker_y), color="0.2",
                    fontsize=8.5, fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")

    for dim_s, dd in sorted(data["dims"].items(), key=lambda kv: int(kv[0])):
        dim = int(dim_s); st = DIM_STYLE[dim]
        cur = dd["pod_curve"]
        mem = np.array([c["mem_bytes"] for c in cur]) / 1e6
        med = np.array([c["median"] for c in cur])
        lo = np.array([c["min"] for c in cur])
        hi = np.array([c["max"] for c in cur])
        refs = resolve_refs(dim, dd["N"], dd["nfeat"], dd["d"])   # fresh from polish tables
        rv = refs.get("value")
        rh = refs.get("hermite")

        # (1) faint per-dimension connector: one corpus, three encodings
        #     POD floor (r_shipped) -> value -> full Hermite, in memory order.
        chain = [(mem[-1], med[-1])]
        if rv:
            chain.append((rv["mem_bytes"] / 1e6, rv["guess"]["median"]))
        if rh and rh.get("guess"):
            chain.append((rh["mem_bytes"] / 1e6, rh["guess"]["median"]))
        chain.sort()
        ax.plot([p[0] for p in chain], [p[1] for p in chain], ls=":", color="0.5",
                lw=1.1, alpha=0.7, zorder=1,
                label=("same corpus: POD$\\to$value$\\to$Hermite"
                       if dim == 4 else None))

        # (2) POD rank-sweep curve + min-max band
        ax.fill_between(mem, lo, hi, color=POD_C, alpha=0.12, lw=0, zorder=2)
        ax.plot(mem, med, color=POD_C, ms=5, lw=1.7, zorder=3,
                label=f"{dim}D POD (rank sweep)", **st)
        _steps(mem[-1], med[-1], dd.get("pod_steps"))

        # (3) reference markers: color = model (value C0 / Hermite C1),
        #     marker = dimension (4D o / 8D ^) -- same rule as the POD curves.
        if rv:
            ax.plot(rv["mem_bytes"] / 1e6, rv["guess"]["median"], marker=st["marker"],
                    color=VAL_C, ms=11, ls="none", mec="k", mew=0.5, zorder=4,
                    label=f"{dim}D value Smolyak")
            _steps(rv["mem_bytes"] / 1e6, rv["guess"]["median"], rv.get("median_steps"))
        if rh and rh.get("guess"):
            ax.plot(rh["mem_bytes"] / 1e6, rh["guess"]["median"], marker=st["marker"],
                    color=HERM_C, ms=11, ls="none", mec="k", mew=0.5, zorder=4,
                    label=f"{dim}D full Hermite")
            _steps(rh["mem_bytes"] / 1e6, rh["guess"]["median"], rh.get("median_steps"))
        elif rh:  # memory-only (residual pending on the cluster)
            ax.axvline(rh["mem_bytes"] / 1e6, color=HERM_C, ls=":", alpha=0.6, lw=1.2)
            ax.text(rh["mem_bytes"] / 1e6 * 0.85, 4.5e-3, f"{dim}D Hermite\n(pending)",
                    fontsize=7.5, color=HERM_C, ha="right", va="center")

    ylo = 5e-4
    if az_band is not None:
        lo, hi, lab = az_band
        ax.axhspan(lo, hi, color="0.55", alpha=0.20, lw=0, zorder=0)
        ax.axhline(hi, color="0.4", ls="--", lw=0.9, alpha=0.8, zorder=0)
        ax.text(0.5, np.sqrt(lo * hi), lab, transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=7.5, color="0.3")
        ylo = lo * 0.45
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("stored model memory (MB)")
    ax.set_ylabel(r"bare-guess constraint residual $\|R\|_\infty$")
    npts = data.get("n_points")
    ax.set_title(r"Reduced-basis re-encoding: guess residual vs memory "
                 r"($\chi$, $b\in[2,7]$, $\ell=5$)"
                 + f"\n{npts} off-node points; band = min--max; "
                 r"bold number = median steps to certify $\|R\|_\infty\!\leq\!10^{-10}$",
                 fontsize=9)
    ax.set_ylim(ylo, 6)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.0, ncol=2, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[figure] {out_png} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replot", action="store_true",
                    help="skip the sweep; re-plot from the existing JSON")
    args = ap.parse_args()

    jpath = os.path.join(REPDIR, "guess_vs_memory.json")
    if args.replot:
        with open(jpath) as f:
            data = json.load(f)
    else:
        data = run_sweep(args.n_points, args.seed)
    build_figure(data, os.path.join(REPDIR, "fig_guess_vs_memory.png"))


if __name__ == "__main__":
    main()
