"""PARASOL paper — figure-data driver: check presence, dedup, (re)build.

The committed figure data lives in one place, ``figdata/<stem>.json``, one per figure.  Each is
produced by its own ``<stem>_data.py`` from raw run SOURCE artifacts declared in ``registry.py``.
Sources shared by several figures (e.g. fig09/fig10) are listed once and never rebuilt
twice.

Usage
-----
  python make_figdata.py --check          # present/MISSING matrix for every figure + source (dedup)
  python make_figdata.py --all            # build every figdata/*.json that is missing
  python make_figdata.py --all --force    # rebuild all, even if present
  python make_figdata.py --fig fig03      # build one (stem or number; --force to overwrite)

A figure is "buildable" iff all its raw sources are present under reports/.  Missing sources are
reported with the exact producer command and where it runs (laptop/cluster) — this is the same
list the cluster prompt fills.  ``--check`` never runs anything.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import registry as reg          # noqa: E402
import _figdata as fd           # noqa: E402


def _stem(arg):
    """Accept 'fig03', '3', or a full stem; return the canonical stem."""
    if arg in reg.FIGURES:
        return arg
    hits = [s for s in reg.FIGURES if s.startswith(f"fig{int(arg):02d}_")] if arg.isdigit() \
        else [s for s in reg.FIGURES if s.startswith(arg)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"ambiguous/unknown figure {arg!r}; known: {', '.join(reg.FIGURES)}")


def _source_status():
    """key -> (present bool, meta) for every source, evaluated once (dedup)."""
    return {k: (fd.have_source(k), m) for k, m in reg.SOURCES.items()}


def check():
    st = _source_status()
    print("=== figures ===")
    n_ready = n_partial = 0
    for stem, spec in reg.FIGURES.items():
        srcs = spec.get("sources", [])
        have_fig = os.path.exists(fd.figdata_path(stem))
        if spec.get("inline"):
            miss = []
        else:
            miss = [k for k in srcs if not st[k][0]]
        tag = "figdata OK" if have_fig else "figdata MISSING"
        if spec.get("inline"):
            state = "inline"
        elif not miss:
            state = "buildable"; n_ready += 1
        else:
            state = f"blocked ({len(miss)}/{len(srcs)} sources missing)"; n_partial += 1
        print(f"  {stem:32s} {tag:16s} {state}")
        for k in miss:
            m = st[k][1]
            print(f"        - needs {k:22s} [{m['where']}"
                  f"{'; PENDING' if m.get('status') == 'pending' else ''}]  {m['producer']}")
    print(f"\n=== sources (dedup) ===  {sum(v[0] for v in st.values())}/{len(st)} present")
    for k, (ok, m) in st.items():
        figs = ", ".join(m["figures"])
        mark = "ok " if ok else "MISS"
        print(f"  [{mark}] {k:22s} -> {figs}")
    # committed bundle size
    if os.path.isdir(fd.FIGDATA):
        tot = sum(os.path.getsize(os.path.join(fd.FIGDATA, f))
                  for f in os.listdir(fd.FIGDATA) if f.endswith(".json"))
        print(f"\nfigdata/ committed bundle: {tot/1e3:.0f} kB "
              f"({len([f for f in os.listdir(fd.FIGDATA) if f.endswith('.json')])} files)")
    print(f"\n{n_ready} buildable, {n_partial} blocked on missing (cluster) sources.")


def build(stem, force=False):
    if os.path.exists(fd.figdata_path(stem)) and not force:
        print(f"[{stem}] figdata present — skip (use --force to rebuild)")
        return True
    script = os.path.join(HERE, f"{stem}_data.py")
    if not os.path.exists(script):
        print(f"[{stem}] no data script {os.path.basename(script)} yet — skip")
        return False
    print(f"[{stem}] building via {os.path.basename(script)} ...")
    r = subprocess.run([sys.executable, script], cwd=HERE)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report presence/dedup; run nothing")
    ap.add_argument("--all", action="store_true", help="build every missing figdata json")
    ap.add_argument("--fig", help="build one figure (stem, number, or prefix)")
    ap.add_argument("--force", action="store_true", help="rebuild even if figdata present")
    args = ap.parse_args()

    if args.check or not (args.all or args.fig):
        check()
        return
    stems = [_stem(args.fig)] if args.fig else list(reg.FIGURES)
    ok = sum(build(s, args.force) for s in stems)
    print(f"\nbuilt/kept {ok}/{len(stems)}")


if __name__ == "__main__":
    main()
