"""LM-initial-data paper — figure-data driver: check presence, dedup, (re)build.

The committed figure data lives in one place, ``figdata/<stem>.json``, one per figure.  Each is
produced by its own ``<stem>_data.py`` from raw run SOURCE artifacts declared in ``registry.py``.
Sources shared by several figures (e.g. fig08/fig09) are listed once and never rebuilt
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
import json
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


def _figdata_tag(stem):
    """"figdata OK" / "figdata MISSING" / "figdata STALE" for one figure.

    STALE means the json is present but lacks a top-level key its plotter reads
    (registry.FIGURES[stem]["keys"]) — i.e. it predates a block being added to its producer, so
    ``make figures`` would die inside the plotter.  Reporting it as OK is what let fig02's
    missing ``Q_wall_q`` sit unnoticed.
    """
    p = fd.figdata_path(stem)
    if not os.path.exists(p):
        return "figdata MISSING", None
    try:
        with open(p) as f:
            miss = fd.missing_keys(stem, json.load(f))
    except (ValueError, OSError) as e:                # unreadable/corrupt json
        return "figdata STALE", str(e)
    return ("figdata STALE", f"missing keys {miss}") if miss else ("figdata OK", None)


def check():
    st = _source_status()
    print("=== figures ===")
    n_ready = n_partial = n_stale = 0
    for stem, spec in reg.FIGURES.items():
        srcs = spec.get("sources", [])
        if spec.get("inline"):
            miss = []
        else:
            miss = [k for k in srcs if not st[k][0]]
        tag, why = _figdata_tag(stem)
        if tag == "figdata STALE":
            n_stale += 1
        if spec.get("inline"):
            state = "inline"
        elif not miss:
            state = "buildable"; n_ready += 1
        else:
            state = f"blocked ({len(miss)}/{len(srcs)} sources missing)"; n_partial += 1
        print(f"  {stem:32s} {tag:16s} {state}")
        if why:
            print(f"        - STALE: {why}; rebuild with --fig {stem} --force")
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
    print(f"\n{n_ready} buildable, {n_partial} blocked on missing (cluster) sources"
          f"{f', {n_stale} STALE' if n_stale else ''}.")


def build(stem, force=False):
    # A STALE figdata is rebuilt without --force: keeping it would fail the plotter anyway.
    stale = _figdata_tag(stem)[0] == "figdata STALE"
    if os.path.exists(fd.figdata_path(stem)) and not force and not stale:
        print(f"[{stem}] figdata present — skip (use --force to rebuild)")
        return True
    if stale and not force:
        print(f"[{stem}] figdata STALE ({_figdata_tag(stem)[1]}) — rebuilding")
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
