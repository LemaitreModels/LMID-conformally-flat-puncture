"""PARASOL paper — table-data driver (the ``figures/make_figdata.py`` analogue).

Every paper table ``tabNN_<name>`` has, exactly like a figure:
  * a DATA SCRIPT ``tabNN_<name>_data.py`` — RECOMPUTES the numbers from the solver
    and writes ``tabdata/tabNN_<name>.json`` (a build output, gitignored);
  * a RENDERER    ``tabNN_<name>_tex.py``  — reads ONLY that json and writes
    ``tabNN_<name>.tex``, the ``ruledtabular`` block paper.tex ``\\input``s
    (committed, so ``pdflatex`` works out of the box — as the figure PDFs are).

Unlike the figures, both tables recompute on the laptop in seconds from the solver
alone: no ``reports/`` corpus, no cluster, so there is no SOURCES dedup graph to
declare. The caption of each table stays in paper.tex — it is author-owned prose.

Usage
-----
  python make_tabdata.py --check           # present/MISSING for every table
  python make_tabdata.py --all             # recompute every missing tabdata json
  python make_tabdata.py --all --force     # recompute all
  python make_tabdata.py --tab tab01       # one table (stem, number, or prefix)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _tabdata as td          # noqa: E402

TABLES = {
    "tab01_tangent_operator": dict(
        paper="Table I",
        what="Eq. (tangent) vs central FD of the certified 3-D Newton-Krylov solve",
        producer="pipeline.run_tangent_verification.operator_tangents",
        where="laptop", cost="~1 s"),
    "tab02_tangent_surrogate": dict(
        paper="Table II",
        what="exposed surrogate gradient vs FD of the surrogate and vs Eq. (tangent)",
        producer="pipeline.run_tangent_verification.surrogate_tangents",
        where="laptop", cost="~15 s"),
}


def _stem(arg):
    if arg in TABLES:
        return arg
    hits = [s for s in TABLES if s.startswith(f"tab{int(arg):02d}_")] if arg.isdigit() \
        else [s for s in TABLES if s.startswith(arg)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"ambiguous/unknown table {arg!r}; known: {', '.join(TABLES)}")


def check():
    print("=== tables ===")
    for stem, m in TABLES.items():
        have_json = os.path.exists(td.tabdata_path(stem))
        have_tex = os.path.exists(os.path.join(HERE, f"{stem}.tex"))
        print(f"  {stem:26s} {m['paper']:9s} "
              f"tabdata {'OK     ' if have_json else 'MISSING'} "
              f"tex {'OK     ' if have_tex else 'MISSING'} "
              f"[{m['where']}, {m['cost']}]")
        print(f"       {m['what']}")


def build(stem, force=False):
    if os.path.exists(td.tabdata_path(stem)) and not force:
        print(f"[{stem}] tabdata present — skip (use --force to recompute)")
        return True
    script = os.path.join(HERE, f"{stem}_data.py")
    if not os.path.exists(script):
        print(f"[{stem}] no data script {os.path.basename(script)} — skip")
        return False
    print(f"[{stem}] recomputing via {os.path.basename(script)} ...")
    return subprocess.run([sys.executable, script], cwd=HERE).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tab")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.check or not (args.all or args.tab):
        check()
        return
    stems = [_stem(args.tab)] if args.tab else list(TABLES)
    ok = sum(build(s, args.force) for s in stems)
    print(f"\nbuilt/kept {ok}/{len(stems)}")


if __name__ == "__main__":
    main()
