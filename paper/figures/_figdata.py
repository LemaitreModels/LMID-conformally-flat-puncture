"""LM-initial-data paper — figure-data I/O helper (shared by the data scripts and plotters).

Two jobs:
  * PLOTTERS call ``load(stem)`` to read the committed, plot-ready ``figdata/<stem>.json``.
    That is all a plotter ever touches — no ``reports/``, no ``jax``, no models.
  * DATA SCRIPTS call ``source(key)`` to locate a raw run artifact under ``reports/`` (resolved
    through ``registry.SOURCES``) and ``dump(stem, obj)`` to write the distilled arrays.

``source(key)`` raises a clear, actionable error when the raw artifact is absent (naming the
producer command and whether it runs on the laptop or the cluster), so a missing input never
fails silently.  ``dump`` is numpy-aware.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGDATA = os.path.join(HERE, "figdata")

# The heavy corpora live outside the repo.  This used to be <repo_root>/reports
# while the producers wrote to <pipeline>/reports, so a figure could not see the
# output of the producer that fed it; both now resolve through the one setting,
# $LM_REPORTS.  See lm.initial_data.paths and docs/DATA.md.
from lm.initial_data.paths import reports_root  # noqa: E402  (pure stdlib; no jax)

REPORTS = reports_root()          # display convenience; source_path() resolves live

import sys
sys.path.insert(0, HERE)
import registry as _reg  # noqa: E402


def _to_jsonable(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


# ------------------------------------------------------------------ committed figdata --------
def figdata_path(stem):
    return os.path.join(FIGDATA, f"{stem}.json")


def dump(stem, obj):
    """Write the distilled, plot-ready arrays for a figure to figdata/<stem>.json."""
    os.makedirs(FIGDATA, exist_ok=True)
    with open(figdata_path(stem), "w") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)
    return figdata_path(stem)


def missing_keys(stem, d):
    """Declared plotter keys (registry.FIGURES[stem]['keys']) absent from a loaded figdata."""
    return [k for k in _reg.keys_for(stem) if k not in d]


def load(stem):
    """Read figdata/<stem>.json (the ONLY thing a plotter reads).

    Also verifies the top-level keys the plotter needs are present.  A figdata that predates a
    block being added to its producer loads fine and then dies deep in the plotter with a bare
    ``KeyError``; the check turns that into one actionable message naming the rebuild command.
    """
    p = figdata_path(stem)
    hint = (f"  build it with:  python {stem}_data.py   "
            f"(or: python make_figdata.py --fig {stem} --force)")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"missing committed figure data: {os.path.relpath(p, REPO_ROOT)}\n{hint}")
    with open(p) as f:
        d = json.load(f)
    miss = missing_keys(stem, d)
    if miss:
        srcs = ", ".join(_reg.sources_for(stem)) or "(none)"
        raise KeyError(
            f"stale figure data {os.path.relpath(p, REPO_ROOT)}: missing {miss} "
            f"(has {sorted(d)})\n{hint}\n  raw sources: {srcs}")
    return d


# ------------------------------------------------------------------ raw sources --------------
def source_meta(key):
    if key not in _reg.SOURCES:
        raise KeyError(f"unknown source key {key!r} (not in registry.SOURCES)")
    return _reg.SOURCES[key]


def source_path(key):
    """Absolute path to a raw source artifact under reports/ (may or may not exist).

    Resolved on every call rather than against the import-time ``REPORTS``, so a
    driver (or a test) can relocate the tree via ``$LM_REPORTS`` after import.
    """
    return os.path.join(reports_root(), source_meta(key)["reports"])


def have_source(key):
    return os.path.exists(source_path(key))


def source(key):
    """Resolve a raw source artifact to a path, or raise an actionable error if absent."""
    p = source_path(key)
    if not os.path.exists(p):
        m = source_meta(key)
        raise FileNotFoundError(
            f"missing raw source {key!r}: {os.path.relpath(p, REPO_ROOT)}\n"
            f"  produce it: {m['producer']}   [{m['where']}"
            f"{'; PENDING' if m.get('status') == 'pending' else ''}]\n"
            f"  (heavy cluster runs: see the paper cluster prompt / figures/README.md)")
    return p


def load_source(key):
    """Convenience: json.load a raw source (for the pure-reshape data scripts)."""
    with open(source(key)) as f:
        return json.load(f)
