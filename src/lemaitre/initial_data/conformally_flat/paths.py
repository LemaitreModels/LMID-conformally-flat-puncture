"""Filesystem location of the heavy run corpora (the ``reports/`` tree).

The solve stores, χ surrogate models, and per-figure sweep outputs are multi-GB
and gitignored (see ``docs/DATA.md``), so they live outside version control — and
on a cluster, outside the repo entirely.

Before this module there were **two different roots**, which is why a producer's
output was invisible to the figure that consumed it:

* producers built ``os.path.join(HERE, "reports", …)`` off their own ``__file__``
  → ``src/lemaitre/initial_data/conformally_flat/pipeline/reports/``
* ``paper/figures/_figdata`` used ``<repo_root>/reports``

Both now resolve through :func:`reports_root`, so one setting moves the whole tree
and the two halves of the pipeline always agree:

    export LM_REPORTS=/home/USER/lm-reports
    python -m lemaitre.initial_data.conformally_flat.pipeline.run_3d_sweep

Resolution order:

1. ``$LM_REPORTS`` — explicit, ``~`` expanded, made absolute.  Use this on a
   cluster, where the corpora sit on scratch rather than beside the code.
2. ``<pipeline>/reports`` — the producers' historical location, so their
   behaviour is unchanged when the variable is unset.

Note (2) makes the default relative to the *package source*.  That is a
deliberate compatibility choice, not a recommendation: a pip-installed package
may sit read-only in site-packages, and corpora are data, not code.  Set
``$LM_REPORTS`` for any real run.
"""
from __future__ import annotations

import os

ENV_VAR = "LM_REPORTS"

#: The producers' historical root — ``src/lemaitre/initial_data/conformally_flat/pipeline/reports``.
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pipeline", "reports")


def reports_root(create: bool = False) -> str:
    """Absolute path to the ``reports/`` tree (see the module docstring).

    ``create=True`` makes the directory — pass it from producers that write.
    """
    env = os.environ.get(ENV_VAR)
    root = os.path.abspath(os.path.expanduser(env)) if env else DEFAULT_ROOT
    if create:
        os.makedirs(root, exist_ok=True)
    return root


def reports_path(*parts: str, create_parent: bool = False) -> str:
    """Join ``parts`` onto the reports root; optionally mkdir the parent."""
    p = os.path.join(reports_root(), *parts)
    if create_parent:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def require(*parts: str, hint: str = "") -> str:
    """Resolve a corpus artifact, raising an actionable error if it is absent.

    Producers and figure scripts use this so a missing heavy input names both the
    resolved path and the setting that controls it, instead of surfacing as a bare
    ``FileNotFoundError`` from deep inside a loader.
    """
    p = reports_path(*parts)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"missing corpus artifact: {os.path.join(*parts)}\n"
            f"  looked in: {reports_root()}   (${ENV_VAR}"
            f"{'' if os.environ.get(ENV_VAR) else ' unset -> the in-package default'})\n"
            + (f"  {hint}\n" if hint else "")
            + "  see docs/DATA.md (`make models` / `make oracle` build the heavy tier)"
        )
    return p
