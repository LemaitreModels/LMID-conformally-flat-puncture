"""Filesystem location of the heavy run corpora (the ``reports/`` tree).

The solve stores, χ surrogate models, and per-figure sweep outputs are multi-GB
and gitignored (see ``docs/DATA.md``), so they live outside version control — and
on a cluster, outside the repo entirely.

Before this module there were **two different roots**, which is why a producer's
output was invisible to the figure that consumed it:

* producers built ``os.path.join(HERE, "reports", …)`` off their own ``__file__``
  → ``src/lemaitre/initial_data/conformally_flat_puncture/pipeline/reports/``
* ``paper/figures/_figdata`` used ``<repo_root>/reports``

Both now resolve through :func:`reports_root`, so one setting moves the whole tree
and the two halves of the pipeline always agree:

    export LM_REPORTS=/path/to/lm-reports
    python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.run_3d_sweep

Resolution order:

1. ``$LM_REPORTS`` — explicit, ``~`` expanded, made absolute.  Use this on a
   cluster, where the corpora sit on scratch rather than beside the code.
2. ``<pipeline>/reports`` — the producers' historical location, so their
   behaviour is unchanged when the variable is unset.

**The tree is kept apart by discretisation.**  The returned root is the base above
for ``"paper-v1"`` (where every artifact built before 2026-10-14 lives) and
``<base>/<scheme>`` for any other ``production_model.SCHEME``, so after the paper moved
to ``"v2"`` every producer writes, and every figure-data script reads, under
``<base>/v2``.  A ``"paper-v1"`` source therefore cannot be read under ``"v2"`` by
accident: it is simply absent there, and the error names the producer.
:func:`reports_base` is the unscoped base, for tools that must see both.

Note (2) makes the default relative to the *package source*.  That is a
deliberate compatibility choice, not a recommendation: a pip-installed package
may sit read-only in site-packages, and corpora are data, not code.  Set
``$LM_REPORTS`` for any real run.
"""
from __future__ import annotations

import os

ENV_VAR = "LM_REPORTS"

#: The producers' historical root — ``src/lemaitre/initial_data/conformally_flat_puncture/pipeline/reports``.
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pipeline", "reports")


#: The scheme whose artifacts sit directly in the base (everything built before the
#: subdirectories existed).  Any other scheme gets ``<base>/<scheme>``.
BASE_SCHEME = "paper-v1"


def reports_base() -> str:
    """The unscoped base: ``$LM_REPORTS``, else :data:`DEFAULT_ROOT` (absolute)."""
    env = os.environ.get(ENV_VAR)
    return os.path.abspath(os.path.expanduser(env)) if env else DEFAULT_ROOT


def scheme_subdir(scheme: str) -> str:
    """``""`` for :data:`BASE_SCHEME`, else the scheme name — the directory under the
    base that holds that discretisation's artifacts."""
    return "" if scheme == BASE_SCHEME else scheme


def reports_root(create: bool = False, scheme: str | None = None) -> str:
    """Absolute path to the ``reports/`` tree of ``scheme`` (default: the paper's,
    ``production_model.SCHEME``) — see the module docstring.

    ``create=True`` makes the directory — pass it from producers that write.
    """
    if scheme is None:
        from .pipeline import production_model as pm      # stdlib-only; no jax
        scheme = pm.SCHEME
    root = os.path.normpath(os.path.join(reports_base(), scheme_subdir(scheme)))
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
