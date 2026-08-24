"""Which code produced an artifact — the stamp a producer writes into its ``meta``.

Every producer in this package writes a ``meta`` block describing *what it was asked
for*: the box, the ladder, the sampler seed, the model file.  None of that says *which
code ran*, and that gap has cost a real misattribution.  On 2026-08-20 two TwoPunctures
sweeps were compared whose ``meta`` blocks were identical in all 13 keys while the trees
that produced them were **45 ``src/``-touching commits apart**; a coarse-rung shift was
attributed to the state-vector change and belonged to an evaluator fix three weeks older.
Nothing in either artifact could have revealed that.

So: stamp the tree, not just the inputs.

Two entry points, and the difference between them is deliberate:

* :func:`code_stamp` is **deterministic** — same source tree, byte-identical stamp.  It is
  what goes into *committed* artifacts (the paper's ``figdata/*.json``), because a rebuild
  that changes nothing must produce no diff.  That is the property that makes ``git diff``
  after ``make figures`` mean "the numbers moved", and a wall-clock field would destroy it.
  The *date* of a committed artifact is not lost by leaving it out: it is recoverable, and
  more reliably, from the file's own commit history.
* :func:`run_stamp` adds a UTC timestamp, and is for artifacts written **outside** version
  control — the multi-GB sweep outputs under ``$LM_REPORTS``.  Git cannot date those, so
  the artifact has to date itself, and non-determinism costs nothing because they are
  never diffed.

Both degrade rather than raise.  The package is standalone and pip-installable, so it must
work from a wheel with no ``.git`` and on a machine with no ``git`` at all; every field is
``None`` when it cannot be determined, and a stamp is never a reason for a producer to die.

``src_dirty`` is scoped to ``src/`` on purpose.  A whole-tree dirty flag reports the
author's uncommitted ``paper.tex`` and reads as "these numbers are untrustworthy", which is
false — prose cannot change a solve.  That exact false alarm happened on 2026-08-21.
"""
from __future__ import annotations

import datetime
import os
import subprocess

#: ``src/lemaitre/initial_data/conformally_flat_puncture/`` → four levels up is the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))


def _git(*args, timeout=5.0):
    """``git -C <repo> <args>`` → stripped stdout, or ``None`` if that cannot be answered.

    Returns ``None`` for every failure mode rather than distinguishing them: no ``git`` on
    the machine, no ``.git`` in the tree (a wheel install), a git that hangs, a non-zero
    exit.  A caller that wanted to tell those apart would be doing something this module
    is deliberately not for.
    """
    try:
        out = subprocess.run(("git", "-C", _REPO) + args, capture_output=True,
                             text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def src_tree():
    """SHA of the ``src/`` **tree** at HEAD, or ``None``.

    The tree rather than the commit, because the question a stamp has to answer is "was the
    code the same?"  Two commits that differ only in ``paper/`` or ``docs/`` share a
    ``src/`` tree and should compare equal; ``HEAD`` would say they differ.
    """
    return _git("rev-parse", "HEAD:src")


def src_dirty():
    """``True``/``False`` if ``src/`` has uncommitted changes, ``None`` if unknowable.

    Scoped to ``src/`` — see the module docstring on why a whole-tree flag lies.
    """
    out = _git("status", "--porcelain", "--", "src")
    if out is None:
        # Distinguish "clean" from "cannot tell": a clean tree gives empty stdout, which
        # ``_git`` also returns as None.  Re-ask for something non-empty on success.
        return None if _git("rev-parse", "--is-inside-work-tree") is None else False
    return bool(out)


def package_version():
    """Installed distribution version, or ``None`` — works with no git at all."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                                    # pragma: no cover - py<3.8
        return None
    for dist in ("lm-initial-data-conformally-flat-puncture",
                 "lmid-conformally-flat-puncture"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    return None


def code_stamp():
    """Deterministic record of the code that produced an artifact.

    Same source tree → byte-identical dict, so it is safe to embed in a committed
    artifact without making every rebuild a diff.  Keys are always present, with ``None``
    where the answer is unavailable, so a consumer can test a field without a ``.get``
    dance and cannot mistake "absent" for "clean".
    """
    return {"src_tree": src_tree(),
            "src_dirty": src_dirty(),
            "version": package_version()}


def run_stamp():
    """:func:`code_stamp` plus ``utc``, for artifacts git will never see.

    Not deterministic, by design — see the module docstring.
    """
    stamp = code_stamp()
    stamp["utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    return stamp


def same_code(a, b):
    """Did two stamps come from the same source tree?  ``None`` when undecidable.

    ``None`` rather than ``False`` when either side lacks a usable tree — a legacy
    artifact written before stamping existed is *unknown*, not *different*, and a
    consumer that refuses on a mismatch must not refuse on an absence.  A dirty side is
    undecidable too: the tree SHA describes HEAD, not what actually ran.
    """
    ta, tb = (a or {}).get("src_tree"), (b or {}).get("src_tree")
    if ta is None or tb is None:
        return None
    if (a or {}).get("src_dirty") or (b or {}).get("src_dirty"):
        return None
    return ta == tb
