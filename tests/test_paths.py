"""Acceptance tests for :mod:`lemaitre.initial_data.conformally_flat_puncture.paths` — the heavy-corpora resolver.

The solve stores, χ surrogate models, and per-figure sweep outputs are multi-GB
and gitignored, so they live outside version control.  Before this module there
were two different roots — producers used ``<pipeline>/reports`` (off their own
``__file__``) while ``paper/figures/_figdata`` used ``<repo_root>/reports`` — so a
figure could not see the output of the producer that fed it.  What must hold:

* ``$LM_REPORTS`` wins when set, and is absolutised with ``~`` expanded;
* the default is the producers' historical in-package path, so their behaviour is
  unchanged when the variable is unset;
* both halves of the pipeline agree on the answer;
* ``require`` names the missing artifact, the root searched, and the env var.

Pure stdlib: these must pass on the AVX-less login node, where ``import jax``
aborts.
"""
import os

import pytest

from lemaitre.initial_data.conformally_flat_puncture import paths
from lemaitre.initial_data.conformally_flat_puncture.pipeline import production_model as pm


def _scoped(base):
    """The paper's root under ``base``: the base itself for ``"paper-v1"``, else
    ``<base>/<scheme>`` (see ``paths`` — the tree is kept apart by discretisation)."""
    return os.path.normpath(os.path.join(str(base), paths.scheme_subdir(pm.SCHEME)))


# ==========================================================================
# P-U1 — resolution order
# ==========================================================================
def test_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "corpora"))
    assert paths.reports_base() == str(tmp_path / "corpora")
    assert paths.reports_root() == _scoped(tmp_path / "corpora")


def test_default_is_the_in_package_path(monkeypatch):
    """Unset -> the producers' historical root, so nothing changes for them."""
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    root = paths.reports_base()
    assert root == paths.DEFAULT_ROOT
    assert paths.reports_root() == _scoped(paths.DEFAULT_ROOT)
    # DEFAULT_ROOT is derived from paths.__file__, so it tracks the package
    # location: `<...>/lemaitre/initial_data/conformally_flat_puncture/pipeline/reports`.
    assert root.endswith(
        os.path.join("conformally_flat_puncture", "pipeline", "reports"))


def test_default_is_independent_of_cwd(tmp_path, monkeypatch):
    """A driver that chdirs (make_figdata runs data scripts from paper/figures)
    must not silently pick up a different tree."""
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()            # a decoy beside the cwd
    assert paths.reports_root() == _scoped(paths.DEFAULT_ROOT)


def test_root_is_absolute_and_expands_user(monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, "~/some-corpora")
    root = paths.reports_base()
    assert os.path.isabs(root) and "~" not in root
    assert root == os.path.join(os.path.expanduser("~"), "some-corpora")
    assert paths.reports_root() == _scoped(root)


def test_the_tree_is_kept_apart_by_scheme(tmp_path, monkeypatch):
    """paper-v1 artifacts sit in the base (where everything before 2026-10-14 lives);
    any other scheme in ``<base>/<scheme>``, so a v2 producer or figure cannot read a
    paper-v1 source by accident — it is absent, and the error names the producer."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert paths.reports_root(scheme="paper-v1") == str(tmp_path)
    assert paths.reports_root(scheme="v2") == str(tmp_path / "v2")
    assert paths.reports_root() == paths.reports_root(scheme=pm.SCHEME)
    (tmp_path / "x.json").write_text("{}")                     # a paper-v1 artifact
    assert not os.path.exists(os.path.join(paths.reports_root(scheme="v2"), "x.json"))


def test_a_filesystem_root_base_stays_a_root(monkeypatch):
    """``LM_REPORTS=/`` must not collapse to "" (relative paths, a failing mkdir)."""
    monkeypatch.setenv(paths.ENV_VAR, os.sep)
    assert paths.reports_root(scheme="paper-v1") == os.sep
    assert paths.reports_root(scheme="v2") == os.path.join(os.sep, "v2")


# ==========================================================================
# P-U2 — joining and directory creation
# ==========================================================================
def test_reports_path_joins(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert paths.reports_path("P3", "x.json") == os.path.join(_scoped(tmp_path), "P3", "x.json")


def test_create_flags(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "root"))
    assert not os.path.isdir(str(tmp_path / "root"))
    paths.reports_root(create=True)
    assert os.path.isdir(_scoped(tmp_path / "root"))
    paths.reports_path("P3", "deep", "y.json", create_parent=True)
    assert os.path.isdir(os.path.join(_scoped(tmp_path / "root"), "P3", "deep"))


# ==========================================================================
# P-U3 — `require` is actionable
# ==========================================================================
def test_require_returns_existing(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    d = os.path.join(_scoped(tmp_path), "P3")
    os.makedirs(d)
    open(os.path.join(d, "ok.json"), "w").write("{}")
    assert paths.require("P3", "ok.json") == os.path.join(d, "ok.json")


def test_require_error_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    with pytest.raises(FileNotFoundError) as ei:
        paths.require("P3", "absent.json", hint="run the sweep first")
    msg = str(ei.value)
    assert "absent.json" in msg                 # what is missing
    assert str(tmp_path) in msg                 # where we looked
    assert paths.ENV_VAR in msg                 # the knob that controls it
    assert "run the sweep first" in msg         # the caller's hint
    assert "docs/DATA.md" in msg                # where to read more


def test_require_error_flags_unset_env(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError) as ei:
        paths.require("P3", "definitely-absent-xyz.json")
    assert "unset" in str(ei.value)


# ==========================================================================
# P-T1 — the two halves of the pipeline agree, and nothing regresses
# ==========================================================================
def test_producers_use_the_resolver():
    """No producer may rebuild a ``reports/`` path off its own ``__file__``."""
    import lemaitre.initial_data.conformally_flat_puncture.pipeline as pipe
    pdir = os.path.dirname(os.path.abspath(pipe.__file__))
    offenders = [fn for fn in sorted(os.listdir(pdir))
                 if fn.endswith(".py")
                 and ('HERE, "reports"' in open(os.path.join(pdir, fn)).read()
                      or 'HERE, "reports/' in open(os.path.join(pdir, fn)).read())]
    assert not offenders, offenders


def test_figure_side_agrees_with_producers(monkeypatch, tmp_path):
    """``_figdata`` must resolve to the same tree the producers write to."""
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figdir = os.path.join(repo, "paper", "figures")
    if figdir not in sys.path:
        sys.path.insert(0, figdir)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    import _figdata as fd
    assert fd.reports_root() == paths.reports_root() == _scoped(tmp_path)
    # and a source path lands under it (resolved per call, not at import)
    assert fd.source_path("sweep_3d").startswith(_scoped(tmp_path))


def test_standalone_imports():
    """Imported by login-node figure scripts, so it must not pull jax/numpy."""
    import ast
    tree = ast.parse(open(paths.__file__).read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    for forbidden in ("jax", "numpy", "scipy", "matplotlib", "bbhfm", "torch", "nrpy"):
        assert forbidden not in imported, f"paths must not import {forbidden}"
