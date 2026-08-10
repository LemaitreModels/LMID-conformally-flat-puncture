"""Self-containment guard for ``lemaitre.initial_data.conformally_flat``.

The package is **standalone**: it depends on `jax`, `numpy`, `scipy`,
`matplotlib` and the dependency-free `lemaitre` / `lemaitre-initial-data`
namespace packages, and on nothing else.  It must never reach back into the
BBHFM monorepo it was migrated out of.

Individual modules carry their own ``test_standalone_imports`` spot-checks; this
file is the tree-wide version — it AST-parses every module under ``src/`` so a
new file cannot slip a forbidden import in unnoticed.  It also pins the
namespace ownership, which is the thing most easily broken by an editor
helpfully creating a missing ``__init__.py``.
"""
import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
PKG = os.path.join(SRC, "lemaitre", "initial_data", "conformally_flat")

#: Import roots that would break standalone-ness.  ``src`` covers the monorepo's
#: ``src.bbhfm`` / ``src.*`` form as well as a bare ``import src``.
FORBIDDEN = ("bbhfm", "src", "context", "torch", "nrpy")

#: The only namespace prefix an absolute intra-family import may use.
FAMILY = "lemaitre"


def _modules():
    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _import_roots(path):
    """Every top-level module name imported by ``path``, with the dotted form."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], a.name
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 is a RELATIVE import (`from . import`,
            # `from ..solver import`) — intra-package by construction, and the
            # form the package is required to use internally.  Never flagged.
            if node.level == 0 and node.module:
                yield node.module.split(".")[0], node.module


ALL_MODULES = sorted(_modules())


def test_the_package_tree_was_found():
    """Guard the guard: a moved package must not silently scan nothing."""
    assert len(ALL_MODULES) > 50, f"only {len(ALL_MODULES)} modules under {PKG}"


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda p: os.path.relpath(p, PKG))
def test_no_forbidden_imports(path):
    """No module reaches outside the standalone dependency set."""
    for root, dotted in _import_roots(path):
        assert root not in FORBIDDEN, (
            f"{os.path.relpath(path, REPO)} imports {dotted!r} — "
            f"{root!r} is forbidden (the package is standalone)"
        )


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda p: os.path.relpath(p, PKG))
def test_absolute_family_imports_are_own_leaf(path):
    """An absolute `lemaitre.*` import may only name this leaf.

    Reaching into a sibling leaf (`lemaitre.initial_data.curved`) would invert
    the dependency: `curved` reuses this package, not the other way round.
    """
    own = f"{FAMILY}.initial_data.conformally_flat"
    for root, dotted in _import_roots(path):
        if root == FAMILY:
            assert dotted == own or dotted.startswith(own + "."), (
                f"{os.path.relpath(path, REPO)} imports {dotted!r}; this "
                f"distribution may only import {own}.*"
            )


def test_namespace_levels_are_not_owned_here():
    """This distribution ships ONE package: its own leaf.

    `lemaitre/__init__.py` belongs to the `lemaitre` core and
    `lemaitre/initial_data/__init__.py` to the `lemaitre-initial-data` umbrella.
    A copy of either here would shadow the owner non-deterministically, which is
    exactly the collision the family split was made to prevent.
    """
    for level in (
        os.path.join(SRC, "lemaitre", "__init__.py"),
        os.path.join(SRC, "lemaitre", "initial_data", "__init__.py"),
    ):
        assert not os.path.exists(level), (
            f"{os.path.relpath(level, REPO)} exists — this repo must not own a "
            f"namespace level (see docs/STRUCTURE.md, 'Family bookkeeping')"
        )
    assert os.path.isfile(os.path.join(PKG, "__init__.py")), \
        "the leaf package itself must have an __init__.py"


def test_no_stale_lm_namespace():
    """The pre-rename `lm` namespace root is gone, and nothing re-creates it."""
    assert not os.path.exists(os.path.join(SRC, "lm")), \
        "src/lm/ is the pre-rename namespace root — it must not come back"


# ==========================================================================
# Family wiring — catches a silently-degraded editable install
# ==========================================================================
INSTALL_HINT = (
    "\n\nThe two-level `lemaitre` namespace is not fully wired. This is almost "
    "always the editable-install mode: setuptools' MODERN editable mode installs "
    "each distribution behind a meta-path finder, and because `PathFinder` runs "
    "BEFORE those finders, the leaf packages' bare `lemaitre/initial_data/` "
    "directories win as PEP 420 namespace portions and the umbrella's real "
    "`lemaitre/initial_data/__init__.py` is never loaded. Direct imports still "
    "work; lazy attribute access does not.\n\n"
    "Reinstall every family distribution with the static-path mode:\n"
    "    pip install -e . --config-settings editable_mode=compat\n\n"
    "Normal (non-editable) installs are unaffected — they all land in one "
    "site-packages tree. See docs/STRUCTURE.md, 'Family bookkeeping'."
)


def test_two_level_namespace_is_live():
    """Both namespace levels load their OWNER's `__init__.py`, not a stub.

    `__file__ is None` means Python built an implicit PEP 420 namespace package
    instead, which silently costs the lazy `__getattr__` the whole family idiom
    is built on.
    """
    import lemaitre
    import lemaitre.initial_data

    assert lemaitre.__file__ is not None, "lemaitre/__init__.py not loaded" + INSTALL_HINT
    assert lemaitre.initial_data.__file__ is not None, \
        "lemaitre/initial_data/__init__.py not loaded" + INSTALL_HINT


def test_lazy_attribute_access():
    """`import lemaitre as lm; lm.initial_data.conformally_flat...` resolves.

    The documented idiom, with no explicit submodule import — this is what the
    PEP 562 `__getattr__` on each namespace level exists to provide.
    """
    import lemaitre as lm

    try:
        pkg = lm.initial_data.conformally_flat
    except AttributeError as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(f"lazy access failed: {exc}{INSTALL_HINT}") from exc
    assert pkg.__name__ == "lemaitre.initial_data.conformally_flat"
    assert lm.initial_data.conformally_flat.solver.__name__.endswith(".solver")
