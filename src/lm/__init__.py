"""``lm`` — namespace root for the LemaitreModels package family.

A lightweight *namespace* package: each scientific domain lives in its own
distribution / repository and installs under the shared ``lm`` prefix —
``lm.initial_data`` (this repo, LM-initial-data), and later
``lm.early_inspiral``, ``lm.ringdown``, ``lm.artwork`` from
their sibling repos.

``pkgutil.extend_path`` merges every ``lm/`` directory found on
``sys.path`` into one package, so sibling distributions coexist.  A PEP 562
``__getattr__`` lazily imports subpackages on attribute access, so::

    import lm
    lm.initial_data.solver.solver_3d      # resolves on first access

works without importing each subpackage explicitly.

Family bookkeeping: exactly ONE installed distribution may own this
``lm/__init__.py``.  While LM-initial-data is the only sibling it lives
here; when a second sibling repo is created, factor this file into a dedicated
tiny ``lm-core`` distribution that every sibling depends on.
"""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)


def __getattr__(name):
    import importlib

    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            f"(no installed 'lm.{name}' subpackage)"
        ) from exc
