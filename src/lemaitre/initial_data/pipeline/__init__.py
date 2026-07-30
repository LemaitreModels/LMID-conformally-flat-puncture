"""Canonical producers/builders for the manuscript figures and shipped
models.  Runnable as scripts; importable as ``lemaitre.initial_data.pipeline.*``.
"""
def __getattr__(name):
    """Lazily import a submodule on attribute access (so the parent package's
    modules resolve as attributes without an explicit import)."""
    import importlib

    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
