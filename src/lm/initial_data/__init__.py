"""lm.initial_data — certified, differentiable, parametric black-hole
initial data via spectral collocation (the PARASOL paper package).

Standalone: depends only on ``jax``, ``numpy``, ``scipy``, ``matplotlib``.

  * spatial elliptic solver         :mod:`lm.initial_data.solver`
  * parameter-space ROM             :mod:`lm.initial_data.parametric`
  * applications (targeting, …)     :mod:`lm.initial_data.applications`
  * TwoPunctures validation         :mod:`lm.initial_data.validation`

Producers for the manuscript figures live in ``pipeline/``; the manuscript and
its figure scripts in ``manuscript/``.  See ``docs/`` for the data-regeneration
and validation-oracle setup.
"""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazily expose subpackages as attributes so ``lm.initial_data.solver``
    resolves after ``import lm`` without an explicit submodule import."""
    import importlib

    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
