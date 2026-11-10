"""Sphinx configuration for the LMID-conformally-flat-puncture documentation.

Build locally with ``make docs`` (HTML into ``docs/_site/``, which is gitignored);
Read the Docs builds from ``.readthedocs.yaml`` at the repository root.

The look is the furo theme recoloured with the Lemaitre style tokens. ``_static/lemaitre-tokens.css``
and ``_static/fonts/`` are copies of ``style/`` in the LemaitreArtwork repository; re-copy them from
there rather than editing them here.
"""
import lemaitre.initial_data.conformally_flat_puncture as package

project = "LMID-conformally-flat-puncture"
author = "Frederik De Ceuster"
copyright = "2026, Frederik De Ceuster"
release = package.__version__
version = release

extensions = [
    "myst_parser",               # the existing Markdown pages are read as they are
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # the few NumPy-style docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",    # :func: links into numpy, scipy and jax resolve
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_site", "_autosummary/**/.*", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["dollarmath", "amsmath", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Producers in pipeline/ are scripts run per figure; they are documented on the
# "Regenerating the paper's data" page, not imported here.
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

html_theme = "furo"
html_title = "LMID-conformally-flat-puncture"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["lemaitre-tokens.css", "lemaitre.css", "spin.css", "psi.css", "bernstein.css", "smolyak.css", "hermite.css", "abtgrid.css"]
html_js_files = ["lemaitre.js", "psi-data.js", "psi.js", "spin.js", "bernstein.js", "smolyak.js", "hermite.js", "abtgrid.js"]
html_theme_options = {
    "source_repository": "https://github.com/LemaitreModels/LMID-conformally-flat-puncture",
    "source_branch": "main",
    "source_directory": "docs/",
}
