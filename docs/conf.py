"""Sphinx configuration for BrainEvo."""
from __future__ import annotations

project = "BrainEvo"
author = "Theo Dodd, Bart R. H. Geurten"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]
autodoc_mock_imports = ["matplotlib", "tqdm"]
napoleon_google_docstring = True
html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build"]
