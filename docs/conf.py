"""Sphinx configuration for wdflow's Read the Docs build."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "wdflow"
author = "Elena Cuoco"
copyright = "2026, Elena Cuoco"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# wdf.processes/wdf.observers need the compiled p4TSA/pytsa core, and
# wdf.analysis.gnn imports torch/torch_geometric at module level; RTD's build
# environment doesn't need any of these actually installed to document the
# API -- autodoc just needs the imports to not fail.
autodoc_mock_imports = ["pytsa", "torch", "torch_geometric"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
