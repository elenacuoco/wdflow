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
    # myst_nb supersedes myst_parser (it registers the Markdown parser too) and
    # renders the tutorial notebooks.
    "myst_nb",
]

# The tutorials are committed with their outputs, and executing them here would
# need py4tsa, which the docs build does not install.
nb_execution_mode = "off"

# wdf.processes/wdf.observers need the compiled p4TSA/py4tsa core, and
# wdf.analysis.gnn imports torch/torch_geometric at module level; RTD's build
# environment doesn't need any of these actually installed to document the
# API -- autodoc just needs the imports to not fail.
autodoc_mock_imports = ["py4tsa", "torch", "torch_geometric", "pycbc", "gwpy"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
root_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

# Each inventory is read from its site first and, when the site cannot be
# reached, from the copy committed under _inventories/. The build runs with -W,
# and intersphinx warns only when every location of an inventory fails, so an
# outage of one of these sites no longer fails the build: on 1 Oct 2026
# docs.python.org answered 503 for hours, and that alone failed the docs check.
# The copies only serve while a site is down. Refresh them now and then with
#     curl -sSL -o docs/_inventories/<name>.inv <site>/objects.inv
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", (None, "_inventories/python.inv")),
    "numpy": ("https://numpy.org/doc/stable/", (None, "_inventories/numpy.inv")),
    "pandas": ("https://pandas.pydata.org/docs/", (None, "_inventories/pandas.inv")),
}
