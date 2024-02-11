# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'paref'
copyright = '2023, Nicolai Palm'
author = 'Nicolai Palm'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_design',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.autosectionlabel',
]

nbsphinx_execute = 'never'

myst_enable_extensions = ['colon_fence']

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_theme_path = ['_themes', ]

html_theme_options = {
    'github_url': 'https://github.com/nicolaipalm/paref',
    'search_bar_text': 'Search for treasure...',
}

html_sidebars = {
  'path/to/page': [],
}

html_title = 'Paref'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Autodoc ---------------------------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    'show-inheritance': True,
    'members': True,
    'member-order': 'groupwise',
    'special-members': '__call__',
    'undoc-members': True,
}
autoclass_content = 'both'
autodoc_inherit_docstrings = False

# -- Read the Docs ---------------------------------------------------------------------------------
master_doc = 'index'
