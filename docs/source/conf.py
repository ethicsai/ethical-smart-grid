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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# -- Project information -----------------------------------------------------

project = 'EthicalSmartGrid'
copyright = '2022, Clément Scheirlinck & Rémy Chaput'
author = 'Clément Scheirlinck & Rémy Chaput'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Include documentation from docstrings
    'sphinx.ext.autodoc',
    # Generate summaries (tables/listings) for autodoc
    'sphinx.ext.autosummary',
    # Link to external (other projects') documentation
    'sphinx.ext.intersphinx',
    # Matplotlib plot in documentation
    'matplotlib.sphinxext.plot_directive',
]

# Enable autosummary
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Mapping to other projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
    # gymnasium is often used as `gym`, so we alias it here
    'gym': ('https://gymnasium.farama.org/', None),
}

# Allow to automatically place the project name in ReST documents
rst_epilog = '.. |project_name| replace:: {}'.format(project)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
