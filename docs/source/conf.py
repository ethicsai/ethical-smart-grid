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
copyright = '2023, Clément Scheirlinck & Rémy Chaput'
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
    # Automatically add a 'copy button' to our code blocks
    'sphinx_copybutton',
    # Support for Jupyter Notebooks
    'myst_nb',
    # Add a link to the source code on each of the "API pages"
    'sphinx.ext.viewcode',
    # Render the docs for multiple versions (tags, branches, ...)
    'sphinx_multiversion',
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


# -- Options for Sphinx-multiversion

# Whitelist pattern for tags (in our case, versions)
# We ignore explicitly `v1.1.0-joss-paper` because it was used for Zenodo, it
# is not a "real" version.
smv_tag_whitelist = r'^(?!v1.1.0-joss-paper).*$'

# Accept all branches except:
# - `paper` specifically (only holds the JOSS paper's source)
#    (note: the `$` is at the end of `paper` so that it matches this exact
#     string; anything other, e.g. `papers`, can pass)
# - any `wip/**` branch (because they are not ready)
smv_branch_whitelist = r'^(?!wip/|paper$).*'

# Allow remote branches from `origin` only (required for building all branches
# on GitHub Actions, because they are not automatically fetched).
smv_remote_whitelist = r'^origin$'

# A version is considered "released" only if it is a tag beginning with `v`.
smv_released_pattern = r'^tags/v.*$'

# Each version gets a subdir based on its name (e.g., `master`, `v1.0.0`, ...)
smv_outputdir_format = '{ref.name}'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Override Furo's default sidebar widgets to add our version selector
# At some point we should be able to use `variant-selector`, when Furo will add
# support for versions.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "versioning.html",  # Our custom template
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/variant-selector.html",
        "sidebar/scroll-end.html",
    ]
}
