import os
import sys

# Configuration file for the Sphinx documentation builder.

# -- Path Setup

sys.path.insert(0, os.path.abspath('../../'))
autodoc_mock_imports = ["numpy", "scipy", "matplotlib", "sklearn", "pandas",
        "numdifftools", "random"]

# -- Project information

project = 'mastercurves'
copyright = '2022, Kyle R. Lennon'
author = 'Kyle R. Lennon'

release = '0.2'
version = '0.2.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
