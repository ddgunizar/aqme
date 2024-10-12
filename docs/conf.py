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
# Ensure that modules can be imported without installing aqme
sys.path.insert(0, os.path.abspath('..')) 



# -- Project information -----------------------------------------------------

project = 'aqme'
copyright = '2024, Juan V. Alegre Requena, Shree Sowndarya S. V., Alturaifi T., Pérez-Soto R., Robert S. Paton'
author = '2020, Juan V. Alegre Requena, Shree Sowndarya S. V., Alturaifi T., Pérez-Soto R., Robert S. Paton'

# The full version, including alpha/beta/rc tags
release = 'v1.7.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',]

# Avoid paths in class names i.e. 
# class aqme.aqme.csearch.csearch -> class csearch
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Disable  smartquotes which might transform '--' into a different character
smartquotes = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
