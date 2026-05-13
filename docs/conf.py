# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "langgoap"
copyright = "2024, Integrallis Software"
author = "Integrallis Software"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_copybutton",
    "_extension.gallery_directive",
    "myst_nb",
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

pygments_style = "friendly"
pygments_dark_style = "monokai"

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_title = "langgoap"

html_context = {
    "github_user": "integrallis",
    "github_repo": "langgoap",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

html_theme_options = {
    "repository_url": "https://github.com/LangGOAP/LangGOAP",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "navigation_depth": 4,
    "show_toc_level": 3,
    "home_page_in_toc": True,
    "logo": {
        "text": "langgoap",
        "image_light": "_static/images/logo-light.svg",
        "image_dark": "_static/images/logo-dark.svg",
        "alt_text": "LangGOAP",
    },
}

# -- Favicon (sphinx-favicon) ------------------------------------------------
favicons = [
    {"rel": "icon", "href": "images/icon.svg", "type": "image/svg+xml"},
]

autoclass_content = "both"
add_module_names = False

# Suppress warnings that are either expected artifacts of our build
# (TypedDict classes generate an empty ``__init__`` that autosummary
# templates try to document; docutils complains about a cosmetic
# "explicit markup" line emitted by the autodoc-generated summary
# pages) or cosmetic link targets inside notebook markdown that point
# at files outside the docs tree.
suppress_warnings = [
    "autodoc",
    "docutils",
    "myst.xref_missing",
    # Pygments has no built-in mermaid lexer; the plan_visualization
    # notebook prints rendered ```mermaid``` blocks in its outputs.
    "misc.highlighting_failure",
]

nb_execution_mode = "off"

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

# -- Sidebar with version switcher ------------------------------------------
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
        "versioning.html",
    ],
}
