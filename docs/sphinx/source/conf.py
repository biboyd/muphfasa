# Configuration file for the Sphinx documentation builder.
# muphfasa docs — styled after the AMReX documentation
# (https://amrex-codes.github.io/amrex/docs_html/index.html).

project = "MuPhFASa"
copyright = "2026, MuPhFASa contributors"
author = "MuPhFASa contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
]

autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML output ──────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}

# ── LaTeX / math ─────────────────────────────────────────────────────────
numfig = True
math_number_all = False
