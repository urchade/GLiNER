import os
import sys

# ----- Paths -----
DOCS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)

project = "Home"
copyright = "2025, GLiNER community"
author = "Urchade Zaratiana, Ihor Stepanov"
release = "0.2.24"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

html_theme = "shibuya"
html_theme_options = {
    "navigation_depth": 4,
    "show_sidebar": True,
    "sidebar_toctree_depth": 2,
}

html_context = {"github_user": "urchade", "github_repo": "GLiNER"}
html_static_path = ["_static"]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "show-inheritance": True,
    "undoc-members": True,
    "inherited-members": False,
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autoclass_content = "both"

# Autosummary settings
autosummary_generate = True  # Enable autosummary


def run_apidoc(_):
    from sphinx.ext.apidoc import main

    # Path to your module (package "gliner" in repo root)
    module_path = os.path.join(PROJECT_ROOT, "gliner")
    # Output path for API docs (inside docs/)
    output_path = os.path.join(DOCS_DIR, "api")

    main([
        "--force",
        "--separate",
        "--module-first",
        "-o", output_path,
        module_path,
    ])


def setup(app):
    app.connect("builder-inited", run_apidoc)
