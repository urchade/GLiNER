import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "Home"
copyright = "2025, GLiNER community"
author = "Ihor Stepanov"
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
    import os

    # Path to your module
    module_path = os.path.abspath('../gliner')
    # Output path for API docs
    output_path = os.path.abspath('./api')
    
    # Run sphinx-apidoc
    main([
        '--force',
        '--separate',
        '--module-first',
        '-o', output_path,
        module_path,
    ])

def setup(app):
    app.connect('builder-inited', run_apidoc)