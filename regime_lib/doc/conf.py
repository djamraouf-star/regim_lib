"""
Configuration Sphinx pour la documentation de regime_lib.

Build :
    sphinx-build -b html regime_lib/doc regime_lib/doc/_build/html

Docstrings : tout en français, conformément au contrat du projet.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permet à autodoc de trouver le package sans installation préalable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from regime_lib._version import __version__

project = "regime_lib"
author = "Équipe regime_lib"
copyright = "2026, Équipe regime_lib"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",     # docstrings style NumPy/Google
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",             # pages en Markdown
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = []

language = "fr"
