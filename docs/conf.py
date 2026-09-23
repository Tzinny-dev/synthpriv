"""Sphinx configuration for the synthpriv documentation.

Site: https://tzinny-dev.github.io/synthpriv/
"""

from __future__ import annotations

import pathlib
import re

_root = pathlib.Path(__file__).resolve().parents[1]

# Single-source version without importing the package (torch-heavy deps).
_init = (_root / "src" / "synthpriv" / "__init__.py").read_text(encoding="utf-8")
release = re.search(r'__version__\s*=\s*"([^"]+)"', _init).group(1)
version = release

project = "synthpriv"
copyright = "2026, Carlos"  # noqa: A001
author = "Carlos"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"synthpriv {release}"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}