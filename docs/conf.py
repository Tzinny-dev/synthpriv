"""Sphinx configuration for the synthpriv documentation.

Site: https://tzinny-dev.github.io/synthpriv/
"""

from __future__ import annotations

import json
import os
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

# -- Versioned docs (mike) --------------------------------------------------
# `docs/deploy_version.py` materialises versions.json as docs/_versions.json
# before invoking Sphinx, so the version selector is rendered at build time.
templates_path = ["_templates"]

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/versions.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}


def _sort_key(version: str) -> tuple:
    parts = version.split(".")
    if all(part.isdigit() for part in parts):
        return (1, [int(part) for part in parts])
    return (0, [version])


def _load_versions() -> list[dict]:
    path = pathlib.Path(
        os.environ.get("MIKE_VERSIONS_FILE", str(_root / "docs" / "_versions.json"))
    )
    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))
    base_url = os.environ.get("DOCS_BASE_URL", "").rstrip("/")
    current = os.environ.get("DOCS_VERSION", "")
    versions = []
    for entry in sorted(entries, key=lambda item: _sort_key(str(item["version"])), reverse=True):
        name = str(entry["version"])
        versions.append(
            {
                "name": name,
                "label": str(entry.get("title") or name),
                "aliases": [str(alias) for alias in entry.get("aliases") or []],
                "url": f"{base_url}/{name}/" if base_url else f"../{name}/",
                "current": name == current,
            }
        )
    return versions


_html_base_url = os.environ.get("DOCS_BASE_URL", "").rstrip("/")
if _html_base_url:
    html_baseurl = f"{_html_base_url}/"

html_context = {
    "versions": _load_versions(),
    "current_version": os.environ.get("DOCS_VERSION", ""),
}