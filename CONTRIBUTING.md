# Contributing to synthpriv

Thanks for helping out. This covers the workflow for code, documentation and
releases.

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"    # code + test/build tooling
.venv/bin/pip install -e ".[docs]"   # documentation toolchain
```

`dp-gan` pulls `torch` through Opacus (CPU wheels by default; install a
CUDA build separately if you train on GPU).

## Tests

```bash
.venv/bin/python -m pytest -q          # fast suite (default)
.venv/bin/python -m pytest -m slow -q  # trains the deep models
```

CI runs the fast suite plus a package build on Python 3.10–3.12
(`.github/workflows/tests.yml`).

## Documentation

The public site is Sphinx + Furo, built from `docs/` and published at
https://tzinny-dev.github.io/synthpriv/.

```bash
.venv/bin/python -m sphinx -W -b html docs docs/_build/html
```

- **Warnings are errors** (`-W`) locally and in CI: a broken reference or a
  missing template fails the build.
- Page structure comes from `docs/_templates/`; `docs/_templates/base.html`
  extends the theme's `base.html` to inject the analytics snippet.
- Sidebar components live in `docs/_templates/sidebar/` and are registered in
  `html_sidebars` (`docs/conf.py`). Shared styles go in `docs/_static/custom.css`.

### Site settings

| Env var | Meaning | Default |
|---|---|---|
| `DOCS_BASE_URL` | Absolute site URL; enables canonical URLs and `sitemap.xml` | unset (no canonical/sitemap) |
| `DOCS_VERSION` | Version directory the build is deployed into (`0.2`, `dev`) | unset |
| `GOOGLE_ANALYTICS_ID` | GA4 measurement ID; `off`/`none` disables analytics | `G-7HBPMKCQXE` |
| `MIKE_VERSIONS_FILE` | Path to `versions.json` used by the sidebar version selector | `docs/_versions.json` |

`docs.yml` forwards the repository variable `GOOGLE_ANALYTICS_ID`, so the ID
can be changed or turned off without editing the code.

### How the site is deployed

`.github/workflows/docs.yml`:

| Event | Result |
|---|---|
| `pull_request` | `sphinx-build -W` check only, nothing is published |
| push to `main` | `docs/deploy_version.py dev` → the `dev` version on `gh-pages` (default when no other version exists) |
| tag `vX.Y.Z` | version `X.Y` with the `latest` alias, made the site default |
| `workflow_dispatch` | an arbitrary version you type in, with the `latest` alias |

Every deployment publishes the `gh-pages` branch through
`actions/deploy-pages` and writes a root `robots.txt` (mike keeps each build
inside its version directory, so `robots.txt` cannot come from the Sphinx
output). `mike` renders the sidebar version selector from `versions.json`,
which `docs/deploy_version.py` materialises before invoking Sphinx.

## Release checklist

1. Bump `__version__` in `src/synthpriv/__init__.py` (single source of truth;
   `pyproject.toml` reads it dynamically).
2. Move the `[Unreleased]` entries in `CHANGELOG.md` into a new
   `## [X.Y.Z] - YYYY-MM-DD` section and update the comparison links.
3. Update `CITATION.cff` (`version` and `date-released`).
4. Commit, then tag `vX.Y.Z` and push the tag.
5. `publish.yml` runs the test matrix, publishes to PyPI and creates the
   GitHub release; `docs.yml` deploys the `X.Y` documentation line.

## Style

- Docs, docstrings and comments are in English; docstrings must build cleanly
  under `sphinx-build -W`.
- `docs/conf.py` must not import the package (it is torch-heavy).
- Keep commits scoped, and extend the README phase table when a phase lands.
