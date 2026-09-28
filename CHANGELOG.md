# Changelog

All notable changes to `synthpriv` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/).

## [Unreleased]

### Added
- README: publish-workflow badge; phase-table rows for the publication
  polish commits (`c19fc3c`, `58a144a`).
- `CHANGELOG.md`: `[Unreleased]` section and Keep-a-Changelog comparison
  link definitions.
- Documentation site: Sphinx + Furo + autodoc/Napoleon (numpydoc-style
  docstrings) + MyST, with quickstart, privacy, sweep/benchmark, CLI and
  API reference pages; `docs.yml` builds with `-W` (warnings = errors) and
  deploys to GitHub Pages via `actions/deploy-pages`; `docs` extra; the
  `Documentation` project URL now points to
  https://tzinny-dev.github.io/synthpriv/.
- Versioned documentation with **mike**: tag pushes deploy the `X.Y` docs
  line (alias `latest`, site root redirected to it) and `main` pushes deploy
  `dev`; the Sphinx build renders a version selector in Furo's sidebar from
  `versions.json` (`docs/deploy_version.py`, `DOCS_BASE_URL`/`DOCS_VERSION`).
- Custom-domain ready Pages deployment: the URLs and Sphinx `html_baseurl`
  come from `actions/configure-pages` (`base_url`), so moving the site to
  its own domain needs no code change (`CNAME` is not required for
  Actions-based publishing).
- Site analytics: Google Analytics 4 (`gtag.js`, IP anonymized) injected in
  the `<head>` of every page through a Furo `extrahead` override
  (`docs/_templates/base.html`). The measurement ID defaults to
  `G-7HBPMKCQXE` in `docs/conf.py`, can be overridden with
  `GOOGLE_ANALYTICS_ID` (forwarded from the `GOOGLE_ANALYTICS_ID`
  repository variable in `docs.yml`) and turned off with
  `GOOGLE_ANALYTICS_ID=off`; disclosed in the README and in
  `docs/index.rst`.
- Support buttons (PayPal, Buy Me a Coffee) in the sidebar of every page
  (`docs/_templates/sidebar/support.html`, wired via `html_sidebars`, with
  styles in `docs/_static/custom.css`).
- `sitemap.xml` from the new `sphinx-sitemap` extension (only when
  `DOCS_BASE_URL` is set, so local/PR builds are unaffected) and a
  `robots.txt` written at the site root by `docs.yml` — necessary because
  mike keeps every build inside a version directory.
- `CONTRIBUTING.md` (dev setup, tests, docs build/deploy, release
  checklist) and `CITATION.cff`.
- README: `Documentation` section (versioned site + local build), `Support`
  section with the donation links, and phase-table rows for the docs site
  commits (`308434a`, `5339aa9`).

### Changed
- `docs/conf.py`: `html_baseurl` now includes the `DOCS_VERSION` directory
  and `sitemap_url_scheme` is `{link}` (the extension's default would
  prepend Sphinx's `language`/`version`, which are not part of the deployed
  path).
- `publish.yml` hardening:
  - **tests gate**: the tag pipeline now runs the full test matrix before
    `build`/`publish` (a tag no longer publishes untested code).
  - **idempotent re-runs**: `skip-existing` on the PyPI upload and
    `gh release view || create` (upload `--clobber`) for the GitHub release.
  - **post-publish verification**: new `verify-pypi` job installs
    `synthpriv==<version>` from real PyPI (with propagation retries) and
    checks `importlib.metadata` version + `synthpriv --help`.
  - `twine check --strict`.

### Fixed
- Documentation canonical URLs: `html_baseurl` gained the mike version
  directory, so `rel="canonical"` points at `<base>/0.2/page.html` instead of
  `<base>/page.html`, which is a 404 at the site root.

### Known limitations
- **Python 3.13 unsupported**: `anonymeter 1.1.0` (latest) pins
  `numpy>=1.22,<1.27`, and numpy <1.27 ships no 3.13 wheels, so
  installation on 3.13 fails with `ResolutionImpossible` (CI run
  `35905206866`; `sdv`/`sdmetrics` themselves already support 3.13).
  `requires-python` is now capped to `>=3.10,<3.13` (effective from the
  next release) so `pip` fails fast with a clear message; drop the cap
  when anonymeter relaxes the numpy pin.

## [0.2.1] - 2026-09-21

Packaging and release-pipeline polish. No changes to the public API.

### Added
- Single-source version: `pyproject.toml` reads `version` dynamically from
  `src/synthpriv/__init__.py` (`importlib.metadata.version("synthpriv")`
  compatible).
- `publish.yml` workflow: tag `v*` → build → tag/version guard → PyPI
  (Trusted Publisher OIDC + PEP 740 attestations) → GitHub release with
  artifacts. The release is only created after a successful PyPI publish.
- `MANIFEST.in`: sdist now ships `CHANGELOG.md`, `examples/` and `tests/`.
- CI: test matrix 3.10/3.11/3.12 + `package` job (build, `twine check`,
  wheel smoke test).

### Changed
- PEP 639 metadata: `license-files` explicitly declared; `Project-URL`s
  `Documentation` and `Changelog` added (side links on PyPI).
- README: PyPI/Python/License/CI badges; `pip install synthpriv` quick install;
  CUDA/torch note.

## [0.2.0] - 2026-09-15

Beta validated: clean PyPI install, functional smoke (`dp-gan`/`dp-copula`),
fast + slow suites, E2E demo. Same content as `0.2.0b1`, promoted to final.

## [0.2.0b1] - 2026-09-15

Beta pre-release: public API frozen for the `0.2.x` line.

### Added
- `run_epsilon_sweep(..., generator_key="dp-gan"|"dp-copula")`: sweep any
  DP-capable generator; rows carry `model`, `SweepResult.generator` records it.
- `run_benchmark(..., dp_generator="dp-gan")`: benchmark any DP generator vs
  non-DP SDV baselines; `BenchmarkResult.dp_generator`, `kind="dp"|"baseline"`,
  `dp_value()/utility_gap()/best_dp_point()` resolve the configured generator.
- CLI `sweep/benchmark --generator/-g [dp-gan|dp-copula]` with dp-gan-only flag
  validation (`--numeric/--rectify-marginals/--ecdf-epsilon`).
- `tests/test_dp_matrix.py`: dp-copula sweep/benchmark matrix (measured==target,
  `kind=dp`, dynamic report title).
- `.github/workflows/tests.yml`: fast CI (`pip install -e .[dev]`, `pytest -q`).

### Changed
- **BREAKING (beta freeze reference):** `BenchmarkResult` rows use
  `kind="dp"` instead of `"dp-gan"`; use `model` for the generator name and
  `result.dp_generator` for the swept generator.
- `SweepResult` rows now include `model`; reports/footers no longer assume
  RDP-only (`dp-copula` is pure DP).
- Docs fully in English; `README` CLI/demo/tests updated.

### Fixed
- Stale `phase 0.1 / dp-gan only` docstrings (`mechanisms`, `assurance`,
  `pipeline` leftover Spanish comments).

## [0.1.0] - 2026-09-08

Alpha: tabular synthesis with differential privacy (`dp-gan` DP-SGD,
`dp-copula` pure DP), DP marginal ECDFs, budget split, benchmark/sweep, HTML
report, CLI, `assert_dp`, persistence.

[Unreleased]: https://github.com/Tzinny-dev/synthpriv/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Tzinny-dev/synthpriv/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Tzinny-dev/synthpriv/compare/v0.2.0b1...v0.2.0
[0.2.0b1]: https://github.com/Tzinny-dev/synthpriv/compare/v0.1.0...v0.2.0b1
[0.1.0]: https://github.com/Tzinny-dev/synthpriv/releases/tag/v0.1.0
