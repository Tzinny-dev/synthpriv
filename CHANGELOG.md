# Changelog

All notable changes to `synthpriv` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Python 3.13: CI test matrix and trove classifier (SDV/SDMetrics declare
  3.13 support; torch ships 3.13 wheels).
- README: publish-workflow badge; phase-table rows for the publication
  polish commits (`c19fc3c`, `58a144a`).
- `CHANGELOG.md`: `[Unreleased]` section and Keep-a-Changelog comparison
  link definitions.

### Changed
- `publish.yml` hardening:
  - **tests gate**: the tag pipeline now runs the full test matrix before
    `build`/`publish` (a tag no longer publishes untested code).
  - **idempotent re-runs**: `skip-existing` on the PyPI upload and
    `gh release view || create` (upload `--clobber`) for the GitHub release.
  - **post-publish verification**: new `verify-pypi` job installs
    `synthpriv==<version>` from real PyPI (with propagation retries) and
    checks `importlib.metadata` version + `synthpriv --help`.
  - `twine check --strict`.

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
