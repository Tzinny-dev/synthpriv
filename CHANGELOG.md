# Changelog

All notable changes to `synthpriv` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/).

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
