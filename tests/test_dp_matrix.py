"""dp-copula in the sweep/benchmark matrix (fast, no training epochs)."""

from __future__ import annotations

import pytest

from synthpriv import run_epsilon_sweep
from synthpriv.benchmark import BenchmarkResult, run_benchmark


def test_sweep_dp_copula_reports_model_and_generator(real_data):
    result = run_epsilon_sweep(
        real_data,
        epsilons=(1.0, 5.0),
        generator_key="dp-copula",
        utility_metrics=["ks_test"],
        privacy_metrics=[],
        num_rows=100,
    )
    assert result.generator == "dp-copula"
    assert len(result.rows) == 2
    assert all(r["model"] == "dp-copula" for r in result.rows)
    measured = [r["measured_epsilon"] for r in result.rows]
    assert measured == [1.0, 5.0]  # pure DP: measured == target
    assert all("util_ks_test" in r for r in result.rows)


def test_benchmark_dp_copula_kind_dp(real_data):
    result = run_benchmark(
        real_data,
        epsilons=(2.0,),
        dp_generator="dp-copula",
        baselines=("gaussian-copula",),
        utility_metrics=["ks_test"],
        privacy_metrics=[],
        num_rows=100,
    )
    assert result.dp_generator == "dp-copula"
    models = [r["model"] for r in result.rows]
    assert models == ["dp-copula", "gaussian-copula"]
    dp_row = [r for r in result.rows if r["model"] == "dp-copula"][0]
    assert dp_row["kind"] == "dp"
    assert dp_row["measured_epsilon"] == pytest.approx(2.0)
    assert result.curve("dp-copula", "util_ks_test")


def test_benchmark_dp_value_defaults_to_configured_generator(tmp_path):
    result = BenchmarkResult(
        rows=[
            {"model": "dp-copula", "kind": "dp", "target_epsilon": 2.0,
             "measured_epsilon": 2.0, "util_ks_test": 0.30, "fit_seconds": 0.1,
             "delta": 1e-5},
            {"model": "gaussian-copula", "kind": "baseline", "target_epsilon": None,
             "measured_epsilon": None, "util_ks_test": 0.50, "fit_seconds": 0.1,
             "delta": 1e-5},
        ],
        baselines=["gaussian-copula"],
        utility_metrics=["ks_test"],
        dp_generator="dp-copula",
    )
    assert result.dp_value("util_ks_test") == pytest.approx(0.30)
    assert result.utility_gap("util_ks_test", "gaussian-copula") == pytest.approx(0.20)
    html = result.save_report(tmp_path / "b.html")
    assert "dp-copula (DP) vs SDV baselines" in html.read_text()
