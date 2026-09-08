from __future__ import annotations

import pandas as pd
import pytest

from synthpriv import SweepResult, run_epsilon_sweep
from synthpriv.report import render_sweep_html


def test_sweep_result_dataframe_and_csv(tmp_path):
    result = SweepResult(
        rows=[
            {"target_epsilon": 1.0, "measured_epsilon": 0.9, "delta": 1e-5,
             "util_ks_test": 0.4, "util_correlation_mae": 0.08, "fit_seconds": 1.0},
            {"target_epsilon": 50.0, "measured_epsilon": 45.0, "delta": 1e-5,
             "util_ks_test": 0.1, "util_correlation_mae": 0.03, "fit_seconds": 1.0},
        ],
        utility_metrics=["ks_test", "correlation_mae"],
    )
    df = result.dataframe()
    assert list(df["measured_epsilon"]) == [0.9, 45.0]  # ascending order
    result.to_csv(tmp_path / "sweep.csv")
    assert (tmp_path / "sweep.csv").exists()


def test_best_tradeoff():
    result = SweepResult(rows=[
        {"target_epsilon": 0.5, "measured_epsilon": 0.45, "util_correlation_mae": 0.12},
        {"target_epsilon": 2.0, "measured_epsilon": 1.9, "util_correlation_mae": 0.04},
        {"target_epsilon": 50.0, "measured_epsilon": 46.0, "util_correlation_mae": 0.02},
    ])
    best = result.best_tradeoff("util_correlation_mae", threshold=0.05)
    assert best["measured_epsilon"] == 1.9  # the most private that meets MAE <= 0.05
    assert result.best_tradeoff("util_correlation_mae", threshold=0.001) is None


def test_sweep_report_render_with_svg(tmp_path):
    result = SweepResult(rows=[
        {"target_epsilon": 0.5, "measured_epsilon": 0.45, "util_correlation_mae": 0.12},
        {"target_epsilon": 5.0, "measured_epsilon": 4.6, "util_correlation_mae": 0.05},
        {"target_epsilon": 50.0, "measured_epsilon": 46.0, "util_correlation_mae": 0.02},
    ], utility_metrics=["correlation_mae"])
    out = render_sweep_html(result, tmp_path / "sweep.html")
    html = out.read_text()
    assert "<svg" in html and "<polyline" in html
    assert "correlation_mae" in html


@pytest.mark.slow
def test_run_epsilon_sweep_smoke(real_data, tmp_path):
    """Short sweep with loose budgets to finish quickly."""
    result = run_epsilon_sweep(
        real_data,
        epsilons=(5.0, 50.0),
        delta=1e-3,
        generator_kwargs={"epochs": 2, "batch_size": 128, "latent_dim": 16, "hidden_dim": 32},
        utility_metrics=["correlation_mae"],
        privacy_metrics=["nndr"],
        num_rows=60,
    )
    assert len(result.rows) == 2
    measured = [r["measured_epsilon"] for r in result.rows]
    assert all(m is not None for m in measured)
    assert measured[0] <= 5.0 * 1.5  # close to the first point budget
    assert measured[1] > measured[0]  # higher budget -> higher measured epsilon
    assert all("util_correlation_mae" in r for r in result.rows)
    assert result.save_report(tmp_path / "sweep.html").exists()