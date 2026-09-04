"""Benchmark dp-gan vs baselines SDV (fase benchmark)."""

from __future__ import annotations

import pandas as pd
import pytest

from synthpriv.benchmark import BenchmarkResult, run_benchmark

# ---------------------------------------------------------------------------
# Fast: pruebas de estructura/reportes sin entrenar nada
# ---------------------------------------------------------------------------

_BASELINE = "gaussian-copula"


def _sample_result() -> BenchmarkResult:
    rows = [
        {"model": "dp-gan", "kind": "dp-gan", "target_epsilon": 5.0, "measured_epsilon": 4.9,
         "util_ks_test": 0.20, "util_correlation_mae": 0.10, "priv_nndr": 0.75,
         "fit_seconds": 5.0, "delta": 1e-5},
        {"model": "dp-gan", "kind": "dp-gan", "target_epsilon": 50.0, "measured_epsilon": 49.0,
         "util_ks_test": 0.40, "util_correlation_mae": 0.04, "priv_nndr": 0.60,
         "fit_seconds": 5.0, "delta": 1e-5},
        {"model": _BASELINE, "kind": "baseline", "target_epsilon": None, "measured_epsilon": None,
         "util_ks_test": 0.60, "util_correlation_mae": 0.03, "priv_nndr": 0.50,
         "fit_seconds": 0.1, "delta": 1e-5},
    ]
    return BenchmarkResult(rows=rows, baselines=[_BASELINE],
                           utility_metrics=["ks_test", "correlation_mae"],
                           privacy_metrics=["nndr"])


def test_result_sort_dp_by_epsilon_and_baselines_last():
    df = _sample_result().dataframe()
    models = list(df["model"])
    assert models == ["dp-gan", "dp-gan", _BASELINE]
    eps = list(df["measured_epsilon"])
    assert eps[0] == pytest.approx(4.9)
    assert eps[1] == pytest.approx(49.0)
    assert pd.isna(eps[2])


def test_curve_ordered_and_dropna():
    curve = _sample_result().curve("dp-gan", "util_ks_test")
    assert [p["x"] for p in curve] == [4.9, 49.0]
    assert curve[0]["y"] == 0.20
    assert _sample_result().curve("dp-gan", "util_desconocida") == []


def test_baseline_value_es_media():
    assert _sample_result().baseline_value(_BASELINE, "util_ks_test") == 0.60


def test_dp_value_mas_privado_y_a_epsilon_objetivo():
    result = _sample_result()
    assert result.dp_value("util_ks_test") == 0.20  # minimo epsilon medido
    assert result.dp_value("util_ks_test", target_epsilon=50.0) == 0.40


def test_utility_gap_direccion():
    result = _sample_result()
    # higher_is_better: gap = baseline - dp (positivo => dp-gan pierde)
    assert result.utility_gap("util_ks_test", _BASELINE) == pytest.approx(0.40)
    # lower_is_better: gap = dp - baseline
    assert result.utility_gap("util_correlation_mae", _BASELINE) == pytest.approx(0.07)


def test_best_dp_point_por_umbral():
    result = _sample_result()
    # correlation_mae (lower): el punto mas privado con mae <= 0.10
    best = result.best_dp_point("util_correlation_mae", threshold=0.10)
    assert best == {"x": 4.9, "y": 0.10}
    assert result.best_dp_point("util_correlation_mae", threshold=0.03) is None


def test_to_csv_y_save_report(tmp_path):
    result = _sample_result()
    csv = result.to_csv(tmp_path / "bench.csv")
    assert csv.exists()
    assert "model" in csv.read_text().splitlines()[0]

    html = result.save_report(tmp_path / "bench.html")
    content = html.read_text()
    assert "Benchmark dp-gan" in content
    assert "gaussian-copula" in content
    assert "Utilidad: ks_test" in content  # curva con refs de baseline


# ---------------------------------------------------------------------------
# Slow: benchmark real con una copula como baseline
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_benchmark_con_copula(real_data):
    result = run_benchmark(
        real_data,
        epsilons=(1.0, 50.0),
        baselines=(_BASELINE,),
        generator_kwargs={"epochs": 3, "batch_size": 64, "latent_dim": 16, "hidden_dim": 32},
        num_rows=200,
    )
    models = [r["model"] for r in result.rows]
    assert models == ["dp-gan", "dp-gan", _BASELINE]

    dp_eps = [r["measured_epsilon"] for r in result.rows if r["model"] == "dp-gan"]
    assert all(e is not None for e in dp_eps)
    baseline = [r for r in result.rows if r["model"] == _BASELINE][0]
    assert baseline["measured_epsilon"] is None
    assert "util_ks_test" in baseline and baseline["util_ks_test"] is not None

    curve = result.curve("dp-gan", "util_ks_test")
    assert len(curve) == 2 and curve[0]["x"] < curve[1]["x"]
    assert result.baseline_value(_BASELINE, "util_ks_test") is not None