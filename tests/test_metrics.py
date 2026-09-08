from __future__ import annotations

from synthpriv.metrics import evaluate_privacy, evaluate_utility, list_metric_names, MetricResult


def test_utility_metrics_run(real_data, synth_data):
    results = evaluate_utility(real_data, synth_data)
    assert {"ks_test", "correlation_mae", "ml_utility"} <= set(results)
    for r in results.values():
        assert isinstance(r, MetricResult)


def test_ks_test_reports_columns(real_data, synth_data):
    res = evaluate_utility(real_data, synth_data, ["ks_test"])["ks_test"]
    assert res.details["columns_checked"]
    assert res.value is not None


def test_correlation_mae_value(real_data, synth_data):
    res = evaluate_utility(real_data, synth_data, ["correlation_mae"])["correlation_mae"]
    assert 0 <= res.value <= 1


def test_privacy_metrics_run(real_data, synth_data):
    results = evaluate_privacy(real_data, synth_data)
    assert {"nndr", "mia_auc"} <= set(results)
    for r in results.values():
        assert isinstance(r, MetricResult)


def test_nndr_close_if_copies_reals():
    """Synthetic rows very close to real records => ratios < 1 (high copy risk)."""
    import pandas as pd

    from synthpriv.metrics import evaluate_privacy

    real = pd.DataFrame({"a": [0, 0, 10, 10, 20]})
    synth = pd.DataFrame({"a": [0.4, 9.6, 19.7, 9.9, 0.3]})
    res = evaluate_privacy(real, synth, ["nndr"], {"nndr": {"sample": 10}})
    assert res["nndr"].details["min_ratio"] < 0.1  # at least one near-copy case
    assert res["nndr"].details["pct_below_1"] > 0.0

    # well-separated data => higher ratios (less risk)
    far = pd.DataFrame({"a": [100.0, 110.0, 120.0, 130.0, 140.0]})
    res2 = evaluate_privacy(real, far, ["nndr"], {"nndr": {"sample": 10}})
    assert res2["nndr"].details["min_ratio"] > 1.0


def test_mia_same_data_auc_low():
    """Identical data -> indistinguishable -> AUC ~0.5 (low distinguishability, good result)."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    from synthpriv.metrics import evaluate_privacy

    res = evaluate_privacy(df, df, ["mia_auc"], {"mia_auc": {"folds": 2}})
    assert 0.3 <= res["mia_auc"].value <= 0.7


def test_metric_registry_lists():
    assert "nndr" in list_metric_names()
    assert "ks_test" in list_metric_names()