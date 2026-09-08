"""Utility metrics: how close the synthetic data is to the real one."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from synthpriv.metrics.base import MetricResult, evaluate_status


def _numeric_columns(real: pd.DataFrame, synth: pd.DataFrame) -> list[str]:
    num = real.select_dtypes(include=[np.number]).columns
    return [c for c in num if c in synth.columns]


def ks_test(real: pd.DataFrame, synth: pd.DataFrame, significance: float = 0.05) -> MetricResult:
    """Kolmogorov-Smirnov test per numeric column.

    Null hypothesis: the marginal distributions match. ``p >= significance``
    implies equality cannot be rejected -> the column is useful.
    """
    cols = _numeric_columns(real, synth)
    per_col = {}
    p_values = []
    for c in cols:
        D, p = stats.ks_2samp(real[c].dropna(), synth[c].dropna())
        per_col[c] = {"statistic": round(float(D), 4), "p_value": round(float(p), 4)}
        p_values.append(float(p))
    if not p_values:
        return MetricResult(name="ks_test", status="error", message="No numeric columns for KS.")
    min_p = min(p_values)
    status, msg = evaluate_status(min_p, significance, "higher_is_better")
    return MetricResult(
        name="ks_test",
        description="KS per column: p>=alpha implies statistically equal distributions",
        value=round(min_p, 4),
        threshold=significance,
        direction="higher_is_better",
        status=status,
        message=msg,
        details={
            "columns_checked": cols,
            "passed_columns": [c for c, d in per_col.items() if d["p_value"] >= significance],
            "per_column": per_col,
        },
    )


def correlation_mae(real: pd.DataFrame, synth: pd.DataFrame, max_mae: float = 0.05) -> MetricResult:
    """Mean absolute error between Pearson correlation matrices."""
    cols = _numeric_columns(real, synth)
    if len(cols) < 2:
        return MetricResult(name="correlation_mae", status="reported",
                            message="Need >=2 numeric columns.")
    corr_real = real[cols].corr().values
    corr_synth = synth[cols].corr().values
    mae = float(np.nanmean(np.abs(corr_real - corr_synth)))
    status, msg = evaluate_status(mae, max_mae, "lower_is_better")
    return MetricResult(
        name="correlation_mae",
        description="MAE between correlation matrices (lower = better)",
        value=round(mae, 4),
        threshold=max_mae,
        direction="lower_is_better",
        status=status,
        message=msg,
        details={"columns_checked": cols},
    )


def _infer_target(real: pd.DataFrame, target: str | None) -> str:
    if target is not None:
        return target
    for c in real.columns:
        if real[c].dtype == object or real[c].dtype == "category":
            return c
    return real.columns[-1]


def _is_classification(y: pd.Series) -> bool:
    return y.dtype == object or y.dtype == "category" or len(y.unique()) <= 10


def _encode_features(real: pd.DataFrame, synth: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in real.columns if c != target]
    combined = pd.concat([real[cols], synth[cols]], axis=0)
    combined = pd.get_dummies(combined)
    return combined.iloc[: len(real)], combined.iloc[len(real):]


def ml_utility(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target: str | None = None,
    model=None,
    test_size: float = 0.3,
    min_score: float = 0.6,
) -> MetricResult:
    """Train-on-Synthetic-Test-on-Real (TSTR) performance.

    Trains a model on the synthetic data and evaluates it on the real one
    (TSTR); also trains and evaluates on real data (TRTS) as the maximum
    reachable reference. ``value`` is the TSTR metric.
    """
    target = _infer_target(real, target)
    if target not in synth.columns:
        return MetricResult(name="ml_utility", status="error",
                            message=f"Target column {target!r} does not exist in the synthetic data.")

    X_real, X_synth = _encode_features(real, synth, target)
    y_real, y_synth = real[target], synth[target]

    classification = _is_classification(y_real) and _is_classification(y_synth)
    y_real = y_real.astype(str) if classification else y_real.astype(float)
    y_synth = y_synth.astype(str) if classification else y_synth.astype(float)

    if classification:
        model = model or RandomForestClassifier(n_estimators=100, random_state=0)
        scorer = accuracy_score
    else:
        model = model or RandomForestRegressor(n_estimators=100, random_state=0)
        scorer = r2_score

    X_trts, X_te, y_trts, y_te = train_test_split(
        X_real, y_real, test_size=test_size, random_state=0,
        stratify=y_real if classification else None,
    )
    model.fit(X_trts, y_trts)
    trts = scorer(y_te, model.predict(X_te))

    model_tstr = type(model)(**model.get_params())
    model_tstr.fit(X_synth, y_synth)
    pred = model_tstr.predict(X_te)
    if classification:
        try:
            n_classes = len(np.unique(np.concatenate([y_synth.unique(), y_te.unique()])))
            tstr = scorer(y_te, pred)
        except ValueError:
            tstr = scorer(y_te, pred)
    else:
        tstr = scorer(y_te, pred)

    status, msg = evaluate_status(tstr, min_score, "higher_is_better")
    return MetricResult(
        name="ml_utility",
        description="TSTR: model trained on synthetic evaluated on real (higher = better)",
        value=round(float(tstr), 4),
        threshold=min_score,
        direction="higher_is_better",
        status=status,
        message=msg,
        details={
            "task": "classification" if classification else "regression",
            "target": target,
            "tstr": round(float(tstr), 4),
            "trts_reference": round(float(trts), 4),
            "delta_vs_trts": round(float(trts - tstr), 4),
        },
    )