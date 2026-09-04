"""Registro de metricas y funciones de orquestacion."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from synthpriv.metrics.base import MetricResult, summarize
from synthpriv.metrics import privacy as _privacy_metrics
from synthpriv.metrics import utility as _utility_metrics

# Las metricas son callables ``f(real: DataFrame, synth: DataFrame, **kw) -> MetricResult``.
_METRICS: dict[str, Callable[..., MetricResult]] = {}


def register_metric(fn: Callable[..., MetricResult], name: str | None = None):
    """Registra una funcion de metrica bajo ``name`` (sys.modules[fn.__module__].__name__)."""
    key = name or fn.__name__
    if key in _METRICS:
        raise ValueError(f"La metrica {key!r} ya esta registrada")
    _METRICS[key] = fn
    return fn


for _fn in (_utility_metrics.ks_test,
            _utility_metrics.correlation_mae,
            _utility_metrics.ml_utility,
            _privacy_metrics.nndr,
            _privacy_metrics.mia_auc,
            _privacy_metrics.anonymeter_discovery,
            _privacy_metrics.anonymeter_inference,
            _privacy_metrics.anonymeter_linkability):
    register_metric(_fn)


def list_metric_names() -> list[str]:
    return sorted(_METRICS)


def run_metric(name: str, real: pd.DataFrame, synth: pd.DataFrame, **kwargs) -> MetricResult:
    try:
        return _METRICS[name](real, synth, **kwargs)
    except Exception as exc:  # pragma: no cover - defensivo
        return MetricResult(
            name=name,
            status="error",
            message=f"La metrica {name!r} fallo: {exc}",
        )


def evaluate_metrics(real, synth, metric_names, metric_options=None) -> dict[str, MetricResult]:
    """Ejecuta una lista de metricas sobre (real, synth)."""
    metric_options = metric_options or {}
    results = {}
    for name in metric_names:
        results[name] = run_metric(name, real, synth, **metric_options.get(name, {}))
    return results


_UTILITY_DEFAULTS = ["ks_test", "correlation_mae", "ml_utility"]
_PRIVACY_DEFAULTS = ["nndr", "mia_auc", "anonymeter_discovery"]


def evaluate_utility(real, synth, metric_names=None, metric_options=None) -> dict[str, MetricResult]:
    return evaluate_metrics(real, synth, metric_names or _UTILITY_DEFAULTS, metric_options)


def evaluate_privacy(real, synth, metric_names=None, metric_options=None) -> dict[str, MetricResult]:
    return evaluate_metrics(real, synth, metric_names or _PRIVACY_DEFAULTS, metric_options)


def metrics_summary(*groups: dict[str, MetricResult]) -> dict[str, int]:
    """Resumen combinado de varios grupos de metricas."""
    combined: dict[str, MetricResult] = {}
    for group in groups:
        combined.update(group)
    return summarize(combined)