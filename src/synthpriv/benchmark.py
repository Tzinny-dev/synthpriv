"""Benchmark de utilidad dp-gan (DP) frente a generadores SDV sin DP.

Entrena ``dp-gan`` con varios presupuestos (medidos con el accountant RDP) y a la
vez entrena generadores de referencia sin privacidad (``ctgan``, ``tvae``,
``copula-gan``, ``gaussian-copula``) sobre el mismo dataset y con la misma
evaluacion. Resultado: una tabla con la utilidad de cada punto y un informe HTML
con la curva privacidad/utilidad del dp-gan frente a las lineas de referencia de
cada baseline.

Interpretacion: si el dp-gan con epsilon ~ sin-DP (50) se acerca al baseline
mejor, la arquitectura es la limitacion; la distancia a epsilon bajo es el coste
de la privacidad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from synthpriv.core.registry import build_generator
from synthpriv.pipeline import PrivacyPreservingSynthesizer
from synthpriv.privacy.mechanisms import DPSGD, NoPrivacy
from synthpriv.report import render_benchmark_html
from synthpriv.utils import get_logger

logger = get_logger("benchmark")

_DEFAULT_EPSILONS = (1.0, 2.0, 5.0, 10.0, 50.0)
_DEFAULT_BASELINES = ("gaussian-copula", "ctgan", "tvae", "copula-gan")

# Direccion util para interpretar el gap de utilidad.
_DIRECTION: dict[str, str] = {
    "ks_test": "higher",
    "correlation_mae": "lower",
    "ml_utility": "higher",
    "nndr": "higher",
    "mia_auc": "lower",
}


def _metric_direction(name: str, default: str = "higher") -> str:
    for prefix in ("util_", "priv_"):
        key = name[len(prefix):] if name.startswith(prefix) else name
        if key in _DIRECTION:
            return _DIRECTION[key]
    return default


@dataclass
class BenchmarkResult:
    """Resultado de ``run_benchmark``.

    Cada ``row`` es un dict con: ``model`` (nombre del generador o "dp-gan"),
    ``kind`` ("dp-gan" | "baseline"), ``target_epsilon`` y ``measured_epsilon``
    (solo dp-gan), ``util_<metrica>``, ``priv_<metrica>`` y ``fit_seconds``.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    utility_metrics: list[str] = field(default_factory=list)
    privacy_metrics: list[str] = field(default_factory=list)

    def dataframe(self) -> pd.DataFrame:
        """Filas ordenadas: dp-gan por epsilon medido y luego los baselines."""
        df = pd.DataFrame(self.rows)
        if "measured_epsilon" in df.columns:
            df = df.sort_values("measured_epsilon", na_position="last")
        return df.reset_index(drop=True)

    def to_csv(self, path: str | Path) -> Path:
        self.dataframe().to_csv(path, index=False)
        return Path(path)

    def save_report(self, path: str | Path) -> Path:
        """Informe HTML con la tabla y las curvas dp-gan vs baselines."""
        return render_benchmark_html(self, Path(path))

    def curve(self, generator: str, metric: str) -> list[dict[str, float]]:
        """Serie (measured_epsilon, metric) del dp-gan, ascendente."""
        out = [
            {"x": r["measured_epsilon"], "y": r[metric]}
            for r in self.rows
            if r.get("model") == generator
            and r.get("measured_epsilon") is not None
            and r.get(metric) is not None
        ]
        return sorted(out, key=lambda p: p["x"])

    def baseline_value(self, baseline: str, metric: str) -> float | None:
        """Valor medio del baseline para una metrica (1 sola ejecucion o repetida)."""
        vals = [r[metric] for r in self.rows
                if r.get("model") == baseline and r.get(metric) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def dp_value(self, metric: str, target_epsilon: float | None = None) -> float | None:
        """Valor del dp-gan en el punto mas privado (o cercano a ``target_epsilon``)."""
        pts = self.curve("dp-gan", metric)
        if not pts:
            return None
        if target_epsilon is None:
            return min(pts, key=lambda p: p["x"])["y"]  # minimo epsilon medido
        return min(pts, key=lambda p: abs(p["x"] - target_epsilon))["y"]

    def utility_gap(self, metric: str, baseline: str,
                    target_epsilon: float | None = None) -> float | None:
        """Coste de la privacidad: cuan lejos queda dp-gan del baseline en una metrica.

        Positivo = dp-gan pierde frente al baseline; negativo = dp-gan gana.
        """
        base = self.baseline_value(baseline, metric)
        dp = self.dp_value(metric, target_epsilon=target_epsilon)
        if base is None or dp is None:
            return None
        if _metric_direction(metric) == "higher":
            return base - dp
        return dp - base

    def best_dp_point(self, metric: str, threshold: float,
                      lower_is_better: bool | None = None) -> dict[str, float] | None:
        """Punto dp-gan de menor epsilon cuyo valor cumple el umbral de ``metric``."""
        direction = _metric_direction(metric)
        lower = direction == "lower" if lower_is_better is None else lower_is_better
        candidates = [
            p for p in self.curve("dp-gan", metric)
            if ((p["y"] <= threshold) if lower else (p["y"] >= threshold))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p["x"])


def _fit_and_evaluate(
    real_data: pd.DataFrame,
    *,
    generator_key: str,
    generator_kwargs: dict[str, Any],
    privacy_mechanism,
    utility_metrics: list[str],
    privacy_metrics: list[str],
    metric_options: dict[str, dict[str, Any]] | None,
    num_rows: int | None,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Entrena un generador y devuelve (row_de_metricas, timings)."""
    synth = PrivacyPreservingSynthesizer(
        generator_key=generator_key,
        generator_kwargs=generator_kwargs,
        privacy_mechanism=privacy_mechanism,
        utility_metrics=utility_metrics,
        privacy_metrics=privacy_metrics,
        metric_options=metric_options,
        random_state=random_state,
    )
    synth.fit(real_data)
    n = num_rows or len(real_data)
    synthetic = synth.sample(n)
    report = synth.evaluate(real_data, synthetic)
    measured = synth.accountant.get_epsilon()

    row = {
        "measured_epsilon": round(measured, 4) if measured is not None else None,
        "fit_seconds": round(synth.timings.get("fit_seconds", 0.0), 2),
    }
    for name, res in report.data["utility"].items():
        row[f"util_{name}"] = res.value
    for name, res in report.data["privacy_metrics"].items():
        row[f"priv_{name}"] = res.value
    return row, report.data


def run_benchmark(
    real_data: pd.DataFrame,
    epsilons: tuple[float, ...] = _DEFAULT_EPSILONS,
    delta: float = 1e-5,
    baselines: tuple[str, ...] = ("gaussian-copula",),
    generator_kwargs: dict[str, Any] | None = None,
    baseline_kwargs: dict[str, dict[str, Any]] | None = None,
    utility_metrics: list[str] | None = None,
    privacy_metrics: list[str] | None = None,
    metric_options: dict[str, dict[str, Any]] | None = None,
    num_rows: int | None = None,
    random_state: int = 0,
) -> BenchmarkResult:
    """Benchmarka el dp-gan (a los ``epsilons`` dados) frente a ``baselines`` sin DP.

    Los baselines se entrenan una sola vez (no dependen de epsilon) con su
    configuracion por defecto; para los deep generators p.ej. pasar
    ``baseline_kwargs={"ctgan": {"epochs": 300}}``. Usa las mismas metricas en
    todos los puntos.
    """
    generator_kwargs = generator_kwargs or {}
    baseline_kwargs = baseline_kwargs or {}
    utility_metrics = list(utility_metrics or ["ks_test", "correlation_mae", "ml_utility"])
    privacy_metrics = list(privacy_metrics or ["nndr", "mia_auc"])
    rows: list[dict[str, Any]] = []

    for eps in epsilons:
        privacy = DPSGD(epsilon=float(eps), delta=float(delta))
        logger.info("[benchmark] dp-gan | epsilon objetivo %.2f ...", eps)
        row, _ = _fit_and_evaluate(
            real_data,
            generator_key="dp-gan",
            generator_kwargs={**generator_kwargs, "privacy": privacy},
            privacy_mechanism=privacy,
            utility_metrics=utility_metrics,
            privacy_metrics=privacy_metrics,
            metric_options=metric_options,
            num_rows=num_rows,
            random_state=random_state,
        )
        rows.append({
            "model": "dp-gan", "kind": "dp-gan",
            "target_epsilon": float(eps), "delta": float(delta), **row,
        })
        logger.info("[benchmark]   medido %.4f | util=%s", row["measured_epsilon"],
                    {k: v for k, v in row.items() if k.startswith("util_")})

    for name in baselines:
        try:
            build_generator(name)  # valida que exista antes de entrenar
        except KeyError as exc:  # pragma: no cover - defensivo
            raise ValueError(f"Baseline desconocido: {name!r}") from exc
        logger.info("[benchmark] baseline %s (sin DP) ...", name)
        row, _ = _fit_and_evaluate(
            real_data,
            generator_key=name,
            generator_kwargs=baseline_kwargs.get(name, {}),
            privacy_mechanism=NoPrivacy(),
            utility_metrics=utility_metrics,
            privacy_metrics=privacy_metrics,
            metric_options=metric_options,
            num_rows=num_rows,
            random_state=random_state,
        )
        rows.append({
            "model": name, "kind": "baseline",
            "target_epsilon": None, "delta": float(delta), **row,
        })
        logger.info("[benchmark]   %s util=%s", name,
                    {k: v for k, v in row.items() if k.startswith("util_")})

    return BenchmarkResult(
        rows=rows,
        baselines=list(baselines),
        utility_metrics=utility_metrics,
        privacy_metrics=privacy_metrics,
    )