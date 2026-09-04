"""Barrido epsilon vs utilidad para explorar el trade-off privacidad/utilidad.

Entrena ``dp-gan`` con varios presupuestos de privacidad, mide el epsilon real
(accountant RDP) y evalua la utilidad de cada punto. El resultado es una tabla
ordenada por epsilon medido y un informe HTML con las curvas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from synthpriv.pipeline import PrivacyPreservingSynthesizer
from synthpriv.privacy.mechanisms import DPSGD
from synthpriv.report import render_sweep_html
from synthpriv.utils import get_logger

logger = get_logger("sweep")

_DEFAULT_EPSILONS = (0.1, 0.5, 1.0, 2.0, 5.0, 50.0)


@dataclass
class SweepResult:
    """Resultado de un barrido epsilon-utilidad.

    Cada row es un dict: ``target_epsilon``, ``measured_epsilon``,
    ``util_<metrica>``, ``priv_<metrica>`` y ``fit_seconds``.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    utility_metrics: list[str] = field(default_factory=list)
    privacy_metrics: list[str] = field(default_factory=list)

    def dataframe(self) -> pd.DataFrame:
        """Filas ordenadas por epsilon medido ascendentemente."""
        df = pd.DataFrame(self.rows)
        if "measured_epsilon" in df.columns:
            df = df.sort_values("measured_epsilon", na_position="last")
        return df.reset_index(drop=True)

    def to_csv(self, path: str | Path) -> Path:
        self.dataframe().to_csv(path, index=False)
        return Path(path)

    def save_report(self, path: str | Path) -> Path:
        """Informe HTML con la curva epsilon medido vs valor de cada metrica."""
        return render_sweep_html(self, Path(path))

    def best_tradeoff(self, metric: str, threshold: float, lower_is_better: bool = True):
        """Mejor punto: el de menor epsilon medido que cumple el umbral de ``metric``.

        Por ejemplo ``best_tradeoff("util_correlation_mae", 0.05)`` devuelve el
        punto mas privado cuya utilidad (MAE de correlaciones) sigue dentro de 0.05.
        """
        candidates = [
            r for r in self.rows
            if r.get(metric) is not None
            and ((r[metric] <= threshold) if lower_is_better else (r[metric] >= threshold))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r["measured_epsilon"] or float("inf"))


def run_epsilon_sweep(
    real_data: pd.DataFrame,
    epsilons: tuple[float, ...] = _DEFAULT_EPSILONS,
    delta: float = 1e-5,
    generator_kwargs: dict[str, Any] | None = None,
    utility_metrics: list[str] | None = None,
    privacy_metrics: list[str] | None = None,
    metric_options: dict[str, dict[str, Any]] | None = None,
    num_rows: int | None = None,
    random_state: int = 0,
) -> SweepResult:
    """Entrena ``dp-gan`` por cada ``epsilons`` y registra utilidad + epsilon real.

    Un epsilon muy grande (p.ej. 50) equivale practicamente a "sin DP": sirve de
    tope de utilidad para la arquitectura. El epsilon medido (accountant RDP)
    es el que se presenta en la curva.
    """
    generator_kwargs = generator_kwargs or {}
    utility_metrics = utility_metrics or ["ks_test", "correlation_mae", "ml_utility"]
    privacy_metrics = privacy_metrics or ["nndr", "mia_auc"]
    rows: list[dict[str, Any]] = []

    for eps in epsilons:
        privacy = DPSGD(epsilon=float(eps), delta=float(delta))
        synthesizer = PrivacyPreservingSynthesizer(
            generator_key="dp-gan",
            generator_kwargs={**generator_kwargs, "privacy": privacy},
            privacy_mechanism=privacy,
            utility_metrics=utility_metrics,
            privacy_metrics=privacy_metrics,
            metric_options=metric_options,
            random_state=random_state,
        )
        logger.info("[sweep] epsilon objetivo %.2f -> entrenando dp-gan ...", eps)
        synthesizer.fit(real_data)
        n = num_rows or len(real_data)
        synthetic = synthesizer.sample(n)
        report = synthesizer.evaluate(real_data, synthetic)
        measured = synthesizer.accountant.get_epsilon()

        row: dict[str, Any] = {
            "target_epsilon": float(eps),
            "measured_epsilon": round(measured, 4) if measured is not None else None,
            "delta": float(delta),
            "fit_seconds": round(synthesizer.timings.get("fit_seconds", 0.0), 2),
        }
        for name, res in report.data["utility"].items():
            row[f"util_{name}"] = res.value
        for name, res in report.data["privacy_metrics"].items():
            row[f"priv_{name}"] = res.value
        rows.append(row)
        logger.info("[sweep] eps objetivo %.2f -> medido %s | util=%s",
                    eps, row["measured_epsilon"], {k: v for k, v in row.items() if k.startswith("util_")})

    return SweepResult(rows=rows, utility_metrics=list(utility_metrics),
                       privacy_metrics=list(privacy_metrics))