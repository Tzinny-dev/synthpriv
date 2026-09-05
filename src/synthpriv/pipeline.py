"""Pipeline orquestador: entrenar, muestrear, evaluar y reportar."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from synthpriv.core.base import BaseSynthesizer
from synthpriv.core.registry import build_generator, get_generator
from synthpriv.metrics.base import summarize
from synthpriv.metrics.core import (
    evaluate_privacy,
    evaluate_utility,
    metrics_summary,
)
from synthpriv.privacy.accountant import PrivacyAccountant
from synthpriv.privacy.assurance import DpAssurance, assert_dp
from synthpriv.privacy.mechanisms import NoPrivacy, PrivacyMechanism
from synthpriv.utils import get_logger, timed_block
from synthpriv.report import render_html

logger = get_logger("pipeline")


@dataclass
class EvaluationReport:
    """Contenedor del resultado de ``evaluate``."""

    data: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        def _fmt(group: dict) -> str:
            counts = summarize(group)
            return f"{counts['passed']} OK / {counts['failed']} FAIL / {counts['reported']} n/a"

        util = self.data.get("utility", {})
        priv = self.data.get("privacy_metrics", {})
        return f"Utilidad [{_fmt(util)}] | Privacidad [{_fmt(priv)}]"

    def save(self, path: str | Path) -> Path:
        """Genera el informe HTML autocontenido en ``path``."""
        return render_html(self.data, Path(path))


class PrivacyPreservingSynthesizer:
    """Orquesta generador + mecanismo de privacidad + metricas.

    Examples
    --------
    ```python
    from synthpriv import PrivacyPreservingSynthesizer
    from synthpriv.privacy import NoPrivacy

    synth = PrivacyPreservingSynthesizer(
        generator_key="ctgan",
        generator_kwargs={"epochs": 100},
        privacy_mechanism=NoPrivacy(),
        utility_metrics=["ks_test", "correlation_mae", "ml_utility"],
        privacy_metrics=["nndr", "mia_auc"],
    )
    synth.fit(df)
    synthetic = synth.sample(5000)
    report = synth.evaluate(df, synthetic)
    report.save("report.html")
    ```
    """

    def __init__(
        self,
        generator: BaseSynthesizer | type | str | None = None,
        generator_key: str | None = None,
        generator_kwargs: dict[str, Any] | None = None,
        privacy_mechanism: PrivacyMechanism | None = None,
        privacy_metrics: list[str] | None = None,
        utility_metrics: list[str] | None = None,
        metric_options: dict[str, dict[str, Any]] | None = None,
        random_state: int = 0,
    ):
        self.generator = self._resolve_generator(generator, generator_key, generator_kwargs, random_state)
        self.privacy_mechanism = privacy_mechanism or NoPrivacy()
        self._validate_privacy(self.generator, self.privacy_mechanism)
        self.accountant = PrivacyAccountant(self.privacy_mechanism)
        self.privacy_metrics = privacy_metrics
        self.utility_metrics = utility_metrics
        self.metric_options = metric_options or {}
        self.timings: dict[str, float] = {}

    # -- construccion ------------------------------------------------------
    @staticmethod
    def _resolve_generator(generator, key, kwargs, random_state: int) -> BaseSynthesizer:
        kwargs = kwargs or {}
        if generator is None:
            if key is None:
                key = "ctgan"
                logger.warning("Sin generador: se usa el por defecto 'ctgan'.")
            return build_generator(key, random_state=random_state, **kwargs)
        if isinstance(generator, str):
            return build_generator(generator, random_state=random_state, **kwargs)
        if isinstance(generator, BaseSynthesizer):
            return generator
        if isinstance(generator, type) and issubclass(generator, BaseSynthesizer):
            return generator(random_state=random_state, **kwargs)
        raise TypeError(
            f"generator debe ser instancia, clase o clave; se recibio {type(generator)}."
        )

    @staticmethod
    def _validate_privacy(generator: BaseSynthesizer, mechanism: PrivacyMechanism) -> None:
        if mechanism.is_dp and not getattr(generator, "dp_capable", False):
            raise ValueError(
                f"El generador '{getattr(generator, 'name', None)}' no es capaz de entrenar "
                "diferencialmente privado. Con un mecanismo DPSGD usa generator_key='dp-gan'."
            )

    # -- interfaz principal ------------------------------------------------
    def fit(self, real_data: pd.DataFrame) -> "PrivacyPreservingSynthesizer":
        with timed_block(f"Entrenamiento {self.generator.name}") as timer:
            self.generator.fit(real_data)
            self.timings["fit_seconds"] = timer()
        measured = getattr(self.generator, "accounted_epsilon", None)
        if measured is not None:
            self.accountant.set_effective_epsilon(measured)
        self.real_shape = real_data.shape
        return self

    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        with timed_block(f"Muestreo {num_rows} filas") as timer:
            out = self.generator.sample(num_rows=num_rows, **kwargs)
            self.timings["sample_seconds"] = timer()
        return out

    def generate(self, real_data: pd.DataFrame, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        """Atajo: fit + sample."""
        return self.fit(real_data).sample(num_rows, **kwargs)

    def assert_dp(self, declared_epsilon: float | None = None, *,
                  tolerance: float = 0.05) -> DpAssurance:
        """Valida la garantia DP del sintetizador contra el epsilon del accountant.

        ``declared_epsilon`` por defecto es el del mecanismo configurado. El
        resultado reconcilia el epsilon medido por el generador con el del
        accountant (si discrepan, manda el del accountant tras ``fit``).
        """
        declared = declared_epsilon if declared_epsilon is not None else \
            getattr(self.privacy_mechanism, "epsilon", None)
        assurance = assert_dp(self.generator, declared, tolerance=tolerance)
        effective = self.accountant.get_epsilon()
        if effective is not None and assurance.measured_epsilon is not None:
            if assurance.measured_epsilon != effective:
                assurance.measured_epsilon = effective
                assurance.budget_respected = (
                    effective <= (assurance.declared_epsilon or float("inf"))
                )
        return assurance

    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame | None = None,
        num_rows: int | None = None,
        **sample_kwargs,
    ) -> EvaluationReport:
        """Evalua privacidad y utilidad de los datos sinteticos frente a los reales."""
        if synthetic_data is None:
            synthetic_data = self.sample(num_rows or len(real_data), **sample_kwargs)

        with timed_block("Evaluacion de utilidad"):
            utility = evaluate_utility(real_data, synthetic_data, self.utility_metrics, self.metric_options)
        with timed_block("Evaluacion de privacidad"):
            privacy_metrics = evaluate_privacy(real_data, synthetic_data, self.privacy_metrics, self.metric_options)

        try:
            spec = get_generator(self.generator.name)
        except KeyError:
            spec = getattr(self.generator, "name", "unknown")

        acc = self.accountant.report()
        ecdf_eps = getattr(self.generator, "ecdf_epsilon", None)
        if ecdf_eps is not None:
            acc["ecdf_epsilon"] = float(ecdf_eps)
            base = acc.get("effective_epsilon")
            acc["total_epsilon"] = (base + float(ecdf_eps)) if base is not None else float(ecdf_eps)
            acc["explanation"] = (
                "Epsilon total = DP-SGD del entrenamiento + DP-ECDF de marginales "
                "(composicion secuencial)."
            )

        report_data = {
            "generator": {
                "key": getattr(self.generator, "name", None),
                "description": getattr(self.generator, "description", "") or getattr(spec, "description", ""),
            },
            "rows": {"real": len(real_data), "synthetic": len(synthetic_data)},
            "privacy_mechanism": {
                "configured": self.privacy_mechanism.get_report(),
                "accountant": acc,
                "assurance": self.assert_dp().as_dict(),
            },
            "utility": utility,
            "privacy_metrics": privacy_metrics,
            "summary": metrics_summary(utility, privacy_metrics),
            "timings": dict(self.timings),
        }
        return EvaluationReport(data=report_data)

    # -- persistencia del sintetizador completo ----------------------------
    def save_model(self, path: str | Path) -> Path:
        """Persiste el sintetizador entrenado (generador + privacidad + metricas).

        Genera ``path`` (el generador) y ``path.meta`` (config de privacidad,
        epsilon medio y metricas). Al ``load_model`` no haria falta reentrenar.
        """
        path = Path(path)
        generator_file = self.generator.save(path)
        meta = {
            "version": 1,
            "generator_name": self.generator.name,
            "privacy_mechanism": self.privacy_mechanism,
            "effective_epsilon": self.accountant.get_epsilon(),
            "ecdf_epsilon": getattr(self.generator, "ecdf_epsilon", None),
            "privacy_metrics": self.privacy_metrics,
            "utility_metrics": self.utility_metrics,
            "metric_options": self.metric_options,
            "timings": dict(self.timings),
            "real_shape": getattr(self, "real_shape", None),
        }
        torch.save(meta, str(path) + ".meta")
        logger.info("Sintetizador persistido en %s (epsilon medio %.3f)",
                    path, meta["effective_epsilon"] or -1.0)
        return generator_file

    @classmethod
    def load_model(cls, path: str | Path) -> "PrivacyPreservingSynthesizer":
        """Reconstruye un sintetizador persistido con ``save_model`` (sin reentrenar)."""
        path = Path(path)
        meta = torch.load(str(path) + ".meta", map_location="cpu", weights_only=False)
        spec = get_generator(meta["generator_name"])
        generator = spec.cls.load(path)
        synthesizer = cls(
            generator=generator,
            privacy_mechanism=meta["privacy_mechanism"],
            privacy_metrics=meta.get("privacy_metrics"),
            utility_metrics=meta.get("utility_metrics"),
            metric_options=meta.get("metric_options") or {},
        )
        if meta.get("effective_epsilon") is not None:
            synthesizer.accountant.set_effective_epsilon(meta["effective_epsilon"])
        synthesizer.timings = dict(meta.get("timings") or {})
        synthesizer.real_shape = meta.get("real_shape")
        logger.info("Sintetizador recargado de %s (%s, epsilon medio %.3f)",
                    path, meta["generator_name"], meta.get("effective_epsilon") or -1.0)
        return synthesizer

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PrivacyPreservingSynthesizer generator={self.generator.name}>"