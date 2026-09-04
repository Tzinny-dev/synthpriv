"""Modelo de resultados de metricas y evaluacion de umbrales."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Estados posibles de una metrica.
PASSED = "passed"
FAILED = "failed"
REPORTED = "reported"
ERROR = "error"


@dataclass
class MetricResult:
    """Resultado de una metrica unica."""

    name: str
    description: str = ""
    value: Any = None
    threshold: Any = None
    direction: str = "lower_is_better"  # "lower_is_better" | "higher_is_better" | "none"
    status: str = REPORTED
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == PASSED


def evaluate_status(
    value: float,
    threshold: float | None,
    direction: str,
) -> tuple[str, str]:
    """Decide estado PASSED/FAILED/REPORTED comparando contra un umbral."""
    if threshold is None or value is None:
        return REPORTED, "Sin umbral configurado; solo se reporta el valor."
    if direction == "higher_is_better":
        ok = value >= threshold
        op = ">="
    elif direction == "lower_is_better":
        ok = value <= threshold
        op = "<="
    else:
        return REPORTED, "Direccion 'none': sin comparacion automatica."
    status = PASSED if ok else FAILED
    return status, f"Valor {value:.4f} vs umbral {threshold:.4f} ({op})"


def summarize(results: dict[str, MetricResult]) -> dict[str, int]:
    """Cuenta estados para el resumen del informe."""
    counts = {s: 0 for s in (PASSED, FAILED, REPORTED, ERROR)}
    for res in results.values():
        if res.status in counts:
            counts[res.status] += 1
        else:  # pragma: no cover - defensivo
            counts[REPORTED] += 1
    return counts