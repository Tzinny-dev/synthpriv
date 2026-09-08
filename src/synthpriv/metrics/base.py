"""Metric result model and threshold evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Possible states of a metric.
PASSED = "passed"
FAILED = "failed"
REPORTED = "reported"
ERROR = "error"


@dataclass
class MetricResult:
    """Result of a single metric."""

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
    """Decide PASSED/FAILED/REPORTED status by comparing against a threshold."""
    if threshold is None or value is None:
        return REPORTED, "No threshold configured; value is only reported."
    if direction == "higher_is_better":
        ok = value >= threshold
        op = ">="
    elif direction == "lower_is_better":
        ok = value <= threshold
        op = "<="
    else:
        return REPORTED, "Direction 'none': no automatic comparison."
    status = PASSED if ok else FAILED
    return status, f"Value {value:.4f} vs threshold {threshold:.4f} ({op})"


def summarize(results: dict[str, MetricResult]) -> dict[str, int]:
    """Count statuses for the report summary."""
    counts = {s: 0 for s in (PASSED, FAILED, REPORTED, ERROR)}
    for res in results.values():
        if res.status in counts:
            counts[res.status] += 1
        else:  # pragma: no cover - defensive
            counts[REPORTED] += 1
    return counts