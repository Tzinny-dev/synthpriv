"""Utility benchmark of dp-gan (DP) vs non-DP SDV generators.

Trains ``dp-gan`` with several budgets (measured with the RDP accountant) and
also trains reference generators without privacy (``ctgan``, ``tvae``,
``copula-gan``, ``gaussian-copula``) on the same dataset with the same
evaluation. Result: a table with the utility of each point and an HTML report
with the dp-gan privacy/utility curve against the reference lines of each
baseline.

Interpretation: if dp-gan at epsilon ~ no-DP (50) gets close to the best
baseline, the architecture is the limitation; the distance at low epsilon is the
cost of privacy.
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

# Direction useful to interpret the utility gap.
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
    """Result of ``run_benchmark``.

    Each ``row`` is a dict with: ``model`` (generator name or "dp-gan"),
    ``kind`` ("dp-gan" | "baseline"), ``target_epsilon`` and ``measured_epsilon``
    (dp-gan only), ``util_<metric>``, ``priv_<metric>`` and ``fit_seconds``.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    utility_metrics: list[str] = field(default_factory=list)
    privacy_metrics: list[str] = field(default_factory=list)

    def dataframe(self) -> pd.DataFrame:
        """Rows ordered: dp-gan by measured epsilon, then the baselines."""
        df = pd.DataFrame(self.rows)
        if "measured_epsilon" in df.columns:
            df = df.sort_values("measured_epsilon", na_position="last")
        return df.reset_index(drop=True)

    def to_csv(self, path: str | Path) -> Path:
        self.dataframe().to_csv(path, index=False)
        return Path(path)

    def save_report(self, path: str | Path) -> Path:
        """HTML report with the table and the dp-gan vs baselines curves."""
        return render_benchmark_html(self, Path(path))

    def curve(self, generator: str, metric: str) -> list[dict[str, float]]:
        """(measured_epsilon, metric) series of the DP generator, ascending."""
        out = [
            {"x": r["measured_epsilon"], "y": r[metric]}
            for r in self.rows
            if r.get("model") == generator
            and r.get("measured_epsilon") is not None
            and r.get(metric) is not None
        ]
        return sorted(out, key=lambda p: p["x"])

    def baseline_value(self, baseline: str, metric: str) -> float | None:
        """Mean baseline value for a metric (single run or repeated)."""
        vals = [r[metric] for r in self.rows
                if r.get("model") == baseline and r.get(metric) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def dp_value(self, metric: str, target_epsilon: float | None = None) -> float | None:
        """dp-gan value at the most private point (or close to ``target_epsilon``)."""
        pts = self.curve("dp-gan", metric)
        if not pts:
            return None
        if target_epsilon is None:
            return min(pts, key=lambda p: p["x"])["y"]  # minimum measured epsilon
        return min(pts, key=lambda p: abs(p["x"] - target_epsilon))["y"]

    def utility_gap(self, metric: str, baseline: str,
                    target_epsilon: float | None = None) -> float | None:
        """Cost of privacy: how far dp-gan is from the baseline on a metric.

        Positive = dp-gan loses to the baseline; negative = dp-gan wins.
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
        """Lowest-epsilon dp-gan point whose value meets the ``metric`` threshold."""
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
    """Train a generator and return (metrics_row, timings)."""
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
    """Benchmark dp-gan (at the given ``epsilons``) against non-DP ``baselines``.

    Baselines are trained once (they do not depend on epsilon) with their default
    configuration; for deep generators e.g. pass
    ``baseline_kwargs={"ctgan": {"epochs": 300}}``. Uses the same metrics on all
    points.
    """
    generator_kwargs = generator_kwargs or {}
    baseline_kwargs = baseline_kwargs or {}
    utility_metrics = list(utility_metrics or ["ks_test", "correlation_mae", "ml_utility"])
    privacy_metrics = list(privacy_metrics or ["nndr", "mia_auc"])
    rows: list[dict[str, Any]] = []

    for eps in epsilons:
        privacy = DPSGD(epsilon=float(eps), delta=float(delta))
        logger.info("[benchmark] dp-gan | target epsilon %.2f ...", eps)
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
        logger.info("[benchmark]   measured %.4f | util=%s", row["measured_epsilon"],
                    {k: v for k, v in row.items() if k.startswith("util_")})

    for name in baselines:
        try:
            build_generator(name)  # validates it exists before training
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown baseline: {name!r}") from exc
        logger.info("[benchmark] baseline %s (no DP) ...", name)
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