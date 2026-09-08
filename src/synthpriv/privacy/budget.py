"""Split of the DP budget between training and marginals.

The ``dp-gan`` synthesizer consumes ``ecdf_epsilon`` on the marginals ECDFs and
``epsilon`` on DP-SGD training; the total guarantee is
``epsilon + ecdf_epsilon`` (additive and exact). This module helps split a total
budget into those two items coherently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSplit:
    """Result of the split: ``train`` and ``margins`` items for ``dp-gan``.

    ``total`` (final composed guarantee) = ``train + margins``. For ``dp-copula``
    use ``DPCopulaGenerator`` directly with its internal fractions
    (``margins_fraction``/``corr_fraction``), which split the total automatically.
    """

    total: float
    margins_fraction: float
    train: float
    margins: float

    def describe(self) -> str:
        return (
            f"Total budget {self.total:.3f} -> training (DP-SGD) "
            f"{self.train:.3f} + marginals (DP-ECDF) {self.margins:.3f} "
            f"({self.margins_fraction:.0%} to marginals). "
            "for dp-copula the split is internal via margins/corr_fraction."
        )


def split_budget(total_epsilon: float, margins_fraction: float = 0.3) -> BudgetSplit:
    """Divide ``total_epsilon`` into training and marginals for ``dp-gan``.

    A low ``margins_fraction`` (0.1-0.4) is usually enough for marginals with many
    rows; increase it on small datasets (Laplace histogram variance grows with less
    data) or when tails matter a lot and KS drops.
    """
    if total_epsilon <= 0:
        raise ValueError(f"total_epsilon must be > 0, got {total_epsilon!r}")
    if not 0 < margins_fraction < 1:
        raise ValueError(f"margins_fraction must be in (0, 1), got {margins_fraction!r}")
    train = total_epsilon * (1.0 - margins_fraction)
    margins = total_epsilon - train
    return BudgetSplit(total=float(total_epsilon),
                       margins_fraction=float(margins_fraction),
                       train=float(train), margins=float(margins))