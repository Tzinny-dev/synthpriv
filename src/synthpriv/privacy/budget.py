"""Reparto de presupuesto DP entre el entrenamiento y los marginales.

El sintetizador ``dp-gan`` consume ``ecdf_epsilon`` en las ECDF de marginales y
``epsilon`` en el entrenamiento DP-SGD; la garantia total es
``epsilon + ecdf_epsilon`` (aditiva y exacta). Este módulo ayuda a partir un
presupuesto total en esas dos partidas de forma coherente.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSplit:
    """Resultado del reparto: partidas ``train`` y ``margins`` para ``dp-gan``.

    ``total`` (garantia compuesta final) = ``train + margins``. Para ``dp-copula``
    usa directamente ``DPCopulaGenerator`` con las fracciones internas
    (``margins_fraction``/``corr_fraction``), que reparten el total automaticamente.
    """

    total: float
    margins_fraction: float
    train: float
    margins: float

    def describe(self) -> str:
        return (
            f"Presupuesto total {self.total:.3f} -> entrenamiento (DP-SGD) "
            f"{self.train:.3f} + marginales (DP-ECDF) {self.margins:.3f} "
            f"({self.margins_fraction:.0%} a marginales). "
            "para dp-copula el reparto es interno via margins/corr_fraction."
        )


def split_budget(total_epsilon: float, margins_fraction: float = 0.3) -> BudgetSplit:
    """Divide ``total_epsilon`` en entrenamiento y marginales para ``dp-gan``.

    ``margins_fraction`` baja (0.1-0.4) suele bastar para marginales con muchas
    filas; aumentala si el dataset es pequeno (DCvariance de los histogramas
    Laplace crece con menos datos) o si las colas importan mucho y el KS cae.
    """
    if total_epsilon <= 0:
        raise ValueError(f"total_epsilon debe ser > 0, se recibio {total_epsilon!r}")
    if not 0 < margins_fraction < 1:
        raise ValueError(f"margins_fraction debe estar en (0, 1), se recibio {margins_fraction!r}")
    train = total_epsilon * (1.0 - margins_fraction)
    margins = total_epsilon - train
    return BudgetSplit(total=float(total_epsilon),
                       margins_fraction=float(margins_fraction),
                       train=float(train), margins=float(margins))