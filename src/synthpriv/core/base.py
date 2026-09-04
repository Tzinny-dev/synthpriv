"""Contratos base para generadores de datos sinteticos.

Todos los generadores de synthpriv implementan ``BaseSynthesizer``.
La interfaz minima es ``fit`` + ``sample``, de modo que los callers
(pipeline, CLI, REST) son agnosticos al algoritmo concreto.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from synthpriv.utils import check_fitted


class BaseSynthesizer(ABC):
    """Interfaz comun para cualquier generador de datos sinteticos.

    Parameters
    ----------
    metadata:
        Metadatos opcionales (p.ej. ``SingleTableMetadata`` de SDV o dict).
        Si es ``None``, el generador los infiere en ``fit``.
    """

    name: str = "base"
    description: str = ""

    def __init__(self, metadata=None, **kwargs):  # noqa: B027
        self.metadata = metadata
        self._model = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """True si ``fit`` se ha completado correctamente."""
        return self._fitted

    @property
    def model(self):
        """Modelo subyacente entrenado (depende de la implementacion)."""
        return self._model

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseSynthesizer":
        """Entrena el generador sobre los datos reales."""

    @abstractmethod
    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        """Genera ``num_rows`` registros sinteticos."""

    def fit_and_sample(self, data: pd.DataFrame, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        """Atajo: ``fit`` + ``sample`` en un solo paso."""
        self.fit(data)
        return self.sample(num_rows=num_rows, **kwargs)

    def get_params(self) -> dict[str, Any]:
        """Parametros de configuracion reproducibles (para guardar/reesperar)."""
        return {}

    # ------------------------------------------------------------------
    # persistencia
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persiste el generador entrenado en ``path`` (formato por implementacion)."""
        raise NotImplementedError(f"{self.__class__.__name__} no implementa save()")

    @classmethod
    def load(cls, path: str | Path) -> "BaseSynthesizer":
        """Reconstruye un generador entrenado desde ``path``."""
        raise NotImplementedError(f"{cls.__name__} no implementa load()")

    def __repr__(self) -> str:  # pragma: no cover - utilidad de depuracion
        return f"<{self.__class__.__name__} name={self.name!r} fitted={self.fitted}>"