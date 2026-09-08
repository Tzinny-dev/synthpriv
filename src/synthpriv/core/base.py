"""Base contracts for synthetic data generators.

All synthpriv generators implement ``BaseSynthesizer``.
The minimal interface is ``fit`` + ``sample``, so callers
(pipeline, CLI, REST) are agnostic to the concrete algorithm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from synthpriv.utils import check_fitted


class BaseSynthesizer(ABC):
    """Common interface for any synthetic data generator.

    Parameters
    ----------
    metadata:
        Optional metadata (e.g. SDV ``SingleTableMetadata`` or a dict).
        If ``None``, the generator infers it in ``fit``.
    """

    name: str = "base"
    description: str = ""

    def __init__(self, metadata=None, **kwargs):  # noqa: B027
        self.metadata = metadata
        self._model = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """True if ``fit`` completed successfully."""
        return self._fitted

    @property
    def model(self):
        """Trained underlying model (implementation-specific)."""
        return self._model

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseSynthesizer":
        """Train the generator on the real data."""

    @abstractmethod
    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        """Generate ``num_rows`` synthetic records."""

    def fit_and_sample(self, data: pd.DataFrame, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        """Shortcut: ``fit`` + ``sample`` in a single step."""
        self.fit(data)
        return self.sample(num_rows=num_rows, **kwargs)

    def get_params(self) -> dict[str, Any]:
        """Reproducible configuration parameters (for save/resample)."""
        return {}

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist the trained generator to ``path`` (implementation-defined format)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save()")

    @classmethod
    def load(cls, path: str | Path) -> "BaseSynthesizer":
        """Rebuild a trained generator from ``path``."""
        raise NotImplementedError(f"{cls.__name__} does not implement load()")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.__class__.__name__} name={self.name!r} fitted={self.fitted}>"