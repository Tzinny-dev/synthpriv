"""Tabular generators.

Thin wrappers over SDV single-table synthesizers. They keep the
``BaseSynthesizer`` interface (fit/sample) and register in the registry.
"""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import (
    CTGANSynthesizer,
    CopulaGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from synthpriv.core.base import BaseSynthesizer
from synthpriv.core.registry import register_generator
from synthpriv.utils import check_fitted, get_logger

logger = get_logger("generators")


class _SDVWrapper(BaseSynthesizer):
    """Base to wrap an SDV single-table synthesizer."""

    _sdv_class = None

    def __init__(self, metadata=None, random_state: int | None = None, **kwargs):
        super().__init__(metadata=metadata)
        raw = {k: v for k, v in kwargs.items() if v is not None}
        if random_state is not None:
            raw["random_state"] = random_state
        self._kwargs = self._filter_sdv_kwargs(raw)

    @classmethod
    def _filter_sdv_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop parameters the SDV synthesizer does not accept."""
        sig = inspect.signature(cls._sdv_class.__init__)
        allowed = set(sig.parameters) - {"self", "metadata"}
        return {k: v for k, v in kwargs.items() if k in allowed}

    def fit(self, data: pd.DataFrame, **kwargs) -> "_SDVWrapper":
        metadata = self.metadata
        if metadata is None:
            metadata = Metadata.detect_from_dataframe(data)
        self._model = self._sdv_class(metadata, **self._kwargs)
        logger.info("Training %s on %d rows...", self.name, len(data))
        self._model.fit(data, **kwargs)
        if self.metadata is None:
            self.metadata = metadata
        self._fitted = True
        return self

    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        check_fitted(self)
        return self._model.sample(num_rows=num_rows, **kwargs)

    def get_params(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def save(self, path: str | Path) -> Path:
        """Persist model + metadata + kwargs in a single pickle file."""
        payload = {
            "version": 1,
            "class": self.__class__.__name__,
            "name": self.name,
            "fitted": self._fitted,
            "metadata": self.metadata,
            "kwargs": self._kwargs,
            "model": self._model,
        }
        path = Path(path)
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "_SDVWrapper":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("name") != cls.name:
            raise ValueError(
                f"The file stores '{payload.get('name')}', expected '{cls.name}'."
            )
        obj = cls(metadata=payload.get("metadata"), **payload.get("kwargs", {}))
        obj._model = payload.get("model")
        obj._fitted = bool(payload.get("fitted"))
        obj.metadata = payload.get("metadata")
        return obj


@register_generator("ctgan", description="Conditional Tabular GAN (mixed numeric/categorical data)")
class CTGANGenerator(_SDVWrapper):
    """CTGAN: conditional generative adversarial network for tabular data."""

    name = "ctgan"
    _sdv_class = CTGANSynthesizer

    def __init__(self, metadata=None, epochs=300, batch_size=500, embedding_dim=128,
                 generator_dim=(256, 256), discriminator_dim=(256, 256), **kwargs):
        super().__init__(metadata=metadata,
                         epochs=epochs, batch_size=batch_size,
                         embedding_dim=embedding_dim,
                         generator_dim=generator_dim,
                         discriminator_dim=discriminator_dim, **kwargs)


@register_generator("tvae", description="Tabular Variational Autoencoder")
class TVAEGenerator(_SDVWrapper):
    """TVAE: variational autoencoder for mixed tabular data."""

    name = "tvae"
    _sdv_class = TVAESynthesizer

    def __init__(self, metadata=None, epochs=300, batch_size=500, embedding_dim=128, **kwargs):
        super().__init__(metadata=metadata, epochs=epochs, batch_size=batch_size,
                         embedding_dim=embedding_dim, **kwargs)


@register_generator("copula-gan", description="Copula + GAN for bivariate dependencies")
class CopulaGANGenerator(_SDVWrapper):
    """CopulaGAN: combines Gaussian copulas with the CTGAN architecture."""

    name = "copula-gan"
    _sdv_class = CopulaGANSynthesizer

    def __init__(self, metadata=None, epochs=300, batch_size=500, embedding_dim=128,
                 generator_dim=(256, 256), discriminator_dim=(256, 256), **kwargs):
        super().__init__(metadata=metadata, epochs=epochs, batch_size=batch_size,
                         embedding_dim=embedding_dim, generator_dim=generator_dim,
                         discriminator_dim=discriminator_dim, **kwargs)


@register_generator("gaussian-copula", description="Gaussian copula (fast, baseline)")
class GaussianCopulaGenerator(_SDVWrapper):
    """Classic Gaussian copula: fast, useful as baseline and for tests."""

    name = "gaussian-copula"
    _sdv_class = GaussianCopulaSynthesizer

    def __init__(self, metadata=None, default_distribution=None, **kwargs):
        if default_distribution is not None:
            kwargs["default_distribution"] = default_distribution
        super().__init__(metadata=metadata, **kwargs)