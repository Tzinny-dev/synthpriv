"""Generadores tabulares.

Wrappers finos sobre los sintetizadores de SDV single-table. Mantienen la
interfaz ``BaseSynthesizer`` (fit/sample) y se registran en el registry.
"""

from __future__ import annotations

import inspect
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
    """Base para envolver un sintetizador single-table de SDV."""

    _sdv_class = None

    def __init__(self, metadata=None, random_state: int | None = None, **kwargs):
        super().__init__(metadata=metadata)
        raw = {k: v for k, v in kwargs.items() if v is not None}
        if random_state is not None:
            raw["random_state"] = random_state
        self._kwargs = self._filter_sdv_kwargs(raw)

    @classmethod
    def _filter_sdv_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Descarta parametros que el sintetizador SDV no acepta."""
        sig = inspect.signature(cls._sdv_class.__init__)
        allowed = set(sig.parameters) - {"self", "metadata"}
        return {k: v for k, v in kwargs.items() if k in allowed}

    def fit(self, data: pd.DataFrame, **kwargs) -> "_SDVWrapper":
        metadata = self.metadata
        if metadata is None:
            metadata = Metadata.detect_from_dataframe(data)
        self._model = self._sdv_class(metadata, **self._kwargs)
        logger.info("Entrenando %s sobre %d filas...", self.name, len(data))
        self._model.fit(data, **kwargs)
        if self.metadata is None:
            self.metadata = metadata
        self._fitted = True
        return self

    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        check_fitted(self)
        return self._model.sample(num_rows=num_rows, **kwargs)


@register_generator("ctgan", description="Conditional Tabular GAN (datos mixtos numericos/categoricos)")
class CTGANGenerator(_SDVWrapper):
    """CTGAN: modelo generativo adversario condicionado para tabulares."""

    name = "ctgan"
    _sdv_class = CTGANSynthesizer

    def __init__(self, metadata=None, epochs=300, batch_size=500, embedding_dim=128,
                 generator_dim=(256, 256), discriminator_dim=(256, 256), **kwargs):
        super().__init__(metadata=metadata,
                         epochs=epochs, batch_size=batch_size,
                         embedding_dim=embedding_dim,
                         generator_dim=generator_dim,
                         discriminator_dim=discriminator_dim, **kwargs)


@register_generator("tvae", description="Variational Autoencoder tabular")
class TVAEGenerator(_SDVWrapper):
    """TVAE: autoencoder variacional para datos tabulares mixtos."""

    name = "tvae"
    _sdv_class = TVAESynthesizer

    def __init__(self, metadata=None, epochs=300, batch_size=500, embedding_dim=128, **kwargs):
        super().__init__(metadata=metadata, epochs=epochs, batch_size=batch_size,
                         embedding_dim=embedding_dim, **kwargs)


@register_generator("copula-gan", description="Copula + GAN para dependencias bivariadas")
class CopulaGANGenerator(_SDVWrapper):
    """CopulaGAN: combina copulas gaussianas con la arquitectura CTGAN."""

    name = "copula-gan"
    _sdv_class = CopulaGANSynthesizer

    def __init__(self, metadata=None, epochs=300, batch_size=500, embedding_dim=128,
                 generator_dim=(256, 256), discriminator_dim=(256, 256), **kwargs):
        super().__init__(metadata=metadata, epochs=epochs, batch_size=batch_size,
                         embedding_dim=embedding_dim, generator_dim=generator_dim,
                         discriminator_dim=discriminator_dim, **kwargs)


@register_generator("gaussian-copula", description="Copula gaussiana (rapido, baseline)")
class GaussianCopulaGenerator(_SDVWrapper):
    """Copula gaussiana clasica: rapida, util como baseline y para tests."""

    name = "gaussian-copula"
    _sdv_class = GaussianCopulaSynthesizer

    def __init__(self, metadata=None, default_distribution=None, **kwargs):
        if default_distribution is not None:
            kwargs["default_distribution"] = default_distribution
        super().__init__(metadata=metadata, **kwargs)