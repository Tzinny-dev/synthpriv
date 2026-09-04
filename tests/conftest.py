"""Tests de synthpriv (Fase 0 + 1).

Los tests entrenan solo el generador rapido (GaussianCopula) para mantener el
suite veloz. Los deep generators (CTGAN/TVAE/CopulaGAN) quedan marcados como
``slow`` y no se ejecutan por defecto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def real_data() -> pd.DataFrame:
    """Dataset mixto pequeno y determinista (numerico + categorico)."""
    rng = np.random.default_rng(42)
    n = 400
    return pd.DataFrame({
        "age": rng.normal(45, 12, n).clip(18, 90).round(),
        "income": rng.gamma(3.0, 5000.0, n),
        "score": rng.normal(0.5, 0.2, n).clip(0, 1),
        "education": rng.choice(["high_school", "bachelor", "master", "phd"], n, p=[0.35, 0.4, 0.15, 0.1]),
        "is_fraud": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })


@pytest.fixture
def synth_data(real_data) -> pd.DataFrame:
    """Datos 'sinteticos' generados rapidos con la copula gaussiana."""
    from synthpriv import GaussianCopulaGenerator

    gen = GaussianCopulaGenerator(random_state=0)
    return gen.fit_and_sample(real_data, num_rows=400)