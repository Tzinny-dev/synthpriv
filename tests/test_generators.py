from __future__ import annotations

import pytest

from synthpriv import GaussianCopulaGenerator
from synthpriv.utils import check_fitted


def test_gaussian_copula_roundtrip(real_data):
    gen = GaussianCopulaGenerator(random_state=7)
    out = gen.fit_and_sample(real_data, num_rows=250)
    assert out.shape[0] == 250
    assert set(real_data.columns) <= set(out.columns)


def test_sample_before_fit_raises(real_data):
    gen = GaussianCopulaGenerator()
    with pytest.raises(RuntimeError, match="fit"):
        gen.sample(10)


def test_check_fitted_positive(real_data):
    gen = GaussianCopulaGenerator()
    gen.fit(real_data)
    check_fitted(gen)  # no debe lanzar


def test_categorical_values_preserved(real_data):
    gen = GaussianCopulaGenerator(random_state=1)
    out = gen.fit_and_sample(real_data, num_rows=1000)
    allowed = set(real_data["education"].unique())
    assert set(out["education"].dropna().unique()) <= allowed


@pytest.mark.slow
def test_ctgan_smoke(real_data):
    """Comprobacion de humo con pocas epochs (no se ejecuta por defecto)."""
    from synthpriv import CTGANGenerator

    gen = CTGANGenerator(epochs=1, batch_size=100)
    out = gen.fit_and_sample(real_data, num_rows=50)
    assert out.shape[0] == 50