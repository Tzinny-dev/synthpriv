from __future__ import annotations

import pytest

from synthpriv import (
    CTGANGenerator,
    GaussianCopulaGenerator,
    GeneratorNotFoundError,
    TVAEGenerator,
    build_generator,
    get_generator,
    list_generators,
    register_generator,
)


def test_registry_lists_builtins():
    keys = list_generators()
    assert {"ctgan", "tvae", "copula-gan", "gaussian-copula"} <= set(keys)


def test_build_known_generator():
    gen = build_generator("gaussian-copula")
    assert isinstance(gen, GaussianCopulaGenerator)


def test_build_unknown_raises():
    with pytest.raises(GeneratorNotFoundError):
        build_generator("does-not-exist")


def test_register_duplicate_key_raises():
    """Registering an existing key again must fail at decoration time."""
    with pytest.raises(ValueError):
        @register_generator("gaussian-copula")
        class Dummy:  # pragma: no cover
            pass


def test_generators_expose_interface():
    for cls in (CTGANGenerator, GaussianCopulaGenerator, TVAEGenerator):
        assert hasattr(cls, "fit")
        assert hasattr(cls, "sample")
        assert getattr(cls, "name", None)


def test_get_generator_spec():
    spec = get_generator("ctgan")
    assert spec.key == "ctgan"
    assert spec.description