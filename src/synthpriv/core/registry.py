"""Generator registry.

Generators are registered with the ``@register_generator`` decorator and
instantiated by name (key). This lets the pipeline, the CLI or a REST API
resolve generators without importing each class directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from synthpriv.core.base import BaseSynthesizer

SynthesizerT = TypeVar("SynthesizerT", bound="BaseSynthesizer")


class GeneratorNotFoundError(KeyError):
    """A generator that is not registered was requested."""


@dataclass
class GeneratorSpec:
    """Registry entry."""

    key: str
    cls: type
    description: str = field(default="")
    supports: tuple[str, ...] = field(default=("tabular",))


_REGISTRY: dict[str, GeneratorSpec] = {}


def register_generator(key: str, description: str = "", supports: tuple[str, ...] = ("tabular",)):
    """Decorator to register a class as a generator under ``key``."""

    def decorate(cls: type) -> type:
        if key in _REGISTRY:
            raise ValueError(f"Generator {key!r} is already registered")
        doc = description or getattr(cls, "description", "") or cls.__doc__ or ""
        _REGISTRY[key] = GeneratorSpec(key=key, cls=cls, description=doc.strip(), supports=supports)
        return cls

    return decorate


def list_generators() -> list[str]:
    """Keys of all registered generators."""
    return sorted(_REGISTRY)


def get_generator(key: str) -> GeneratorSpec:
    """Return the specification of a registered generator."""
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise GeneratorNotFoundError(
            f"Generator {key!r} not registered. Available: {list_generators()}"
        ) from exc


def build_generator(key: str, *args, **kwargs) -> "BaseSynthesizer":
    """Instantiate a generator from its registered key."""
    spec = get_generator(key)
    return spec.cls(*args, **kwargs)