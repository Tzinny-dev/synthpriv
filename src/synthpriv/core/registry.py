"""Registro de generadores.

Los generadores se dan de alta con el decorador ``@register_generator`` y se
instancian por nombre (clave). Esto permite al pipeline, la CLI o una API REST
resolver generadores sin importar directamente cada clase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from synthpriv.core.base import BaseSynthesizer

SynthesizerT = TypeVar("SynthesizerT", bound="BaseSynthesizer")


class GeneratorNotFoundError(KeyError):
    """Se pidio un generador que no esta registrado."""


@dataclass
class GeneratorSpec:
    """Entrada del registro."""

    key: str
    cls: type
    description: str = field(default="")
    supports: tuple[str, ...] = field(default=("tabular",))


_REGISTRY: dict[str, GeneratorSpec] = {}


def register_generator(key: str, description: str = "", supports: tuple[str, ...] = ("tabular",)):
    """Decorador para registrar una clase como generador bajo ``key``."""

    def decorate(cls: type) -> type:
        if key in _REGISTRY:
            raise ValueError(f"El generador {key!r} ya esta registrado")
        doc = description or getattr(cls, "description", "") or cls.__doc__ or ""
        _REGISTRY[key] = GeneratorSpec(key=key, cls=cls, description=doc.strip(), supports=supports)
        return cls

    return decorate


def list_generators() -> list[str]:
    """Claves de todos los generadores registrados."""
    return sorted(_REGISTRY)


def get_generator(key: str) -> GeneratorSpec:
    """Devuelve la especificacion de un generador registrado."""
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise GeneratorNotFoundError(
            f"Generador {key!r} no registrado. Disponibles: {list_generators()}"
        ) from exc


def build_generator(key: str, *args, **kwargs) -> "BaseSynthesizer":
    """Instancia un generador a partir de su clave registrada."""
    spec = get_generator(key)
    return spec.cls(*args, **kwargs)