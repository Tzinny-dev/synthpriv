"""Utilidades compartidas (logging, validacion)."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

LOGGER_NAME = "synthpriv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Logger de synthpriv (hijo de la raiz ``synthpriv``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name != LOGGER_NAME else LOGGER_NAME)


def check_fitted(instance) -> None:
    """Levanta si el generador/metodo no ha sido entrenado todavia."""
    if not getattr(instance, "fitted", False):
        raise RuntimeError(
            f"{instance.__class__.__name__} no esta entrenado: llama a .fit(real_data) primero."
        )


@contextmanager
def timed_block(label: str):
    """Registra cuanto tarda un bloque; el valor devuelto reporta los segundos transcurridos."""
    logger = get_logger("timing")
    start = time.perf_counter()
    logger.info("Iniciando: %s", label)
    try:
        yield lambda: time.perf_counter() - start
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Completado: %s (%.2fs)", label, elapsed)