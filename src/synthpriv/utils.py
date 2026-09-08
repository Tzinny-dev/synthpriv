"""Shared utilities (logging, validation)."""

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
    """synthpriv logger (child of the ``synthpriv`` root)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name != LOGGER_NAME else LOGGER_NAME)


def check_fitted(instance) -> None:
    """Raise if the generator/method has not been trained yet."""
    if not getattr(instance, "fitted", False):
        raise RuntimeError(
            f"{instance.__class__.__name__} is not fitted: call .fit(real_data) first."
        )


@contextmanager
def timed_block(label: str):
    """Record how long a block takes; the returned value reports elapsed seconds."""
    logger = get_logger("timing")
    start = time.perf_counter()
    logger.info("Starting: %s", label)
    try:
        yield lambda: time.perf_counter() - start
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Completed: %s (%.2fs)", label, elapsed)