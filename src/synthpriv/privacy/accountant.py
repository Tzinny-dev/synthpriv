"""Accountant de privacidad.

En la fase 2 el epsilon real lo calcula Opacus (RDP) durante ``fit`` del
generador ``dp-gan`` y lo registra aqui; el informe muestra ese valor, nunca un
objetivo configurado.
"""

from __future__ import annotations

from typing import Any

from synthpriv.privacy.mechanisms import PrivacyMechanism


class PrivacyAccountant:
    """Traduce la configuracion de ruido a un epsilon acumulado real."""

    def __init__(self, mechanism: PrivacyMechanism):
        self.mechanism = mechanism
        self._effective_epsilon: float | None = None

    def set_effective_epsilon(self, epsilon: float) -> None:
        """Registra el epsilon medido por el generador DP tras ``fit``."""
        self._effective_epsilon = float(epsilon)

    def get_epsilon(self) -> float | None:
        """Epsilon efectivo (``None`` si no hay garantia formal o no se midio)."""
        if not self.mechanism.is_dp:
            return None
        return self._effective_epsilon

    def report(self) -> dict[str, Any]:
        """Resumen del estado de garantias para el informe."""
        return {
            "mechanism": self.mechanism.name,
            "dp": self.mechanism.is_dp,
            "effective_epsilon": self.get_epsilon(),
            "delta": getattr(self.mechanism, "delta", None),
            "noise_multiplier": getattr(self.mechanism, "used_noise_multiplier", None),
            "explanation": (
                ""
                if self.get_epsilon() is not None
                else "Sin epsilon efectivo medido (mecanismo no DP o sin entrenar)."
            ),
        }