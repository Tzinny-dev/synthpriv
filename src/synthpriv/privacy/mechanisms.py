"""Mecanismos de privacidad diferencial (DP).

Fase actual (0.1): solo ``NoPrivacy`` esta operativo. ``DPSGD`` queda como
configuracion declarativa; la integracion real con Opacus llegara en la fase 2.

Regla de oro: si ``is_dp`` es ``True`` pero ``available`` es ``False``, el
pipeline se niega a proseguir o avisa claramente para no vender garantias que
el codigo todavia no entrega.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrivacyMechanism:
    """Base de todo mecanismo de privacidad."""

    name: str = "base"
    is_dp: bool = False
    available: bool = True
    notes: str = ""

    def get_report(self) -> dict[str, Any]:
        """Dict descriptivo para incluir en el informe de evaluacion."""
        return {
            "mechanism": self.name,
            "dp": self.is_dp,
            "available": self.available,
            "epsilon": getattr(self, "epsilon", None),
            "delta": getattr(self, "delta", None),
            "notes": self.notes,
        }


@dataclass
class NoPrivacy(PrivacyMechanism):
    """Sin mecanismo formal de DP.

    La privacidad queda mitigada empiricamente (calidad del generador + metricas
    de riesgo). Warning de jaque permanente: no da garantia formal alguna.
    """

    name: str = "no-privacy"
    is_dp: bool = False
    notes: str = (
        "Sin garantia formal de privacidad diferencial. El riesgo de re-identificacion "
        "debe evaluarse con las metricas del informe."
    )


@dataclass
class DPSGD(PrivacyMechanism):
    """Entrenamiento con DP-SGD (Opacus).

    Tras ``fit``, ``used_noise_multiplier`` guarda el ruido aplicado y el
    accountant RDP devuelve el epsilon *acumulado real* (<= objetivo si el tamano
    de muestra lo permite). Usa el generador ``dp-gan``.
    """

    name: str = "dp-sgd"
    is_dp: bool = True
    available: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float | None = None
    max_grad_norm: float = 1.0
    used_noise_multiplier: float | None = field(default=None, init=False)
    notes: str = (
        "DP-SGD con accountant RDP de Opacus. El epsilon del informe es el acumulado "
        "real tras entrenar, no el objetivo configurado. Exige el generador 'dp-gan'."
    )