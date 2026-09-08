"""Differential privacy (DP) mechanisms.

Current phase (0.1): only ``NoPrivacy`` is operational. ``DPSGD`` is a
declarative configuration; the real Opacus integration arrives in phase 2.

Golden rule: if ``is_dp`` is ``True`` but ``available`` is ``False``, the
pipeline refuses to proceed or warns clearly so that no guarantees are
claimed that the code does not yet deliver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrivacyMechanism:
    """Base for every privacy mechanism."""

    name: str = "base"
    is_dp: bool = False
    available: bool = True
    notes: str = ""

    def get_report(self) -> dict[str, Any]:
        """Descriptive dict to include in the evaluation report."""
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
    """No formal DP mechanism.

    Privacy is only mitigated empirically (generator quality + risk metrics).
    Permanent check warning: it gives no formal guarantee at all.
    """

    name: str = "no-privacy"
    is_dp: bool = False
    notes: str = (
        "No formal differential privacy guarantee. Re-identification risk must be "
        "assessed with the report metrics."
    )


@dataclass
class DPSGD(PrivacyMechanism):
    """Training with DP-SGD (Opacus).

    After ``fit``, ``used_noise_multiplier`` holds the applied noise and the
    RDP accountant returns the *real accumulated* epsilon (<= target when the
    sample size allows it). Uses the ``dp-gan`` generator.
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
        "DP-SGD with Opacus's RDP accountant. The report epsilon is the real "
        "accumulated one after training, not the configured target. Requires the "
        "'dp-gan' generator."
    )