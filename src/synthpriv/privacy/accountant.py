"""Privacy accountant.

In phase 2 the real epsilon is computed by Opacus (RDP) during ``fit`` of the
``dp-gan`` generator and recorded here; the report shows that value, never a
configured target.
"""

from __future__ import annotations

from typing import Any

from synthpriv.privacy.mechanisms import PrivacyMechanism


class PrivacyAccountant:
    """Translates noise configuration into a real accumulated epsilon."""

    def __init__(self, mechanism: PrivacyMechanism):
        self.mechanism = mechanism
        self._effective_epsilon: float | None = None

    def set_effective_epsilon(self, epsilon: float) -> None:
        """Record the epsilon measured by the DP generator after ``fit``."""
        self._effective_epsilon = float(epsilon)

    def get_epsilon(self) -> float | None:
        """Effective epsilon (``None`` if there is no formal guarantee or it was not measured)."""
        if not self.mechanism.is_dp:
            return None
        return self._effective_epsilon

    def report(self) -> dict[str, Any]:
        """Summary of the guarantee state for the report."""
        return {
            "mechanism": self.mechanism.name,
            "dp": self.mechanism.is_dp,
            "effective_epsilon": self.get_epsilon(),
            "delta": getattr(self.mechanism, "delta", None),
            "noise_multiplier": getattr(self.mechanism, "used_noise_multiplier", None),
            "explanation": (
                ""
                if self.get_epsilon() is not None
                else "No effective epsilon measured (non-DP mechanism or not trained)."
            ),
        }