"""Formal assurance of the declared privacy.

``assert_dp`` does not invent guarantees: it verifies the two operational facts
that make Opacus's RDP bound valid and checks that the privacy claim does not
exceed what was accounted.

1. **Step integrity**: each ``step()`` of the discriminator optimizer consumes DP
   budget. If more steps ran than the RDP accountant accounted, the guarantee is
   void and ``assert_dp`` fails. (Generator steps are only post-processing of the
   DP discriminator: they do not leak.)
2. **Budget not exceeded**: the epsilon measured by the accountant must stay
   within ``declared_epsilon * (1 + tolerance)``. The operational window is
   ``[measured_epsilon, declared_budget]``: any claim above what was measured
   would be technically defensible, equal or below would be inflated.

If the mechanism is ``NoPrivacy`` or the generator provides no measured epsilon,
the result is ``fail`` with an explicit message: there is no formal guarantee to
validate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synthpriv.core.base import BaseSynthesizer

STATUS_OK = "ok"
STATUS_FAIL = "fail"


@dataclass
class DpAssurance:
    """Result of ``assert_dp``: state of each check and the epsilon window."""

    status: str = STATUS_FAIL
    declared_epsilon: float | None = None
    measured_epsilon: float | None = None
    delta: float | None = None
    noise_multiplier: float | None = None
    max_grad_norm: float | None = None
    accounted_steps: int | None = None
    actual_private_steps: int | None = None
    steps_match: bool = False
    budget_respected: bool = False
    checks: list[dict[str, str]] = field(default_factory=list)
    message: str = ""

    # ------------------------------------------------------------------
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "declared_epsilon": self.declared_epsilon,
            "measured_epsilon": self.measured_epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "accounted_steps": self.accounted_steps,
            "actual_private_steps": self.actual_private_steps,
            "steps_match": self.steps_match,
            "budget_respected": self.budget_respected,
            "window": self.window,
            "checks": self.checks,
            "message": self.message,
        }

    @property
    def window(self) -> tuple[float | None, float | None]:
        """Operational window `(measured_epsilon, declared_budget)`."""
        return (self.measured_epsilon, self.declared_epsilon)

    def raise_if_not_passed(self) -> "DpAssurance":
        """Raise ``AssertionError`` if the guarantee is not validated (test/reporting)."""
        if not self.ok():
            raise AssertionError(self.message)
        return self


def _add_check(assurance: DpAssurance, name: str, ok: bool, detail: str) -> None:
    assurance.checks.append({"name": name, "status": STATUS_OK if ok else STATUS_FAIL,
                             "detail": detail})


def assert_dp(
    generator: BaseSynthesizer,
    declared_epsilon: float | None = None,
    *,
    tolerance: float = 0.05,
    delta: float | None = None,
) -> DpAssurance:
    """Validate the declared DP guarantee of a trained generator.

    Parameters
    ----------
    generator:
        Trained generator (``dp-gan``). Must expose ``accounted_epsilon`` and the
        step counters ``_disc_steps_accounted``/``_disc_steps_actual``.
    declared_epsilon:
        Claimed budget (defaults to the generator mechanism's).
    tolerance:
        Relative margin allowed over the declared budget.
    """
    assurance = DpAssurance()

    mechanism = getattr(generator, "privacy", None)
    declared = declared_epsilon if declared_epsilon is not None else getattr(mechanism, "epsilon", None)
    measured = getattr(generator, "accounted_epsilon", None)
    is_dp = bool(getattr(mechanism, "is_dp", False)) and getattr(generator, "dp_capable", False)

    assurance.declared_epsilon = declared
    assurance.measured_epsilon = measured
    assurance.delta = delta if delta is not None else getattr(mechanism, "delta", None)
    assurance.noise_multiplier = getattr(mechanism, "used_noise_multiplier", None)
    assurance.max_grad_norm = getattr(mechanism, "max_grad_norm", None)

    # -- 1. formal guarantee available ------------------------------------
    if not is_dp or measured is None:
        assurance.message = (
            "No formal privacy guarantee: the mechanism is not DP or the accumulated "
            "epsilon was not measured during training. Privacy is only mitigated "
            "empirically (risk metrics of the report)."
        )
        _add_check(assurance, "dp_mechanism", False,
                   "mechanism not DP or epsilon unmeasured")
        return assurance

    # -- 2. step integrity (only sequential mechanisms like DP-SGD) --------
    accounted = getattr(generator, "_disc_steps_accounted", None)
    actual = getattr(generator, "_disc_steps_actual", None)
    steps_applicable = getattr(generator, "name", "") == "dp-gan" or (
        accounted is not None and actual is not None)
    if not steps_applicable:
        # dp-copula and similar: guarantee by composition of pure mechanisms
        assurance.steps_match = True
        _add_check(assurance, "dp_steps", True,
                   "compositional DP mechanism without sequential steps (Opacus step "
                   "verdict does not apply)")
    elif accounted is None or actual is None:
        assurance.steps_match = False
        _add_check(assurance, "dp_steps", False,
                   "no step counters (was it trained with this version?)")
        assurance.message = "Could not verify DP step integrity."
        return assurance
    else:
        assurance.accounted_steps = accounted
        assurance.actual_private_steps = actual
        assurance.steps_match = actual == accounted
        if actual > accounted:
            _add_check(assurance, "dp_steps", False,
                       f"{actual} steps executed > {accounted} accounted: "
                       "there are unregistered steps that leak data.")
            assurance.message = (
                f"ALERT: {actual} discriminator steps executed vs "
                f"{accounted} accounted. The RDP guarantee is void."
            )
            return assurance
        if actual < accounted:
            _add_check(assurance, "dp_steps", False,
                       f"{actual} steps executed < {accounted} accounted: "
                       "more privacy is declared than actually consumed.")
            assurance.message = (
                f"The executed steps ({actual}) differ from the accounted ones "
                f"({accounted}); the bound is not exact."
            )
            return assurance
        _add_check(assurance, "dp_steps", True, f"{accounted} steps all accounted")

    # -- 3. budget respected ----------------------------------------------
    if declared is None:
        assurance.budget_respected = True  # no declared budget, no excess
        _add_check(assurance, "budget", True, "no declared budget")
    else:
        allowance = declared * (1.0 + tolerance)
        assurance.budget_respected = measured <= allowance
        _add_check(
            assurance,
            "budget",
            assurance.budget_respected,
            f"measured epsilon {measured:.4f} <= {allowance:.4f} "
            f"(declared {declared} + {tolerance:.0%})",
        )

    if assurance.steps_match and assurance.budget_respected:
        assurance.status = STATUS_OK
        _noise = assurance.noise_multiplier if assurance.noise_multiplier is not None else float("nan")
        note = ""
        ecdf = getattr(generator, "ecdf_epsilon", None)
        if ecdf is not None:
            note = (f" (validates the training DP-SGD; synthesizer total = "
                    f"{measured:.3f} + {ecdf} = {measured + float(ecdf):.3f} with DP-ECDF, "
                    f"see report)")
        if steps_applicable:
            _pasos = f"{accounted} accounted and executed steps, noise {_noise:.3f}"
        else:
            _pasos = "guarantee by composition of pure mechanisms (see component epsilons in report)"
        assurance.message = (
            f"DP guarantee validated: RDP epsilon {measured:.3f} "
            f"(window [{measured:.3f}, {declared}]), {_pasos}.{note}"
        )
    else:
        assurance.message = (
            f"Privacy claim not validated: steps "
            f"{'OK' if assurance.steps_match else 'NO'}, budget "
            f"{'OK' if assurance.budget_respected else 'NO'}."
        )
    return assurance