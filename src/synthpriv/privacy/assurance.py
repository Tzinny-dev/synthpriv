"""Aseguramento formal de la privacidad declarada.

``assert_dp`` no inventa garantias: verifica los dos hechos operativos que hacen
valida la cota RDP de Opacus y comprueba que la reivindicacion de privacidad no
excede lo contabilizado.

1. **Integridad de pasos**: cada ``step()`` del optimizador del discriminador
   consume presupuesto DP. Si se ejecutaron mas pasos que los que el accountant
   RDP contabilizo, la garantia queda viciada y ``assert_dp`` falla. (Los pasos
   del generador son solo post-proceso del discriminador DP: no filtran.)
2. **No exceder el presupuesto**: el epsilon medido por el accountant debe quedar
   dentro de ``declared_epsilon * (1 + tolerance)``. La ventana operativa es
   ``[epsilon_medido, presupuesto_declarado]``: cualquier reivindicacion superior
   a lo medido seria tecnicamente defensible, igual o inferior seria inflada.

Si el mecanismo es ``NoPrivacy`` o el generador no aporta epsilon medido, el
resultado es ``fail`` con mensaje explicito: no hay garantia formal que validar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synthpriv.core.base import BaseSynthesizer

STATUS_OK = "ok"
STATUS_FAIL = "fail"


@dataclass
class DpAssurance:
    """Resultado de ``assert_dp``: estado de cada comprobacion y ventana de epsilon."""

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
        """Ventana operativa `(epsilon_medido, presupuesto_declarado)`."""
        return (self.measured_epsilon, self.declared_epsilon)

    def raise_if_not_passed(self) -> "DpAssurance":
        """Lanza ``AssertionError`` si la garantia no se valida (test/reporting)."""
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
    """Valida la garantia DP declarada de un generador entrenado.

    Parameters
    ----------
    generator:
        Generador entrenado (``dp-gan``). Debe exponer ``accounted_epsilon`` y
        los contadores de pasos ``_disc_steps_accounted``/``_disc_steps_actual``.
    declared_epsilon:
        Presupuesto que se reclama (por defecto el del mecanismo del generador).
    tolerance:
        Margen relativo permitido sobre el presupuesto declarado.
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

    # -- 1. garantia formal disponible -----------------------------------
    if not is_dp or measured is None:
        assurance.message = (
            "Sin garantia formal de privacidad: el mecanismo no es DP o el epsilon "
            "acumulado no se midio al entrenar. La privacidad queda solo mitigada "
            "empiricamente (metricas de riesgo del informe)."
        )
        _add_check(assurance, "mecanismo_dp", False,
                   "mecanismo no DP o epsilon sin medir")
        return assurance

    # -- 2. integridad de pasos (solo mecanismos secuenciales tipo DP-SGD) --
    accounted = getattr(generator, "_disc_steps_accounted", None)
    actual = getattr(generator, "_disc_steps_actual", None)
    steps_applicable = getattr(generator, "name", "") == "dp-gan" or (
        accounted is not None and actual is not None)
    if not steps_applicable:
        # dp-copula y similares: garantia por composicion de mecanismos puros
        assurance.steps_match = True
        _add_check(assurance, "pasos_dp", True,
                   "mecanismo DP composicional sin pasos secuenciales (no aplica "
                   "veredicto de pasos Opacus)")
    elif accounted is None or actual is None:
        assurance.steps_match = False
        _add_check(assurance, "pasos_dp", False,
                   "sin contadores de pasos (¿se entreno con esta version?)")
        assurance.message = "No se pudo verificar la integridad de pasos DP."
        return assurance
    else:
        assurance.accounted_steps = accounted
        assurance.actual_private_steps = actual
        assurance.steps_match = actual == accounted
        if actual > accounted:
            _add_check(assurance, "pasos_dp", False,
                       f"{actual} pasos ejecutados > {accounted} contabilizados: "
                       "hay pasos no registrados que filtran datos.")
            assurance.message = (
                f"ALERTA: {actual} pasos del discriminador ejecutados frente a "
                f"{accounted} contabilizados. La garantia RDP queda viciada."
            )
            return assurance
        if actual < accounted:
            _add_check(assurance, "pasos_dp", False,
                       f"{actual} pasos ejecutados < {accounted} contabilizados: "
                       "se declara mas privacidad de la realmente consumida.")
            assurance.message = (
                f"Los pasos ejecutados ({actual}) difieren de los contabilizados "
                f"({accounted}); la cota no es exacta."
            )
            return assurance
        _add_check(assurance, "pasos_dp", True, f"{accounted} pasos todos contabilizados")

    # -- 3. presupuesto respetado -------------------------------------------
    if declared is None:
        assurance.budget_respected = True  # sin presupuesto declarado no hay exceso
        _add_check(assurance, "presupuesto", True, "sin presupuesto declarado")
    else:
        allowance = declared * (1.0 + tolerance)
        assurance.budget_respected = measured <= allowance
        _add_check(
            assurance,
            "presupuesto",
            assurance.budget_respected,
            f"epsilon medido {measured:.4f} <= {allowance:.4f} "
            f"(declarado {declared} + {tolerance:.0%})",
        )

    if assurance.steps_match and assurance.budget_respected:
        assurance.status = STATUS_OK
        _noise = assurance.noise_multiplier if assurance.noise_multiplier is not None else float("nan")
        note = ""
        ecdf = getattr(generator, "ecdf_epsilon", None)
        if ecdf is not None:
            note = (f" (valida el DP-SGD del entrenamiento; total del sintetizador = "
                    f"{measured:.3f} + {ecdf} = {measured + float(ecdf):.3f} con DP-ECDF, "
                    f"ver informe)")
        if steps_applicable:
            _pasos = f"{accounted} pasos contabilizados y ejecutados, ruido {_noise:.3f}"
        else:
            _pasos = "garantia por composicion de mecanismos puros (ver componente epsilons en informe)"
        assurance.message = (
            f"Garantia DP validada: epsilon RDP {measured:.3f} "
            f"(ventana [{measured:.3f}, {declared}]), {_pasos}.{note}"
        )
    else:
        assurance.message = (
            f"Reinvindicacion de privacidad no validada: pasos "
            f"{'OK' if assurance.steps_match else 'NO'}, presupuesto "
            f"{'OK' if assurance.budget_respected else 'NO'}."
        )
    return assurance