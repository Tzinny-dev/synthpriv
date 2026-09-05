# synthpriv

Generación de datos sintéticos preservando privacidad. Construido sobre
[SDV](https://docs.sdv.dev/), [SDMetrics](https://docs.sdv.dev/sdmetrics) y
[Opacus](https://opacus.ai) para privacidad diferencial (DP-SGD) con accountant RDP.

El proyecto se desarrolla por fases acumulativas, cada una con su batería de tests y su commit.

## Fases implementadas

| Fase | Commit | Qué aporta |
|---|---|---|
| 0 | `ebb3700` | Núcleo: registrador de generadores, base, pipeline, informe HTML, CLI |
| 1 | `ebb3700` | Generadores tabulares sobre SDV (`ctgan`, `tvae`, `copula-gan`, `gaussian-copula`) |
| 2 | `ebb3700` | DP real: `dp-gan` con DP-SGD + accountant RDP y epsilon medido |
| 3 | `6d21828` | Utilidad del `dp-gan`: condicionamiento AC-GAN estilo CTGAN, cobertura de clases minoritarias |
| 4 | `1dc5108` | Barrido `epsilon ↔ utilidad` (curva privacidad/utilidad) |
| 5 | `2c2b05e` | Serialización: `save_model` / `load_model` (regenerar sin reentrenar) |
| 6 | `603445f` | `assert_dp`: validación de integridad de pasos DP y presupuesto ε |
| 7 | `90b485f` | Docs + demo reproducible end-to-end (README, `examples/demo.py`) |
| 8 | `e979fb7` | Benchmark `dp-gan` vs baselines SDV sin DP (curvas de utilidad + gap) |
| 9 | *(esta fase)* | Endurecimiento de utilidad del `dp-gan`: `numeric="uniform"` (gaussianización + cuantil interpolado), `rectify_marginals` (KS garantizado), `label_smoothing`; diagnóstico y límites documentados |

## Instalación

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"   # (usa CUDA si está disponible)
```

Dependencias: Python ≥ 3.10, numpy, pandas, scipy, scikit-learn, SDV < 2, SDMetrics,
anonymeter, Opacus, click, Jinja2.

## Uso rápido

### API Python

```python
from synthpriv import PrivacyPreservingSynthesizer
from synthpriv.privacy.mechanisms import DPSGD

privacy = DPSGD(epsilon=8.0, delta=1e-5)
synth = PrivacyPreservingSynthesizer(
    generator_key="dp-gan",
    generator_kwargs={"epochs": 100, "privacy": privacy, "random_state": 0},
    privacy_mechanism=privacy,
    utility_metrics=["ks_test", "correlation_mae", "ml_utility"],
    privacy_metrics=["nndr", "mia_auc"],
)
synthetic = synth.generate(real_df, num_rows=5000)   # fit + sample
report = synth.evaluate(real_df, synthetic)
report.save("report.html")                          # informe HTML autocontenido

# persiste y regenera sin reentrenar
synth.save_model("demo_model.sz")
loaded = PrivacyPreservingSynthesizer.load_model("demo_model.sz")
loaded.sample(5000).to_csv("resample.csv", index=False)

# valida la garantía DP
assurance = loaded.assert_dp()
print(assurance.status, assurance.message)
```

> Nota: en `generator_kwargs` no pases `random_state` a `PrivacyPreservingSynthesizer`
> cuando éste ya lo recibe en el constructor (conflicto con el del constructor).

### CLI

```bash
# generar con garantía DP
synthpriv generate --data real.csv --epsilon 8 --rows 5000 --save demo_model.sz -o synthetic.csv

# regenerar sin reentrenar (mismo epsilon contabilizado)
synthpriv sample --model demo_model.sz --rows 5000 -o resample.csv

# evaluar utilidad y privacidad
synthpriv evaluate --real real.csv --synthetic synthetic.csv --epsilon 8 -o report.html

# barrido epsilon <-> utilidad
synthpriv sweep --data real.csv --epsilons "0.1,0.5,1,2,5,50" -o sweep_report.html

# benchmark dp-gan vs generadores SDV sin DP (curvas + gap de utilidad)
synthpriv benchmark --data real.csv --epsilons "1,5,50" --baselines gaussian-copula -o bench.html

# auditar que la garantía DP de un modelo persistido no se excede
synthpriv dpcheck --model demo_model.sz --tolerance 0.05
```

### Demo reproducible

```bash
.venv/bin/python examples/demo.py [epsilon] [epochs]
```

Entrena el `dp-gan` sobre un dataset de ejemplo, genera, evalúa, persiste/recarga,
regenera y ejecuta `assert_dp`. Artefactos en `/tmp/synthpriv_demo/`.

## Mecánica de privacidad diferencial

- **Solo el discriminador entrena con DP-SGD** (recorte de gradiente + ruido, muestreo
  de Poisson). El generador es post-proceso del discriminador, por lo que el resultado
  es DP con el epsilon contabilizado por el accountant RDP de Opacus.
- El **epsilon acumulado real** (`accountant.get_epsilon()`) es el que se reporta, no el
  objetivo: dependerá del tamaño de muestra, epochs y ruido resultante.
- `assert_dp` comprueba dos condiciones sobre un modelo (persistido o en memoria):
  1. **Integridad de pasos**: los pasos DP contabilizados por el accountant == los
     ejecutados por el discriminador en el entrenamiento.
  2. **Presupuesto**: epsilon medido ≤ epsilon declarado × (1 + tolerancia).

  Si cualquiera falla, el estado es `fail` y la garantía RDP queda en entredicho.

### Mecanismos

- `NoPrivacy()` — sin garantía formal (solo mitigación empírica + métricas de riesgo).
- `DPSGD(epsilon=1.0, delta=1e-5)` — DP-SGD (Opacus) con el generador `dp-gan`.
  `noise_multiplier`: si se fija se usa ese ruido; si no, Opacus lo calcula para alcanzar
  el presupuesto. Tras entrenar, `used_noise_multiplier` guarda el aplicado.

## Generadores

| Clave | Descripción | DP |
|---|---|---|
| `gaussian-copula` | Cópula gaussiana (rápido, determinista) | no |
| `ctgan` | GAN tabular (SDV) | no |
| `tvae` | Autoencoder variacional tabular (SDV) | no |
| `copula-gan` | GAN con normalización cópula (SDV) | no |
| `dp-gan` | GAN MLP propia con DP-SGD y condicionamiento AC-GAN | sí |

### `dp-gan`

Solo el discriminador ve datos reales y entrena con DP-SGD. Hiperparámetros relevantes:

- `privacy` — mecanismo `DPSGD` con ε/δ objetivo.
- `num_modes` (3) — `ModeEncoder` Gaussian Mixture por columna numérica; `1` = z-score.
- `numeric` (`"mode"`) — codificación de las numéricas: `"mode"` (modo-specific, mix de
  Gauss) o `"uniform"` (rango percentil gaussianizado `Φ⁻¹(rank)` + cuantil empírico).
  `"uniform"` maneja mucho mejor colas y es la opción recomendada cuando el marginal
  importa; el inverso interpola entre cuantiles (no devuelve valores reales exactos).
- `rectify_marginals` (`False`) — solo con `numeric="uniform"`: rectifica al muestrear los
  marginales continuos al ECDF real (transformación monótona por columna, preserva la
  cópula). Garantiza KS ≈ 1 para las numéricas; comparte el trade-off de cuantiles
  empíricos en el marginal (utilidad vs. fuga per-columna) documentado en Limitaciones.
- `condition_column` (`None`) — columna que condiciona la generación; `None` elige la más
  imbalanced (menor entropía) para que las clases minoritarias no colapsen.
- `aux_lambda` (1.0) — peso de las pérdidas auxiliares (clasificador + consistencia de clase).
- `generator_steps` (2) — pasos del generador por paso del discriminador (el presupuesto DP
  solo cuenta el discriminador).
- `label_smoothing` (0.0) — suavizado de etiquetas del discriminador; útil para estabilizar.

Resultados de la fase de endurecimiento (dataset tabular con correlaciones reales,
1500 filas, `numeric="uniform"` + `rectify_marginals`): KS ≈ 1.0 en los tres numéricos
para ε = 1/5/25 y `ml_utility` a la par de la cópula gaussiana de SDV (≈0.35). La cópula
sigue ganando en correlaciones (corrMAE ≈0.02 vs ≈0.2–0.37): la estructura de dependencia
es lo que menos aprende un DP-GAN MLP en datasets pequeños (ver Limitaciones).

## Métricas

- **Utilidad**: KS (distribuciones), MAE de correlaciones, utilidad ML (TSTR).
- **Privacidad**: NNDR (distancia al vecino real más cercano), AUC de ataques de
  inferencia (MIA) y ataques de anonymeter.

## Tests

```bash
.venv/bin/python -m pytest -q              # fast (por defecto)
.venv/bin/python -m pytest -m slow -q      # entrena modelos profundos
```

- `test_registry`, `test_generators`, `test_metrics`, `test_pipeline`, `test_cli`
- `test_dp` (DP end-to-end + cobertura de clases minoritarias)
- `test_sweep` (barrido ε-utilidad)
- `test_benchmark` (benchmark dp-gan vs baselines: estructura, gaps, reportes)
- `test_serialization` (persistencia/recarga)
- `test_assurance` (integridad de pasos DP y presupuesto)

## Limitaciones

- El `dp-gan` es una GAN MLP: para datasets numéricos/categóricos pequeños, no para
  imágenes ni secuencias.
- La garantía DP depende del accountant RDP de Opacus y del muestreo de Poisson; auditoría
  formal con librerías dedicadas queda fuera de alcance.
- Aunque `dp-gan` respeta el presupuesto, la utilidad real a ε bajo depende del dataset
  (ver el barrido `synthpriv sweep`).
- La estructura de dependencia (correlaciones) es el punto débil del `dp-gan`: en datasets
  pequeños aprendida parcialmente y con ruido; ahí las cópulas (SDV) son más precisas.
- Con `numeric="uniform"` el marginal queda ligado a los cuantiles empíricos reales
  (interpolados), lo que da utilidad fuerte pero comparte información per-columna; se
  recomienda con `rectify_marginals=True` solo cuando la utilidad del marginal es prioritaria
  frente a esa consideración.

## Nota de privacidad

Los datos sintéticos **sin DP no son anonimización garantizada**. El informe sugiere
qué nivel de ε usar y qué riesgo empírico se mide, pero la protección formal solo la da
entrenar con un mecanismo DP (`DPSGD` + `dp-gan`) y verificarla con `assert_dp`.