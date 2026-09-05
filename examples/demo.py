"""Demo end-to-end de synthpriv: pipeline DP, utilidad/privacidad y persistencia.

Recorrido:
1. Construye un dataset tabular sintetico de partida (mezcla numerico + categorico,
   con una columna desbalanceada que hace de condicion del dp-gan).
2. Entrena el ``dp-gan`` con un presupuesto DP (epsilon, delta) y genera filas.
3. Evalua utilidad y privacidad, y guarda el informe HTML autocontenido.
4. Persiste el sintetizador, lo recarga y regenera sin reentrenar.
5. Ejecuta ``assert_dp`` para validar que la garantia DP no se excede.

Uso (desde la raiz del repo, con el entorno activo):
    .venv/bin/python examples/demo.py [epsilon] [epochs] [ecdf_epsilon]

Salida: /tmp/synthpriv_demo/ con
    - synthetic.csv          (filas generadas)
    - report.html            (informe de utilidad y privacidad)
    - demo_model.sz          (sintetizador persistido) + demo_model.sz.meta
    - resample.csv           (regenerado desde el modelo recargado, sin reentrenar)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from synthpriv import DpAssurance, PrivacyPreservingSynthesizer
from synthpriv.privacy.mechanisms import DPSGD

OUT = Path("/tmp/synthpriv_demo")


def make_real_data(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Dataset de partida determinista: numericas con modos y categorias."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.normal(45, 12, n).clip(18, 90).round(),
        "income": rng.gamma(3.0, 5000.0, n),
        "score": rng.normal(0.5, 0.2, n).clip(0, 1),
        "education": rng.choice(
            ["high_school", "bachelor", "master", "phd"], n,
            p=[0.35, 0.4, 0.15, 0.1]),
        "is_fraud": rng.choice(["no", "yes"], n, p=[0.9, 0.1]),
    })


def main() -> int:
    epsilon = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ecdf_eps = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    OUT.mkdir(parents=True, exist_ok=True)

    real = make_real_data()
    print(f"=== Dataset real: {real.shape} ===")
    print(real.head(3).to_string(index=False))

    privacy = DPSGD(epsilon=epsilon, delta=1e-5)
    synth = PrivacyPreservingSynthesizer(
        generator_key="dp-gan",
        generator_kwargs={
            "epochs": epochs,
            "batch_size": 128,
            "latent_dim": 32,
            "hidden_dim": 128,
            "numeric": "uniform",
            "rectify_marginals": True,
            "ecdf_epsilon": ecdf_eps,
            "privacy": privacy,
        },
        privacy_mechanism=privacy,
        utility_metrics=["ks_test", "correlation_mae", "ml_utility"],
        privacy_metrics=["nndr", "mia_auc"],
        random_state=0,
    )

    print(f"\n=== Entrenando dp-gan: epsilon objetivo {epsilon} (DP-SGD) + "
          f"{ecdf_eps} (DP-ECDF) ===")
    print("    codificacion numerica 'uniform' + DP-ECDF de marginales con"
          " composicion secuencial")
    synthetic = synth.generate(real, num_rows=len(real))
    synthetic.to_csv(OUT / "synthetic.csv", index=False)

    eps_measured = synth.accountant.get_epsilon()
    print(f"epsilon acumulado real (RDP): {eps_measured:.4f}")
    print(f"epsilon TOTAL (DP-SGD + DP-ECDF): {eps_measured + ecdf_eps:.4f}")

    print("\n=== Evaluando utilidad y privacidad ===")
    report = synth.evaluate(real, synthetic)
    report.save(OUT / "report.html")
    print(report.summary())

    print("\n=== Persistiendo el sintetizador sin reentrenar ===")
    model_path = synth.save_model(OUT / "demo_model.sz")

    print("\n=== Recargando y regenerando ===")
    loaded = PrivacyPreservingSynthesizer.load_model(model_path)
    resample = loaded.sample(len(real))
    resample.to_csv(OUT / "resample.csv", index=False)
    print(f"regeneradas {len(resample)} filas, epsilon registrado "
          f"{loaded.accountant.get_epsilon():.4f}")

    print("\n=== Validando la garantia DP (assert_dp) ===")
    assurance: DpAssurance = loaded.assert_dp()
    print(f"estado: {assurance.status}")
    print(f"ventana [medido, presupuesto]: {assurance.window}")
    print(f"pasos contabilizados/ejecutados: {assurance.accounted_steps}/{assurance.actual_private_steps} "
          f"({'OK' if assurance.steps_match else 'NO'})")
    print(assurance.message)

    print(f"\n=== Artefactos en {OUT} ===")
    for p in sorted(OUT.iterdir()):
        print(f"  - {p.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())