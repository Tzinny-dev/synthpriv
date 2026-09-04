"""Interfaz de linea de comandos de synthpriv.

Usage:
    synthpriv generate --data real.csv -m ctgan --rows 5000 -o synthetic.csv
    synthpriv evaluate --real real.csv --synthetic synthetic.csv -o report.html
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import click

from synthpriv.core.registry import list_generators
from synthpriv.pipeline import PrivacyPreservingSynthesizer
from synthpriv.privacy.mechanisms import DPSGD, NoPrivacy
from synthpriv.sweep import run_epsilon_sweep
from synthpriv.utils import get_logger

logger = get_logger("cli")


@click.group()
def cli():
    """Generacion de datos sinteticos preservando privacidad."""


@cli.command()
@click.option("--data", "-d", "data", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Ruta al CSV con los datos reales.")
@click.option("--method", "-m", "method", default="ctgan", type=click.Choice(list_generators()),
              help="Generador a utilizar.")
@click.option("--rows", "-n", "rows", default=5000, type=int, show_default=True,
              help="Numero de filas sinteticas.")
@click.option("--epochs", default=300, type=int, show_default=True, help="Epochs de entrenamiento.")
@click.option("--epsilon", default=None, type=float,
              help="Privacidad diferencial: entrena 'dp-gan' con DP-SGD y este presupuesto.")
@click.option("--output", "-o", "output", default="synthetic.csv", type=click.Path(dir_okay=False),
              help="CSV de salida.")
@click.option("--save", "save_model", default=None, type=click.Path(dir_okay=False),
              help="Persistir el sintetizador entrenado en esta ruta (para 'synthpriv sample').")
def generate(data, method, rows, epochs, epsilon, output, save_model):
    """Entrena un generador y produce datos sinteticos."""
    df = pd.read_csv(data)
    logger.info("Datos reales: %d filas x %d columnas", *df.shape)

    if epsilon is not None:
        privacy = DPSGD(epsilon=epsilon)
        method = "dp-gan"
        generator_kwargs: dict = {"epochs": epochs, "privacy": privacy}
        logger.info("Modo DP activo: generador 'dp-gan' con epsilon objetivo %.3f", epsilon)
    else:
        privacy = NoPrivacy()
        generator_kwargs = {"epochs": epochs}

    synthesizer = PrivacyPreservingSynthesizer(
        generator_key=method,
        generator_kwargs=generator_kwargs,
        privacy_mechanism=privacy,
    )
    synthetic = synthesizer.generate(df, num_rows=rows)
    synthetic.to_csv(output, index=False)
    if save_model:
        model_path = synthesizer.save_model(save_model)
        logger.info("Sintetizador guardado en %s", model_path)
    eps = synthesizer.accountant.get_epsilon()
    extra = f" | epsilon acumulado real %.3f" % eps if eps is not None else ""
    logger.info("Sinteticos generados en %s (%.2fs de entrenamiento)%s", output,
                synthesizer.timings.get("fit_seconds", 0.0), extra)


@cli.command()
@click.option("--model", "-m", "model", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Sintetizador guardado con 'synthpriv generate --save'.")
@click.option("--rows", "-n", "rows", default=1000, type=int, show_default=True,
              help="Numero de filas sinteticas.")
@click.option("--output", "-o", "output", default="synthetic.csv", type=click.Path(dir_okay=False),
              help="CSV de salida.")
def sample(model, rows, output):
    """Genera filas desde un sintetizador persistido sin reentrenar.

    Entrenar con DP cuesta una vez; recuperando el modelo se regenera el dataset
    con la misma garantia de privacidad (accounted epsilon) en segundos.
    """
    synthesizer = PrivacyPreservingSynthesizer.load_model(model)
    out = synthesizer.sample(rows)
    out.to_csv(output, index=False)
    eps = synthesizer.accountant.get_epsilon()
    extra = f" | epsilon acumulado real %.3f" % eps if eps is not None else ""
    logger.info("Muestreadas %d filas desde %s -> %s%s",
                rows, model, output, extra)


@cli.command()
@click.option("--real", "-r", "real", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV con los datos reales.")
@click.option("--synthetic", "-s", "synthetic", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV con los datos sinteticos.")
@click.option("--epsilon", default=None, type=float,
              help="Presupuesto DP que se uso al generar (para etiquetar el informe).")
@click.option("--delta", default=1e-5, type=float, show_default=True, help="Delta DP asociado.")
@click.option("--output", "-o", "output", default="report.html", type=click.Path(dir_okay=False),
              help="Informe HTML de salida.")
def evaluate(real, synthetic, epsilon, delta, output):
    """Evalua utilidad y privacidad de unos datos sinteticos.""" 
    real_df, synth_df = pd.read_csv(real), pd.read_csv(synthetic)

    if epsilon is not None:
        privacy = DPSGD(epsilon=epsilon, delta=delta)
        generator_key = "dp-gan"  # solo para pasar la validacion DP-capable; no se entrena
    else:
        privacy = NoPrivacy()
        generator_key = "gaussian-copula"
    synthesizer = PrivacyPreservingSynthesizer(
        generator_key=generator_key,
        privacy_mechanism=privacy,
    )
    report = synthesizer.evaluate(real_df, synth_df)
    report.save(output)
    logger.info("Informe generado en %s", output)
    click.echo(report.summary())


@cli.command()
@click.option("--data", "-r", "data", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV con los datos reales.")
@click.option("--epsilons", "-e", "epsilons", default="0.1,0.5,1,2,5,50", show_default=True,
              help="Presupuestos DP separados por comas (50 ~ casi sin DP).")
@click.option("--delta", default=1e-5, type=float, show_default=True, help="Delta DP.")
@click.option("--epochs", default=50, type=int, show_default=True, help="Epochs por punto.")
@click.option("--rows", "-n", "rows", default=2000, type=int, show_default=True,
              help="Filas sinteticas por punto.")
@click.option("--output", "-o", "output", default="sweep_report.html", type=click.Path(dir_okay=False),
              help="Informe HTML de salida.")
def sweep(data, epsilons, delta, epochs, rows, output):
    """Entrena dp-gan con varios epsilon y dibuja la curva privacidad/utilidad."""
    df = pd.read_csv(data)
    eps = tuple(float(e.strip()) for e in epsilons.split(",") if e.strip())
    result = run_epsilon_sweep(
        df,
        epsilons=eps,
        delta=delta,
        generator_kwargs={"epochs": epochs, "batch_size": 256},
        num_rows=rows,
    )
    result.to_csv(str(Path(output).with_suffix(".csv")))
    result.save_report(output)
    for r in result.rows:
        click.echo("eps objetivo {:>6} | medido {:>8} | {}".format(
            r["target_epsilon"], r["measured_epsilon"],
            {k: v for k, v in r.items() if k.startswith("util_")}))
    click.echo(f"Informe guardado en {output}")


if __name__ == "__main__":  # pragma: no cover
    cli()