"""Interfaz de linea de comandos de synthpriv.

Usage:
    synthpriv generate --data real.csv -m ctgan --rows 5000 -o synthetic.csv
    synthpriv evaluate --real real.csv --synthetic synthetic.csv -o report.html
"""

from __future__ import annotations

import pandas as pd
import click

from synthpriv.core.registry import list_generators
from synthpriv.pipeline import PrivacyPreservingSynthesizer
from synthpriv.privacy.mechanisms import DPSGD, NoPrivacy
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
def generate(data, method, rows, epochs, epsilon, output):
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
    eps = synthesizer.accountant.get_epsilon()
    extra = f" | epsilon acumulado real %.3f" % eps if eps is not None else ""
    logger.info("Sinteticos generados en %s (%.2fs de entrenamiento)%s", output,
                synthesizer.timings.get("fit_seconds", 0.0), extra)


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


if __name__ == "__main__":  # pragma: no cover
    cli()