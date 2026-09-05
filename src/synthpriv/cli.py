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
from synthpriv.benchmark import run_benchmark
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
@click.option("--ecdf-epsilon", default=None, type=float,
              help="Presupuesto DP extra para las ECDF de marginales del dp-gan "
                   "(solo con DP; requiere numericas 'uniform').")
@click.option("--output", "-o", "output", default="synthetic.csv", type=click.Path(dir_okay=False),
              help="CSV de salida.")
@click.option("--save", "save_model", default=None, type=click.Path(dir_okay=False),
              help="Persistir el sintetizador entrenado en esta ruta (para 'synthpriv sample').")
def generate(data, method, rows, epochs, epsilon, ecdf_epsilon, output, save_model):
    """Entrena un generador y produce datos sinteticos."""
    if ecdf_epsilon is not None and epsilon is None:
        raise click.ClickException("--ecdf-epsilon requiere --epsilon (modo DP con dp-gan).")
    df = pd.read_csv(data)
    logger.info("Datos reales: %d filas x %d columnas", *df.shape)

    if epsilon is not None:
        privacy = DPSGD(epsilon=epsilon)
        generator_kwargs: dict = {"privacy": privacy}
        if method == "dp-copula":
            logger.info("Modo DP activo: copula gaussiana privada (dp-copula) "
                        "con epsilon total %.3f", epsilon)
            if ecdf_epsilon is not None:
                raise click.ClickException(
                    "--ecdf-epsilon es del dp-gan; en dp-copula el presupuesto "
                    "se divide en marginales/copula/explicita.")
        elif method == "dp-gan":
            generator_kwargs.update(epochs=epochs)
            if ecdf_epsilon is not None:
                generator_kwargs.update(numeric="uniform", rectify_marginals=True,
                                        ecdf_epsilon=ecdf_epsilon)
                logger.info("  + DP-ECDF de marginales: epsilon total = %.3f + %.3f",
                            epsilon, ecdf_epsilon)
            logger.info("Modo DP activo: generador 'dp-gan' con epsilon objetivo %.3f", epsilon)
        else:
            fallback = method
            method = "dp-gan"  # solo los generadores DP-capable entrenan con privacidad
            generator_kwargs.update(epochs=epochs)
            logger.info("Modo DP activo: generador 'dp-gan' (el %r no es DP-capable)",
                        fallback)
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


@cli.command()
@click.option("--data", "-d", "data", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV con los datos reales.")
@click.option("--epsilons", "-e", "epsilons", default="1,2,5,10,50", show_default=True,
              help="Presupuestos DP del dp-gan separados por comas.")
@click.option("--delta", default=1e-5, type=float, show_default=True, help="Delta DP.")
@click.option("--baselines", "-b", "baselines", default="gaussian-copula", show_default=True,
              help="Generadores SDV sin DP separados por comas; usa 'all' para todos.")
@click.option("--epochs", default=100, type=int, show_default=True, help="Epochs del dp-gan.")
@click.option("--numeric", default="mode", type=click.Choice(["mode", "uniform"]), show_default=True,
              help="Codificacion numerica del dp-gan: 'mode' (GMM) o 'uniform' (gaussianizada).")
@click.option("--rectify-marginals", "rectify_marginals", is_flag=True,
              help="Rectificar marginales al muestrear (requiere --numeric uniform).")
@click.option("--ecdf-epsilon", default=None, type=float,
              help="Presupuesto DP-ECDF de marginales (requiere --numeric uniform). "
                   "El total informado por punto es entrenamiento + este valor.")
@click.option("--baseline-epochs", default=0, type=int, show_default=True,
              help="Epochs de los baselines SDV (0 = el default de cada generador).")
@click.option("--rows", "-n", "rows", default=None, type=int,
              help="Filas sinteticas por punto (default: mismas que reales).")
@click.option("--output", "-o", "output", default="benchmark_report.html", type=click.Path(dir_okay=False),
              help="Informe HTML de salida.")
def benchmark(data, epsilons, delta, baselines, epochs, numeric, rectify_marginals,
              ecdf_epsilon, baseline_epochs, rows, output):
    """Compara dp-gan (varios epsilon) frente a generadores SDV sin DP."""
    if ecdf_epsilon is not None and numeric != "uniform":
        raise click.ClickException("--ecdf-epsilon requiere --numeric uniform")
    if rectify_marginals and numeric != "uniform":
        raise click.ClickException("--rectify-marginals requiere --numeric uniform")
    df = pd.read_csv(data)
    eps = tuple(float(e.strip()) for e in epsilons.split(",") if e.strip())
    bl = ["gaussian-copula", "ctgan", "tvae", "copula-gan"] if baselines.strip() == "all" \
        else tuple(b.strip() for b in baselines.split(",") if b.strip())
    baseline_kwargs = {b: {"epochs": baseline_epochs} for b in bl if baseline_epochs > 0}

    result = run_benchmark(
        df,
        epsilons=eps,
        delta=delta,
        baselines=bl,
        generator_kwargs={"epochs": epochs, "batch_size": 128,
                          "numeric": numeric, "rectify_marginals": rectify_marginals,
                          "ecdf_epsilon": ecdf_epsilon},
        baseline_kwargs=baseline_kwargs,
        num_rows=rows,
    )
    result.to_csv(str(Path(output).with_suffix(".csv")))
    result.save_report(output)
    for r in result.dataframe().to_dict("records"):
        extra = f" | eps {r['target_epsilon']:.1f} -> medido {r['measured_epsilon']}" \
            if r.get("measured_epsilon") is not None else " (baseline, sin DP)"
        click.echo("{:<16} {}{}".format(r["model"], extra,
                                        {k: v for k, v in r.items() if k.startswith("util_")}))
    click.echo(f"Informe guardado en {output}")


@cli.command()
@click.option("--model", "-m", "model", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Sintetizador guardado con 'synthpriv generate --save'.")
@click.option("--tolerance", default=0.05, type=float, show_default=True,
              help="Margen relativo permitido sobre el presupuesto declarado.")
def dpcheck(model, tolerance):
    """Valida que la garantia DP de un modelo persistido no se excede."""
    synthesizer = PrivacyPreservingSynthesizer.load_model(model)
    assurance = synthesizer.assert_dp(tolerance=tolerance)
    click.echo(f"estado: {assurance.status}")
    click.echo(f"ventana [epsilon medido, presupuesto]: {assurance.window}")
    click.echo(f"pasos contabilizados/ejecutados: {assurance.accounted_steps}/{assurance.actual_private_steps} "
               f"({'OK' if assurance.steps_match else 'NO'})")
    click.echo(f"ruido: {assurance.noise_multiplier} | delta: {assurance.delta}")
    for check in assurance.checks:
        click.echo(f"  [{'OK' if check['status'] == 'ok' else 'X'}] {check['name']}: {check['detail']}")
    click.echo(assurance.message)


if __name__ == "__main__":  # pragma: no cover
    cli()