"""synthpriv command-line interface.

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
    """Synthetic data generation preserving privacy."""


@cli.command()
@click.option("--data", "-d", "data", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to the CSV with the real data.")
@click.option("--method", "-m", "method", default="ctgan", type=click.Choice(list_generators()),
              help="Generator to use.")
@click.option("--rows", "-n", "rows", default=5000, type=int, show_default=True,
              help="Number of synthetic rows.")
@click.option("--epochs", default=300, type=int, show_default=True, help="Training epochs.")
@click.option("--epsilon", default=None, type=float,
              help="Differential privacy: trains 'dp-gan' with DP-SGD and this budget.")
@click.option("--ecdf-epsilon", default=None, type=float,
              help="Extra DP budget for the dp-gan marginal ECDFs "
                   "(DP only; requires 'uniform' numerics).")
@click.option("--output", "-o", "output", default="synthetic.csv", type=click.Path(dir_okay=False),
              help="Output CSV.")
@click.option("--save", "save_model", default=None, type=click.Path(dir_okay=False),
              help="Persist the trained synthesizer at this path (for 'synthpriv sample').")
def generate(data, method, rows, epochs, epsilon, ecdf_epsilon, output, save_model):
    """Train a generator and produce synthetic data."""
    if ecdf_epsilon is not None and epsilon is None:
        raise click.ClickException("--ecdf-epsilon requires --epsilon (DP mode with dp-gan).")
    df = pd.read_csv(data)
    logger.info("Real data: %d rows x %d columns", *df.shape)

    if epsilon is not None:
        privacy = DPSGD(epsilon=epsilon)
        generator_kwargs: dict = {"privacy": privacy}
        if method == "dp-copula":
            logger.info("DP mode active: private Gaussian copula (dp-copula) "
                        "with total epsilon %.3f", epsilon)
            if ecdf_epsilon is not None:
                raise click.ClickException(
                    "--ecdf-epsilon belongs to dp-gan; in dp-copula the budget "
                    "is split into marginals/copula/categorical.")
        elif method == "dp-gan":
            generator_kwargs.update(epochs=epochs)
            if ecdf_epsilon is not None:
                generator_kwargs.update(numeric="uniform", rectify_marginals=True,
                                        ecdf_epsilon=ecdf_epsilon)
                logger.info("  + marginal DP-ECDF: total epsilon = %.3f + %.3f",
                            epsilon, ecdf_epsilon)
            logger.info("DP mode active: generator 'dp-gan' with target epsilon %.3f", epsilon)
        else:
            fallback = method
            method = "dp-gan"  # only DP-capable generators train with privacy
            generator_kwargs.update(epochs=epochs)
            logger.info("DP mode active: generator 'dp-gan' (%r is not DP-capable)",
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
        logger.info("Synthesizer saved at %s", model_path)
    eps = synthesizer.accountant.get_epsilon()
    extra = f" | real accumulated epsilon %.3f" % eps if eps is not None else ""
    logger.info("Synthetic data generated at %s (%.2fs training)%s", output,
                synthesizer.timings.get("fit_seconds", 0.0), extra)


@cli.command()
@click.option("--model", "-m", "model", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Synthesizer saved with 'synthpriv generate --save'.")
@click.option("--rows", "-n", "rows", default=1000, type=int, show_default=True,
              help="Number of synthetic rows.")
@click.option("--output", "-o", "output", default="synthetic.csv", type=click.Path(dir_okay=False),
              help="Output CSV.")
def sample(model, rows, output):
    """Generate rows from a persisted synthesizer without retraining.

    Training with DP costs once; reloading the model regenerates the dataset
    with the same privacy guarantee (accounted epsilon) in seconds.
    """
    synthesizer = PrivacyPreservingSynthesizer.load_model(model)
    out = synthesizer.sample(rows)
    out.to_csv(output, index=False)
    eps = synthesizer.accountant.get_epsilon()
    extra = f" | real accumulated epsilon %.3f" % eps if eps is not None else ""
    logger.info("Sampled %d rows from %s -> %s%s",
                rows, model, output, extra)


@cli.command()
@click.option("--real", "-r", "real", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV with the real data.")
@click.option("--synthetic", "-s", "synthetic", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV with the synthetic data.")
@click.option("--epsilon", default=None, type=float,
              help="DP budget used when generating (to label the report).")
@click.option("--delta", default=1e-5, type=float, show_default=True, help="Associated DP delta.")
@click.option("--output", "-o", "output", default="report.html", type=click.Path(dir_okay=False),
              help="Output HTML report.")
def evaluate(real, synthetic, epsilon, delta, output):
    """Evaluate utility and privacy of some synthetic data."""
    real_df, synth_df = pd.read_csv(real), pd.read_csv(synthetic)

    if epsilon is not None:
        privacy = DPSGD(epsilon=epsilon, delta=delta)
        generator_key = "dp-gan"  # only to pass the DP-capable validation; not trained
    else:
        privacy = NoPrivacy()
        generator_key = "gaussian-copula"
    synthesizer = PrivacyPreservingSynthesizer(
        generator_key=generator_key,
        privacy_mechanism=privacy,
    )
    report = synthesizer.evaluate(real_df, synth_df)
    report.save(output)
    logger.info("Report generated at %s", output)
    click.echo(report.summary())


@cli.command()
@click.option("--data", "-r", "data", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV with the real data.")
@click.option("--epsilons", "-e", "epsilons", default="0.1,0.5,1,2,5,50", show_default=True,
              help="Comma-separated DP budgets (50 ~ almost no DP).")
@click.option("--delta", default=1e-5, type=float, show_default=True, help="DP delta.")
@click.option("--epochs", default=50, type=int, show_default=True, help="Epochs per point.")
@click.option("--rows", "-n", "rows", default=2000, type=int, show_default=True,
              help="Synthetic rows per point.")
@click.option("--output", "-o", "output", default="sweep_report.html", type=click.Path(dir_okay=False),
              help="Output HTML report.")
def sweep(data, epsilons, delta, epochs, rows, output):
    """Train dp-gan with several epsilon and plot the privacy/utility curve."""
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
        click.echo("target eps {:>6} | measured {:>8} | {}".format(
            r["target_epsilon"], r["measured_epsilon"],
            {k: v for k, v in r.items() if k.startswith("util_")}))
    click.echo(f"Report saved at {output}")


@cli.command()
@click.option("--data", "-d", "data", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV with the real data.")
@click.option("--epsilons", "-e", "epsilons", default="1,2,5,10,50", show_default=True,
              help="Comma-separated DP budgets for dp-gan.")
@click.option("--delta", default=1e-5, type=float, show_default=True, help="DP delta.")
@click.option("--baselines", "-b", "baselines", default="gaussian-copula", show_default=True,
              help="Comma-separated non-DP SDV generators; use 'all' for all of them.")
@click.option("--epochs", default=100, type=int, show_default=True, help="dp-gan epochs.")
@click.option("--numeric", default="mode", type=click.Choice(["mode", "uniform"]), show_default=True,
              help="dp-gan numeric encoding: 'mode' (GMM) or 'uniform' (gaussianized).")
@click.option("--rectify-marginals", "rectify_marginals", is_flag=True,
              help="Rectify marginals when sampling (requires --numeric uniform).")
@click.option("--ecdf-epsilon", default=None, type=float,
              help="DP-ECDF marginals budget (requires --numeric uniform). "
                   "The point total reported is training + this value.")
@click.option("--baseline-epochs", default=0, type=int, show_default=True,
              help="Epochs of the SDV baselines (0 = each generator's default).")
@click.option("--rows", "-n", "rows", default=None, type=int,
              help="Synthetic rows per point (default: same as real).")
@click.option("--output", "-o", "output", default="benchmark_report.html", type=click.Path(dir_okay=False),
              help="Output HTML report.")
def benchmark(data, epsilons, delta, baselines, epochs, numeric, rectify_marginals,
              ecdf_epsilon, baseline_epochs, rows, output):
    """Compare dp-gan (several epsilon) vs non-DP SDV generators."""
    if ecdf_epsilon is not None and numeric != "uniform":
        raise click.ClickException("--ecdf-epsilon requires --numeric uniform")
    if rectify_marginals and numeric != "uniform":
        raise click.ClickException("--rectify-marginals requires --numeric uniform")
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
        extra = f" | eps {r['target_epsilon']:.1f} -> measured {r['measured_epsilon']}" \
            if r.get("measured_epsilon") is not None else " (baseline, no DP)"
        click.echo("{:<16} {}{}".format(r["model"], extra,
                                        {k: v for k, v in r.items() if k.startswith("util_")}))
    click.echo(f"Report saved at {output}")


@cli.command()
@click.option("--model", "-m", "model", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Synthesizer saved with 'synthpriv generate --save'.")
@click.option("--tolerance", default=0.05, type=float, show_default=True,
              help="Relative margin allowed over the declared budget.")
def dpcheck(model, tolerance):
    """Validate that a persisted model's DP guarantee is not exceeded."""
    synthesizer = PrivacyPreservingSynthesizer.load_model(model)
    assurance = synthesizer.assert_dp(tolerance=tolerance)
    click.echo(f"status: {assurance.status}")
    click.echo(f"window [measured epsilon, budget]: {assurance.window}")
    click.echo(f"accounted/executed steps: {assurance.accounted_steps}/{assurance.actual_private_steps} "
               f"({'OK' if assurance.steps_match else 'NO'})")
    click.echo(f"noise: {assurance.noise_multiplier} | delta: {assurance.delta}")
    for check in assurance.checks:
        click.echo(f"  [{'OK' if check['status'] == 'ok' else 'X'}] {check['name']}: {check['detail']}")
    click.echo(assurance.message)


if __name__ == "__main__":  # pragma: no cover
    cli()