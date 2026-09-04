from __future__ import annotations

from click.testing import CliRunner

from synthpriv.cli import cli


def test_cli_generate_and_evaluate(real_data, tmp_path):
    real_csv = tmp_path / "real.csv"
    real_data.to_csv(real_csv, index=False)
    synth_csv = tmp_path / "synthetic.csv"

    runner = CliRunner()
    gen = runner.invoke(cli, [
        "generate", "--data", str(real_csv),
        "--method", "gaussian-copula", "--rows", "120", "--epochs", "1",
        "--output", str(synth_csv),
    ])
    assert gen.exit_code == 0, gen.output
    assert synth_csv.exists()

    report_path = tmp_path / "report.html"
    ev = runner.invoke(cli, [
        "evaluate", "--real", str(real_csv),
        "--synthetic", str(synth_csv), "--output", str(report_path),
    ])
    assert ev.exit_code == 0, ev.output
    assert report_path.exists()


def test_cli_epsilon_trains_dp(real_data, tmp_path, caplog):
    """'--epsilon' activa el generador dp-gan con DP real (no puede avisar de "sin garantia")."""
    import logging

    real_csv = tmp_path / "real.csv"
    real_data.to_csv(real_csv, index=False)
    synth_csv = tmp_path / "s.csv"

    runner = CliRunner()
    with caplog.at_level(logging.INFO, logger="synthpriv"):
        res = runner.invoke(cli, [
            "generate", "--data", str(real_csv),
            "--method", "gaussian-copula", "--rows", "50",
            "--epochs", "2", "--epsilon", "50.0", "--output", str(synth_csv),
        ])
    assert res.exit_code == 0, res.output
    assert synth_csv.exists()
    assert "sin garantia formal" not in caplog.text.lower()
    assert "epsilon objetivo" in caplog.text.lower()