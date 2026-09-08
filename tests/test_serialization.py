"""Serialization tests: save/load of generators and of the full synthesizer."""

from __future__ import annotations

import pandas as pd
import pytest

from synthpriv import DPSGDGenerator, GaussianCopulaGenerator, PrivacyPreservingSynthesizer
from synthpriv.privacy import DPSGD


# ---------------------------------------------------------------------------
# Generators (SDV wrappers) - fast
# ---------------------------------------------------------------------------

def test_wrapper_roundtrip(real_data, tmp_path):
    gen = GaussianCopulaGenerator(random_state=0).fit(real_data)
    path = gen.save(tmp_path / "gaussian.pkl")
    assert path.exists()

    loaded = GaussianCopulaGenerator.load(path)
    assert loaded.fitted
    out = loaded.sample(100)
    assert out.shape[0] == 100
    assert list(out.columns) == list(real_data.columns)
    assert set(out["is_fraud"]).issubset({"no", "yes"})


def test_wrapper_load_wrong_class_rejected(real_data, tmp_path):
    path = GaussianCopulaGenerator(random_state=0).fit(real_data).save(tmp_path / "g.pkl")
    with pytest.raises(ValueError, match="expected"):
        from synthpriv import CTGANGenerator
        CTGANGenerator.load(path)


# ---------------------------------------------------------------------------
# dp-gan - slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_dpgan_save_load_preserves_privacy_and_generation(real_data, tmp_path):
    privacy = DPSGD(epsilon=20.0, delta=1e-3)
    gen = DPSGDGenerator(privacy=privacy, epochs=3, batch_size=64,
                         latent_dim=16, hidden_dim=32, random_state=0)
    gen.fit(real_data)
    original_eps = gen.accounted_epsilon

    path = gen.save(tmp_path / "dpgan.pt")
    loaded = DPSGDGenerator.load(path)

    assert loaded.fitted
    assert loaded.accounted_epsilon == original_eps          # DP guarantee preserved
    assert loaded.privacy.used_noise_multiplier == privacy.used_noise_multiplier
    out = loaded.sample(120)
    assert len(out) == 120
    assert set(real_data.columns) <= set(out.columns)         # retrains nothing
    assert loaded._encoder.condition_column_ == "is_fraud"


@pytest.mark.slow
def test_pipeline_save_load_model(real_data, tmp_path):
    privacy = DPSGD(epsilon=18.0, delta=1e-3)
    synth = PrivacyPreservingSynthesizer(
        generator_key="dp-gan",
        generator_kwargs={"epochs": 3, "batch_size": 64, "latent_dim": 16,
                          "hidden_dim": 32, "privacy": privacy},
        privacy_mechanism=privacy,
        utility_metrics=["ks_test"],
        privacy_metrics=["nndr"],
    )
    synth.generate(real_data, num_rows=40)
    expected_eps = synth.accountant.get_epsilon()

    model = synth.save_model(tmp_path / "synth.sz")
    assert model.exists() and str(model).endswith(".sz")
    assert (tmp_path / "synth.sz.meta").exists()

    restored = PrivacyPreservingSynthesizer.load_model(model)
    assert restored.generator.name == "dp-gan"
    assert restored.accountant.get_epsilon() == expected_eps  # same measured epsilon
    restored_sample = restored.sample(30)
    assert len(restored_sample) == 30
    # the restored model report must still show active DP
    report = restored.evaluate(real_data, restored_sample)
    assert report.data["privacy_mechanism"]["accountant"]["effective_epsilon"] == expected_eps


# ---------------------------------------------------------------------------
# CLI: generate --save + sample --model
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cli_generate_save_and_sample(real_data, tmp_path):
    from click.testing import CliRunner

    from synthpriv.cli import cli

    real_csv = tmp_path / "real.csv"
    real_data.to_csv(real_csv, index=False)
    model = tmp_path / "synth.sz"
    runner = CliRunner()
    res = runner.invoke(cli, [
        "generate", "--data", str(real_csv), "--method", "gaussian-copula",
        "--rows", "60", "--epsilon", "50.0", "--epochs", "2",
        "--output", str(tmp_path / "s.csv"), "--save", str(model),
    ])
    assert res.exit_code == 0, res.output
    assert model.exists()

    out_csv = tmp_path / "regenerated.csv"
    res2 = runner.invoke(cli, [
        "sample", "--model", str(model), "--rows", "80", "--output", str(out_csv),
    ])
    assert res2.exit_code == 0, res2.output
    df = pd.read_csv(out_csv)
    assert len(df) == 80