from __future__ import annotations

from pathlib import Path

import pytest

from synthpriv import PrivacyPreservingSynthesizer
from synthpriv.privacy import DPSGD, NoPrivacy


def test_pipeline_generate_and_evaluate(real_data, tmp_path):
    synth = PrivacyPreservingSynthesizer(
        generator_key="gaussian-copula",
        privacy_mechanism=NoPrivacy(),
        utility_metrics=["ks_test", "correlation_mae", "ml_utility"],
        privacy_metrics=["nndr", "mia_auc"],
    )
    synthetic = synth.generate(real_data, num_rows=200)
    assert len(synthetic) == 200

    report = synth.evaluate(real_data)
    assert set(report.data["utility"]) == {"ks_test", "correlation_mae", "ml_utility"}
    assert set(report.data["privacy_metrics"]) == {"nndr", "mia_auc"}
    assert report.data["privacy_mechanism"]["configured"]["dp"] is False

    out = report.save(tmp_path / "report.html")
    assert out.exists()
    assert "synthpriv" in out.read_text()


def test_pipeline_default_generator_is_ctgan(real_data):
    synth = PrivacyPreservingSynthesizer(generator_key="gaussian-copula")
    assert synth.generator.name == "gaussian-copula"


def test_pipeline_requires_dp_capable_generator():
    """DPSGD requires the 'dp-gan' generator; a regular generator must be rejected."""
    with pytest.raises(ValueError, match="dp-gan"):
        PrivacyPreservingSynthesizer(
            generator_key="gaussian-copula",
            privacy_mechanism=DPSGD(epsilon=1.0),
        )


def test_pipeline_builds_from_instance(real_data):
    from synthpriv import GaussianCopulaGenerator

    gen = GaussianCopulaGenerator()
    synth = PrivacyPreservingSynthesizer(generator=gen)
    out = synth.generate(real_data, num_rows=50)
    assert out.shape[0] == 50


def test_save_report_returns_path(real_data, tmp_path):
    synth = PrivacyPreservingSynthesizer(generator_key="gaussian-copula")
    synth.fit(real_data)
    synthetic = synth.sample(100)
    report = synth.evaluate(real_data, synthetic)
    path = report.save(Path(tmp_path) / "r.html")
    assert isinstance(path, Path)