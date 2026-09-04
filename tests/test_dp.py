from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthpriv import DPSGDGenerator, PrivacyPreservingSynthesizer
from synthpriv.dp.encoder import TabularEncoder
from synthpriv.privacy import DPSGD


def test_encoder_roundtrip(real_data):
    enc = TabularEncoder().fit(real_data)
    X = enc.transform(real_data)
    assert X.shape == (len(real_data), enc.total_dims)
    back = enc.inverse(X)
    assert list(back.columns) == list(real_data.columns)
    assert back.shape == real_data.shape
    assert np.allclose(back[real_data.select_dtypes(include=[np.number]).columns],
                       real_data[real_data.select_dtypes(include=[np.number]).columns])
    assert set(back["education"]).issubset(set(real_data["education"]))


def test_encoder_handles_zero_variance():
    df = pd.DataFrame({"const": [5.0, 5.0, 5.0], "cat": ["a", "b", "a"]})
    enc = TabularEncoder().fit(df)
    X = enc.transform(df)
    assert np.isfinite(X).all()
    back = enc.inverse(X)
    assert list(back["const"]).count(5.0) == len(df)


@pytest.mark.slow
def test_dpsgd_generator_fit_and_sample(real_data):
    """Entrenamiento DP con presupuesto holgado para que termine rapido."""
    privacy = DPSGD(epsilon=30.0, delta=1e-3)
    gen = DPSGDGenerator(privacy=privacy, epochs=3, batch_size=64,
                         latent_dim=16, hidden_dim=32, random_state=0)
    out = gen.fit_and_sample(real_data, num_rows=40)
    assert out.shape[0] == 40
    assert set(real_data.columns) <= set(out.columns)
    assert gen.accounted_epsilon is not None
    assert gen.accounted_epsilon <= 30.0 * 1.5  # cercano al presupuesto (RDP puede superarlo ligeramente)


@pytest.mark.slow
def test_pipeline_dp_end_to_end(real_data, tmp_path):
    privacy = DPSGD(epsilon=20.0, delta=1e-3)
    synth = PrivacyPreservingSynthesizer(
        generator_key="dp-gan",
        generator_kwargs={"epochs": 3, "batch_size": 64, "latent_dim": 16, "hidden_dim": 32,
                          "privacy": privacy},
        privacy_mechanism=privacy,
        privacy_metrics=["nndr"],
        utility_metrics=["ks_test"],
    )
    gen_out = synth.generate(real_data, num_rows=40)
    assert len(gen_out) == 40
    assert synth.accountant.get_epsilon() is not None
    report = synth.evaluate(real_data)
    assert report.data["privacy_mechanism"]["accountant"]["effective_epsilon"] is not None
    html = report.save(tmp_path / "dp_report.html")
    assert "DP activo" in html.read_text()