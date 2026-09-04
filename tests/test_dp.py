from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthpriv import DPSGDGenerator, PrivacyPreservingSynthesizer
from synthpriv.dp.encoder import ModeEncoder, TabularEncoder
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


# ---------------------------------------------------------------------------
# ModeEncoder: normalizacion mode-specific + condicionamiento
# ---------------------------------------------------------------------------

def test_mode_encoder_roundtrip(real_data):
    """Modos capturan multimodalidad en numericas y el inverso respeta rangos."""
    enc = ModeEncoder(num_modes=3).fit(real_data)
    X = enc.transform(real_data)
    assert X.shape == (len(real_data), enc.total_dims)
    back = enc.inverse(X)
    assert list(back.columns) == list(real_data.columns)
    assert back.shape == real_data.shape
    num = real_data.select_dtypes(include=[np.number]).columns
    # el cuantil 0.01/0.99 del rango real se preserva (clip solo recorta colas)
    for col in num:
        assert back[col].min() >= real_data[col].min() - 1e-6
        assert back[col].max() <= real_data[col].max() + 1e-6
    assert back["age"].isna().sum() == 0
    assert set(back["is_fraud"]).issubset({"no", "yes"})


def test_mode_encoder_condition_selects_imbalanced(real_data):
    """La condicion automatica debe ser is_fraud (la mas imbalanced)."""
    enc = ModeEncoder().fit(real_data)
    assert enc.condition_column_ == "is_fraud"
    assert enc.n_cond == 2
    cond = enc.condition_vectors(real_data)
    assert cond.shape == (len(real_data), 2)
    assert cond.sum(axis=1).all()
    rng = np.random.default_rng(0)
    sampled = enc.sample_conditions(400, rng)
    # el muestreo respeta frecuencias empiricas (ambas clases presentes)
    assert sampled[:, 0].sum() > 50
    assert sampled[:, 1].sum() > 10


def test_mode_encoder_num_modes_one_is_zscore(real_data):
    enc = TabularEncoder().fit(real_data)
    mode1 = ModeEncoder(num_modes=1).fit(real_data)
    X_plain = enc.transform(real_data)
    X_one = mode1.transform(real_data)
    num = real_data.select_dtypes(include=[np.number]).columns
    # con 1 modo, el "valor" de cada numerica es su normalizacion z-score
    vals = {b["col"]: b["val"] for b in mode1.blocks if b["type"] == "num"}
    for i, col in enumerate(num):
        mask = np.abs(X_plain[:, i]) < mode1.clip_value - 1e-3  # filas sin recortar
        assert np.allclose(X_one[mask, vals[col]], X_plain[mask, i])


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
def test_dpgan_conditioning_preserves_imbalanced_class(real_data):
    """El condicionamiento debe generar ambas clases de is_fraud."""
    privacy = DPSGD(epsilon=25.0, delta=1e-3)
    gen = DPSGDGenerator(privacy=privacy, epochs=6, batch_size=32,
                         latent_dim=16, hidden_dim=48, random_state=0,
                         num_modes=3, generator_steps=2)
    out = gen.fit_and_sample(real_data, num_rows=300)
    counts = out["is_fraud"].value_counts()
    assert counts.get("yes", 0) >= 5   # clase minoritaria cubierta
    assert "education" in out.columns


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