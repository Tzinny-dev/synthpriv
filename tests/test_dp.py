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


def test_uniform_encoder_roundtrip_and_tails(real_data):
    """El inverso uniform tiene rango acotado por cola real (interpolado)."""
    enc = ModeEncoder(num_modes=3, numeric="uniform").fit(real_data)
    v = enc.transform(real_data)
    assert v.shape == (len(real_data), enc.total_dims)
    assert np.isfinite(v).all()
    back = enc.inverse(v)
    assert list(back.columns) == list(real_data.columns)
    assert back.isna().sum().sum() == 0
    for col in real_data.select_dtypes(include=[np.number]).columns:
        if col in ("age", "score"):
            assert back[col].min() >= real_data[col].min() - 1e-6
            assert back[col].max() <= real_data[col].max() + 1e-6


def test_uniform_rectify_matches_marginals(real_data):
    """Rectificar marginales sobre numericas uniform deja cada marginal
    exactamente uniforme: el inverso reproduce los cuantiles empiricos reales."""
    from scipy import stats
    enc = ModeEncoder(num_modes=3, numeric="uniform").fit(real_data)
    X = np.random.default_rng(0).standard_normal((2000, enc.total_dims)).astype(np.float32)
    out = enc.inverse(enc.rectify(X.copy()))
    for col in real_data.select_dtypes(include=[np.number]).columns:
        assert stats.ks_2samp(real_data[col], out[col]).pvalue > 0.99


def test_uniform_rectify_preserves_rank_dependence(real_data):
    """La rectificacion es monotona por columna: la correlacion de Spearman
    muestral no cambia (la copula se conserva)."""
    from scipy import stats
    enc = ModeEncoder(num_modes=3, numeric="uniform").fit(real_data)
    X0 = np.random.default_rng(1).multivariate_normal(
        [0, 0], [[1, 0.6], [0.6, 1]], 500).astype(np.float32)
    X = np.zeros((500, enc.total_dims), dtype=np.float32)
    b = [b for b in enc.blocks if b["type"] == "num" and b.get("kind") == "uniform"]
    X[:, 0:2] = X0
    a, c = b[0], b[1]
    rho_before = float(stats.spearmanr(X[:, a["val"]], X[:, c["val"]]).statistic)
    Xr = enc.rectify(X.copy())
    rho_after = float(stats.spearmanr(Xr[:, a["val"]], Xr[:, c["val"]]).statistic)
    assert abs(rho_after - rho_before) < 1e-3


def test_dpecdf_quantile_health():
    """La ECDF privada reconstruye los cuantiles (mediana/cola) de forma monotona."""
    from synthpriv import DPEcdf
    rng = np.random.default_rng(0)
    v = rng.gamma(3.0, 5000.0, 4000)
    ecdf = DPEcdf(epsilon=2.0, bins=400).fit(v, rng=default_num_rng())
    med = ecdf.quantile(np.array([0.5]))[0]
    assert np.isfinite(med) and abs(med - np.median(v)) / np.median(v) < 0.25
    u = np.linspace(0.01, 0.99, 99)
    q = ecdf.quantile(u)
    assert ((q[1:] - q[:-1]) >= 0).all()
    assert ecdf.epsilon == 2.0
    assert ecdf.report()["dp"] is True


def default_num_rng():
    return np.random.default_rng(0)


def test_encoder_uniform_with_dpecdf(real_data):
    """Con dp_ecdf_epsilon, el inverso usa la ECDF privada: marginales cercanos
    a los reales, presupuesto registrado y sin excepciones."""
    from scipy import stats
    enc = ModeEncoder(num_modes=3, numeric="uniform",
                      dp_ecdf_epsilon=6.0, ecdf_bins=120).fit(real_data)
    assert enc.ecdf_epsilon == 6.0
    X = np.random.default_rng(0).standard_normal((2000, enc.total_dims)).astype(np.float32)
    out = enc.inverse(enc.rectify(X.copy()))
    for col in real_data.select_dtypes(include=[np.number]).columns:
        p = stats.ks_2samp(real_data[col], out[col]).pvalue
        assert p > 0.05, f"{col}: ks p={p}"  # DP-ECDF pierde algo vs ECDF cruda
    num_blocks = [b for b in enc.blocks if b["type"] == "num" and b.get("kind") == "uniform"]
    assert all(b["ecdf"] is not None for b in num_blocks)


def test_encoder_dpecdf_requires_uniform():
    with pytest.raises(ValueError):
        ModeEncoder(num_modes=3, numeric="mode", dp_ecdf_epsilon=1.0)


def test_dpecdf_public_bounds_no_warning():
    """Con bounds publicos el mecanismo es DP pura estricta (sin warning)."""
    import warnings as _warn
    from synthpriv import DPEcdf
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 500)
    with _warn.catch_warnings(record=True) as rec:
        _warn.simplefilter("always")
        ecdf = DPEcdf(epsilon=1.0, bins=50, bounds=(0.0, 1.0)).fit(v, rng=rng)
    assert not any("soporte derivado" in str(w.message) for w in rec)
    assert ecdf.bounds == (0.0, 1.0)
    q = ecdf.quantile(np.array([0.0, 1.0]))
    assert q[0] >= 0.0 and q[1] <= 1.0
    assert ecdf.report()["bounds_public"] is True


def test_dpecdf_data_derived_range_warns():
    """Sin bounds el soporte viene de los datos: se avisa explicitamente."""
    from synthpriv import DPEcdf
    with pytest.warns(UserWarning, match="soporte derivado"):
        DPEcdf(epsilon=1.0, bins=50).fit(np.linspace(0, 10, 100))


def test_encoder_uniform_with_dpecdf_public_bounds(real_data):
    from synthpriv import DPEcdf
    enc = ModeEncoder(num_modes=3, numeric="uniform",
                      dp_ecdf_epsilon=6.0, ecdf_bins=120,
                      ecdf_bounds=(0.0, 100.0)).fit(real_data)
    for b in (b for b in enc.blocks if b.get("kind") == "uniform"):
        assert b["ecdf"].bounds == (0.0, 100.0)
        assert b["ecdf"].report()["bounds_public"] is True


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
def test_dp_ecdf_pipeline_reports_total(real_data, tmp_path):
    """dp-gan con DP-ECDF: el informe compone entrenamiento + marginales."""
    privacy = DPSGD(epsilon=30.0, delta=1e-3)
    synth = PrivacyPreservingSynthesizer(
        generator_key="dp-gan",
        generator_kwargs={"epochs": 2, "batch_size": 64, "latent_dim": 16, "hidden_dim": 32,
                          "numeric": "uniform", "rectify_marginals": True,
                          "ecdf_epsilon": 2.0, "privacy": privacy},
        privacy_mechanism=privacy,
        utility_metrics=["ks_test"],
        privacy_metrics=["nndr"],
    )
    gen_out = synth.generate(real_data, num_rows=40)
    assert len(gen_out) == 40
    report = synth.evaluate(real_data, gen_out)
    acc = report.data["privacy_mechanism"]["accountant"]
    assert acc["ecdf_epsilon"] == 2.0
    base = synth.accountant.get_epsilon()
    assert base is not None and acc["total_epsilon"] == pytest.approx(base + 2.0)
    p = synth.save_model(tmp_path / "ecdf_model.sz")
    loaded = PrivacyPreservingSynthesizer.load_model(p)
    assert loaded.generator.ecdf_epsilon == 2.0


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