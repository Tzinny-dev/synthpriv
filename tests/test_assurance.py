"""assert_dp tests: validates accountable steps, budget and the epsilon window."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from synthpriv import DPSGDGenerator, GaussianCopulaGenerator, PrivacyPreservingSynthesizer
from synthpriv.privacy import DPSGD, DpAssurance, NoPrivacy, assert_dp


class _FakeDP:
    """Minimal duplicate of a trained dp-gan generator for logic tests."""

    name = "dp-gan"
    dp_capable = True

    def __init__(self, measured=1.0, privacy=None, steps=10, actual=10):
        self.accounted_epsilon = measured
        self.privacy = privacy if privacy is not None else DPSGD(epsilon=2.0, delta=1e-5)
        self._disc_steps_accounted = steps
        self._disc_steps_actual = actual


def test_non_dp_generator_fails():
    gen = GaussianCopulaGenerator(random_state=0)
    assurance = assert_dp(gen)
    assert not assurance.ok()
    assert assurance.status == "fail"
    assert "No formal privacy guarantee" in assurance.message


def test_budget_respected_ok():
    gen = _FakeDP(measured=1.9, privacy=DPSGD(epsilon=2.0), steps=10, actual=10)
    assurance = assert_dp(gen)
    assert assurance.ok()
    assert assurance.steps_match and assurance.budget_respected
    assert assurance.window == (1.9, 2.0)
    assert assurance.accounted_steps == assurance.actual_private_steps == 10


def test_unaccounted_extra_steps_fail():
    # 12 steps executed, only 10 accounted -> void guarantee
    gen = _FakeDP(measured=1.9, privacy=DPSGD(epsilon=2.0), steps=10, actual=12)
    assurance = assert_dp(gen)
    assert not assurance.ok()
    assert "void" in assurance.message


def test_budget_overrun_fails():
    gen = _FakeDP(measured=5.0, privacy=DPSGD(epsilon=2.0), steps=10, actual=10)
    assurance = assert_dp(gen)
    assert not assurance.ok()
    assert not assurance.budget_respected
    assert assurance.measured_epsilon > assurance.declared_epsilon


def test_fewer_steps_than_accounted_fails():
    gen = _FakeDP(measured=1.0, privacy=DPSGD(epsilon=2.0), steps=10, actual=8)
    assurance = assert_dp(gen)
    assert not assurance.ok()


def test_raise_if_not_passed():
    assurance = DpAssurance(status="fail", message="no")
    with pytest.raises(AssertionError):
        assurance.raise_if_not_passed()


@pytest.mark.slow
def test_dpgan_assert_dp_after_fit(real_data):
    privacy = DPSGD(epsilon=20.0, delta=1e-3)
    gen = DPSGDGenerator(privacy=privacy, epochs=3, batch_size=64,
                         latent_dim=16, hidden_dim=32, random_state=0).fit(real_data)

    assurance = gen.assert_dp()
    assert assurance.ok(), assurance.message
    assert assurance.steps_match
    assert assurance.budget_respected
    assert assurance.accounted_steps == assurance.actual_private_steps > 0
    assert assurance.measured_epsilon <= 20.0 * 1.05

    # impossible declared budget -> must fail
    strict = gen.assert_dp(declared_epsilon=0.5)
    assert not strict.ok()


@pytest.mark.slow
def test_pipeline_assert_dp_in_report(real_data, tmp_path):
    privacy = DPSGD(epsilon=15.0, delta=1e-3)
    synth = PrivacyPreservingSynthesizer(
        generator_key="dp-gan",
        generator_kwargs={"epochs": 3, "batch_size": 64, "latent_dim": 16,
                          "hidden_dim": 32, "privacy": privacy},
        privacy_mechanism=privacy,
        utility_metrics=["ks_test"],
        privacy_metrics=["nndr"],
    )
    synth.generate(real_data, num_rows=40)
    assurance = synth.assert_dp()
    assert assurance.ok(), assurance.message
    assert assurance.measured_epsilon == synth.accountant.get_epsilon()

    report = synth.evaluate(real_data)
    ar = report.data["privacy_mechanism"]["assurance"]
    assert ar["status"] == "ok"
    assert ar["steps_match"] is True
    html = report.save(tmp_path / "dp_report.html").read_text()
    assert "DP ok" in html


@pytest.mark.slow
def test_assert_dp_persisted_model(tmp_path, real_data):
    """A saved model must keep the counters and keep validating DP."""
    privacy = DPSGD(epsilon=20.0, delta=1e-3)
    synth = PrivacyPreservingSynthesizer(
        generator_key="dp-gan",
        generator_kwargs={"epochs": 3, "batch_size": 64, "latent_dim": 16,
                          "hidden_dim": 32, "privacy": privacy},
        privacy_mechanism=privacy,
    )
    synth.generate(real_data, num_rows=40)
    model = synth.save_model(tmp_path / "s.sz")
    restored = PrivacyPreservingSynthesizer.load_model(model)
    ar = restored.assert_dp()
    assert ar.ok(), ar.message

    runner = CliRunner()
    from synthpriv.cli import cli
    res = runner.invoke(cli, ["dpcheck", "--model", str(model)])
    assert res.exit_code == 0, res.output
    assert "status: ok" in res.output
    assert "budget" in res.output