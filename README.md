# synthpriv

**Tabular data synthesis preserving differential privacy**: generators
(`dp-gan` with DP-SGD, `dp-copula` with pure DP), DP marginal ECDFs, benchmark and
budget sweep, and an HTML privacy/utility report. Built on
[SDV](https://docs.sdv.dev/), [SDMetrics](https://docs.sdv.dev/sdmetrics) and
[Opacus](https://opacus.ai) for differential privacy (DP-SGD) with an RDP accountant.

The project evolves in cumulative phases, each with its own test suite and commit.

## Implemented phases

| Phase | Commit | What it adds |
|---|---|---|
| 0 | `ebb3700` | Core: generator registry, base classes, pipeline, HTML report, CLI |
| 1 | `ebb3700` | SDV tabular generators (`ctgan`, `tvae`, `copula-gan`, `gaussian-copula`) |
| 2 | `ebb3700` | Real DP: `dp-gan` with DP-SGD + RDP accountant and measured epsilon |
| 3 | `6d21828` | `dp-gan` utility: AC-GAN conditioning (CTGAN-style), minority class coverage |
| 4 | `1dc5108` | `epsilon ↔ utility` sweep (privacy/utility curve) |
| 5 | `2c2b05e` | Serialization: `save_model` / `load_model` (regenerate without retraining) |
| 6 | `603445f` | `assert_dp`: DP step integrity and epsilon budget validation |
| 7 | `90b485f` | Docs + reproducible end-to-end demo (README, `examples/demo.py`) |
| 8 | `e979fb7` | `dp-gan` benchmark vs non-DP SDV baselines (utility curves + gap) |
| 9 | `6001540` | `dp-gan` utility hardening: `numeric="uniform"` (gaussianization + interpolated quantile), `rectify_marginals` (guaranteed KS), `label_smoothing`; documented diagnosis and limits |
| 10 | `0194a23` | `DPEcdf`: marginal ECDF with Laplace noise (epsilon per column, parallel composition by bins); `ecdf_epsilon` with sequential composition with training (`total_epsilon` in the report) |
| 11 | `6458cfa` | `dp-copula`: private Gaussian copula (pure DP, δ=0) — captures dependence, the `dp-gan` weak spot |
| 12 | `d1296ad` | `split_budget`: split DP budget between training and marginals |
| — | `84f4fef` | Fix: HTML report without DP-ECDF (`ecdf_epsilon`/`total_epsilon` keys always present) |
| — | `b939af3` | Build: PyPI publication prep (MIT license, PEP 639/URIs/classifiers, dev extras build+twine, README intro) |

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"   # (uses CUDA if available)
```

Dependencies: Python ≥ 3.10, numpy, pandas, scipy, scikit-learn, SDV < 2, SDMetrics,
anonymeter, Opacus, click, Jinja2.

## Quick start

### Python API

```python
from synthpriv import PrivacyPreservingSynthesizer
from synthpriv.privacy.mechanisms import DPSGD

privacy = DPSGD(epsilon=8.0, delta=1e-5)
synth = PrivacyPreservingSynthesizer(
    generator_key="dp-gan",
    generator_kwargs={"epochs": 100, "privacy": privacy, "random_state": 0},
    privacy_mechanism=privacy,
    utility_metrics=["ks_test", "correlation_mae", "ml_utility"],
    privacy_metrics=["nndr", "mia_auc"],
)
synthetic = synth.generate(real_df, num_rows=5000)   # fit + sample
report = synth.evaluate(real_df, synthetic)
report.save("report.html")                          # self-contained HTML report

# persist and regenerate without retraining
synth.save_model("demo_model.sz")
loaded = PrivacyPreservingSynthesizer.load_model("demo_model.sz")
loaded.sample(5000).to_csv("resample.csv", index=False)

# validate the DP guarantee
assurance = loaded.assert_dp()
print(assurance.status, assurance.message)
```

> Note: do not pass `random_state` in `generator_kwargs` when the synthesizer
> already receives it in the constructor (conflict with the constructor's one).

### CLI

```bash
# generate with a DP guarantee
synthpriv generate --data real.csv --epsilon 8 --rows 5000 --save demo_model.sz -o synthetic.csv

# regenerate without retraining (same accounted epsilon)
synthpriv sample --model demo_model.sz --rows 5000 -o resample.csv

# evaluate utility and privacy
synthpriv evaluate --real real.csv --synthetic synthetic.csv --epsilon 8 -o report.html

# epsilon <-> utility sweep
synthpriv sweep --data real.csv --epsilons "0.1,0.5,1,2,5,50" -o sweep_report.html

# benchmark dp-gan vs non-DP SDV generators (curves + utility gap)
synthpriv benchmark --data real.csv --epsilons "1,5,50" --baselines gaussian-copula -o bench.html

# audit that a persisted model's DP guarantee is not exceeded
synthpriv dpcheck --model demo_model.sz --tolerance 0.05
```

### Reproducible demo

```bash
.venv/bin/python examples/demo.py [epsilon] [epochs]
```

Trains `dp-gan` on a sample dataset, generates, evaluates, persists/reloads,
regenerates and runs `assert_dp`. Artifacts under `/tmp/synthpriv_demo/`.

## Differential privacy mechanics

- **Only the discriminator trains with DP-SGD** (gradient clipping + noise, Poisson
  sampling). The generator is post-processing of the discriminator, so the result is
  DP with the epsilon accounted by Opacus's RDP accountant.
- The **real accumulated epsilon** (`accountant.get_epsilon()`) is what is reported,
  not the target: it depends on sample size, epochs and resulting noise.
- `assert_dp` checks two conditions on a model (persisted or in memory):
  1. **Step integrity**: the DP steps accounted by the accountant == those executed
     by the discriminator during training.
  2. **Budget**: measured epsilon ≤ declared epsilon × (1 + tolerance).

  If either fails, the state is `fail` and the RDP guarantee is in question.

### Mechanisms

- `NoPrivacy()` — no formal guarantee (only empirical mitigation + risk metrics).
- `DPSGD(epsilon=1.0, delta=1e-5)` — DP-SGD (Opacus) for the `dp-gan` generator; it
  also configures the total budget for the `dp-copula` generator (pure DP, δ=0).
  `noise_multiplier`: if set, that noise is used; otherwise Opacus computes it to
  reach the budget. After training, `used_noise_multiplier` holds the applied value.

## Generators

| Key | Description | DP |
|---|---|---|
| `gaussian-copula` | Gaussian copula (fast, deterministic) | no |
| `ctgan` | Tabular GAN (SDV) | no |
| `tvae` | Tabular variational autoencoder (SDV) | no |
| `copula-gan` | GAN with copula normalization (SDV) | no |
| `dp-gan` | Custom MLP GAN with DP-SGD and AC-GAN conditioning | yes |
| `dp-copula` | Private Gaussian copula: Laplace ECDF + private correlation (fast) | yes |

### `dp-copula`

Parametric model that attacks the `dp-gan` weak spot (the dependence structure)
with **pure DP** (δ=0, all sub-mechanisms are Laplace). The total budget is split
into `margins_fraction` (≈ default 0.4, private ECDFs per column), `corr_fraction`
(≈ default 0.4, per-entry Laplace-perturbed correlation matrix projected to PSD)
and the rest into categorical frequencies. On purely numeric datasets the whole
remainder goes to the copula. Benchmark on correlated data: learned correlation
≈0.61 vs 0.60 real (dp-gan ≈0.72-0.83) at ε=1-5. In the CLI:
`synthpriv generate --method dp-copula --epsilon E`.

### `dp-gan`

Only the discriminator sees real data and trains with DP-SGD. Relevant hyperparameters:

- `privacy` — `DPSGD` mechanism with target ε/δ.
- `num_modes` (3) — `ModeEncoder` Gaussian Mixture per numeric column; `1` = z-score.
- `numeric` (`"mode"`) — numeric encoding: `"mode"` (mode-specific, mix of Gaussians)
  or `"uniform"` (gaussianized percentile range `Φ⁻¹(rank)` + empirical quantile).
  `"uniform"` handles tails much better and is the recommended option when the
  marginal matters; its inverse interpolates between quantiles (does not return
  exact real values).
- `rectify_marginals` (`False`) — only with `numeric="uniform"`: rectifies at sample
  time the continuous marginals to the real ECDF (monotone per-column transformation,
  preserves the copula). Guarantees KS ≈ 1 for the numerics; it shares the
  per-column empirical-quantile trade-off (utility vs. leakage) documented in
  Limitations.
- `condition_column` (`None`) — column that conditions generation; `None` picks the
  most imbalanced one (lowest entropy) so minority classes do not collapse.
- `aux_lambda` (1.0) — weight of auxiliary losses (classifier + class consistency).
- `generator_steps` (2) — generator steps per discriminator step (the DP budget only
  counts the discriminator).
- `label_smoothing` (0.0) — discriminator label smoothing; useful for stability.
- `ecdf_epsilon` (`None`) — DP budget for the **marginals** (only with
  `numeric="uniform"`): builds a private ECDF per column (Laplace histogram, ε per
  column = `ecdf_epsilon / #numeric`, parallel composition by bins). The
  synthesizer's **total** guarantee is the sequential composition
  `epsilon(accumulated) + ecdf_epsilon` (additive and exact: the ECDF is pure DP,
  δ=0), exposed in the report as `accountant.ecdf_epsilon` / `accountant.total_epsilon`.
  With `None` the ECDF is the raw empirical one (no formal guarantee on the marginal).
  CLI: `synthpriv generate --epsilon E --ecdf-epsilon EE`.
- `ecdf_bounds` (`None`) — **public** support `(min, max)` of the private ECDFs. With
  `bounds` the guarantee is strictly pure DP (fixed grid, no data-derived range);
  without them the support is derived from the 0.001/0.999 quantiles (with margin)
  and a warning documents that nuance.
- Budget split — `from synthpriv import split_budget; b = split_budget(total=10.0,
  margins_fraction=0.3)` returns `b.train` (for `epsilon`) and `b.margins` (for
  `ecdf_epsilon`), with `b.total = train + margins`. The `dp-gan` total is exactly
  additive.

Hardening phase results (tabular dataset with real correlations, 1500 rows,
`numeric="uniform"` + `rectify_marginals`): KS ≈ 1.0 on all three numerics for
ε = 1/5/25 and `ml_utility` on par with SDV's Gaussian copula (≈0.35). The copula
still wins on correlations (corrMAE ≈0.02 vs ≈0.2–0.37): the dependence structure
is what an MLP DP-GAN learns least on small datasets (see Limitations).

## Metrics

- **Utility**: KS (distributions), correlation MAE, ML utility (TSTR).
- **Privacy**: NNDR (distance to nearest real neighbor), inference attack AUC (MIA)
  and anonymeter attacks.

## Tests

```bash
.venv/bin/python -m pytest -q              # fast (default)
.venv/bin/python -m pytest -m slow -q      # trains deep models
```

- `test_registry`, `test_generators`, `test_metrics`, `test_pipeline`, `test_cli`
- `test_dp` (end-to-end DP + minority class coverage)
- `test_sweep` (ε-utility sweep)
- `test_benchmark` (dp-gan vs baselines: structure, gaps, reports)
- `test_serialization` (persistence/reload)
- `test_assurance` (DP step integrity and budget)

## Limitations

- `dp-gan` is an MLP GAN: for small numeric/categorical datasets, not for images or
  sequences.
- The DP guarantee relies on Opacus's RDP accountant and Poisson sampling; formal
  audit with dedicated libraries is out of scope.
- `assert_dp` validates the integrity of the **training DP-SGD** (steps + budget). If
  you also use `ecdf_epsilon`, the synthesizer's **total** guarantee is the
  composition `epsilon(accumulated) + ecdf_epsilon` (additive and exact: the ECDF is
  pure DP, δ=0); the report and the `assert_dp` message show it.
- With `ecdf_epsilon`, the marginal budget is split **equally across columns** and the
  support of each ECDF is trimmed to the empirical 0.001/0.999 quantiles (+margin):
  exact extremes are neither emitted nor published, at the cost of a slight
  generated-range trim.
- Although `dp-gan` respects the budget, real utility at low ε depends on the dataset
  (see the `synthpriv sweep` sweep).
- The dependence structure (correlations) is the `dp-gan` weak spot: partially learned
  and noisy on small datasets; there copulas (SDV) are more accurate.
- With `numeric="uniform"` the marginal is tied to the real empirical quantiles
  (interpolated), which gives strong utility but shares per-column information; use it
  with `rectify_marginals=True` only when marginal utility is the priority over that
  consideration.

## Privacy note

Synthetic data **without DP is not guaranteed anonymization**. The report suggests
which ε level to use and which empirical risk is measured, but formal protection only
comes from training with a DP mechanism (`DPSGD` + `dp-gan`) and verifying it with
`assert_dp`.