Differential privacy
====================

Mechanics
---------

- **Only the discriminator trains with DP-SGD** (gradient clipping + noise,
  Poisson sampling). The generator is post-processing of the discriminator,
  so the result is DP with the epsilon accounted by Opacus's RDP accountant.
- The **real accumulated epsilon** (:meth:`synthpriv.privacy.accountant.PrivacyAccountant.get_epsilon`)
  is what is reported, not the target: it depends on sample size, epochs and
  resulting noise.
- ``assert_dp`` checks two conditions on a model (persisted or in memory):

  1. **Step integrity**: the DP steps accounted by the accountant == those
     executed by the discriminator during training.
  2. **Budget**: measured epsilon ≤ declared epsilon × (1 + tolerance).

  If either fails, the state is ``fail`` and the RDP guarantee is in question.

Mechanisms
----------

- :class:`~synthpriv.privacy.mechanisms.NoPrivacy` — no formal guarantee
  (only empirical mitigation + risk metrics).
- :class:`~synthpriv.privacy.mechanisms.DPSGD` ``(epsilon=1.0, delta=1e-5)``
  — DP-SGD (Opacus) for the ``dp-gan`` generator; it also configures the
  total budget for the ``dp-copula`` generator (pure DP, δ=0).
  ``noise_multiplier``: if set, that noise is used; otherwise Opacus computes
  it to reach the budget. After training, ``used_noise_multiplier`` holds the
  applied value.

Generators
----------

- ``gaussian-copula`` — Gaussian copula (fast, deterministic) — no DP.
- ``ctgan`` — tabular GAN (SDV) — no DP.
- ``tvae`` — tabular variational autoencoder (SDV) — no DP.
- ``copula-gan`` — GAN with copula normalization (SDV) — no DP.
- ``dp-gan`` — custom MLP GAN with DP-SGD and AC-GAN conditioning — **DP**.
- ``dp-copula`` — private Gaussian copula: Laplace ECDF + private
  correlation (fast) — **DP**.

``dp-copula``
-------------

Parametric model that attacks the ``dp-gan`` weak spot (the dependence
structure) with **pure DP** (δ=0, all sub-mechanisms are Laplace). The total
budget is split into ``margins_fraction`` (≈ default 0.4, private ECDFs per
column), ``corr_fraction`` (≈ default 0.4, per-entry Laplace-perturbed
correlation matrix projected to PSD) and the rest into categorical
frequencies. On purely numeric datasets the whole remainder goes to the
copula. Benchmark on correlated data: learned correlation ≈0.61 vs 0.60 real
(``dp-gan`` ≈0.72–0.83) at ε=1–5. In the CLI:
``synthpriv generate --method dp-copula --epsilon E``.

``dp-gan``
----------

Only the discriminator sees real data and trains with DP-SGD. Relevant
hyperparameters (via ``generator_kwargs``):

- ``privacy`` — ``DPSGD`` mechanism with target ε/δ.
- ``num_modes`` *(3)* — ``ModeEncoder`` Gaussian Mixture per numeric column;
  ``1`` = z-score.
- ``numeric`` *(``"mode"``)* — numeric encoding: ``"mode"`` (mode-specific,
  mix of Gaussians) or ``"uniform"`` (gaussianized percentile range
  ``Φ⁻¹(rank)`` + empirical quantile). ``"uniform"`` handles tails much
  better and is the recommended option when the marginal matters; its
  inverse interpolates between quantiles (does not return exact real values).
- ``rectify_marginals`` *(False)* — only with ``numeric="uniform"``:
  rectifies at sample time the continuous marginals to the real ECDF
  (monotone per-column transformation, preserves the copula). Guarantees
  KS ≈ 1 for the numerics; it shares the per-column empirical-quantile
  trade-off (utility vs. leakage) documented in Limitations.
- ``condition_column`` *(None)* — column that conditions generation;
  ``None`` picks the most imbalanced one (lowest entropy) so minority
  classes do not collapse.
- ``aux_lambda`` *(1.0)* — weight of auxiliary losses (classifier + class
  consistency).
- ``generator_steps`` *(2)* — generator steps per discriminator step (the DP
  budget only counts the discriminator).
- ``label_smoothing`` *(0.0)* — discriminator label smoothing; useful for
  stability.

- ``ecdf_epsilon`` *(None)* — DP budget for the **marginals** (only with
  ``numeric="uniform"``): builds a private ECDF per column (Laplace
  histogram, ε per column = ``ecdf_epsilon / #numeric``, parallel
  composition by bins). The synthesizer's **total** guarantee is the
  sequential composition ``epsilon(accumulated) + ecdf_epsilon`` (additive
  and exact: the ECDF is pure DP, δ=0), exposed in the report as
  ``accountant.ecdf_epsilon`` / ``accountant.total_epsilon``. With ``None``
  the ECDF is the raw empirical one (no formal guarantee on the marginal).
  CLI: ``synthpriv generate --epsilon E --ecdf-epsilon EE``.
- ``ecdf_bounds`` *(None)* — **public** support ``(min, max)`` of the
  private ECDFs. With ``bounds`` the guarantee is strictly pure DP (fixed
  grid, no data-derived range); without them the support is derived from the
  0.001/0.999 quantiles (with margin) and a warning documents that nuance.

Hardening results (tabular dataset with real correlations, 1500 rows,
``numeric="uniform"`` + ``rectify_marginals``): KS ≈ 1.0 on all three
numerics for ε = 1/5/25 and ``ml_utility`` on par with SDV's Gaussian copula
(≈0.35). The copula still wins on correlations (corrMAE ≈0.02 vs ≈0.2–0.37):
the dependence structure is what an MLP DP-GAN learns least on small datasets
(see Limitations).

Budget split
------------

:func:`~synthpriv.privacy.budget.split_budget` —
``from synthpriv import split_budget; b = split_budget(total=10.0,
margins_fraction=0.3)`` returns ``b.train`` (for ``epsilon``) and
``b.margins`` (for ``ecdf_epsilon``), with ``b.total = train + margins``.
The ``dp-gan`` total is exactly additive.

Limitations
-----------

- ``dp-gan`` is an MLP GAN: for small numeric/categorical datasets, not for
  images or sequences.
- The DP guarantee relies on Opacus's RDP accountant and Poisson sampling;
  formal audit with dedicated libraries is out of scope.
- ``assert_dp`` validates the integrity of the **training DP-SGD** (steps +
  budget). If you also use ``ecdf_epsilon``, the synthesizer's **total**
  guarantee is the composition ``epsilon(accumulated) + ecdf_epsilon``
  (additive and exact: the ECDF is pure DP, δ=0); the report and the
  ``assert_dp`` message show it.
- With ``ecdf_epsilon``, the marginal budget is split **equally across
  columns** and the support of each ECDF is trimmed to the empirical
  0.001/0.999 quantiles (+margin): exact extremes are neither emitted nor
  published, at the cost of a slight generated-range trim.
- Although ``dp-gan`` respects the budget, real utility at low ε depends on
  the dataset (see ``synthpriv sweep``).
- The dependence structure (correlations) is the ``dp-gan`` weak spot:
  partially learned and noisy on small datasets; there copulas (SDV) are
  more accurate.
- With ``numeric="uniform"`` the marginal is tied to the real empirical
  quantiles (interpolated), which gives strong utility but shares
  per-column information; use it with ``rectify_marginals=True`` only when
  marginal utility is the priority over that consideration.

Privacy note
------------

Synthetic data **without DP is not guaranteed anonymization**. The report
suggests which ε level to use and which empirical risk is measured, but
formal protection only comes from training with a DP mechanism (``DPSGD`` +
``dp-gan``/``dp-copula``) and verifying it with ``assert_dp``.