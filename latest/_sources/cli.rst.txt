Command-line interface
======================

Entry point: ``synthpriv`` (module docstring in :mod:`synthpriv.cli`).

synthpriv generate
------------------

Train a generator and produce synthetic data.

.. code-block:: bash

   synthpriv generate --data real.csv --epsilon 8 --rows 5000 \
     --save demo_model.sz -o synthetic.csv

Options:

- ``--data/-d`` *(required)* — CSV with the real data.
- ``--method/-m`` *(default: ``ctgan``)* — generator registry key
  (``dp-gan``, ``dp-copula``, ``gaussian-copula``, ``ctgan``, ``tvae``,
  ``copula-gan``).
- ``--rows/-n`` *(default: 5000)* — number of synthetic rows.
- ``--epochs`` *(default: 300)* — training epochs (``dp-gan`` only).
- ``--epsilon`` — differential privacy budget: trains ``dp-gan`` (DP-SGD) or
  ``dp-copula`` (pure DP) with this ε.
- ``--ecdf-epsilon`` — extra DP budget for the ``dp-gan`` marginal ECDFs
  (requires ``--epsilon``; ``dp-gan`` only).
- ``--output/-o`` *(default: ``synthetic.csv``)* — output CSV.
- ``--save`` — persist the trained synthesizer (for ``synthpriv sample``).

synthpriv sample
----------------

Generate rows from a persisted synthesizer **without retraining**: training
with DP costs once; reloading regenerates the dataset with the same privacy
guarantee (accounted epsilon) in seconds.

.. code-block:: bash

   synthpriv sample --model demo_model.sz --rows 5000 -o resample.csv

Options:

- ``--model/-m`` *(required)* — synthesizer saved with ``generate --save``.
- ``--rows/-n`` *(default: 1000)* — number of synthetic rows.
- ``--output/-o`` *(default: ``synthetic.csv``)* — output CSV.

synthpriv evaluate
------------------

Evaluate utility and privacy of some synthetic data and render the HTML
report.

.. code-block:: bash

   synthpriv evaluate --real real.csv --synthetic synthetic.csv \
     --epsilon 8 -o report.html

Options:

- ``--real/-r`` *(required)* — CSV with the real data.
- ``--synthetic/-s`` *(required)* — CSV with the synthetic data.
- ``--epsilon`` — DP budget used when generating (labels the report).
- ``--delta`` *(default: 1e-5)* — associated DP delta.
- ``--output/-o`` *(default: ``report.html``)* — output HTML report.

synthpriv sweep
---------------

Train a DP generator with several epsilons and plot the privacy/utility
curve.

.. code-block:: bash

   synthpriv sweep --data real.csv --generator dp-copula \
     --epsilons "0.1,0.5,1,2,5,50" -o sweep_report.html

Options:

- ``--data/-r`` *(required)* — CSV with the real data.
- ``--generator/-g`` *(default: ``dp-gan``)* — DP-capable generator:
  ``dp-gan`` or ``dp-copula``.
- ``--epsilons/-e`` *(default: ``0.1,0.5,1,2,5,50``)* — comma-separated
  budgets (50 ≈ almost no DP).
- ``--delta`` *(default: 1e-5)* — DP delta.
- ``--epochs`` *(default: 50)* — epochs per point (``dp-gan`` only).
- ``--rows/-n`` *(default: 2000)* — synthetic rows per point.
- ``--output/-o`` *(default: ``sweep_report.html``)* — output HTML report.

synthpriv benchmark
-------------------

Compare a DP generator (several epsilons) vs non-DP SDV generators.

.. code-block:: bash

   synthpriv benchmark --data real.csv --generator dp-gan \
     --epsilons "1,5,50" --baselines gaussian-copula -o bench.html

Options:

- ``--data/-d`` *(required)* — CSV with the real data.
- ``--generator/-g`` *(default: ``dp-gan``)* — ``dp-gan`` or ``dp-copula``.
- ``--epsilons/-e`` *(default: ``1,2,5,10,50``)* — comma-separated budgets.
- ``--delta`` *(default: 1e-5)* — DP delta.
- ``--baselines/-b`` *(default: ``gaussian-copula``)* — comma-separated
  non-DP SDV generators, or ``all``.
- ``--epochs`` *(default: 100)* — ``dp-gan`` epochs (ignored by
  ``dp-copula``).
- ``--numeric`` *(default: ``mode``)* — ``dp-gan`` numeric encoding:
  ``mode`` (GMM) or ``uniform`` (gaussianized).
- ``--rectify-marginals`` *(flag)* — rectify marginals at sampling
  (requires ``--numeric uniform``; ``dp-gan`` only).
- ``--ecdf-epsilon`` — DP-ECDF marginals budget (requires
  ``--numeric uniform``; ``dp-gan`` only); the point's reported total is
  training + this value.
- ``--baseline-epochs`` *(default: 0)* — epochs of the SDV baselines
  (0 = each generator's default).
- ``--rows/-n`` — synthetic rows per point (default: same as real).
- ``--output/-o`` *(default: ``benchmark_report.html``)* — output HTML
  report.

synthpriv dpcheck
-----------------

Validate that a persisted model's DP guarantee is not exceeded (see
:func:`~synthpriv.privacy.assurance.assert_dp`).

.. code-block:: bash

   synthpriv dpcheck --model demo_model.sz --tolerance 0.05

Options:

- ``--model/-m`` *(required)* — synthesizer saved with ``generate --save``.
- ``--tolerance`` *(default: 0.05)* — relative margin allowed over the
  declared budget.
