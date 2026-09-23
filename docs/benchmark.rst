Sweep and benchmark
===================

Metrics
-------

- **Utility**: KS (distributions), correlation MAE, ML utility (TSTR).
- **Privacy**: NNDR (distance to nearest real neighbor), inference attack AUC
  (MIA) and anonymeter attacks.

Epsilon vs utility sweep
------------------------

:func:`~synthpriv.sweep.run_epsilon_sweep` trains a DP-capable generator
(``dp-gan`` or ``dp-copula``) for each budget in ``epsilons`` and records the
**measured** epsilon together with utility/privacy metrics:

.. code-block:: python

   from synthpriv import run_epsilon_sweep

   result = run_epsilon_sweep(
       real_df,
       epsilons=(0.1, 0.5, 1.0, 2.0, 5.0, 50.0),
       generator_key="dp-copula",
   )
   result.dataframe()          # rows ordered by measured epsilon
   result.to_csv("sweep.csv")
   result.save_report("sweep_report.html")

   # most private point whose correlation MAE is still within 0.05
   best = result.best_tradeoff("util_correlation_mae", 0.05)

Each row carries ``model``, ``target_epsilon``, ``measured_epsilon``,
``util_<metric>``, ``priv_<metric>`` and ``fit_seconds``. A very large epsilon
(e.g. 50) is practically equivalent to "no DP": it works as the architecture's
utility ceiling.

CLI:

.. code-block:: bash

   synthpriv sweep --data real.csv --generator dp-gan \
     --epsilons "0.1,0.5,1,2,5,50" -o sweep_report.html

Benchmark: DP vs non-DP baselines
---------------------------------

:func:`~synthpriv.benchmark.run_benchmark` compares a DP generator (several
epsilons) against non-DP SDV generators (utility curves + gap):

.. code-block:: python

   from synthpriv import run_benchmark

   result = run_benchmark(
       real_df,
       epsilons=(1.0, 5.0, 50.0),
       dp_generator="dp-gan",
       baselines=("gaussian-copula", "ctgan"),
   )
   result.dataframe()      # rows carry model + kind ("dp" | "baseline")
   result.utility_gap()    # utility gap vs baselines
   result.best_dp_point()  # best DP point of the sweep
   result.save_report("benchmark.html")

CLI:

.. code-block:: bash

   synthpriv benchmark --data real.csv --generator dp-gan \
     --epsilons "1,5,50" --baselines gaussian-copula -o bench.html

Use ``--baselines all`` for every SDV baseline
(``gaussian-copula``, ``ctgan``, ``tvae``, ``copula-gan``).