Quick start
===========

Installation
------------

.. code-block:: bash

   pip install synthpriv

Requires Python 3.10–3.12. (Python 3.13 is unsupported for now:
``anonymeter`` pins ``numpy<1.27``, which ships no 3.13 wheels — tracked
upstream.)

For development (editable install + test/build tooling):

.. code-block:: bash

   python -m venv .venv
   .venv/bin/pip install -e ".[dev]"

.. note::

   ``dp-gan`` pulls ``torch`` via Opacus. The default PyPI install is
   CPU-only; install a CUDA-enabled torch build separately if you need GPU
   training (see https://pytorch.org/get-started/locally/).

Python API
----------

.. code-block:: python

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

.. note::

   Do not pass ``random_state`` in ``generator_kwargs`` when the synthesizer
   already receives it in the constructor (conflict with the constructor's
   one).

CLI
---

.. code-block:: bash

   # generate with a DP guarantee
   synthpriv generate --data real.csv --epsilon 8 --rows 5000 --save demo_model.sz -o synthetic.csv

   # regenerate without retraining (same accounted epsilon)
   synthpriv sample --model demo_model.sz --rows 5000 -o resample.csv

   # evaluate utility and privacy
   synthpriv evaluate --real real.csv --synthetic synthetic.csv --epsilon 8 -o report.html

   # epsilon <-> utility sweep (dp-gan or dp-copula)
   synthpriv sweep --data real.csv --generator dp-copula --epsilons "0.1,0.5,1,2,5,50" -o sweep_report.html

   # benchmark DP generator vs non-DP SDV generators (curves + utility gap)
   synthpriv benchmark --data real.csv --generator dp-copula --epsilons "1,5,50" --baselines gaussian-copula -o bench.html

   # audit that a persisted model's DP guarantee is not exceeded
   synthpriv dpcheck --model demo_model.sz --tolerance 0.05

See :doc:`cli` for the full option reference.

Reproducible demo
-----------------

.. code-block:: bash

   .venv/bin/python examples/demo.py [epsilon] [epochs] [ecdf_epsilon]

Trains ``dp-gan`` on a sample dataset, generates, evaluates,
persists/reloads, regenerates and runs ``assert_dp``. Artifacts under
``/tmp/synthpriv_demo/``.