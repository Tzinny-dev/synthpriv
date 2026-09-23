synthpriv
=========

**Tabular data synthesis preserving differential privacy**: generators
(``dp-gan`` with DP-SGD, ``dp-copula`` with pure DP), DP marginal ECDFs,
benchmark and budget sweep, and an HTML privacy/utility report. Built on
SDV_, SDMetrics_ and Opacus_.

.. _SDV: https://docs.sdv.dev/
.. _SDMetrics: https://docs.sdv.dev/sdmetrics
.. _Opacus: https://opacus.ai

.. note::

   Synthetic data **without DP is not guaranteed anonymization**. Formal
   protection only comes from training with a DP mechanism
   (:class:`~synthpriv.privacy.mechanisms.DPSGD`) and verifying it with
   :func:`~synthpriv.privacy.assurance.assert_dp`.

Install with::

   pip install synthpriv

.. toctree::
   :maxdepth: 2
   :caption: User guide

   quickstart
   privacy
   benchmark
   cli

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   changelog

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`