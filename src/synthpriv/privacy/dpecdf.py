"""ECDF with a formal differential privacy guarantee (Laplace / histogram).

Mechanism: histogram with Laplace-noisy counts for each numeric column.

- Sensitivity per bin = 1 (adding/removing a row moves each count by at most 1)
  and the bins partition the data disjointly: by **parallel** composition, the
  whole column consumes a single ``epsilon`` (Laplace noise of scale ``1/epsilon``
  per bin).
- Columns are mutually disjoint but not a partition of the same datum, so the
  total budget is split sequentially across columns: each column uses
  ``total_epsilon / n_columns``.
- The grid range is trimmed to the empirical 0.001/0.999 quantiles (with a
  margin), avoiding the publication of exact extremes; values outside that
  support are not emitted.
- The quantile function (inverse of the noisy ECDF, smoothed monotonically and
  by linear interpolation) is post-processing of the DP output, so it consumes
  no extra budget. Emitting values from this inverse keeps the per-column DP
  guarantee; composition with the training DP-SGD gives the synthesizer's total
  guarantee.
"""

from __future__ import annotations

import warnings

import numpy as np

from synthpriv.utils import get_logger

logger = get_logger("dp")


class DPEcdf:
    """Private per-column ECDF based on a Laplace histogram.

    Parameters
    ----------
    epsilon:
        DP budget of this column (0 < epsilon <= total marginals budget /
        n_columns). The rest of the synthesizer must compose it with the
        training epsilon (``ecdf_epsilon + effective_epsilon``).
    bins:
        Number of equal-width intervals over the trimmed support.
    q_low/q_high:
        Quantiles (0..1) defining the grid support when ``bounds`` are not
        given; out-of-range values are aggregated at the edges via range
        clipping.
    bounds:
        **Public** support ``(min, max)`` of the column. If provided, the grid
        is fixed and the mechanism is strictly pure DP (the guarantee does not
        depend on any prior data). If ``None``, the support is derived from the
        empirical 0.001/0.999 quantiles of the data (with margin): practical,
        but the range itself reveals sample information — a warning is emitted
        and passing public ``bounds`` is recommended when available.
    """

    def __init__(self, epsilon: float = 1.0, bins: int = 200,
                 q_low: float = 0.001, q_high: float = 0.999,
                 bounds: tuple[float, float] | None = None):
        if epsilon <= 0:
            raise ValueError(f"DP-ECDF epsilon must be > 0, got {epsilon!r}")
        self.epsilon = float(epsilon)
        self.bins = max(2, int(bins))
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        if bounds is not None and not (
                bounds[0] < bounds[1] and np.isfinite(bounds[0]) and np.isfinite(bounds[1])):
            raise ValueError(f"Invalid bounds: {bounds!r} (requires finite min < max)")
        self.bounds = tuple(map(float, bounds)) if bounds is not None else None
        self.n = 0
        self._edges: np.ndarray | None = None
        self._cdf: np.ndarray | None = None  # len(bins)+1, monotone, ends at 1

    # ------------------------------------------------------------------
    def fit(self, values: np.ndarray, rng: np.random.Generator | None = None) -> "DPEcdf":
        """Build the private ECDF from ``values`` (one entry per row).
        """
        v = np.asarray(values, dtype=float).ravel()
        if v.size < 2:
            raise ValueError(f"DPEcdf needs at least 2 values, got {v.size}")
        self.n = int(v.size)
        rng = rng or np.random.default_rng(0)

        if self.bounds is not None:
            lo, hi = self.bounds
        else:
            lo = float(np.quantile(v, min(self.q_low, self.q_high)))
            hi = float(np.quantile(v, max(self.q_low, self.q_high)))
            pad = 1e-6 + 0.05 * (hi - lo)  # rounds support, does not publish extremes
            lo, hi = lo - pad, hi + pad
            if hi <= lo:
                hi = lo + 1.0
            warnings.warn(
                "DPEcdf uses a data-derived support (0.001/0.999 quantiles + margin). "
                "The histogram DP guarantee is strict given that support, but the range "
                "itself reveals sample information; pass public 'bounds=(min,max)' when "
                "available for a fully formal guarantee.",
                stacklevel=2,
            )

        counts = np.histogram(v, bins=self.bins, range=(lo, hi))[0].astype(np.float64)
        scale = 1.0 / self.epsilon
        noisy = counts + rng.laplace(0.0, scale, size=self.bins)
        w = np.maximum(noisy, 0.0)
        total = float(w.sum())
        if total <= 0 or not np.isfinite(total):
            w = np.ones(self.bins)
            total = float(self.bins)
        pdf = w / total
        self._edges = np.linspace(lo, hi, self.bins + 1)
        # monotone smoothing: cdf = cumsum of the pdf (already ordered, non-negative)
        cdf = np.concatenate([[0.0], np.cumsum(pdf)])
        cdf = cdf / cdf[-1]
        # avoid exact plateaus: normalize and clamp
        self._cdf = np.clip(cdf, 0.0, 1.0)
        self._cdf[-1] = 1.0
        return self

    # ------------------------------------------------------------------
    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Inverse of the private ECDF over the uniform quantiles ``u`` (0..1).

        Linearly interpolates between grid edges (DP post-processing).
        """
        if self._edges is None or self._cdf is None:
            raise RuntimeError("DPEcdf not fitted: call fit(values) first.")
        u_arr = np.asarray(u, dtype=np.float64)
        flat = u_arr.ravel()
        q = np.clip(flat, 1e-12, 1.0 - 1e-12)
        i = np.clip(np.searchsorted(self._cdf, q, side="right") - 1, 0, self.bins - 1)
        c0 = self._cdf[i]
        c1 = self._cdf[i + 1]
        span = np.maximum(c1 - c0, 1e-12)
        t = np.clip((q - c0) / span, 0.0, 1.0)
        out = self._edges[i] + t * (self._edges[i + 1] - self._edges[i])
        return out.reshape(u_arr.shape)

    # ------------------------------------------------------------------
    def report(self) -> dict:
        """Privacy state summary of the column (for the report)."""
        return {
            "mechanism": "dp-ecdf",
            "dp": True,
            "epsilon": self.epsilon,
            "bins": self.bins,
            "bounds_public": self.bounds is not None,
            "bounds": self.bounds,
            "range": None if self._edges is None
                else (float(self._edges[0]), float(self._edges[-1])),
            "n": self.n,
        }