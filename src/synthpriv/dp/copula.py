"""Gaussian copula with differential privacy (``dp-copula``).

Fully parametric, fast model that attacks the ``dp-gan`` weak spot: the
dependence structure. The guarantee is **pure DP** (delta 0) because all
sub-mechanisms are Laplace (no RDP Gaussian noise):

1. **Marginals**: ``DPEcdf`` per numeric column (Laplace histogram, parallel
   composition by bins; budget ``margins_fraction * epsilon`` split sequentially
   across columns).
2. **Dependence (copula)**: over the real Gaussians ``z = Phi^-1(rank)``
   (deterministic transformation from n, with values bounded by
   ``B = Phi^-1((n-0.5)/n)``, a public quantity), the sample covariance is
   perturbed entry-wise with Laplace noise of scale ``(B^2 / n) / epsilon_entry``.
   Sensitivity per entry <= B^2/n (removing a row changes one per-row term of the
   sum). As entries share the data, the composition is **sequential** across the
   ``d(d-1)/2`` correlations: ``epsilon_entry = corr_fraction * epsilon /
   n_entries``. It is then projected to the correlation sphere (PSD + diagonal 1).
3. **Categoricals**: Laplace frequencies (parallel composition across
   categories, sequential across columns; budget the rest of ``epsilon``).

Sampling is post-processing of these DP outputs: private Gaussian copula
``N(0, R)`` -> ``u = Phi(z)`` -> ``DPEcdf.quantile(u)`` and private
multinomials. Dependence is the main gain over independent marginals, and its
cost is total: the budget is split between marginals, copula and categoricals
according to the configurable fractions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from synthpriv.core.base import BaseSynthesizer
from synthpriv.core.registry import register_generator
from synthpriv.privacy.dpecdf import DPEcdf
from synthpriv.privacy.mechanisms import DPSGD
from synthpriv.utils import get_logger

logger = get_logger("dp")

_EPS = 1e-12


@register_generator(
    key="dp-copula",
    description="Gaussian copula with pure differential privacy (laplace): "
                "DP-ECDF marginals + private correlation with PSD projection.",
    supports=("tabular",),
)
class DPCopulaGenerator(BaseSynthesizer):
    """Differentially private Gaussian copula (parametric and fast).

    Parameters
    ----------
    privacy:
        ``DPSGD`` mechanism with the total ``epsilon`` budget. For this
        generator DP is pure (delta 0): the mechanism's ``delta`` only applies
        to the ``dp-gan``'s DP-SGD.
    margins_fraction:
        Fraction of ``epsilon`` dedicated to the numeric marginals (split
        equally across columns).
    corr_fraction:
        Fraction dedicated to the copula's correlation matrix. With categorical
        columns, the rest goes to their frequencies.
    bins/bounds:
        Grid and public support of the ``DPEcdf``s (see ``DPEcdf``).
    """

    name = "dp-copula"
    dp_capable = True

    def __init__(
        self,
        privacy: DPSGD | None = None,
        margins_fraction: float = 0.4,
        corr_fraction: float = 0.4,
        bins: int = 200,
        bounds: dict[str, tuple[float, float]] | None = None,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.privacy = privacy or DPSGD()
        if not self.privacy.is_dp:
            raise ValueError("dp-copula requires a DPSGD mechanism with epsilon>0")
        if not 0 < margins_fraction < 1 or not 0 < corr_fraction < 1:
            raise ValueError("margins_fraction and corr_fraction must be in (0, 1)")
        self.margins_fraction = float(margins_fraction)
        self.corr_fraction = float(corr_fraction)
        self.bins = max(2, int(bins))
        self.bounds = dict(bounds or {})
        self.bounds = {c: tuple(map(float, b)) for c, b in self.bounds.items()}
        self._seed = int(random_state)
        self._rng = np.random.default_rng(self._seed)

        self._fitted = False
        self.accounted_epsilon: float | None = None
        self.components: dict[str, float] = {}
        self.num_columns: list[str] = []
        self.cat_columns: list[str] = []
        self._ecdfs: list[DPEcdf] = []
        self._corr: np.ndarray | None = None
        self._cat_probs: dict[str, tuple[list[str], np.ndarray]] = {}

    # ------------------------------------------------------------------
    def fit(self, real_data: pd.DataFrame) -> "DPCopulaGenerator":
        data = real_data.copy()
        num_mask = data.select_dtypes(include=[np.number]).columns
        self.num_columns = [c for c in data.columns if c in num_mask]
        self.cat_columns = [c for c in data.columns if c not in num_mask]
        n_num = len(self.num_columns)
        n_cat = len(self.cat_columns)
        eps_total = float(self.privacy.epsilon)
        margins_total = self.margins_fraction * eps_total
        corr_total = self.corr_fraction * eps_total
        cat_share = 1.0 - self.margins_fraction - self.corr_fraction
        if n_cat == 0:
            corr_total = (1.0 - self.margins_fraction) * eps_total  # everything to the copula
            cat_share = 0.0
        self.components = {"margins": margins_total, "corr": corr_total,
                           "cats": cat_share * eps_total}
        self.accounted_epsilon = eps_total

        n = len(data)
        B = float(norm.ppf((n - 0.5) / n)) if n > 2 else 1.0  # public bound of |z|
        G = np.empty((n, n_num))
        self._ecdfs = []
        col_eps = margins_total / n_num if n_num else 0.0
        for j, col in enumerate(self.num_columns):
            v = data[col].astype(float).to_numpy()
            ecdf = DPEcdf(epsilon=col_eps, bins=self.bins,
                          bounds=self.bounds.get(col)).fit(v, rng=self._rng)
            self._ecdfs.append(ecdf)
            u = (v.argsort().argsort() + 0.5) / n  # normalized rank (0,1)
            G[:, j] = np.clip(norm.ppf(u), -B, B)

        if n_num > 1:
            S = (G.T @ G) / n
            entries = n_num * (n_num - 1) // 2
            eps_entry = corr_total / entries if entries else corr_total
            scale = (B * B / max(n, 2)) / eps_entry
            off = S.copy()
            for i in range(n_num):
                for k in range(i + 1, n_num):
                    off[i, k] = S[i, k] + self._rng.laplace(0.0, scale)
                    off[k, i] = off[i, k]
            np.fill_diagonal(off, 1.0)
            self._corr = _project_to_correlation(off)
        else:
            self._corr = np.ones((1, 1))

        self._cat_probs = {}
        if n_cat and cat_share > 0:
            col_eps_cat = cat_share * eps_total / n_cat
            for col in self.cat_columns:
                cats = data[col].astype(str)
                counts = cats.value_counts().sort_index()
                keys = list(counts.index)
                cnt = np.array(counts.to_numpy(), dtype=np.float64)
                noisy = cnt + self._rng.laplace(0.0, 1.0 / col_eps_cat, size=len(keys))
                probs = np.clip(noisy, 0.0, None)
                total = probs.sum() or len(keys)
                self._cat_probs[col] = (keys, probs / total)
        self._fitted = True
        logger.info("dp-copula fitted: n=%d, %d numeric, %d categorical(s); "
                    "total epsilon %.3f = marginals %.3f + copula %.3f + categoricals %.3f",
                    n, n_num, n_cat, eps_total, self.components["margins"],
                    self.components["corr"], self.components["cats"])
        return self

    # ------------------------------------------------------------------
    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("DPCopulaGenerator is not fitted: call fit(real_data).")
        n = int(num_rows)
        frame: dict[str, np.ndarray] = {}
        if self.num_columns:
            n_num = len(self.num_columns)
            Z = self._rng.multivariate_normal(np.zeros(n_num), self._corr, size=n)
            U = norm.cdf(Z)
            for j, col in enumerate(self.num_columns):
                frame[col] = self._ecdfs[j].quantile(U[:, j])
        for col, (keys, probs) in self._cat_probs.items():
            idx = self._rng.choice(len(keys), size=n, p=probs)
            frame[col] = np.asarray(keys)[idx]
        return pd.DataFrame(frame, columns=self.num_columns + self.cat_columns)

    # ------------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        return {
            "margins_fraction": self.margins_fraction,
            "corr_fraction": self.corr_fraction,
            "bins": self.bins,
            "bounds": self.bounds,
            "random_state": int(self._seed),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload = {
            "version": 1,
            "class": self.__class__.__name__,
            "name": self.name,
            "fitted": self._fitted,
            "params": self.get_params(),
            "privacy": self.privacy,
            "accounted_epsilon": self.accounted_epsilon,
            "components": self.components,
            "num_columns": self.num_columns,
            "cat_columns": self.cat_columns,
            "ecdfs": self._ecdfs,
            "corr": self._corr,
            "cat_probs": self._cat_probs,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "DPCopulaGenerator":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        params = dict(payload.get("params", {}))
        params.update(overrides)
        obj = cls(privacy=payload.get("privacy"), **params)
        obj._seed = params.get("random_state", 0)
        obj._rng = np.random.default_rng(obj._seed)
        obj._fitted = bool(payload.get("fitted"))
        obj.accounted_epsilon = payload.get("accounted_epsilon")
        obj.components = dict(payload.get("components") or {})
        obj.num_columns = payload.get("num_columns") or []
        obj.cat_columns = payload.get("cat_columns") or []
        obj._ecdfs = list(payload.get("ecdfs") or [])
        obj._corr = payload.get("corr")
        obj._cat_probs = dict(payload.get("cat_probs") or {})
        return obj


def _project_to_correlation(S: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the correlation sphere (PSD, diag 1)."""
    S = (S + S.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, 0.0, None)
    P = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(P), _EPS, None))
    R = P / np.outer(d, d)
    return np.clip(R, -1.0, 1.0)