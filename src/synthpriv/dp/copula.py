"""Copula gaussiana con privacidad diferencial (``dp-copula``).

Modelo completamente parametrico y rapido que ataca el punto debil del
``dp-gan``: la estructura de dependencia. La garantia es **DP pura** (delta 0)
porque todos los sub-mecanismos son Laplace (ningun ruido gaussiano RDP):

1. **Marginales**: ``DPEcdf`` por columna numerica (histograma Laplace,
   composicion paralela por bins; presupuesto ``margins_fraction * epsilon``
   repartido secuencialmente entre columnas).
2. **Dependencia (copula)**: sobre los gaussianos ``z = Phi^-1(rank)`` reales
   (transformacion determinista a partir de n, con valores acotados por
   ``B = Phi^-1((n-0.5)/n)``, cifra publica), la covarianza muestral se
   perturba entrada a entrada con Laplace de escala
   ``(B^2 / n) / epsilon_entry``. Sensibilidad por entrada <= B^2/n (quitar una
   fila cambia un termino del sumatorio por fila). Como las entradas comparten
   los datos, la composicion es **secuencial** entre las ``d(d-1)/2``
   correlaciones: ``epsilon_entry = corr_fraction * epsilon / n_entries``.
   Despues se proyecta a la esfera de correlaciones (PSD + diagonal 1).
3. **Categoricas**: frecuencias con Laplace (composicion paralela entre
   categorias, secuencial entre columnas; presupuesto el resto de ``epsilon``).

El muestreo es post-proceso de estas salidas DP: copula gaussiana privada
``N(0, R)`` -> ``u = Phi(z)`` -> ``DPEcdf.quantile(u)`` y multinomiales
privadas. La dependencia es la principal ganancia frente a las marginals
independientes, y su coste es total: el presupuesto se reparte entre marginales,
copula y categoricas segun las fracciones configurables.
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
    description="Copula gaussiana con privacidad diferencial pura (laplace): "
                "marginales DP-ECDF + correlacion privada con proyeccion PSD.",
    supports=("tabular",),
)
class DPCopulaGenerator(BaseSynthesizer):
    """Copula gaussiana diferencialmente privada (parametrica y rapida).

    Parameters
    ----------
    privacy:
        Mecanismo ``DPSGD`` con el presupuesto ``epsilon`` total. Para esta
        generacion DP es pura (delta 0): el ``delta`` del mecanismo solo aplica
        al DP-SGD del ``dp-gan``.
    margins_fraction:
        Fraccion de ``epsilon`` dedicada a los marginales numericos (repartida
        equitativamente entre columnas).
    corr_fraction:
        Fraccion dedicada a la matriz de correlacion de la copula. Con
        columnas categoricas, el resto va a sus frecuencias.
    bins/bounds:
        Cuadricula y soporte publico de las ``DPEcdf`` (ver ``DPEcdf``).
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
            raise ValueError("dp-copula exige un mecanismo DPSGD con epsilon>0")
        if not 0 < margins_fraction < 1 or not 0 < corr_fraction < 1:
            raise ValueError("margins_fraction y corr_fraction deben estar en (0, 1)")
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
            corr_total = (1.0 - self.margins_fraction) * eps_total  # todo a la copula
            cat_share = 0.0
        self.components = {"margins": margins_total, "corr": corr_total,
                           "cats": cat_share * eps_total}
        self.accounted_epsilon = eps_total

        n = len(data)
        B = float(norm.ppf((n - 0.5) / n)) if n > 2 else 1.0  # cota publica de |z|
        G = np.empty((n, n_num))
        self._ecdfs = []
        col_eps = margins_total / n_num if n_num else 0.0
        for j, col in enumerate(self.num_columns):
            v = data[col].astype(float).to_numpy()
            ecdf = DPEcdf(epsilon=col_eps, bins=self.bins,
                          bounds=self.bounds.get(col)).fit(v, rng=self._rng)
            self._ecdfs.append(ecdf)
            u = (v.argsort().argsort() + 0.5) / n  # rango normalizado (0,1)
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
        logger.info("dp-copula ajustada: n=%d, %d numericas, %d categorica(s); "
                    "epsilon total %.3f = marginales %.3f + copula %.3f + categorias %.3f",
                    n, n_num, n_cat, eps_total, self.components["margins"],
                    self.components["corr"], self.components["cats"])
        return self

    # ------------------------------------------------------------------
    def sample(self, num_rows: int = 1000, **kwargs) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("DPCopulaGenerator no entrenado: llama a fit(real_data).")
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
    """Proyecta una matriz simetrica a la esfera de correlaciones (PSD, diag 1)."""
    S = (S + S.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, 0.0, None)
    P = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(P), _EPS, None))
    R = P / np.outer(d, d)
    return np.clip(R, -1.0, 1.0)