"""Encoders tabulares hacia un espacio numerico uniforme (y su inverso).

- ``TabularEncoder``: z-score + one-hot (basico, estable).
- ``ModeEncoder``: normalizacion mode-specific con Gaussian Mixture +
  condicionamiento por clase imbalanced (estilo CTGAN) para mejor utilidad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from synthpriv.privacy.dpecdf import DPEcdf


class TabularEncoder:
    """Transforma un DataFrame a un array continuo y lo reconstruye.

    El bloque numerico usa los primeros ``num_dims`` componentes; las columnas
    categoricas se one-hot-codean despues, sus intervalos se exponen en
    ``categorical_spans`` como listas ``(start, end)``.
    """

    def __init__(self):
        self.columns: list[str] = []
        self.num_columns: list[str] = []
        self.cat_columns: list[str] = []
        self._num_mean: dict[str, float] = {}
        self._num_std: dict[str, float] = {}
        self._cat_categories: dict[str, list[str]] = {}
        self.categorical_spans: list[tuple[int, int]] = []
        self.num_dims = 0
        self.total_dims = 0

    def fit(self, data: pd.DataFrame) -> "TabularEncoder":
        self.columns = list(data.columns)
        num_mask = data.select_dtypes(include=[np.number]).columns
        self.num_columns = [c for c in data.columns if c in num_mask]
        self.cat_columns = [c for c in data.columns if c not in num_mask]

        for col in self.num_columns:
            values = data[col].astype(float)
            self._num_mean[col] = float(values.mean())
            self._num_std[col] = float(values.std()) or 1.0
            self.num_dims += 1

        offset = self.num_dims
        for col in self.cat_columns:
            categories = sorted(pd.unique(data[col].dropna().astype(str)))
            end = offset + len(categories)
            self.categorical_spans.append((offset, end))
            self._cat_categories[col] = categories
            offset = end
        self.total_dims = offset
        return self

    def dump(self) -> dict:
        return {
            "num_columns": self.num_columns,
            "cat_columns": self.cat_columns,
            "num_mean": self._num_mean,
            "num_std": self._num_std,
            "cat_categories": self._cat_categories,
        }

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        rows = len(data)
        out = np.zeros((rows, self.total_dims), dtype=np.float32)
        for i, col in enumerate(self.num_columns):
            z = (data[col].astype(float) - self._num_mean[col]) / self._num_std[col]
            out[:, i] = z.to_numpy()
        for col, (start, end) in zip(self.cat_columns, self.categorical_spans):
            cats = self._cat_categories[col]
            idx = {c: k for k, c in enumerate(cats)}
            for j, raw in enumerate(data[col].astype(str)):
                if raw in idx:
                    out[j, start + idx[raw]] = 1.0
                else:
                    out[j, start] = 1.0  # categorias invisibles -> primera
        return out

    def inverse(self, X: np.ndarray) -> pd.DataFrame:
        X = np.asarray(X, dtype=np.float32)
        frame: dict[str, np.ndarray] = {}
        for i, col in enumerate(self.num_columns):
            frame[col] = X[:, i] * self._num_std[col] + self._num_mean[col]
        for col, (start, end) in zip(self.cat_columns, self.categorical_spans):
            idx = np.argmax(X[:, start:end], axis=1)
            cats = np.array(self._cat_categories[col])
            frame[col] = cats[idx]
        return pd.DataFrame(frame, columns=self.columns)


class ModeEncoder:
    """Encoder tabular de mejor utilidad, inspirado en CTGAN.

    Numericas: dos modos de normalizacion.
    - ``mode`` (mode-specific): cada valor se asigna al modo mas probable de una
      Gaussian Mixture (por columna), se normaliza dentro del modo y se codifica
      junto con un one-hot del modo. Captura distribuciones multimodales que el
      z-score aplasta. ``num_modes=1`` degenera a z-score.
    - ``uniform`` (empirico-CDF/rank gaussianizado): mapea cada valor a su rango
      percentil y luego a un valor normal estandar via ``Phi^-1`` (gaussianizacion
      por columna). El inverso aplica ``Phi`` y el cuantil empirico, con lo que el
      marginal se reconstruye por construccion **si el generador emite valores
      normales estandar** (marginal mas facil de aprender y sin saturacion de
      ``tanh``).

    Categoricas: one-hot. Ademas, se elige la columna categorica mas
    imbalanced como ``condition_column_`` (menor entropia) para el
    condicionamiento AC-GAN del generador; sus one-hots se exponen en
    ``condition_vectors``/``sample_conditions``.

    La salida usa ``blocks``: cada bloque numerico es
    ``[valor normalizado (+ one-hot del modo en modo ``mode``)]`` y cada
    categorico es su one-hot.
    """

    def __init__(self, num_modes: int = 5, clip_value: float = 3.0,
                 condition_column: str | None = None,
                 numeric: str = "mode",
                 dp_ecdf_epsilon: float | None = None,
                 ecdf_bins: int = 200,
                 ecdf_bounds: tuple[float, float] | None = None):
        self.num_modes = max(1, int(num_modes))
        self.clip_value = float(clip_value)
        self.condition_column = condition_column
        self.numeric = numeric
        if numeric not in ("mode", "uniform"):
            raise ValueError(f"numeric debe ser 'mode' o 'uniform', se recibio {numeric!r}")
        if dp_ecdf_epsilon is not None and numeric != "uniform":
            raise ValueError("dp_ecdf_epsilon solo se aplica con numeric='uniform'")
        self.dp_ecdf_epsilon = float(dp_ecdf_epsilon) if dp_ecdf_epsilon is not None else None
        self.ecdf_bins = max(2, int(ecdf_bins))
        self.ecdf_bounds = tuple(map(float, ecdf_bounds)) if ecdf_bounds is not None else None
        self.ecdf_epsilon = None  # presupuesto total consumido por marginales DP (tras fit)
        self.columns: list[str] = []
        self.num_columns: list[str] = []
        self.cat_columns: list[str] = []
        self.blocks: list[dict] = []          # (start, end) posiciones inward
        self.total_dims = 0
        self.condition_column_: str | None = None
        self.n_cond = 0
        self._cond_block: dict | None = None
        self._cond_freq: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> "ModeEncoder":
        self.columns = list(data.columns)
        num_mask = data.select_dtypes(include=[np.number]).columns
        self.num_columns = [c for c in data.columns if c in num_mask]
        self.cat_columns = [c for c in data.columns if c not in num_mask]

        pos = 0
        n_num = len(self.num_columns)
        col_eps = self.dp_ecdf_epsilon / n_num if self.dp_ecdf_epsilon else None
        for col in self.num_columns:
            v = data[col].astype(float).to_numpy()
            if self.numeric == "uniform":
                block = {
                    "type": "num", "col": col, "val": pos, "kind": "uniform",
                    "min": float(np.min(v)), "max": float(np.max(v)),
                    "values": np.sort(v), "n": len(v),
                }
                if col_eps is not None:
                    block["ecdf"] = DPEcdf(epsilon=col_eps, bins=self.ecdf_bins,
                                           bounds=self.ecdf_bounds).fit(v)
                self.blocks.append(block)
                pos += 1
                continue
            n_unique = len(np.unique(v))
            k = max(1, min(self.num_modes, n_unique))
            gmm = None
            modes = np.zeros(len(v), dtype=int)
            if k > 1 and len(v) >= max(8, 4 * k):
                g = GaussianMixture(n_components=min(k, len(v) // 2), random_state=0)
                try:
                    g.fit(v.reshape(-1, 1))
                    gmm = g
                    modes = g.predict(v.reshape(-1, 1))
                except Exception:  # degenerado -> modo unico
                    gmm = None
            if gmm is None:
                means = [float(np.mean(v))]
                stds = [float(np.std(v, ddof=1)) or 1.0]
            else:
                means = [float(m) for m in gmm.means_.ravel()]
                stds = [float(s) for s in np.sqrt(gmm.covariances_.ravel())]
            self.blocks.append({
                "type": "num", "col": col, "val": pos, "kind": "mode",
                "modes": (pos + 1, pos + 1 + len(means)), "k": len(means),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "means": means, "stds": stds, "gmm": gmm,
            })
            pos += 1 + len(means)

        for col in self.cat_columns:
            cats = sorted(pd.unique(data[col].dropna().astype(str)).tolist())
            if not cats:
                cats = ["NA"]
            self.blocks.append({"type": "cat", "col": col, "start": pos,
                               "end": pos + len(cats), "categories": cats})
            pos += len(cats)
        self.total_dims = pos
        self.ecdf_epsilon = self.dp_ecdf_epsilon

        self._fit_condition(data)
        return self

    def _entropy(self, values: pd.Series) -> float:
        freqs = values.astype(str).value_counts(normalize=True).to_numpy()
        return float(-np.sum(freqs * np.log(freqs)))

    def _fit_condition(self, data: pd.DataFrame) -> None:
        if self.condition_column is not None:
            if self.condition_column not in self.cat_columns:
                raise ValueError(f"Columna de condicion '{self.condition_column}' no es categorica")
            self.condition_column_ = self.condition_column
        else:
            cands = [b for b in self.blocks if b["type"] == "cat"]
            if cands:
                self.condition_column_ = min(cands, key=lambda b: self._entropy(data[b["col"]]))["col"]
            else:
                self.condition_column_ = None

        self._cond_block = next(
            (b for b in self.blocks
             if b["type"] == "cat" and b["col"] == self.condition_column_), None)
        if self._cond_block is not None:
            self.n_cond = len(self._cond_block["categories"])
            cats = self._cond_block["categories"]
            freqs = data[self.condition_column_].astype(str).value_counts(normalize=True)
            self._cond_freq = np.array([float(freqs.get(c, 0.0)) for c in cats])
        else:
            self.n_cond = 0
            self._cond_freq = None

    # ------------------------------------------------------------------
    def _encode_numeric_block(self, b: dict, col: np.ndarray, out: np.ndarray) -> None:
        v = col
        if b.get("kind") == "uniform":
            left = np.searchsorted(b["values"], v, side="left")
            right = np.searchsorted(b["values"], v, side="right")
            rank = (left + right) / 2.0 / b["n"]  # promedio de empates en (0,1)
            out[:, b["val"]] = np.clip(norm.ppf(rank), -self.clip_value, self.clip_value).astype(np.float32)
            return
        if b["gmm"] is not None:
            mode = b["gmm"].predict(v.reshape(-1, 1))
        else:
            mode = np.zeros(len(v), dtype=int)
        means = np.asarray(b["means"]); stds = np.asarray(b["stds"])
        nrm = (v - means[mode]) / np.maximum(stds[mode], 1e-6)
        out[:, b["val"]] = np.clip(nrm, -self.clip_value, self.clip_value)
        out[:, b["modes"][0]:b["modes"][1]] = _onehot(mode, b["k"])

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        rows = len(data)
        out = np.zeros((rows, self.total_dims), dtype=np.float32)
        for b in self.blocks:
            if b["type"] == "num":
                self._encode_numeric_block(b, data[b["col"]].astype(float).to_numpy(), out)
            else:
                idx = {c: k for k, c in enumerate(b["categories"])}
                start, end = b["start"], b["end"]
                for j, raw in enumerate(data[b["col"]].astype(str)):
                    out[j, start + idx.get(raw, 0)] = 1.0
        return out

    def inverse(self, X: np.ndarray) -> pd.DataFrame:
        X = np.asarray(X, dtype=np.float32)
        frame: dict[str, np.ndarray] = {}
        for b in self.blocks:
            if b["type"] == "num":
                if b.get("kind") == "uniform":
                    u = norm.cdf(X[:, b["val"]])
                    if b.get("ecdf") is not None:
                        frame[b["col"]] = b["ecdf"].quantile(u)
                        continue
                    f = np.clip(u * (b["n"] - 1), 0.0, b["n"] - 1)
                    i = np.floor(f).astype(int)
                    i = np.clip(i, 0, b["n"] - 2)
                    t = (f - i)[:, None]
                    lo = b["values"][i][:, None]
                    hi = b["values"][i + 1][:, None]
                    frame[b["col"]] = (lo * (1.0 - t) + hi * t).ravel()
                    continue
                mode = np.argmax(X[:, b["modes"][0]:b["modes"][1]], axis=1)
                means = np.asarray(b["means"]); stds = np.asarray(b["stds"])
                values = X[:, b["val"]] * stds[mode] + means[mode]
                frame[b["col"]] = np.clip(values, b["min"], b["max"])
            else:
                idx = np.argmax(X[:, b["start"]:b["end"]], axis=1)
                frame[b["col"]] = np.asarray(b["categories"])[idx]
        return pd.DataFrame(frame, columns=self.columns)

    def rectify(self, X: np.ndarray) -> np.ndarray:
        """Rectifica los vecores numericos continuos a marginales uniformes.

        Para cada bloque ``uniform`` sustituye el valor por su rango percentil
        dentro de la propia muestra ``(rank-0.5)/n`` pasado por ``Phi^-1``.
        Como es una transformacion monotona por columna, la copula muestral
        (correlaciones de rango/estructura de dependencia) se conserva intacta
        mientras que el marginal de cada columna queda exactamente uniforme: el
        inverso devuelve entonces los cuantiles empiricos reales. Util para
        corregir el sesgo marginal del generador sin tocar la estructura
        conjunta aprendida. Aplica sobre numericas ``uniform``; el resto del
        vector no se modifica.
        """
        X = np.asarray(X, dtype=np.float32)
        for b in self.blocks:
            if b["type"] != "num" or b.get("kind") != "uniform":
                continue
            v = X[:, b["val"]]
            order = np.argsort(np.argsort(v))
            ranks = np.clip((order + 0.5) / len(v), 1e-6, 1.0 - 1e-6)
            X[:, b["val"]] = np.clip(norm.ppf(ranks), -self.clip_value, self.clip_value)
        return X

    # ------------------------------------------------------------------
    def condition_vectors(self, data: pd.DataFrame) -> np.ndarray | None:
        """One-hot de la clase de cada fila real (para el discriminador)."""
        if self.n_cond == 0:
            return None
        b = self._cond_block
        cats = b["categories"]
        idx = {c: k for k, c in enumerate(cats)}
        out = np.zeros((len(data), self.n_cond), dtype=np.float32)
        for j, raw in enumerate(data[b["col"]].astype(str)):
            out[j, idx.get(raw, 0)] = 1.0
        return out

    def sample_conditions(self, n: int, rng: np.random.Generator) -> np.ndarray | None:
        """One-hot de clases para generar ``n`` filas, con la frecuencia empirica."""
        if self.n_cond == 0 or self._cond_freq is None:
            return None
        c = rng.choice(self.n_cond, size=n, p=self._cond_freq)
        out = np.zeros((n, self.n_cond), dtype=np.float32)
        out[np.arange(n), c] = 1.0
        return out


def _onehot(idx: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(idx), k), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out