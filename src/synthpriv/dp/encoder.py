"""Encoder tabular hacia un espacio numerico uniforme (y su inverso).

Numericas: z-score manual (std=1 cuando es 0) para evitar NaNs.
Categoricas: one-hot con las categorias vistas en ``fit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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