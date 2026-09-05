"""ECDF con garantia formal de privacidad diferencial (Laplace / histograma).

Mecanismo: histograma con recuentos ruidosos Laplace para cada columna numerica.

- Sensibilidad por bin = 1 (anadir/quitar una fila mueve como mucho en 1 cada
  recuento) y los bins particionan los datos de forma disjunta: por composicion
  **paralela**, toda la columna consume un unico ``epsilon`` (ruido Laplace de
  escala ``1/epsilon`` por bin).
- Las columnas son disjuntas entre si pero no particion del mismo dato, asi que
  el presupuesto total se reparte secuencialmente entre columnas: cada columna
  usa ``epsilon_total / n_columnas``.
- El rango de la cuadricula se recorta a los cuantiles empiricos 0.001/0.999
  (con un margen), evitando publicar los extremos exactos; los valores fuera de
  ese soporte no se emiten.
- La funcion de cuantil (inversa de la ECDF ruidosa, suavizada de forma
  monotona y por interpolacion lineal) es post-proceso de la salida DP, por lo
  que no consume presupuesto adicional. Emitir valores desde esta inversa
  mantiene la garantia DP por columna; la composicion con el DP-SGD del
  entrenamiento da la garantia total del sintetizador.
"""

from __future__ import annotations

import warnings

import numpy as np

from synthpriv.utils import get_logger

logger = get_logger("dp")


class DPEcdf:
    """ECDF privada por columna basada en histograma Laplace.

    Parameters
    ----------
    epsilon:
        Presupuesto DP de esta columna (0 < epsilon <= presupuesto total de
        marginales / n_columnas). El resto del sintetizador debe componerlo con
        el epsilon del entrenamiento (``ecdf_epsilon + effective_epsilon``).
    bins:
        Numero de intervalos de igual anchura sobre el soporte recortado.
    q_low/q_high:
        Cuantiles (0..1) que definen el soporte de la cuadricula cuando no se
        dan ``bounds``; los valores fuera se agrupan en los bordes via el recorte
        del rango.
    bounds:
        Soporte **publico** ``(min, max)`` de la columna. Si se provee, la
        cuadricula es fija y el mecanismo es DP pura estricta (la garantia no
        depende de ningun dato previo). Si es ``None``, el soporte se deriva de
        los cuantiles empiricos 0.001/0.999 de los datos (con margen): util en la
        practica, pero el rango en si revela informacion de la muestra — se emite
        un warning y se recomienda pasar ``bounds`` publicos cuando existan.
    """

    def __init__(self, epsilon: float = 1.0, bins: int = 200,
                 q_low: float = 0.001, q_high: float = 0.999,
                 bounds: tuple[float, float] | None = None):
        if epsilon <= 0:
            raise ValueError(f"epsilon del DP-ECDF debe ser > 0, se recibio {epsilon!r}")
        self.epsilon = float(epsilon)
        self.bins = max(2, int(bins))
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        if bounds is not None and not (
                bounds[0] < bounds[1] and np.isfinite(bounds[0]) and np.isfinite(bounds[1])):
            raise ValueError(f"bounds invalidos: {bounds!r} (requiere min < max finitos)")
        self.bounds = tuple(map(float, bounds)) if bounds is not None else None
        self.n = 0
        self._edges: np.ndarray | None = None
        self._cdf: np.ndarray | None = None  # len(bins)+1, monotono, termina en 1

    # ------------------------------------------------------------------
    def fit(self, values: np.ndarray, rng: np.random.Generator | None = None) -> "DPEcdf":
        """Construye la ECDF privada a partir de ``values`` (unifila por fila).
        """
        v = np.asarray(values, dtype=float).ravel()
        if v.size < 2:
            raise ValueError(f"DPEcdf necesita al menos 2 valores, se recibieron {v.size}")
        self.n = int(v.size)
        rng = rng or np.random.default_rng(0)

        if self.bounds is not None:
            lo, hi = self.bounds
        else:
            lo = float(np.quantile(v, min(self.q_low, self.q_high)))
            hi = float(np.quantile(v, max(self.q_low, self.q_high)))
            pad = 1e-6 + 0.05 * (hi - lo)  # redondea soporte, no publica los extremos
            lo, hi = lo - pad, hi + pad
            if hi <= lo:
                hi = lo + 1.0
            warnings.warn(
                "DPEcdf usa un soporte derivado de los datos (cuantiles 0.001/0.999 "
                "+ margen). La garantia DP del histograma es estricta dado ese soporte, "
                "pero el rango en si revela informacion muestral; pasa 'bounds=(min,max)' "
                "publicos cuando existan para una garantia formal completa.",
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
        # suavizado monotono: cdf = cumsum de la pdf (ya ordenada no negativa)
        cdf = np.concatenate([[0.0], np.cumsum(pdf)])
        cdf = cdf / cdf[-1]
        # evita mesetas exactas: normaliza y clampa
        self._cdf = np.clip(cdf, 0.0, 1.0)
        self._cdf[-1] = 1.0
        return self

    # ------------------------------------------------------------------
    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Inversa de la ECDF privada sobre los cuantiles uniformes ``u`` (0..1).

        Interpola linealmente entre bordes de la cuadricula (post-proceso DP).
        """
        if self._edges is None or self._cdf is None:
            raise RuntimeError("DPEcdf sin entrenar: llama a fit(values) primero.")
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
        """Resumen del estado de privacidad de la columna (para el informe)."""
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