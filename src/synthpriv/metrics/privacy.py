"""Privacy metrics: re-identification and inference risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from synthpriv.metrics.base import MetricResult, evaluate_status


def _encode_mixed(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Encode real and synth into a uniform numeric space (one-hot + scaling)."""
    union_cols = [c for c in real.columns if c in synth.columns]
    combined = pd.concat([real[union_cols], synth[union_cols]], axis=0)
    dummies = pd.get_dummies(combined)
    num_idx = dummies.select_dtypes(include=[np.number]).columns
    encoded = dummies[num_idx].astype(float).to_numpy()
    return encoded[: len(real)], encoded[len(real):]


def nndr(real: pd.DataFrame, synth: pd.DataFrame, threshold: float = 0.8, sample: int = 2000) -> MetricResult:
    """Nearest Neighbor Distance Ratio.

    For each synthetic row ``i``: ratio = d(nearest real) / d(nearest synthetic excluding itself).
    Ratios << 1 imply near-duplicates of real records among the synthetic ones (re-identification
    risk). Values >= 1 indicate the synthetic points are not stuck to the real ones.
    """
    Xr, Xs = _encode_mixed(real, synth)
    if len(Xs) > sample:  # bound the cost of kNN on large datasets
        rng = np.random.default_rng(0)
        Xs = Xs[rng.choice(len(Xs), size=sample, replace=False)]
    if len(Xs) < 2:
        return MetricResult(name="nndr", status="error",
                            message="Need >=2 synthetic rows for NNDR.")

    nn_real = NearestNeighbors(n_neighbors=1).fit(Xr)
    nn_synth = NearestNeighbors(n_neighbors=2).fit(Xs)
    d_real, _ = nn_real.kneighbors(Xs, 1)
    d_synth, _ = nn_synth.kneighbors(Xs, 2)
    d_real, d_synth = d_real[:, 0], d_synth[:, -1]  # 2nd neighbor excludes the row itself

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(d_synth > 0, d_real / np.where(d_synth > 0, d_synth, np.nan), np.nan)
    ratios = ratios[~np.isnan(ratios)]
    if ratios.size == 0:
        return MetricResult(name="nndr", status="error", message="No valid distances for NNDR.")

    value = float(np.mean(ratios))
    status, msg = evaluate_status(value, threshold, "higher_is_better")
    return MetricResult(
        name="nndr",
        description="Mean d(real_knn)/d(synth_knn); >= threshold suggests low copy risk",
        value=round(value, 4),
        threshold=threshold,
        direction="higher_is_better",
        status=status,
        message=msg,
        details={
            "min_ratio": round(float(np.min(ratios)), 4),
            "pct_below_1": round(float(np.mean(ratios < 1.0)), 4),
            "synthetic_checked": int(ratios.size),
        },
    )


def mia_auc(real: pd.DataFrame, synth: pd.DataFrame, threshold: float = 0.7, folds: int = 5) -> MetricResult:
    """Membership inference attack.

    Trains a classifier to distinguish real from synthetic rows
    (cross-validation). AUC ~0.5 = indistinguishable (good); high AUC = the
    synthetic rows are distinguishable and an attacker could infer membership.
    """
    Xr, Xs = _encode_mixed(real, synth)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])
    if len(np.unique(y)) < 2 or len(y) < folds * 4:
        return MetricResult(name="mia_auc", status="reported",
                            message="Insufficient data for the MIA attack.")

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    aucs, bas = [], []
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=0).split(X, y):
        pipeline.fit(X[tr], y[tr])
        pred = pipeline.predict_proba(X[te])[:, 1]
        y_te = y[te]
        if len(np.unique(pred.round())) > 1 and len(np.unique(y_te)) == 2:
            aucs.append(roc_auc_score(y_te, pred))
        bas.append(balanced_accuracy_score(y_te, pipeline.predict(X[te])))

    value = float(np.mean(aucs)) if aucs else np.nan
    status, msg = evaluate_status(value, threshold, "lower_is_better")
    return MetricResult(
        name="mia_auc",
        description="Membership attack AUC; ~0.5 = good privacy, >0.7 = distinguishable",
        value=round(value, 4) if not np.isnan(value) else None,
        threshold=threshold,
        direction="lower_is_better",
        status=status,
        message=msg,
        details={
            "balanced_accuracy": round(float(np.mean(bas)), 4),
            "real_rows": len(real),
            "synthetic_rows": len(synth),
        },
    )


# ---------------------------------------------------------------------------
# anonymeter attacks (discovery, inference, linkability)
# ---------------------------------------------------------------------------

def _anonymeter_guard(exc: Exception, name: str) -> MetricResult:
    return MetricResult(name=name, status="error",
                        message=f"The anonymeter evaluator could not complete: {exc}")


def anonymeter_discovery(real: pd.DataFrame, synth: pd.DataFrame,
                         n_attacks: int = 5, threshold: float = 0.1) -> MetricResult:
    """Univariate discovery ratio: % of real rows "recovered" in the synthetic ones."""
    try:
        from anonymeter.evaluators import UnivariateDiscoveryEvaluator
    except Exception as exc:  # pragma: no cover - dependencia opcional
        return _anonymeter_guard(exc, "anonymeter_discovery")

    try:
        df_real = real.astype(str)
        df_synth = synth[list(real.columns) if real.columns.isin(synth.columns).all() else real.columns].astype(str)
        evaluator = UnivariateDiscoveryEvaluator(
            ori=df_real, sim=df_synth, n_attacks=n_attacks, n_neighbors=3,
        )
        evaluator.evaluate()
        finding = evaluator.result()
        rate = getattr(finding, "rate", None)
        if rate is None or (isinstance(rate, float) and np.isnan(rate)):
            return MetricResult(name="anonymeter_discovery", status="reported",
                                message="The evaluator found no sufficient matches (success_rate NaN).")
        status, msg = evaluate_status(float(rate), threshold, "lower_is_better")
        return MetricResult(
            name="anonymeter_discovery",
            description="% of real rows recoverable in the synthetic set (lower = better)",
            value=round(float(rate), 4),
            threshold=threshold,
            direction="lower_is_better",
            status=status,
            message=msg,
            details={"control_attack": round(float(getattr(finding, "control", np.nan)), 4)},
        )
    except Exception as exc:
        return _anonymeter_guard(exc, "anonymeter_discovery")


def anonymeter_inference(real: pd.DataFrame, synth: pd.DataFrame,
                         n_attacks: int = 3, threshold: float = 0.1) -> MetricResult:
    """Sensitive-attribute inference attack from auxiliary attributes."""
    try:
        from anonymeter.evaluators import InferenceEvaluator
    except Exception as exc:  # pragma: no cover
        return _anonymeter_guard(exc, "anonymeter_inference")

    cols = list(real.columns)
    if len(cols) < 3:
        return MetricResult(name="anonymeter_inference", status="reported",
                            message="Need >=3 columns (auxiliaries + secret) for the attack.")
    aux, secret = cols[:2], [cols[2]]
    try:
        evaluator = InferenceEvaluator(
            ori=real.astype(str), sim=synth.astype(str),
            aux_cols=aux, secret_cols=secret, n_attacks=n_attacks,
        )
        evaluator.evaluate()
        finding = evaluator.result()
        rate = getattr(finding, "rate", None)
        if rate is None or (isinstance(rate, float) and np.isnan(rate)):
            return MetricResult(name="anonymeter_inference", status="reported",
                                message="Success rate NaN: insufficient data for inference.")
        status, msg = evaluate_status(float(rate), threshold, "lower_is_better")
        return MetricResult(
            name="anonymeter_inference",
            description="Success inferring the secret attribute from auxiliaries (lower = better)",
            value=round(float(rate), 4),
            threshold=threshold,
            direction="lower_is_better",
            status=status,
            message=msg,
            details={"aux_cols": aux, "secret_cols": secret},
        )
    except Exception as exc:
        return _anonymeter_guard(exc, "anonymeter_inference")


def anonymeter_linkability(real: pd.DataFrame, synth: pd.DataFrame,
                           n_attacks: int = 3, threshold: float = 0.1) -> MetricResult:
    """Linkability attack: join two attribute halves to re-identify."""
    try:
        from anonymeter.evaluators import LinkabilityEvaluator
    except Exception as exc:  # pragma: no cover
        return _anonymeter_guard(exc, "anonymeter_linkability")

    cols = list(real.columns)
    if len(cols) < 2:
        return MetricResult(name="anonymeter_linkability", status="reported",
                            message="Need >=2 columns to split auxiliary attributes.")
    aux = (cols[: len(cols) // 2], cols[len(cols) // 2:])
    try:
        evaluator = LinkabilityEvaluator(
            ori=real.astype(str)[aux[0] + aux[1]][cols], sim=synth.astype(str)[cols],
            n_attacks=n_attacks, aux_cols=aux,
        )
        evaluator.evaluate()
        finding = evaluator.result()
        rate = getattr(finding, "rate", None)
        if rate is None or (isinstance(rate, float) and np.isnan(rate)):
            return MetricResult(name="anonymeter_linkability", status="reported",
                                message="Success rate NaN: insufficient data for linkability.")
        status, msg = evaluate_status(float(rate), threshold, "lower_is_better")
        return MetricResult(
            name="anonymeter_linkability",
            description="Success linking halves of records (lower = better)",
            value=round(float(rate), 4),
            threshold=threshold,
            direction="lower_is_better",
            status=status,
            message=msg,
            details={"aux_split": [list(a) for a in aux]},
        )
    except Exception as exc:
        return _anonymeter_guard(exc, "anonymeter_linkability")