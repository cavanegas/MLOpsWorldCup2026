"""Train the match-outcome classifier with MLflow experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # xgboost is optional at import time — mlflow.log_model still works with sklearn.
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - environment without xgboost compiled wheels
    _HAS_XGB = False

from ..config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, PATHS
from ..features.build_features import FEATURE_COLUMNS, Dataset
from ..logging_utils import get_logger

log = get_logger(__name__)

CLASS_NAMES = ("home_win", "draw", "away_win")


@dataclass
class TrainedModel:
    estimator: Pipeline
    metrics: dict[str, float]
    params: dict[str, Any]
    model_name: str
    run_id: str | None


def _time_split(ds: Dataset, test_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays for train/test based on chronological order."""
    order = ds.meta["date"].argsort()
    n = len(order)
    cutoff = int(n * (1 - test_fraction))
    train_idx = order.iloc[:cutoff].to_numpy()
    test_idx = order.iloc[cutoff:].to_numpy()
    return train_idx, test_idx


def _build_estimator(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial", **params)
    elif model_name == "rf":
        clf = RandomForestClassifier(n_jobs=-1, **params)
    elif model_name == "gbt":
        clf = GradientBoostingClassifier(**params)
    elif model_name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed — choose 'logreg' or 'gbt'")
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            **params,
        )
    else:
        raise ValueError(f"unknown model_name: {model_name}")
    return Pipeline([("scaler", StandardScaler(with_mean=True)), ("clf", clf)])


def _cross_validate(estimator_builder, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict[str, float]:
    """Time-series cross-validation. Returns averaged metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, f1s, lls = [], [], []
    for fold, (tr, va) in enumerate(tscv.split(X)):
        est = estimator_builder()
        est.fit(X.iloc[tr], y.iloc[tr])
        pr = est.predict_proba(X.iloc[va])
        pred = pr.argmax(axis=1)
        accs.append(accuracy_score(y.iloc[va], pred))
        f1s.append(f1_score(y.iloc[va], pred, average="macro"))
        lls.append(log_loss(y.iloc[va], pr, labels=[0, 1, 2]))
        log.info("  cv fold %d  acc=%.3f  f1=%.3f  ll=%.3f", fold + 1, accs[-1], f1s[-1], lls[-1])
    return {
        "cv_accuracy": float(np.mean(accs)),
        "cv_f1_macro": float(np.mean(f1s)),
        "cv_log_loss": float(np.mean(lls)),
        "cv_log_loss_std": float(np.std(lls)),
    }


def train_model(
    ds: Dataset,
    model_name: str = "gbt",
    params: dict[str, Any] | None = None,
    test_fraction: float = 0.15,
    register: bool = True,
    cv_splits: int = 5,
    promote_to_stage: str | None = None,
) -> TrainedModel:
    """Train, evaluate with time-series CV, log to MLflow and optionally promote.

    ``promote_to_stage`` can be ``"Staging"`` or ``"Production"`` — when set and
    the MLflow registry is reachable, the freshly-trained model version is
    transitioned to that stage.
    """
    params = params or _default_params(model_name)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_idx, test_idx = _time_split(ds, test_fraction=test_fraction)
    X_train, X_test = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
    y_train, y_test = ds.y.iloc[train_idx], ds.y.iloc[test_idx]
    log.info("train rows=%d  test rows=%d", len(X_train), len(X_test))

    estimator = _build_estimator(model_name, params)

    with mlflow.start_run(run_name=f"train_{model_name}") as run:
        mlflow.log_params({f"model.{k}": v for k, v in params.items()})
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_fraction", test_fraction)
        mlflow.log_param("cv_splits", cv_splits)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        log.info("running %d-fold TimeSeries CV on %s", cv_splits, model_name)
        cv_metrics = _cross_validate(
            lambda: _build_estimator(model_name, params), X_train, y_train, n_splits=cv_splits
        )
        mlflow.log_metrics(cv_metrics)

        estimator.fit(X_train, y_train)
        proba = estimator.predict_proba(X_test)
        preds = proba.argmax(axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
            "log_loss": float(log_loss(y_test, proba, labels=[0, 1, 2])),
        }
        metrics.update(cv_metrics)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k not in cv_metrics})

        ds_path = PATHS.processed / "training_dataset.parquet"
        if ds_path.exists():
            mlflow.log_artifact(ds_path.as_posix(), artifact_path="dataset")

        model_info = mlflow.sklearn.log_model(
            estimator,
            artifact_path="model",
            registered_model_name="wc2026_match_classifier" if register else None,
        )

        if register and promote_to_stage:
            try:
                client = mlflow.tracking.MlflowClient()
                # The latest version is the one we just logged.
                latest = client.get_latest_versions("wc2026_match_classifier", stages=["None"])
                if latest:
                    client.transition_model_version_stage(
                        name="wc2026_match_classifier",
                        version=latest[0].version,
                        stage=promote_to_stage,
                        archive_existing_versions=(promote_to_stage == "Production"),
                    )
                    client.set_model_version_tag(
                        "wc2026_match_classifier", latest[0].version, "algorithm", model_name
                    )
                    log.info("promoted v%s -> %s", latest[0].version, promote_to_stage)
            except Exception as exc:  # MLflow registry requires a server URI in some modes.
                log.warning("model staging skipped: %s", exc)

        log.info("metrics=%s", metrics)

        PATHS.models.mkdir(parents=True, exist_ok=True)
        artifact_path = PATHS.model_artifact
        joblib.dump({"pipeline": estimator, "features": list(FEATURE_COLUMNS),
                     "classes": list(CLASS_NAMES)}, artifact_path)
        mlflow.log_artifact(artifact_path.as_posix(), artifact_path="joblib")
        log.info("persisted model -> %s", artifact_path)

        return TrainedModel(
            estimator=estimator,
            metrics=metrics,
            params=params,
            model_name=model_name,
            run_id=run.info.run_id,
        )


def _default_params(model_name: str) -> dict[str, Any]:
    if model_name == "logreg":
        return {"C": 1.0}
    if model_name == "rf":
        return {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 3, "random_state": 42}
    if model_name == "gbt":
        return {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.05, "random_state": 42}
    if model_name == "xgb":
        return {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "random_state": 42}
    raise ValueError(model_name)


def predict_match_outcome(
    artifact: dict[str, Any],
    features: pd.DataFrame,
) -> np.ndarray:
    """Return an array of shape (n_rows, 3) with probabilities per class."""
    pipeline: Pipeline = artifact["pipeline"]
    cols = artifact["features"]
    return pipeline.predict_proba(features[cols])
