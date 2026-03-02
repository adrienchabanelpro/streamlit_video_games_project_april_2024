"""Model evaluation: metrics, residual analysis, cross-validation reports."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_transform: bool = True,
) -> dict[str, float]:
    """Compute full set of regression metrics.

    If log_transform is True, y_pred is in log-space and y_true is raw.
    Inverse-transforms before computing metrics.
    """
    if log_transform:
        y_pred = np.expm1(y_pred)

    # Clip negative predictions to 0 (sales can't be negative)
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

    # MAPE (only where y_true > 0 to avoid division by zero)
    mask = y_true > 0
    if mask.sum() > 0:
        metrics["mape"] = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))
    else:
        metrics["mape"] = float("nan")

    return metrics


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    global_mean: float,
    log_transform: bool = True,
) -> dict[str, float]:
    """Evaluate a single model and compare against baseline."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, log_transform)

    # Baseline: always predict training mean
    y_baseline = np.full_like(y_test, global_mean)
    baseline_metrics = compute_metrics(y_test, y_baseline, log_transform=False)
    for k, v in baseline_metrics.items():
        metrics[f"baseline_{k}"] = v

    return metrics


def evaluate_stacking(
    base_models: list[Any],
    meta_learner: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    global_mean: float,
    log_transform: bool = True,
) -> dict[str, float]:
    """Evaluate stacking ensemble."""
    from scripts.training.stacking import predict_stacking

    y_pred = predict_stacking(base_models, meta_learner, X_test)
    metrics = compute_metrics(y_test, y_pred, log_transform)

    # Also compute simple average for comparison
    simple_avg = np.mean([m.predict(X_test) for m in base_models], axis=0)
    simple_metrics = compute_metrics(y_test, simple_avg, log_transform)
    metrics["simple_avg_r2"] = simple_metrics["r2"]
    metrics["simple_avg_rmse"] = simple_metrics["rmse"]

    # Baseline
    y_baseline = np.full_like(y_test, global_mean)
    baseline_metrics = compute_metrics(y_test, y_baseline, log_transform=False)
    metrics["baseline_r2"] = baseline_metrics["r2"]
    metrics["baseline_rmse"] = baseline_metrics["rmse"]

    # Per-model disagreement as uncertainty proxy
    preds_all = np.column_stack([m.predict(X_test) for m in base_models])
    if log_transform:
        preds_all = np.expm1(preds_all)
    metrics["pred_std_mean"] = float(preds_all.std(axis=1).mean())
    metrics["pred_std_median"] = float(np.median(preds_all.std(axis=1)))

    return metrics


def print_metrics_table(
    all_metrics: dict[str, dict[str, float]],
) -> None:
    """Pretty-print a comparison table of model metrics."""
    header = f"{'Model':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, m in all_metrics.items():
        mape_str = f"{m.get('mape', float('nan')):.4f}" if not np.isnan(m.get("mape", float("nan"))) else "N/A"
        logger.info(
            f"{name:<20} {m['r2']:>8.4f} {m['rmse']:>8.4f} {m['mae']:>8.4f} {mape_str:>8}"
        )
