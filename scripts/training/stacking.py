"""Stacking ensemble: Level-0 base models + Level-1 Ridge meta-learner.

Uses out-of-fold predictions from base models to train a meta-learner,
preventing overfitting that occurs with simple averaging.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def generate_oof_predictions(
    models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """Generate out-of-fold predictions for each base model.

    Parameters
    ----------
    models:
        Dict of {name: (train_func, best_params)} for each base model.
    X_train, y_train:
        Training data.
    n_splits:
        Number of CV folds.

    Returns
    -------
    oof_preds: np.ndarray of shape (n_samples, n_models)
    """
    n_models = len(models)
    oof_preds = np.zeros((len(X_train), n_models))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        logger.info(f"  OOF fold {fold_idx + 1}/{n_splits}")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]

        for model_idx, (name, (train_func, params)) in enumerate(models.items()):
            model = train_func(X_tr, y_tr, params)
            oof_preds[val_idx, model_idx] = model.predict(X_val)

    return oof_preds


def train_meta_learner(
    oof_preds: np.ndarray,
    y_train: np.ndarray,
) -> RidgeCV:
    """Train a Ridge meta-learner on out-of-fold predictions.

    Uses RidgeCV with built-in cross-validation to select alpha.
    """
    meta = RidgeCV(alphas=np.logspace(-3, 3, 50))
    meta.fit(oof_preds, y_train)

    logger.info(f"  Meta-learner alpha: {meta.alpha_:.6f}")
    logger.info(f"  Meta-learner weights (coef): {meta.coef_}")
    logger.info(f"  Meta-learner intercept: {meta.intercept_:.6f}")

    return meta


def predict_stacking(
    base_models: list[Any],
    meta_learner: RidgeCV,
    X: np.ndarray,
) -> np.ndarray:
    """Generate stacking ensemble predictions.

    Parameters
    ----------
    base_models:
        List of trained base models (same order as OOF training).
    meta_learner:
        Trained Ridge meta-learner.
    X:
        Feature matrix.

    Returns
    -------
    Stacked predictions.
    """
    base_preds = np.column_stack([m.predict(X) for m in base_models])
    return meta_learner.predict(base_preds)


def train_stacking_ensemble(
    model_configs: dict[str, tuple],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> tuple[list[Any], RidgeCV]:
    """Full stacking pipeline: OOF predictions → meta-learner → final models.

    Parameters
    ----------
    model_configs:
        Dict of {name: (train_func, best_params)}.
    X_train, y_train:
        Training data.
    n_splits:
        CV folds for OOF generation.

    Returns
    -------
    (trained_base_models, meta_learner)
    """
    logger.info("Generating out-of-fold predictions...")
    oof_preds = generate_oof_predictions(model_configs, X_train, y_train, n_splits)

    logger.info("Training meta-learner on OOF predictions...")
    meta = train_meta_learner(oof_preds, y_train)

    logger.info("Training final base models on full training data...")
    trained_models = []
    for name, (train_func, params) in model_configs.items():
        logger.info(f"  Training {name}...")
        model = train_func(X_train, y_train, params)
        trained_models.append(model)

    return trained_models, meta
