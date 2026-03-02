"""Model definitions and hyperparameter spaces for Optuna tuning.

Models: LightGBM, XGBoost, CatBoost, Random Forest, HistGradientBoosting,
ElasticNet, TabNet (optional).
"""

from __future__ import annotations

import logging
from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Optuna objective factories
# ---------------------------------------------------------------------------
def _cv_score(model: Any, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """Compute mean R2 across k-fold CV."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in kf.split(X):
        model_copy = _clone_model(model)
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if hasattr(model_copy, "fit"):
            if isinstance(model_copy, lgb.LGBMRegressor):
                model_copy.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            elif isinstance(model_copy, xgb.XGBRegressor):
                model_copy.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif isinstance(model_copy, cb.CatBoostRegressor):
                model_copy.fit(
                    X_tr, y_tr,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                )
            else:
                model_copy.fit(X_tr, y_tr)

        scores.append(r2_score(y_val, model_copy.predict(X_val)))

    return float(np.mean(scores))


def _clone_model(model: Any) -> Any:
    """Clone a model instance with the same parameters."""
    from sklearn.base import clone

    try:
        return clone(model)
    except Exception:
        # For models that don't support sklearn clone
        return model.__class__(**model.get_params())


def objective_lgb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for LightGBM."""
    model = lgb.LGBMRegressor(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    return _cv_score(model, X, y)


def objective_xgb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for XGBoost."""
    model = xgb.XGBRegressor(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 100),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    return _cv_score(model, X, y)


def objective_cb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for CatBoost."""
    model = cb.CatBoostRegressor(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        depth=trial.suggest_int("depth", 3, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        iterations=trial.suggest_int("iterations", 100, 500),
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    return _cv_score(model, X, y)


def objective_rf(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for Random Forest."""
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return _cv_score(model, X, y)


def objective_hgb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for HistGradientBoosting."""
    model = HistGradientBoostingRegressor(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 15),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        max_bins=trial.suggest_int("max_bins", 128, 255),
        random_state=RANDOM_STATE,
    )
    return _cv_score(model, X, y)


def objective_elastic(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for ElasticNet."""
    model = ElasticNet(
        alpha=trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    return _cv_score(model, X, y)


# ---------------------------------------------------------------------------
# Model training with best params
# ---------------------------------------------------------------------------
def train_lgb(X: np.ndarray, y: np.ndarray, params: dict) -> lgb.LGBMRegressor:
    """Train LightGBM with given hyperparameters."""
    model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    model.fit(X, y)
    return model


def train_xgb(X: np.ndarray, y: np.ndarray, params: dict) -> xgb.XGBRegressor:
    """Train XGBoost with given hyperparameters."""
    clean = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**clean, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    model.fit(X, y)
    return model


def train_cb(X: np.ndarray, y: np.ndarray, params: dict) -> cb.CatBoostRegressor:
    """Train CatBoost with given hyperparameters."""
    model = cb.CatBoostRegressor(**params, random_seed=RANDOM_STATE, verbose=0)
    model.fit(X, y)
    return model


def train_rf(X: np.ndarray, y: np.ndarray, params: dict) -> RandomForestRegressor:
    """Train Random Forest with given hyperparameters."""
    model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X, y)
    return model


def train_hgb(X: np.ndarray, y: np.ndarray, params: dict) -> HistGradientBoostingRegressor:
    """Train HistGradientBoosting with given hyperparameters."""
    model = HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


def train_elastic(X: np.ndarray, y: np.ndarray, params: dict) -> ElasticNet:
    """Train ElasticNet with given hyperparameters."""
    model = ElasticNet(**params, max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model
