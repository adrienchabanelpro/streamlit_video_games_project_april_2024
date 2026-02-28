"""Training pipeline for Video Game Sales Prediction v2.

Implements:
- Temporal train/test split (no data leakage)
- Feature engineering computed on training data only
- Target encoding for Publisher (replaces 567-column one-hot)
- Optuna hyperparameter tuning with 5-fold CV (LightGBM, XGBoost, CatBoost)
- Stacking ensemble (average of 3 models)
- SHAP feature importance plots
- Full artifact saving for reproducibility

Usage:
    python scripts/train_model.py

Outputs saved to models/ and reports/:
    - reports/model_v2_optuna.txt     (LightGBM model)
    - models/model_v2_xgboost.json    (XGBoost model)
    - models/model_v2_catboost.cbm    (CatBoost model)
    - models/scaler_v2.joblib         (StandardScaler)
    - models/target_encoder_v2.joblib (Publisher target encoder)
    - models/feature_means_v2.joblib  (genre/platform means + cumulative stats)
    - reports/shap_summary.png
    - reports/shap_bar.png
    - reports/training_log.json       (params, metrics, timestamp)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import catboost as cb
import category_encoders as ce
import joblib
import lightgbm as lgb
import matplotlib
import mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "Global_Sales"
LOG_TRANSFORM = True  # Apply log1p to target before training
NUMERICAL_FEATURES = [
    "Year",
    "meta_score",
    "user_review",
    "Global_Sales_mean_genre",
    "Global_Sales_mean_platform",
    "Year_Global_Sales_mean_genre",
    "Year_Global_Sales_mean_platform",
    "Cumulative_Sales_Genre",
    "Cumulative_Sales_Platform",
    "Publisher_encoded",
]


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV, drop leakage columns, clean NaN and types."""
    df = pd.read_csv(csv_path)

    # Drop rows with missing Publisher or Year
    df = df.dropna(subset=["Publisher", "Year"])

    # Convert Year to int
    df["Year"] = df["Year"].astype(int)

    # Drop regional sales columns (they compose Global_Sales = leakage)
    regional_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    df = df.drop(columns=[c for c in regional_cols if c in df.columns])

    # Drop non-feature columns
    df = df.drop(columns=["Rank", "Name"], errors="ignore")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # Fill missing meta_score and user_review with median
    df["meta_score"] = df["meta_score"].fillna(df["meta_score"].median())
    df["user_review"] = df["user_review"].fillna(df["user_review"].median())

    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def temporal_train_test_split(
    df: pd.DataFrame, split_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train <= split_year, test > split_year."""
    df_train = df[df["Year"] <= split_year].copy()
    df_test = df[df["Year"] > split_year].copy()
    return df_train, df_test


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_train_stats(df_train: pd.DataFrame) -> dict:
    """Compute feature engineering statistics from training data only.

    These stats are saved and reused at inference time so that no
    test/future data ever leaks into feature values.
    """
    stats: dict = {}

    # Mean Global_Sales by Genre
    stats["genre_means"] = df_train.groupby("Genre")[TARGET].mean().to_dict()

    # Mean Global_Sales by Platform
    stats["platform_means"] = df_train.groupby("Platform")[TARGET].mean().to_dict()

    # Cumulative sales by Genre and Year
    stats["cumsum_genre"] = {}
    for genre in df_train["Genre"].unique():
        genre_data = (
            df_train[df_train["Genre"] == genre].groupby("Year")[TARGET].sum().sort_index().cumsum()
        )
        stats["cumsum_genre"][genre] = genre_data.to_dict()

    # Cumulative sales by Platform and Year
    stats["cumsum_platform"] = {}
    for platform in df_train["Platform"].unique():
        platform_data = (
            df_train[df_train["Platform"] == platform]
            .groupby("Year")[TARGET]
            .sum()
            .sort_index()
            .cumsum()
        )
        stats["cumsum_platform"][platform] = platform_data.to_dict()

    # Unique values for prediction page dropdowns
    stats["publishers"] = sorted(df_train["Publisher"].unique().tolist())
    stats["genres"] = sorted(df_train["Genre"].unique().tolist())
    stats["platforms"] = sorted(df_train["Platform"].unique().tolist())

    # Default values for UI inputs
    stats["meta_score_mean"] = float(df_train["meta_score"].mean())
    stats["user_review_mean"] = float(df_train["user_review"].mean())

    # Global mean of target (for baseline & target-encoding fallback)
    stats["global_sales_mean"] = float(df_train[TARGET].mean())

    return stats


def _lookup_cumulative(cumsum_dict: dict, category: str, year: int) -> float:
    """Look up cumulative sales for *category* up to *year*."""
    if category not in cumsum_dict:
        return 0.0
    yearly = cumsum_dict[category]
    relevant_years = [y for y in yearly if y <= year]
    if not relevant_years:
        return 0.0
    return yearly[max(relevant_years)]


def compute_engineered_features(df: pd.DataFrame, train_stats: dict) -> pd.DataFrame:
    """Apply feature engineering using pre-computed training statistics."""
    df = df.copy()

    # Mean sales by genre / platform (from training data)
    df["Global_Sales_mean_genre"] = (
        df["Genre"].map(train_stats["genre_means"]).fillna(train_stats["global_sales_mean"])
    )
    df["Global_Sales_mean_platform"] = (
        df["Platform"].map(train_stats["platform_means"]).fillna(train_stats["global_sales_mean"])
    )

    # Interaction features
    df["Year_Global_Sales_mean_genre"] = df["Year"] * df["Global_Sales_mean_genre"]
    df["Year_Global_Sales_mean_platform"] = df["Year"] * df["Global_Sales_mean_platform"]

    # Cumulative sales
    df["Cumulative_Sales_Genre"] = df.apply(
        lambda row: _lookup_cumulative(train_stats["cumsum_genre"], row["Genre"], row["Year"]),
        axis=1,
    )
    df["Cumulative_Sales_Platform"] = df.apply(
        lambda row: _lookup_cumulative(
            train_stats["cumsum_platform"], row["Platform"], row["Year"]
        ),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """Optuna objective: temporal split + 5-fold CV on train set."""
    split_year = trial.suggest_categorical("split_year", [2013, 2014, 2015])

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }

    # Temporal split
    df_train, _ = temporal_train_test_split(df, split_year)
    if len(df_train) < 100:
        return float("-inf")

    # Feature engineering (train stats only)
    train_stats = compute_train_stats(df_train)
    df_train = compute_engineered_features(df_train, train_stats)

    # Target-encode Publisher
    encoder = ce.TargetEncoder(cols=["Publisher"], smoothing=10)
    df_train["Publisher_encoded"] = encoder.fit_transform(
        df_train[["Publisher"]], df_train[TARGET]
    )["Publisher"]

    X = df_train[NUMERICAL_FEATURES].values
    y = df_train[TARGET].values
    if LOG_TRANSFORM:
        y = np.log1p(y)

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(
            **params,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_pred = model.predict(X_val)
        scores.append(r2_score(y_val, y_pred))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Optuna objectives — XGBoost & CatBoost
# ---------------------------------------------------------------------------
def objective_xgb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for XGBoost with 5-fold CV."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []

    for train_idx, val_idx in kf.split(X):
        model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(
            X[train_idx],
            y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        scores.append(r2_score(y[val_idx], model.predict(X[val_idx])))

    return float(np.mean(scores))


def objective_cb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for CatBoost with 5-fold CV."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "iterations": trial.suggest_int("iterations", 100, 1000),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []

    for train_idx, val_idx in kf.split(X):
        model = cb.CatBoostRegressor(**params, random_seed=RANDOM_STATE, verbose=0)
        model.fit(
            X[train_idx],
            y[train_idx],
            eval_set=(X[val_idx], y[val_idx]),
            early_stopping_rounds=50,
        )
        scores.append(r2_score(y[val_idx], model.predict(X[val_idx])))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Final training
# ---------------------------------------------------------------------------
def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
) -> lgb.LGBMRegressor:
    """Train the final LightGBM model with best hyperparameters."""
    model_params = {k: v for k, v in best_params.items() if k != "split_year"}

    model = lgb.LGBMRegressor(
        **model_params,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_final_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
) -> xgb.XGBRegressor:
    """Train the final XGBoost model with best hyperparameters."""
    model = xgb.XGBRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_final_cb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
) -> cb.CatBoostRegressor:
    """Train the final CatBoost model with best hyperparameters."""
    model = cb.CatBoostRegressor(**best_params, random_seed=RANDOM_STATE, verbose=0)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _compute_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R2, MSE, RMSE, MAE for a set of predictions."""
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    global_mean: float,
) -> dict:
    """Evaluate model on test set + compare against mean-predictor baseline.

    If LOG_TRANSFORM is True, predictions are in log-space and y_test is raw,
    so we inverse-transform predictions before computing metrics.
    """
    y_pred = model.predict(X_test)
    if LOG_TRANSFORM:
        y_pred = np.expm1(y_pred)

    metrics = _compute_metrics(y_test, y_pred)

    # Baseline: always predict the training mean
    y_baseline = np.full_like(y_test, global_mean)
    metrics["baseline_r2"] = float(r2_score(y_test, y_baseline))
    metrics["baseline_mse"] = float(mean_squared_error(y_test, y_baseline))
    metrics["baseline_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_baseline)))
    metrics["baseline_mae"] = float(mean_absolute_error(y_test, y_baseline))

    return metrics


def evaluate_ensemble(
    models: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    global_mean: float,
) -> dict:
    """Evaluate average ensemble of multiple models."""
    preds = np.array([m.predict(X_test) for m in models])
    y_ensemble = preds.mean(axis=0)
    if LOG_TRANSFORM:
        y_ensemble = np.expm1(y_ensemble)

    metrics = _compute_metrics(y_test, y_ensemble)

    y_baseline = np.full_like(y_test, global_mean)
    metrics["baseline_r2"] = float(r2_score(y_test, y_baseline))
    metrics["baseline_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_baseline)))

    return metrics


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def generate_shap_plots(
    model: lgb.LGBMRegressor,
    X_test: np.ndarray,
    feature_names: list[str],
    save_dir: Path,
) -> None:
    """Generate and save SHAP summary (beeswarm) and bar plots."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Beeswarm summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (mean |SHAP value|)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"  SHAP plots saved to {save_dir}")


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------
def save_artifacts(
    lgb_model: lgb.LGBMRegressor,
    xgb_model: xgb.XGBRegressor,
    cb_model: cb.CatBoostRegressor,
    scaler: StandardScaler,
    encoder: ce.TargetEncoder,
    train_stats: dict,
    all_params: dict,
    all_metrics: dict,
) -> None:
    """Save all model artifacts for reproducibility and inference."""
    # LightGBM (native format)
    lgb_model.booster_.save_model(str(REPORTS_DIR / "model_v2_optuna.txt"))

    # XGBoost (JSON format)
    xgb_model.save_model(str(MODELS_DIR / "model_v2_xgboost.json"))

    # CatBoost (native format)
    cb_model.save_model(str(MODELS_DIR / "model_v2_catboost.cbm"))

    # Transformers
    joblib.dump(scaler, MODELS_DIR / "scaler_v2.joblib")
    joblib.dump(encoder, MODELS_DIR / "target_encoder_v2.joblib")
    joblib.dump(train_stats, MODELS_DIR / "feature_means_v2.joblib")

    # Training log
    log = {
        "timestamp": datetime.now().isoformat(),
        "best_params": all_params,
        "metrics": all_metrics,
        "features": NUMERICAL_FEATURES,
        "target": TARGET,
        "log_transform": LOG_TRANSFORM,
        "random_state": RANDOM_STATE,
    }
    with open(REPORTS_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    logger.info(f"  LightGBM  -> {REPORTS_DIR / 'model_v2_optuna.txt'}")
    logger.info(f"  XGBoost   -> {MODELS_DIR / 'model_v2_xgboost.json'}")
    logger.info(f"  CatBoost  -> {MODELS_DIR / 'model_v2_catboost.cbm'}")
    logger.info(f"  Scaler    -> {MODELS_DIR / 'scaler_v2.joblib'}")
    logger.info(f"  Encoder   -> {MODELS_DIR / 'target_encoder_v2.joblib'}")
    logger.info(f"  Stats     -> {MODELS_DIR / 'feature_means_v2.joblib'}")
    logger.info(f"  Log       -> {REPORTS_DIR / 'training_log.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _print_metrics(name: str, m: dict) -> None:
    """Pretty-print evaluation metrics for a model."""
    logger.info(f"  {name:12s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")


def main() -> None:
    """Run the full training pipeline (LightGBM + XGBoost + CatBoost ensemble)."""
    logger.info("=" * 60)
    logger.info("Video Game Sales - Training Pipeline v2 (Ensemble)")
    logger.info("=" * 60)

    # ---- 1. Load & clean ----
    logger.info("\n[1/9] Loading and cleaning data...")
    df = load_and_clean_data(DATA_DIR / "Ventes_jeux_video_final.csv")
    logger.info(f"  {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")
    logger.info(f"  Unique publishers: {df['Publisher'].nunique()}")

    # ---- 2. Optuna — LightGBM ----
    logger.info("\n[2/9] Optuna: LightGBM (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_lgb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study_lgb.optimize(
        lambda trial: objective(trial, df),
        n_trials=50,
        show_progress_bar=True,
    )

    best_lgb_params = study_lgb.best_params
    best_split_year = best_lgb_params["split_year"]
    logger.info(f"  Best CV R2: {study_lgb.best_value:.4f}")
    logger.info(f"  Best split_year: {best_split_year}")

    # ---- 3. Prepare data with best split_year ----
    logger.info(f"\n[3/9] Splitting data at year {best_split_year}...")
    df_train, df_test = temporal_train_test_split(df, best_split_year)
    logger.info(f"  Train: {len(df_train)} rows (<= {best_split_year})")
    logger.info(f"  Test:  {len(df_test)} rows (> {best_split_year})")

    # ---- 4. Feature engineering ----
    logger.info("\n[4/9] Feature engineering (train stats only)...")
    train_stats = compute_train_stats(df_train)
    df_train = compute_engineered_features(df_train, train_stats)
    df_test = compute_engineered_features(df_test, train_stats)

    logger.info("  Fitting target encoder on Publisher...")
    encoder = ce.TargetEncoder(cols=["Publisher"], smoothing=10)
    df_train["Publisher_encoded"] = encoder.fit_transform(
        df_train[["Publisher"]], df_train[TARGET]
    )["Publisher"]
    df_test["Publisher_encoded"] = encoder.transform(df_test[["Publisher"]])["Publisher"]

    logger.info("  Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[NUMERICAL_FEATURES])
    y_train_raw = df_train[TARGET].values
    X_test = scaler.transform(df_test[NUMERICAL_FEATURES])
    y_test_raw = df_test[TARGET].values

    if LOG_TRANSFORM:
        logger.info("  Applying log1p transform to target...")
        y_train = np.log1p(y_train_raw)
        y_test = y_test_raw  # Keep raw for evaluation (we'll inverse-transform preds)
    else:
        y_train = y_train_raw
        y_test = y_test_raw

    # ---- 5. Optuna — XGBoost ----
    logger.info("\n[5/9] Optuna: XGBoost (30 trials)...")
    study_xgb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 1),
    )
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train),
        n_trials=30,
        show_progress_bar=True,
    )
    best_xgb_params = study_xgb.best_params
    logger.info(f"  Best CV R2: {study_xgb.best_value:.4f}")

    # ---- 6. Optuna — CatBoost ----
    logger.info("\n[6/9] Optuna: CatBoost (30 trials)...")
    study_cb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 2),
    )
    study_cb.optimize(
        lambda trial: objective_cb(trial, X_train, y_train),
        n_trials=30,
        show_progress_bar=True,
    )
    best_cb_params = study_cb.best_params
    logger.info(f"  Best CV R2: {study_cb.best_value:.4f}")

    # ---- 7. Train final models ----
    logger.info("\n[7/9] Training final models...")
    lgb_model = train_final_model(X_train, y_train, best_lgb_params)
    logger.info("  LightGBM trained")
    xgb_model = train_final_xgb(X_train, y_train, best_xgb_params)
    logger.info("  XGBoost trained")
    cb_model = train_final_cb(X_train, y_train, best_cb_params)
    logger.info("  CatBoost trained")

    # ---- 8. Evaluate all models + ensemble ----
    logger.info("\n[8/9] Evaluating on test set...")
    gm = train_stats["global_sales_mean"]
    metrics_lgb = evaluate_model(lgb_model, X_test, y_test, gm)
    metrics_xgb = evaluate_model(xgb_model, X_test, y_test, gm)
    metrics_cb = evaluate_model(cb_model, X_test, y_test, gm)
    metrics_ens = evaluate_ensemble([lgb_model, xgb_model, cb_model], X_test, y_test, gm)

    _print_metrics("LightGBM", metrics_lgb)
    _print_metrics("XGBoost", metrics_xgb)
    _print_metrics("CatBoost", metrics_cb)
    _print_metrics("Ensemble", metrics_ens)
    logger.info(
        f"  {'Baseline':12s}  R2={metrics_lgb['baseline_r2']:.4f}  "
        f"RMSE={metrics_lgb['baseline_rmse']:.4f}"
    )

    # ---- 9. SHAP (using LightGBM) ----
    logger.info("\n[9/9] Generating SHAP plots (LightGBM)...")
    generate_shap_plots(lgb_model, X_test, NUMERICAL_FEATURES, REPORTS_DIR)

    # ---- Save ----
    logger.info("\nSaving artifacts...")
    all_params = {
        "lightgbm": best_lgb_params,
        "xgboost": best_xgb_params,
        "catboost": best_cb_params,
    }
    all_metrics = {
        "lightgbm": metrics_lgb,
        "xgboost": metrics_xgb,
        "catboost": metrics_cb,
        "ensemble": metrics_ens,
    }
    save_artifacts(
        lgb_model,
        xgb_model,
        cb_model,
        scaler,
        encoder,
        train_stats,
        all_params,
        all_metrics,
    )

    # ---- MLflow logging ----
    logger.info("\nLogging to MLflow...")
    mlflow.set_experiment("video-game-sales-prediction")
    with mlflow.start_run(run_name=f"ensemble_{datetime.now():%Y%m%d_%H%M%S}"):
        # Log parameters
        mlflow.log_param("split_year", best_split_year)
        mlflow.log_param("log_transform", LOG_TRANSFORM)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_size", len(df_train))
        mlflow.log_param("test_size", len(df_test))
        for model_name, params in all_params.items():
            for k, v in params.items():
                if k != "split_year":
                    mlflow.log_param(f"{model_name}__{k}", v)

        # Log metrics
        for model_name, metrics in all_metrics.items():
            for metric_key, metric_val in metrics.items():
                mlflow.log_metric(f"{model_name}__{metric_key}", metric_val)

        # Log artifacts
        mlflow.log_artifact(str(REPORTS_DIR / "model_v2_optuna.txt"))
        mlflow.log_artifact(str(MODELS_DIR / "model_v2_xgboost.json"))
        mlflow.log_artifact(str(MODELS_DIR / "model_v2_catboost.cbm"))
        mlflow.log_artifact(str(REPORTS_DIR / "training_log.json"))
        mlflow.log_artifact(str(REPORTS_DIR / "shap_summary.png"))
        mlflow.log_artifact(str(REPORTS_DIR / "shap_bar.png"))

    logger.info("  MLflow run logged successfully")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete! (3 models + ensemble)")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
