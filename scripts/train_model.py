"""Training pipeline for Video Game Sales Prediction v2.

Implements:
- Temporal train/test split (no data leakage)
- Feature engineering computed on training data only
- Target encoding for Publisher (replaces 567-column one-hot)
- Optuna hyperparameter tuning with 5-fold CV
- SHAP feature importance plots
- Full artifact saving for reproducibility

Usage:
    python scripts/train_model.py

Outputs saved to models/ and reports/:
    - reports/model_v2_optuna.txt    (LightGBM model)
    - models/scaler_v2.joblib        (StandardScaler)
    - models/target_encoder_v2.joblib (Publisher target encoder)
    - models/feature_means_v2.joblib (genre/platform means + cumulative stats)
    - reports/shap_summary.png
    - reports/shap_bar.png
    - reports/training_log.json      (params, metrics, timestamp)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import category_encoders as ce
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
    stats["platform_means"] = (
        df_train.groupby("Platform")[TARGET].mean().to_dict()
    )

    # Cumulative sales by Genre and Year
    stats["cumsum_genre"] = {}
    for genre in df_train["Genre"].unique():
        genre_data = (
            df_train[df_train["Genre"] == genre]
            .groupby("Year")[TARGET]
            .sum()
            .sort_index()
            .cumsum()
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


def compute_engineered_features(
    df: pd.DataFrame, train_stats: dict
) -> pd.DataFrame:
    """Apply feature engineering using pre-computed training statistics."""
    df = df.copy()

    # Mean sales by genre / platform (from training data)
    df["Global_Sales_mean_genre"] = (
        df["Genre"]
        .map(train_stats["genre_means"])
        .fillna(train_stats["global_sales_mean"])
    )
    df["Global_Sales_mean_platform"] = (
        df["Platform"]
        .map(train_stats["platform_means"])
        .fillna(train_stats["global_sales_mean"])
    )

    # Interaction features
    df["Year_Global_Sales_mean_genre"] = (
        df["Year"] * df["Global_Sales_mean_genre"]
    )
    df["Year_Global_Sales_mean_platform"] = (
        df["Year"] * df["Global_Sales_mean_platform"]
    )

    # Cumulative sales
    df["Cumulative_Sales_Genre"] = df.apply(
        lambda row: _lookup_cumulative(
            train_stats["cumsum_genre"], row["Genre"], row["Year"]
        ),
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
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 10.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),
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
# Final training
# ---------------------------------------------------------------------------
def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
) -> lgb.LGBMRegressor:
    """Train the final model with best hyperparameters on full train set."""
    model_params = {k: v for k, v in best_params.items() if k != "split_year"}

    model = lgb.LGBMRegressor(
        **model_params,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: lgb.LGBMRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    global_mean: float,
) -> dict:
    """Evaluate model on test set + compare against mean-predictor baseline."""
    y_pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }

    # Baseline: always predict the training mean
    y_baseline = np.full_like(y_test, global_mean)
    metrics["baseline_r2"] = float(r2_score(y_test, y_baseline))
    metrics["baseline_mse"] = float(mean_squared_error(y_test, y_baseline))
    metrics["baseline_rmse"] = float(
        np.sqrt(mean_squared_error(y_test, y_baseline))
    )
    metrics["baseline_mae"] = float(
        mean_absolute_error(y_test, y_baseline)
    )

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

    print(f"  SHAP plots saved to {save_dir}")


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------
def save_artifacts(
    model: lgb.LGBMRegressor,
    scaler: StandardScaler,
    encoder: ce.TargetEncoder,
    train_stats: dict,
    best_params: dict,
    metrics: dict,
) -> None:
    """Save all artifacts for reproducibility and inference."""
    # LightGBM model (native format)
    model.booster_.save_model(str(REPORTS_DIR / "model_v2_optuna.txt"))

    # Transformers
    joblib.dump(scaler, MODELS_DIR / "scaler_v2.joblib")
    joblib.dump(encoder, MODELS_DIR / "target_encoder_v2.joblib")
    joblib.dump(train_stats, MODELS_DIR / "feature_means_v2.joblib")

    # Training log
    log = {
        "timestamp": datetime.now().isoformat(),
        "best_params": best_params,
        "metrics": metrics,
        "features": NUMERICAL_FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
    }
    with open(REPORTS_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"  Model        -> {REPORTS_DIR / 'model_v2_optuna.txt'}")
    print(f"  Scaler       -> {MODELS_DIR / 'scaler_v2.joblib'}")
    print(f"  Encoder      -> {MODELS_DIR / 'target_encoder_v2.joblib'}")
    print(f"  Feature means-> {MODELS_DIR / 'feature_means_v2.joblib'}")
    print(f"  Training log -> {REPORTS_DIR / 'training_log.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full training pipeline."""
    print("=" * 60)
    print("Video Game Sales - Training Pipeline v2")
    print("=" * 60)

    # ---- 1. Load & clean ----
    print("\n[1/7] Loading and cleaning data...")
    df = load_and_clean_data(DATA_DIR / "Ventes_jeux_video_final.csv")
    print(f"  {len(df)} rows, {len(df.columns)} columns")
    print(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"  Unique publishers: {df['Publisher'].nunique()}")

    # ---- 2. Optuna tuning ----
    print("\n[2/7] Optuna hyperparameter tuning (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, df),
        n_trials=50,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_score = study.best_value
    best_split_year = best_params["split_year"]
    print(f"  Best CV R2: {best_score:.4f}")
    print(f"  Best split_year: {best_split_year}")
    print(
        f"  Best params: {json.dumps({k: v for k, v in best_params.items() if k != 'split_year'}, indent=4)}"
    )

    # ---- 3. Final temporal split ----
    print(f"\n[3/7] Splitting data at year {best_split_year}...")
    df_train, df_test = temporal_train_test_split(df, best_split_year)
    print(f"  Train: {len(df_train)} rows (<= {best_split_year})")
    print(f"  Test:  {len(df_test)} rows (> {best_split_year})")

    # ---- 4. Feature engineering ----
    print("\n[4/7] Feature engineering (train stats only)...")
    train_stats = compute_train_stats(df_train)
    df_train = compute_engineered_features(df_train, train_stats)
    df_test = compute_engineered_features(df_test, train_stats)

    # Target-encode Publisher
    print("  Fitting target encoder on Publisher...")
    encoder = ce.TargetEncoder(cols=["Publisher"], smoothing=10)
    df_train["Publisher_encoded"] = encoder.fit_transform(
        df_train[["Publisher"]], df_train[TARGET]
    )["Publisher"]
    df_test["Publisher_encoded"] = encoder.transform(
        df_test[["Publisher"]]
    )["Publisher"]

    # Scale numerical features
    print("  Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[NUMERICAL_FEATURES])
    y_train = df_train[TARGET].values
    X_test = scaler.transform(df_test[NUMERICAL_FEATURES])
    y_test = df_test[TARGET].values

    # ---- 5. Train final model ----
    print("\n[5/7] Training final model with best hyperparameters...")
    model = train_final_model(X_train, y_train, best_params)

    # ---- 6. Evaluate ----
    print("\n[6/7] Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test, train_stats["global_sales_mean"])
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  --- Baseline (mean predictor) ---")
    print(f"  R2:   {metrics['baseline_r2']:.4f}")
    print(f"  RMSE: {metrics['baseline_rmse']:.4f}")

    # ---- 7. SHAP ----
    print("\n[7/7] Generating SHAP plots...")
    generate_shap_plots(model, X_test, NUMERICAL_FEATURES, REPORTS_DIR)

    # ---- Save ----
    print("\nSaving artifacts...")
    save_artifacts(model, scaler, encoder, train_stats, best_params, metrics)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
