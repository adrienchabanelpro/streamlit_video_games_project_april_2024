"""Training pipeline v3 orchestrator.

Runs the full pipeline: data prep → Optuna tuning → stacking ensemble
→ evaluation → SHAP → artifact saving.

Usage:
    python -m scripts.training.run_training
    python scripts/training/run_training.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import optuna

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from scripts.training.data_prep import (
    TARGET,
    clean_data,
    compute_train_stats,
    load_dataset,
    prepare_training_data,
    temporal_split,
)
from scripts.training.evaluation import (
    evaluate_model,
    evaluate_stacking,
    print_metrics_table,
)
from scripts.training.models import (
    RANDOM_STATE,
    objective_cb,
    objective_elastic,
    objective_hgb,
    objective_lgb,
    objective_rf,
    objective_xgb,
    train_cb,
    train_elastic,
    train_hgb,
    train_lgb,
    train_rf,
    train_xgb,
)
from scripts.training.stacking import train_stacking_ensemble

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

LOG_TRANSFORM = True
SPLIT_YEAR = 2015  # Default; Optuna may override


def _tune_model(
    name: str,
    objective_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    seed_offset: int = 0,
) -> dict:
    """Run Optuna tuning for a single model."""
    logger.info(f"  Tuning {name} ({n_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + seed_offset),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    logger.info(f"  {name} best CV R²: {study.best_value:.4f}")
    return study.best_params


def main(
    n_trials_lgb: int = 50,
    n_trials_xgb: int = 30,
    n_trials_cb: int = 30,
    n_trials_rf: int = 20,
    n_trials_hgb: int = 20,
    n_trials_elastic: int = 20,
    split_year: int = SPLIT_YEAR,
) -> None:
    """Run the full v3 training pipeline."""
    logger.info("=" * 60)
    logger.info("Video Game Sales — Training Pipeline v3 (Stacking Ensemble)")
    logger.info("=" * 60)

    # ---- 1. Load & clean ----
    logger.info("\n[1/8] Loading and cleaning data...")
    df = load_dataset()
    df = clean_data(df)
    logger.info(f"  {len(df):,} rows, Year range: {df['Year'].min()}-{df['Year'].max()}")

    # ---- 2. Split ----
    logger.info(f"\n[2/8] Temporal split at year {split_year}...")
    df_train, df_test = temporal_split(df, split_year)

    # ---- 3. Feature engineering + preparation ----
    logger.info("\n[3/8] Feature engineering (train stats only)...")
    train_stats = compute_train_stats(df_train)
    X_train, y_train, X_test, y_test_raw, features, scaler, encoder = (
        prepare_training_data(df_train, df_test, train_stats, log_transform=LOG_TRANSFORM)
    )
    logger.info(f"  {len(features)} features, X_train shape: {X_train.shape}")

    # ---- 4. Optuna tuning for all models ----
    logger.info("\n[4/8] Hyperparameter tuning...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_lgb = _tune_model("LightGBM", objective_lgb, X_train, y_train, n_trials_lgb, 0)
    best_xgb = _tune_model("XGBoost", objective_xgb, X_train, y_train, n_trials_xgb, 1)
    best_cb = _tune_model("CatBoost", objective_cb, X_train, y_train, n_trials_cb, 2)
    best_rf = _tune_model("RandomForest", objective_rf, X_train, y_train, n_trials_rf, 3)
    best_hgb = _tune_model("HistGBR", objective_hgb, X_train, y_train, n_trials_hgb, 4)
    best_elastic = _tune_model("ElasticNet", objective_elastic, X_train, y_train, n_trials_elastic, 5)

    # ---- 5. Stacking ensemble ----
    logger.info("\n[5/8] Training stacking ensemble (5 base models + Ridge meta-learner)...")

    # We use the 5 tree-based models for stacking (ElasticNet as separate baseline)
    model_configs = {
        "lightgbm": (train_lgb, best_lgb),
        "xgboost": (train_xgb, best_xgb),
        "catboost": (train_cb, best_cb),
        "random_forest": (train_rf, best_rf),
        "hist_gbr": (train_hgb, best_hgb),
    }

    base_models, meta_learner = train_stacking_ensemble(
        model_configs, X_train, y_train, n_splits=5
    )

    # Also train ElasticNet as a separate baseline
    logger.info("  Training ElasticNet baseline...")
    elastic_model = train_elastic(X_train, y_train, best_elastic)

    # ---- 6. Evaluate ----
    logger.info("\n[6/8] Evaluating on test set...")
    gm = train_stats["global_sales_mean"]

    all_metrics: dict[str, dict] = {}
    model_names = list(model_configs.keys())
    for i, name in enumerate(model_names):
        all_metrics[name] = evaluate_model(
            base_models[i], X_test, y_test_raw, gm, LOG_TRANSFORM
        )

    all_metrics["elastic_net"] = evaluate_model(
        elastic_model, X_test, y_test_raw, gm, LOG_TRANSFORM
    )

    # Stacking ensemble
    stacking_metrics = evaluate_stacking(
        base_models, meta_learner, X_test, y_test_raw, gm, LOG_TRANSFORM
    )
    all_metrics["stacking_ensemble"] = stacking_metrics

    logger.info("\n  Results:")
    print_metrics_table(all_metrics)

    logger.info(f"\n  Stacking vs Simple Avg: "
                f"R²={stacking_metrics['r2']:.4f} vs {stacking_metrics['simple_avg_r2']:.4f}")
    logger.info(f"  Prediction uncertainty (mean std): {stacking_metrics['pred_std_mean']:.4f}")

    # ---- 7. SHAP ----
    logger.info("\n[7/8] Generating SHAP plots (LightGBM)...")
    _generate_shap(base_models[0], X_test, features)

    # ---- 8. Save artifacts ----
    logger.info("\n[8/8] Saving artifacts...")
    _save_artifacts(
        base_models, meta_learner, elastic_model,
        scaler, encoder, train_stats, features,
        {name: best for name, (_, best) in zip(model_names, model_configs.values())},
        best_elastic, all_metrics, split_year,
    )

    logger.info("\n" + "=" * 60)
    best_r2 = stacking_metrics["r2"]
    logger.info(f"Training complete! Stacking R² = {best_r2:.4f}")
    logger.info("=" * 60)


def _generate_shap(model, X_test: np.ndarray, features: list[str]) -> None:
    """Generate SHAP plots from the LightGBM base model."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_summary_v3.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_bar_v3.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"  SHAP plots saved to {REPORTS_DIR}")
    except Exception as exc:
        logger.warning(f"  SHAP generation failed: {exc}")


def _save_artifacts(
    base_models: list,
    meta_learner,
    elastic_model,
    scaler,
    encoder,
    train_stats: dict,
    features: list[str],
    all_params: dict,
    elastic_params: dict,
    all_metrics: dict,
    split_year: int,
) -> None:
    """Save all model artifacts for inference and reproducibility."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    # Save each base model
    for model in base_models:
        if isinstance(model, lgb.LGBMRegressor):
            model.booster_.save_model(str(REPORTS_DIR / "model_v3_lgb.txt"))
        elif isinstance(model, xgb.XGBRegressor):
            model.save_model(str(MODELS_DIR / "model_v3_xgb.json"))
        elif isinstance(model, cb.CatBoostRegressor):
            model.save_model(str(MODELS_DIR / "model_v3_cb.cbm"))

    # Save sklearn-based models
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )

    for model in base_models:
        if isinstance(model, RandomForestRegressor):
            joblib.dump(model, MODELS_DIR / "model_v3_rf.joblib")
        elif isinstance(model, HistGradientBoostingRegressor):
            joblib.dump(model, MODELS_DIR / "model_v3_hgb.joblib")

    # Meta-learner
    joblib.dump(meta_learner, MODELS_DIR / "meta_learner_v3.joblib")

    # ElasticNet baseline
    joblib.dump(elastic_model, MODELS_DIR / "model_v3_elastic.joblib")

    # Preprocessing artifacts
    joblib.dump(scaler, MODELS_DIR / "scaler_v3.joblib")
    joblib.dump(encoder, MODELS_DIR / "target_encoder_v3.joblib")
    joblib.dump(train_stats, MODELS_DIR / "feature_means_v3.joblib")

    # Training log
    log = {
        "version": 3,
        "timestamp": datetime.now().isoformat(),
        "split_year": split_year,
        "log_transform": LOG_TRANSFORM,
        "random_state": RANDOM_STATE,
        "features": features,
        "n_features": len(features),
        "target": TARGET,
        "best_params": {**all_params, "elastic_net": elastic_params},
        "metrics": all_metrics,
        "stacking_meta_weights": meta_learner.coef_.tolist(),
        "stacking_meta_intercept": float(meta_learner.intercept_),
        "stacking_meta_alpha": float(meta_learner.alpha_),
    }
    with open(REPORTS_DIR / "training_log_v3.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    logger.info("  Artifacts saved to models/ and reports/")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
