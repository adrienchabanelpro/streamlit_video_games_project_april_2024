"""ML prediction logic: model loading, feature engineering, and ensemble inference.

Supports two model versions:
- **v3 (preferred):** 5 base models + Ridge meta-learner (stacking ensemble), 50 features.
- **v2 (fallback):** 3 base models simple average, 10 features.

This module is Streamlit-agnostic. Caching decorators are applied in the page
layer (``pages/prediction.py``) via thin wrappers.
"""

from __future__ import annotations

import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
from config import MODELS_DIR, REPORTS_DIR

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# v2 fallback feature list (10 features)
_V2_FEATURES: list[str] = [
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


def _model_version() -> int:
    """Detect which model version is available (3 or 2)."""
    if (REPORTS_DIR / "model_v3_lgb.txt").exists():
        return 3
    return 2


def get_feature_names() -> list[str]:
    """Load feature names from training log, or fall back to v2 defaults."""
    for name in ("training_log_v3.json", "training_log.json"):
        path = REPORTS_DIR / name
        if path.exists():
            with open(path) as f:
                features = json.load(f).get("features")
                if features:
                    return features
    return list(_V2_FEATURES)


# Public accessor (used by prediction pages)
NUMERICAL_FEATURES: list[str] = _V2_FEATURES  # overridden at load time


def load_models() -> tuple[list, object | None, int]:
    """Load ensemble models. Returns (base_models, meta_learner, version).

    v3: list of 5 base models + Ridge meta-learner.
    v2: list of 3 base models + None meta-learner.
    """
    version = _model_version()
    if version == 3:
        return _load_v3_models()
    return _load_v2_models()


def _load_v3_models() -> tuple[list, object, int]:
    """Load v3 stacking ensemble: 5 base models + Ridge meta-learner."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    models = []

    # LightGBM (loaded as Booster — .predict() works natively)
    lgb_model = lgb.Booster(model_file=str(REPORTS_DIR / "model_v3_lgb.txt"))
    models.append(lgb_model)

    # XGBoost
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(MODELS_DIR / "model_v3_xgb.json"))
    models.append(xgb_model)

    # CatBoost
    cb_model = cb.CatBoostRegressor()
    cb_model.load_model(str(MODELS_DIR / "model_v3_cb.cbm"))
    models.append(cb_model)

    # RandomForest
    rf_path = MODELS_DIR / "model_v3_rf.joblib"
    if rf_path.exists():
        models.append(joblib.load(rf_path))

    # HistGradientBoosting
    hgb_path = MODELS_DIR / "model_v3_hgb.joblib"
    if hgb_path.exists():
        models.append(joblib.load(hgb_path))

    meta_learner = joblib.load(MODELS_DIR / "meta_learner_v3.joblib")
    return models, meta_learner, 3


def _load_v2_models() -> tuple[list, None, int]:
    """Load v2 simple ensemble: LGB + XGB + CB average."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    lgb_model = lgb.Booster(model_file=str(REPORTS_DIR / "model_v2_optuna.txt"))
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(MODELS_DIR / "model_v2_xgboost.json"))
    cb_model = cb.CatBoostRegressor()
    cb_model.load_model(str(MODELS_DIR / "model_v2_catboost.cbm"))
    return [lgb_model, xgb_model, cb_model], None, 2


def load_numerical_transformer() -> object:
    """Load the StandardScaler fitted during training."""
    v3 = MODELS_DIR / "scaler_v3.joblib"
    return joblib.load(v3 if v3.exists() else MODELS_DIR / "scaler_v2.joblib")


def load_target_encoder() -> object:
    """Load the target encoder for Publisher column."""
    v3 = MODELS_DIR / "target_encoder_v3.joblib"
    return joblib.load(v3 if v3.exists() else MODELS_DIR / "target_encoder_v2.joblib")


def load_feature_means() -> dict:
    """Load pre-computed feature means (training statistics)."""
    v3 = MODELS_DIR / "feature_means_v3.joblib"
    path = v3 if v3.exists() else MODELS_DIR / "feature_means_v2.joblib"
    stats = joblib.load(path)
    # Safety: replace NaN means with 0.0
    for key in ("meta_score_mean", "user_review_mean"):
        if key in stats and (stats[key] is None or (isinstance(stats[key], float) and math.isnan(stats[key]))):
            stats[key] = 0.0
    return stats


def is_log_transformed() -> bool:
    """Check if the trained model used log-transform on the target."""
    for name in ("training_log_v3.json", "training_log.json"):
        path = REPORTS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f).get("log_transform", False)
    return False


def lookup_cumulative(cumsum_dict: dict, category: str, year: int) -> float:
    """Look up cumulative sales for a category up to a given year."""
    if category not in cumsum_dict:
        return 0.0
    yearly = cumsum_dict[category]
    relevant_years = [y for y in yearly if y <= year]
    if not relevant_years:
        return 0.0
    return yearly[max(relevant_years)]


def _build_v3_features(
    input_data: dict[str, float],
    train_stats: dict,
    feature_names: list[str],
) -> None:
    """Populate all v3 engineered features in input_data using training stats."""
    gm = train_stats.get("global_sales_mean", 0.0)

    # Publisher track record
    pub_stats = train_stats.get("publisher_stats", {})
    publisher = input_data.get("_publisher", "")
    pub_info = pub_stats.get(publisher, {})
    input_data.setdefault("publisher_avg_sales_prior", pub_info.get("pub_avg_sales", gm))
    input_data.setdefault("publisher_game_count_prior", pub_info.get("pub_game_count", 0))
    input_data.setdefault(
        "publisher_hit_rate",
        train_stats.get("publisher_hit_rate", {}).get(publisher, 0.0),
    )

    # Competition density
    games_per_year = train_stats.get("games_per_year", {})
    year = input_data.get("Year", 2020)
    input_data.setdefault("competition_density", games_per_year.get(int(year), 0))

    # Genre market share
    genre = input_data.get("_genre", "")
    genre_market = train_stats.get("genre_market_share", {})
    input_data.setdefault(
        "genre_market_share",
        genre_market.get(genre, {}).get(int(year), 0.0),
    )

    # Fill any remaining features with 0 (enrichment features the user didn't provide)
    for feat in feature_names:
        input_data.setdefault(feat, 0.0)


def get_features(
    input_data: dict,
    train_stats: dict,
    genre_input: str,
    platform_input: str,
) -> pd.DataFrame:
    """Build feature vector using pre-computed training statistics."""
    gm = train_stats.get("global_sales_mean", 0.0)

    input_data["Global_Sales_mean_genre"] = train_stats["genre_means"].get(genre_input, gm)
    input_data["Global_Sales_mean_platform"] = train_stats["platform_means"].get(platform_input, gm)
    input_data["Year_Global_Sales_mean_genre"] = (
        input_data["Year"] * input_data["Global_Sales_mean_genre"]
    )
    input_data["Year_Global_Sales_mean_platform"] = (
        input_data["Year"] * input_data["Global_Sales_mean_platform"]
    )
    input_data["Cumulative_Sales_Genre"] = lookup_cumulative(
        train_stats["cumsum_genre"], genre_input, input_data["Year"]
    )
    input_data["Cumulative_Sales_Platform"] = lookup_cumulative(
        train_stats["cumsum_platform"], platform_input, input_data["Year"]
    )

    # For v3, build additional features
    feature_names = get_feature_names()
    if len(feature_names) > len(_V2_FEATURES):
        input_data["_genre"] = genre_input
        input_data["_publisher"] = input_data.get("_publisher", "")
        _build_v3_features(input_data, train_stats, feature_names)
        # Clean up internal keys
        input_data.pop("_genre", None)
        input_data.pop("_publisher", None)

    return pd.DataFrame(input_data, index=[0])


def prepare_for_prediction(
    df_input: pd.DataFrame, publisher_input: str
) -> pd.DataFrame:
    """Target-encode Publisher, scale features, return prediction-ready df."""
    encoder = load_target_encoder()
    scaler = load_numerical_transformer()
    features = get_feature_names()

    pub_df = pd.DataFrame({"Publisher": [publisher_input]})
    df_input["Publisher_encoded"] = encoder.transform(pub_df)["Publisher"].values
    df_input[features] = scaler.transform(df_input[features])
    return df_input


def _predict_v3_stacking(
    base_models: list,
    meta_learner: object,
    X: np.ndarray,
    log_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run v3 stacking prediction. Returns (predictions, uncertainty_std)."""
    base_preds = np.column_stack([m.predict(X) for m in base_models])
    pred = meta_learner.predict(base_preds)
    uncertainty = np.std(base_preds, axis=1)

    if log_transform:
        pred = np.expm1(pred)
        uncertainty = np.expm1(uncertainty)

    return np.maximum(pred, 0), uncertainty


def _predict_v2_average(
    models: list,
    X: np.ndarray,
    log_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run v2 simple average prediction."""
    import lightgbm as lgb

    preds = []
    for m in models:
        if isinstance(m, lgb.Booster):
            preds.append(m.predict(X))
        else:
            preds.append(m.predict(X))
    preds_arr = np.array(preds)
    pred = preds_arr.mean(axis=0)
    uncertainty = preds_arr.std(axis=0)

    if log_transform:
        pred = np.expm1(pred)
        uncertainty = np.expm1(uncertainty)

    return np.maximum(pred, 0), uncertainty


def predict_single(
    models: list,
    meta_learner: object | None,
    scaler: object,
    encoder: object,
    train_stats: dict,
    genre: str,
    platform: str,
    publisher: str,
    year: int,
    meta_score: float,
    user_review: float,
    version: int = 2,
    **extra_features: float,
) -> tuple[float, float]:
    """Build features and run ensemble prediction for a single game.

    Returns (predicted_sales, uncertainty).
    """
    feature_names = get_feature_names()

    input_data: dict[str, float] = {
        "Year": year,
        "meta_score": meta_score,
        "user_review": user_review,
        **extra_features,
    }
    input_data["_publisher"] = publisher

    gm = train_stats.get("global_sales_mean", 0.0)
    input_data["Global_Sales_mean_genre"] = train_stats["genre_means"].get(genre, gm)
    input_data["Global_Sales_mean_platform"] = train_stats["platform_means"].get(platform, gm)
    input_data["Year_Global_Sales_mean_genre"] = year * input_data["Global_Sales_mean_genre"]
    input_data["Year_Global_Sales_mean_platform"] = year * input_data["Global_Sales_mean_platform"]
    input_data["Cumulative_Sales_Genre"] = lookup_cumulative(train_stats["cumsum_genre"], genre, year)
    input_data["Cumulative_Sales_Platform"] = lookup_cumulative(train_stats["cumsum_platform"], platform, year)

    if version == 3:
        input_data["_genre"] = genre
        _build_v3_features(input_data, train_stats, feature_names)
        input_data.pop("_genre", None)
        input_data.pop("_publisher", None)

    # Encode publisher
    pub_df = pd.DataFrame({"Publisher": [publisher]})
    input_data["Publisher_encoded"] = encoder.transform(pub_df)["Publisher"].values[0]

    df = pd.DataFrame(input_data, index=[0])
    df[feature_names] = scaler.transform(df[feature_names])

    X = df[feature_names].values
    log_transform = is_log_transformed()

    if version == 3 and meta_learner is not None:
        pred, unc = _predict_v3_stacking(models, meta_learner, X, log_transform)
    else:
        pred, unc = _predict_v2_average(models, X, log_transform)

    return float(pred[0]), float(unc[0])


def predict_ensemble(
    models: list,
    meta_learner: object | None,
    X: pd.DataFrame | np.ndarray,
    version: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ensemble prediction on a prepared feature matrix.

    Returns (predictions, uncertainty_std).
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    log_transform = is_log_transformed()

    if version == 3 and meta_learner is not None:
        return _predict_v3_stacking(models, meta_learner, X, log_transform)
    return _predict_v2_average(models, X, log_transform)
