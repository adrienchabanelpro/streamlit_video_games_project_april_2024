"""ML prediction logic: model loading, feature engineering, and ensemble inference.

This module is Streamlit-agnostic. Caching decorators are applied in the page
layer (``pages/prediction.py``) via thin wrappers.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from config import MODELS_DIR, REPORTS_DIR

if TYPE_CHECKING:
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Feature order must match training script exactly
NUMERICAL_FEATURES: list[str] = [
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


def load_models() -> tuple[lgb.Booster, xgb.XGBRegressor, cb.CatBoostRegressor]:
    """Load all 3 ensemble models (LightGBM, XGBoost, CatBoost)."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    lgb_model = lgb.Booster(model_file=str(REPORTS_DIR / "model_v2_optuna.txt"))
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(MODELS_DIR / "model_v2_xgboost.json"))
    cb_model = cb.CatBoostRegressor()
    cb_model.load_model(str(MODELS_DIR / "model_v2_catboost.cbm"))
    return lgb_model, xgb_model, cb_model


def load_numerical_transformer() -> object:
    """Load the StandardScaler fitted during training."""
    return joblib.load(MODELS_DIR / "scaler_v2.joblib")


def load_target_encoder() -> object:
    """Load the target encoder for Publisher column."""
    return joblib.load(MODELS_DIR / "target_encoder_v2.joblib")


def load_feature_means() -> dict:
    """Load pre-computed feature means from training data."""
    return joblib.load(MODELS_DIR / "feature_means_v2.joblib")


def is_log_transformed() -> bool:
    """Check if the trained model used log-transform on the target."""
    log_path = REPORTS_DIR / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
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


def get_features(
    input_data: dict,
    train_stats: dict,
    genre_input: str,
    platform_input: str,
) -> pd.DataFrame:
    """Build feature vector using pre-computed training statistics."""
    input_data["Global_Sales_mean_genre"] = train_stats["genre_means"].get(
        genre_input, train_stats["global_sales_mean"]
    )
    input_data["Global_Sales_mean_platform"] = train_stats["platform_means"].get(
        platform_input, train_stats["global_sales_mean"]
    )
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
    return pd.DataFrame(input_data, index=[0])


def prepare_for_prediction(
    df_input: pd.DataFrame, publisher_input: str
) -> pd.DataFrame:
    """Target-encode Publisher, scale features, return prediction-ready df."""
    encoder = load_target_encoder()
    scaler = load_numerical_transformer()

    pub_df = pd.DataFrame({"Publisher": [publisher_input]})
    df_input["Publisher_encoded"] = encoder.transform(pub_df)["Publisher"].values
    df_input[NUMERICAL_FEATURES] = scaler.transform(df_input[NUMERICAL_FEATURES])
    return df_input


def predict_single(
    lgb_model: lgb.Booster,
    xgb_model: xgb.XGBRegressor,
    cb_model: cb.CatBoostRegressor,
    scaler: object,
    encoder: object,
    train_stats: dict,
    genre: str,
    platform: str,
    publisher: str,
    year: int,
    meta_score: float,
    user_review: float,
) -> float:
    """Build features and run ensemble prediction for a single game."""
    input_data: dict[str, float] = {
        "Year": year,
        "meta_score": meta_score,
        "user_review": user_review,
    }

    input_data["Global_Sales_mean_genre"] = train_stats["genre_means"].get(
        genre, train_stats["global_sales_mean"]
    )
    input_data["Global_Sales_mean_platform"] = train_stats["platform_means"].get(
        platform, train_stats["global_sales_mean"]
    )
    input_data["Year_Global_Sales_mean_genre"] = (
        input_data["Year"] * input_data["Global_Sales_mean_genre"]
    )
    input_data["Year_Global_Sales_mean_platform"] = (
        input_data["Year"] * input_data["Global_Sales_mean_platform"]
    )
    input_data["Cumulative_Sales_Genre"] = lookup_cumulative(
        train_stats["cumsum_genre"], genre, year
    )
    input_data["Cumulative_Sales_Platform"] = lookup_cumulative(
        train_stats["cumsum_platform"], platform, year
    )

    pub_df = pd.DataFrame({"Publisher": [publisher]})
    input_data["Publisher_encoded"] = encoder.transform(pub_df)["Publisher"].values[0]

    df = pd.DataFrame(input_data, index=[0])
    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])

    X = df[NUMERICAL_FEATURES]
    pred_lgb = np.atleast_1d(lgb_model.predict(X))
    pred_xgb = np.atleast_1d(xgb_model.predict(X.values))
    pred_cb = np.atleast_1d(cb_model.predict(X.values))
    result = float(((pred_lgb + pred_xgb + pred_cb) / 3)[0])
    if is_log_transformed():
        result = float(np.expm1(result))
    return result


def predict_ensemble(
    lgb_model: lgb.Booster,
    xgb_model: xgb.XGBRegressor,
    cb_model: cb.CatBoostRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Run ensemble prediction on a prepared feature matrix."""
    pred = (
        lgb_model.predict(X) + xgb_model.predict(X.values) + cb_model.predict(X.values)
    ) / 3
    if is_log_transformed():
        pred = np.expm1(pred)
    return pred
