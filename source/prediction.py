"""Prediction module: Streamlit-cached wrappers around ml.predict functions.

This module provides the ``@st.cache_resource`` / ``@st.cache_data``
wrappers and re-exports the core ML functions so that other Streamlit
pages can ``from prediction import load_models, predict_single, ...``
without importing ``ml.predict`` directly.
"""

import streamlit as st
from ml.predict import (
    NUMERICAL_FEATURES,
    get_features,
    is_log_transformed,
    lookup_cumulative,
    predict_ensemble,
    predict_single,
    prepare_for_prediction,
)
from ml.predict import (
    load_feature_means as _load_feature_means,
)
from ml.predict import (
    load_models as _load_models,
)
from ml.predict import (
    load_numerical_transformer as _load_numerical_transformer,
)
from ml.predict import (
    load_target_encoder as _load_target_encoder,
)

# Re-export for backwards compatibility (used by tests and other pages)
_lookup_cumulative = lookup_cumulative
_NUMERICAL_FEATURES = NUMERICAL_FEATURES


@st.cache_resource
def load_models():
    """Load all 3 ensemble models (cached)."""
    return _load_models()


@st.cache_resource
def load_numerical_transformer():
    """Load the StandardScaler (cached)."""
    return _load_numerical_transformer()


@st.cache_resource
def load_target_encoder():
    """Load the target encoder (cached)."""
    return _load_target_encoder()


@st.cache_resource
def load_feature_means():
    """Load pre-computed feature means (cached)."""
    return _load_feature_means()


@st.cache_data
def _is_log_transformed() -> bool:
    """Check if the trained model used log-transform (cached)."""
    return is_log_transformed()


# Re-export page function from pages package
from pages.prediction import prediction_page  # noqa: E402, F401

__all__ = [
    "NUMERICAL_FEATURES",
    "get_features",
    "load_feature_means",
    "load_models",
    "load_numerical_transformer",
    "load_target_encoder",
    "lookup_cumulative",
    "predict_ensemble",
    "predict_single",
    "prediction_page",
    "prepare_for_prediction",
]
