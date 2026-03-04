"""Home page: project overview, key metrics, and pipeline diagram."""

import json

import pandas as pd
import streamlit as st
from components import info_card, metric_card, pipeline_step, section_header
from config import DATA_DIR, REPORTS_DIR


@st.cache_data
def _load_overview_stats() -> dict:
    """Load dataset stats and model metrics for the dashboard."""
    stats: dict = {"rows": 0, "cols": 0, "features": 0, "best_r2": 0.0}

    # Dataset stats
    for csv_name in ["Ventes_jeux_video_v3.csv", "Ventes_jeux_video_final.csv"]:
        path = DATA_DIR / csv_name
        if path.exists():
            df = pd.read_csv(path, nrows=1)
            stats["cols"] = len(df.columns)
            # Count rows without loading full file
            with open(path) as f:
                stats["rows"] = sum(1 for _ in f) - 1
            stats["dataset"] = csv_name
            break

    # Training log
    for log_name in ["training_log_v3.json", "training_log.json"]:
        path = REPORTS_DIR / log_name
        if path.exists():
            with open(path) as f:
                log = json.load(f)
            metrics = log.get("metrics", {})
            ensemble_key = "stacking_ensemble" if "stacking_ensemble" in metrics else "ensemble"
            if ensemble_key in metrics:
                stats["best_r2"] = metrics[ensemble_key].get("r2", 0.0)
            stats["features"] = log.get("n_features", len(log.get("features", [])))
            stats["version"] = log_name.replace("training_log", "").replace(".json", "").strip("_") or "v2"
            break

    return stats


def home_page() -> None:
    """Render the home/overview page."""
    st.title("Video Game Sales Prediction")
    st.caption(
        "Data Science Project: collection, analysis, and prediction of global "
        "video game sales from multiple sources"
    )

    stats = _load_overview_stats()

    # Key metrics row
    section_header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Games in Dataset", f"{stats['rows']:,}", icon="🎮")
    with c2:
        metric_card("Columns", stats["cols"], icon="📊")
    with c3:
        metric_card("Features ML", stats["features"], icon="⚙️")
    with c4:
        metric_card("Best R²", f"{stats['best_r2']:.3f}", icon="🎯")

    st.divider()

    # Project description
    section_header("Project Objective")
    st.write(
        """
        This project uses **Machine Learning** to predict global video game sales.
        It combines **9 data sources** (VGChartz, SteamSpy, RAWG, IGDB, HowLongToBeat,
        Wikipedia, Steam Store, OpenCritic, Gamedatacrunch) and an **ensemble of models**
        (LightGBM, XGBoost, CatBoost, Random Forest, HistGradientBoosting) with a
        **Ridge meta-learner** (stacking) for optimal predictions.

        Each step is documented in the corresponding pages.
        """
    )

    st.divider()

    # Pipeline overview
    section_header("Data Pipeline", "From collection to prediction")

    col1, col2 = st.columns(2)
    with col1:
        pipeline_step(1, "Data Collection", "9 sources: VGChartz, SteamSpy, RAWG, IGDB, HLTB, Wikipedia, Steam Store, OpenCritic, Gamedatacrunch")
        pipeline_step(2, "Merging & Cleaning", "Fuzzy matching, deduplication, missing value handling")
        pipeline_step(3, "Exploratory Analysis", "Distributions, correlations, temporal trends")
    with col2:
        pipeline_step(4, "Feature Engineering", "50 features: temporal, engagement, track record, market, critics, Steam Store, OpenCritic")
        pipeline_step(5, "Training", "7 models + Optuna tuning + stacking ensemble")
        pipeline_step(6, "Evaluation", "R², RMSE, MAE, SHAP, residuals, learning curves")

    st.divider()

    # Tech stack
    section_header("Tech Stack")
    c1, c2, c3 = st.columns(3)
    with c1:
        info_card(
            "Data & ML",
            "Python, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, SHAP",
        )
    with c2:
        info_card(
            "Visualization",
            "Plotly, Streamlit, matplotlib",
            accent="#8B5CF6",
        )
    with c3:
        info_card(
            "NLP & APIs",
            "Transformers (DistilBERT), RAWG, IGDB, SteamSpy, HLTB, Wikipedia, Steam Store, OpenCritic",
            accent="#10B981",
        )
