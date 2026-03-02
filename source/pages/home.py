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
    st.title("Prediction des Ventes de Jeux Video")
    st.caption(
        "Projet de Data Science : collecte, analyse et prediction des ventes "
        "mondiales de jeux video a partir de sources multiples"
    )

    stats = _load_overview_stats()

    # Key metrics row
    section_header("Vue d'ensemble")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Jeux dans le dataset", f"{stats['rows']:,}", icon="🎮")
    with c2:
        metric_card("Colonnes", stats["cols"], icon="📊")
    with c3:
        metric_card("Features ML", stats["features"], icon="⚙️")
    with c4:
        metric_card("Meilleur R²", f"{stats['best_r2']:.3f}", icon="🎯")

    st.divider()

    # Project description
    section_header("Objectif du projet")
    st.write(
        """
        Ce projet utilise le **Machine Learning** pour predire les ventes mondiales
        de jeux video. Il combine **5 sources de donnees** (VGChartz, SteamSpy,
        RAWG, IGDB, HowLongToBeat) et un **ensemble de modeles** (LightGBM, XGBoost,
        CatBoost, Random Forest, HistGradientBoosting) avec un **meta-learner Ridge**
        (stacking) pour des predictions optimales.

        Chaque etape est documentee dans les pages correspondantes.
        """
    )

    st.divider()

    # Pipeline overview
    section_header("Pipeline de donnees", "De la collecte a la prediction")

    col1, col2 = st.columns(2)
    with col1:
        pipeline_step(1, "Collecte de donnees", "5 sources: VGChartz, SteamSpy, RAWG API, IGDB API, HLTB")
        pipeline_step(2, "Fusion & nettoyage", "Fuzzy matching, deduplication, gestion des valeurs manquantes")
        pipeline_step(3, "Analyse exploratoire", "Distributions, correlations, tendances temporelles")
    with col2:
        pipeline_step(4, "Feature Engineering", "30+ variables : temporelles, engagement, track record, marche")
        pipeline_step(5, "Entrainement", "7 modeles + Optuna tuning + stacking ensemble")
        pipeline_step(6, "Evaluation", "R², RMSE, MAE, SHAP, residus, courbes d'apprentissage")

    st.divider()

    # Tech stack
    section_header("Stack technique")
    c1, c2, c3 = st.columns(3)
    with c1:
        info_card(
            "Data & ML",
            "Python, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, SHAP",
        )
    with c2:
        info_card(
            "Visualisation",
            "Plotly, Streamlit, matplotlib",
            accent="#8B5CF6",
        )
    with c3:
        info_card(
            "NLP & APIs",
            "Transformers (DistilBERT), RAWG, IGDB, SteamSpy, HLTB",
            accent="#10B981",
        )
