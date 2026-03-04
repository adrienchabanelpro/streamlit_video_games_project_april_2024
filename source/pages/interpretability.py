"""Interpretability page: SHAP analysis, feature importance, individual explanations."""

import json

import pandas as pd
import streamlit as st
from components import info_card, section_header
from config import DATA_DIR, REPORTS_DIR


@st.cache_data
def _load_training_log() -> dict | None:
    for name in ["training_log_v3.json", "training_log.json"]:
        path = REPORTS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


@st.cache_data
def _load_dataset() -> pd.DataFrame:
    for name in ["Ventes_jeux_video_v3.csv", "Ventes_jeux_video_final.csv"]:
        path = DATA_DIR / name
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def interpretability_page() -> None:
    """Render the model interpretability page."""
    st.title("Model Interpretability")
    st.caption("Understanding why the model makes its predictions: SHAP, feature importance")

    log = _load_training_log()

    # SHAP section
    section_header("SHAP (SHapley Additive exPlanations)")

    info_card(
        "What is SHAP?",
        """
        SHAP assigns each feature a <b>marginal contribution</b> to the prediction.
        Unlike classical importance (based on gain), SHAP provides
        <b>consistent and local</b> explanations: each prediction can be explained
        individually, not just the model as a whole.
        """,
    )

    shap_bar = REPORTS_DIR / "shap_bar_v3.png"
    if not shap_bar.exists():
        shap_bar = REPORTS_DIR / "shap_bar.png"
    shap_summary = REPORTS_DIR / "shap_summary_v3.png"
    if not shap_summary.exists():
        shap_summary = REPORTS_DIR / "shap_summary.png"

    if shap_bar.exists() or shap_summary.exists():
        tab1, tab2 = st.tabs(["Global Importance (Bar)", "Distribution (Beeswarm)"])
        with tab1:
            if shap_bar.exists():
                st.image(str(shap_bar), use_container_width=True)
                st.caption(
                    "Each bar represents the mean absolute contribution "
                    "of the feature to predictions. The longer the bar, "
                    "the more important the feature."
                )
        with tab2:
            if shap_summary.exists():
                st.image(str(shap_summary), use_container_width=True)
                st.caption(
                    "Each dot represents a game. The color indicates the feature "
                    "value (red = high, blue = low). The horizontal position "
                    "shows the impact on the prediction."
                )
    else:
        st.info("SHAP plots will be generated after training (`make train`).")

    st.divider()

    # Feature descriptions
    section_header("Feature Descriptions", "What each feature measures and why it is useful")

    if log:
        features = log.get("features", [])
        if features:
            feature_desc = _get_feature_descriptions()
            rows = []
            for f in features:
                rows.append({
                    "Feature": f,
                    "Description": feature_desc.get(f, "—"),
                    "Category": _categorize_feature(f),
                })
            df_features = pd.DataFrame(rows)

            # Group by category
            for cat in df_features["Category"].unique():
                with st.expander(f"**{cat}** ({len(df_features[df_features['Category'] == cat])} features)"):
                    subset = df_features[df_features["Category"] == cat][["Feature", "Description"]]
                    st.dataframe(subset, use_container_width=True, hide_index=True)

    st.divider()

    # Methodology explanation
    section_header("Interpretability Methodology")

    info_card(
        "Multi-Level Approach",
        """
        <b>1. SHAP TreeExplainer</b> — Exact computation of SHAP values for tree-based
        models (LightGBM, XGBoost, CatBoost). Complexity O(TLD²).<br><br>
        <b>2. Permutation Importance</b> — Measures performance degradation when
        a feature is randomly shuffled. Model-agnostic.<br><br>
        <b>3. Residual Analysis</b> — Verifies that errors are random and
        not systematic (no structural bias).
        """,
    )


def _get_feature_descriptions() -> dict[str, str]:
    """Return human-readable descriptions for known features."""
    return {
        "Year": "Game release year",
        "meta_score": "Metacritic score (professional critics)",
        "user_review": "User review score",
        "Global_Sales_mean_genre": "Average sales of games in the same genre (training data)",
        "Global_Sales_mean_platform": "Average sales of games on the same platform",
        "Year_Global_Sales_mean_genre": "Interaction: Year x average genre sales",
        "Year_Global_Sales_mean_platform": "Interaction: Year x average platform sales",
        "Cumulative_Sales_Genre": "Cumulative genre sales up to the release year",
        "Cumulative_Sales_Platform": "Cumulative platform sales up to the release year",
        "Publisher_encoded": "Publisher encoded via target encoding (publisher average sales)",
        "publisher_avg_sales_prior": "Average sales of the publisher's previous games",
        "publisher_game_count_prior": "Number of the publisher's previous games",
        "publisher_hit_rate": "Publisher hit rate (sales > median)",
        "developer_avg_sales_prior": "Average sales of the developer's previous games",
        "competition_density": "Number of games released the same year",
        "genre_market_share": "Genre market share in the release year",
        "review_count_total": "Total number of Steam reviews (positive + negative)",
        "review_ratio": "Ratio of positive Steam reviews",
        "playtime_avg": "Average playtime on Steam (minutes)",
        "concurrent_users": "Peak concurrent players on Steam",
        "rawg_playtime": "Estimated average playtime (RAWG)",
        "rawg_ratings_count": "Number of RAWG community ratings",
        "rawg_rating": "RAWG community rating (0-5)",
        "rawg_metacritic": "Metacritic score via RAWG (0-100)",
        "hltb_main": "Time to finish the main story (hours)",
        "hltb_main_extra": "Main story + extras time (hours)",
        "hltb_completionist": "Time for 100% completion (hours)",
        "hltb_depth_ratio": "Completionist / main story ratio (depth)",
        "release_month": "Release month (1-12)",
        "release_quarter": "Release quarter (1-4)",
        "is_holiday_release": "Released during the holiday season (Oct-Dec)",
        "cross_platform_count": "Number of supported platforms",
        "is_multi_platform": "Cross-platform game (yes/no)",
        "esrb_encoded": "ESRB age rating (ordinal)",
        "has_franchise": "Belongs to a known franchise",
        "is_remake": "Is a remake",
        "is_remaster": "Is a remaster",
        "igdb_total_rating": "IGDB overall rating (0-100)",
        "igdb_hypes": "Pre-release hype score (IGDB)",
        "igdb_follows": "Number of game followers (IGDB)",
        "steam_price": "Current Steam price",
        "steam_initialprice": "Launch price on Steam",
        "steam_review_pct": "Percentage of positive Steam reviews",
        "oc_top_critic_score": "OpenCritic top critic average score",
        "oc_percent_recommended": "Percentage of critics recommending the game (OpenCritic)",
        "critic_score_combined": "Weighted combination of Metacritic + OpenCritic scores",
        "steam_store_price_usd": "Current Steam Store price in USD",
        "has_dlc": "Game has downloadable content available",
        "steam_store_recommendations": "Number of Steam Store user recommendations",
        "has_verified_sales": "Game has verified (non-estimated) sales figures",
    }


def _categorize_feature(name: str) -> str:
    """Categorize a feature by its type."""
    if name in ("Year", "release_month", "release_quarter", "is_holiday_release"):
        return "Temporal"
    if "publisher" in name or "developer" in name or "Publisher" in name:
        return "Publisher/Developer Track Record"
    if name.startswith(("Global_Sales_mean", "Year_Global_Sales", "Cumulative")):
        return "Genre/Platform History"
    if name.startswith(("review_", "playtime", "concurrent", "steam_review_pct")):
        return "Steam Engagement"
    if name.startswith("rawg_"):
        return "RAWG Metadata"
    if name.startswith("hltb_"):
        return "Completion Time (HLTB)"
    if name.startswith(("cross_platform", "is_multi")):
        return "Cross-Platform"
    if name.startswith(("esrb", "has_franchise", "is_remake", "is_remaster")):
        return "Game Characteristics"
    if name.startswith("igdb_"):
        return "IGDB"
    if name.startswith("oc_"):
        return "OpenCritic"
    if name == "critic_score_combined":
        return "Critic Scores"
    if name.startswith("steam_store_") or name == "has_dlc":
        return "Steam Store"
    if name == "has_verified_sales":
        return "Data Quality"
    if name.startswith("steam_") and "price" in name:
        return "Price"
    if name in ("meta_score", "user_review"):
        return "Critic Scores"
    return "Other"
