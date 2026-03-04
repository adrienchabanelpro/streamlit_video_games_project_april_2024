"""Feature Engineering page: dataset evolution, data quality, feature categories."""

import json

import pandas as pd
import streamlit as st
from components import info_card, section_header
from config import DATA_DIR, REPORTS_DIR


@st.cache_data
def _load_training_log() -> dict | None:
    """Load the most recent training log."""
    for name in ["training_log_v3.json", "training_log.json"]:
        path = REPORTS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


@st.cache_data
def _get_dataset_row_count() -> int:
    """Get the row count of the current dataset."""
    for csv_name in ["Ventes_jeux_video_v3.csv", "Ventes_jeux_video_final.csv"]:
        path = DATA_DIR / csv_name
        if path.exists():
            with open(path) as f:
                return sum(1 for _ in f) - 1
    return 0


_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "Year": "Game release year",
    "meta_score": "Metacritic score (professional critics, 0-100)",
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
    "steam_price": "Current Steam price (SteamSpy)",
    "steam_initialprice": "Launch price on Steam (SteamSpy)",
    "steam_review_pct": "Percentage of positive Steam reviews",
    "rawg_playtime": "Estimated average playtime (RAWG)",
    "rawg_ratings_count": "Number of RAWG community ratings",
    "rawg_rating": "RAWG community rating (0-5)",
    "rawg_metacritic": "Metacritic score via RAWG (0-100)",
    "hltb_main": "Time to finish the main story (hours)",
    "hltb_main_extra": "Main story + extras time (hours)",
    "hltb_completionist": "Time for 100% completion (hours)",
    "hltb_depth_ratio": "Completionist / main story ratio (game depth)",
    "release_month": "Release month (1-12)",
    "release_quarter": "Release quarter (1-4)",
    "is_holiday_release": "Released during the holiday season (Oct-Dec)",
    "cross_platform_count": "Number of supported platforms",
    "is_multi_platform": "Cross-platform game (yes/no)",
    "esrb_encoded": "ESRB age rating (ordinal encoded)",
    "has_franchise": "Belongs to a known franchise",
    "is_remake": "Is a remake",
    "is_remaster": "Is a remaster",
    "igdb_total_rating": "IGDB overall rating (0-100)",
    "igdb_hypes": "Pre-release hype score (IGDB)",
    "igdb_follows": "Number of game followers (IGDB)",
    "oc_top_critic_score": "OpenCritic top critic average score",
    "oc_percent_recommended": "Percentage of critics recommending the game (OpenCritic)",
    "critic_score_combined": "Weighted combination of Metacritic + OpenCritic scores",
    "steam_store_price_usd": "Current Steam Store price in USD",
    "has_dlc": "Game has downloadable content available",
    "steam_store_recommendations": "Number of Steam Store user recommendations",
    "has_verified_sales": "Game has verified (non-estimated) sales figures",
}

_FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Temporal": [
        "Year", "release_month", "release_quarter", "is_holiday_release",
    ],
    "Publisher/Developer Track Record": [
        "Publisher_encoded", "publisher_avg_sales_prior",
        "publisher_game_count_prior", "publisher_hit_rate",
        "developer_avg_sales_prior",
    ],
    "Genre/Platform History": [
        "Global_Sales_mean_genre", "Global_Sales_mean_platform",
        "Year_Global_Sales_mean_genre", "Year_Global_Sales_mean_platform",
        "Cumulative_Sales_Genre", "Cumulative_Sales_Platform",
        "competition_density", "genre_market_share",
    ],
    "Steam Engagement (SteamSpy)": [
        "review_count_total", "review_ratio", "playtime_avg",
        "concurrent_users", "steam_price", "steam_initialprice",
        "steam_review_pct",
    ],
    "RAWG Metadata": [
        "rawg_playtime", "rawg_ratings_count", "rawg_rating", "rawg_metacritic",
    ],
    "Completion Time (HLTB)": [
        "hltb_main", "hltb_main_extra", "hltb_completionist", "hltb_depth_ratio",
    ],
    "Cross-Platform": [
        "cross_platform_count", "is_multi_platform",
    ],
    "Game Characteristics": [
        "esrb_encoded", "has_franchise", "is_remake", "is_remaster",
    ],
    "IGDB": [
        "igdb_total_rating", "igdb_hypes", "igdb_follows",
    ],
    "OpenCritic": [
        "oc_top_critic_score", "oc_percent_recommended",
    ],
    "Critic Scores": [
        "meta_score", "user_review", "critic_score_combined",
    ],
    "Steam Store": [
        "steam_store_price_usd", "has_dlc", "steam_store_recommendations",
    ],
    "Data Quality": [
        "has_verified_sales",
    ],
}


def feature_engineering_page() -> None:
    """Render the feature engineering and pre-processing page."""
    st.title("Feature Engineering & Pre-processing")
    st.caption(
        "From raw multi-source data to 50 predictive features — "
        "data quality pipeline, encoding, and feature design"
    )

    log = _load_training_log()
    n_features = log.get("n_features", 50) if log else 50
    features = log.get("features", []) if log else []
    row_count = _get_dataset_row_count()

    # Dataset evolution
    section_header("Dataset Evolution")

    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "v2 — Initial Dataset",
            """
            <ul style="margin:0;padding-left:16px">
                <li><b>~64,000 rows</b> from VGChartz + SteamSpy</li>
                <li><b>10 features</b> for prediction</li>
                <li>2 data sources</li>
                <li>Simple target encoding + StandardScaler</li>
            </ul>
            """,
        )
    with c2:
        info_card(
            "v3 — Current Dataset",
            f"""
            <ul style="margin:0;padding-left:16px">
                <li><b>{row_count:,} rows</b> after quality filtering</li>
                <li><b>{n_features} features</b> for prediction</li>
                <li><b>9 data sources</b> merged via fuzzy matching</li>
                <li>Target encoding + StandardScaler + data quality tiers</li>
            </ul>
            """,
            accent="#10B981",
        )

    st.divider()

    # Data quality pipeline
    section_header("Data Quality Pipeline", "From 64K+ raw records to a clean dataset")

    st.write(
        f"""
        The v3 pipeline applies a rigorous quality process that reduces the raw
        dataset to **{row_count:,} high-quality game records**:

        1. **Source Tier Classification** — Each data source is ranked by reliability
           (Tier 1: official figures, Tier 2: APIs, Tier 3: estimates, Tier 4: scraped)
        2. **Zero-Sales Removal** — Games with zero or missing sales are excluded
        3. **Deduplication** — Exact match first, then fuzzy matching (85% threshold
           via rapidfuzz) with preference for the most complete record
        4. **Multi-Source Enrichment** — 9 sources merged via fuzzy matching to
           maximize feature coverage
        5. **Outlier Handling** — Winsorization of extreme values, log1p target transform
        """
    )

    st.divider()

    # Pre-processing
    section_header("Pre-processing")

    st.write(
        """
        - **Target Encoding** on Publisher — replaces ~600 unique values with mean
          sales (1 column instead of 576 one-hot columns)
        - **StandardScaler** — normalizes all numerical features
        - **log1p Transform** on the target variable (Global_Sales) to handle
          the heavy right skew
        - **Temporal Split** — train on games released ≤ 2015, test on games after
          2015 (no data leakage)
        """
    )

    st.divider()

    # Feature categories
    section_header(
        f"Feature Catalog ({n_features} Features)",
        "Organized by source and category",
    )

    if features:
        for category, cat_features in _FEATURE_CATEGORIES.items():
            # Only show features actually in the model
            active = [f for f in cat_features if f in features]
            if not active:
                continue
            with st.expander(f"**{category}** ({len(active)} features)"):
                rows = []
                for f in active:
                    rows.append({
                        "Feature": f,
                        "Description": _FEATURE_DESCRIPTIONS.get(f, "—"),
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

        # Check for uncategorized features
        categorized = {f for feats in _FEATURE_CATEGORIES.values() for f in feats}
        uncategorized = [f for f in features if f not in categorized]
        if uncategorized:
            with st.expander(f"**Other** ({len(uncategorized)} features)"):
                rows = [
                    {"Feature": f, "Description": _FEATURE_DESCRIPTIONS.get(f, "—")}
                    for f in uncategorized
                ]
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Run `make train` to generate the training log with feature details.")

    st.divider()

    # V2 historical context
    with st.expander("Historical: v2 Feature Set (10 features)"):
        st.write(
            """
            The v2 model used ~64,000 rows and **10 features**:

            `Year`, `meta_score`, `user_review`, `Publisher_encoded`,
            `Global_Sales_mean_genre`, `Global_Sales_mean_platform`,
            `Year_Global_Sales_mean_genre`, `Year_Global_Sales_mean_platform`,
            `Cumulative_Sales_Genre`, `Cumulative_Sales_Platform`

            The v3 pipeline expanded this to 50 features by integrating 7 additional
            data sources and engineering new variables around publisher/developer
            track record, Steam engagement, RAWG metadata, HLTB completion times,
            IGDB hype signals, OpenCritic scores, and Steam Store data.
            """
        )
