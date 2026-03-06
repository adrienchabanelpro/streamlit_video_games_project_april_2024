"""Data loading, cleaning, splitting, and feature engineering for v3 pipeline.

Handles both v2 (10 features) and v3 (30+ features) datasets.
All feature engineering is computed from training data only (no leakage).
"""

from __future__ import annotations

import logging
from pathlib import Path

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

TARGET = "Global_Sales"
TARGET_ESTIMATED = "estimated_total_sales"
RANDOM_STATE = 42

# Sample weights by sales confidence tier
CONFIDENCE_WEIGHTS = {
    "high": 2.0,
    "medium": 1.0,
    "none": 0.5,
}

# Regional sales columns — NEVER used as features (data leakage)
REGIONAL_COLS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

# Columns never used as features
DROP_COLS = [
    "Rank", "Name", "img", "release_date", "last_update",
    "rawg_slug", "rawg_genres", "rawg_platforms", "rawg_tags_top5",
    "rawg_developers", "rawg_publishers",
    "igdb_slug", "igdb_developers", "igdb_publishers",
    "igdb_themes", "igdb_game_modes", "igdb_perspectives", "igdb_franchises",
    "steam_appid", "steam_owners", "steam_tags",
    "query_name", "hltb_name",
    # Sales estimation metadata (not features)
    "quality_tier", "sales_provenance", "sales_confidence",
    "review_estimated_sales", "review_multiplier",
    "wiki_sales_millions", "wiki_platform", "wiki_publisher",
    "wiki_developer", "wiki_release_date", "wiki_source_page", "wiki_sales_type",
    "steam_store_appid",
]


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the clean dataset (preferred), v3, or v2 fallback."""
    if path is None:
        clean = DATA_DIR / "Ventes_jeux_video_clean.csv"
        v3 = DATA_DIR / "Ventes_jeux_video_v3.csv"
        v2 = DATA_DIR / "Ventes_jeux_video_final.csv"
        path = clean if clean.exists() else (v3 if v3.exists() else v2)

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} cols from {path.name}")
    return df


def clean_data(
    df: pd.DataFrame,
    target: str = TARGET,
) -> pd.DataFrame:
    """Clean raw data: drop leakage, handle types, remove NaN/zero target.

    Parameters
    ----------
    target:
        Target column to use. Either TARGET (Global_Sales) or
        TARGET_ESTIMATED (estimated_total_sales).
    """
    df = df.copy()

    # Drop rows missing target or key categoricals
    df = df.dropna(subset=[target])
    df = df.dropna(subset=["Publisher", "Year"])

    # Filter zero-sales rows: use quality tiers if available, else simple > 0
    if "quality_tier" in df.columns:
        if target == TARGET_ESTIMATED:
            # Include tiers 1-4 (verified + physical + estimated + marginal)
            valid_tiers = [
                "tier_1_verified", "tier_2_physical",
                "tier_3_estimated", "tier_4_marginal",
            ]
        else:
            # Original behavior: only tiers with VGChartz physical sales
            valid_tiers = [
                "tier_1_verified", "tier_2_physical",
                "tier_3_estimated", "tier_4_marginal",
            ]
        before = len(df)
        df = df[df["quality_tier"].isin(valid_tiers)]
        logger.info(f"Quality tier filter: {before:,} → {len(df):,} rows")
    else:
        before = len(df)
        df = df[df[target] > 0]
        logger.info(f"Zero-sales filter: {before:,} → {len(df):,} rows")

    # Filter to rows with positive target
    before = len(df)
    df = df[df[target] > 0]
    if len(df) < before:
        logger.info(f"Zero-target filter: {before:,} → {len(df):,} rows")

    # Build sample weights from confidence tiers
    if "sales_confidence" in df.columns:
        df["sample_weight"] = df["sales_confidence"].map(CONFIDENCE_WEIGHTS).fillna(1.0)
    else:
        df["sample_weight"] = 1.0

    # Convert Year to int
    df["Year"] = df["Year"].astype(int)

    # Drop regional sales (leakage) and non-feature columns
    # Also drop the non-active target to avoid leakage
    other_target = TARGET_ESTIMATED if target == TARGET else TARGET
    cols_to_drop = REGIONAL_COLS + DROP_COLS + [other_target]
    cols_to_drop += [c for c in df.columns if c.endswith("_match_score")]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    logger.info(f"After cleaning: {len(df):,} rows, {len(df.columns)} cols")
    return df


def temporal_split(
    df: pd.DataFrame, split_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train <= split_year, test > split_year."""
    train = df[df["Year"] <= split_year].copy()
    test = df[df["Year"] > split_year].copy()
    logger.info(f"Split at {split_year}: train={len(train):,}, test={len(test):,}")
    return train, test


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_train_stats(df_train: pd.DataFrame, target: str = TARGET) -> dict:
    """Compute all feature engineering statistics from training data only."""
    stats: dict = {}

    # Mean sales by Genre / Platform
    stats["genre_means"] = df_train.groupby("Genre")[target].mean().to_dict()
    stats["platform_means"] = df_train.groupby("Platform")[target].mean().to_dict()
    stats["global_sales_mean"] = float(df_train[target].mean())

    # Cumulative sales by Genre and Year
    stats["cumsum_genre"] = {}
    for genre in df_train["Genre"].unique():
        data = (
            df_train[df_train["Genre"] == genre]
            .groupby("Year")[target].sum().sort_index().cumsum()
        )
        stats["cumsum_genre"][genre] = data.to_dict()

    stats["cumsum_platform"] = {}
    for platform in df_train["Platform"].unique():
        data = (
            df_train[df_train["Platform"] == platform]
            .groupby("Year")[target].sum().sort_index().cumsum()
        )
        stats["cumsum_platform"][platform] = data.to_dict()

    # --- v3 track record features ---
    # Publisher historical performance (train data only, per year)
    pub_stats = df_train.groupby("Publisher").agg(
        pub_avg_sales=(target, "mean"),
        pub_game_count=(target, "count"),
        pub_total_sales=(target, "sum"),
    ).to_dict("index")
    stats["publisher_stats"] = pub_stats

    # Publisher hit rate (% of games above median)
    median_sales = df_train[target].median()
    stats["median_sales"] = float(median_sales)

    pub_hits: dict[str, float] = {}
    for pub, group in df_train.groupby("Publisher"):
        hits = (group[target] > median_sales).sum()
        pub_hits[str(pub)] = float(hits / len(group)) if len(group) > 0 else 0.0
    stats["publisher_hit_rate"] = pub_hits

    # Developer historical performance
    if "developer" in df_train.columns:
        dev_stats = df_train.groupby("developer").agg(
            dev_avg_sales=(target, "mean"),
            dev_game_count=(target, "count"),
        ).to_dict("index")
        stats["developer_stats"] = dev_stats

    # Competition density: games per year-month (use Year for now)
    games_per_year = df_train.groupby("Year").size().to_dict()
    stats["games_per_year"] = games_per_year

    # Genre game count per year
    genre_year_count = df_train.groupby(["Genre", "Year"]).size().to_dict()
    stats["genre_year_count"] = {f"{g}_{y}": v for (g, y), v in genre_year_count.items()}

    # Genre market share per year
    total_by_year = df_train.groupby("Year")[target].sum().to_dict()
    stats["total_sales_by_year"] = total_by_year

    genre_sales_by_year: dict[str, dict] = {}
    for (genre, year), group in df_train.groupby(["Genre", "Year"]):
        if genre not in genre_sales_by_year:
            genre_sales_by_year[str(genre)] = {}
        total = total_by_year.get(year, 1)
        genre_sales_by_year[str(genre)][int(year)] = float(group[target].sum() / total) if total > 0 else 0.0
    stats["genre_market_share"] = genre_sales_by_year

    # Dropdown values for prediction UI
    stats["publishers"] = sorted(df_train["Publisher"].unique().tolist())
    stats["genres"] = sorted(df_train["Genre"].unique().tolist())
    stats["platforms"] = sorted(df_train["Platform"].unique().tolist())
    stats["meta_score_mean"] = float(df_train["meta_score"].mean()) if "meta_score" in df_train.columns else 0.0
    stats["user_review_mean"] = float(df_train["user_review"].mean()) if "user_review" in df_train.columns else 0.0

    return stats


def _lookup_cumulative(cumsum_dict: dict, category: str, year: int) -> float:
    """Look up cumulative sales for category up to year."""
    if category not in cumsum_dict:
        return 0.0
    yearly = cumsum_dict[category]
    relevant = [y for y in yearly if y <= year]
    return yearly[max(relevant)] if relevant else 0.0


def engineer_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply all feature engineering using pre-computed training statistics.

    Works for both v2 (10 features) and v3 (30+ features) datasets.
    """
    df = df.copy()
    gm = stats["global_sales_mean"]

    # --- Core v2 features (always available) ---
    df["Global_Sales_mean_genre"] = df["Genre"].map(stats["genre_means"]).fillna(gm)
    df["Global_Sales_mean_platform"] = df["Platform"].map(stats["platform_means"]).fillna(gm)
    df["Year_Global_Sales_mean_genre"] = df["Year"] * df["Global_Sales_mean_genre"]
    df["Year_Global_Sales_mean_platform"] = df["Year"] * df["Global_Sales_mean_platform"]

    df["Cumulative_Sales_Genre"] = df.apply(
        lambda r: _lookup_cumulative(stats["cumsum_genre"], r["Genre"], r["Year"]),
        axis=1,
    )
    df["Cumulative_Sales_Platform"] = df.apply(
        lambda r: _lookup_cumulative(stats["cumsum_platform"], r["Platform"], r["Year"]),
        axis=1,
    )

    # Fill missing meta_score / user_review
    if "meta_score" in df.columns:
        df["meta_score"] = df["meta_score"].fillna(df["meta_score"].median())
    if "user_review" in df.columns:
        df["user_review"] = df["user_review"].fillna(df["user_review"].median())

    # --- v3 Publisher/Developer track record ---
    pub_stats = stats.get("publisher_stats", {})
    df["publisher_avg_sales_prior"] = df["Publisher"].map(
        lambda p: pub_stats.get(p, {}).get("pub_avg_sales", gm)
    )
    df["publisher_game_count_prior"] = df["Publisher"].map(
        lambda p: pub_stats.get(p, {}).get("pub_game_count", 0)
    )
    df["publisher_hit_rate"] = df["Publisher"].map(
        stats.get("publisher_hit_rate", {})
    ).fillna(0.0)

    if "developer_stats" in stats and "developer" in df.columns:
        dev_stats = stats["developer_stats"]
        df["developer_avg_sales_prior"] = df["developer"].map(
            lambda d: dev_stats.get(d, {}).get("dev_avg_sales", gm)
        )
    elif "developer" in df.columns:
        df["developer_avg_sales_prior"] = gm

    # --- v3 Market context ---
    games_per_year = stats.get("games_per_year", {})
    df["competition_density"] = df["Year"].map(games_per_year).fillna(0)

    genre_market = stats.get("genre_market_share", {})
    df["genre_market_share"] = df.apply(
        lambda r: genre_market.get(r["Genre"], {}).get(int(r["Year"]), 0.0),
        axis=1,
    )

    # --- v3 Enrichment features (only if columns exist) ---
    # Steam engagement features
    if "steam_positive" in df.columns and "steam_negative" in df.columns:
        pos = pd.to_numeric(df["steam_positive"], errors="coerce").fillna(0)
        neg = pd.to_numeric(df["steam_negative"], errors="coerce").fillna(0)
        df["review_count_total"] = pos + neg
        total = pos + neg
        df["review_ratio"] = np.where(total > 0, pos / total, np.nan)
        df["review_ratio"] = df["review_ratio"].fillna(df["review_ratio"].median())

    if "steam_average_forever" in df.columns:
        df["playtime_avg"] = pd.to_numeric(df["steam_average_forever"], errors="coerce").fillna(0)

    if "steam_ccu" in df.columns:
        df["concurrent_users"] = pd.to_numeric(df["steam_ccu"], errors="coerce").fillna(0)

    # RAWG features
    if "rawg_playtime" in df.columns:
        df["rawg_playtime"] = pd.to_numeric(df["rawg_playtime"], errors="coerce").fillna(0)

    if "rawg_ratings_count" in df.columns:
        df["rawg_ratings_count"] = pd.to_numeric(df["rawg_ratings_count"], errors="coerce").fillna(0)

    if "rawg_rating" in df.columns:
        df["rawg_rating"] = pd.to_numeric(df["rawg_rating"], errors="coerce").fillna(0)

    if "rawg_metacritic" in df.columns:
        df["rawg_metacritic"] = pd.to_numeric(df["rawg_metacritic"], errors="coerce").fillna(0)

    # HLTB features
    for hltb_col in ["hltb_main", "hltb_main_extra", "hltb_completionist"]:
        if hltb_col in df.columns:
            df[hltb_col] = pd.to_numeric(df[hltb_col], errors="coerce").fillna(0)

    if "hltb_main" in df.columns and "hltb_completionist" in df.columns:
        main = df["hltb_main"].replace(0, np.nan)
        df["hltb_depth_ratio"] = (df["hltb_completionist"] / main).fillna(0)

    # Temporal features (from RAWG release date or derived columns)
    if "release_month" in df.columns:
        df["release_month"] = pd.to_numeric(df["release_month"], errors="coerce").fillna(6)
        df["is_holiday_release"] = df["release_month"].isin([10, 11, 12]).astype(int)
    if "release_quarter" in df.columns:
        df["release_quarter"] = pd.to_numeric(df["release_quarter"], errors="coerce").fillna(2)

    # Cross-platform
    if "cross_platform_count" in df.columns:
        df["cross_platform_count"] = pd.to_numeric(df["cross_platform_count"], errors="coerce").fillna(1)
        df["is_multi_platform"] = (df["cross_platform_count"] > 1).astype(int)

    # ESRB encoding
    if "esrb_encoded" in df.columns:
        df["esrb_encoded"] = pd.to_numeric(df["esrb_encoded"], errors="coerce").fillna(-1)

    # Franchise / game type from IGDB
    if "has_franchise" in df.columns:
        df["has_franchise"] = pd.to_numeric(df["has_franchise"], errors="coerce").fillna(0)
    if "is_remake" in df.columns:
        df["is_remake"] = pd.to_numeric(df["is_remake"], errors="coerce").fillna(0)
    if "is_remaster" in df.columns:
        df["is_remaster"] = pd.to_numeric(df["is_remaster"], errors="coerce").fillna(0)

    # IGDB ratings
    if "igdb_total_rating" in df.columns:
        df["igdb_total_rating"] = pd.to_numeric(df["igdb_total_rating"], errors="coerce").fillna(0)
    if "igdb_hypes" in df.columns:
        df["igdb_hypes"] = pd.to_numeric(df["igdb_hypes"], errors="coerce").fillna(0)
    if "igdb_follows" in df.columns:
        df["igdb_follows"] = pd.to_numeric(df["igdb_follows"], errors="coerce").fillna(0)

    # Price features (SteamSpy)
    if "steam_price" in df.columns:
        df["steam_price"] = pd.to_numeric(df["steam_price"], errors="coerce").fillna(0)
    if "steam_initialprice" in df.columns:
        df["steam_initialprice"] = pd.to_numeric(df["steam_initialprice"], errors="coerce").fillna(0)
    if "steam_review_pct" in df.columns:
        df["steam_review_pct"] = pd.to_numeric(df["steam_review_pct"], errors="coerce").fillna(0)

    # --- v3 OpenCritic features ---
    if "oc_top_critic_score" in df.columns:
        df["oc_top_critic_score"] = pd.to_numeric(df["oc_top_critic_score"], errors="coerce").fillna(0)
    if "oc_percent_recommended" in df.columns:
        df["oc_percent_recommended"] = pd.to_numeric(df["oc_percent_recommended"], errors="coerce").fillna(0)

    # Critic consensus: prefer OpenCritic, fallback to meta_score
    if "critic_score_combined" in df.columns:
        df["critic_score_combined"] = pd.to_numeric(df["critic_score_combined"], errors="coerce").fillna(0)

    # --- v3 Steam Store features ---
    if "steam_store_price_usd" in df.columns:
        df["steam_store_price_usd"] = pd.to_numeric(df["steam_store_price_usd"], errors="coerce").fillna(0)
    if "has_dlc" in df.columns:
        df["has_dlc"] = pd.to_numeric(df["has_dlc"], errors="coerce").fillna(0)
    if "steam_store_recommendations" in df.columns:
        df["steam_store_recommendations"] = pd.to_numeric(df["steam_store_recommendations"], errors="coerce").fillna(0)

    # --- v3 Quality flags ---
    if "has_verified_sales" in df.columns:
        df["has_verified_sales"] = pd.to_numeric(df["has_verified_sales"], errors="coerce").fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Detect available numerical feature columns from a DataFrame.

    Returns columns in consistent order. This auto-detects which features
    are available (v2 vs v3 dataset).
    """
    # Always-available v2 core features
    core = [
        "Year", "meta_score", "user_review",
        "Global_Sales_mean_genre", "Global_Sales_mean_platform",
        "Year_Global_Sales_mean_genre", "Year_Global_Sales_mean_platform",
        "Cumulative_Sales_Genre", "Cumulative_Sales_Platform",
        "Publisher_encoded",
    ]

    # v3 engineered features (added if present)
    v3_features = [
        "publisher_avg_sales_prior", "publisher_game_count_prior", "publisher_hit_rate",
        "developer_avg_sales_prior",
        "competition_density", "genre_market_share",
        # Steam enrichment (SteamSpy)
        "review_count_total", "review_ratio", "playtime_avg", "concurrent_users",
        "steam_price", "steam_initialprice", "steam_review_pct",
        # RAWG enrichment
        "rawg_playtime", "rawg_ratings_count", "rawg_rating", "rawg_metacritic",
        # HLTB
        "hltb_main", "hltb_main_extra", "hltb_completionist", "hltb_depth_ratio",
        # Temporal
        "release_month", "release_quarter", "is_holiday_release",
        # Cross-platform
        "cross_platform_count", "is_multi_platform",
        # ESRB
        "esrb_encoded",
        # IGDB
        "has_franchise", "is_remake", "is_remaster",
        "igdb_total_rating", "igdb_hypes", "igdb_follows",
        # OpenCritic
        "oc_top_critic_score", "oc_percent_recommended", "critic_score_combined",
        # Steam Store
        "steam_store_price_usd", "has_dlc", "steam_store_recommendations",
        # Quality flags
        "has_verified_sales",
    ]

    features = list(core)
    for f in v3_features:
        if f in df.columns:
            features.append(f)

    return features


def prepare_training_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    stats: dict,
    log_transform: bool = True,
    target: str = TARGET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler, ce.TargetEncoder]:
    """Full preparation: feature engineering + encoding + scaling.

    Returns
    -------
    X_train, y_train, X_test, y_test_raw, w_train, feature_names, scaler, encoder
    """
    # Feature engineering
    df_train = engineer_features(df_train, stats)
    df_test = engineer_features(df_test, stats)

    # Target-encode Publisher
    encoder = ce.TargetEncoder(cols=["Publisher"], smoothing=10)
    df_train["Publisher_encoded"] = encoder.fit_transform(
        df_train[["Publisher"]], df_train[target]
    )["Publisher"]
    df_test["Publisher_encoded"] = encoder.transform(df_test[["Publisher"]])["Publisher"]

    # Detect available features
    features = get_feature_columns(df_train)
    logger.info(f"Using {len(features)} features: {features}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    X_test = scaler.transform(df_test[features])

    y_train_raw = df_train[target].values
    y_test_raw = df_test[target].values

    # Sample weights
    w_train = df_train["sample_weight"].values if "sample_weight" in df_train.columns else np.ones(len(df_train))

    if log_transform:
        y_train = np.log1p(y_train_raw)
    else:
        y_train = y_train_raw

    return X_train, y_train, X_test, y_test_raw, w_train, features, scaler, encoder
