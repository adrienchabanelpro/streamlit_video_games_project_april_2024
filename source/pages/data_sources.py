"""Data Sources page: document all 5 sources, merge methodology, schema."""

import pandas as pd
import streamlit as st
from components import info_card, metric_card, section_header, source_card
from config import DATA_DIR


@st.cache_data
def _load_dataset_info() -> dict:
    """Load dataset for schema display."""
    for name in ["Ventes_jeux_video_v3.csv", "Ventes_jeux_video_final.csv"]:
        path = DATA_DIR / name
        if path.exists():
            df = pd.read_csv(path, nrows=5)
            with open(path) as f:
                row_count = sum(1 for _ in f) - 1
            return {"df_sample": df, "rows": row_count, "cols": len(df.columns), "name": name}
    return {"df_sample": pd.DataFrame(), "rows": 0, "cols": 0, "name": "N/A"}


def data_sources_page() -> None:
    """Render the Data Sources documentation page."""
    st.title("Data Sources")
    st.caption("Documentation of the 5 sources used and the merge methodology")

    info = _load_dataset_info()

    # Overview metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total Games", f"{info['rows']:,}", icon="🎮")
    with c2:
        metric_card("Columns", info["cols"], icon="📋")
    with c3:
        metric_card("Sources", "5", icon="🔗")

    st.divider()

    # Data sources
    section_header("Data Sources", "Each source provides complementary information")

    source_card(
        name="VGChartz (2024)",
        description=(
            "Primary dataset with worldwide physical sales (NA, EU, JP, Other, Global). "
            "Includes Metacritic scores, publishers, developers, release dates."
        ),
        row_count="64 000+",
        fields="Global_Sales, meta_score, Publisher, Genre, Platform, Year",
        url="https://www.vgchartz.com",
    )

    c1, c2 = st.columns(2)
    with c1:
        source_card(
            name="SteamSpy",
            description=(
                "PC digital sales estimates: number of owners, "
                "positive/negative reviews, playtime, price, concurrent users."
            ),
            row_count="60 000+",
            fields="owners, positive, negative, playtime, price, ccu",
            url="https://steamspy.com",
            accent="#8B5CF6",
        )
    with c2:
        source_card(
            name="RAWG API",
            description=(
                "Rich metadata: Metacritic scores (0-100), average playtime, "
                "ESRB rating, genres, tags, developers/publishers."
            ),
            row_count="500 000+",
            fields="metacritic, playtime, esrb_rating, genres, tags",
            url="https://rawg.io/apidocs",
            accent="#10B981",
        )

    c1, c2 = st.columns(2)
    with c1:
        source_card(
            name="IGDB / Twitch API",
            description=(
                "Unique data: themes (horror, sci-fi...), game modes, "
                "perspectives, franchises, game type (remake, remaster), pre-release hype."
            ),
            row_count="700 000+",
            fields="themes, game_modes, franchises, category, hypes, follows",
            url="https://api.igdb.com",
            accent="#F59E0B",
        )
    with c2:
        source_card(
            name="HowLongToBeat",
            description=(
                "Completion times: main story, extras, completionist. "
                "Indicator of game depth and replayability."
            ),
            row_count="~10 000",
            fields="main_story, main_extra, completionist",
            url="https://howlongtobeat.com",
            accent="#EF4444",
        )

    st.divider()

    # Merge methodology
    section_header("Merge Methodology", "How the 5 sources are combined")

    info_card(
        "Matching Strategy",
        """
        <ol style="margin:0;padding-left:20px">
            <li><b>Normalization</b>: lowercase names, removal of punctuation,
            edition suffixes (Remastered, GOTY...) and articles (The, A, Le...)</li>
            <li><b>Exact match</b>: direct matching of normalized names (fast)</li>
            <li><b>Fuzzy match</b>: rapidfuzz WRatio with 85% threshold for remaining names</li>
            <li><b>Deduplication</b>: VGChartz preferred for sales, RAWG for metadata,
            IGDB for themes/franchises, HLTB for completion times</li>
        </ol>
        """,
    )

    st.divider()

    # Schema
    section_header("Dataset Schema", f"File: {info['name']}")

    if not info["df_sample"].empty:
        # Column types and descriptions
        schema_data = []
        for col in info["df_sample"].columns:
            dtype = str(info["df_sample"][col].dtype)
            source = _infer_source(col)
            schema_data.append({"Column": col, "Type": dtype, "Source": source})

        st.dataframe(
            pd.DataFrame(schema_data),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Data preview (first 5 rows)"):
            st.dataframe(info["df_sample"], use_container_width=True, hide_index=True)


def _infer_source(col: str) -> str:
    """Infer the data source from column name prefix."""
    if col.startswith("steam_"):
        return "SteamSpy"
    if col.startswith("rawg_"):
        return "RAWG"
    if col.startswith("igdb_"):
        return "IGDB"
    if col.startswith("hltb_"):
        return "HLTB"
    if col in ("cross_platform_count", "is_multi_platform", "release_month",
               "release_quarter", "release_day_of_week", "esrb_encoded",
               "has_franchise", "is_remake", "is_remaster", "price_tier"):
        return "Derived"
    return "VGChartz"
