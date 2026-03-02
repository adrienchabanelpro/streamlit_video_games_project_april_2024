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
    st.title("Sources de Donnees")
    st.caption("Documentation des 5 sources utilisees et de la methodologie de fusion")

    info = _load_dataset_info()

    # Overview metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Jeux totaux", f"{info['rows']:,}", icon="🎮")
    with c2:
        metric_card("Colonnes", info["cols"], icon="📋")
    with c3:
        metric_card("Sources", "5", icon="🔗")

    st.divider()

    # Data sources
    section_header("Sources de donnees", "Chaque source apporte des informations complementaires")

    source_card(
        name="VGChartz (2024)",
        description=(
            "Base principale avec les ventes physiques mondiales (NA, EU, JP, Other, Global). "
            "Inclut les scores Metacritic, editeurs, developpeurs, dates de sortie."
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
                "Estimations des ventes digitales PC : nombre de proprietaires, "
                "avis positifs/negatifs, temps de jeu, prix, joueurs simultanes."
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
                "Metadonnees riches : scores Metacritic (0-100), temps de jeu moyen, "
                "classification ESRB, genres, tags, developpeurs/editeurs."
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
                "Donnees uniques : themes (horreur, sci-fi...), modes de jeu, "
                "perspectives, franchises, type de jeu (remake, remaster), hype pre-sortie."
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
                "Temps de completion : histoire principale, extras, completionniste. "
                "Indicateur de profondeur et rejouabilite du jeu."
            ),
            row_count="~10 000",
            fields="main_story, main_extra, completionist",
            url="https://howlongtobeat.com",
            accent="#EF4444",
        )

    st.divider()

    # Merge methodology
    section_header("Methodologie de fusion", "Comment les 5 sources sont combinees")

    info_card(
        "Strategie de matching",
        """
        <ol style="margin:0;padding-left:20px">
            <li><b>Normalisation</b> : noms en minuscules, suppression de la ponctuation,
            des suffixes d'edition (Remastered, GOTY...) et des articles (The, A, Le...)</li>
            <li><b>Match exact</b> : correspondance directe des noms normalises (rapide)</li>
            <li><b>Match flou</b> : rapidfuzz WRatio avec seuil de 85% pour les noms restants</li>
            <li><b>Deduplication</b> : preference VGChartz pour les ventes, RAWG pour les metadonnees,
            IGDB pour les themes/franchises, HLTB pour les temps de completion</li>
        </ol>
        """,
    )

    st.divider()

    # Schema
    section_header("Schema du dataset", f"Fichier : {info['name']}")

    if not info["df_sample"].empty:
        # Column types and descriptions
        schema_data = []
        for col in info["df_sample"].columns:
            dtype = str(info["df_sample"][col].dtype)
            source = _infer_source(col)
            schema_data.append({"Colonne": col, "Type": dtype, "Source": source})

        st.dataframe(
            pd.DataFrame(schema_data),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Apercu des donnees (5 premieres lignes)"):
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
        return "Derive"
    return "VGChartz"
