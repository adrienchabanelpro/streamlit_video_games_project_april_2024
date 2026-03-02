"""Historical trend explorer: interactive timeline of genres, platforms, and publishers."""

import pandas as pd
import plotly.express as px
import streamlit as st
from config import DATA_DIR, PLOTLY_LAYOUT
from data_validation import validate_dataframe


@st.cache_data
def _load_data() -> pd.DataFrame:
    """Load and prepare the main dataset for trend analysis."""
    df = pd.read_csv(DATA_DIR / "Ventes_jeux_video_final.csv")
    # Advisory validation — warn but continue
    is_valid, errors = validate_dataframe(df)
    if not is_valid:
        st.warning(
            f"Validation des donnees : {len(errors)} probleme(s) detecte(s). "
            "Les tendances peuvent etre affectees."
        )
    df = df.dropna(subset=["Year", "Publisher", "Genre", "Platform"])
    df["Year"] = df["Year"].astype(int)
    return df


def trends_page() -> None:
    """Historical trend explorer page."""
    st.title("Explorateur de tendances historiques")
    st.write(
        "Visualisez comment les genres, plateformes et editeurs ont evolue "
        "au fil des decennies dans l'industrie du jeu video."
    )

    with st.spinner("Chargement des donnees..."):
        df = _load_data()

    st.markdown("---")

    # ------------------------------------------------------------------
    # View selector
    # ------------------------------------------------------------------
    view = st.radio(
        "Dimension a explorer",
        ["Genres", "Plateformes", "Editeurs"],
        horizontal=True,
        key="trend_view",
    )

    if view == "Genres":
        _genre_trends(df)
    elif view == "Plateformes":
        _platform_trends(df)
    else:
        _publisher_trends(df)


# ------------------------------------------------------------------
# Genre trends
# ------------------------------------------------------------------


def _genre_trends(df: pd.DataFrame) -> None:
    st.subheader("Evolution des genres")

    all_genres = sorted(df["Genre"].unique())
    sel = st.multiselect(
        "Selectionner des genres",
        all_genres,
        default=all_genres[:5],
        key="trend_genres",
    )
    if not sel:
        st.warning("Selectionnez au moins un genre.")
        return

    df_g = df[df["Genre"].isin(sel)]

    # Number of releases per year
    releases = df_g.groupby(["Year", "Genre"]).size().reset_index(name="Nb_jeux")
    fig = px.line(
        releases,
        x="Year",
        y="Nb_jeux",
        color="Genre",
        title="Nombre de sorties par genre et par annee",
        labels={"Nb_jeux": "Nombre de jeux", "Year": "Annee"},
        markers=True,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Sales per year
    sales = df_g.groupby(["Year", "Genre"])["Global_Sales"].sum().reset_index()
    fig = px.area(
        sales,
        x="Year",
        y="Global_Sales",
        color="Genre",
        title="Ventes globales par genre et par annee",
        labels={"Global_Sales": "Ventes (millions)", "Year": "Annee"},
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Market share over time
    total_by_year = df.groupby("Year")["Global_Sales"].sum().rename("Total")
    genre_by_year = df_g.groupby(["Year", "Genre"])["Global_Sales"].sum().reset_index()
    genre_by_year = genre_by_year.merge(total_by_year, on="Year")
    genre_by_year["Part_marche"] = genre_by_year["Global_Sales"] / genre_by_year["Total"] * 100

    fig = px.bar(
        genre_by_year,
        x="Year",
        y="Part_marche",
        color="Genre",
        title="Part de marche par genre (%)",
        labels={"Part_marche": "Part de marche (%)", "Year": "Annee"},
    )
    fig.update_layout(barmode="stack", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Average scores over time
    scores = (
        df_g.groupby(["Year", "Genre"])
        .agg(
            meta_score=("meta_score", "mean"),
            user_review=("user_review", "mean"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            scores,
            x="Year",
            y="meta_score",
            color="Genre",
            title="Score Metacritic moyen par genre",
            labels={"meta_score": "Metacritic", "Year": "Annee"},
            markers=True,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            scores,
            x="Year",
            y="user_review",
            color="Genre",
            title="Score utilisateur moyen par genre",
            labels={"user_review": "Score utilisateur", "Year": "Annee"},
            markers=True,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Platform trends
# ------------------------------------------------------------------


def _platform_trends(df: pd.DataFrame) -> None:
    st.subheader("Evolution des plateformes")

    # Top 10 platforms by sales
    top_platforms = df.groupby("Platform")["Global_Sales"].sum().nlargest(10).index.tolist()
    all_platforms = sorted(df["Platform"].unique())
    sel = st.multiselect(
        "Selectionner des plateformes",
        all_platforms,
        default=top_platforms,
        key="trend_platforms",
    )
    if not sel:
        st.warning("Selectionnez au moins une plateforme.")
        return

    df_p = df[df["Platform"].isin(sel)]

    # Releases timeline
    releases = df_p.groupby(["Year", "Platform"]).size().reset_index(name="Nb_jeux")
    fig = px.line(
        releases,
        x="Year",
        y="Nb_jeux",
        color="Platform",
        title="Nombre de sorties par plateforme et par annee",
        labels={"Nb_jeux": "Nombre de jeux", "Year": "Annee"},
        markers=True,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Sales timeline
    sales = df_p.groupby(["Year", "Platform"])["Global_Sales"].sum().reset_index()
    fig = px.area(
        sales,
        x="Year",
        y="Global_Sales",
        color="Platform",
        title="Ventes globales par plateforme et par annee",
        labels={"Global_Sales": "Ventes (millions)", "Year": "Annee"},
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Platform lifecycle — active years heatmap
    lifecycle = (
        df_p.groupby(["Platform", "Year"])
        .agg(
            Nb_jeux=("Global_Sales", "count"),
            Ventes=("Global_Sales", "sum"),
        )
        .reset_index()
    )

    fig = px.density_heatmap(
        lifecycle,
        x="Year",
        y="Platform",
        z="Ventes",
        title="Cycle de vie des plateformes (ventes par annee)",
        labels={"Ventes": "Ventes (M)", "Year": "Annee"},
        color_continuous_scale="Viridis",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Publisher trends
# ------------------------------------------------------------------


def _publisher_trends(df: pd.DataFrame) -> None:
    st.subheader("Evolution des editeurs")

    # Top 10 publishers by total sales
    top_publishers = df.groupby("Publisher")["Global_Sales"].sum().nlargest(10).index.tolist()
    all_publishers = sorted(df["Publisher"].unique())
    sel = st.multiselect(
        "Selectionner des editeurs",
        all_publishers,
        default=top_publishers,
        key="trend_publishers",
    )
    if not sel:
        st.warning("Selectionnez au moins un editeur.")
        return

    df_pub = df[df["Publisher"].isin(sel)]

    # Sales over time
    sales = df_pub.groupby(["Year", "Publisher"])["Global_Sales"].sum().reset_index()
    fig = px.line(
        sales,
        x="Year",
        y="Global_Sales",
        color="Publisher",
        title="Ventes globales par editeur et par annee",
        labels={"Global_Sales": "Ventes (millions)", "Year": "Annee"},
        markers=True,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Genre distribution per publisher (sunburst)
    genre_dist = df_pub.groupby(["Publisher", "Genre"])["Global_Sales"].sum().reset_index()
    fig = px.sunburst(
        genre_dist,
        path=["Publisher", "Genre"],
        values="Global_Sales",
        title="Repartition des ventes par editeur et genre",
        color="Global_Sales",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        paper_bgcolor="#0D0D0D",
        font=dict(color="#E0E0E0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Number of releases per publisher per year
    releases = df_pub.groupby(["Year", "Publisher"]).size().reset_index(name="Nb_jeux")
    fig = px.bar(
        releases,
        x="Year",
        y="Nb_jeux",
        color="Publisher",
        title="Nombre de sorties par editeur et par annee",
        labels={"Nb_jeux": "Nombre de jeux", "Year": "Annee"},
        barmode="group",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Average Global Sales per title over time
    avg_sales = df_pub.groupby(["Year", "Publisher"])["Global_Sales"].mean().reset_index()
    fig = px.line(
        avg_sales,
        x="Year",
        y="Global_Sales",
        color="Publisher",
        title="Ventes moyennes par jeu par editeur et par annee",
        labels={"Global_Sales": "Ventes moyennes (millions)", "Year": "Annee"},
        markers=True,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
