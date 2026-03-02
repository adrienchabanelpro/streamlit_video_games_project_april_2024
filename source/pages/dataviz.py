"""DataViz page: interactive charts with global search & filter controls."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import CYAN, DATA_DIR, PINK, PLOTLY_LAYOUT, PURPLE, YELLOW
from data_validation import validate_dataframe


@st.cache_data
def _load_dataviz_data() -> pd.DataFrame:
    """Load and prepare the main dataset for visualization."""
    df = pd.read_csv(DATA_DIR / "Ventes_jeux_video_final.csv")
    # Advisory validation — warn but continue
    is_valid, errors = validate_dataframe(df)
    if not is_valid:
        st.warning(
            f"Validation des donnees : {len(errors)} probleme(s) detecte(s). "
            "Les visualisations peuvent etre affectees."
        )
    # Only drop rows missing critical viz columns (not steam_* which are mostly NaN)
    df = df.dropna(subset=["Year", "Genre", "Platform", "Publisher", "Global_Sales"])
    df["Year"] = df["Year"].astype(int)
    # Fill optional scores with median for charts that use them
    df["meta_score"] = df["meta_score"].fillna(df["meta_score"].median())
    df["user_review"] = df["user_review"].fillna(df["user_review"].median())
    return df


def dataviz_page() -> None:
    """Render the DataViz page with interactive filtered charts."""
    with st.spinner("Chargement des visualisations..."):
        df = _load_dataviz_data()

    st.title("Page de DataViz")

    # ------------------------------------------------------------------
    # Global filters
    # ------------------------------------------------------------------
    st.subheader("Filtres")
    with st.expander("Filtrer les donnees", expanded=True):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            all_genres = sorted(df["Genre"].unique())
            sel_genres = st.multiselect("Genres", all_genres, default=all_genres, key="dv_genre")
            all_platforms = sorted(df["Platform"].unique())
            sel_platforms = st.multiselect(
                "Plateformes", all_platforms, default=all_platforms, key="dv_platform"
            )
        with col_f2:
            all_publishers = sorted(df["Publisher"].unique())
            sel_publishers = st.multiselect(
                "Editeurs", all_publishers, default=all_publishers, key="dv_publisher"
            )
            year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
            sel_years = st.slider(
                "Periode", year_min, year_max, (year_min, year_max), key="dv_year"
            )

    # Apply filters
    mask = (
        df["Genre"].isin(sel_genres)
        & df["Platform"].isin(sel_platforms)
        & df["Publisher"].isin(sel_publishers)
        & df["Year"].between(sel_years[0], sel_years[1])
    )
    df_f = df[mask]

    if df_f.empty:
        st.warning("Aucun jeu ne correspond aux filtres selectionnes.")
        return

    st.caption(f"{len(df_f):,} jeux selectionnes sur {len(df):,}")

    st.markdown("---")

    _REGION_COLORS = [CYAN, PINK, YELLOW, PURPLE]

    # ------------------------------------------------------------------
    # 1. Global sales over time
    # ------------------------------------------------------------------
    st.header("Evolution des ventes globales par annee")
    sales_by_year = df_f.groupby("Year")["Global_Sales"].sum()
    mean_sales = sales_by_year.mean()
    median_sales = sales_by_year.median()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sales_by_year.index,
            y=sales_by_year,
            mode="lines+markers",
            name="Ventes annuelles",
            line=dict(color=CYAN),
        )
    )
    fig.add_hline(
        y=mean_sales,
        line=dict(color=PINK, dash="solid"),
        annotation_text=f"Moyenne: {mean_sales:.2f} M",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=median_sales,
        line=dict(color=YELLOW, dash="dash"),
        annotation_text=f"Mediane: {median_sales:.2f} M",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Evolution des ventes globales par annee",
        xaxis_title="Annee",
        yaxis_title="Ventes globales (millions)",
        legend_title="Legende",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 2. Regional sales over time
    # ------------------------------------------------------------------
    st.header("Evolution des ventes par region")
    df_sales_year = df_f.groupby("Year").agg(
        {
            "NA_Sales": "sum",
            "EU_Sales": "sum",
            "JP_Sales": "sum",
            "Other_Sales": "sum",
        }
    )
    fig = px.bar(
        df_sales_year,
        x=df_sales_year.index,
        y=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"],
        title="Evolution des ventes par region",
        color_discrete_sequence=_REGION_COLORS,
    )
    fig.update_layout(
        xaxis_title="Annee",
        yaxis_title="Ventes (millions)",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Total by region
    region_sales = df_f[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum().reset_index()
    region_sales.columns = ["Region", "Total_Sales"]
    fig = px.bar(
        region_sales,
        x="Region",
        y="Total_Sales",
        title="Ventes totales par region",
        labels={"Total_Sales": "Ventes totales (millions)", "Region": "Region"},
        color="Region",
        color_discrete_sequence=_REGION_COLORS,
    )
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Ventes totales (millions)",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3. Regional vs global scatter
    # ------------------------------------------------------------------
    st.header("Relation entre les ventes regionales et les ventes globales")
    region_cols = [
        ("NA_Sales", "Ventes NA"),
        ("EU_Sales", "Ventes EU"),
        ("JP_Sales", "Ventes JP"),
        ("Other_Sales", "Autres ventes"),
    ]
    for region_col, region_label in region_cols:
        fig = px.scatter(
            df_f,
            x=region_col,
            y="Global_Sales",
            color="Genre",
            title=f"{region_label} vs Ventes Globales",
            labels={
                region_col: f"{region_label} (millions)",
                "Global_Sales": "Ventes Globales (millions)",
            },
            opacity=0.6,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Top 10 publishers by region
    # ------------------------------------------------------------------
    st.header("Ventes par region pour les 10 principaux editeurs")
    ventes_par_editeur = (
        df_f.groupby("Publisher")
        .agg(
            {
                "Global_Sales": "sum",
                "NA_Sales": "sum",
                "EU_Sales": "sum",
                "JP_Sales": "sum",
                "Other_Sales": "sum",
            }
        )
        .reset_index()
    )
    top_editeurs = ventes_par_editeur.sort_values(by="Global_Sales", ascending=False).head(10)
    ventes_top = top_editeurs.melt(
        id_vars="Publisher",
        value_vars=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"],
        var_name="Region",
        value_name="Sales",
    )
    fig = px.bar(
        ventes_top,
        x="Publisher",
        y="Sales",
        color="Region",
        title="Ventes par region pour les 10 principaux editeurs",
        labels={"Sales": "Ventes (millions)", "Publisher": "Editeur"},
        text_auto=True,
        color_discrete_sequence=_REGION_COLORS,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. Sales distribution by genre
    # ------------------------------------------------------------------
    st.header("Distribution des ventes globales par genre de jeu")
    fig = px.box(
        df_f,
        x="Genre",
        y="Global_Sales",
        color="Genre",
        title="Distribution des ventes globales par genre de jeu",
        labels={"Global_Sales": "Ventes globales (millions)", "Genre": "Genre"},
        notched=True,
        points="all",
    )
    fig.update_layout(
        xaxis_tickangle=45,
        yaxis=dict(type="log", autorange=True),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 6. Top 5 genres over time
    # ------------------------------------------------------------------
    st.header("Evolution des ventes par genre (top 5)")
    top_5_genre = df_f.groupby("Genre")["Global_Sales"].sum().nlargest(5).index.tolist()
    if top_5_genre:
        df_top5 = df_f[df_f["Genre"].isin(top_5_genre)]
        pivot = df_top5.pivot_table(
            values="Global_Sales",
            index="Year",
            columns="Genre",
            aggfunc="sum",
        ).fillna(0)
        fig = px.bar(
            pivot,
            x=pivot.index,
            y=pivot.columns,
            title="Evolution des ventes en fonction des genres (top 5)",
        )
        fig.update_layout(
            xaxis_title="Annee",
            yaxis_title="Ventes globales (millions)",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 7. Meta Score vs User Review
    # ------------------------------------------------------------------
    st.header("Correlation entre Meta Score et User Review")
    fig = px.scatter(
        df_f,
        x="meta_score",
        y="user_review",
        title="Relation entre Meta Score et User Review",
        labels={"meta_score": "Meta Score", "user_review": "User Review"},
        trendline="ols",
        color="Genre",
        opacity=0.5,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 8. Sales by Meta Score / User Review
    # ------------------------------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        st.header("Ventes vs Meta Score")
        fig = px.histogram(
            df_f,
            x="meta_score",
            y="Global_Sales",
            title="Ventes globales vs Meta Score",
            labels={"meta_score": "Meta Score", "Global_Sales": "Ventes (millions)"},
            log_y=True,
            color_discrete_sequence=[CYAN],
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.header("Ventes vs User Review")
        fig = px.histogram(
            df_f,
            x="user_review",
            y="Global_Sales",
            title="Ventes globales vs User Review",
            labels={"user_review": "User Review", "Global_Sales": "Ventes (millions)"},
            log_y=True,
            color_discrete_sequence=[PINK],
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 9. Median scores by genre
    # ------------------------------------------------------------------
    st.header("Moyenne des avis par genre")
    df_score = (
        df_f.groupby("Genre")
        .agg(
            {
                "user_review": "mean",
                "meta_score": "mean",
            }
        )
        .reset_index()
    )

    col_c, col_d = st.columns(2)
    with col_c:
        df_sorted = df_score.sort_values("user_review")
        fig = px.bar(
            df_sorted,
            x="Genre",
            y="user_review",
            color="Genre",
            title="Moyenne des avis joueurs par genre",
            labels={"user_review": "Avis joueurs"},
        )
        fig.update_layout(yaxis_title="Avis joueurs", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        df_sorted = df_score.sort_values("meta_score")
        fig = px.bar(
            df_sorted,
            x="Genre",
            y="meta_score",
            color="Genre",
            title="Moyenne des avis presse par genre",
            labels={"meta_score": "Avis presse"},
        )
        fig.update_layout(yaxis_title="Avis presse", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
