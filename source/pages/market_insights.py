"""Market Insights page: trends, publisher analytics, genre evolution, recommendations."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from components import section_header
from config import ACCENT, DATA_DIR, PLOTLY_LAYOUT, SECONDARY, SUCCESS


@st.cache_data
def _load_data() -> pd.DataFrame:
    for name in ["Ventes_jeux_video_v3.csv", "Ventes_jeux_video_final.csv"]:
        path = DATA_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            df = df.dropna(subset=["Year", "Genre", "Platform", "Publisher", "Global_Sales"])
            df["Year"] = df["Year"].astype(int)
            return df
    return pd.DataFrame()


def market_insights_page() -> None:
    """Render the Market Insights page."""
    st.title("Market Trends")
    st.caption("Sales evolution, genres, platforms and publishers over time")

    df = _load_data()
    if df.empty:
        st.warning("Dataset not found.")
        return

    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "Temporal Evolution", "Genres", "Platforms", "Publishers"
    ])

    with tab1:
        _temporal_tab(df)

    with tab2:
        _genre_tab(df)

    with tab3:
        _platform_tab(df)

    with tab4:
        _publisher_tab(df)


def _temporal_tab(df: pd.DataFrame) -> None:
    """Sales and releases over time."""
    section_header("Global Sales Evolution")

    yearly = df.groupby("Year").agg(
        total_sales=("Global_Sales", "sum"),
        avg_sales=("Global_Sales", "mean"),
        game_count=("Global_Sales", "count"),
    ).reset_index()

    # Total sales by year
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly["total_sales"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=5),
        name="Total Sales (M$)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Total Global Sales by Year",
        xaxis_title="Year",
        yaxis_title="Sales (millions $)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Games released and avg sales
    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.bar(
            yearly, x="Year", y="game_count",
            color_discrete_sequence=[SECONDARY],
        )
        fig2.update_layout(**PLOTLY_LAYOUT, title="Games Released per Year", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = px.line(
            yearly, x="Year", y="avg_sales",
            color_discrete_sequence=[SUCCESS],
        )
        fig3.update_layout(**PLOTLY_LAYOUT, title="Average Sales per Game", height=350)
        st.plotly_chart(fig3, use_container_width=True)


def _genre_tab(df: pd.DataFrame) -> None:
    """Genre analysis."""
    section_header("Genre Analysis")

    # Genre evolution (stacked area)
    genre_year = df.groupby(["Year", "Genre"])["Global_Sales"].sum().reset_index()
    top_genres = df.groupby("Genre")["Global_Sales"].sum().nlargest(8).index.tolist()
    genre_year_top = genre_year[genre_year["Genre"].isin(top_genres)]

    fig = px.area(
        genre_year_top, x="Year", y="Global_Sales", color="Genre",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Sales Evolution by Genre (Top 8)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Genre comparison
    genre_stats = df.groupby("Genre").agg(
        total_sales=("Global_Sales", "sum"),
        avg_sales=("Global_Sales", "mean"),
        count=("Global_Sales", "count"),
    ).sort_values("total_sales", ascending=True).reset_index()

    fig2 = px.bar(
        genre_stats, y="Genre", x="total_sales",
        orientation="h",
        color="avg_sales",
        color_continuous_scale="Blues",
    )
    fig2.update_layout(**PLOTLY_LAYOUT, title="Total Sales by Genre", height=500)
    st.plotly_chart(fig2, use_container_width=True)


def _platform_tab(df: pd.DataFrame) -> None:
    """Platform analysis."""
    section_header("Platform Analysis")

    # Platform lifecycle heatmap
    platform_year = df.groupby(["Year", "Platform"])["Global_Sales"].sum().reset_index()
    top_platforms = df.groupby("Platform")["Global_Sales"].sum().nlargest(12).index.tolist()
    pf_filtered = platform_year[platform_year["Platform"].isin(top_platforms)]

    pivot = pf_filtered.pivot_table(index="Platform", columns="Year", values="Global_Sales", fill_value=0)

    fig = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Platform Lifecycle (Sales)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Platform market share over time
    total_by_year = df.groupby("Year")["Global_Sales"].sum()
    pf_share = pf_filtered.copy()
    pf_share["share"] = pf_share.apply(
        lambda r: r["Global_Sales"] / total_by_year.get(r["Year"], 1) * 100, axis=1
    )

    fig2 = px.area(
        pf_share, x="Year", y="share", color="Platform",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title="Platform Market Share (%)",
        yaxis_title="Share (%)",
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)


def _publisher_tab(df: pd.DataFrame) -> None:
    """Publisher analysis."""
    section_header("Publisher Analysis")

    pub_stats = df.groupby("Publisher").agg(
        total_sales=("Global_Sales", "sum"),
        avg_sales=("Global_Sales", "mean"),
        game_count=("Global_Sales", "count"),
        best_game_sales=("Global_Sales", "max"),
    ).sort_values("total_sales", ascending=False).reset_index()

    # Top publishers
    top_n = st.slider("Number of publishers to display", 5, 30, 15)
    top_pubs = pub_stats.head(top_n)

    fig = px.bar(
        top_pubs, x="Publisher", y="total_sales",
        color="avg_sales",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Top {top_n} Publishers by Total Sales",
        xaxis_tickangle=-45,
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Publisher table
    with st.expander("Detailed Publisher Table"):
        st.dataframe(
            top_pubs.rename(columns={
                "Publisher": "Publisher",
                "total_sales": "Total Sales (M$)",
                "avg_sales": "Average Sales",
                "game_count": "Game Count",
                "best_game_sales": "Best Game (M$)",
            }),
            use_container_width=True,
            hide_index=True,
        )
