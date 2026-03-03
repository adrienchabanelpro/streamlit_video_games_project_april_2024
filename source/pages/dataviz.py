"""DataViz page: interactive charts with global search & filter controls."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import ACCENT, DATA_DIR, PLOTLY_LAYOUT, SECONDARY
from data_validation import validate_dataframe


@st.cache_data
def _load_dataviz_data() -> pd.DataFrame:
    """Load and prepare the main dataset for visualization."""
    df = pd.read_csv(DATA_DIR / "Ventes_jeux_video_final.csv")
    # Advisory validation — warn but continue
    is_valid, errors = validate_dataframe(df)
    if not is_valid:
        st.warning(
            f"Data validation: {len(errors)} issue(s) detected. "
            "Visualizations may be affected."
        )
    # Only drop rows missing critical viz columns (not steam_* which are mostly NaN)
    df = df.dropna(subset=["Year", "Genre", "Platform", "Publisher", "Global_Sales"])
    df["Year"] = df["Year"].astype(int)
    # Fill optional scores with median for charts that use them
    # (fallback to 0 if median is NaN, e.g. user_review is 100% NaN in 64K dataset)
    df["meta_score"] = df["meta_score"].fillna(df["meta_score"].median()).fillna(0)
    df["user_review"] = df["user_review"].fillna(df["user_review"].median()).fillna(0)
    return df


def dataviz_page() -> None:
    """Render the DataViz page with interactive filtered charts."""
    with st.spinner("Loading visualizations..."):
        df = _load_dataviz_data()

    st.title("DataViz Page")

    # ------------------------------------------------------------------
    # Global filters
    # ------------------------------------------------------------------
    st.subheader("Filters")
    with st.expander("Filter data", expanded=True):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            all_genres = sorted(df["Genre"].unique())
            sel_genres = st.multiselect("Genres", all_genres, default=all_genres, key="dv_genre")
            all_platforms = sorted(df["Platform"].unique())
            sel_platforms = st.multiselect(
                "Platforms", all_platforms, default=all_platforms, key="dv_platform"
            )
        with col_f2:
            all_publishers = sorted(df["Publisher"].unique())
            sel_publishers = st.multiselect(
                "Publishers", all_publishers, default=all_publishers, key="dv_publisher"
            )
            year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
            sel_years = st.slider(
                "Period", year_min, year_max, (year_min, year_max), key="dv_year"
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
        st.warning("No games match the selected filters.")
        return

    st.caption(f"{len(df_f):,} games selected out of {len(df):,}")

    st.markdown("---")

    _REGION_COLORS = [ACCENT, SECONDARY, "#10B981", "#F59E0B"]

    # ------------------------------------------------------------------
    # 1. Global sales over time
    # ------------------------------------------------------------------
    st.header("Global Sales Trend by Year")
    sales_by_year = df_f.groupby("Year")["Global_Sales"].sum()
    mean_sales = sales_by_year.mean()
    median_sales = sales_by_year.median()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sales_by_year.index,
            y=sales_by_year,
            mode="lines+markers",
            name="Annual Sales",
            line=dict(color=ACCENT),
        )
    )
    fig.add_hline(
        y=mean_sales,
        line=dict(color=SECONDARY, dash="solid"),
        annotation_text=f"Mean: {mean_sales:.2f} M",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=median_sales,
        line=dict(color="#F59E0B", dash="dash"),
        annotation_text=f"Median: {median_sales:.2f} M",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Global Sales Trend by Year",
        xaxis_title="Year",
        yaxis_title="Global Sales (millions)",
        legend_title="Legend",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 2. Regional sales over time
    # ------------------------------------------------------------------
    st.header("Sales Trend by Region")
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
        title="Sales Trend by Region",
        color_discrete_sequence=_REGION_COLORS,
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Sales (millions)",
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
        title="Total Sales by Region",
        labels={"Total_Sales": "Total Sales (millions)", "Region": "Region"},
        color="Region",
        color_discrete_sequence=_REGION_COLORS,
    )
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Total Sales (millions)",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3. Regional vs global scatter
    # ------------------------------------------------------------------
    st.header("Regional Sales vs Global Sales")
    region_cols = [
        ("NA_Sales", "NA Sales"),
        ("EU_Sales", "EU Sales"),
        ("JP_Sales", "JP Sales"),
        ("Other_Sales", "Other Sales"),
    ]
    for region_col, region_label in region_cols:
        fig = px.scatter(
            df_f,
            x=region_col,
            y="Global_Sales",
            color="Genre",
            title=f"{region_label} vs Global Sales",
            labels={
                region_col: f"{region_label} (millions)",
                "Global_Sales": "Global Sales (millions)",
            },
            opacity=0.6,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Top 10 publishers by region
    # ------------------------------------------------------------------
    st.header("Sales by Region for the Top 10 Publishers")
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
        title="Sales by Region for the Top 10 Publishers",
        labels={"Sales": "Sales (millions)", "Publisher": "Publisher"},
        text_auto=True,
        color_discrete_sequence=_REGION_COLORS,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. Sales distribution by genre
    # ------------------------------------------------------------------
    st.header("Global Sales Distribution by Game Genre")
    fig = px.box(
        df_f,
        x="Genre",
        y="Global_Sales",
        color="Genre",
        title="Global Sales Distribution by Game Genre",
        labels={"Global_Sales": "Global Sales (millions)", "Genre": "Genre"},
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
    st.header("Sales Trend by Genre (Top 5)")
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
            title="Sales Trend by Genre (Top 5)",
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Global Sales (millions)",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 7. Meta Score vs User Review
    # ------------------------------------------------------------------
    # Check if user_review has meaningful data (not all zeros / all same value)
    _has_user_review = df_f["user_review"].nunique() > 1

    if _has_user_review:
        st.header("Correlation Between Meta Score and User Review")
        fig = px.scatter(
            df_f,
            x="meta_score",
            y="user_review",
            title="Relationship Between Meta Score and User Review",
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
    st.header("Sales vs Meta Score")
    fig = px.histogram(
        df_f,
        x="meta_score",
        y="Global_Sales",
        title="Global Sales vs Meta Score",
        labels={"meta_score": "Meta Score", "Global_Sales": "Sales (millions)"},
        log_y=True,
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    if _has_user_review:
        st.header("Sales vs User Review")
        fig = px.histogram(
            df_f,
            x="user_review",
            y="Global_Sales",
            title="Global Sales vs User Review",
            labels={"user_review": "User Review", "Global_Sales": "Sales (millions)"},
            log_y=True,
            color_discrete_sequence=[SECONDARY],
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 9. Average scores by genre
    # ------------------------------------------------------------------
    st.header("Average Critics Score by Genre")
    df_score = (
        df_f.groupby("Genre")
        .agg(meta_score=("meta_score", "mean"))
        .reset_index()
    )

    df_sorted = df_score.sort_values("meta_score")
    fig = px.bar(
        df_sorted,
        x="Genre",
        y="meta_score",
        color="Genre",
        title="Average Critics Score by Genre",
        labels={"meta_score": "Critics Score"},
    )
    fig.update_layout(yaxis_title="Critics Score", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    if _has_user_review:
        df_score_user = (
            df_f.groupby("Genre")
            .agg(user_review=("user_review", "mean"))
            .reset_index()
        )
        df_sorted = df_score_user.sort_values("user_review")
        fig = px.bar(
            df_sorted,
            x="Genre",
            y="user_review",
            color="Genre",
            title="Average Player Review Score by Genre",
            labels={"user_review": "Player Review"},
        )
        fig.update_layout(yaxis_title="Player Review", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
