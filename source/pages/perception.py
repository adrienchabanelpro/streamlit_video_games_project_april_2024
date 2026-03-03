"""User perception analysis page with DistilBERT sentiment, word clouds, and aspect analysis."""

import plotly.graph_objects as go
import streamlit as st
from config import ACCENT, BG, PLOTLY_LAYOUT, SECONDARY, TEXT_COLOR
from sentiment_analysis import analyze_aspects, has_transformers, predict_user_reviews


def perception_page() -> None:
    """Render the user perception / sentiment analysis page."""
    if not has_transformers():
        st.title("Perception")
        st.info(
            "Sentiment analysis requires the `transformers` library "
            "(and PyTorch), which is not available in this cloud environment. "
            "To use this feature, run the application locally with:\n\n"
            "```\npip install -r requirements-dev.txt\nstreamlit run source/main.py\n```"
        )
        return

    st.title("Perception")
    st.write(
        "Analyze user sentiment with our NLP models. "
        "Upload a CSV file and let our AI predict your customers' perception."
    )

    # --- Mode selector ---
    mode = st.radio(
        "Analysis Mode",
        ["Binary (Positive/Negative)", "5 Stars (1-5)", "Aspect Analysis"],
        horizontal=True,
        key="sentiment_mode",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.caption(
        "The CSV file must have a column named 'user_review'. "
        "The 5-star model supports French, English, German, "
        "Spanish, Italian, and Dutch."
    )

    if uploaded_file is None:
        return

    if not st.button("Run Analysis"):
        return

    if mode == "Aspect Analysis":
        _run_aspect_analysis(uploaded_file)
    elif mode == "5 Stars (1-5)":
        _run_star_analysis(uploaded_file)
    else:
        _run_binary_analysis(uploaded_file)

    # Disclaimer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #94A3B8;'>"
        "Models: DistilBERT (binary) and BERT multilingual (5 stars). "
        "These models are in beta and may make errors. "
        "Consider verifying important information.</p>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Binary sentiment analysis
# ------------------------------------------------------------------


def _run_binary_analysis(uploaded_file: object) -> None:
    """Run binary sentiment and display results with word clouds."""
    with st.spinner("Binary analysis with DistilBERT..."):
        data, positive_pct, negative_pct = predict_user_reviews(uploaded_file, granularity="binary")

    if data is None or positive_pct is None:
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Positive Reviews", f"{positive_pct:.1f}%")
    with col2:
        st.metric("Negative Reviews", f"{negative_pct:.1f}%")
    with col3:
        avg_conf = data["confidence"].mean() * 100
        st.metric("Average Confidence", f"{avg_conf:.1f}%")

    # Gauge chart
    _display_gauge(positive_pct)

    # Word clouds
    _display_word_clouds(data)

    # Per-review details
    st.subheader("Prediction Details")
    display_df = data[["user_review", "sentiment", "confidence"]].copy()
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
    display_df.columns = ["Review", "Sentiment", "Confidence"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# 5-star sentiment analysis
# ------------------------------------------------------------------


def _run_star_analysis(uploaded_file: object) -> None:
    """Run 5-star sentiment analysis and display results."""
    with st.spinner("5-star analysis with BERT multilingual..."):
        data, avg_stars, _ = predict_user_reviews(uploaded_file, granularity="5-star")

    if data is None or avg_stars is None:
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Rating", f"{avg_stars:.1f} / 5")
    with col2:
        avg_conf = data["confidence"].mean() * 100
        st.metric("Average Confidence", f"{avg_conf:.1f}%")
    with col3:
        st.metric("Number of Reviews", str(len(data)))

    # Star distribution chart
    star_counts = data["stars"].value_counts().sort_index()
    colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#10B981"]

    fig = go.Figure()
    for star_val in range(1, 6):
        count = star_counts.get(star_val, 0)
        fig.add_trace(
            go.Bar(
                x=[f"{star_val} star{'s' if star_val > 1 else ''}"],
                y=[count],
                name=f"{star_val} star{'s' if star_val > 1 else ''}",
                marker_color=colors[star_val - 1],
                text=[count],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Rating Distribution",
        xaxis_title="Rating",
        yaxis_title="Number of Reviews",
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-review details
    st.subheader("Prediction Details")
    display_df = data[["user_review", "stars", "sentiment", "confidence"]].copy()
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
    display_df.columns = ["Review", "Stars", "Sentiment", "Confidence"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# Aspect-based sentiment analysis
# ------------------------------------------------------------------


def _run_aspect_analysis(uploaded_file: object) -> None:
    """Run aspect-based sentiment analysis and display results."""
    import pandas as pd

    try:
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return

    if "user_review" not in data.columns:
        st.warning("The CSV file must contain a 'user_review' column.")
        return

    data = data.dropna(subset=["user_review"]).reset_index(drop=True)
    reviews = data["user_review"].astype(str).tolist()

    if not reviews:
        st.warning("No valid reviews found.")
        return

    with st.spinner("Aspect analysis with DistilBERT..."):
        aspect_results = analyze_aspects(reviews)

    # Build chart data
    aspects = []
    pos_counts = []
    neg_counts = []
    totals = []
    for aspect, counts in aspect_results.items():
        aspects.append(aspect.capitalize())
        pos_counts.append(counts["positive"])
        neg_counts.append(counts["negative"])
        totals.append(counts["total"])

    if sum(totals) == 0:
        st.info("No game aspects detected in the reviews.")
        return

    # Grouped bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Positive",
            x=aspects,
            y=pos_counts,
            marker_color=ACCENT,
            text=pos_counts,
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Negative",
            x=aspects,
            y=neg_counts,
            marker_color=SECONDARY,
            text=neg_counts,
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Sentiment by Game Aspect",
        xaxis_title="Aspect",
        yaxis_title="Number of Reviews",
        barmode="group",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Aspect Summary")
    summary_rows = []
    for aspect, counts in aspect_results.items():
        total = counts["total"]
        if total > 0:
            pos_pct = counts["positive"] / total * 100
            summary_rows.append(
                {
                    "Aspect": aspect.capitalize(),
                    "Mentions": total,
                    "Positive (%)": f"{pos_pct:.0f}%",
                    "Negative (%)": f"{100 - pos_pct:.0f}%",
                }
            )
    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------------
# Word clouds
# ------------------------------------------------------------------


def _display_word_clouds(data: object) -> None:
    """Generate and display word clouds for positive and negative reviews."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        st.info("Install 'wordcloud' (pip install wordcloud) to display word clouds.")
        return

    from PIL import Image

    pos_reviews = data[data["predictions"] == 1]["user_review"].str.cat(sep=" ")
    neg_reviews = data[data["predictions"] == 0]["user_review"].str.cat(sep=" ")

    if not pos_reviews.strip() and not neg_reviews.strip():
        return

    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)

    with col1:
        if pos_reviews.strip():
            wc = WordCloud(
                width=600,
                height=400,
                background_color=BG,
                colormap="cool",
                max_words=100,
            ).generate(pos_reviews)
            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source=Image.fromarray(wc.to_array()),
                    x=0,
                    y=1,
                    xref="paper",
                    yref="paper",
                    sizex=1,
                    sizey=1,
                    xanchor="left",
                    yanchor="top",
                    layer="above",
                )
            )
            fig.update_layout(
                title="Positive Reviews",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                paper_bgcolor=BG,
                margin=dict(t=40, b=0, l=0, r=0),
                font=dict(color=TEXT_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positive reviews to generate a word cloud.")

    with col2:
        if neg_reviews.strip():
            wc = WordCloud(
                width=600,
                height=400,
                background_color=BG,
                colormap="spring",
                max_words=100,
            ).generate(neg_reviews)
            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source=Image.fromarray(wc.to_array()),
                    x=0,
                    y=1,
                    xref="paper",
                    yref="paper",
                    sizex=1,
                    sizey=1,
                    xanchor="left",
                    yanchor="top",
                    layer="above",
                )
            )
            fig.update_layout(
                title="Negative Reviews",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                paper_bgcolor=BG,
                margin=dict(t=40, b=0, l=0, r=0),
                font=dict(color=TEXT_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No negative reviews to generate a word cloud.")


# ------------------------------------------------------------------
# Gauge chart (no image background)
# ------------------------------------------------------------------


def _display_gauge(positive_percentage: float) -> None:
    """Render the Plotly gauge for positive sentiment percentage."""

    def _get_color(value: float, alpha: float = 0.5) -> str:
        """Return an RGBA color string for the given gauge *value*."""
        if value <= 25:
            red, green, blue = 255, int((value / 25) * 165), 0
        elif value <= 50:
            red = 255
            green = 165 + int(((value - 25) / 25) * (255 - 165))
            blue = 0
        elif value <= 75:
            red = 255 - int(((value - 50) / 25) * 200)
            green = 240 - int((value / 100) * (255 - 128))
            blue = 0
        else:
            red, green = 0, 255 - int(((value - 55) / 25) * (255 - 128))
            blue = 0
        return f"rgba({red},{green},{blue},{alpha})"

    steps = [{"range": [i, i + 1], "color": _get_color(i, alpha=0.7)} for i in range(101)]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=positive_percentage,
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickfont": {"size": 18, "family": "Inter, sans-serif", "color": "#F1F5F9"},
                },
                "bar": {"color": ACCENT},
                "steps": steps,
                "threshold": {
                    "line": {"color": ACCENT, "width": 5},
                    "thickness": 0.75,
                    "value": positive_percentage,
                },
            },
        )
    )

    fig.update_layout(
        title={
            "text": "Positive Review Percentage",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": ACCENT},
        },
        paper_bgcolor=BG,
        font=dict(family="Inter, sans-serif", size=18, color=TEXT_COLOR),
        margin=dict(t=50, b=0, l=0, r=0),
    )

    st.plotly_chart(fig)
