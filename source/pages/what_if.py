"""What-if analysis: interactive exploration of how each feature impacts predictions."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config import ACCENT, PLOTLY_LAYOUT, SECONDARY
from prediction import (
    load_feature_means,
    load_models,
    load_numerical_transformer,
    load_target_encoder,
    predict_single,
)


def what_if_page() -> None:
    """What-if analysis page: sweep one variable and see impact on predictions."""
    st.title("What-If Analysis")
    st.caption("Explore how each variable influences sales predictions")
    st.write(
        "Select a base configuration, then choose a variable "
        "to sweep and observe its impact in real time."
    )

    try:
        lgb_model, xgb_model, cb_model = load_models()
        train_stats = load_feature_means()
        scaler = load_numerical_transformer()
        encoder = load_target_encoder()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    st.markdown("---")

    # --- Base configuration ---
    st.subheader("Base Configuration")
    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("Genre", train_stats["genres"])
        platform = st.selectbox("Platform", train_stats["platforms"])
    with col2:
        publisher = st.selectbox("Publisher", train_stats["publishers"])
        year = st.number_input("Year", min_value=1980, max_value=2030, value=2015)

    col3, col4 = st.columns(2)
    with col3:
        base_meta = st.number_input(
            "Metacritic Score (base)",
            min_value=0.0,
            max_value=10.0,
            value=train_stats["meta_score_mean"],
            format="%.1f",
        )
    with col4:
        base_user = st.number_input(
            "User Score (base)",
            min_value=0.0,
            max_value=10.0,
            value=train_stats["user_review_mean"],
            format="%.1f",
        )

    st.markdown("---")

    # --- Variable to sweep ---
    st.subheader("Variable to Analyze")
    sweep_var = st.selectbox(
        "Which variable do you want to sweep?",
        ["meta_score", "user_review", "Year"],
    )

    if sweep_var == "meta_score":
        sweep_range = np.linspace(0, 10, 50)
        x_label = "Metacritic Score"
    elif sweep_var == "user_review":
        sweep_range = np.linspace(0, 10, 50)
        x_label = "User Score"
    else:  # Year
        sweep_range = np.arange(1990, 2026)
        x_label = "Year"

    # --- Compute predictions across sweep ---
    if st.button("Run Analysis"):
        with st.spinner("Computing predictions..."):
            predictions = []
            for val in sweep_range:
                meta = float(val) if sweep_var == "meta_score" else base_meta
                user = float(val) if sweep_var == "user_review" else base_user
                yr = int(val) if sweep_var == "Year" else year

                pred = predict_single(
                    lgb_model,
                    xgb_model,
                    cb_model,
                    scaler,
                    encoder,
                    train_stats,
                    genre,
                    platform,
                    publisher,
                    yr,
                    meta,
                    user,
                )
                predictions.append(pred)

        # --- Plot ---
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sweep_range,
                y=predictions,
                mode="lines+markers",
                name="Predicted Sales",
                line=dict(color=ACCENT, width=3),
                marker=dict(size=4),
            )
        )

        # Mark the base value
        if sweep_var == "meta_score":
            base_val = base_meta
        elif sweep_var == "user_review":
            base_val = base_user
        else:
            base_val = year

        base_pred = predict_single(
            lgb_model,
            xgb_model,
            cb_model,
            scaler,
            encoder,
            train_stats,
            genre,
            platform,
            publisher,
            year,
            base_meta,
            base_user,
        )
        fig.add_trace(
            go.Scatter(
                x=[base_val],
                y=[base_pred],
                mode="markers",
                name="Base Value",
                marker=dict(color=SECONDARY, size=14, symbol="star"),
            )
        )

        fig.update_layout(
            title=f"Impact of {x_label} on Predicted Sales",
            xaxis_title=x_label,
            yaxis_title="Predicted Sales (millions)",
            **PLOTLY_LAYOUT,
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Summary stats ---
        col_min, col_max, col_range = st.columns(3)
        with col_min:
            st.metric("Min Prediction", f"{min(predictions):.4f} M")
        with col_max:
            st.metric("Max Prediction", f"{max(predictions):.4f} M")
        with col_range:
            st.metric("Range", f"{max(predictions) - min(predictions):.4f} M")

        st.caption(
            f"Configuration: {genre} / {platform} / {publisher} / "
            f"Year={year} / Meta={base_meta:.0f} / User={base_user:.1f}"
        )
