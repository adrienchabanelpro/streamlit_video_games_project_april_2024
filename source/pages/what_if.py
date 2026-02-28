"""What-if analysis: interactive exploration of how each feature impacts predictions."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config import CYAN, PINK, PLOTLY_LAYOUT
from prediction import (
    load_feature_means,
    load_models,
    load_numerical_transformer,
    load_target_encoder,
    predict_single,
)
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header


def what_if_page() -> None:
    """What-if analysis page: sweep one variable and see impact on predictions."""
    colored_header(
        label="Analyse What-If",
        description="Explorez comment chaque variable influence les predictions de ventes",
        color_name="light-blue-70",
    )
    add_vertical_space(1)
    st.write(
        "Selectionnez une configuration de base, puis choisissez une variable "
        "a faire varier pour observer son impact en temps reel."
    )

    try:
        lgb_model, xgb_model, cb_model = load_models()
        train_stats = load_feature_means()
        scaler = load_numerical_transformer()
        encoder = load_target_encoder()
    except Exception as e:
        st.error(f"Erreur lors du chargement des modeles : {e}")
        return

    st.markdown("---")

    # --- Base configuration ---
    st.subheader("Configuration de base")
    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("Genre", train_stats["genres"])
        platform = st.selectbox("Plateforme", train_stats["platforms"])
    with col2:
        publisher = st.selectbox("Editeur", train_stats["publishers"])
        year = st.number_input("Annee", min_value=1980, max_value=2030, value=2015)

    col3, col4 = st.columns(2)
    with col3:
        base_meta = st.number_input(
            "Score Metacritic (base)",
            min_value=0.0,
            max_value=100.0,
            value=train_stats["meta_score_mean"],
            format="%.0f",
        )
    with col4:
        base_user = st.number_input(
            "Score utilisateur (base)",
            min_value=0.0,
            max_value=100.0,
            value=train_stats["user_review_mean"],
            format="%.1f",
        )

    st.markdown("---")

    # --- Variable to sweep ---
    st.subheader("Variable a analyser")
    sweep_var = st.selectbox(
        "Quelle variable voulez-vous faire varier ?",
        ["meta_score", "user_review", "Year"],
    )

    if sweep_var == "meta_score":
        sweep_range = np.linspace(0, 100, 50)
        x_label = "Score Metacritic"
    elif sweep_var == "user_review":
        sweep_range = np.linspace(0, 100, 50)
        x_label = "Score utilisateur"
    else:  # Year
        sweep_range = np.arange(1990, 2026)
        x_label = "Annee"

    # --- Compute predictions across sweep ---
    if st.button("Lancer l'analyse"):
        with st.spinner("Calcul des predictions..."):
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
                name="Ventes predites",
                line=dict(color=CYAN, width=3),
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
                name="Valeur de base",
                marker=dict(color=PINK, size=14, symbol="star"),
            )
        )

        fig.update_layout(
            title=f"Impact de {x_label} sur les ventes predites",
            xaxis_title=x_label,
            yaxis_title="Ventes predites (millions)",
            **PLOTLY_LAYOUT,
        )

        st.plotly_chart(fig, use_container_width=True)
        add_vertical_space(1)

        # --- Summary stats ---
        col_min, col_max, col_range = st.columns(3)
        with col_min:
            st.metric("Prediction min", f"{min(predictions):.4f} M")
        with col_max:
            st.metric("Prediction max", f"{max(predictions):.4f} M")
        with col_range:
            st.metric("Amplitude", f"{max(predictions) - min(predictions):.4f} M")

        st.caption(
            f"Configuration : {genre} / {platform} / {publisher} / "
            f"Annee={year} / Meta={base_meta:.0f} / User={base_user:.1f}"
        )
