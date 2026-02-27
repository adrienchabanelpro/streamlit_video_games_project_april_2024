"""What-if analysis: interactive exploration of how each feature impacts predictions."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prediction import (
    _NUMERICAL_FEATURES,
    _lookup_cumulative,
    load_feature_means,
    load_models,
    load_numerical_transformer,
    load_target_encoder,
)


def _predict_single(
    lgb_model, xgb_model, cb_model, scaler, encoder,
    train_stats: dict,
    genre: str, platform: str, publisher: str,
    year: int, meta_score: float, user_review: float,
) -> float:
    """Build features and run ensemble prediction for a single input."""
    input_data = {
        "Year": year,
        "meta_score": meta_score,
        "user_review": user_review,
    }

    # Feature engineering
    input_data["Global_Sales_mean_genre"] = train_stats["genre_means"].get(
        genre, train_stats["global_sales_mean"]
    )
    input_data["Global_Sales_mean_platform"] = train_stats["platform_means"].get(
        platform, train_stats["global_sales_mean"]
    )
    input_data["Year_Global_Sales_mean_genre"] = (
        input_data["Year"] * input_data["Global_Sales_mean_genre"]
    )
    input_data["Year_Global_Sales_mean_platform"] = (
        input_data["Year"] * input_data["Global_Sales_mean_platform"]
    )
    input_data["Cumulative_Sales_Genre"] = _lookup_cumulative(
        train_stats["cumsum_genre"], genre, year
    )
    input_data["Cumulative_Sales_Platform"] = _lookup_cumulative(
        train_stats["cumsum_platform"], platform, year
    )

    # Target encode publisher
    pub_df = pd.DataFrame({"Publisher": [publisher]})
    input_data["Publisher_encoded"] = encoder.transform(pub_df)["Publisher"].values[0]

    # Build DataFrame and scale
    df = pd.DataFrame(input_data, index=[0])
    df[_NUMERICAL_FEATURES] = scaler.transform(df[_NUMERICAL_FEATURES])

    # Ensemble prediction
    X = df[_NUMERICAL_FEATURES]
    pred_lgb = lgb_model.predict(X)
    pred_xgb = xgb_model.predict(X.values)
    pred_cb = cb_model.predict(X.values)
    return float((pred_lgb + pred_xgb + pred_cb) / 3)


def what_if_page():
    """What-if analysis page: sweep one variable and see impact on predictions."""
    st.title("Analyse What-If")
    st.write(
        "Explorez comment chaque variable influence les predictions de ventes. "
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
            "Score Metacritic (base)", min_value=0.0, max_value=100.0,
            value=train_stats["meta_score_mean"], format="%.0f",
        )
    with col4:
        base_user = st.number_input(
            "Score utilisateur (base)", min_value=0.0, max_value=100.0,
            value=train_stats["user_review_mean"], format="%.1f",
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

                pred = _predict_single(
                    lgb_model, xgb_model, cb_model, scaler, encoder,
                    train_stats, genre, platform, publisher,
                    yr, meta, user,
                )
                predictions.append(pred)

        # --- Plot ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_range,
            y=predictions,
            mode="lines+markers",
            name="Ventes predites",
            line=dict(color="#00FFCC", width=3),
            marker=dict(size=4),
        ))

        # Mark the base value
        if sweep_var == "meta_score":
            base_val = base_meta
        elif sweep_var == "user_review":
            base_val = base_user
        else:
            base_val = year

        base_pred = _predict_single(
            lgb_model, xgb_model, cb_model, scaler, encoder,
            train_stats, genre, platform, publisher,
            year, base_meta, base_user,
        )
        fig.add_trace(go.Scatter(
            x=[base_val],
            y=[base_pred],
            mode="markers",
            name="Valeur de base",
            marker=dict(color="#FF6EC7", size=14, symbol="star"),
        ))

        fig.update_layout(
            title=f"Impact de {x_label} sur les ventes predites",
            xaxis_title=x_label,
            yaxis_title="Ventes predites (millions)",
            template="plotly_dark",
            paper_bgcolor="#0D0D0D",
            plot_bgcolor="#1A1A2E",
            font=dict(color="#E0E0E0"),
        )

        st.plotly_chart(fig, use_container_width=True)

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
