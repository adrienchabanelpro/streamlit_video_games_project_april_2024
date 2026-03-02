"""Game comparison tool: side-by-side prediction of two game configurations."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import BG, BG_SECONDARY, CYAN, PINK, TEXT_COLOR, YELLOW
from prediction import (
    load_feature_means,
    load_models,
    load_numerical_transformer,
    load_target_encoder,
    predict_single,
)
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

# ---------------------------------------------------------------------------
# Input widget helpers
# ---------------------------------------------------------------------------


def _game_inputs(
    column_key: str,
    train_stats: dict,
    label: str,
) -> dict[str, str | int | float]:
    """Render input widgets for one game configuration and return values.

    Args:
        column_key: Unique key prefix so Streamlit widgets do not collide.
        train_stats: Pre-computed training statistics (genres, platforms, etc.).
        label: Display label for the configuration (e.g. "Jeu A").

    Returns:
        Dictionary with keys: genre, platform, publisher, year, meta_score, user_review.
    """
    st.markdown(
        f"<h3 style='text-align:center; color:{CYAN};'>{label}</h3>",
        unsafe_allow_html=True,
    )

    genre = st.selectbox(
        "Genre",
        train_stats["genres"],
        key=f"{column_key}_genre",
    )
    platform = st.selectbox(
        "Plateforme",
        train_stats["platforms"],
        key=f"{column_key}_platform",
    )
    publisher = st.selectbox(
        "Editeur",
        train_stats["publishers"],
        key=f"{column_key}_publisher",
    )
    years = list(range(1980, 2031))
    year = st.selectbox(
        "Annee",
        years,
        index=years.index(2015),
        key=f"{column_key}_year",
    )
    meta_score = st.number_input(
        "Score Metacritic",
        min_value=0.0,
        max_value=10.0,
        value=train_stats["meta_score_mean"],
        format="%.1f",
        key=f"{column_key}_meta",
    )
    user_review = st.number_input(
        "Score utilisateur",
        min_value=0.0,
        max_value=10.0,
        value=train_stats["user_review_mean"],
        format="%.1f",
        key=f"{column_key}_user",
    )

    return {
        "genre": genre,
        "platform": platform,
        "publisher": publisher,
        "year": int(year),
        "meta_score": float(meta_score),
        "user_review": float(user_review),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _build_bar_chart(
    label_a: str,
    label_b: str,
    pred_a: float,
    pred_b: float,
) -> go.Figure:
    """Create a horizontal bar chart comparing predicted sales for both games."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=[label_a],
            x=[pred_a],
            orientation="h",
            name=label_a,
            marker_color=CYAN,
            text=[f"{pred_a:.4f} M"],
            textposition="auto",
            textfont=dict(color=BG, size=13),
        )
    )
    fig.add_trace(
        go.Bar(
            y=[label_b],
            x=[pred_b],
            orientation="h",
            name=label_b,
            marker_color=PINK,
            text=[f"{pred_b:.4f} M"],
            textposition="auto",
            textfont=dict(color=BG, size=13),
        )
    )

    fig.update_layout(
        title="Ventes predites (millions d'unites)",
        xaxis_title="Ventes (M)",
        yaxis_title="",
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG_SECONDARY,
        font=dict(color=TEXT_COLOR),
        barmode="group",
        showlegend=False,
        height=280,
        margin=dict(l=20, r=20, t=50, b=30),
    )

    return fig


def _build_radar_chart(
    label_a: str,
    label_b: str,
    cfg_a: dict[str, str | int | float],
    cfg_b: dict[str, str | int | float],
    pred_a: float,
    pred_b: float,
) -> go.Figure:
    """Create a radar chart comparing feature values of both game configurations.

    Axes: Annee (normalized 1980-2030), Metacritic (0-10 → 0-100), Score utilisateur
    (0-10 → 0-100), Ventes predites (normalized to max of both).
    """
    # Normalize Year to 0-100 scale for visual consistency
    year_min, year_max = 1980, 2030
    year_a_norm = (cfg_a["year"] - year_min) / (year_max - year_min) * 100
    year_b_norm = (cfg_b["year"] - year_min) / (year_max - year_min) * 100

    # Normalize predicted sales to 0-100 (relative to max of the two)
    max_pred = max(pred_a, pred_b, 0.0001)  # avoid division by zero
    pred_a_norm = pred_a / max_pred * 100
    pred_b_norm = pred_b / max_pred * 100

    # Scale scores from 0-10 to 0-100 for radar visual consistency
    meta_a_norm = cfg_a["meta_score"] * 10
    meta_b_norm = cfg_b["meta_score"] * 10
    user_a_norm = cfg_a["user_review"] * 10
    user_b_norm = cfg_b["user_review"] * 10

    categories = [
        "Annee",
        "Score Metacritic",
        "Score utilisateur",
        "Ventes predites",
    ]

    values_a = [year_a_norm, meta_a_norm, user_a_norm, pred_a_norm]
    values_b = [year_b_norm, meta_b_norm, user_b_norm, pred_b_norm]

    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_a_closed = values_a + [values_a[0]]
    values_b_closed = values_b + [values_b[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_a_closed,
            theta=categories_closed,
            fill="toself",
            name=label_a,
            line_color=CYAN,
            fillcolor="rgba(0, 255, 204, 0.15)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=values_b_closed,
            theta=categories_closed,
            fill="toself",
            name=label_b,
            line_color=PINK,
            fillcolor="rgba(255, 110, 199, 0.15)",
        )
    )

    fig.update_layout(
        title="Comparaison des profils",
        template="plotly_dark",
        paper_bgcolor=BG,
        font=dict(color=TEXT_COLOR),
        polar=dict(
            bgcolor=BG_SECONDARY,
            radialaxis=dict(
                visible=True,
                range=[0, 105],
                gridcolor="rgba(255, 255, 255, 0.1)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.1)",
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(color=TEXT_COLOR),
        ),
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


# ---------------------------------------------------------------------------
# Summary card helper
# ---------------------------------------------------------------------------


def _render_summary_card(
    label: str,
    cfg: dict[str, str | int | float],
    prediction: float,
    accent_color: str,
) -> None:
    """Display a styled summary card for a game configuration."""
    st.markdown(
        f"""
        <div style="
            background-color: {BG_SECONDARY};
            border: 1px solid {accent_color};
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 0 12px {accent_color}40;
        ">
            <h4 style="color: {accent_color}; text-align: center; margin-bottom: 12px;">
                {label}
            </h4>
            <p style="color: {TEXT_COLOR}; margin: 4px 0;"><b>Genre :</b> {cfg["genre"]}</p>
            <p style="color: {TEXT_COLOR}; margin: 4px 0;"><b>Plateforme :</b> {cfg["platform"]}</p>
            <p style="color: {TEXT_COLOR}; margin: 4px 0;"><b>Editeur :</b> {cfg["publisher"]}</p>
            <p style="color: {TEXT_COLOR}; margin: 4px 0;"><b>Annee :</b> {cfg["year"]}</p>
            <p style="color: {TEXT_COLOR}; margin: 4px 0;"><b>Metacritic :</b> {cfg["meta_score"]:.0f}</p>
            <p style="color: {TEXT_COLOR}; margin: 4px 0;"><b>Score utilisateur :</b> {cfg["user_review"]:.1f}</p>
            <hr style="border-color: {accent_color}40;">
            <p style="
                text-align: center;
                font-size: 1.4em;
                font-weight: bold;
                color: {accent_color};
                text-shadow: 0 0 8px {accent_color}80;
                margin: 8px 0 0 0;
            ">
                {prediction:.4f} M
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main page function
# ---------------------------------------------------------------------------


def comparison_page() -> None:
    """Game comparison page: side-by-side prediction of two game configurations."""
    colored_header(
        label="Comparaison de jeux",
        description="Comparez les ventes predites de deux configurations de jeux cote a cote",
        color_name="light-blue-70",
    )
    add_vertical_space(1)
    st.write(
        "Configurez deux jeux hypothetiques et comparez leurs ventes predites "
        "cote a cote. Ajustez les parametres de chaque jeu pour explorer "
        "comment les choix de genre, plateforme, editeur et scores influencent "
        "les predictions."
    )

    # Load models and statistics
    try:
        lgb_model, xgb_model, cb_model = load_models()
        train_stats = load_feature_means()
        scaler = load_numerical_transformer()
        encoder = load_target_encoder()
    except Exception as e:
        st.error(f"Erreur lors du chargement des modeles : {e}")
        return

    st.markdown("---")

    # --- Side-by-side game configuration inputs ---
    col_a, col_spacer, col_b = st.columns([5, 1, 5])

    with col_a:
        cfg_a = _game_inputs("game_a", train_stats, "Jeu A")

    with col_spacer:
        # Visual separator between the two columns
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
                min-height: 400px;
            ">
                <div style="
                    width: 2px;
                    height: 380px;
                    background: linear-gradient({CYAN}, {PINK});
                    border-radius: 2px;
                "></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        cfg_b = _game_inputs("game_b", train_stats, "Jeu B")

    st.markdown("---")

    # --- Compare button ---
    _, btn_col, _ = st.columns([4, 3, 4])
    with btn_col:
        compare_clicked = st.button("Comparer", use_container_width=True)

    if compare_clicked:
        with st.spinner("Calcul des predictions..."):
            pred_a = predict_single(
                lgb_model,
                xgb_model,
                cb_model,
                scaler,
                encoder,
                train_stats,
                cfg_a["genre"],
                cfg_a["platform"],
                cfg_a["publisher"],
                cfg_a["year"],
                cfg_a["meta_score"],
                cfg_a["user_review"],
            )
            pred_b = predict_single(
                lgb_model,
                xgb_model,
                cb_model,
                scaler,
                encoder,
                train_stats,
                cfg_b["genre"],
                cfg_b["platform"],
                cfg_b["publisher"],
                cfg_b["year"],
                cfg_b["meta_score"],
                cfg_b["user_review"],
            )

        st.markdown("---")

        # --- Summary cards side by side ---
        st.subheader("Resultats")
        res_a, res_b = st.columns(2)
        with res_a:
            _render_summary_card("Jeu A", cfg_a, pred_a, CYAN)
        with res_b:
            _render_summary_card("Jeu B", cfg_b, pred_b, PINK)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Difference metric ---
        diff = pred_a - pred_b
        diff_pct = (diff / max(abs(pred_b), 0.0001)) * 100
        if diff > 0:
            winner_text = "Jeu A vend plus"
            diff_color = CYAN
        elif diff < 0:
            winner_text = "Jeu B vend plus"
            diff_color = PINK
        else:
            winner_text = "Egalite"
            diff_color = YELLOW

        st.markdown(
            f"""
            <div style="
                text-align: center;
                padding: 14px;
                background-color: {BG_SECONDARY};
                border: 1px solid {diff_color};
                border-radius: 10px;
                box-shadow: 0 0 10px {diff_color}30;
                margin: 10px auto;
                max-width: 500px;
            ">
                <p style="color: {diff_color}; font-size: 1.1em; font-weight: bold; margin: 0;">
                    {winner_text}
                </p>
                <p style="color: {TEXT_COLOR}; margin: 4px 0 0 0;">
                    Ecart : {abs(diff):.4f} M ({abs(diff_pct):.1f}%)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Charts side by side ---
        chart_left, chart_right = st.columns(2)

        with chart_left:
            bar_fig = _build_bar_chart("Jeu A", "Jeu B", pred_a, pred_b)
            st.plotly_chart(bar_fig, use_container_width=True)

        with chart_right:
            radar_fig = _build_radar_chart(
                "Jeu A",
                "Jeu B",
                cfg_a,
                cfg_b,
                pred_a,
                pred_b,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        # --- Export comparison as CSV ---
        st.markdown("---")
        export_df = pd.DataFrame(
            {
                "Configuration": ["Jeu A", "Jeu B"],
                "Genre": [cfg_a["genre"], cfg_b["genre"]],
                "Plateforme": [cfg_a["platform"], cfg_b["platform"]],
                "Editeur": [cfg_a["publisher"], cfg_b["publisher"]],
                "Annee": [cfg_a["year"], cfg_b["year"]],
                "Metacritic": [cfg_a["meta_score"], cfg_b["meta_score"]],
                "Score_utilisateur": [cfg_a["user_review"], cfg_b["user_review"]],
                "Ventes_predites_M": [round(pred_a, 4), round(pred_b, 4)],
            }
        )
        st.download_button(
            "Telecharger la comparaison (CSV)",
            export_df.to_csv(index=False),
            file_name="comparaison_jeux.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Les predictions sont basees sur un ensemble de 3 modeles "
        "(LightGBM + XGBoost + CatBoost). Les resultats sont indicatifs.</p>",
        unsafe_allow_html=True,
    )
