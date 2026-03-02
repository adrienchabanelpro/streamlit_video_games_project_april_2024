"""Prediction page UI: single and batch predictions using the 3-model ensemble."""

import numpy as np
import pandas as pd
import streamlit as st
from config import IMAGES_DIR
from ml.predict import NUMERICAL_FEATURES, get_features, is_log_transformed, prepare_for_prediction
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header


def _get_input(train_stats: dict) -> tuple[str, str, str, dict]:
    """Render sidebar inputs and return user selections."""
    st.sidebar.header("Selection des entrees")

    input_data: dict = {}
    publisher_input = st.sidebar.selectbox("Selectionnez l'editeur", train_stats["publishers"])
    genre_input = st.sidebar.selectbox("Selectionnez le genre", train_stats["genres"])
    platform_input = st.sidebar.selectbox("Selectionnez la plateforme", train_stats["platforms"])
    years = list(range(1970, 2031))
    year_input = st.sidebar.selectbox("Selectionnez l'annee", years, index=years.index(2024))
    input_data["Year"] = year_input

    meta_input = st.sidebar.number_input(
        "Selectionnez le score Metacritic",
        min_value=0.0,
        max_value=10.0,
        value=train_stats["meta_score_mean"],
        format="%.1f",
    )
    input_data["meta_score"] = meta_input

    user_input = st.sidebar.number_input(
        "Selectionnez le score utilisateur",
        min_value=0.0,
        max_value=10.0,
        value=train_stats["user_review_mean"],
        format="%.1f",
    )
    input_data["user_review"] = user_input

    return publisher_input, genre_input, platform_input, input_data


def prediction_page() -> None:
    """Render the prediction page."""
    # Import cached loaders from the wrapper module (not ml.predict directly)
    from prediction import load_feature_means, load_models

    colored_header(
        label="Prediction des ventes de jeux video",
        description="Estimez les ventes mondiales d'un jeu video a l'aide de notre ensemble de modeles",
        color_name="light-blue-70",
    )
    add_vertical_space(1)

    try:
        lgb_model, xgb_model, cb_model = load_models()
        train_stats = load_feature_means()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modele : {e}")
        return

    # CSS for the arcade screen overlay
    st.markdown(
        """
        <style>
        .arcade-container {
            position: relative;
            text-align: center;
            color: white;
            max-width: 700px;
            margin: 0 auto;
        }
        .arcade-image {
            width: 100%;
            height: auto;
        }
        .arcade-screen {
            font-family: 'Press Start 2P', cursive;
            color: #00FFCC;
            text-shadow: 0 0 10px rgba(0, 255, 204, 0.6);
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size : 22px;
            position: absolute;
            top: -300px;
            left: 52%;
            transform: translate(-50%, -50%);
            width: 65%;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Arcade machine image
    image_path = IMAGES_DIR / "street_arcade.jpg"
    if image_path.exists():
        st.image(str(image_path), width=1000)
    else:
        st.write(
            f"Erreur : l'image {image_path.name} est introuvable. Verifiez le dossier images/."
        )

    # User inputs
    publisher_input, genre_input, platform_input, input_data = _get_input(train_stats)

    if st.sidebar.button("Predire"):
        with st.spinner("Calcul de la prediction..."):
            try:
                df_input = get_features(input_data, train_stats, genre_input, platform_input)
                df_ready = prepare_for_prediction(df_input, publisher_input)

                X = df_ready[NUMERICAL_FEATURES]
                pred_lgb = lgb_model.predict(X)
                pred_xgb = xgb_model.predict(X.values)
                pred_cb = cb_model.predict(X.values)
                user_pred = (pred_lgb + pred_xgb + pred_cb) / 3
                if is_log_transformed():
                    user_pred = np.expm1(user_pred)

                st.markdown(
                    f"""
                    <div class="arcade-container">
                        <div class="arcade-screen">Prediction pour les ventes:<br><br> {user_pred[0]:.4f} millions d'unites</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                export_df = pd.DataFrame(
                    {
                        "Publisher": [publisher_input],
                        "Genre": [genre_input],
                        "Platform": [platform_input],
                        "Year": [input_data["Year"]],
                        "meta_score": [input_data["meta_score"]],
                        "user_review": [input_data["user_review"]],
                        "Predicted_Sales_M": [round(user_pred[0], 4)],
                    }
                )
                st.download_button(
                    "Telecharger la prediction (CSV)",
                    export_df.to_csv(index=False),
                    file_name="prediction.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Erreur lors de la prediction : {e}")
    else:
        st.markdown(
            """
            <div class="arcade-container">
                <div class="arcade-screen">Entrez les informations necessaires pour predire les ventes globales d'un jeu video</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Batch prediction ---
    st.markdown("---")
    st.subheader("Prediction par lot")
    st.write(
        "Telechargez un fichier CSV avec les colonnes : "
        "`Publisher`, `Genre`, `Platform`, `Year`, `meta_score`, `user_review`"
    )
    batch_file = st.file_uploader("Fichier CSV", type="csv", key="batch")

    if batch_file is not None and st.button("Predire le lot"):
        with st.spinner("Predictions en cours..."):
            try:
                batch_df = pd.read_csv(batch_file)
                required = ["Publisher", "Genre", "Platform", "Year", "meta_score", "user_review"]
                missing = [c for c in required if c not in batch_df.columns]
                if missing:
                    st.error(f"Colonnes manquantes : {', '.join(missing)}")
                else:
                    results = []
                    for _, row in batch_df.iterrows():
                        inp = {
                            "Year": int(row["Year"]),
                            "meta_score": float(row["meta_score"]),
                            "user_review": float(row["user_review"]),
                        }
                        df_feat = get_features(inp, train_stats, row["Genre"], row["Platform"])
                        df_r = prepare_for_prediction(df_feat, row["Publisher"])
                        X = df_r[NUMERICAL_FEATURES]
                        p = (
                            lgb_model.predict(X)
                            + xgb_model.predict(X.values)
                            + cb_model.predict(X.values)
                        ) / 3
                        if is_log_transformed():
                            p = np.expm1(p)
                        results.append(round(float(p[0]), 4))

                    batch_df["Predicted_Sales_M"] = results
                    st.dataframe(batch_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Telecharger les resultats (CSV)",
                        batch_df.to_csv(index=False),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Erreur lors de la prediction par lot : {e}")

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Ce modele est en version beta et peut faire des erreurs. "
        "Envisagez de verifier les informations importantes.</p>",
        unsafe_allow_html=True,
    )
