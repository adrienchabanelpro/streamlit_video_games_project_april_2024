import json
import os
import streamlit as st
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

_BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

# Feature order must match training script exactly
_NUMERICAL_FEATURES = [
    "Year",
    "meta_score",
    "user_review",
    "Global_Sales_mean_genre",
    "Global_Sales_mean_platform",
    "Year_Global_Sales_mean_genre",
    "Year_Global_Sales_mean_platform",
    "Cumulative_Sales_Genre",
    "Cumulative_Sales_Platform",
    "Publisher_encoded",
]


@st.cache_resource
def load_models():
    """Load all 3 ensemble models (LightGBM, XGBoost, CatBoost)."""
    lgb_model = lgb.Booster(
        model_file=os.path.join(_BASE_DIR, 'reports', 'model_v2_optuna.txt')
    )
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(
        os.path.join(_BASE_DIR, 'models', 'model_v2_xgboost.json')
    )
    cb_model = cb.CatBoostRegressor()
    cb_model.load_model(
        os.path.join(_BASE_DIR, 'models', 'model_v2_catboost.cbm')
    )
    return lgb_model, xgb_model, cb_model


@st.cache_resource
def load_numerical_transformer():
    return joblib.load(os.path.join(_BASE_DIR, 'models', 'scaler_v2.joblib'))


@st.cache_resource
def load_target_encoder():
    return joblib.load(os.path.join(_BASE_DIR, 'models', 'target_encoder_v2.joblib'))


@st.cache_resource
def load_feature_means():
    return joblib.load(os.path.join(_BASE_DIR, 'models', 'feature_means_v2.joblib'))


@st.cache_data
def _is_log_transformed() -> bool:
    """Check if the trained model used log-transform on the target."""
    log_path = os.path.join(_BASE_DIR, 'reports', 'training_log.json')
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f).get("log_transform", False)
    return False


def _lookup_cumulative(cumsum_dict: dict, category: str, year: int) -> float:
    """Look up cumulative sales for a category up to a given year."""
    if category not in cumsum_dict:
        return 0.0
    yearly = cumsum_dict[category]
    relevant_years = [y for y in yearly if y <= year]
    if not relevant_years:
        return 0.0
    return yearly[max(relevant_years)]


def get_input(train_stats: dict):
    st.sidebar.header('Selection des entrees')

    input_data = {}
    publisher_input = st.sidebar.selectbox(
        "Selectionnez l'editeur", train_stats['publishers']
    )
    genre_input = st.sidebar.selectbox(
        'Selectionnez le genre', train_stats['genres']
    )
    platform_input = st.sidebar.selectbox(
        'Selectionnez la plateforme', train_stats['platforms']
    )
    years = list(range(1980, 2031))
    year_input = st.sidebar.selectbox(
        "Selectionnez l'annee", years, index=years.index(2024)
    )
    input_data['Year'] = year_input

    meta_input = st.sidebar.number_input(
        'Selectionnez le score Metacritic',
        min_value=0.0,
        max_value=100.0,
        value=train_stats['meta_score_mean'],
        format="%.0f",
    )
    input_data['meta_score'] = meta_input

    user_input = st.sidebar.number_input(
        'Selectionnez le score utilisateur',
        min_value=0.0,
        max_value=100.0,
        value=train_stats['user_review_mean'],
        format="%.1f",
    )
    input_data['user_review'] = user_input

    return publisher_input, genre_input, platform_input, input_data


def get_features(
    input_data: dict,
    train_stats: dict,
    genre_input: str,
    platform_input: str,
) -> pd.DataFrame:
    """Build feature vector using pre-computed training statistics."""
    # Mean sales by genre/platform from training data
    input_data['Global_Sales_mean_genre'] = train_stats['genre_means'].get(
        genre_input, train_stats['global_sales_mean']
    )
    input_data['Global_Sales_mean_platform'] = train_stats['platform_means'].get(
        platform_input, train_stats['global_sales_mean']
    )

    # Interaction features
    input_data['Year_Global_Sales_mean_genre'] = (
        input_data['Year'] * input_data['Global_Sales_mean_genre']
    )
    input_data['Year_Global_Sales_mean_platform'] = (
        input_data['Year'] * input_data['Global_Sales_mean_platform']
    )

    # Cumulative sales from training data
    input_data['Cumulative_Sales_Genre'] = _lookup_cumulative(
        train_stats['cumsum_genre'], genre_input, input_data['Year']
    )
    input_data['Cumulative_Sales_Platform'] = _lookup_cumulative(
        train_stats['cumsum_platform'], platform_input, input_data['Year']
    )

    return pd.DataFrame(input_data, index=[0])


def prepare_for_prediction(
    df_input: pd.DataFrame, publisher_input: str
) -> pd.DataFrame:
    """Target-encode Publisher, scale features, return prediction-ready df."""
    encoder = load_target_encoder()
    scaler = load_numerical_transformer()

    # Target encode Publisher (1 column instead of 567 one-hot columns)
    pub_df = pd.DataFrame({"Publisher": [publisher_input]})
    df_input["Publisher_encoded"] = encoder.transform(pub_df)["Publisher"].values

    # Scale all numerical features
    df_input[_NUMERICAL_FEATURES] = scaler.transform(
        df_input[_NUMERICAL_FEATURES]
    )

    return df_input


def prediction_page():
    st.title("Prediction des ventes de jeux video")

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
    image_path = os.path.join(
        os.path.dirname(__file__), '..', 'images', 'street_arcade.jpg'
    )
    if os.path.exists(image_path):
        st.image(image_path, width=1000)
    else:
        st.write(
            f"Erreur : l'image {os.path.basename(image_path)} est introuvable. "
            "Verifiez le dossier images/."
        )

    # User inputs (dropdowns populated from training stats)
    publisher_input, genre_input, platform_input, input_data = get_input(
        train_stats
    )

    if st.sidebar.button('Predire'):
        with st.spinner("Calcul de la prediction..."):
            try:
                # Build feature vector from training stats
                df_input = get_features(
                    input_data, train_stats, genre_input, platform_input
                )

                # Encode + scale
                df_ready = prepare_for_prediction(df_input, publisher_input)

                # Ensemble predict (average of 3 models)
                X = df_ready[_NUMERICAL_FEATURES]
                pred_lgb = lgb_model.predict(X)
                pred_xgb = xgb_model.predict(X.values)
                pred_cb = cb_model.predict(X.values)
                user_pred = (pred_lgb + pred_xgb + pred_cb) / 3
                if _is_log_transformed():
                    user_pred = np.expm1(user_pred)

                st.markdown(
                    f"""
                    <div class="arcade-container">
                        <div class="arcade-screen">Prediction pour les ventes:<br><br> {user_pred[0]:.4f} millions d'unites</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Export prediction as CSV
                export_df = pd.DataFrame({
                    "Publisher": [publisher_input],
                    "Genre": [genre_input],
                    "Platform": [platform_input],
                    "Year": [input_data["Year"]],
                    "meta_score": [input_data["meta_score"]],
                    "user_review": [input_data["user_review"]],
                    "Predicted_Sales_M": [round(user_pred[0], 4)],
                })
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
                        df_feat = get_features(
                            inp, train_stats, row["Genre"], row["Platform"]
                        )
                        df_r = prepare_for_prediction(df_feat, row["Publisher"])
                        X = df_r[_NUMERICAL_FEATURES]
                        p = (
                            lgb_model.predict(X)
                            + xgb_model.predict(X.values)
                            + cb_model.predict(X.values)
                        ) / 3
                        if _is_log_transformed():
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


if __name__ == "__main__":
    prediction_page()
