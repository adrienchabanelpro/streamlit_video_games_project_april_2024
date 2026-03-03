"""Prediction page UI: single and batch predictions using the 3-model ensemble."""

import numpy as np
import pandas as pd
import streamlit as st
from ml.predict import NUMERICAL_FEATURES, get_features, is_log_transformed, prepare_for_prediction


def _get_input(train_stats: dict) -> tuple[str, str, str, dict]:
    """Render sidebar inputs and return user selections."""
    st.sidebar.header("Input Selection")

    input_data: dict = {}
    publisher_input = st.sidebar.selectbox("Select Publisher", train_stats["publishers"])
    genre_input = st.sidebar.selectbox("Select Genre", train_stats["genres"])
    platform_input = st.sidebar.selectbox("Select Platform", train_stats["platforms"])
    years = list(range(1970, 2031))
    year_input = st.sidebar.selectbox("Select Release Year", years, index=years.index(2024))
    input_data["Year"] = year_input

    meta_input = st.sidebar.number_input(
        "Select Metacritic Score",
        min_value=0.0,
        max_value=10.0,
        value=train_stats["meta_score_mean"],
        format="%.1f",
    )
    input_data["meta_score"] = meta_input

    user_input = st.sidebar.number_input(
        "Select User Score",
        min_value=0.0,
        max_value=10.0,
        value=train_stats["user_review_mean"],
        format="%.1f",
    )
    input_data["user_review"] = user_input

    return publisher_input, genre_input, platform_input, input_data


def prediction_page() -> None:
    """Render the prediction page."""
    from prediction import load_feature_means, load_models

    st.title("Video Game Sales Prediction")
    st.caption(
        "Estimate global sales of a video game using our model ensemble"
    )

    try:
        lgb_model, xgb_model, cb_model = load_models()
        train_stats = load_feature_means()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # User inputs
    publisher_input, genre_input, platform_input, input_data = _get_input(train_stats)

    if st.sidebar.button("Predict"):
        with st.spinner("Computing prediction..."):
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

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Sales", f"{user_pred[0]:.4f} M")
                with col2:
                    st.metric("Genre", genre_input)
                with col3:
                    st.metric("Platform", platform_input)

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
                    "Download Prediction (CSV)",
                    export_df.to_csv(index=False),
                    file_name="prediction.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.info(
            "Enter the required information in the sidebar "
            "and click 'Predict' to estimate global sales."
        )

    # --- Batch prediction ---
    st.markdown("---")
    st.subheader("Batch Prediction")
    st.write(
        "Upload a CSV file with the following columns: "
        "`Publisher`, `Genre`, `Platform`, `Year`, `meta_score`, `user_review`"
    )
    batch_file = st.file_uploader("CSV File", type="csv", key="batch")

    if batch_file is not None and st.button("Predict Batch"):
        with st.spinner("Running predictions..."):
            try:
                batch_df = pd.read_csv(batch_file)
                required = ["Publisher", "Genre", "Platform", "Year", "meta_score", "user_review"]
                missing = [c for c in required if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
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
                        "Download Results (CSV)",
                        batch_df.to_csv(index=False),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error during batch prediction: {e}")

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #94A3B8;'>"
        "This model is in beta and may make errors. "
        "Consider verifying important information.</p>",
        unsafe_allow_html=True,
    )
