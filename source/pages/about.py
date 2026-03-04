"""About page: methodology, tech stack, limitations, author info."""

import streamlit as st
from components import info_card, section_header


def about_page() -> None:
    """Render the About / Methodology page."""
    st.title("About the Project")
    st.caption("Methodology, tools, limitations, and future work")

    # Methodology
    section_header("Methodology")
    st.write(
        """
        This project follows a **CRISP-DM** (Cross-Industry Standard Process
        for Data Mining) approach in 6 phases:

        1. **Business Understanding** — Predict global video game sales
           from metadata (genre, platform, publisher, scores, etc.)
        2. **Data Understanding** — 9 sources: VGChartz (physical sales),
           SteamSpy (digital PC), RAWG (metadata), IGDB (themes/franchises),
           HowLongToBeat (completion times), Wikipedia (verified sales),
           Steam Store (pricing/DLC), OpenCritic (critic scores),
           Gamedatacrunch (market data)
        3. **Data Preparation** — Fuzzy matching merge, 5-tier data quality
           classification, deduplication, zero-sales removal, imputation
           (17.5K clean games from 64K+ raw records)
        4. **Feature Engineering** — 50 engineered features: publisher/developer
           track record, market context, Steam engagement, RAWG metadata,
           HLTB completion, critics, Steam Store, OpenCritic
        5. **Modeling** — 7 models with Optuna tuning, stacking ensemble with
           Ridge meta-learner
        6. **Evaluation** — R², RMSE, MAE, MAPE, residual analysis, SHAP
        """
    )

    st.divider()

    # Tech stack
    section_header("Tech Stack")

    c1, c2 = st.columns(2)
    with c1:
        info_card(
            "Data & ML",
            """
            <ul style="margin:0;padding-left:16px">
                <li><b>Python 3.11+</b></li>
                <li><b>pandas</b> — data manipulation</li>
                <li><b>scikit-learn</b> — preprocessing, metrics, RF, HGB, ElasticNet</li>
                <li><b>LightGBM</b> — gradient boosting</li>
                <li><b>XGBoost</b> — gradient boosting</li>
                <li><b>CatBoost</b> — gradient boosting</li>
                <li><b>Optuna</b> — Bayesian hyperparameter tuning</li>
                <li><b>SHAP</b> — model interpretability</li>
                <li><b>category_encoders</b> — target encoding</li>
                <li><b>rapidfuzz</b> — fuzzy matching for data merging</li>
            </ul>
            """,
        )

    with c2:
        info_card(
            "Application & NLP",
            """
            <ul style="margin:0;padding-left:16px">
                <li><b>Streamlit</b> — interactive web framework</li>
                <li><b>Plotly</b> — interactive visualizations</li>
                <li><b>Transformers</b> — DistilBERT for sentiment analysis</li>
                <li><b>pytest</b> — unit testing</li>
                <li><b>ruff</b> — linting and formatting</li>
                <li><b>GitHub Actions</b> — CI/CD</li>
                <li><b>Docker</b> — containerization</li>
                <li><b>MLflow</b> — experiment tracking</li>
            </ul>
            """,
            accent="#8B5CF6",
        )

    st.divider()

    # Decisions
    section_header("Key Technical Decisions")

    decisions = [
        ("Why stacking instead of simple averaging?",
         "Stacking learns optimal weights via a Ridge meta-learner, "
         "whereas averaging assumes all models are equally good. "
         "Stacking assigns more weight to the best-performing models on "
         "out-of-fold data."),
        ("Why target encoding?",
         "With ~600 unique publishers, one-hot encoding would create 600 sparse "
         "columns. Target encoding replaces each publisher with its mean sales, "
         "going from 576 columns to 1 while preserving the information."),
        ("Why the log transformation?",
         "Video game sales follow a highly skewed distribution "
         "(a few massive hits, many small sellers). The log1p() transformation "
         "normalizes this distribution and improves model predictions "
         "across the full range."),
        ("Why a temporal split?",
         "A random split would allow games from 2020 in the training set and "
         "games from 2010 in the test set — the model would 'cheat' by seeing "
         "the future. A temporal split (train <= 2015, test > 2015) simulates "
         "real-world usage."),
    ]

    for q, a in decisions:
        with st.expander(q):
            st.write(a)

    st.divider()

    # Limitations
    section_header("Known Limitations")
    st.write(
        """
        - **Physical sales only** for the target variable (VGChartz).
          Digital sales (Steam, PS Store, etc.) are not included in
          Global_Sales.
        - **Data completeness** — Not all games are present across all
          sources. Fuzzy matching may introduce false positives.
        - **Moderate R²** — Predicting video game sales is an inherently
          difficult problem (high variance, unobservable factors such as
          marketing and word-of-mouth).
        - **Temporal bias** — The model is trained on historical data.
          Market trends evolve (rise of free-to-play, cloud gaming).
        """
    )

    st.divider()

    # Future work
    section_header("Future Work")
    st.write(
        """
        - Integration of digital sales data (Epic Games Store, PS Store)
        - Tabular deep learning (TabNet, FT-Transformer)
        - Time series for trend forecasting (Prophet, ARIMA)
        - Review sentiment analysis as a predictive feature
        - LLM for natural language data querying
        """
    )
