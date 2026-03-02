# Video Game Sales Prediction — Streamlit App

## Quick Reference

- **Run:** `streamlit run source/main.py` → http://localhost:8501
- **Language:** UI in French, docs in English, code mixed
- **Python:** 3.11+ required

## Architecture

- `source/main.py` — Entry point, sidebar nav routing to 12 pages
- `source/config.py` — Paths, constants, Plotly layout
- `source/style.py` — CSS injection (neon dark theme, Press Start 2P + Tiny5 fonts)
- `source/pages/dataviz.py` — 20+ charts (Plotly)
- `source/pages/prediction.py` — Ensemble prediction UI (arcade theme)
- `source/pages/perception.py` — NLP sentiment analysis (DistilBERT + LR fallback)
- `source/ml/predict.py` — Inference pipeline (loads 3 models + transformers)
- `source/analyse_avis_utilisateurs.py` — `clean_text()`, `predict_user_reviews()`
- `scripts/train_model.py` — Full training pipeline (ensemble + Optuna)
- `scripts/data_collection/` — Kaggle + SteamSpy data collection + fuzzy merge

## Data & Models

- `data/Ventes_jeux_video_final.csv` — Main dataset (64,016 rows, 30 cols: VGChartz 2024 + SteamSpy enrichment)
- `reports/model_v2_optuna.txt` — LightGBM (Optuna-optimized)
- `models/model_v2_xgboost.json` — XGBoost (Optuna-optimized)
- `models/model_v2_catboost.cbm` — CatBoost (Optuna-optimized)
- `models/scaler_v2.joblib` — StandardScaler
- `models/target_encoder_v2.joblib` — Publisher target encoder
- `models/feature_means_v2.joblib` — Training stats (genre/platform means, cumulative sales)
- `reports/training_log.json` — Training log (params, metrics, timestamp)
- `models/logistic_regression_model.pkl` — Sentiment classifier (v1)
- `models/tfidf_vectorizer.pkl` — TF-IDF vectorizer

## Key Conventions

- Streamlit CSS via `st.markdown(unsafe_allow_html=True)`
- Models: `joblib.load()` for sklearn, `lgb.Booster(model_file=...)` for LightGBM
- Plotly for interactive charts, Matplotlib for static
- `@st.cache_data` for data loading, `@st.cache_resource` for model loading
- See `IMPROVEMENT.md` for full roadmap
- See `CLAUDE.md` (root) for detailed technical context

## Known Critical Issues

1. ~~Possible data leakage~~ — Fixed in v2 (regional sales excluded, temporal split)
2. ~~OneHotEncoder bloat~~ — Fixed in v2 (target encoding: 10 features instead of 576)
3. ~~No caching~~ — Fixed (st.cache_data/st.cache_resource on all loaders)
4. ~~Pygame games~~ — Removed (incompatible with Streamlit Cloud)
