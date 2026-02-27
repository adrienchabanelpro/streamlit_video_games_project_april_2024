# Video Game Sales Prediction — Streamlit App

## Quick Reference

- **Run:** `streamlit run source/main.py` → http://localhost:8501
- **Language:** UI in French, docs in English, code mixed
- **Python:** 3.11+ required

## Architecture

- `source/main.py` — Entry point, sidebar nav (radio buttons) routing to 9 pages
- `source/style.py` — CSS injection (orange sidebar, Press Start 2P + Tiny5 fonts, blue #003366 headers)
- `source/dataviz.py` — 20+ charts (Plotly/Matplotlib)
- `source/prediction.py` — Loads LightGBM model + transformers, arcade UI
- `source/perception.py` — NLP sentiment analysis (file upload + gauge)
- `source/analyse_avis_utilisateurs.py` — `clean_text()`, `predict_user_reviews()`
- `source/snake.py`, `source/casse_brique.py` — Pygame games

## Data & Models

- `data/Ventes_jeux_video_final.csv` — Main dataset (16,325 rows)
- `reports/model_final.txt` — LightGBM (500 trees, 576 features, R²=0.988)
- `models/logistic_regression_model.pkl` — Sentiment classifier
- `models/tfidf_vectorizer.pkl` — TF-IDF vectorizer
- `models/*.joblib` — StandardScaler + OneHotEncoder transformers

## Key Conventions

- Streamlit CSS via `st.markdown(unsafe_allow_html=True)`
- Models: `joblib.load()` for sklearn, `lgb.Booster(model_file=...)` for LightGBM
- Plotly for interactive charts, Matplotlib for static
- No tests, no caching, no error handling currently
- See `IMPROVEMENT.md` for full roadmap
- See `CLAUDE.md` (root) for detailed technical context

## Known Critical Issues

1. Possible data leakage (regional sales as features)
2. OneHotEncoder bloat (576 cols from Publisher)
3. No `@st.cache_data` / `@st.cache_resource`
4. Pygame games need local display (no cloud support)
