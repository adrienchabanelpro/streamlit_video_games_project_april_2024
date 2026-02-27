# Improvement Roadmap

This document outlines an ambitious roadmap to take this video game sales prediction project to the next level. Improvements are organized by category and priority.

---

## Table of Contents

- [1. Machine Learning & Models](#1-machine-learning--models)
- [2. UI/Design](#2-uidesign)
- [3. New Features](#3-new-features)
- [4. Data Pipeline & Sources](#4-data-pipeline--sources)
- [5. NLP & Sentiment Analysis](#5-nlp--sentiment-analysis)
- [6. Code Quality & Architecture](#6-code-quality--architecture)
- [7. Deployment & DevOps](#7-deployment--devops)
- [8. Mini-Games](#8-mini-games)
- [Priority Matrix](#priority-matrix)

---

## 1. Machine Learning & Models

### Quick Wins
- [x] **Hyperparameter tuning with Optuna** — 50-trial Bayesian optimization with 5-fold CV. Split year also optimized. See `scripts/train_model.py`.
- [x] **Proper cross-validation** — 5-fold CV integrated into Optuna objective function with temporal train/test split.
- [x] **Feature importance visualization** — SHAP summary (beeswarm) and bar plots generated and displayed on Modelisation page. See `reports/shap_summary.png` and `reports/shap_bar.png`.

### Medium Effort
- [x] **Ensemble modeling** — LightGBM + XGBoost + CatBoost averaging ensemble. Each model Optuna-tuned (50+30+30 trials). Ensemble used in prediction app.
- [x] **Target encoding** — Publisher now uses target encoding (1 column) instead of one-hot (567 columns). See `models/target_encoder_v2.joblib`.
- [x] **Time-series aware splitting** — Temporal split (train <= split_year, test > split_year). Split year optimized by Optuna from [2013, 2014, 2015].
- [x] **Log-transform target** — `np.log1p(y)` before training, `np.expm1(pred)` at inference. Flag stored in `reports/training_log.json`. Applied across single, batch, and what-if predictions.
- [x] **Remove data leakage** — Regional sales dropped. Target-derived features (genre/platform means, cumulative sales) now computed on training data only. See `scripts/train_model.py`.

### Ambitious
- [ ] **Deep learning with PyTorch/TensorFlow** — Build a neural network (tabular model) using embeddings for categorical features. Libraries like `pytorch-tabnet` or `fast.ai` make this accessible.
- [ ] **Time-series forecasting** — Use Prophet, ARIMA, or temporal fusion transformers to forecast future sales trends by genre/platform.
- [ ] **AutoML comparison** — Run AutoML frameworks (H2O, AutoGluon, FLAML) to benchmark against manual approach. This shows what the ceiling performance looks like.

---

## 2. UI/Design

### Quick Wins
- [x] **Streamlit page config** — Use `st.set_page_config()` with proper title, favicon, and wide layout mode.
- [x] **Loading states** — Add `st.spinner()` around model loading and predictions so users know something is happening.
- [x] **Better color palette** — Define a consistent color scheme (not just orange sidebar) using CSS variables. Consider a retro neon palette (dark background, neon green/pink/cyan accents).
- [x] **Responsive charts** — Replace static Matplotlib plots with Plotly everywhere for consistency and interactivity.

### Medium Effort
- [x] **Multi-page app (native)** — Migrated to `st.navigation()` API with proper URL routing, page icons, and global sidebar branding.
- [x] **Dark mode** — Implement a proper dark theme that matches the retro arcade aesthetic. Dark backgrounds with neon accents.
- [ ] **Streamlit-extras components** — Use `streamlit-extras` for animated counters, card layouts, metric displays, and other modern UI components.
- [x] **Custom Streamlit theme** — Create a `.streamlit/config.toml` with custom primary colors, background, and font settings.
- [ ] **Animated transitions** — Add Lottie animations (via `streamlit-lottie`) for page transitions and loading states.

### Ambitious
- [ ] **Full CSS overhaul** — Redesign the entire app with a cohesive retro-futuristic theme. Custom CSS for every component.
- [ ] **Mobile-responsive layout** — Test and optimize for mobile viewing with conditional layouts.
- [ ] **Interactive dashboard mode** — Create a single-page dashboard view (using `streamlit-elements`) with draggable/resizable widgets.

---

## 3. New Features

### Quick Wins
- [x] **Export predictions as CSV/PDF** — Let users download prediction results with `st.download_button()`.
- [ ] **Game comparison tool** — Side-by-side comparison of two games' predicted sales based on different parameters.
- [x] **Search & filter on DataViz** — Multi-select filters (genre, platform, publisher) + year range slider at top of DataViz page. All charts respond to filters. Matplotlib replaced with Plotly + dark neon theme.

### Medium Effort
- [x] **Recommendation engine** — "Games like this" feature: given a game's attributes, find similar games using cosine similarity or k-NN on feature vectors.
- [x] **What-if analysis** — Interactive sliders to see how changing one variable (e.g., meta_score from 60 to 90) impacts predicted sales, with real-time chart updates.
- [x] **Batch prediction** — Upload a CSV of multiple games and get predictions for all of them at once.
- [ ] **Historical trend explorer** — Interactive timeline showing how genres, platforms, and publishers evolved over decades.
- [ ] **Publisher analytics dashboard** — Deep-dive into any publisher's historical performance, genre distribution, and predicted future.

### Ambitious
- [ ] **Real-time market tracker** — Live connection to gaming APIs to show current market trends alongside historical data.
- [ ] **Multiplayer prediction game** — Users compete to predict which game will sell more, earning points for accuracy.
- [ ] **Natural language query** — "Show me the best-selling RPGs on PlayStation" processed via LLM to generate dynamic visualizations.

---

## 4. Data Pipeline & Sources

### Quick Wins
- [ ] **Data validation** — Add schema validation (with `pandera` or `great_expectations`) to catch data quality issues early.
- [x] **Caching** — `@st.cache_data` and `@st.cache_resource` added to all data/model loading functions.

### Medium Effort
- [ ] **Steam API integration** — Pull real-time data from Steam's public API (player counts, reviews, pricing, tags).
- [ ] **IGDB API** — Integrate the Internet Game Database API for richer game metadata (ratings, screenshots, release info).
- [ ] **RAWG API** — Another rich gaming API with 500,000+ games, ratings, and metadata.
- [ ] **Automated web scraping** — Set up scheduled scraping of VGChartz/Metacritic with `BeautifulSoup` or `Scrapy` to keep data fresh.
- [ ] **Data versioning** — Use DVC (Data Version Control) to track dataset versions alongside code.

### Ambitious
- [ ] **Real-time data pipeline** — Automated ETL pipeline (Airflow/Prefect) that scrapes, cleans, and updates the dataset on a schedule.
- [ ] **Digital sales data** — Incorporate digital download data (currently only physical sales). This is the biggest data gap.
- [ ] **Social media signals** — Scrape Twitter/Reddit mentions as predictive features for upcoming game sales.

---

## 5. NLP & Sentiment Analysis

### Quick Wins
- [x] **Show confidence scores** — Display prediction probability alongside positive/negative labels.
- [ ] **Word clouds** — Generate word clouds for positive vs negative reviews to visualize common themes.
- [ ] **More sentiment granularity** — Move from binary (positive/negative) to 5-star rating prediction or fine-grained sentiment (very negative, negative, neutral, positive, very positive).

### Medium Effort
- [x] **Transformer models** — Replace Logistic Regression + TF-IDF with a pre-trained transformer:
  - `distilbert-base-uncased-finetuned-sst-2-english` for sentiment
  - Or fine-tune on gaming review data for domain-specific accuracy
- [ ] **Aspect-based sentiment** — Extract sentiment per aspect (gameplay, graphics, story, value) rather than overall sentiment.
- [ ] **Review summarization** — Use an LLM to generate concise summaries of review collections.
- [ ] **Multilingual support** — Support French, Japanese, German reviews using multilingual transformers.

### Ambitious
- [ ] **LLM-powered analysis** — Integrate Claude API or GPT for advanced review analysis, trend detection, and natural language insights.
- [ ] **Review authenticity detection** — Detect fake/bot reviews using anomaly detection.

---

## 6. Code Quality & Architecture

### Quick Wins
- [ ] **Type hints** — Add type annotations to all functions for better IDE support and documentation.
- [ ] **Consistent naming** — Standardize file names (all snake_case), function names, and variable names.
- [ ] **Environment variables** — Move any configuration (file paths, model paths) to a `config.py` or `.env` file.
- [ ] **Proper logging** — Replace print statements with Python's `logging` module.

### Medium Effort
- [x] **Unit tests** — Write tests with `pytest` for:
  - Data loading and preprocessing functions
  - Feature engineering pipeline
  - Model prediction pipeline
  - NLP text cleaning and prediction
- [ ] **Modular architecture** — Refactor into proper packages:
  ```
  src/
  ├── app/           # Streamlit pages
  ├── ml/            # ML models, training, prediction
  ├── data/          # Data loading, cleaning, feature engineering
  ├── nlp/           # Sentiment analysis
  ├── games/         # Pygame mini-games
  ├── utils/         # Shared utilities
  └── config.py      # Central configuration
  ```
- [ ] **Pre-commit hooks** — Set up `ruff` (linter + formatter), `mypy` (type checking), and `pytest` as pre-commit hooks.
- [ ] **Docstrings** — Add Google-style docstrings to all public functions and classes.

### Ambitious
- [x] **CI/CD pipeline** — GitHub Actions workflow (`.github/workflows/ci.yml`) runs pytest on every push and PR to main. Uses Python 3.12 with pip caching.
- [ ] **ML pipeline with MLflow** — Track experiments, log metrics, version models, and compare runs.
- [ ] **Makefile / Task runner** — Centralize common commands (run, test, lint, train, deploy).

---

## 7. Deployment & DevOps

### Quick Wins
- [x] **Streamlit Cloud** — Deploy for free on [share.streamlit.io](https://share.streamlit.io) — just connect the GitHub repo.
- [ ] **Secrets management** — Use `st.secrets` for any API keys instead of hardcoding.

### Medium Effort
- [ ] **Docker** — Create a proper Dockerfile (not just dev container) for reproducible production deployment.
- [ ] **Railway / Render** — Deploy on a managed platform with custom domain support.
- [ ] **Performance monitoring** — Add basic analytics (page views, prediction counts) with a lightweight solution.

### Ambitious
- [ ] **Kubernetes** — Container orchestration for scaling under load.
- [ ] **CDN for assets** — Move large images to a CDN to reduce app load time.
- [ ] **A/B testing** — Test different model versions or UI layouts with real users.

---

## 8. Mini-Games

### Quick Wins
- [ ] **High score persistence** — Save high scores to a local file or Streamlit session state.
- [ ] **Difficulty levels** — Add easy/medium/hard modes to Snake and Breakout.

### Medium Effort
- [ ] **More games** — Add:
  - Space Invaders (fits the retro theme)
  - Pong (classic 2-player)
  - Tetris
- [ ] **Leaderboard** — Global leaderboard using a simple database (SQLite or Firebase).
- [ ] **Streamlit-native games** — Rebuild mini-games using `streamlit-elements` or JavaScript components instead of Pygame (which requires local display).

### Ambitious
- [ ] **ML-powered game AI** — Train a reinforcement learning agent to play the mini-games and let users compete against it.
- [ ] **Game analytics** — Track gameplay metrics and visualize them (play time, scores, completion rates).

---

## Priority Matrix

| Priority | Category | Item | Impact | Effort |
|----------|----------|------|--------|--------|
| ~~1~~ | ML | ~~Hyperparameter tuning (Optuna)~~ | High | Low | **DONE** |
| ~~2~~ | ML | ~~Remove data leakage~~ | Critical | Low | **DONE** |
| ~~3~~ | ML | ~~SHAP feature importance~~ | High | Low | **DONE** |
| ~~4~~ | Code | ~~Caching with @st.cache_data~~ | High | Low | **DONE** |
| ~~5~~ | UI | ~~Streamlit native multi-page~~ | Medium | Low | **DONE** |
| ~~6~~ | ML | ~~Ensemble (LightGBM + XGBoost + CatBoost)~~ | High | Medium | **DONE** |
| ~~7~~ | ML | ~~Target encoding (replace one-hot)~~ | High | Medium | **DONE** |
| ~~8~~ | NLP | ~~Transformer-based sentiment~~ | High | Medium | **DONE** |
| ~~9~~ | UI | ~~Dark mode + retro neon theme~~ | Medium | Medium | **DONE** |
| 10 | Data | Steam/IGDB API integration | High | Medium |
| ~~11~~ | Code | ~~Unit tests (pytest)~~ | Medium | Medium | **DONE** |
| ~~12~~ | Deploy | ~~Streamlit Cloud deployment~~ | High | Low | **DONE** |
| ~~13~~ | Feature | ~~What-if analysis tool~~ | Medium | Medium | **DONE** |
| ~~14~~ | Feature | ~~Recommendation engine~~ | High | Medium | **DONE** |
| 15 | ML | Deep learning (TabNet/neural net) | Medium | High |
| 16 | Code | Full modular refactor | Medium | High |
| 17 | NLP | LLM-powered analysis | High | High |
| 18 | Data | Real-time data pipeline | Medium | High |

---

*This roadmap is a living document. Items will be checked off and reprioritized as the project evolves.*
