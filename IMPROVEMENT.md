# Improvement Roadmap

Current state and next steps for the video game sales prediction project.

---

## Current State (v3)

### What's been built

**Machine Learning**
- Stacking ensemble: 5 base models (LightGBM, XGBoost, CatBoost, RandomForest, HistGBR) + Ridge meta-learner
- Optuna hyperparameter tuning (50+30+30+20+20 trials per model)
- 50 engineered features from 9 data sources
- Temporal split (train <= 2015, test > 2015), log-transformed target
- Sample weights by data confidence tier (verified > estimated > none)
- Best R² = 0.500 on test set (baseline: -0.014)
- SHAP feature importance (beeswarm + bar plots)

**Data Pipeline**
- 9-source collection pipeline: VGChartz (Kaggle), SteamSpy, Steam Reviews, RAWG, IGDB, HLTB, Wikipedia, Steam Store, OpenCritic, Gamedatacrunch
- Fuzzy merge with rapidfuzz (85% threshold, exact + fuzzy phases)
- Data quality pipeline: deduplication, validation, sample weighting
- Review-multiplier sales estimation for digital-only titles
- 26K clean rows from 64K raw after quality filtering

**Application (11 pages)**
- Home, Data Sources, Exploratory Analysis, Feature Engineering, Training, Predictions, Interpretability, What-If, Market Trends, Sentiment NLP, About
- Lazy-loaded pages for < 200 MB startup memory
- Dark slate theme with Plotly interactive charts
- Single + batch prediction, CSV export
- DistilBERT + BERT multilingual sentiment analysis with aspect detection

**Code Quality**
- 139 pytest tests
- ruff linting + formatting, pre-commit hooks
- GitHub Actions CI (Python 3.12)
- Modular architecture: source/, scripts/training/, scripts/data_collection/
- Docker deployment ready

---

## Next Steps

### High Priority

- [ ] **Improve R² score** — Current 0.500 has room to grow. Investigate:
  - Feature selection (remove low-importance features, reduce noise)
  - Better handling of missing values (currently many enrichment features are 0-filled)
  - Train on estimated_total_sales target (review-multiplier estimates) instead of VGChartz physical-only
  - Tune stacking: try different meta-learners, cross-validation strategies
  - Add TabNet or neural network as additional base model

- [ ] **Digital sales gap** — VGChartz only tracks physical sales. The review-multiplier estimation exists but isn't the training target yet. Switching to `estimated_total_sales` or a blended target could capture the full market.

- [ ] **Data freshness** — Current data is a snapshot. Consider:
  - Scheduled pipeline runs (cron / GitHub Actions)
  - Incremental collection (only fetch new games)
  - Track data staleness per source

### Medium Priority

- [ ] **Deep learning** — Try pytorch-tabnet or fast.ai tabular learner as an additional base model in the stacking ensemble.

- [ ] **Time-series forecasting** — Use Prophet or temporal fusion transformers to forecast future sales trends by genre/platform.

- [ ] **LLM-powered analysis** — Integrate Claude API for:
  - Natural language queries ("best-selling RPGs on PlayStation")
  - Review summarization
  - Trend detection and insights generation

- [ ] **Interactive dashboard mode** — Single-page dashboard with draggable/resizable widgets (streamlit-elements).

- [ ] **Review summarization** — Use an LLM to generate concise summaries of review collections per game.

### Low Priority

- [ ] **Real-time market tracker** — Live API connections for current market trends alongside historical data.

- [ ] **Secrets management** — Move API keys to `st.secrets` for Streamlit Cloud deployment.

- [ ] **Performance monitoring** — Basic analytics (page views, prediction counts).

- [ ] **AutoML benchmark** — Run H2O / AutoGluon / FLAML to find the performance ceiling.

- [ ] **Review authenticity detection** — Detect fake/bot reviews using anomaly detection.

---

## Completed Items

<details>
<summary>Click to expand full history of completed improvements</summary>

### Machine Learning
- [x] Hyperparameter tuning with Optuna (50-trial Bayesian optimization, 5-fold CV)
- [x] Proper cross-validation (5-fold CV in Optuna objective)
- [x] SHAP feature importance visualization (beeswarm + bar)
- [x] Ensemble modeling (v2: 3-model average, v3: 5-model stacking)
- [x] Target encoding for Publisher (replaces 567 one-hot columns)
- [x] Time-series aware splitting (temporal split at 2015)
- [x] Log-transform target (log1p/expm1)
- [x] Remove data leakage (regional sales dropped, train-only feature computation)
- [x] Sample weights by data confidence tier

### UI/Design
- [x] Streamlit page config (title, favicon, wide layout)
- [x] Loading states (st.spinner)
- [x] Consistent color palette (dark slate theme)
- [x] Responsive Plotly charts
- [x] Native multi-page app (st.navigation API)
- [x] Dark mode theme
- [x] Custom Streamlit theme (.streamlit/config.toml)
- [x] Full CSS overhaul (modern dark slate, responsive)

### Features
- [x] Export predictions as CSV
- [x] Search & filter on DataViz (multi-select + year slider)
- [x] What-if analysis (variable sweep with real-time charts)
- [x] Batch prediction (CSV upload)
- [x] Market trends explorer (genre, platform, publisher analytics)

### Data Pipeline
- [x] Data validation (Pandera schemas)
- [x] Caching (@st.cache_data, @st.cache_resource)
- [x] 9-source data collection pipeline
- [x] Fuzzy merge with rapidfuzz
- [x] Data quality pipeline with sample weighting
- [x] Review-multiplier sales estimation

### NLP
- [x] Confidence scores on sentiment predictions
- [x] Word clouds for positive/negative clusters
- [x] 5-star rating mode (BERT multilingual)
- [x] Transformer models (DistilBERT for sentiment)
- [x] Aspect-based sentiment analysis (gameplay, graphics, story, etc.)
- [x] Multilingual support (EN, FR, DE, ES, IT, NL)

### Code Quality
- [x] Type hints on all functions
- [x] Consistent naming conventions
- [x] Central config (pathlib paths, theme constants)
- [x] Proper logging in training pipeline
- [x] 139 unit tests (pytest)
- [x] Modular architecture (source/, scripts/training/, scripts/data_collection/)
- [x] Pre-commit hooks (ruff lint + format)
- [x] Google-style docstrings
- [x] CI/CD (GitHub Actions, pytest on push/PR)
- [x] Makefile automation

### Deployment
- [x] Streamlit Cloud deployment
- [x] Docker (Python 3.12-slim)

</details>

---

*Last updated: 2026-03-06*
