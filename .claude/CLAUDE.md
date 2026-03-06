# Video Game Sales Prediction — Streamlit App

## Behavioral Rules

- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary — prefer editing existing files
- NEVER save working files, text/mds, or tests to the root folder
- ALWAYS read a file before editing it
- ALWAYS run tests after code changes
- ALWAYS verify the app starts before committing

## Build & Test

```bash
make install-dev    # Install all deps (dev, training, NLP, CI)
make run            # Streamlit app (cd source && streamlit run main.py)
make test           # Full pytest suite
make lint           # ruff check source/ scripts/ tests/
make format         # ruff format source/ scripts/ tests/
make train          # Run training pipeline
make collect-data   # Run data collection pipeline
```

Single test: `python -m pytest tests/test_file.py::test_name -v`

CI: GitHub Actions runs pytest on push/PR to `main` (Python 3.12). See `.github/workflows/ci.yml`.

## Architecture

**Entry point:** `source/main.py` — `st.navigation()` + `st.Page()` with lazy imports (`importlib.import_module`) to keep startup memory under 200 MB.

**CWD quirk:** `make run` does `cd source && streamlit run main.py`, so imports in `source/` are relative to `source/` (e.g. `from config import ...`, not `from source.config`).

**Core modules:**
- `source/config.py` — Path constants (`ROOT`, `DATA_DIR`, `MODELS_DIR`), theme colors, `PLOTLY_LAYOUT`
- `source/style.py` — CSS injection via `components.html()` JS into parent `<head>` (bypasses Streamlit sanitizer)
- `source/components.py` — Shared UI: `metric_card`, `info_card`, `source_card`, `pipeline_step`, `section_header`
- `source/ml/predict.py` — Streamlit-agnostic inference (v3: 5-model stacking + Ridge meta-learner, v2 fallback)
- `source/sentiment_analysis.py` — NLP sentiment (DistilBERT + BERT multilingual)

**Pipelines:**
- Training: `scripts/training/` — `data_prep.py` → `models.py` → `stacking.py` → `evaluation.py` via `run_training.py`
- Data collection: `scripts/data_collection/` — 9 sources (RAWG, IGDB, HLTB, SteamSpy, Kaggle, Wikipedia, Steam Store, OpenCritic, Gamedatacrunch) + fuzzy merge

## Pages (11)

| Page | File | Description |
|------|------|-------------|
| Home | `home.py` | Dashboard, key metrics, pipeline diagram |
| Data Sources | `data_sources.py` | Sources, merge methodology, schema |
| Exploratory Analysis | `dataviz.py` | Interactive Plotly charts with filters |
| Feature Engineering | `feature_engineering.py` | Feature explanations |
| Training | `model_training.py` | Model comparison, stacking, SHAP |
| Predictions | `prediction.py` | Single + batch prediction UI |
| Interpretability | `interpretability.py` | SHAP beeswarm, feature descriptions |
| What-If | `what_if.py` | Variable sweep analysis |
| Market Trends | `market_insights.py` | Genre/platform/publisher analytics |
| Sentiment NLP | `perception.py` | DistilBERT sentiment analysis |
| About | `about.py` | Methodology, tech stack, limitations |

## Data & Models

- `data/Ventes_jeux_video_clean.csv` — Primary training dataset (26K rows, quality-filtered, sample-weighted)
- `data/Ventes_jeux_video_v3.csv` — Full merged dataset (64K rows, 9-source, used by app pages)
- `data/Ventes_jeux_video_final.csv` — v2 legacy fallback (64K rows, VGChartz + SteamSpy)
- **v3 (production):** 5 base models (LGB, XGB, CB, RF, HGB) + Ridge meta-learner. R²=0.500, 50 features. Artifacts: `models/model_v3_*`, `reports/model_v3_lgb.txt`, `reports/training_log_v3.json`
- **v2 (fallback):** LGB + XGB + CB simple average, 10 features. Artifacts: `models/*_v2.*`, `reports/model_v2_optuna.txt`
- Model loading: `joblib.load()` for sklearn, `lgb.Booster(model_file=...)` for LightGBM

## Security

- NEVER hardcode API keys or credentials in source files
- NEVER commit .env files — use `.env.example` as template
