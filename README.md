# Video Game Sales Predictor

Interactive data science web application built with Streamlit for exploring, visualizing, and predicting global video game sales using Machine Learning.

## Project Overview

This project combines data analysis, machine learning, and rich visualizations in a modern dark-themed web application. It predicts global video game sales using a **stacking ensemble** of 5 base models with a Ridge meta-learner, trained on data from 9 sources and optimized with Optuna.

### Model Results (v3 — Production)

| Model | R² | RMSE | MAE |
|-------|------|------|-----|
| CatBoost | 0.506 | 0.621 | 0.288 |
| LightGBM | 0.481 | 0.637 | 0.287 |
| HistGBR | 0.477 | 0.639 | 0.298 |
| XGBoost | 0.376 | 0.698 | 0.425 |
| RandomForest | 0.297 | 0.741 | 0.384 |
| **Stacking Ensemble** | **0.500** | **0.625** | **0.313** |

*v3: 26K cleaned rows (from 64K raw), 50 features, temporal split (2015), log-transformed target, sample weights by confidence tier.*

## Features

The application has **11 interactive pages**:

| Page | Description |
|------|-------------|
| **Home** | Project overview, key metrics, data pipeline diagram |
| **Data Sources** | Documentation of all 9 data sources and merge methodology |
| **Exploratory Analysis** | 20+ interactive Plotly charts with global filters |
| **Feature Engineering** | Preprocessing pipeline, target encoding, 50 engineered features |
| **Training** | Model comparison, stacking architecture, SHAP, hyperparameters |
| **Predictions** | Single and batch prediction interface |
| **Interpretability** | SHAP beeswarm plots, feature descriptions |
| **What-If** | Sensitivity analysis — how each parameter impacts sales |
| **Market Trends** | Genre, platform, and publisher analytics over time |
| **Sentiment NLP** | DistilBERT + BERT multilingual sentiment analysis |
| **About** | Methodology, tech stack, limitations, and future work |

## Data Sources (9)

| Source | Type | Data |
|--------|------|------|
| **VGChartz 2024** (Kaggle) | Physical sales | ~64,000 games, worldwide sales |
| **SteamSpy** | Digital estimates | ~46,000 Steam games, owners, reviews, playtime, price |
| **RAWG API** | Metadata | Ratings, tags, platforms, playtime |
| **IGDB API** | Metadata | Themes, game modes, age ratings, hypes, follows |
| **HowLongToBeat** | Playtime | Main story, completionist, extras completion times |
| **Wikipedia** | Verified sales | Official physical + digital figures |
| **Steam Store API** | Store data | Player counts, review scores, pricing, DLC |
| **OpenCritic** | Reviews | Aggregated scores from 100+ outlets |
| **Gamedatacrunch** | Estimates | Revenue estimates, concurrent players |

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Web Framework** | Streamlit |
| **Data** | Pandas, NumPy, Pandera |
| **Machine Learning** | LightGBM, XGBoost, CatBoost, Scikit-learn, Optuna |
| **NLP** | DistilBERT, BERT Multilingual (HuggingFace Transformers) |
| **Visualization** | Plotly, SHAP |
| **Data Collection** | kagglehub, steamspypi, rapidfuzz, requests |
| **Code Quality** | ruff, pre-commit, pytest (139 tests), GitHub Actions CI |
| **Deployment** | Docker, Streamlit Cloud |

## Installation

### Prerequisites

- Python 3.11+
- pip

### Steps

```bash
# Clone the repository
git clone git@github.com-pro:adrienchabanelpro/streamlit-video-game-advanced.git
cd streamlit-video-game-advanced

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
make install-dev    # All deps (dev, training, NLP, CI)
# or
make install        # Production only

# Run the application
make run
```

The app will be available at `http://localhost:8501`.

### Makefile Commands

```bash
make run            # Run the Streamlit app
make test           # Run pytest suite (139 tests)
make lint           # Lint with ruff
make format         # Format with ruff
make train          # Run v3 training pipeline
make collect-data   # Run 9-source data collection pipeline
make clean          # Remove Python cache files
```

## Project Structure

```
streamlit-video-game-advanced/
├── source/
│   ├── main.py                      # Entry point, navigation (11 pages)
│   ├── config.py                    # Paths, constants, Plotly layout
│   ├── style.py                     # CSS injection (dark slate theme)
│   ├── components.py                # Reusable UI components
│   ├── prediction.py                # Cached model loading wrapper
│   ├── sentiment_analysis.py        # NLP sentiment (DistilBERT + BERT)
│   ├── data_validation.py           # Dataset schema validation
│   ├── pages/                       # 11 Streamlit page modules
│   └── ml/
│       └── predict.py               # Inference pipeline (v3 stacking + v2 fallback)
├── scripts/
│   ├── training/                    # v3 modular training pipeline
│   │   ├── run_training.py          # Orchestrator (8-step pipeline)
│   │   ├── data_prep.py             # Data loading, feature engineering
│   │   ├── models.py                # Individual model trainers + Optuna
│   │   ├── stacking.py              # Stacking ensemble (5 base + Ridge)
│   │   └── evaluation.py            # Metrics, baseline comparison, SHAP
│   └── data_collection/             # 9-source collection + merge pipeline
│       ├── run_pipeline.py          # Pipeline orchestrator
│       ├── merge_all_sources.py     # 9-source fuzzy merge
│       ├── build_clean_dataset.py   # Quality pipeline + sample weights
│       ├── estimate_sales.py        # Review-multiplier sales estimation
│       └── collect_*.py             # Individual source collectors
├── data/                            # Datasets (raw/ is gitignored)
├── models/                          # Trained model artifacts (v3 + v2 fallback)
├── reports/                         # Training logs, SHAP plots
├── tests/                           # pytest suite (139 tests)
├── .claude/                         # Claude Code configuration
├── .github/workflows/ci.yml         # GitHub Actions CI
├── Makefile                         # Build automation
├── Dockerfile                       # Docker deployment
├── requirements.txt                 # Production dependencies (15 packages)
└── requirements-dev.txt             # Development dependencies (+12 packages)
```

## License

This project is for educational and portfolio purposes.
