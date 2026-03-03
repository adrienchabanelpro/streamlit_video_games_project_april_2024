# Video Game Sales Predictor

Interactive data science web application built with Streamlit for exploring, visualizing, and predicting global video game sales using Machine Learning.

## Project Overview

This project combines data analysis, machine learning, and rich visualizations in a modern dark-themed web application. It predicts global video game sales using a **stacking ensemble** of 7 models (LightGBM, XGBoost, CatBoost, RandomForest, HistGradientBoosting, ElasticNet, TabNet) with a Ridge meta-learner, optimized with Optuna.

### Model Results (v2 — Production)

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| LightGBM | 0.3740 | 0.3606 | 0.1006 |
| XGBoost | 0.3754 | 0.3602 | 0.1009 |
| CatBoost | 0.3556 | 0.3658 | 0.1021 |
| **Ensemble** | **0.3811** | **0.3585** | **0.0998** |

*v2: 64K rows, 10 features, temporal split, no data leakage.*

## Features

The application has **11 interactive pages**:

| Page | Description |
|------|-------------|
| **Home** | Project overview, key metrics, data pipeline diagram |
| **Data Sources** | Documentation of all data sources and merge methodology |
| **Exploratory Analysis** | 20+ interactive Plotly charts with global filters |
| **Feature Engineering** | Preprocessing pipeline, target encoding, engineered features |
| **Training** | Model comparison, stacking architecture, SHAP, hyperparameters |
| **Predictions** | Single and batch prediction interface |
| **Interpretability** | SHAP beeswarm plots, feature descriptions |
| **What-If** | Sensitivity analysis — how each parameter impacts sales |
| **Market Trends** | Genre, platform, and publisher analytics over time |
| **Sentiment NLP** | DistilBERT sentiment analysis on user reviews |
| **About** | Methodology, tech stack, limitations, and future work |

## Data Sources

- **VGChartz 2024** (Kaggle) — ~64,000 games, worldwide physical sales
- **SteamSpy** — ~46,000 Steam games, owner estimates, reviews, playtime, price
- **RAWG API** — Rich metadata (ratings, tags, platforms)
- **IGDB API** — Themes, game modes, age ratings
- **HowLongToBeat** — Completion times
- **Wikipedia** — Verified official sales figures (physical + digital)
- **Steam Store API** — Player counts, review scores, pricing
- **OpenCritic** — Aggregated review scores from 100+ outlets
- **Gamedatacrunch** — Revenue estimates, concurrent players

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Web Framework** | Streamlit |
| **Data** | Pandas, NumPy |
| **Machine Learning** | LightGBM, XGBoost, CatBoost, Scikit-learn, Optuna |
| **NLP** | DistilBERT (Transformers/HuggingFace), BERT Multilingual |
| **Visualization** | Plotly, SHAP |
| **Data Collection** | kagglehub, steamspypi, rapidfuzz, requests |
| **Code Quality** | ruff, pre-commit, pytest, GitHub Actions CI |
| **Deployment** | Docker, Streamlit Cloud |

## Installation

### Prerequisites

- Python 3.11+
- pip

### Steps

```bash
# Clone the repository
git clone <repo-url>
cd streamlit_video_games_project_april_2024

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
make run
```

The app will be available at `http://localhost:8501`.

### Makefile Commands

```bash
make run            # Run the Streamlit app
make test           # Run pytest suite
make lint           # Lint with ruff
make format         # Format with ruff
make train          # Run model training pipeline
make collect-data   # Run data collection pipeline
make clean          # Remove Python cache files
```

## Project Structure

```
streamlit_video_games_project_april_2024/
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
│       └── predict.py               # Inference pipeline (ensemble)
├── scripts/
│   ├── training/                    # v3 modular training pipeline
│   └── data_collection/             # Multi-source collection + merge
├── data/                            # Datasets
├── models/                          # Trained model artifacts
├── reports/                         # Training logs, SHAP plots
├── tests/                           # pytest suite (124 tests)
├── .claude/                         # Claude Code configuration
├── .github/workflows/ci.yml         # GitHub Actions CI
├── Makefile                         # Build automation
├── Dockerfile                       # Docker deployment
├── requirements.txt                 # Production dependencies
└── requirements-dev.txt             # Development dependencies
```

## License

This project is for educational and portfolio purposes.
