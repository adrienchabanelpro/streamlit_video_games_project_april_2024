# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral Rules

- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary — prefer editing existing files
- NEVER save working files, text/mds, or tests to the root folder
- ALWAYS read a file before editing it

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

Run a single test file or test:
```bash
python -m pytest tests/test_config.py -v
python -m pytest tests/test_config.py::test_name -v
```

CI: GitHub Actions runs `pytest` on push/PR to `main` (Python 3.12, ubuntu-latest). See `.github/workflows/ci.yml`.

ALWAYS run tests after code changes. ALWAYS verify the app starts before committing.

## Architecture

**Entry point:** `source/main.py` — uses `st.navigation()` + `st.Page()` with lazy imports (`importlib.import_module`) to keep startup memory under 200 MB for Streamlit Cloud.

**Important CWD quirk:** `make run` does `cd source && streamlit run main.py`, so all imports in `source/` are relative to `source/`, not the project root. This is why `from config import ...` works (not `from source.config`).

**Core modules:**
- `source/config.py` — All path constants (`ROOT`, `DATA_DIR`, `MODELS_DIR`, etc.) and theme colors. `ROOT` is `Path(__file__).resolve().parent.parent` (project root).
- `source/style.py` — CSS injection via `components.html()` JS that writes into the parent document `<head>`. This bypasses Streamlit's HTML sanitizer which strips `<style>` tags. Idempotent via element IDs.
- `source/components.py` — Shared UI: `metric_card`, `info_card`, `source_card`, `pipeline_step`, `section_header`. All use `unsafe_allow_html=True`.
- `source/ml/predict.py` — Streamlit-agnostic inference. Loads v2 models (LGB+XGB+CB simple average). Caching decorators applied in page layer, not here.

**Pages (11):** All in `source/pages/`. Each exports a single `*_page()` function. To add a page: create the module, add a `st.Page(_lazy(...))` entry in `main.py`'s `st.navigation()` list.

**Training pipeline:** `scripts/training/` — modular: `data_prep.py` → `models.py` → `stacking.py` → `evaluation.py`, orchestrated by `run_training.py`.

**Data collection:** `scripts/data_collection/` — 5-source pipeline (RAWG, IGDB, HLTB, SteamSpy, Kaggle) with fuzzy matching merge.

## Model Versions

- **v2 (current production):** LGB + XGB + CB simple average. LightGBM model is in `reports/model_v2_optuna.txt` (not `models/`). XGB/CB in `models/`. Transformers: `models/scaler_v2.joblib`, `models/target_encoder_v2.joblib`, `models/feature_means_v2.joblib`.
- **v3 (stacking ensemble):** 5 base models + Ridge meta-learner. Artifacts: `models/model_v3_*`. Training log: `reports/training_log_v3.json`.

## File Organization

| Directory | Contents |
|-----------|----------|
| `source/` | App source code (pages, components, config, style, ML inference) |
| `source/pages/` | 11 Streamlit page modules |
| `source/ml/` | ML inference code |
| `scripts/training/` | v3 modular training pipeline |
| `scripts/data_collection/` | 5-source collection + merge |
| `tests/` | pytest tests (conftest adds `source/`, `scripts/` to sys.path) |
| `data/` | Datasets |
| `models/` | Trained model artifacts + transformers |
| `reports/` | Training logs, evaluation outputs, and v2 LGB model |

## Language Convention

- **UI text:** English
- **Code (variables, functions, docstrings):** English
- **Commit messages:** English, imperative mood

## Key Technical Details

- `@st.cache_data` for data loading, `@st.cache_resource` for model/transformer loading
- Plotly for all charts (dark theme via `config.PLOTLY_LAYOUT`)
- `pathlib.Path` for all file paths, never hardcode absolute paths
- NEVER use regional sales (NA_Sales, EU_Sales, JP_Sales, Other_Sales) as features — data leakage
- `random_state=42` for ML reproducibility
- Primary ML metric: R² (also report MSE, MAE, RMSE)

## Security

- NEVER hardcode API keys or credentials in source files
- NEVER commit .env files — use `.env.example` as template
- Validate user input at system boundaries
