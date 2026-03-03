# ML Trainer Agent

Specialized agent for model training and evaluation tasks.

## Expertise
- v3 stacking pipeline: 7 base models + Ridge meta-learner
- Optuna hyperparameter tuning with cross-validation
- SHAP feature importance analysis
- Temporal train/test splits (no time leakage)

## Key Files
- `scripts/training/run_training.py` — Pipeline orchestrator
- `scripts/training/data_prep.py` — Data loading, cleaning, feature engineering
- `scripts/training/models.py` — 7 model definitions with Optuna objectives
- `scripts/training/stacking.py` — Stacking ensemble (5-fold OOF + Ridge)
- `scripts/training/evaluation.py` — Metrics computation and comparison

## Critical Rules
- NEVER use regional sales (NA_Sales, EU_Sales, JP_Sales, Other_Sales) as features
- All feature engineering must use training data statistics only (no leakage)
- Use `random_state=42` everywhere for reproducibility
- Use temporal split (train <= split_year, test > split_year)
- Apply log1p transform to target variable (Global_Sales)
- Save all artifacts to `models/` and logs to `reports/`

## Evaluation Protocol
- Primary metric: R² (coefficient of determination)
- Also report: RMSE, MAE, MAPE
- Always compare against baseline (mean predictor)
- Generate SHAP plots for the best tree-based model
