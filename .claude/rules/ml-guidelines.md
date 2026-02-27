# ML Guidelines

## Data Integrity
- NEVER use regional sales (NA_Sales, EU_Sales, JP_Sales, Other_Sales) as features for predicting Global_Sales — this is data leakage
- Always use temporal splits for time-series-like data (train on older, test on newer)
- Validate with cross-validation, not just single train/test split

## Model Training
- Log all experiments with parameters and metrics
- Save models with clear versioning (e.g., `model_v2_optuna.txt`)
- Always save the preprocessing pipeline alongside the model
- Use `random_state=42` for reproducibility

## Feature Engineering
- Document every engineered feature: name, formula, rationale
- Test feature importance before and after adding new features
- Prefer target encoding over one-hot encoding for high-cardinality categoricals

## Evaluation
- Primary metric: R² (coefficient of determination)
- Also report: MSE, MAE, RMSE
- Always compare against a baseline (mean predictor)
- Use SHAP for model interpretability
