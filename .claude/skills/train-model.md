# Train Model Skill

Guide for retraining the sales prediction model.

## Prerequisites
- `data/df_features.csv` or `data/df_topfeats.csv` must exist
- All dependencies installed (`pip install -r requirements.txt`)

## Steps
1. Load dataset from `data/df_features.csv`
2. Separate features (X) and target (Global_Sales)
3. Remove leaky columns: NA_Sales, EU_Sales, JP_Sales, Other_Sales, Rank, Name
4. Apply preprocessing (StandardScaler for numerical, encoding for categorical)
5. Train LightGBM with tuned hyperparameters
6. Evaluate with R², MSE, MAE on test set
7. Save model to `reports/model_v<N>.txt`
8. Save transformers to `models/`

## Current Baseline
- R²=0.988, MSE=0.0007, MAE=0.0132

## Notes
- Always version new models, never overwrite `model_final.txt`
- Update CLAUDE.md if model architecture changes
