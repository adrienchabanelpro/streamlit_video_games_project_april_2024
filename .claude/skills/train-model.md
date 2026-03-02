# Train Model Skill

Guide for retraining the sales prediction model.

## Prerequisites
- `data/Ventes_jeux_video_final.csv` must exist (64K rows, 30 columns)
- All dependencies installed (`pip install -r requirements.txt`)

## Steps
1. Load dataset from `data/Ventes_jeux_video_final.csv`
2. Drop non-feature columns: Name, Rank, regional sales, img, developer, release_date, last_update, steam_* cols
3. Clean NaN (drop missing Publisher/Year/Global_Sales, fill median for meta_score/user_review)
4. Temporal train/test split (Optuna selects best split year from 2013-2015)
5. Compute feature engineering stats from training data only (genre/platform means, cumulative sales)
6. Target-encode Publisher (smoothing=10)
7. StandardScaler on 10 numerical features
8. log1p transform on target (Global_Sales)
9. Optuna hyperparameter tuning: LightGBM (50 trials), XGBoost (30 trials), CatBoost (30 trials)
10. Train final models with best params
11. Evaluate all models + ensemble on test set
12. Generate SHAP plots
13. Save all artifacts to `models/` and `reports/`
14. Log to MLflow

## Command
```bash
python scripts/train_model.py
```

## Current Baseline (64K data, v2 ensemble)
- Ensemble: R²=0.3811, RMSE=0.3585, MAE=0.0998
- LightGBM: R²=0.3740, XGBoost: R²=0.3754, CatBoost: R²=0.3556
- Baseline (mean predictor): R²=-0.0136, RMSE=0.4588
- See `reports/training_log.json` for full details.

## Notes
- Always version new models, never overwrite `model_final.txt`
- Update CLAUDE.md if model architecture changes
- Training takes ~15-20 min with Optuna (50+30+30 trials)
