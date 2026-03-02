# Prediction des Ventes de Jeux Video

Application web interactive de data science construite avec Streamlit, permettant d'explorer, visualiser et predire les ventes mondiales de jeux video a l'aide du Machine Learning.

## Apercu du Projet

Ce projet combine l'analyse de donnees, le machine learning et des mini-jeux interactifs dans une application web au theme retro arcade (Street Fighter). Il s'articule autour de la prediction des ventes globales de jeux video en utilisant un **ensemble de 3 modeles** (LightGBM + XGBoost + CatBoost) optimises avec Optuna.

### Resultats cles du modele v2

| Modele | R² | RMSE | MAE |
|--------|-----|------|-----|
| LightGBM | 0.3740 | 0.3606 | 0.1006 |
| XGBoost | 0.3754 | 0.3602 | 0.1009 |
| CatBoost | 0.3556 | 0.3658 | 0.1021 |
| **Ensemble** | **0.3811** | **0.3585** | **0.0998** |

*Modele v2 : 64K donnees, 10 features, split temporel, pas de fuite de donnees.*

## Fonctionnalites

L'application comporte **12 pages interactives** :

| Page | Description |
|------|-------------|
| **Presentation** | Vue d'ensemble du projet, objectifs et description du dataset |
| **Methodologie** | Approche collaborative, outils utilises, repartition du travail |
| **DataViz** | 20+ graphiques interactifs (Plotly) explorant les tendances de ventes |
| **Feature Engineering** | Pipeline de preprocessing v2, target encoding, features engineerees |
| **Modelisation** | Presentation de LightGBM, comparaison v1/v2, SHAP, reglage interactif |
| **Prediction** | Interface arcade pour predire les ventes d'un jeu en temps reel (ensemble) |
| **What-If** | Analyse de sensibilite : comment chaque parametre impacte les ventes |
| **Recommandations** | Suggestions de jeux similaires basees sur les features |
| **Comparaison** | Comparaison cote a cote de jeux video |
| **Tendances** | Exploration des tendances par genre, plateforme, editeur dans le temps |
| **Perception** | Analyse de sentiment NLP sur les avis utilisateurs (DistilBERT + Logistic Regression) |
| **Perspectives** | Axes d'amelioration, quiz interactif, formulaire de feedback |

## Sources de Donnees

- **VGChartz 2024** (Kaggle) — ~64 000 jeux video, ventes physiques mondiales (NA, EU, JP, Other, Global)
- **SteamSpy** — ~46 000 jeux Steam, estimations de proprietaires, avis, temps de jeu, prix
- **Fusion par correspondance floue** (rapidfuzz, seuil 85%) : 64 016 lignes, 30 colonnes

**Dataset final** : ~64 000 lignes x 30 colonnes (dont 13 colonnes `steam_*` enrichies)

## Stack Technique

| Categorie | Technologies |
|-----------|-------------|
| **Framework Web** | Streamlit |
| **Data** | Pandas, NumPy |
| **Machine Learning** | LightGBM, XGBoost, CatBoost, Scikit-learn, Optuna, Joblib |
| **NLP** | DistilBERT (Transformers/HuggingFace), Logistic Regression, TF-IDF, NLTK |
| **Visualisation** | Plotly, Matplotlib, Seaborn, SHAP |
| **Data Collection** | kagglehub, steamspypi, rapidfuzz |
| **Code Quality** | ruff, pre-commit, pytest, GitHub Actions CI |
| **Deploiement** | Docker, Streamlit Cloud |
| **Autres** | Pillow, Requests, Statsmodels, MLflow |

## Installation

### Prerequis

- Python 3.11+
- pip

### Etapes

```bash
# Cloner le depot
git clone <url-du-repo>
cd streamlit_video_games_project_april_2024

# Creer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dependances
pip install -r requirements.txt

# Lancer l'application
streamlit run source/main.py
```

L'application sera accessible sur `http://localhost:8501`.

### Commandes utiles (Makefile)

```bash
make run        # Lancer l'app
make test       # Lancer les tests
make lint       # Verifier le code (ruff)
make format     # Formater le code (ruff)
make train      # Re-entrainer les modeles
make clean      # Nettoyer les fichiers temporaires
```

### Dev Container

Un Dev Container est configure pour VS Code (Python 3.11, Debian Bullseye). Il installe automatiquement les dependances et lance Streamlit au demarrage.

## Structure du Projet

```
streamlit_video_games_project_april_2024/
├── source/
│   ├── main.py                          # Point d'entree, navigation (12 pages)
│   ├── config.py                        # Chemins, constantes, layout Plotly
│   ├── style.py                         # CSS personnalise (theme retro neon)
│   ├── prediction.py                    # Wrapper de chargement modeles (cache)
│   ├── data_validation.py               # Validation du dataset
│   ├── analyse_avis_utilisateurs.py     # Module d'analyse de sentiment
│   ├── pages/
│   │   ├── presentation.py              # Page de presentation
│   │   ├── methodologie.py              # Page methodologie
│   │   ├── dataviz.py                   # Visualisation des donnees (20+ graphiques)
│   │   ├── feature_engineering.py       # Pipeline de feature engineering
│   │   ├── modelisation.py              # Page modelisation + SHAP
│   │   ├── prediction.py               # Interface de prediction interactive
│   │   ├── what_if.py                   # Analyse What-If
│   │   ├── recommendation.py           # Recommandations de jeux
│   │   ├── comparison.py               # Comparaison de jeux
│   │   ├── trends.py                   # Explorateur de tendances
│   │   ├── perception.py               # Analyse de sentiment NLP
│   │   ├── perspectives.py             # Perspectives et quiz
│   │   ├── pong_streamlit.py           # Jeu Pong dans le navigateur
│   │   └── leaderboard.py             # Classement des scores
│   ├── ml/
│   │   └── predict.py                  # Pipeline d'inference (ensemble)
│   └── games/
│       ├── snake.py                    # Mini-jeu Snake (Pygame)
│       ├── casse_brique.py             # Mini-jeu Casse-Brique (Pygame)
│       └── space_invaders.py           # Mini-jeu Space Invaders (Pygame)
├── scripts/
│   ├── train_model.py                  # Pipeline d'entrainement v2 (ensemble + Optuna)
│   └── data_collection/               # Scripts Kaggle + SteamSpy + fusion
├── data/
│   ├── Ventes_jeux_video_final.csv     # Dataset principal (~64 000 lignes, 30 colonnes)
│   ├── df_features.csv                 # Dataset avec features engineerees
│   └── df_topfeats.csv                 # Features les plus importantes
├── models/
│   ├── model_v2_xgboost.json          # XGBoost optimise (Optuna)
│   ├── model_v2_catboost.cbm          # CatBoost optimise (Optuna)
│   ├── scaler_v2.joblib               # StandardScaler
│   ├── target_encoder_v2.joblib       # Target encoder (Publisher)
│   ├── feature_means_v2.joblib        # Stats d'entrainement (moyennes, cumuls)
│   ├── logistic_regression_model.pkl  # Modele de sentiment
│   └── tfidf_vectorizer.pkl           # Vectoriseur TF-IDF
├── reports/
│   ├── model_v2_optuna.txt            # LightGBM optimise (Optuna)
│   ├── training_log.json              # Log d'entrainement (params, metrics)
│   ├── shap_summary.png              # SHAP beeswarm plot
│   └── shap_bar.png                  # SHAP bar plot
├── images/                            # Assets visuels (GIFs, PNGs)
├── fonts/
│   └── PressStart2P-Regular.ttf       # Police retro pixel
├── tests/                             # Tests pytest
├── .claude/                           # Configuration Claude Code
├── .github/workflows/ci.yml          # GitHub Actions CI
├── Makefile                           # Commandes utiles
├── Dockerfile                         # Deploiement Docker
├── .pre-commit-config.yaml            # Pre-commit hooks (ruff)
├── ruff.toml                          # Configuration ruff
├── requirements.txt                   # Dependances Python
└── .gitignore
```

## Modeles de Machine Learning

### 1. Ensemble v2 — Prediction des ventes (LightGBM + XGBoost + CatBoost)

- **Objectif** : Predire `Global_Sales` (ventes mondiales en millions)
- **Donnees** : ~64 000 lignes (VGChartz 2024 + SteamSpy), split temporel
- **Features** (10) : Year, meta_score, user_review, Publisher_encoded (target encoding), + 6 features engineerees (moyennes par genre/plateforme, interactions annee, ventes cumulatives)
- **Optimisation** : Optuna (50 trials LightGBM, 30 XGBoost, 30 CatBoost)
- **Transformation cible** : log1p(Global_Sales) pour normaliser la distribution
- **Ensemble** : Moyenne des predictions des 3 modeles
- **Pas de fuite de donnees** : ventes regionales exclues, split temporel

### 2. DistilBERT — Analyse de sentiment (v2)

- **Objectif** : Classifier les avis utilisateurs (positif/negatif)
- **Modele** : DistilBERT fine-tune (HuggingFace Transformers)
- **Fallback** : Logistic Regression + TF-IDF (modele v1)

### 3. Logistic Regression — Analyse de sentiment (v1)

- **Objectif** : Classifier les avis utilisateurs (positif/negatif)
- **Vectorisation** : TF-IDF
- **Preprocessing** : Lowercase, suppression ponctuation, stop words, lemmatisation (NLTK)

## Lancement

```bash
streamlit run source/main.py
```

L'application s'ouvre dans le navigateur avec une sidebar de navigation permettant d'acceder aux 12 pages.
