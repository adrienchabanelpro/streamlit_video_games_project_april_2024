# Prediction des Ventes de Jeux Video

Application web interactive de data science construite avec Streamlit, permettant d'explorer, visualiser et predire les ventes mondiales de jeux video a l'aide du Machine Learning.

## Apercu du Projet

Ce projet combine l'analyse de donnees, le machine learning et des mini-jeux interactifs dans une application web au theme retro arcade (Street Fighter). Il s'articule autour de la prediction des ventes globales de jeux video en utilisant un modele **LightGBM Regressor**.

### Resultats cles du modele

| Metrique | Avant Feature Engineering | Apres Feature Engineering |
|----------|--------------------------|--------------------------|
| R²       | 0.3107                   | **0.9880**               |
| MSE      | 0.0400                   | **0.0007**               |
| MAE      | 0.1432                   | **0.0132**               |

## Fonctionnalites

L'application comporte **9 pages interactives** :

| Page | Description |
|------|-------------|
| **Presentation** | Vue d'ensemble du projet, objectifs et description du dataset |
| **Methodologie** | Approche collaborative, outils utilises, repartition du travail |
| **DataViz** | 20+ graphiques interactifs (Plotly/Matplotlib) explorant les tendances de ventes |
| **Feature Engineering** | Pipeline de preprocessing, creation de features, analyse PCA |
| **Modelisation** | Presentation de LightGBM, reglage interactif des parametres |
| **Prediction** | Interface arcade pour predire les ventes d'un jeu en temps reel |
| **Perception** | Analyse de sentiment NLP sur les avis utilisateurs (Logistic Regression + TF-IDF) |
| **Perspectives** | Axes d'amelioration, quiz interactif, formulaire de feedback |
| **Jeu Surprise** | Mini-jeu aleatoire : Snake ou Casse-Brique (Pygame) |

## Sources de Donnees

- **VGChartz** — 16 500 lignes de donnees de ventes physiques de jeux video
- **Metacritic** — 18 799 lignes de scores critiques (meta_score) et avis utilisateurs (user_review)
- **Video Game Sales Statistics** — 62 000 lignes de donnees supplementaires

**Dataset final** : ~14 500 entrees x 576 colonnes (apres nettoyage et encodage)

## Stack Technique

| Categorie | Technologies |
|-----------|-------------|
| **Framework Web** | Streamlit |
| **Data** | Pandas, NumPy |
| **Machine Learning** | LightGBM, Scikit-learn, Joblib |
| **NLP** | NLTK, TF-IDF (Scikit-learn) |
| **Visualisation** | Plotly, Matplotlib, Seaborn, Altair |
| **Jeux** | Pygame |
| **Autres** | Pillow, Requests, Statsmodels |

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

### Dev Container

Un Dev Container est configure pour VS Code (Python 3.11, Debian Bullseye). Il installe automatiquement les dependances et lance Streamlit au demarrage.

## Structure du Projet

```
streamlit_video_games_project_april_2024/
├── source/
│   ├── main.py                          # Point d'entree, navigation sidebar
│   ├── presentation.py                  # Page de presentation
│   ├── methodologie.py                  # Page methodologie
│   ├── dataviz.py                       # Visualisation des donnees (20+ graphiques)
│   ├── feature_engineering.py           # Pipeline de feature engineering
│   ├── modelisation.py                  # Page modelisation LightGBM
│   ├── prediction.py                    # Interface de prediction interactive
│   ├── perception.py                    # Analyse de sentiment NLP
│   ├── perspectives.py                  # Perspectives et quiz
│   ├── snake.py                         # Mini-jeu Snake (Pygame)
│   ├── casse_brique.py                  # Mini-jeu Casse-Brique (Pygame)
│   ├── style.py                         # CSS personnalise (theme retro)
│   └── analyse_avis_utilisateurs.py     # Module d'analyse de sentiment
├── data/
│   ├── Ventes_jeux_video_final.csv      # Dataset principal (16 325 lignes)
│   ├── df_features.csv                  # Dataset avec features engineerees
│   └── df_topfeats.csv                  # Features les plus importantes
├── models/
│   ├── logistic_regression_model.pkl    # Modele de sentiment
│   ├── tfidf_vectorizer.pkl             # Vectoriseur TF-IDF
│   ├── numerical_transformer.joblib     # StandardScaler
│   └── categorical_transformer.joblib   # OneHotEncoder
├── reports/
│   └── model_final.txt                  # Modele LightGBM entraine (500 arbres)
├── images/                              # Assets visuels (GIFs, PNGs)
├── fonts/
│   └── PressStart2P-Regular.ttf         # Police retro pixel
├── .devcontainer/                       # Configuration Dev Container
├── requirements.txt                     # Dependances Python
└── .gitignore
```

## Modeles de Machine Learning

### 1. LightGBM Regressor — Prediction des ventes

- **Objectif** : Predire `Global_Sales` (ventes mondiales en millions)
- **Features** : Year, meta_score, user_review, Publisher (one-hot encoded), + 6 features engineerees
- **Features engineerees** :
  - `Global_Sales_mean_genre` / `Global_Sales_mean_platform`
  - `Year_Global_Sales_mean_genre` / `Year_Global_Sales_mean_platform`
  - `Cumulative_Sales_Genre` / `Cumulative_Sales_Platform`
- **Configuration** : 500 arbres de decision, ~3000 feuilles par arbre
- **Selection** : Choisi via LazyRegressor parmi 29 modeles testes

### 2. Logistic Regression — Analyse de sentiment

- **Objectif** : Classifier les avis utilisateurs (positif/negatif)
- **Vectorisation** : TF-IDF
- **Preprocessing** : Lowercase, suppression ponctuation, stop words, lemmatisation (NLTK)

## Lancement

```bash
streamlit run source/main.py
```

L'application s'ouvre dans le navigateur avec une sidebar de navigation permettant d'acceder aux 9 pages.
