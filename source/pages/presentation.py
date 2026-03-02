# presentation.py
"""Presentation page: project overview and dataset display."""

import pandas as pd
import streamlit as st
from config import DATA_DIR, IMAGES_DIR
from data_validation import validate_dataframe


@st.cache_data
def _load_presentation_data() -> pd.DataFrame:
    """Load the main video game sales dataset for the presentation page."""
    df = pd.read_csv(DATA_DIR / "Ventes_jeux_video_final.csv")
    # Advisory validation — warn but continue
    is_valid, errors = validate_dataframe(df)
    if not is_valid:
        st.warning(
            f"Validation des donnees : {len(errors)} probleme(s) detecte(s). "
            "Les donnees affichees peuvent contenir des anomalies."
        )
    return df


def presentation_page() -> None:
    """Render the project presentation and objectives page."""
    # Construire les chemins absolus pour les images
    image_path1 = IMAGES_DIR / "image_pres.png"
    image_path2 = IMAGES_DIR / "street.png"

    # Vérifier et afficher les images
    for image_path in [image_path1, image_path2]:
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.write(f"Erreur : l'image {image_path.name} est introuvable.")

    st.title("Présentation du projet")

    st.markdown("""
        Dans le cadre de notre formation en analyse de données et en vue de développer nos compétences pratiques, nous avons choisi d'explorer et d'analyser les ventes de jeux vidéo.
    """)

    # Charger les données (needed for dynamic stats below)
    df = _load_presentation_data()

    st.header("Définition des Objectifs Principaux")
    st.markdown("""
        Pour ce projet, il faudra estimer les ventes totales d'un jeu vidéo à l'aide d'informations descriptives. Il va alors falloir passer par plusieurs étapes dont :

        - **Exploration des Données** : Comprendre la structure des données et identifier les principales caractéristiques qui influencent les ventes de jeux vidéo.
        - **Visualisation des Données** : Créer des visualisations interactives pour mettre en évidence les tendances et les relations dans les données.
        - **Pré-processing des Données et Features Engineering** : Nettoyer et préparer et enrichir les données pour une analyse plus approfondie pour des modèles prédictifs.
        - **Modélisation** : Entraînements de modèles de machine learning sur des données afin de prédire les ventes globales (nombre d'unités vendues) des jeux vidéo.
        - **Analyse et Interprétation** : Comprendre les résultats obtenus et en tirer des conclusions pertinentes.
    """)

    st.markdown("""
    Voici une description detaillee des jeux de donnees utilises :

    ### VGChartz 2024 (Kaggle)
    ~64 000 jeux video — ventes physiques mondiales (NA, EU, JP, Other, Global).

    ### SteamSpy
    ~46 000 jeux Steam — estimations de proprietaires, avis positifs/negatifs,
    temps de jeu moyen et median, prix actuel et initial.

    ### Fusion par correspondance floue (fuzzy matching)
    Les deux sources ont ete fusionnees par correspondance floue sur les noms de jeux
    (librairie **rapidfuzz**), avec un seuil de similarite de 85%.
    """)

    st.markdown(f"""
    ### Volumetrie du Jeu de Donnees Final
    Le dataset fusionne contient **{len(df):,} lignes** et **{len(df.columns)} colonnes**,
    couvrant des jeux de {int(df['Year'].min()) if 'Year' in df.columns and df['Year'].notna().any() else '?'}
    a {int(df['Year'].max()) if 'Year' in df.columns and df['Year'].notna().any() else '?'}.

    Les 13 colonnes `steam_*` enrichissent les donnees VGChartz avec des metriques
    de la plateforme Steam (proprietaires, avis, temps de jeu, prix).
    """)

    st.write("Dataset fusionne (VGChartz + SteamSpy)")
    st.write(df)
