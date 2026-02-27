# presentation.py
import os

import pandas as pd
import streamlit as st

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")


@st.cache_data
def _load_presentation_data():
    return pd.read_csv(os.path.join(_BASE_DIR, "data", "Ventes_jeux_video_final.csv"))


def presentation_et_objectif():
    # Obtenir le chemin absolu du dossier contenant ce fichier
    base_path = os.path.dirname(__file__)

    # Construire les chemins absolus pour les images
    image_path1 = os.path.join(base_path, "..", "images", "image_pres.png")
    image_path2 = os.path.join(base_path, "..", "images", "street.png")

    # Vérifier et afficher les images
    for image_path in [image_path1, image_path2]:
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.write(f"Erreur : l'image {os.path.basename(image_path)} est introuvable.")

    st.title("Présentation du projet")

    st.markdown("""
        Dans le cadre de notre formation en analyse de données et en vue de développer nos compétences pratiques, nous avons choisi d'explorer et d'analyser les ventes de jeux vidéo.
    """)

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
    Voici une description détaillée des jeux de données utilisés :

    ### VGChartz
    16500 lignes 
                 
    ### Statistiques des Jeux Vidéos
    62000 lignes
                
    Initialement, nous avons utilisé les données disponibles sur VGChartz. Cependant, nous avons constaté que certaines informations cruciales manquaient. Pour pallier ce manque, nous avons :

    - **Rescrappé VGChartz** : Pour obtenir une version plus complète et actualisée des données de vente.
    - **Scrappé Metacritic** : Pour intégrer des données supplémentaires sur les scores des utilisateurs et de la presse.

    ### Metacritic
    18799 lignes
                
    ### Volumétrie du Jeu de Données Final
    Après avoir combiné les données scrappées de VGChartz, Metacritic et le jeu de données supplémentaires, notre DataFrame final nettoyé contient plus de 14 500 entrées (sans outliers) et 16 325 (avec outliers).
    """)

    # Charger les données
    df = _load_presentation_data()
    st.write("Dataset après scrapping")
    st.write(df)
