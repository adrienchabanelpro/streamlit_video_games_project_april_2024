import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")


@st.cache_data
def _load_feature_data():
    return pd.read_csv(os.path.join(_BASE_DIR, "data", "df_topfeats.csv"))


def feature_engineering():

    # Titre et description
    st.title("Feature Engineering et Pre-processing")

    # Afficher le GIF depuis Giphy
    gif_url = "https://giphy.com/embed/QJDOwyyvgIcPS"
    components.html(
        f"""
    <iframe src="{gif_url}" width="480" height="480" style="border:0;" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
    <p><a href="https://giphy.com/gifs/arcade-street-fighter-QJDOwyyvgIcPS">via GIPHY</a></p>
    """,
        height=500,
    )

    # Définir les sections
    st.header("Dataset Initial")
    st.markdown("""
    **Taille:** 16 500 entrées et 5 colonnes.  
    **Colonnes Exclues:** 'Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank'.
    """)

    st.header("Nettoyage")
    st.markdown("""
    - Suppression des doublons.  
    - Correction des formats des colonnes ('Year' convertie en datetime, 'user_review' en numérique).  
    - Alignement des informations des colonnes 'Platform' pour faciliter les merges.  
    - Suppression des lignes avec des valeurs manquantes ('Publisher' et 'Year').  
    - Remplacement des valeurs manquantes dans 'user_review' et 'meta_score' par les médianes selon les plateformes et les genres.
    """)

    st.header("Transformation")
    st.markdown("""
    - Application du OneHotEncoder sur les variables catégorielles ('Publisher') et sur les autres variables catégorielles.  
    - Normalisation de toutes les autres colonnes numériques.
    """)

    st.header("Normalisation et Encodage")
    st.markdown("""
    **Objectif:** Préparer les données pour la modélisation en normalisant les colonnes numériques et en encodant les variables catégorielles.  
    **Résultat:** Après transformation, chaque dataset avait en moyenne entre 14 000 entrées et 17 000 entrées et entre 400 et 600 colonnes.
    """)

    st.header("Analyse en Composantes Principales (PCA)")
    st.markdown("""
    **Problème Identifié:**  
    - La variable "Éditeur" avait un grand nombre de valeurs uniques, compliquant l'ACP.  
    - Plus de 30 composantes principales étaient nécessaires pour expliquer environ 90 % de la variance, rendant la réduction de dimension peu efficace.  
    - La complexité de la visualisation et de l'interprétation des résultats.

    **Conclusion:**  
    L'ACP n'a pas donné les résultats escomptés, et nous avons opté pour l'utilisation directe des techniques de réduction de dimension dans notre modèle ML (LightGBM Regressor) via l'EFB (exclusive feature bundling).
    """)

    st.header("Feature Engineering")
    st.markdown("""
    **Processus Créatif:**  
    - Création de nouvelles variables pour améliorer la performance du modèle.  
    - Test de plusieurs itérations pour trouver les meilleures variables.

    **Nouvelles Variables Créées:**  
    - Global_Sales_Mean_Platform et Global_Sales_Mean_genre: Moyennes des ventes globales par genre et par plateforme.  
    - Year_Global_Sales_mean_platform et Year_Global_Sales_mean_genre: Interactions entre l'année de sortie et les moyennes des ventes globales.  
    - Cumulative_Sales_Platform et Cumulative_Sales_Genre: Indicateurs de popularité basés sur les ventes historiques.
    """)

    st.subheader("Hypothèses Non Concluantes")
    st.markdown("""
    Les variables suivantes n'ont pas amélioré les performances du modèle :  
    - Genre_Count  
    - Publisher_Count  
    - Platform_Count  
    - Publisher_Popularity_Sales  
    - Age  
    - Decade  
    - Score_Interaction
    """)

    st.header("Dataset Final")
    st.markdown("""
    **Taille:** 14 500 entrées et 11 colonnes.  
    **Colonnes:**  
    - Year  
    - Publisher  
    - Global_Sales  
    - meta_score  
    - user_review  
    - Global_Sales_mean_genre  
    - Global_Sales_mean_platform  
    - Year_Global_Sales_mean_genre  
    - Year_Global_Sales_mean_platform  
    - Cumulative_Sales_Genre  
    - Cumulative_Sales_Platform  

    Après encodage, ce dataset comptait toujours 14 500 entrées mais 576 colonnes.
    """)

    data = _load_feature_data()
    st.write(data)
