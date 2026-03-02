import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from config import DATA_DIR


@st.cache_data
def _load_feature_data() -> pd.DataFrame:
    """Load feature engineering example dataset."""
    return pd.read_csv(DATA_DIR / "df_topfeats.csv")


def feature_engineering_page() -> None:
    """Render the feature engineering and pre-processing page."""

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
    **Taille:** ~64 000 lignes et 30 colonnes (VGChartz 2024 + SteamSpy).
    **Colonnes exclues pour la prediction :** 'Name', 'Rank', 'NA_Sales', 'EU_Sales',
    'JP_Sales', 'Other_Sales' (fuite de donnees), 'img', 'developer', 'release_date',
    'last_update', et toutes les colonnes `steam_*` (non utilisees comme features).
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
    - **v1 (ancien)** : OneHotEncoder sur Publisher → 576 colonnes creuses.
    - **v2 (actuel)** : Target Encoding sur Publisher → 1 seule colonne (`Publisher_encoded`),
      beaucoup plus efficace pour les variables a haute cardinalite.
    - Normalisation de toutes les colonnes numeriques avec StandardScaler.
    """)

    st.header("Normalisation et Encodage")
    st.markdown("""
    **Objectif :** Preparer les donnees pour la modelisation en normalisant les colonnes
    numeriques et en encodant les variables categorielles.
    **Resultat (v2) :** Apres transformation, le dataset d'entrainement contient ~60 000+
    lignes et seulement **10 features** (grace au target encoding au lieu du one-hot).
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

    st.header("Dataset Final (v2)")
    st.markdown("""
    **Taille :** ~60 000+ lignes et **10 features** pour la prediction.
    **Features utilisees :**
    - `Year`
    - `meta_score`, `user_review`
    - `Publisher_encoded` (target encoding — 1 colonne au lieu de 576)
    - `Global_Sales_mean_genre`, `Global_Sales_mean_platform`
    - `Year_Global_Sales_mean_genre`, `Year_Global_Sales_mean_platform`
    - `Cumulative_Sales_Genre`, `Cumulative_Sales_Platform`

    **Cible :** `Global_Sales` (avec transformation log1p).
    **Split temporel** : entrainement sur les jeux anterieurs a l'annee de split,
    test sur les jeux posterieurs (pas de fuite de donnees).
    """)

    data = _load_feature_data()
    st.write(data)
