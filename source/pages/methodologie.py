from io import BytesIO

import altair as alt
import pandas as pd
import requests
import streamlit as st
from config import IMAGES_DIR
from PIL import Image


# Fonction pour afficher la page méthodologie
def methodologie_page() -> None:
    """Render the project methodology page."""

    # Fonction pour charger une image depuis une URL
    def load_image(url: str) -> Image.Image:
        """Fetch an image from *url* and return a PIL Image."""
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image

    # Titre de la page avec icône de méthodologie
    st.title("🕹️ Méthodologie du Projet 🎮")

    # Chemin de l'image locale
    image_path = IMAGES_DIR / "collab.png"

    # Afficher l'image sous le titre
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.write(
            f"Erreur : l'image {image_path.name} est introuvable. Vérifiez le dossier images/."
        )

    # Présentation de la méthodologie
    st.header("Méthodologie")

    st.markdown("""
    ### Approche Structurée et Collaborative

    - **Approche structurée et collaborative** : Nous avons adopté une méthode organisée et coopérative pour assurer le succès du projet.

    ### Réunions Quotidiennes

    - **Réunions Discord quotidiennes** : Chaque après-midi, nous avons consacré du temps au projet. À la fin de chaque journée, nous nous réunissions sur Discord pour :
      - Partager les résultats obtenus.
      - Discuter des observations faites.
      - Planifier les prochaines étapes.

    ### Utilisation de Google Colab

    - **Centralisation des informations sur Google Colab** :
      - Facilité de collaboration et accès aux documents.
      - Garantie que chaque membre puisse y accéder à tout moment.

    ### Communication Continue

    - **Canaux de communication sur Discord et Slack** :
      - Maintien d'un flux constant d'échanges et de discussions.

    ### Apprentissage Collectif

    - **Uniformisation des étapes** :
      - Chaque membre a suivi les mêmes étapes pour acquérir et maîtriser les compétences enseignées.
      - Comparaison régulière des résultats et apprentissage collectif.

    ### Collaboration Intensive

    - **Élément clé : Réunions** :
      - Coordination des efforts.
      - Partage des découvertes.
      - Résolution collective des problèmes.

    Grâce à cette **collaboration intensive**, nous avons pu mener des recherches approfondies et proposer un projet complet et cohérent, reflétant notre engagement et notre rigueur tout au long du processus.
    """)

    # Ajout d'un graphique interactif pour illustrer la répartition du temps de travail
    st.subheader("Répartition du temps de travail")

    # Exemple de données fictives pour le graphique
    data = pd.DataFrame(
        {
            "Activité": ["Réunions Discord", "Travail sur Google Colab", "Autres"],
            "Heures par semaine": [14, 25, 2],
        }
    )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Activité", sort=None),
            y="Heures par semaine",
            color=alt.Color("Activité", legend=None),
            tooltip=["Activité", "Heures par semaine"],
        )
        .properties(width=600, height=400)
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_title(fontSize=16)
    )

    st.altair_chart(chart, use_container_width=True)
