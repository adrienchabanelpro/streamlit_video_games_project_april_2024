# main.py
import streamlit as st
import os
import subprocess
import random
from style import apply_style
from presentation import presentation_et_objectif
from methodologie import methodologie
from dataviz import dataviz
from feature_engineering import feature_engineering
from modelisation import modelisation
from perspectives import perspectives
from perception import perception
from prediction import prediction_page

st.set_page_config(page_title="Prediction Jeux Video", page_icon="🎮", layout="wide")

# Appliquer le style
apply_style()
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        .stRadio > label {
            font-family: 'Press Start 2P', cursive;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

# Chemin du GIF de navigation
navigation_gif_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'chun-li-walking-animation.gif')


# Titre de la barre latérale
st.sidebar.title("Navigation")

# Vérifier si le GIF de navigation existe
if os.path.exists(navigation_gif_path):
    st.sidebar.image(navigation_gif_path, width=200)
else:
    st.sidebar.write(f"Erreur : l'image {os.path.basename(navigation_gif_path)} est introuvable. Vérifiez le dossier images/.")


# Configuration de la barre latérale pour la navigation
page = st.sidebar.radio("Choisissez une page", 
                        ["Présentation", "Méthodologie", "DataViz", 
                         "Feature Engineering", "Modélisation", 
                         "Prédiction", "Perception", 
                         "Perspectives", "Jeu Surprise"])

# Choisir la page à afficher
if page == "Présentation":
    presentation_et_objectif()
elif page == "Méthodologie":
    methodologie()
elif page == "DataViz":
    st.title("DataViz")
    dataviz()
    
elif page == "Feature Engineering":
    feature_engineering()
elif page == "Modélisation":
    st.title("Modélisation")
    modelisation()
elif page == "Prédiction":
    prediction_page()
elif page == "Perspectives":
    perspectives()
elif page == "Perception":
    perception()
        #else:
          #  st.error("Le fichier CSV doit contenir une colonne 'user_review'.")
elif page == "Jeu Surprise":
    st.title("THE GAME!!")
    st.write("Es-tu prêt à donner le meilleur de toi même?")
    if st.button("Clique ICI"):
    # Utiliser random.choices avec des poids égaux
        game_choice = random.choices(['casse_brique.py', 'snake.py'], weights=[1, 1], k=1)[0]
        game_path = os.path.join(os.path.dirname(__file__), game_choice)
        subprocess.Popen(['python', game_path])
        st.write(f"Le jeu {game_choice.split('.')[0]} se lance dans une nouvelle fenêtre.")
